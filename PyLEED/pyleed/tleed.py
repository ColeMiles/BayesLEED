""" Python code for interfacing with the TensErLEED code.
"""
from __future__ import annotations # Python 3.7+ required

import os
import shutil
import subprocess
import logging
import enum
import time
import itertools
import multiprocessing as mp
import copy
from typing import List, Tuple, Collection, Optional, Union

import numpy as np

from .structure import AtomicStructure, Site, Layer, Atom, LayerType
from .searchspace import DeltaSearchDim, DeltaSearchSpace, Constraint, optimize_delta_anneal
from .curves import IVCurve, IVCurveSet, parse_ivcurves, avg_rfactors


_MNLMBS = [19, 126, 498, 1463, 3549, 7534, 14484, 25821, 43351,
           69322, 106470, 158067, 227969, 320664, 441320]
_MNLMOS = [1, 70, 264, 759, 1820, 3836, 7344, 13053, 21868, 34914,
           53560, 79443, 114492, 160952, 221408]
_MNLMS = [1, 76, 284, 809, 1925, 4032, 7680, 13593, 22693, 36124,
          55276, 81809, 117677, 165152, 226848]


class BeamInfo:
    """ A simple struct-like class for holding information about a set of beams
         measured in experiment.
    """
    def __init__(self, theta: float, phi: float, beams: List[Tuple[int, int]],
                 energy_min: float, energy_max: float, energy_step: float):
        self.theta = theta
        self.phi = phi
        self.beams = beams
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_step = energy_step
        if not self.energy_min < self.energy_max:
            raise ValueError("Cannot have energy_min >= energy_max")

    def __len__(self):
        return len(self.beams)

    def __iter__(self):
        return iter(self.beams)


class BeamList:
    """ A class representing the beamlist generated as a pre-processing step in
         the LEED dynamical structure calculations.
        Note a distinction: BeamInfo is used for the set of beams measured by experiment
         that you want to compare simulated results against. BeamList is the list of all
         beams that TensErLEED has determined it needs to track to perform this
         calculation. (Produced by beamgen.f).
    """
    def __init__(self, beams, energies):
        self.beams: List[Tuple[float, float]] = beams
        self.energies: List[float] = energies
        assert len(self.beams) == len(self.energies)

    def __len__(self):
        return len(self.beams)

    def __iter__(self):
        return zip(iter(self.beams), iter(self.energies))

    def to_script(self) -> str:
        """ Write out beamlist in format expected by TensErLEED
        """
        script = str(len(self.beams)) + "\n"
        for i, (beam, energy) in enumerate(zip(self.beams, self.energies)):
            script += "{:>10.5f}{:>10.5f}  1  1          E ={:>11.4f}  NR.{:>3d}\n".format(
                beam[0], beam[1], energy, i+1
            )
        return script


def parse_beamlist(filename: str) -> BeamList:
    try:
        with open(filename, "r") as f:
            contents = f.readlines()
        beamlist = _parse_beamlist_str(contents)
    except Exception as e:
        logging.error("Error when parsing beamlist {}!".format(filename))
        raise e

    return beamlist


def _parse_beamlist_str(lines: List[str]) -> BeamList:
    beams, energies = [], []
    linenum = 1
    while linenum < len(lines):
        line = lines[linenum]
        line_s = line.split()
        beams.append((float(line_s[0]), float(line_s[1])))
        energies.append(float(line_s[6]))
        linenum += 1
    return BeamList(beams, energies)


class Phaseshifts:
    """ Class representing a set of phaseshifts for a LEED calculation """
    def __init__(self, filename: str, energies: np.ndarray, phases: np.ndarray):

        self.filename = os.path.abspath(filename)
        self.energies = energies  # Energies at which each phaseshift is calculated (Hartrees)
        self.phases = phases      # [NENERGIES, NELEM, NANG_MOM] array of phaseshifts (Radians)

        # Do some basic validation
        if len(self.energies) != len(self.phases):
            raise ValueError("Number of energies not commensurate with number of phaseshifts")
        if len(self.phases.shape) != 3:
            raise ValueError(
                "Phases should be a 3-dimensional array of shape [NENERGIES, NELEM, NANG_MOM]"
            )

        self.num_energies = self.phases.shape[0]  # Number of energies tabulated
        self.num_elem = self.phases.shape[1]      # Number of elements
        self.lmax = self.phases.shape[2] - 1      # Maximum angular momentum quantum number

    """ Returns a string to be inserted into scripts which expect phaseshifts as text.
        Produces the string in the gross Fortran format of 10 F7.4's per line.
    """
    def to_script(self) -> str:
        script = ""
        for energy, phases in zip(self.energies, self.phases):
            script += "{:7.4f}\n".format(energy)

            for elem_row in phases:
                # Always write out 16 phases for TLEED formatting, no matter what
                elem_row = elem_row.tolist() + [0] * max(0, 16 - len(elem_row))
                for i, phas in enumerate(elem_row):
                    script += "{:7.4f}".format(phas)
                    # Wrap phases to next line every 10 phases
                    if i % 10 == 9:
                        script += "\n"
                script += "\n"
        return script

    def trunc(self, trunc_lmax) -> Phaseshifts:
        new_phases = self.phases[:, :, :trunc_lmax+1]
        return Phaseshifts(
            self.filename, self.energies, new_phases
        )


""" Parse a set of phaseshifts from a file. The maximum angular momentum number contained
     in the file must be specified due to the ambiguous format of phaseshift files.
    This function is compatible with the Fortran 10-phases-per-line formatting, or a more
     sane format with each t-matrix on its own line.
"""
def parse_phaseshifts(filename: str, num_el: int, l_max: int) -> Phaseshifts:
    try:
        with open(filename, 'r') as f:
            contents = f.readlines()
        phaseshifts = _parse_phaseshifts_str(contents, num_el, l_max)
        phaseshifts.filename = filename
    except Exception as e:
        logging.error("Error in parsing phaseshifts file {}!".format(filename))
        raise e

    return phaseshifts


def _parse_phaseshifts_str(lines: List[str], num_el: int, l_max: int) -> Phaseshifts:
    energies = []
    # Will become a triply-nested list of dimensions [NENERGIES, NELEM, NANG_MOM],
    #  converted to a numpy array at the end
    phases = []

    try:
        linenum = 0
        while linenum < len(lines):
            line = lines[linenum]
            energies.append(float(line))
            elem_phases = [[] for _ in range(num_el)]
            for n in range(num_el):
                linenum += 1
                line = lines[linenum]
                num_phases_line = len(line) // 7
                elem_phases[n] = [float(line[7*i:7*(i+1)]) for i in range(num_phases_line)]
                # Detect weird Fortran formatting
                if num_phases_line != (l_max + 1):
                    linenum += 1
                    line = lines[linenum]
                    for i in range(l_max + 1 - num_phases_line):
                        elem_phases[n].append(float(line[7*i:7*(i+1)]))
                # Truncate phases to lmax+1
                elem_phases[n] = elem_phases[n][:l_max+1]
            phases.append(elem_phases)
            linenum += 1
    except Exception as e:
        # Re-raise the exception with a line number
        logging.error("Error on line number " + str(linenum) + "\n")
        raise e

    return Phaseshifts("", np.array(energies), np.array(phases))


class CalcState(enum.Enum):
    """ Enumeration defining the current state of a calculation
    """
    INIT       = enum.auto()  # Defined, but has not been run yet
    RUNNING    = enum.auto()  # Currently running
    COMPLETED  = enum.auto()  # Successfully completed
    TERMINATED = enum.auto()  # Terminated by some signal


class RefCalc:
    """ Class representing a single reference calculation, responsible for orchestrating the
        necessary scripts to run the calculation, as well as keeping track of Tensors needed
        for perturbative calculations
    """
    def __init__(self, struct: AtomicStructure, phaseshifts: Phaseshifts,
                 beaminfo: BeamInfo, beamlist: BeamList, leed_exe: str,
                 workdir: str, produce_tensors=False, epsilon=1e-3, layer_iter=8,
                 decay_thresh=1e-4, name="Reference Calculation"):
        self.struct = struct
        self.phaseshifts = phaseshifts
        self.beaminfo = beaminfo
        self.beamlist = beamlist
        self.leed_exe = os.path.abspath(leed_exe)
        self.workdir = os.path.abspath(workdir)
        self.produce_tensors = produce_tensors
        self.epsilon = epsilon            # EPS in input file
        self.layer_iter = layer_iter      # LITER in input file
        self.decay_thresh = decay_thresh  # TST in input file
        self.name = name
        self.tensorfiles = []
        if produce_tensors:
            self.tensorfiles = [
                os.path.join(self.workdir, "LAY{}_{}".format(i+1, j+1))
                for i in range(len(self.struct.layers) - 1)
                for j in range(len(self.struct.layers[i]))
            ]

        # Check that beams in BeamInfo are contained in BeamList, and record their indices
        self.beam_idxs = []
        for beam in self.beaminfo:
            try:
                self.beam_idxs.append(self.beamlist.beams.index(beam))
            except ValueError:
                # Re-raise the error, just with a bit more information
                raise ValueError(
                    "Beam {} present in BeamInfo not contained in given BeamList.".format(beam)
                )

        self.state = CalcState.INIT

        self.script_filename = os.path.join(self.workdir, "FIN")
        self.result_filename = os.path.join(self.workdir, "fd.out")
        self._process = None

    def _write_script(self, filename):
        with open(filename, "w") as ofile:
            # File title and energy range
            ofile.write(self.name + "\n")
            ofile.write("{:>7.2f}{:>7.2f}{:>7.2f}\n".format(
                self.beaminfo.energy_min, self.beaminfo.energy_max, self.beaminfo.energy_step
            ))

            # Bulk vectors
            ofile.write("{:>7.4f} 0.0000          ARA1 *\n".format(self.struct.cell_params[0]))
            ofile.write(" 0.0000{:>7.4f}          ARA2 *\n".format(self.struct.cell_params[1]))

            # (Unused) registry shift lines
            ofile.write(
                " 0.0    0.0             SS1\n"
                " 0.0    0.0             SS2\n"
                " 0.0    0.0             SS3\n"
                " 0.0    0.0             SS4\n"
            )

            # Surface lattice vectors
            ofile.write("{:>7.4f} 0.0000          ARB1 *\n".format(self.struct.cell_params[0]))
            ofile.write(" 0.0000{:>7.4f}          ARB2 *\n".format(self.struct.cell_params[1]))

            # (Unused) registry shift lines
            ofile.write(
                " 0.0    0.0             SO1\n"
                " 0.0    0.0             SO2\n"
                " 0.0    0.0             SO3\n"
            )
            # Unused FR parameter, and ASE = distance to vacuum from top layer
            # TODO: Should ASE be smartly set based on the structure?
            ofile.write(" 0.5    1.2237          FR ASE\n")

            # Write out beamlist
            ofile.write(self.beamlist.to_script())

            # Various simulation parameters
            ofile.write("{:>7.4f}".format(self.decay_thresh) + 20 * " " + "TST\n")
            for idx in self.beam_idxs:
                ofile.write("{:>3d}".format(idx+1))
            ofile.write(12 * " " + "NPU(K)\n")
            ofile.write("{:>6.1f} {:>6.1f}".format(self.beaminfo.theta, self.beaminfo.phi))
            ofile.write(16 * " " + "THETA PHI\n")
            ofile.write("{:>6.3f}".format(self.epsilon) + 21 * " " + "EPS\n")
            ofile.write("{:>3d}".format(self.layer_iter) + 24 * " " + "LITER\n")
            ofile.write("{:>3d}".format(self.phaseshifts.lmax) + 24 * " " + "LMAX\n")
            ofile.write("{:>3d}".format(self.phaseshifts.num_elem) + 24 * " " + "NEL\n")

            # Phaseshift table
            ofile.write(self.phaseshifts.to_script())

            # Specifying output beams
            ofile.write("   1               IFORM - ASCII output of tensor components\n")
            for beam in self.beaminfo:
                ofile.write("{:>10.5f}{:>10.5f}\n".format(beam[0], beam[1]))

            # Site description section
            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- define chem. and vib. properties for different atomic sites ---\n"
                "-------------------------------------------------------------------\n"
            )
            ofile.write("{:>3d}".format(len(self.struct.sites)))
            ofile.write(23 * " " + "NSITE: number of different site types\n")
            for i, site in enumerate(self.struct.sites):
                ofile.write("-   site type {}  {}---\n".format(i + 1, site.name))
                ofile.write(site.to_script())

            # Layer description section
            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- define different layer types                            *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            ofile.write("{:>3d}".format(len(self.struct.layers)))
            ofile.write(23 * " " + "NLTYPE: number of different layer types\n")
            for i, layer in enumerate(self.struct.layers):
                ofile.write("-   layer num {}  {}---\n".format(i + 1, layer.name))
                ofile.write("{:>3d}".format(layer.lay_type.value))
                ofile.write(23 * " " + "LAY = {}: {} periodicity\n".format(
                    layer.lay_type.value, layer.lay_type.name
                ))
                ofile.write(layer.to_script(self.struct.cell_params))

            num_layers = len(self.struct.layers)

            # Bulk stacking section
            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- define bulk stacking sequence                           *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            # List holds (layer_num, layer) for all bulk layers
            bulk_layers = [
                (i+1, lay) for i, lay in enumerate(self.struct.layers)
                if lay.lay_type == LayerType.BULK
            ]
            double_bulk = len(bulk_layers) == 2
            top_layer_num = bulk_layers[0][0]
            bulk_intraunit_vec = bulk_layers[0][1].interlayer_vec
            if double_bulk:
                bulk_interunit_vec = bulk_layers[1][1].interlayer_vec
                bot_layer_num = bulk_layers[1][0]
            else:
                bulk_interunit_vec = bulk_layers[0][1].interlayer_vec
                bot_layer_num = bulk_layers[0][0]

            ofile.write("  0" + 23 * " " + "TSLAB = 0: compute bulk using subras\n")
            ofile.write("{:>7.4f}{:>7.4f}{:>7.4f}".format(
                bulk_interunit_vec[2], *bulk_interunit_vec[:2]
            ))
            ofile.write("     ASA interlayer vector between different bulk units *\n")
            ofile.write("{:>3d}".format(top_layer_num))
            ofile.write(23 * " " + "top layer of bulk unit: num {}\n".format(top_layer_num))
            ofile.write("{:>3d}".format(bot_layer_num))
            ofile.write(23 * " " + "bottom layer of bulk unit: num {}\n".format(bot_layer_num))
            ofile.write("{:>7.4f}{:>7.4f}{:>7.4f}".format(
                bulk_intraunit_vec[2], *bulk_intraunit_vec[:2]
            ))
            ofile.write("     ASBULK between the two bulk unit layers (may differ from ASA)\n")

            # Surface layer stacking sequence
            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- define layer stacking sequence and Tensor LEED output   *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            # List holds (layer_num, layer) for all surface layers
            surf_layers = [
                (i+1, lay) for i, lay in enumerate(self.struct.layers)
                if lay.lay_type == LayerType.SURF
            ]
            # Need to write out surface layers in reverse order
            surf_layers.reverse()
            num_surf_layers = len(surf_layers)
            ofile.write("{:>3d}".format(num_surf_layers))
            ofile.write(23 * " " + "Number of surface layers(?)\n")
            for layer_num, layer in surf_layers:
                ofile.write("{:>3d}{:>7.4f}{:>7.4f}{:>7.4f}".format(
                    layer_num, layer.interlayer_vec[2], *layer.interlayer_vec[:2]
                ))
                ofile.write("  this layer is of kind {}: ".format(layer_num) +
                            "interlayer vector connecting it to bulk\n")
                if self.produce_tensors:
                    ofile.write("  1" + 23 * " " + "Tensor output is required for this layer\n")
                    for j in range(len(layer)):
                        ofile.write("LAY{}_{}".format(layer_num, j + 1) + 22 * " " +
                                    "Tensorfile, layer {}, sublayer {}\n".format(layer_num, j+1))
                else:
                    ofile.write("  0" + 23 * " " + "Tensor output is NOT required for this layer\n")

            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- end geometrical input                                       ---\n"
                "-------------------------------------------------------------------\n"
            )

    def run(self):
        """ Starts a process in the background to run this reference calculation.
            Non-blocking. Call wait() to block until completion.
        """
        self._write_script(self.script_filename)
        stdout_filename = os.path.join(os.path.dirname(self.script_filename), "log.txt")
        process = subprocess.Popen(
            [self.leed_exe],
            stdin=open(self.script_filename, "r"),
            stdout=open(stdout_filename, "w"),
            cwd=self.workdir,
            text=True
        )
        self._process = process
        self.state = CalcState.RUNNING

    def wait(self) -> CalcState:
        """ Waits for completion.
        """
        completion = self._process.wait()
        if completion < 0:
            self.state = CalcState.TERMINATED
            logging.error(
                "Reference calculation {} was terminated!".format(self.script_filename)
            )
            raise RuntimeError(
                "Reference calculation {} was terminated!".format(self.script_filename)
            )
        self.state = CalcState.COMPLETED
        return self.state

    def poll(self) -> CalcState:
        completion = self._process.poll()
        if completion is not None:
            if completion < 0:
                self.state = CalcState.TERMINATED
                logging.error(
                    "Reference calculation {} was terminated!".format(self.script_filename)
                )
                raise RuntimeError(
                    "Reference calculation {} was terminated!".format(self.script_filename)
                )
            self.state = CalcState.COMPLETED
        return self.state

    def produce_curves(self) -> IVCurveSet:
        """ Produce the set of IV curves resulting from the reference calculation.
            If the calculation has not been run yet, or is still running, wait
             for completion.
        """
        if self.state is CalcState.INIT:
            self.run()
        if self.state is CalcState.RUNNING:
            self.wait()
        return parse_ivcurves(self.result_filename, format='RCOUT')

    def rfactor(self, exp_curves: IVCurveSet, **kwargs) -> float:
        calc_curves = self.produce_curves()
        calc_rfactor = min(avg_rfactors(exp_curves, calc_curves, **kwargs))
        return calc_rfactor


def parse_ref_calc(filename: str) -> RefCalc:
    """ Parses a RefCalc from the input script. Checks for completion by the presence
         of a fd.out file in the same directory as the script. Attempts to look for
         tensors from completed run.
        Warning: Some subfields will have incorrect filenames because this information
         is not present in the input file (e.g. ref_calc.phaseshifts.filename)
    """
    try:
        with open(filename, 'r') as f:
            contents = f.readlines()
        ref_calc = _parse_ref_calc_str(contents)
        ref_calc.workdir = os.path.abspath(os.path.dirname(filename))
        ref_calc.script_filename = os.path.join(ref_calc.workdir, os.path.basename(filename))
        ref_calc.result_filename = os.path.join(ref_calc.workdir, "fd.out")
        ref_calc.tensorfiles = [
            os.path.abspath(os.path.join(ref_calc.workdir, fname)) for fname in ref_calc.tensorfiles
        ]
        if os.path.exists(ref_calc.result_filename):
            ref_calc.state = CalcState.COMPLETED
    except Exception as e:
        logging.error("Error in parsing reference calc file {}.".format(filename))
        raise e
    return ref_calc


# TODO: Needs testing that AtomicStructure parsed in is correct
# Beware all ye who enter below
def _parse_ref_calc_str(lines: List[str]) -> RefCalc:
    name = lines[0][:-1]
    initE, finalE, stepE = map(float, lines[1].split())
    lat_vec_a = np.array(list(map(float, lines[2].split()[:2])))
    lat_vec_b = np.array(list(map(float, lines[3].split()[:2])))
    # Lines 4, 5, 6, 7 unused
    # Lines 8, 9 identical to 2, 3 for me currently
    # Lines 10, 11, 12 unused
    fr, ase = map(float, lines[13].split()[:2])

    beamlist_len = int(lines[14])
    beamlist = _parse_beamlist_str(lines[14:14+beamlist_len+1])
    linenum = 14 + beamlist_len + 1

    decay_thresh = float(lines[linenum][:7])
    linenum += 1
    # This will break if beam idxs get to three digits
    beam_idx_line = lines[linenum]
    beam_idxs = []
    # Gross
    try:
        for i in range(len(beam_idx_line) // 3):
            beam_idxs.append(int(beam_idx_line[3*i:3*(i+1)]))
            i += 1
    except ValueError:
        pass
    num_beams = len(beam_idxs)

    beam_idxs = [int(beam_idx_line[3*i:3*(i+1)]) for i in range(num_beams)]
    linenum += 1
    theta, phi = map(float, lines[linenum].split()[:2])
    linenum += 1
    epsilon = float(lines[linenum][:6])
    linenum += 1
    layer_iter = int(lines[linenum][:3])
    linenum += 1
    lmax = int(lines[linenum][:3])
    linenum += 1
    num_elem = int(lines[linenum][:3])
    linenum += 1

    # Search for the end of the phaseshifts
    phaseshifts_start_idx = linenum
    line = lines[linenum]
    while line[:3] != "   ":
        linenum += 1
        line = lines[linenum]

    phaseshifts = _parse_phaseshifts_str(lines[phaseshifts_start_idx:linenum], lmax)

    linenum += 1
    beams = [(int(float(l[:10])), int(float(l[10:20]))) for l in lines[linenum:linenum+num_beams]]
    beaminfo = BeamInfo(theta, phi, beams, initE, finalE, stepE)
    linenum += num_beams

    # Skip 3 header lines -- TODO: Make this auto-detected
    sites = []
    linenum += 3
    nsites = int(lines[linenum][:3])
    linenum += 1
    for _ in range(nsites):
        linenum += 1
        concs, vibs, elems = [], [], []
        for _ in range(num_elem):
            line = lines[linenum]
            concs.append(float(line[:7]))
            vibs.append(float(line[7:14]))
            elems.append(line[-2:])
            linenum += 1
        # TODO: Different vibs?
        sites.append(Site(concs, vibs[-1], elems))

    linenum += 3
    nlayer = int(lines[linenum][:3])
    layers = []
    linenum += 1
    for _ in range(nlayer):
        linenum += 2
        numsublayer = int(lines[linenum][:3])
        linenum += 1
        atoms = []
        for _ in range(numsublayer):
            sitenum = int(lines[linenum][:3])
            z = float(lines[linenum][3:10])
            x = float(lines[linenum][10:17])
            y = float(lines[linenum][17:24])
            sitenum = int(sitenum)
            atoms.append(Atom(sitenum, x, y, z))
            linenum += 1
        layers.append(Layer(atoms))

    linenum += 7
    bulk_interlayer_dist = float(lines[linenum][:7])

    linenum += 5
    surf_interlayer_dist = float(lines[linenum][3:10])

    cell_a, cell_b = lat_vec_a[0], lat_vec_b[1]
    # TODO: This is certainly not true in general
    cell_c = bulk_interlayer_dist

    # Normalize all Layer coordinates
    for layer in layers:
        layer.xs /= cell_a
        layer.ys /= cell_b
        layer.zs /= cell_c

    struct = AtomicStructure(sites, layers, [cell_a, cell_b, cell_c])

    linenum += 1
    produce_tensors = bool(int(lines[linenum][:3]))
    linenum += 1
    tensorfiles = [l.split()[0] for l in lines[linenum:linenum+len(layers[0])]]

    ref_calc = RefCalc(struct, phaseshifts, beaminfo, beamlist, "", "",
                       produce_tensors=produce_tensors, epsilon=epsilon,
                       layer_iter=layer_iter, decay_thresh=decay_thresh)
    ref_calc.tensorfiles = tensorfiles
    return ref_calc


class DeltaCalc:
    """ Class representing a single perturbative calculation away from a reference calculation.
        Note that structure is not very `natural` from the perspective of the TensErLEED program,
         which preferes to return results for a whole grid of perturbations at the same time.
        However, this class exposes an identical API to RefCalc, and so makes it easier to set
         up an optimization loop which uses both calculations.
    """
    def __init__(self, struct: AtomicStructure, ref_calc: RefCalc,
                 delta_exe: str, search_vibs: Collection[float] = None):
        """ Initialize a calculation of the given AtomicStructure, viewed as a perturbation from
             the given RefCalc.
            Note: The cell_params must be identical between struct and ref_calc.struct!
        """
        if not np.all(struct.cell_params == ref_calc.struct.cell_params):
            raise ValueError(
                "TLEED calculations must be made on a structure with identical"
                " unit cell parameters as the reference structure perturbed from"
            )
        if not len(struct.layers) == len(ref_calc.struct.layers):
            raise ValueError(
                "TLEED calculations must be made on a structure with the same number"
                " of layers as the reference structure perturbed from"
            )
        if not ref_calc.produce_tensors:
            raise ValueError(
                "Tensors are required to be output by the reference calc to compute deltas!"
            )

        self.struct = struct
        self.ref_calc = ref_calc
        self.delta_exe = delta_exe
        self.state = CalcState.INIT
        self._script_paths: List[str] = []
        self._processes: List[subprocess.Popen] = []

        # Determine the delta displacements which brings the ref_calc to the target struct
        self._disps, self._vibs = [], []
        a, b, c = self.struct.cell_params
        for ref_layer, delta_layer in zip(ref_calc.struct.layers, struct.layers):
            for ref_atom, delta_atom in zip(ref_layer, delta_layer):
                self._disps.append(np.array([
                    a * (delta_atom.x - ref_atom.x),
                    b * (delta_atom.y - ref_atom.y),
                    c * (delta_atom.z - ref_atom.z)
                ]))
                self._vibs.append(
                    struct.sites[delta_atom.sitenum].vib
                )

    def _write_scripts(self, directory: Optional[str] = None) -> List[str]:
        """ Writes one script into the target directory for each atom which we need to perturb.
            Returns a list of the script paths.
        """
        directory = self.ref_calc.workdir if directory is None else directory
        script_paths = []
        for iatom, atom in enumerate(self.struct.layers[0]):
            subworkdir = os.path.join(directory, "delta_tmp" + str(iatom + 1))
            try:
                os.mkdir(subworkdir)
            except FileExistsError:
                pass
            script_filename = os.path.join(subworkdir, "delta{}.in".format(iatom + 1))

            # Determine which element the site of this atom is.
            # Unsure how this extends to handle concentration variation.
            site_num = atom.sitenum
            elem_num = np.argmax(self.ref_calc.struct.sites[site_num].concs) + 1
            disp = self._disps[iatom]
            vib = self._vibs[iatom]

            ref_calc = self.ref_calc
            phaseshifts = self.ref_calc.phaseshifts
            beaminfo = self.ref_calc.beaminfo
            with open(script_filename, "w") as f:
                f.write("Delta Script\n")
                f.write("{:>7.2f}{:>7.2f}\n".format(beaminfo.energy_min, beaminfo.energy_max))
                f.write("{:>7.4f}{:>7.4f}\n".format(ref_calc.struct.cell_params[0], 0.0))
                f.write("{:>7.4f}{:>7.4f}\n".format(0.0, ref_calc.struct.cell_params[1]))
                f.write("{:>7.4f}{:>7.4f}\n".format(ref_calc.struct.cell_params[0], 0.0))
                f.write("{:>7.4f}{:>7.4f}\n".format(0.0, ref_calc.struct.cell_params[1]))
                f.write("{:>7.2f}{:>7.2f}\n".format(beaminfo.theta, beaminfo.phi))
                f.write("   1\n")
                for beamx, beamy in beaminfo.beams:
                    f.write("{:>10.5f}{:>10.5f}\n".format(beamx, beamy))
                f.write("{:>3d}\n".format(ref_calc.struct.num_elems))
                f.write(phaseshifts.to_script())
                f.write("   1\n")
                f.write("-------------------------------------------------------------------\n"
                        "--- chemical nature of displaced atom                           ---\n"
                        "-------------------------------------------------------------------\n")
                f.write("{:>4d}\n".format(elem_num))
                f.write("-------------------------------------------------------------------\n"
                        "--- reference undisplaced position of site                      ---\n"
                        "-------------------------------------------------------------------\n")
                f.write(" 0.0000 0.0000 0.0000\n")
                f.write("-------------------------------------------------------------------\n"
                        "--- displaced positions of atomic site in question              ---\n"
                        "-------------------------------------------------------------------\n")
                # Number of displacements -- just one
                f.write("   1\n")
                f.write("{:>7.4f}{:>7.4f}{:>7.4f}\n".format(disp[2], disp[0], disp[1]))
                f.write("-------------------------------------------------------------------\n"
                        "--- vibrational displacements of atomic site in question        ---\n"
                        "-------------------------------------------------------------------\n")
                # Number of vibrational parameters -- just one
                f.write("   1\n")
                f.write("{:>7.4f}\n".format(vib))

            script_paths.append(script_filename)

        self._script_paths = script_paths
        return script_paths

    def run(self, directory: Optional[str] = None):
        self.state = CalcState.RUNNING
        script_paths = self._write_scripts(directory=directory)
        for iatom, script_path in enumerate(script_paths):
            scriptdir = os.path.dirname(script_path)

            # Create a symlink to the correct tensor from the ref calc
            amp_path = os.path.join(scriptdir, "AMP")
            if os.path.exists(amp_path):
                os.remove(amp_path)
            os.symlink(self.ref_calc.tensorfiles[iatom], amp_path)

            self._processes.append(subprocess.Popen(
                [self.delta_exe],
                stdin=open(script_path, "r"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=scriptdir,
                text=True
            ))

    def wait(self) -> CalcState:
        any_terminated = False
        for p in self._processes:
            completion = p.wait()
            any_terminated |= completion < 0
        if any_terminated:
            self.state = CalcState.TERMINATED
            logging.error(
                "Delta calculation {} was terminated!".format(self.delta_exe)
            )
            raise RuntimeError(
                "Reference calculation {} was terminated!".format(self.delta_exe)
            )
        else:
            self.state = CalcState.COMPLETED
        return self.state

    def poll(self) -> CalcState:
        """ Polls to check if ALL delta sub-calculations are done.
            Returns CalcState.RUNNNING if any calculations still ongoing.
        """
        if self.state == CalcState.INIT:
            return CalcState.INIT
        for p in self._processes:
            completion = p.poll()
            if completion is not None:
                if completion < 0:
                    self.state = CalcState.TERMINATED
                    logging.error(
                        "Delta calculation {} was terminated!".format(self.delta_exe)
                    )
                    raise RuntimeError(
                        "Reference calculation {} was terminated!".format(self.delta_exe)
                    )
            else:
                return CalcState.RUNNING
        self.state = CalcState.COMPLETED
        return self.state

    def produce_curves(self) -> IVCurveSet:
        if self.state is CalcState.INIT:
            self.run()
        if self.state is CalcState.RUNNING:
            self.wait()
        # Load all of the MultiDeltaAmps
        delta_amps = []
        for iatom, script_path in enumerate(self._script_paths):
            script_dir = os.path.dirname(script_path)
            delta_amps.append(parse_deltas(os.path.join(script_dir, "DELWV")))

        multi_amps = MultiDeltaAmps(delta_amps)

        return multi_amps.compute_curves([0] * len(multi_amps))


class SiteDeltaAmps:
    """ Class representing a set of delta amplitudes for perturbations of a single site
        produced by delta.f """
    def __init__(self):
        # All of this should be manually initialized, in parse_deltas
        self.theta, self.phi = 0.0, 0.0
        self.recip_a, self.recip_b = np.zeros(2), np.zeros(2)     # RAR1, RAR2
        self.nbeams = 0
        self.natoms = 0  # Unused by TensErLEED, will always be read in as 1
        self.nshifts = 0
        self.nvibs = 0
        self.beams = np.empty((0, 2))
        self.shifts = np.empty((0, 3))   # NOTE: These are z x y in file, but x y z here
        self.thermal_amps = np.empty(1)
        self.crystal_energies = np.empty(0)                       # E
        self.substrate_energies = np.empty(0, dtype=np.complex64) # VV + VPI j
        self.overlayer_energies = np.empty(0, dtype=np.complex64) # VO
        self.real_energies_ev = np.empty(0)                       # (E - VV) to eV
        self.ref_amplitudes = np.empty((0, 0), np.complex64)
        self.delta_amplitudes = np.empty((0, 0, 0, 0), np.complex64)


class MultiDeltaAmps:
    """ Class holding all of the delta amplitudes for a set of sites, and
         can compute modified IV curves.
        TODO: Support different sets of displacements for different sites.
    """
    def __init__(self, delta_amps_list: List[SiteDeltaAmps]):
        self.nsites = len(delta_amps_list)
        self.nbeams = delta_amps_list[0].nbeams
        self.ngeo = delta_amps_list[0].nshifts
        self.nvibs = delta_amps_list[0].nvibs
        self.energies = delta_amps_list[0].real_energies_ev
        self.beam_labels = delta_amps_list[0].beams

        # Check all delta amps correspond to the same reference calculation
        self.ref_amplitudes = delta_amps_list[0].ref_amplitudes
        for i in range(1, self.nsites):
            assert np.allclose(self.ref_amplitudes, delta_amps_list[i].ref_amplitudes)

        self.delta_amps_list = delta_amps_list

        # Compute propagators at every energy in the range
        self.propagators = np.empty((self.nbeams, len(self.energies)))
        # Assumes theta, fi, energy params, recip vectors same across all SiteDeltaAmps
        # All of below is more or less directly copied from lib.tleed.f
        amps = self.delta_amps_list[0]
        Ak = np.sqrt(
            np.maximum(2.0 * (amps.crystal_energies - np.real(amps.substrate_energies)), 0.0)
        )
        C = Ak * np.cos(amps.theta)
        Bk2 = Ak * np.sin(amps.theta) * np.cos(amps.phi)
        Bk3 = Ak * np.sin(amps.theta) * np.sin(amps.phi)
        Bkz = np.sqrt(
            2.0 * (amps.crystal_energies - amps.overlayer_energies)
          - Bk2 * Bk2
          - Bk3 * Bk3
          - 2.0j * np.imag(amps.substrate_energies)
        )

        for ib in range(self.nbeams):
            Ak2 = (Bk2 + self.beam_labels[ib, 0] * amps.recip_a[0]
                       + self.beam_labels[ib, 1] * amps.recip_b[0])
            Ak3 = (Bk3 + self.beam_labels[ib, 0] * amps.recip_a[1]
                       + self.beam_labels[ib, 1] * amps.recip_b[1])
            Ak = 2.0 * amps.crystal_energies - Ak2 * Ak2 - Ak3 * Ak3
            Akz = np.sqrt(
                Ak - 2.0 * amps.overlayer_energies - 2.0j * np.imag(amps.substrate_energies)
            )
            Aperp = np.maximum(Ak - 2.0 * np.real(amps.substrate_energies), 0.0)
            safe_idxs = Aperp > 0.0
            self.propagators[ib][safe_idxs] = np.sqrt(Aperp[safe_idxs]) / C[safe_idxs]

    def __len__(self):
        return self.nsites

    def __iter__(self):
        return iter(self.delta_amps_list)

    def __getitem__(self, idx: int):
        return self.delta_amps_list[idx]

    def compute_curves(self, disps) -> IVCurveSet:
        new_amplitudes = self.ref_amplitudes.copy()
        for isite, disp in enumerate(disps):
            ivib, igeo = divmod(disp, self.ngeo)
            new_amplitudes += self.delta_amps_list[isite].delta_amplitudes[:, ivib, igeo]

        curves = [IVCurve(
            self.energies,
            np.abs(new_amplitudes[ibeam])**2 * self.propagators[ibeam],
            self.beam_labels[ibeam]
        ) for ibeam in range(self.nbeams)]

        return IVCurveSet(curves)


def parse_deltas(filename: str) -> SiteDeltaAmps:
    """ Parses a DELWV file output by delta.f to create a SiteDeltaAmps object.
        Note all energies read in are in Hartrees! (Conversion: 1 Hartree = 27.21 eV).
        To get "energy" to compare to experiment:
            27.21 * (crystal_energy - np.real(substrate_energy))
    """
    try:
        with open(filename, "r") as f:
            contents = f.readlines()
        delta_amp = _parse_deltas_str(contents)
    except Exception as e:
        logging.error("Error when parsing delta file {}!".format(filename))
        # Re-raise exception
        raise e

    return delta_amp


def _parse_deltas_str(lines: List[str]) -> SiteDeltaAmps:
    """ Parses a SiteDeltaAmps from the contents of a DELWV file.
    """
    delta_amp = SiteDeltaAmps()
    linenum = 0
    line = lines[linenum]

    delta_amp.theta = float(line[:13])
    delta_amp.phi = float(line[13:26])
    delta_amp.recip_a[0] = float(line[26:39])
    delta_amp.recip_a[1] = float(line[39:52])
    delta_amp.recip_b[0] = float(line[52:65])
    delta_amp.recip_b[1] = float(line[65:78])

    linenum += 1
    line = lines[linenum]
    delta_amp.nbeams = int(line[:3])
    delta_amp.natoms = int(line[3:6])
    numdeltas = int(line[6:9])
    delta_amp.beams = np.empty((delta_amp.nbeams, 2))
    deltas = np.empty((numdeltas, 3))

    for i in range(delta_amp.nbeams):
        # Only 5 beams per line
        if i % 5 == 0:
            linenum += 1
            line = lines[linenum]
        idx = i % 5
        delta_amp.beams[i, 0] = float(line[20*idx:10+20*idx])
        delta_amp.beams[i, 1] = float(line[10+20*idx:20+20*idx])

    # Skip line of 0.0's from an unused feature in TensErLEED
    linenum += 2

    # Read in the list of considered shifts
    # TODO: This is nasty.
    n = 0
    comp = 0
    while n < numdeltas:
        line = lines[linenum].rstrip('\n')
        for i in range(len(line) // 7):
            deltas[n, (comp + 2) % 3] = float(line[7*i:7*(i+1)])
            if comp == 2:
                n += 1
            comp = (comp + 1) % 3
        linenum += 1

    # Remove duplicate info
    first_delta = deltas[0]
    for n, delta in enumerate(deltas[1:]):
        if np.allclose(first_delta, delta):
            delta_amp.nshifts = (n + 1)
            delta_amp.nvibs = numdeltas // (n + 1)
            break
    else:
        delta_amp.nshifts = numdeltas
        delta_amp.nvibs = 1

    delta_amp.shifts = deltas[:delta_amp.nshifts].copy()
    delta_amp.thermal_amps = np.empty(numdeltas)

    # Read in the list of thermal vibrational amplitudes (?)
    n = 0
    while n < numdeltas:
        line = lines[linenum].rstrip('\n')
        for i in range(len(line) // 7):
            delta_amp.thermal_amps[n] = line[7*i:7*(i+1)]
            n += 1
        linenum += 1

    # Read in the crystal potential energies
    crystal_energies = []
    substrate_energies = []
    overlayer_energies = []
    real_energies_ev = []
    all_ref_amplitudes = []
    all_delta_amplitudes = []
    line = lines[linenum]
    # Each iteration of this is a single energy
    nit = 0
    while linenum < len(lines):
        line = lines[linenum]
        crystal_energy = float(line[:13])
        substrate_energy = float(line[13:26]) * 1j
        overlayer_energy = float(line[26:39])
        substrate_energy += float(line[39:52])
        crystal_energies.append(crystal_energy)
        substrate_energies.append(substrate_energy)
        overlayer_energies.append(overlayer_energy)
        # This is the real energy to compare to experiment
        real_energies_ev.append(round(27.21 * (crystal_energy - substrate_energy.real)))

        # Read in the original reference calculation amplitudes
        ref_amplitudes = np.empty(delta_amp.nbeams, np.complex64)
        n = 0
        while n < delta_amp.nbeams:
            linenum += 1
            line = lines[linenum]
            for i in range(len(line) // 26):
                ref_amplitudes[n] = float(line[26*i:26*i+13])
                ref_amplitudes[n] += float(line[26*i+13:26*i+26]) * 1j
                n += 1
        all_ref_amplitudes.append(ref_amplitudes)

        # Read in the delta amplitudes for each search delta
        delta_amplitudes = np.empty((delta_amp.nbeams, delta_amp.nvibs, delta_amp.nshifts), np.complex64)
        n = 0
        while n < delta_amp.nbeams * delta_amp.nvibs * delta_amp.nshifts:
            linenum += 1
            line = lines[linenum]
            for i in range(len(line) // 26):
                delta_idx, beam_idx = divmod(n, delta_amp.nbeams)
                vib_idx, disp_idx = divmod(delta_idx, delta_amp.nshifts)
                delta_amplitudes[beam_idx, vib_idx, disp_idx] = float(line[26*i:26*i+13])
                delta_amplitudes[beam_idx, vib_idx, disp_idx] += float(line[26*i+13:26*i+26]) * 1j
                n += 1
        all_delta_amplitudes.append(delta_amplitudes)
        nit += 1
        linenum += 1

    delta_amp.crystal_energies = np.array(crystal_energies)
    delta_amp.substrate_energies = np.array(substrate_energies)
    delta_amp.overlayer_energies = np.array(overlayer_energies)
    delta_amp.real_energies_ev = np.array(real_energies_ev)
    delta_amp.ref_amplitudes = np.stack(all_ref_amplitudes, axis=-1)
    delta_amp.delta_amplitudes = np.stack(all_delta_amplitudes, axis=-1)

    return delta_amp


Calc = Union[RefCalc, DeltaCalc]


class LEEDManager:
    def __init__(self, workdir: str, tleed_dir: str, exp_curves: IVCurveSet,
                 phaseshifts: Phaseshifts, beaminfo: BeamInfo, beamlist: BeamList):
        """ Create a LEEDManager to keep track of TensErLEED components, and orchestrate parallel
             executions of multiple calculations. Only one of these should exist per problem.
                basedir: The base directory to do computation in
                leed_executable: Path to the LEED executable
                rfactor_executable: Path to the rfactor executable
                exp_datafile: Path to the experimental datafile
                phss_datafile: Path to the phaseshifts file
                tleed_dir: Path to the base directory of TLEED
            In general, operations on this class are NOT thread-safe. Create an instance
             of this class on a single (main) thread and let it handle asynchronous executions.
        """
        for path in [workdir, tleed_dir]:
            if not os.path.exists(path):
                raise ValueError("Directory not found: {}".format(path))
        self.workdir = os.path.abspath(workdir)
        # These will be compiled and set as needed - they may not exist at this point
        self._ref_exe = os.path.join(self.workdir, "ref-calc.x")
        self._delta_exe = os.path.join(self.workdir, "delta.x")
        self._ref_compiled = False
        self._delta_compiled = False

        self.tleed_dir = os.path.abspath(tleed_dir)
        self.phaseshifts: Phaseshifts = phaseshifts
        self.beaminfo: BeamInfo = beaminfo
        self.beamlist: BeamList = beamlist
        self.exp_curves: IVCurveSet = exp_curves

        self.completed_calcs: List[Tuple[Calc, float]] = []
        self.completed_refcalcs: List[Tuple[RefCalc, float]] = []
        self.completed_deltacalcs: List[Tuple[DeltaCalc, float]] = []
        self.calc_number = 0
        self.active_calcs: List[Calc] = []

    def change_delta_exe(self, ndisps: int, nvibs: int,
                         compiler: str = "gfortran", options: List[str] = None):
        """ Reset self._delta_exe to one which is prepared for a larger number of disps/vibs
        """
        self._compile_delta_program(self._delta_exe, ndisps, nvibs, compiler, options)

    def _spawn_ref_calc(self, structure: AtomicStructure, produce_tensors=False):
        """ Spawns a subprocess running a reference calculation for the given AtomicStructure.
            Adds this subprocess to the manager's list of active calculations.
        """
        # Compile the reference calculation program if this is the first call
        if not self._ref_compiled:
            self._compile_ref_program(structure, self._ref_exe)

        newdir = os.path.join(self.workdir, "ref-calc" + str(self.calc_number))
        os.makedirs(newdir, exist_ok=True)
        ref_calc = RefCalc(structure, self.phaseshifts, self.beaminfo, self.beamlist,
                           self._ref_exe, newdir, produce_tensors=produce_tensors)
        ref_calc.run()
        self.calc_number += 1
        self.active_calcs.append(ref_calc)
        return ref_calc

    def _spawn_delta_calc(self, structure: AtomicStructure, ref_calc: RefCalc):
        """ Spawns a subprocess running a reference calculation for the given AtomicStructure.
            Adds this subprocess to the manager's list of active calculations.
        """
        delta_calc = DeltaCalc(structure, ref_calc, self._delta_exe)
        directory = os.path.join(ref_calc.workdir, "delta_calc" + str(self.calc_number))
        # Make directory if it does not exist
        if not os.path.exists(directory):
            os.mkdir(directory)
        delta_calc.run(directory=directory)
        self.calc_number += 1
        self.active_calcs.append(delta_calc)
        return delta_calc

    def ref_calc_blocking(self, structure: AtomicStructure, produce_tensors=False):
        """ Runs a single reference calculation and blocks until completion.
            Calculations performed with this method are not "kept track of" by the
             manager.
            WARNING: This function is not thread-safe, as there is a race
                condition on self.calc_number. Instead, use batch_ref_calcs.
        """
        refcalc = self._spawn_ref_calc(structure, produce_tensors=produce_tensors)
        calc_curves = refcalc.produce_curves()
        # TODO: imagV a settable parameter somewhere?
        calc_rfactor = min(avg_rfactors(self.exp_curves, calc_curves))
        return calc_rfactor

    def batch_ref_calcs(self, structures: Collection[AtomicStructure], produce_tensors=False):
        """ Starts multiple reference calculations in parallel, one for each
             AtomicStructure. Adds these active processes to the list maintained by
             the manager.
        """
        num_structs = len(structures)

        # Create RefCalc objects for each calculation
        logging.info("Starting {} reference calculations...".format(num_structs))
        refcalcs = [
            self._spawn_ref_calc(struct, produce_tensors=produce_tensors)
            for struct in structures
        ]

        return refcalcs

    def batch_delta_calcs(self, structures: Collection[Tuple[AtomicStructure, RefCalc]]):
        """ Starts multiple TLEED calculations in parallel, one for each pair of
             (AtomicStructure, RefCalc). Adds these active processes to the list
             maintained by the manager.
        """
        num_structs = len(structures)

        # Create DeltaCalc objects for each calculation
        logging.info("Starting {} delta TLEED calculations...".format(num_structs))
        deltacalcs = [
            self._spawn_delta_calc(struct, ref_calc)
            for struct, ref_calc in structures
        ]

        return deltacalcs

    def batch_ref_calc_local_searches(self, structures: Collection[AtomicStructure],
                                      search_dims: List[DeltaSearchDim],
                                      constraints: List[Constraint] = None,
                                      search_epochs: int = 30000, search_indivs: int = 25
                                      ) -> Tuple[List[float], List[AtomicStructure], List[float]]:
        """ Starts (and waits for completion of) multiple reference calculations
             in parallel. Once all are done, computes a local TLEED search within
             the given radii of the reference calc.

            Returns (ref_rfactors, best_delta_structs, best_delta_rfactors)
        """
        num_structs = len(structures)

        # Create RefCalc objects for each calculation
        logging.info("Starting {} reference calculations...".format(num_structs))
        refcalcs = [
            self._spawn_ref_calc(struct, produce_tensors=True)
            for struct in structures
        ]

        # Wait for each to finish
        for calc in refcalcs:
            calc.wait()

        # Calculate r-factors of these initial calculations
        ref_rfactors = [calc.rfactor(self.exp_curves) for calc in refcalcs]
        logging.info("Reference calc rfactors: {}".format(ref_rfactors))
        for calc, rfact in zip(refcalcs, ref_rfactors):
            self.completed_refcalcs.append((calc, rfact))
            self.completed_calcs.append((calc, rfact))

        # Set up the delta search spaces
        search_spaces = [
            DeltaSearchSpace(calc, search_dims, constraints) for calc in refcalcs
        ]

        # Everything used by a multiprocessing.Pool must be pickle-able, so we must
        #  throw away the old processes the ref calcs used
        for space in search_spaces:
            space.ref_calc._process = None

        logging.info("Producing delta amplitudes around {} reference calcs.".format(num_structs))

        # Compile the delta_exe if this has not been done yet
        # This currently assumes that all search dims have the same number of geo/vib disps
        # TODO: Compile separate delta program for each search dimension?
        if not self._delta_compiled:
            self._compile_delta_program(
                self._delta_exe, len(search_dims[0][1]), len(search_dims[0][2])
            )

        # Make MultiDeltaAmps for each, just do this in serial for now
        deltacalcs = [
            self.produce_delta_amps(delta_space, delta_exe=self._delta_exe)
            for delta_space in search_spaces
        ]

        logging.info("Starting local searches around {} reference calcs.".format(num_structs))
        # Run those searches (in parallel)
        with mp.Pool(num_structs) as pool:
            results = pool.starmap(
                optimize_delta_anneal,
                zip(
                    search_spaces,
                    deltacalcs,
                    itertools.repeat(self.exp_curves),
                    itertools.repeat(search_indivs),
                    itertools.repeat(search_epochs),
                )
            )

        # Reconstruct the structures which achieved the minimum of each annealing
        # TODO: Move this inside optimize_delta_anneal?
        delta_structs = []
        delta_rfactors = [r[1] for r in results]
        for i in range(len(structures)):
            (geo_idxs, vib_idxs), delta_rfactor = results[i]
            atom_idxs = search_spaces[i].atoms
            base_struct = search_spaces[i].struct
            delta_struct = copy.deepcopy(base_struct)
            ref_calc = search_spaces[i].ref_calc
            for j, (atom_idx, geo_idx, vib_idx) in enumerate(zip(atom_idxs, geo_idxs, vib_idxs)):
                search_disps = search_spaces[i].search_disps[j]
                search_vibs = search_spaces[i].search_vibs[j]
                geo = search_disps[geo_idx]
                vib = search_vibs[vib_idx]

                delta_struct.layers[0].xs[atom_idx-1] += geo[0] / delta_struct.cell_params[0]
                delta_struct.layers[0].ys[atom_idx-1] += geo[1] / delta_struct.cell_params[1]
                delta_struct.layers[0].zs[atom_idx-1] += geo[2] / delta_struct.cell_params[2]
                sitenum = delta_struct.layers[0].sitenums[atom_idx-1]
                delta_struct.sites[sitenum-1].vib = vib

            delta_structs.append(delta_struct)
            delta_calc = DeltaCalc(delta_struct, ref_calc, self._delta_exe, search_vibs=search_vibs)
            self.completed_deltacalcs.append((delta_calc, delta_rfactor))
            self.completed_calcs.append((delta_calc, delta_rfactor))

        return ref_rfactors, delta_structs, delta_rfactors

    def poll_active_calcs(self) -> List[Tuple[Calc, float]]:
        """ Polls all of the 'active calculations' to check if any have completed,
             updating the status of each calculation.
            Returns a list of (calc, rfactor) for each completed calculation.
        """
        completed_refcalcs = []
        completed_deltacalcs = []
        new_active_calcs = []
        for calc in self.active_calcs:
            calc_state = calc.poll()

            if calc_state is CalcState.RUNNING:  # Calculation still running
                new_active_calcs.append(calc)
                continue
            elif calc_state is CalcState.TERMINATED:    # Calculation terminated by some signal
                logging.error(
                    "Reference calculation {} was terminated!".format(calc.script_filename)
                )
                raise RuntimeError(
                    "Reference calculation {} was terminated!".format(calc.script_filename)
                )
            calc_curves = calc.produce_curves()
            calc_rfactor = min(avg_rfactors(self.exp_curves, calc_curves))
            if type(calc) is RefCalc:
                completed_refcalcs.append((calc, calc_rfactor))
            elif type(calc) is DeltaCalc:
                completed_deltacalcs.append((calc, calc_rfactor))

        self.active_calcs = new_active_calcs
        self.completed_refcalcs.extend(completed_refcalcs)
        self.completed_deltacalcs.extend(completed_deltacalcs)
        return completed_refcalcs + completed_deltacalcs

    def wait_active_calcs(self) -> List[Tuple[Calc, float]]:
        """ Wait for all 'active calculations' to completed.
            Return list of (calc, rfactor) for each completed calculation.
        """
        completed_refcalcs = []
        completed_deltacalcs = []
        for calc in self.active_calcs:
            calc.wait()
            calc_curves = calc.produce_curves()
            calc_rfactor = min(avg_rfactors(self.exp_curves, calc_curves))
            if type(calc) is RefCalc:
                completed_refcalcs.append((calc, calc_rfactor))
            elif type(calc) is DeltaCalc:
                completed_deltacalcs.append((calc, calc_rfactor))

        self.active_calcs = []
        self.completed_refcalcs.extend(completed_refcalcs)
        self.completed_deltacalcs.extend(completed_deltacalcs)
        return completed_refcalcs + completed_deltacalcs

    def _write_delta_script(self, filename: str, ref_calc: RefCalc, search_dim: DeltaSearchDim):
        atom_num, disps, vibs = search_dim
        # Determine which element the site of this atom is.
        # Unsure how this extends to handle concentration variation.
        site_num = ref_calc.struct.layers[0].sitenums[atom_num-1]
        elem_num = np.argmax(ref_calc.struct.sites[site_num].concs) + 1
        with open(filename, "w") as f:
            f.write("Delta Script\n")
            f.write("{:>7.2f}{:>7.2f}\n".format(self.beaminfo.energy_min, self.beaminfo.energy_max))
            f.write("{:>7.4f}{:>7.4f}\n".format(ref_calc.struct.cell_params[0], 0.0))
            f.write("{:>7.4f}{:>7.4f}\n".format(0.0, ref_calc.struct.cell_params[1]))
            f.write("{:>7.4f}{:>7.4f}\n".format(ref_calc.struct.cell_params[0], 0.0))
            f.write("{:>7.4f}{:>7.4f}\n".format(0.0, ref_calc.struct.cell_params[1]))
            f.write("{:>7.2f}{:>7.2f}\n".format(self.beaminfo.theta, self.beaminfo.phi))
            f.write("   1\n")
            for beamx, beamy in self.beaminfo.beams:
                f.write("{:>10.5f}{:>10.5f}\n".format(beamx, beamy))
            f.write("{:>3d}\n".format(ref_calc.struct.num_elems))
            f.write(self.phaseshifts.to_script())
            f.write("   1\n")
            f.write("-------------------------------------------------------------------\n"
                    "--- chemical nature of displaced atom                           ---\n"
                    "-------------------------------------------------------------------\n")
            f.write("{:>4d}\n".format(elem_num))
            f.write("-------------------------------------------------------------------\n"
                    "--- undisplaced position of original site                       ---\n"
                    "-------------------------------------------------------------------\n")
            f.write(" 0.0000 0.0000 0.0000\n")
            f.write("-------------------------------------------------------------------\n"
                    "--- displaced positions of atomic site in question              ---\n"
                    "-------------------------------------------------------------------\n")
            f.write("{:>4d}\n".format(len(disps)))
            for disp in disps:
                f.write("{:>7.4f}{:>7.4f}{:>7.4f}\n".format(disp[2], disp[0], disp[1]))
            f.write("-------------------------------------------------------------------\n"
                    "--- absolute vibrational amplitudes of atomic site in question  ---\n"
                    "-------------------------------------------------------------------\n")
            f.write("{:>4d}\n".format(len(vibs)))
            for vib in vibs:
                f.write("{:>7.4f}\n".format(vib))

    def _compile_delta_program(self, executable_path: str, ndisps: int, nvibs: int,
                               compiler: str = "gfortran", options: List[str] = None):
        """ Compiles the delta.f program. Should only need to be called one at the beginning
             of an optimization problem.
        """
        logging.info("Compiling perturbative TLEED program.")
        exe_dir = os.path.dirname(executable_path)

        # Write PARAM needed to compile the delta executable
        with open(os.path.join(exe_dir, "PARAM"), "w") as f:
            f.write("      PARAMETER( MLMAX = {})\n".format(self.phaseshifts.lmax))
            f.write("      PARAMETER( MNLMB = {})\n".format(_MNLMBS[self.phaseshifts.lmax]))
            f.write("      PARAMETER( MNPSI = {}, MNEL = {})\n".format(
                self.phaseshifts.num_energies, self.phaseshifts.num_elem
            ))
            f.write("      PARAMETER( MNT0 = {} )\n".format(len(self.beaminfo.beams)))
            f.write("      PARAMETER( MNATOMS = 1 )\n")
            f.write("      PARAMETER( MNDEB = {} )\n".format(nvibs))
            f.write("      PARAMETER( MNCSTEP = {} )\n".format(ndisps))

        global_source = os.path.join(self.tleed_dir, "v1.2", "src", "GLOBAL")
        delta_exe_source = os.path.join(self.tleed_dir, "v1.2", "src", "delta.f")
        delta_lib_source = os.path.join(self.tleed_dir, "v1.2", "lib", "lib.delta.f")
        tleed_lib_source = os.path.join(self.tleed_dir, "v1.2", "lib", "lib.tleed.f")
        global_dest = os.path.join(exe_dir, "GLOBAL")
        delta_exe_dest = os.path.join(exe_dir, "delta.f")
        delta_lib_dest = os.path.join(exe_dir, "lib.delta.f")
        tleed_lib_dest = os.path.join(exe_dir, "lib.tleed.f")

        # Create symlinks to source files for local compilation
        for src, dest in zip([global_source, delta_exe_source, delta_lib_source, tleed_lib_source],
                             [global_dest, delta_exe_dest, delta_lib_dest, tleed_lib_dest]):
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(src, dest)

        if options is None:
            options = ["-O3", "-malign-double", "-funroll-loops", "-std=legacy"]

        processes = list()
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "main.o", "-c", delta_exe_dest], cwd=exe_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "lib.tleed.o", "-c", tleed_lib_dest], cwd=exe_dir,
            stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL,
        ))
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "lib.delta.o", "-c", delta_lib_dest], cwd=exe_dir,
            stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL,
        ))
        for p in processes:
            p.wait()
        subprocess.run(
            [compiler] + options + ["-o", executable_path, "main.o", "lib.tleed.o", "lib.delta.o"],
            cwd=exe_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self._delta_compiled = True
        logging.info("Perturbative TLEED program compiled.")

    def _compile_muftin_func(self, compiler: str = "gfortran", options: List[str] = None):
        """ Compiles the muffin tin potential function. For now, uses the potential that is default
             in TensErLEED.
             TODO: Make this edit-able
        """
        executable_dir = os.path.dirname(self._ref_exe)
        with open(os.path.join(executable_dir, "muftin.f"), "w") as f:
            f.write(
                "      subroutine muftin(EEV,VO,VV,VPI,VPIS,VPIO)"
                "      real EEV,VO,VV,VPI,VPIS,VPIO"
                "      real workfn"
                "      workfn = 0."
                "      VV = workfn - max( (0.08-77.73/sqrt(EEV+workfn+30.7)) , -10.73)"
                "      VO = 0."
                "      VPI = 5.0"
                "      VPIS = VPI"
                "      VPIO = VPI"
                "      return"
                "      end"
            )

        subprocess.run(
            [compiler] + options + ["-o", "muftin.o", "-c", "muftin.f"],
            cwd=executable_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def _compile_ref_program(self, struct: AtomicStructure, executable_path: str, sym: int = 2,
                             compiler: str = "gfortran", options: List[str] = None):
        """ Compiles the reference calculation code, using the given AtomicStructure as the
                source of various required dimension sizes in the calculation.
        """
        logging.info("Compiling reference calculation program.")
        exe_dir = os.path.dirname(os.path.abspath(executable_path))

        with open(os.path.join(exe_dir, "PARAM"), "w") as f:
            # 1. Lattice symmetry
            # For now, don't touch these for simplicity.
            f.write("      PARAMETER (MIDEG={},MNL1=1,MNL2=1)\n".format(sym))
            f.write("      PARAMETER (MNL = MNL1*MNL2)\n")

            # 2. General calculational quantities
            # "Number of independent beam sets in beam list"
            f.write("      PARAMETER (MKNBS = 1)\n")
            f.write("      PARAMETER (MKNT = {})\n".format(len(self.beamlist)))
            f.write("      PARAMETER (MNPUN = {0}, MNT0 = {0})\n".format(len(self.beaminfo)))
            f.write("      PARAMETER (MNPSI = {}, MNEL = {})\n".format(
                self.phaseshifts.num_energies, self.phaseshifts.num_elem
            ))
            f.write("      PARAMETER (MLMAX = {})\n".format(self.phaseshifts.lmax))
            f.write("      PARAMETER (MNLMO = {}, MNLM = {})\n".format(
                _MNLMOS[self.phaseshifts.lmax-1], _MNLMS[self.phaseshifts.lmax-1]
            ))

            # 3. Parameters for (3D) geometry within (2D) unit mesh
            f.write("      PARAMETER (MNSITE  = {})\n".format(len(struct.sites)))
            f.write("      PARAMETER (MNLTYPE = {})\n".format(len(struct.layers)))
            # Treat all layers are composite for now
            f.write("      PARAMETER (MNBRAV  = 0)\n")
            f.write("      PARAMETER (MNSUB   = {})\n".format(
                max(len(lay) for lay in struct.layers)
            ))
            f.write("      PARAMETER (MNSTACK = {})\n".format(
                sum(1 for lay in struct.layers if lay.lay_type == LayerType.SURF)
            ))

            # 4. Some derived quantities. Do not modify -- copied as-is from TensErLEED scripts.
            f.write("      PARAMETER (MLMAX1=MLMAX+1)\n")
            f.write("      PARAMETER (MLMMAX = MLMAX1*MLMAX1)\n")
            f.write("      PARAMETER (MNBRAV2 = 1)\n")
            f.write("      PARAMETER (MNCOMP= MNLTYPE-MNBRAV)\n")
            f.write("      PARAMETER (MLMT  = MNSUB*MLMMAX)\n")
            f.write("      PARAMETER (MNSUB2= MNSUB * (MNSUB-1)/2)\n")
            f.write("      PARAMETER (MLMG  = MNSUB2*MLMMAX*2)\n")
            f.write("      PARAMETER (MLMN  = MNSUB * MLMMAX)\n")
            f.write("      PARAMETER (MLM2N = 2*MLMN)\n")
            f.write("      PARAMETER (MLMNI = MNSUB*MLMMAX)\n")

        global_source = os.path.join(self.tleed_dir, "v1.2", "src", "GLOBAL")
        ref_lib_source = os.path.join(self.tleed_dir, "v1.2", "lib", "lib.tleed.f")
        ref_exe_source = os.path.join(self.tleed_dir, "v1.2", "src", "ref-calc.f")
        global_dest = os.path.join(exe_dir, "GLOBAL")
        ref_lib_dest = os.path.join(exe_dir, "lib.tleed.f")
        ref_exe_dest = os.path.join(exe_dir, "ref-calc.f")

        # Create symlinks to source files so that local compilation includes PARAM
        for src, dest in zip([global_source, ref_lib_source, ref_exe_source],
                             [global_dest, ref_lib_dest, ref_exe_dest]):
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(src, dest)

        if options is None:
            options = ["-O3", "-malign-double", "-funroll-loops", "-std=legacy"]

        # Compile the muffin tin potential subroutine
        self._compile_muftin_func(compiler=compiler, options=options)

        # Compile the consituent libraries / programs
        processes = list()
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "main.o", "-c", ref_exe_dest], cwd=exe_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "lib.tleed.o", "-c", ref_lib_dest], cwd=exe_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        for p in processes:
            p.wait()
        # Link together
        subprocess.run(
            [compiler] + options + ["-o", executable_path, "muftin.o", "lib.tleed.o", "main.o"],
            cwd=exe_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self._ref_compiled = True
        logging.info("Reference calculation program compiled.")

    def produce_delta_amps(self, delta_space: DeltaSearchSpace,
                           delta_exe: Optional[str] = None) -> MultiDeltaAmps:
        """ Performs all of the computations needed to produce the delta amplitudes for each point
             in the given search space.
            If re-compilation of the executable is unnecessary, the path to an existing executable
             can be passed.
        """
        if not delta_space.ref_calc.produce_tensors:
            raise ValueError(
                "Tensors are required to be output by the reference calc to compute deltas!"
            )

        ref_calc = delta_space.ref_calc
        workdir = ref_calc.workdir

        processes = []
        for i, search_dim in enumerate(delta_space.search_dims):
            # Produce a temporary work directory
            subworkdir = os.path.join(workdir, "delta_tmp" + str(i + 1))
            try:
                os.mkdir(subworkdir)
            except FileExistsError:
                pass

            dst_exe = os.path.join(subworkdir, "delta.x")
            if delta_exe is None:
                # In that directory, compile a version of delta.f with the parameters
                # of this search dimension
                self._compile_delta_program(dst_exe, len(search_dim[1]), len(search_dim[2]))
            else:
                if os.path.exists(dst_exe):
                    os.remove(dst_exe)
                # Create a symlink in this directory to the executable
                os.symlink(delta_exe, dst_exe)

            # Also in that directory, create the input script to delta.f
            input_scriptname = os.path.join(subworkdir, "delta.in")
            self._write_delta_script(input_scriptname, ref_calc, search_dim)

            # Link the relevant tensor into this directory as AMP
            amp_path = os.path.join(subworkdir, "AMP")
            if os.path.exists(amp_path):
                os.remove(amp_path)
            os.symlink(ref_calc.tensorfiles[i], amp_path)

            # Run the delta executable
            processes.append(subprocess.Popen(
                [dst_exe],
                stdin=open(input_scriptname, "r"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=subworkdir,
                text=True
            ))

        # Wait for all of the processes to complete
        for p in processes:
            retcode = p.wait()
            if retcode != 0:
                raise RuntimeError("Encountered an error while executing delta calculation")

        # Parse all of the produced files into SiteDeltaAmps objects
        delta_amps = []
        for i in range(len(delta_space.search_dims)):
            subworkdir = os.path.join(workdir, "delta_tmp" + str(i + 1))
            delta_amps.append(parse_deltas(os.path.join(subworkdir, "DELWV")))
            # Remove directory once we are done with it
            # shutil.rmtree(subworkdir)

        return MultiDeltaAmps(delta_amps)


