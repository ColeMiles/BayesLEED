""" Python code for interfacing with the TensErLEED code.
"""
from __future__ import annotations # Python 3.7+ required

import os
import shutil
import subprocess
import logging
import enum
from typing import List, Tuple, Collection, Optional, Union

import numpy as np

from .structure import AtomicStructure
from .searchspace import DeltaSearchDim, DeltaSearchSpace
from .curves import IVCurve, IVCurveSet, parse_ivcurves, avg_rfactors


_MNLMBS = [19, 126, 498, 1463, 3549, 7534, 14484, 25821, 43351,
           69322, 106470, 158067, 227969, 320664, 441320]


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

    def __iter__(self):
        return iter(self.beams)


class BeamList:
    """ A class representing the beamlist generated as a pre-processing step in
         the LEED dynamical structure calculations.
        Note a distinction: BeamInfo is used for the set of beams measured by experiment
         that you want to compare simulated results against. BeamList is the list of all
         beams that TensErLEED has determined it needs to track to perform this
         calculation.
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
    beams, energies = [], []
    with open(filename, "r") as f:
        # Number of beams, don't need because we're not Fortran
        f.readline()
        for line in f:
            line_s = line.split()
            beams.append((float(line_s[0]), float(line_s[1])))
            energies.append(float(line_s[6]))

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
        self.lmax = self.phases.shape[2]          # Maximum angular momentum quantum number

    def to_script(self) -> str:
        """ Returns a string to be inserted into scripts which expect phaseshifts as text """
        with open(self.filename, "r") as f:
            return f.read()


def parse_phaseshifts(filename: str, l_max: int) -> Phaseshifts:
    """ Parse a set of phaseshifts from a file. The maximum angular momentum number contained
         in the file must be specified due to the ambiguous format of phaseshift files.
    """
    energies = []
    # Will become a triply-nested list of dimensions [NENERGIES, NELEM, NANG_MOM],
    #  converted to a numpy array at the end
    phases = []

    # All of this parsing is extra weird since Fortran77-formatted files will only place a
    #  maximum # of characters on each line, so information that really belongs together is
    #  strewn across multiple lines, which has to be checked for. Specifically here, only 10
    #  phaseshifts will be placed on a line before wrapping, and the Fortran code does an odd thing
    #  where it adds an extraneous line only if the number of phaseshifts <= 10
    with open(filename, "r") as f:
        line = f.readline()
        line_num = 1
        energies.append(float(line))
        phases.append([])

        # Use the first couple of lines to determine the number of elements present
        line = f.readline()
        line_num += 1

        num_elem = 0
        # Once len(line) == 8, we've hit a new energy rather than more phaseshifts
        while len(line) != 8:
            if len(line) != min(70, 7 * l_max) + 1:
                raise ValueError(
                    "Provided l_max does not agree with phaseshift file: Line {}".format(line_num)
                )

            elem_phases = [float(line[7*i:7*(i+1)]) for i in range(min(10, l_max))]

            # This line is extraneous if l_max <= 9, but will contain more phaseshifts otherwise
            f.readline()
            if l_max > 10:
                for i in range(l_max-10):
                    elem_phases.append(float(line[7*i:7*(i+1)]))

            phases[0].append(elem_phases)
            num_elem += 1
            line = f.readline()

        # Once we know the number of elements, we can loop through the rest of the file simply
        while line != "":
            energies.append(float(line))
            elem_phases = [[] for _ in range(num_elem)]
            for n in range(num_elem):
                line = f.readline()
                elem_phases[n] = [float(line[7*i:7*(i+1)]) for i in range(min(10, l_max))]

                f.readline()
                if l_max > 10:
                    for i in range(l_max-10):
                        elem_phases[n].append(float(line[7*i:7*(i+1)]))
            phases.append(elem_phases)
            line = f.readline()

    return Phaseshifts(filename, np.array(energies), np.array(phases))


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
                os.path.join(self.workdir, "LAY1{}".format(i + 1))
                for i in range(len(self.struct.layers[0]))
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
                ofile.write("{:>3d}".format(idx))
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
                ofile.write("-   layer type {}  {}---\n".format(i + 1, layer.name))
                ofile.write("{:>3d}".format(i + 1))
                ofile.write(23 * " " + "LAY = {}\n".format(i + 1))
                ofile.write(layer.to_script(self.struct.cell_params))

            # Bulk stacking section
            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- define bulk stacking sequence                           *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            # Find bulk interlayer vector from bottom atom of bulk layer
            bulk_maxz = max(self.struct.layers[1].zs)
            num_cells = np.ceil(bulk_maxz)
            bulk_interlayer_dist = (num_cells - bulk_maxz) * self.struct.cell_params[2]

            ofile.write("  0" + 23 * " " + "TSLAB = 0: compute bulk using subras\n")
            ofile.write("{:>7.4f} 0.0000 0.0000".format(bulk_interlayer_dist))
            ofile.write("     ASA interlayer vector between different bulk units *\n")
            ofile.write("  2" + 23 * " " + "top layer of bulk unit: type 2\n")
            ofile.write("  2" + 23 * " " + "bottom layer of bulk unit: type 2\n")
            ofile.write("{:>7.4f} 0.0000 0.0000".format(bulk_interlayer_dist))
            ofile.write("     ASBULK between the two bulk unit layers (may differ from ASA)\n")

            # Surface layer stacking sequence
            ofile.write(
                "-------------------------------------------------------------------\n"
                "--- define layer stacking sequence and Tensor LEED output   *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            # Find surface interlayer vector from bottom atom to bulk
            layer_maxz = max(self.struct.layers[0].zs)
            num_cells = np.ceil(layer_maxz)
            surf_interlayer_dist = (num_cells - layer_maxz) * self.struct.cell_params[2]
            ofile.write("  1\n")
            ofile.write("  1{:>7.4f} 0.0000 0.0000".format(surf_interlayer_dist))
            ofile.write("  surface layer is of type 1: interlayer vector connecting it to bulk\n")
            if self.produce_tensors:
                ofile.write("  1" + 23 * " " + "Tensor output is required for this layer\n")
            else:
                ofile.write("  0" + 23 * " " + "Tensor output is NOT required for this layer\n")

            for i in range(len(self.struct.layers[0])):
                ofile.write("LAY1" + str(i) + 22 * " " + "Tensorfile, sublayer " + str(i+1) + "\n")

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
        for ref_atom, delta_atom in zip(ref_calc.struct.layers[0], struct.layers[0]):
            self._disps.append(np.array([
                a * (delta_atom.x - ref_atom.x),
                b * (delta_atom.y - ref_atom.y),
                c * (delta_atom.z - ref_atom.z)
            ]))
            self._vibs.append(struct.sites[delta_atom.sitenum].vib)

    def _write_scripts(self) -> List[str]:
        """ Writes one script for each atom which we need to perturb.
            Returns a list of the script paths.
        """
        script_paths = []
        for iatom, atom in enumerate(self.struct.layers[0]):
            subworkdir = os.path.join(self.ref_calc.workdir, "delta_tmp" + str(iatom + 1))
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
                        "--- unused relic of the old code                                ---\n"
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

    def run(self):
        self.state = CalcState.RUNNING
        script_paths = self._write_scripts()
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
    """ Class representing a set of delta amplitudes for perturbations of a single site produced by delta.f """
    def __init__(self):
        # All of this should be manually initialized, in parse_deltas
        self.theta, self.phi = 0.0, 0.0
        self.substrate_recip, self.overlayer_recip = np.zeros(2), np.zeros(2)
        self.nbeams = 0
        self.natoms = 0  # Unused by TensErLEED, will always be read in as 1
        self.nshifts = 0
        self.nvibs = 0
        self.beams = np.empty((0, 2))
        self.shifts = np.empty((0, 3))   # NOTE: These are z x y in file, but x y z here
        self.thermal_amps = np.empty(1)
        self.crystal_energies = np.empty(0)
        self.substrate_energies = np.empty(0, dtype=np.complex64)
        self.overlayer_energies = np.empty(0, dtype=np.complex64)
        self.real_energies_ev = np.empty(0)
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
            np.abs(new_amplitudes[ibeam]),
            self.beam_labels[ibeam]
        ) for ibeam in range(self.nbeams)]

        return IVCurveSet(curves)


def parse_deltas(filename: str) -> SiteDeltaAmps:
    """ Parses a DELWV file output by delta.f to create a SiteDeltaAmps object.
        Note all energies read in are in Hartrees! (Conversion: 1 Hartree = 27.21 eV).
        To get "energy" to compare to experiment:
            27.21 * (crystal_energy - np.real(substrate_energy))
    """
    delta_amp = SiteDeltaAmps()
    with open(filename, "r") as f:
        line = f.readline()
        delta_amp.theta = float(line[:13])
        delta_amp.phi = float(line[13:26])
        delta_amp.substrate_recip[0] = float(line[26:39])
        delta_amp.substrate_recip[1] = float(line[39:52])
        delta_amp.overlayer_recip[0] = float(line[52:65])
        delta_amp.overlayer_recip[1] = float(line[65:78])

        line = f.readline()
        delta_amp.nbeams = int(line[:3])
        delta_amp.natoms = int(line[3:6])
        numdeltas = int(line[6:9])
        delta_amp.beams = np.empty((delta_amp.nbeams, 2))
        deltas = np.empty((numdeltas, 3))

        line = f.readline()
        # TODO: Check that this is always on one line, even for more beams
        for i in range(delta_amp.nbeams):
            delta_amp.beams[i, 0] = float(line[20*i:10+20*i])
            delta_amp.beams[i, 1] = float(line[10+20*i:20+20*i])

        # Line of 0.0's from an unused feature in TensErLEED
        line = f.readline()

        # Read in the list of considered shifts
        # TODO: This is nasty.
        n = 0
        comp = 0
        while n < numdeltas:
            line = f.readline().rstrip('\n')
            for i in range(len(line) // 7):
                deltas[n, (comp + 2) % 3] = float(line[7*i:7*(i+1)])
                if comp == 2:
                    n += 1
                comp = (comp + 1) % 3

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
            line = f.readline().rstrip('\n')
            for i in range(len(line) // 7):
                delta_amp.thermal_amps[n] = line[7*i:7*(i+1)]
                n += 1

        # Read in the crystal potential energies
        crystal_energies = []
        substrate_energies = []
        overlayer_energies = []
        real_energies_ev = []
        all_ref_amplitudes = []
        all_delta_amplitudes = []
        line = f.readline()
        # Each iteration of this is a single energy
        nit = 0
        while line != "":
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
                line = f.readline()
                for i in range(len(line) // 26):
                    ref_amplitudes[n] = float(line[26*i:26*i+13])
                    ref_amplitudes[n] += float(line[26*i+13:26*i+26]) * 1j
                    n += 1
            all_ref_amplitudes.append(ref_amplitudes)

            # Read in the delta amplitudes for each search delta
            delta_amplitudes = np.empty((delta_amp.nbeams, delta_amp.nvibs, delta_amp.nshifts), np.complex64)
            n = 0
            while n < delta_amp.nbeams * delta_amp.nvibs * delta_amp.nshifts:
                line = f.readline()
                for i in range(len(line) // 26):
                    delta_idx, beam_idx = divmod(n, delta_amp.nbeams)
                    vib_idx, disp_idx = divmod(delta_idx, delta_amp.nshifts)
                    delta_amplitudes[beam_idx, vib_idx, disp_idx] = float(line[26*i:26*i+13])
                    delta_amplitudes[beam_idx, vib_idx, disp_idx] += float(line[26*i+13:26*i+26]) * 1j
                    n += 1
            all_delta_amplitudes.append(delta_amplitudes)
            nit += 1
            line = f.readline()

        delta_amp.crystal_energies = np.array(crystal_energies)
        delta_amp.substrate_energies = np.array(substrate_energies)
        delta_amp.overlayer_energies = np.array(overlayer_energies)
        delta_amp.real_energies_ev = np.array(real_energies_ev)
        delta_amp.ref_amplitudes = np.stack(all_ref_amplitudes, axis=-1)
        delta_amp.delta_amplitudes = np.stack(all_delta_amplitudes, axis=-1)

    return delta_amp


Calc = Union[RefCalc, DeltaCalc]


class LEEDManager:
    def __init__(self, workdir: str, tleed_dir: str, leed_executable: str, exp_curves: IVCurveSet,
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
        for path in [workdir, leed_executable, tleed_dir]:
            if not os.path.exists(path):
                raise ValueError("File not found: {}".format(path))
        self.workdir = os.path.abspath(workdir)
        self.leed_exe = os.path.abspath(leed_executable)
        self._delta_exe = os.path.join(os.path.dirname(self.leed_exe), "delta.x")
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

        logging.info("Compiling TLEED delta program...")
        self._compile_delta_program(self._delta_exe, 1, 1)

    def _spawn_ref_calc(self, structure: AtomicStructure, produce_tensors=False):
        """ Spawns a subprocess running a reference calculation for the given AtomicStructure.
            Adds this subprocess to the manager's list of active calculations.
        """
        newdir = os.path.join(self.workdir, "ref-calc" + str(self.calc_number))
        os.makedirs(newdir, exist_ok=True)
        ref_calc = RefCalc(structure, self.phaseshifts, self.beaminfo, self.beamlist,
                           self.leed_exe, newdir, produce_tensors=produce_tensors)
        self.calc_number += 1
        self.active_calcs.append(ref_calc)
        return ref_calc

    def _spawn_delta_calc(self, structure: AtomicStructure, ref_calc: RefCalc):
        """ Spawns a subprocess running a reference calculation for the given AtomicStructure.
            Adds this subprocess to the manager's list of active calculations.
        """
        delta_calc = DeltaCalc(structure, ref_calc, self._delta_exe)
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

        # Start up all of the calculation processes
        for r in refcalcs:
            r.run()

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

        # Start up all of the calculation processes
        for r in deltacalcs:
            r.run()

        return deltacalcs

    def poll_active_calcs(self) -> List[Tuple[Calc, float]]:
        """ Polls all of the 'active calculations' to check if any have completed,
             updating the status of each calculation.
            Returns a list of (calc, rfactor) for each completed calculation.
        """
        completed_refcalcs = []
        completed_deltacalcs = []
        new_active_calcs = []
        for calc in self.active_calcs:
            completion = calc.poll()
            if completion is None:  # Calculation still running
                new_active_calcs.append(calc)
                continue
            elif completion < 0:    # Calculation terminated by some signal
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
                    "--- unused relic of the old code                                ---\n"
                    "-------------------------------------------------------------------\n")
            f.write(" 0.0000 0.0000 0.0000\n")
            f.write("-------------------------------------------------------------------\n"
                    "--- displaced positions of atomic site in question              ---\n"
                    "-------------------------------------------------------------------\n")
            f.write("{:>4d}\n".format(len(disps)))
            for disp in disps:
                f.write("{:>7.4f}{:>7.4f}{:>7.4f}\n".format(disp[2], disp[0], disp[1]))
            f.write("-------------------------------------------------------------------\n"
                    "--- vibrational displacements of atomic site in question        ---\n"
                    "-------------------------------------------------------------------\n")
            f.write("{:>4d}\n".format(len(vibs)))
            for vib in vibs:
                f.write("{:>7.4f}\n".format(vib))

    def _compile_delta_program(self, executable_path: str, ndisps: int, nvibs: int,
                               compiler: str = "gfortran", options: List[str] = None):
        """ Compiles the delta.f program. Should only need to be called one at the beginning
             of an optimization problem.
        """
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

        # Create symlinks to source files so that local compilation includes PARAM
        for src, dest in zip([global_source, delta_exe_source, delta_lib_source, tleed_lib_source],
                             [global_dest, delta_exe_dest, delta_lib_dest, tleed_lib_dest]):
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(src, dest)

        if options is None:
            options = ["-O3", "-malign-double", "-funroll-loops", "-std=legacy"]

        processes = list()
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "main.o", "-c", delta_exe_dest], cwd=exe_dir
        ))
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "lib.tleed.o", "-c", tleed_lib_dest], cwd=exe_dir
        ))
        processes.append(subprocess.Popen(
            [compiler] + options + ["-o", "lib.delta.o", "-c", delta_lib_dest], cwd=exe_dir
        ))
        for p in processes:
            p.wait()
        subprocess.run(
            [compiler] + options + ["-o", executable_path, "main.o", "lib.tleed.o", "lib.delta.o"],
            cwd=exe_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

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
                self._compile_delta_program(dst_exe, search_dim)
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
            p.wait()

        # Parse all of the produced files into SiteDeltaAmps objects
        delta_amps = []
        for i in range(len(delta_space.search_dims)):
            subworkdir = os.path.join(workdir, "delta_tmp" + str(i + 1))
            delta_amps.append(parse_deltas(os.path.join(subworkdir, "DELWV")))
            # Remove directory once we are done with it
            shutil.rmtree(subworkdir)

        return MultiDeltaAmps(delta_amps)


