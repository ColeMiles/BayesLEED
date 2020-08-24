""" Python code for interfacing with the TensErLEED code.
"""
from __future__ import annotations # Python 3.7+ required

import os
import shutil
import subprocess
import re
import logging
from typing import List, Tuple, Collection, Optional

import numpy as np

from .structure import AtomicStructure
from .searchspace import DeltaSearchDim, DeltaSearchSpace
from .curves import IVCurve, IVCurveSet, parse_ivcurves


_MNLMBS = [19, 126, 498, 1463, 3549, 7534, 14484, 25821, 43351, 69322, 106470, 158067, 227969, 320664, 441320]


class BeamInfo:
    """ A simple struct-like class for holding information about a set of beams """
    def __init__(self, theta: float, phi: float, beams: List[Tuple[int, int]], energy_min: float, energy_max: float):
        self.theta = theta
        self.phi = phi
        self.beams = beams
        self.energy_min = energy_min
        self.energy_max = energy_max
        if not self.energy_min < self.energy_max:
            raise ValueError("Cannot have energy_min >= energy_max")


class Phaseshifts:
    """ Class representing a set of phaseshifts for a LEED calculation """
    def __init__(self, filename: str, energies: np.ndarray, phases: np.ndarray):

        self.filename = os.path.abspath(filename)
        self.energies = energies        # Energies at which each phaseshift is calculated (Hartrees)
        self.phases = phases            # [NENERGIES, NELEM, NANG_MOM] array of phaseshifts (Radians)

        # Do some basic validation
        if len(self.energies) != len(self.phases):
            raise ValueError("Number of energies not commensurate with number of phaseshifts")
        if len(self.phases.shape) != 3:
            raise ValueError("Phases should be a 3-dimensional array of shape [NENERGIES, NELEM, NANG_MOM]")

        self.num_energies = self.phases.shape[0]  # Number of energies tabulated
        self.num_elem = self.phases.shape[1]      # Number of elements
        self.lmax = self.phases.shape[2]          # Maximum angular momentum quantum number

    def to_script(self) -> str:
        """ Returns a string to be inserted into scripts which expect phaseshifts as text """
        with open(self.filename, "r") as f:
            return f.read()


def parse_phaseshifts(filename: str, l_max: int) -> Phaseshifts:
    """ Parse a set of phaseshifts from a file. The maximum angular momentum number contained in the file
         must be specified due to the ambiguous format of phaseshift files.
    """
    energies = []
    # Will become a triply-nested list of dimensions [NENERGIES, NELEM, NANG_MOM], converted to a numpy array at the end
    phases = []

    # All of this parsing is extra weird since Fortran77-formatted files will only place a maximum # of characters
    #   on each line, so information that really belongs together is strewn across multiple lines, which has to be
    #   checked for. Specifically here, only 10 phaseshift will be placed on a line before wrapping, and the Fortran
    #   code does an odd thing where it adds an extraneous line only if the number of phaseshifts <= 10
    with open(filename, "r") as f:
        line = f.readline()
        line_num = 1
        energies.append(float(line))
        phases.append([])

        # Use the first couple of lines to determine the number of elements present
        line = f.readline()
        line_num += 1

        num_elem = 0
        while len(line) != 8:   # Once len(line) == 8, we've hit a new energy rather than more phaseshifts
            if len(line) != min(70, 7 * l_max) + 1:
                raise ValueError("Provided l_max does not agree with phaseshift file: Line {}".format(line_num))

            elem_phases = [float(line[7*i:7*(i+1)]) for i in range(min(10, l_max))]

            f.readline()      # This line is extraneous if l_max <= 10, but will contain more phaseshifts otherwise
            if l_max > 10:
                for i in range(l_max-10):
                    elem_phases.append(float(line[7*i:7*(i+1)]))

            phases[0].append(elem_phases)
            num_elem += 1
            line = f.readline()

        # Once we know the number of elements, we can loop through the rest of the file simply
        n_energy = 1
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


class RefCalc:
    """ Class representing a single reference calculation, responsible for orchestrating the
        necessary scripts to run the calculation, as well as keeping track of Tensors needed
        for perturbative calculations
    """
    def __init__(self, struct: AtomicStructure, leed_exe: str, rf_exe: str, template: str, workdir: str, produce_tensors=False):
        self.struct = struct
        self.leed_exe = os.path.abspath(leed_exe)
        self.rf_exe = os.path.abspath(rf_exe)
        self.template = template.splitlines(keepends=True)
        self.workdir = os.path.abspath(workdir)
        self.produce_tensors = produce_tensors
        self.tensorfiles = []

        # TODO: Maybe an Enum for a state rather than this?
        self.completed = False
        self.in_progress = False

        self.script_filename = os.path.join(self.workdir, "FIN")
        self.result_filename = os.path.join(self.workdir, "fd.out")
        self._process = None

        if produce_tensors:
            self.tensorfiles = [
                os.path.join(workdir, "LAY1{}".format(i+1)) for i in range(len(struct.layers[0]))
            ]

    def _write_script(self, filename):
        with open(filename, "w") as ofile:
            # File title and energy range
            ofile.writelines(self.template[:2])

            ofile.write("{:>7.4f} 0.0000          ARA1 *\n".format(self.struct.cell_params[0]))
            ofile.write(" 0.0000{:>7.4f}          ARA2 *\n".format(self.struct.cell_params[1]))

            # (Unused) registry shift lines
            ofile.writelines(self.template[4:8])

            ofile.write("{:>7.4f} 0.0000          ARB1 *\n".format(self.struct.cell_params[0]))
            ofile.write(" 0.0000{:>7.4f}          ARB2 *\n".format(self.struct.cell_params[1]))

            # Find line before where new coordinates need to be inserted, as well as the
            #  line which marks the following section
            indbefore, indafter = -1, -1
            for i, line in enumerate(self.template):
                if line.find("define chem. and vib. properties") != -1:
                    indbefore = i - 1
                elif line.find("Tensor output is") != -1:
                    indafter = i+1
            # Check that both lines were found
            if indbefore == -1 or indafter == -1:
                raise ValueError("LEED input file does not contain section marker lines")
            ofile.writelines(self.template[10:indbefore])

            # Site description section
            output = (
                "-------------------------------------------------------------------\n"
                "--- define chem. and vib. properties for different atomic sites ---\n"
                "-------------------------------------------------------------------\n"
            )
            output += "{:>3d}".format(len(self.struct.sites))
            output += 23 * " " + "NSITE: number of different site types\n"
            for i, site in enumerate(self.struct.sites):
                output += "-   site type {}  {}---\n".format(i + 1, site.name)
                output += site.to_script()

            # Layer description section
            output += (
                "-------------------------------------------------------------------\n"
                "--- define different layer types                            *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            output += "{:>3d}".format(len(self.struct.layers))
            output += 23 * " " + "NLTYPE: number of different layer types\n"
            for i, layer in enumerate(self.struct.layers):
                output += "-   layer type {}  {}---\n".format(i + 1, layer.name)
                output += "{:>3d}".format(i + 1)
                output += 23 * " " + "LAY = {}\n".format(i + 1)
                output += layer.to_script(self.struct.cell_params)

            # Bulk stacking section
            output += (
                "-------------------------------------------------------------------\n"
                "--- define bulk stacking sequence                           *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            # Find bulk interlayer vector from bottom atom of bulk layer
            bulk_maxz = max(self.struct.layers[1].zs)
            num_cells = np.ceil(bulk_maxz)
            bulk_interlayer_dist = (num_cells - bulk_maxz) * self.struct.cell_params[2]

            output += "  0" + 23 * " " + "TSLAB = 0: compute bulk using subras\n"
            output += "{:>7.4f} 0.0000 0.0000".format(bulk_interlayer_dist)
            output += "     ASA interlayer vector between different bulk units *\n"
            output += "  2" + 23 * " " + "top layer of bulk unit: type 2\n"
            output += "  2" + 23 * " " + "bottom layer of bulk unit: type 2\n"
            output += "{:>7.4f} 0.0000 0.0000".format(bulk_interlayer_dist)
            output += "     ASBULK between the two bulk unit layers (may differ from ASA)\n"

            # Surface layer stacking sequence
            output += (
                "-------------------------------------------------------------------\n"
                "--- define layer stacking sequence and Tensor LEED output   *   ---\n"
                "-------------------------------------------------------------------\n"
            )
            # Find surface interlayer vector from bottom atom to bulk
            layer_maxz = max(self.struct.layers[0].zs)
            num_cells = np.ceil(layer_maxz)
            surf_interlayer_dist = (num_cells - layer_maxz) * self.struct.cell_params[2]
            output += "  1\n"
            output += "  1{:>7.4f} 0.0000 0.0000".format(surf_interlayer_dist)
            output += "  surface layer is of type 1: interlayer vector connecting it to bulk\n"
            if self.produce_tensors:
                output += "  1" + 23 * " " + "Tensor output is required for this layer\n"
            else:
                output += "  0" + 23 * " " + "Tensor output is NOT required for this layer\n"

            ofile.write(output)
            ofile.writelines(self.template[indafter:])

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
        self.in_progress = True

    def wait(self):
        """ Waits for completion. Call this even if you don't rely on this function for blocking.
        """
        self._process.wait()
        # TODO: Can I make these update without a call to .wait()?
        self.in_progress = False
        self.completed = True
        self.tensorfiles = [os.path.join(self.workdir, "LAY1{}".format(i)) for i in range(len(self.struct.layers[0]))]

    # TODO: Convert to using my own r-factor calculation over the Fortran program
    def rfactor(self):
        if not self.completed:
            raise ValueError("Called .rfactor() on a RefCalc which is not complete!")
        if not os.path.exists(self.rf_exe):
            raise FileNotFoundError("R-factor executable rf.x not found!")
        if not os.path.exists(self.result_filename):
            raise FileNotFoundError("Results from reference calculation, fd.out, not found!")

        result = subprocess.run(
            [self.rf_exe],
            cwd=os.path.dirname(self.rf_exe),
            stdin=open(self.result_filename, "r"),
            capture_output=True,
            text=True
        )
        return extract_rfactor(result.stdout)

    def produce_curves(self) -> IVCurveSet:
        """ Produce the set of IV curves resulting from the reference calculation.
            If the calculation has not been run yet, or is still running, wait
             for completion.
        """
        if not self.in_progress and not self.completed:
            self.run()
        if self.in_progress:
            self.wait()
        return parse_ivcurves(self.result_filename, format='RCOUT')


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
    """ Class holding all of the delta amplitudes for a set of sites, and can compute modified IV curves.
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


class LEEDManager:
    def __init__(self, basedir: str, leed_executable: str, rfactor_executable: str,
                 exp_datafile: str, phaseshifts: Phaseshifts, beaminfo: BeamInfo, templatefile: str,
                 tleed_dir: str):
        """ Create a LEEDManager to keep track of TensErLEED components. Only one of these should exist per problem.
                basedir: The base directory to do computation in
                leed_executable: Path to the LEED executable
                rfactor_executable: Path to the rfactor executable
                exp_datafile: Path to the experimental datafile
                phss_datafile: Path to the phaseshifts file
                templatefile: The base template for the LEED input file (FIN)
                tleed_dir: Path to the base directory of TLEED
        """
        for path in [basedir, leed_executable, rfactor_executable, templatefile]:
            if not os.path.exists(path):
                raise ValueError("File not found: {}".format(path))
        self.basedir = os.path.abspath(basedir)
        self.leed_exe = os.path.abspath(leed_executable)
        self.rfactor_exe = os.path.abspath(rfactor_executable)
        self.exp_datafile = os.path.abspath(exp_datafile)
        self.tleed_dir = os.path.abspath(tleed_dir)
        self.phaseshifts = phaseshifts
        self.beaminfo = beaminfo

        # Copy the exp datafile to the working directory if not already there
        copy_exp_datafile = os.path.join(
            self.basedir,
            os.path.split(self.exp_datafile)[1]
        )
        try:
            shutil.copyfile(self.exp_datafile, copy_exp_datafile)
        except shutil.SameFileError:
            pass
        with open(templatefile, "r") as f:
            self.input_template = f.read()

        self.calc_number = 0
        self.ref_calcs = []

    def _create_ref_calc(self, structure: AtomicStructure, produce_tensors=False):
        newdir = os.path.join(self.basedir, "ref-calc" + str(self.calc_number))
        os.makedirs(newdir, exist_ok=True)
        ref_calc = RefCalc(structure, self.leed_exe, self.rfactor_exe, self.input_template, newdir,
                           produce_tensors=produce_tensors)
        self.ref_calcs.append(ref_calc)
        self.calc_number += 1
        return ref_calc

    def ref_calc(self, structure: AtomicStructure, produce_tensors=False):
        """ Do the full process of performing a reference calculation.
            WARNING: Do not call this function in parallel, as there is a race
                condition on self.calc_number. Instead, use batch_ref_calcs.
        """
        refcalc = self._create_ref_calc(structure, produce_tensors=produce_tensors)
        refcalc.run()
        refcalc.wait()
        self.ref_calcs.append(refcalc)
        return refcalc.rfactor()

    def batch_ref_calcs(self, structures: Collection[AtomicStructure], produce_tensors=False):
        """ Run multiple reference calculations in parallel, one for each
             row of displacements.
        """
        num_structs = len(structures)

        # Create RefCalc objects for each calculation
        logging.info("Starting {} reference calculations...".format(num_structs))
        refcalcs = [
            self._create_ref_calc(struct, produce_tensors=produce_tensors)
            for struct in structures
        ]

        # Start up all of the calculation processes
        for r in refcalcs:
            r.run()

        # Wait for all of them to complete, calculate r-factors for each
        # The r-factor calculations are fast enough that we may as well run
        #   them serially
        rfactors = []
        for refcalc in refcalcs:
            refcalc.wait()
            self.ref_calcs.append(refcalc)
            rfactors.append(refcalc.rfactor())
        logging.info("Reference calculations completed.")
        return np.array(rfactors)

    def _write_delta_script(self, filename: str, ref_calc: RefCalc, search_dim: DeltaSearchDim):
        atom_num, disps, vibs = search_dim
        # Determine which element the site of this atom is. Unsure how this extends to handle concentration variation.
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

    def _compile_delta_program(self, executable_path: str, search_dim: DeltaSearchDim,
                               compiler: str = "gfortran", options: List[str] = None):
        atom_num, disps, vibs = search_dim
        exe_dir = os.path.dirname(executable_path)

        # Write PARAM needed to compile the delta executable
        with open(os.path.join(exe_dir, "PARAM"), "w") as f:
            f.write("      PARAMETER( MLMAX = {})\n".format(self.phaseshifts.lmax))
            f.write("      PARAMETER( MNLMB = {})\n".format(_MNLMBS[self.phaseshifts.lmax]))
            f.write("      PARAMETER( MNPSI = {}, MNEL = {})\n".format(
                self.phaseshifts.num_energies, self.phaseshifts.num_elem)
            )
            f.write("      PARAMETER( MNT0 = {} )\n".format(len(self.beaminfo.beams)))
            f.write("      PARAMETER( MNATOMS = 1 )\n")
            f.write("      PARAMETER( MNDEB = {} )\n".format(len(vibs)))
            f.write("      PARAMETER( MNCSTEP = {} )\n".format(len(disps)))

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
            cwd=exe_dir
        )

    def produce_delta_amps(self, delta_space: DeltaSearchSpace,
                           delta_exe: Optional[str] = None) -> MultiDeltaAmps:
        """ Performs all of the computations needed to produce the delta amplitudes for each point
             in the given search space.
            If re-compliation of the executable is unnecessary, the path to an existing executable
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


def extract_rfactor(output):
    p = re.compile(r"AVERAGE R-FACTOR =  (\d\.\d+)")
    m = re.search(p, output)
    if m is not None:
        return float(m.group(1))
    else:
        import ipdb
        ipdb.set_trace()
        raise ValueError("No average R-factor line found in input")
