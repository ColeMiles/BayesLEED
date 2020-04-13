""" Python code for interfacing with the TensErLEED code.
"""
import os
import shutil
import subprocess
import re
import logging
import enum
from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np


# TODO: Make this and all the search machinery work for searching over concentrations /
#       vibrational parameters for more than two element types
class Site:
    def __init__(self, concs: List[float], vib: float,
                 elems: List[str], name: str = ""):
        self.concs = concs
        self.vib = vib
        self.elems = elems
        self.name = name

    def __str__(self) -> str:
        output = ""
        for conc, elem in zip(self.concs, self.elems):
            output += "{:>7.4f}{:>7.4f}            element {}\n".format(
                conc, self.vib, elem
            )
        return output


class Atom:
    def __init__(self, sitenum: int, x: float, y: float, z: float):
        self.sitenum = sitenum
        self.x = x
        self.y = y
        self.z = z


class Layer:
    def __init__(self, atoms: List[Atom], name: str = ""):
        self.sitenums = np.array([a.sitenum for a in atoms])
        self.xs = np.array([a.x for a in atoms])
        self.ys = np.array([a.y for a in atoms])
        self.zs = np.array([a.z for a in atoms])
        self.name = name

    def __len__(self):
        return len(self.sitenums)

    def __str__(self) -> str:
        output = "{:>3d}".format(len(self.sitenums))
        output += 23 * " " + "number of sublayers\n"
        for sitenum, z, x, y in zip(self.sitenums, self.zs, self.xs, self.ys):
            output += "{:>3d}{:>7.4f}{:>7.4f}{:>7.4f}\n".format(
                sitenum, z, x, y
            )
        return output


class SearchKey(enum.Enum):
    """ Enumeration defining the types of parameters which can be searched over
    """
    CONC = enum.auto()
    VIB = enum.auto()
    ATOMX = enum.auto()
    ATOMY = enum.auto()
    ATOMZ = enum.auto()


class AtomicStructure:
    """ Class holding all of the information needed to write the structure
         section of the input LEED script
    """

    def __init__(self, sites: List[Site], layers: List[Layer]):
        self.sites = sites
        self.layers = layers

    def to_script(self):
        """ Writes to a string the section to be placed inside
            of the LEED script
        """
        # Site description section
        output = (
            "-------------------------------------------------------------------\n"
            "--- define chem. and vib. properties for different atomic sites ---\n"
            "-------------------------------------------------------------------\n"
        )
        output += "{:>3d}".format(len(self.sites))
        output += 23 * " " + "NSITE: number of different site types\n"
        for i, site in enumerate(self.sites):
            output += "-   site type {}  {}---\n".format(i + 1, site.name)
            output += str(site)

        # Layer description section
        output += (
            "-------------------------------------------------------------------\n"
            "--- define different layer types                            *   ---\n"
            "-------------------------------------------------------------------\n"
        )
        output += "{:>3d}".format(len(self.layers))
        output += 23 * " " + "NLTYPE: number of different layer types\n"
        for i, layer in enumerate(self.layers):
            output += "-   layer type {}  {}---\n".format(i + 1, layer.name)
            output += "{:>3d}".format(i + 1)
            output += 23 * " " + "LAY = {}\n".format(i + 1)
            output += str(layer)

        return output

    def __getitem__(self, keyidx: Tuple[SearchKey, int]):
        """ Retrieve a structural parameter. NOTE: 1-based indexing!
        """
        key, idx = keyidx
        idx -= 1
        if key == SearchKey.VIB:
            return self.sites[idx].vib
        elif key == SearchKey.CONC:
            return self.sites[idx].concs
        elif key == SearchKey.ATOMX:
            return self.layers[0].xs[idx]
        elif key == SearchKey.ATOMY:
            return self.layers[0].ys[idx]
        elif key == SearchKey.ATOMZ:
            return self.layers[0].zs[idx]

    def __setitem__(self, keyidx: Tuple[SearchKey, int], value: float):
        """ Set a structural parameter. NOTE: 1-based indexing!
        """
        key, idx = keyidx
        idx -= 1
        if key == SearchKey.VIB:
            self.sites[idx].vib = value
        elif key == SearchKey.CONC:
            self.sites[idx].concs = value
        elif key == SearchKey.ATOMX:
            self.layers[0].xs[idx] = value
        elif key == SearchKey.ATOMY:
            self.layers[0].ys[idx] = value
        elif key == SearchKey.ATOMZ:
            self.layers[0].zs[idx] = value


SearchParam = Tuple[SearchKey, int]
SearchDim = Tuple[SearchKey, int, Tuple[float, float]]
SearchConstraint = Tuple[SearchKey, int, int]
class SearchSpace:
    """ Defines which parameters of an AtomicStructure should be held fixed / 
        searched over in an optimization problem, as well as the domain to
        search over.

        Should be initialized with an AtomicStructure, as well as a dictionary,
         where the keys specify parameters to be searched over, and the values
         are tuples of the intervals defining the search domain (minval, maxval).
        Note these intervals are interpreted as deviations from the values given
         by the atomic structure, not absolute coordinates.
        Also note that this assumes that all atoms to search over are in the
         first layer.

        Constraints can be provided as a list of tuples of the form
            (SEARCH_KEY, SEARCH_IDX, BOUND_IDX)
        which will make it so that whenever the search parameter (SEARCH_KEY, SEARCH_IDX)
         is sampled/changed, the variable at idx BOUND_IDX is changed to be equal.
    """

    def __init__(self, atomic_structure: AtomicStructure,
                 search_dims: List[SearchDim],
                 constraints: List[SearchConstraint] = None):
        if constraints is None:
            constraints = []
        self.atomic_structure = deepcopy(atomic_structure)
        self.search_params = [(key, idx) for key, idx, _ in search_dims]
        self.search_bounds = [bounds for _, _, bounds in search_dims]
        self.num_params = len(self.search_params)

        # Check validity of search_params
        for key, idx in self.search_params:
            if key == SearchKey.CONC:
                raise NotImplementedError("SearchKey.CONC not implemented yet.")
            elif key == SearchKey.VIB:
                if idx > len(self.atomic_structure.sites):
                    raise ValueError("SearchSpace idx out of bounds")
            else:
                if idx > len(self.atomic_structure.layers[0]):
                    raise ValueError("SearchSpace idx out of bounds")
            if idx < 0:
                raise ValueError("SearchSpace idx out of bounds")

        self.constraints = {param: [] for param in self.search_params}
        for key, search_idx, bound_idx in constraints:
            if (key, search_idx) not in self.search_params:
                raise ValueError("Search parameter in constraint not present")
            elif bound_idx in [idx for skey, idx in self.search_params if skey == key]:
                raise ValueError("Bound parameter is in search parameter list")
            self.constraints[(key, search_idx)].append(bound_idx)

    def random_points(self, num_pts: int) -> Tuple[np.ndarray, List[AtomicStructure]]:
        """ Returns num_pts number of random structures in the search space
        """
        random_unit_cube = np.random.random((num_pts, self.num_params))
        return random_unit_cube, self.to_structures(random_unit_cube)

    # TODO
    def warm_start(self, num_pts: int,
                   dist: float, sol: AtomicStructure) -> Tuple[np.ndarray, List[AtomicStructure]]:
        """ Returns n_pts number of random structures which are close to a
             known solution. Note dist is distance in normalized space!
        """
        random_unit_vecs = np.random.random((num_pts, len(self.num_params)))
        random_unit_vecs /= np.linalg.norm(random_unit_vecs, axis=1, keepdims=True)
        sol_vec = self.to_normalized([sol])
        random_pts = sol + dist * random_unit_vecs
        return self.to_structures(sol + random_unit_vecs)

    def _normal_to_structure(self, norm_vec) -> AtomicStructure:
        new_struct = deepcopy(self.atomic_structure)

        for val, param, lims in zip(norm_vec, self.search_params, self.search_bounds):
            key, idx = param
            bound_idxs = self.constraints[param]
            new_struct[key, idx] += lims[0] + val * (lims[1] - lims[0])
            for b_idx in bound_idxs:
                new_struct[key, b_idx] = new_struct[key, idx]

        return new_struct

    def to_structures(self, norm_vecs) -> AtomicStructure:
        """ Converts a normalized feature vector to the corresponding AtomicStructure.
            If norm_vecs is a single vector, returns a single AtomicStructure, if
               norm_vecs is a list of vectors or 2D array, returns a list of AtomicStructures
        """
        if len(norm_vecs.shape) == 1:
            return self._normal_to_structure(norm_vecs)
        structs = [self._normal_to_structure(vec) for vec in norm_vecs]
        return structs

    def to_normalized(self, struct: AtomicStructure) -> np.ndarray:
        """ Converts a single AtomicStructure to a normalized feature vector.
            TODO: Should this also work on list of structures?
        """
        norm_vec = np.empty(self.num_params)
        for i, (param, lims) in enumerate(zip(self.search_params, self.search_bounds)):
            key, idx = param
            norm_vec[i] = ((struct[key, idx] - self.atomic_structure[key, idx] - lims[0])
                           / (lims[1] - lims[0]))

        return norm_vec


class LEEDManager:
    def __init__(self, basedir, leed_executable, rfactor_executable,
                 exp_datafile, templatefile):
        """ Create a LEEDManager to keep track of TensErLEED components.
                basedir: The base directory to do computation in
                leed_executable: Path to the LEED executable
                rfactor_executable: Path to the rfactor executable
                exp_datafile: Path to the experimental datafile
                templatefile: The base template for the LEED input file (FIN)
        """
        for path in [basedir, leed_executable, rfactor_executable, templatefile]:
            if not os.path.exists(path):
                raise ValueError("File not found: {}".format(path))
        self.basedir = os.path.abspath(basedir)
        self.leed_exe = os.path.abspath(leed_executable)
        self.rfactor_exe = os.path.abspath(rfactor_executable)
        self.exp_datafile = os.path.abspath(exp_datafile)
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
            self.input_template = f.readlines()
        self.calc_number = 0

    def _write_structure(self, structure, filename):
        new_coord_sect = structure.to_script()

        # Find line before where new coordinates need to be inserted, as well as the
        #  line which marks the following section
        indbefore, indafter = -1, -1
        for i, line in enumerate(self.input_template):
            if line.find("define chem. and vib. properties") != -1:
                indbefore = i - 1
            elif line.find("define bulk stacking sequence") != -1:
                indafter = i - 1
        # Check that both lines were found
        if indbefore == -1 or indafter == -1:
            raise ValueError("LEED input file does not contain section marker lines")
        with open(filename, "w") as ofile:
            ofile.writelines(self.input_template[:indbefore])
            ofile.write(new_coord_sect)
            ofile.writelines(self.input_template[indafter:])

    def _start_calc(self, structure: AtomicStructure, calcid: int):
        """ Start a new process running TLEED, and return the subprocess
              handle and working directory without waiting for it to finish
        """
        newdir = os.path.join(self.basedir, "ref-calc" + str(calcid))
        os.makedirs(newdir, exist_ok=True)
        # shutil.copy(self.exp_datafile, os.path.join(newdir, "WEXPEL"))
        input_filename = os.path.join(newdir, "FIN")
        stdout_filename = os.path.join(newdir, "protocol")
        self._write_structure(structure, input_filename)
        process = subprocess.Popen(
            [self.leed_exe],
            stdin=open(input_filename, "r"),
            stdout=open(stdout_filename, "w"),
            cwd=newdir,
            text=True
        )
        return process, newdir

    def _rfactor_calc(self, refcalc_outfile):
        """ Runs the r-factor calculation, and parses the output,
             returning the result
        """
        result = subprocess.run(
            [self.rfactor_exe],
            cwd=self.basedir,
            stdin=open(refcalc_outfile, "r"),
            capture_output=True,
            text=True
        )
        return extract_rfactor(result.stdout)

    def ref_calc(self, structure: AtomicStructure):
        """ Do the full process of performing a reference calculation.
            NOTE: Do not call this function in parallel, as there is a race
                condition on self.calc_number. Instead, use batch_ref_calcs.
        """
        self.calc_number += 1
        calc_process, pdir = self._start_calc(structure, self.calc_number)
        calc_process.wait()
        result_filename = os.path.join(pdir, "fd.out")
        return self._rfactor_calc(result_filename)

    def batch_ref_calcs(self, structures: List[AtomicStructure]):
        """ Run multiple reference calculations in parallel, one for each
             row of displacements.
        """
        num_structs = len(structures)

        # Start up all of the calculation processes
        logging.info("Starting {} reference calculations...".format(num_structs))
        processes = []
        for i in range(num_structs):
            self.calc_number += 1
            processes.append(
                self._start_calc(structures[i], self.calc_number)
            )

        # Wait for all of them to complete, calculate r-factors for each
        # The r-factor calculations are fast enough that we may as well run
        #   them serially
        rfactors = []
        for p, pdir in processes:
            p.wait()
            result_filename = os.path.join(pdir, "fd.out")
            rfactors.append(self._rfactor_calc(result_filename))
        logging.info("Reference calculations completed.")
        return np.array(rfactors)


def extract_rfactor(output):
    p = re.compile(r"AVERAGE R-FACTOR =  (\d\.\d+)")
    m = re.search(p, output)
    if m is not None:
        return float(m.group(1))
    else:
        import ipdb
        ipdb.set_trace()
        raise ValueError("No average R-factor line found in input")
