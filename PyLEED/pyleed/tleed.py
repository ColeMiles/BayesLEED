""" Python code for interfacing with the TensErLEED code.
"""
import os
import shutil
import subprocess
import re
import logging
import enum
from copy import deepcopy
from typing import List, Tuple, Union, Collection

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

    def __repr__(self):
        return "Site({}, {}, {}, {})".format(repr(self.concs), repr(self.vib), repr(self.elems), repr(self.name))

    def to_script(self) -> str:
        output = ""
        for conc, elem in zip(self.concs, self.elems):
            output += "{:>7.4f}{:>7.4f}            element {}\n".format(
                conc, self.vib, elem
            )
        return output


class Atom:
    def __init__(self, sitenum: int, x: float, y: float, z: float):
        self.sitenum = sitenum
        self.coord = np.array([x, y, z])

    def __repr__(self):
        return "Atom({}, {}, {}, {})".format(self.sitenum, *self.coord)

    @property
    def x(self):
        return self.coord[0]

    @x.setter
    def x(self, value):
        self.coord[0] = value

    @property
    def y(self):
        return self.coord[1]

    @y.setter
    def y(self, value):
        self.coord[1] = value

    @property
    def z(self):
        return self.coord[2]

    @z.setter
    def z(self, value):
        self.coord[2] = value


class Layer:
    def __init__(self, atoms: List[Atom], name: str = ""):
        self.sitenums = np.array([a.sitenum for a in atoms])
        self.xs = np.array([a.x for a in atoms])
        self.ys = np.array([a.y for a in atoms])
        self.zs = np.array([a.z for a in atoms])
        self.name = name

    def __len__(self):
        return len(self.sitenums)

    def __iter__(self):
        for sitenum, x, y, z in zip(self.sitenums, self.xs, self.ys, self.zs):
            yield Atom(sitenum, x, y, z)

    def __repr__(self):
        result = "Layer([\n"
        for atom in iter(self):
            result += "    " + repr(atom) + ",\n"
        result += "],\n"
        result += "    " + repr(self.name) + "\n"
        result += ")"
        return result

    def to_script(self, lat_params) -> str:
        lat_a, lat_b, lat_c = lat_params
        output = "{:>3d}".format(len(self.sitenums))
        output += 23 * " " + "number of sublayers\n"
        for sitenum, z, x, y in zip(self.sitenums, self.zs, self.xs, self.ys):
            output += "{:>3d}{:>7.4f}{:>7.4f}{:>7.4f}\n".format(
                sitenum, z * lat_c, x * lat_a, y * lat_b
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
    CELLA = enum.auto()
    CELLB = enum.auto()
    CELLC = enum.auto()


class Constraint:
    def __init__(self, search_key, search_idx, bound_key, bound_idx):
        self.search_key = search_key
        self.search_idx = search_idx
        self.bound_key = bound_key
        self.bound_idx = bound_idx


class EqualityConstraint(Constraint):
    def __init__(self, search_key, search_idx, bound_key, bound_idx):
        super().__init__(search_key, search_idx, bound_key, bound_idx)


class EqualShiftConstraint(Constraint):
    def __init__(self, search_key, search_idx, bound_key, bound_idx):
        super().__init__(search_key, search_idx, bound_key, bound_idx)


class AtomicStructure:
    """ Class holding all of the information needed to write the structure
         sections of the input LEED script. Layer coordinates should be in
         fractional coordinates. (This currently assumes orthorhombic unit cells).
    """

    def __init__(self, sites: List[Site], layers: List[Layer], cell_params: List[float]):
        self.sites = sites
        self.layers = layers
        self.cell_params = np.array(cell_params)

    def __repr__(self):
        result = "AtomicStructure(\n"
        result += "    [\n"
        for site in self.sites:
            result += 8 * " " + repr(site) + ",\n"
        result += "    ],\n"
        result += "    [\n"
        for layer in self.layers:
            layer_repr_lines = [8 * " " + line for line in repr(layer).splitlines(keepends=True)]
            result += "".join(layer_repr_lines) + ",\n"
        result += "    ],\n"
        result += "    " + repr(self.cell_params.tolist()) + "\n"
        result += ")"
        return result

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
            output += site.to_script()

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
            output += layer.to_script(self.cell_params)

        return output

    def __getitem__(self, keyidx: Tuple[SearchKey, int]):
        """ Retrieve a structural parameter. NOTE: 1-based indexing!
            Also note: indexing CONC works differently!
            TODO: Also make indexing VIB work like this?
            TODO: Move this logic into SearchSpace?
        """
        key, idx = keyidx
        idx -= 1
        if key == SearchKey.VIB:
            return self.sites[idx].vib
        elif key == SearchKey.CONC:
            site_idx, conc_idx = divmod(idx, len(self.sites[0].concs))
            return self.sites[site_idx].concs[conc_idx]
        elif key == SearchKey.ATOMX:
            return self.layers[0].xs[idx]
        elif key == SearchKey.ATOMY:
            return self.layers[0].ys[idx]
        elif key == SearchKey.ATOMZ:
            return self.layers[0].zs[idx]
        elif key == SearchKey.CELLA:
            return self.cell_params[0]
        elif key == SearchKey.CELLB:
            return self.cell_params[1]
        elif key == SearchKey.CELLC:
            return self.cell_params[2]

    def __setitem__(self, keyidx: Tuple[SearchKey, int], value: float):
        """ Set a structural parameter. NOTE: 1-based indexing!
            Also note: indexing CONC works differently!
        """
        key, idx = keyidx
        idx -= 1
        if key == SearchKey.VIB:
            self.sites[idx].vib = value
        elif key == SearchKey.CONC:
            site_idx, conc_idx = divmod(idx, len(self.sites[0].concs))
            self.sites[site_idx].concs[conc_idx] = value
        elif key == SearchKey.ATOMX:
            self.layers[0].xs[idx] = value
        elif key == SearchKey.ATOMY:
            self.layers[0].ys[idx] = value
        elif key == SearchKey.ATOMZ:
            self.layers[0].zs[idx] = value
        elif key == SearchKey.CELLA:
            self.cell_params[0] = value
        elif key == SearchKey.CELLB:
            self.cell_params[1] = value
        elif key == SearchKey.CELLC:
            self.cell_params[2] = value

    def write_xyz(self, filename: str, comment: str = ""):
        """ Writes the atomic structure to an XYZ file. Overwrites file if it
             already exists.
        """
        with open(filename, "w") as f:
            f.write(str(sum(map(len, self.layers))) + "\n")
            f.write(comment + "\n")
            current_z = 0.0
            for layer in self.layers:
                for atom in layer:
                    site = self.sites[atom.sitenum-1]
                    elem = site.elems[np.argmax(site.concs).item()]
                    f.write("{:>3s}{:>10.5f}{:>10.5f}{:>10.5f}\n".format(
                        elem,
                        atom.x * self.cell_params[0],
                        atom.y * self.cell_params[1],
                        (current_z + atom.z) * self.cell_params[2]
                    ))
                current_z += np.ceil(max(layer.zs))

    def write_cif(self, filename: str, comment: str = ""):
        """ Writes the atomic structure to a CIF file. Overwrites files if it
             already exists.
            Note: All layers are put into a single unit cell, where the c parameters
             is made large enough to fit all layers.
        """
        # Number of unit cells along the c axis spanned by each layer
        layer_num_cells = [np.ceil(np.max(layer.zs)) for layer in self.layers]
        tot_num_cells = sum(layer_num_cells)

        with open(filename, "w") as f:
            f.writelines([
                comment + "\n",
                "_symmetry_space_group_name_H-M   'P 1'\n",
                "_cell_length_a {:>12.8f}\n".format(self.cell_params[0]),
                "_cell_length_b {:>12.8f}\n".format(self.cell_params[1]),
                "_cell_length_c {:>12.8f}\n".format(tot_num_cells * self.cell_params[2]),
                "_cell_length_alpha {:>12.8f}\n".format(90.0),
                "_cell_length_beta {:>12.8f}\n".format(90.0),
                "_cell_length_gamma {:>12.8f}\n".format(90.0),
                "_symmetry_Int_Tables_number   1\n",
                "_chemical_formula_structural\n",
                "_chemical_formula_sum\n",
                "_cell_volume {:>12.8f}\n".format(tot_num_cells * np.prod(self.cell_params)),
                "_cell_formula_unitz_Z   2\n",
                "loop_\n",
                " _symmetry_equiv_pos_site_id\n",
                " _symmetry_equiv_pos_as_xyz\n",
                "  1  'x, y, z'\n",
                "loop_\n",
                " _atom_site_type_symbol\n",
                " _atom_site_label\n",
                " _atom_site_symmetry_multiplicity\n",
                " _atom_site_fract_x\n",
                " _atom_site_fract_y\n",
                " _atom_site_fract_z\n",
                " _atom_site_occupancy\n",
            ])
            atom_num = 0
            cell_num = 0
            for layer_idx, layer in enumerate(self.layers):
                for atom in layer:
                    site = self.sites[atom.sitenum-1]
                    elem = site.elems[np.argmax(site.concs).item()]
                    f.write("{:>4s} {:>5s} {:>3d} {:>10.6f} {:>10.6f} {:>10.6f} {:>3d}\n".format(
                        elem, elem + str(atom_num), 1,
                        atom.x, atom.y, (atom.z + cell_num) / tot_num_cells, 1
                    ))
                    atom_num += 1
                cell_num += layer_num_cells[layer_idx]


SearchParam = Tuple[SearchKey, int]
SearchDim = Tuple[SearchKey, int, Tuple[float, float]]
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
         first layer. (Except for lattice parameters, which also apply to the bulk)

        Constraints can be provided as a list of tuples of the form
            (SEARCH_KEY, SEARCH_IDX, BOUND_IDX)
        which will make it so that whenever the search parameter (SEARCH_KEY, SEARCH_IDX)
         is sampled/changed, the variable at idx BOUND_IDX is changed to be equal.
    """

    def __init__(self, atomic_structure: AtomicStructure,
                 search_dims: List[SearchDim],
                 constraints: List[Constraint] = None):
        if constraints is None:
            constraints = []
        self.atomic_structure = deepcopy(atomic_structure)
        self.search_params = [(key, idx) for key, idx, _ in search_dims]
        self.search_bounds = [bounds for _, _, bounds in search_dims]
        self.num_params = len(self.search_params)

        # Validate search parameters
        for key, idx in self.search_params:
            if key == SearchKey.CONC:
                raise NotImplementedError("SearchKey.CONC not implemented yet.")
            elif key in [SearchKey.CELLA, SearchKey.CELLB, SearchKey.CELLC]:
                continue
            elif key == SearchKey.VIB:
                if idx > len(self.atomic_structure.sites):
                    raise ValueError("SearchSpace idx out of bounds")
            else:
                if idx > len(self.atomic_structure.layers[0]):
                    raise ValueError("SearchSpace idx out of bounds")
            if idx < 0:
                raise ValueError("SearchSpace idx out of bounds")

        self.constraints = {param: [] for param in self.search_params}
        # Validate constraints
        for constraint in constraints:
            search_key = constraint.search_key
            search_idx = constraint.search_idx
            bound_key = constraint.bound_key
            bound_idx = constraint.bound_idx
            if (search_key, search_idx) not in self.search_params:
                raise ValueError("Search parameter in constraint not present")
            elif bound_key in [SearchKey.CELLA, SearchKey.CELLB, SearchKey.CELLC]:
                if bound_key in [skey for skey, idx in self.search_params]:
                    raise ValueError("Bound parameter is in search parameter list")
                self.constraints[(search_key, search_idx)].append(constraint)
                continue
            elif bound_idx in [idx for skey, idx in self.search_params if skey == search_key]:
                raise ValueError("Bound parameter is in search parameter list")
            self.constraints[(search_key, search_idx)].append(constraint)

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

    def _normal_to_structure(self, norm_vec: np.ndarray) -> Union[AtomicStructure, List[AtomicStructure]]:
        new_struct = deepcopy(self.atomic_structure)

        for val, param, lims in zip(norm_vec, self.search_params, self.search_bounds):
            key, idx = param
            bound_constraints = self.constraints[param]
            new_struct[key, idx] += lims[0] + val * (lims[1] - lims[0])

            # Apply all constraints
            for constraint in bound_constraints:
                b_key = constraint.bound_key
                b_idx = constraint.bound_idx
                if isinstance(constraint, EqualityConstraint):
                    new_struct[b_key, b_idx] = new_struct[key, idx]
                elif isinstance(constraint, EqualShiftConstraint):
                    new_struct[b_key, b_idx] += lims[0] + val * (lims[1] - lims[0])

        return new_struct

    def to_structures(self, norm_vecs) -> Union[AtomicStructure, List[AtomicStructure]]:
        """ Converts normalized feature vectors to the corresponding AtomicStructures.
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

        # TODO: Maybe an Enum for a state rather than this?
        self.completed = False
        self.in_progress = False

        self.script_filename = os.path.join(self.workdir, "FIN")
        self.result_filename = os.path.join(self.workdir, "fd.out")
        self._process = None

        if produce_tensors:
            self.tensor_filenames = [
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

    def _write_delta_script(self):
        raise NotImplementedError()

    def run(self):
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
        """ Waits for completion.
        """
        self._process.wait()
        # TODO: Can I make these update without a call to .wait()?
        self.in_progress = False
        self.completed = True

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

    def produce_curves(self):
        # TODO: Returns an IVCurve class, which can normalize and plot itself!
        raise NotImplementedError()



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
            rfactors.append(refcalc.rfactor())
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
