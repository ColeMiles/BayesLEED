from __future__ import annotations
from typing import List, Tuple, Collection

from .searchspace import SearchKey

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


# TODO: I don't like how this class is arranged
class Layer:
    """ Stores the coordinates of atoms within a single 'layer' in the LEED
         context. Coordinates should be normalized, to be converted using
         the unit cell parameters in a surrounding AtomicStructure.
        The interlayer_vec is the vector from the bottom of this layer to the
         (0, 0, 0) coordinate of the next layer. This should be in unnormalized
         coordinates (Angstroms).
    """
    def __init__(self, atoms: List[Atom], interlayer_vec: Collection[float], name: str = ""):
        self.sitenums = np.array([a.sitenum for a in atoms])
        self.xs = np.array([a.x for a in atoms])
        self.ys = np.array([a.y for a in atoms])
        self.zs = np.array([a.z for a in atoms])
        self.name = name
        self.atoms = atoms
        self.interlayer_vec = np.array(interlayer_vec)
        assert len(interlayer_vec) == 3, "Interlayer vector must be 3-dimensional"

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
        result += "    " + repr(self.interlayer_vec) + ",\n"
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


# TODO: Mapping of siteidx -> element name should be global within this class
class AtomicStructure:
    """ Class holding all of the information needed to write the structure
         sections of the input LEED script. Layer coordinates should be in
         fractional coordinates. (This currently assumes orthorhombic unit cells).
    """

    def __init__(self, sites: List[Site], layers: List[Layer], cell_params: List[float]):
        self.sites = sites
        self.layers = layers
        self.cell_params = np.array(cell_params)
        self.num_elems = len(sites[0].elems)

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

    def dist(self, other: AtomicStructure):
        """ Computes total Euclidean distance to another AtomicStructure. Only defined
             if both structures have the same number of atoms in each layer.
        """
        dist = 0.0
        a, b, c = self.cell_params
        oa, ob, oc = other.cell_params
        for layer, other_layer in zip(self.layers, other.layers):
            if len(layer) != len(other_layer):
                raise ValueError(
                    "Cannot call .dist() between structures with unequal numbers of"
                    " atoms in matching layers."
                )
            dist += np.sum(np.square(a * layer.xs - oa * other_layer.xs)).item()
            dist += np.sum(np.square(b * layer.ys - ob * other_layer.ys)).item()
            dist += np.sum(np.square(c * layer.zs - oc * other_layer.zs)).item()
        return np.sqrt(dist)
