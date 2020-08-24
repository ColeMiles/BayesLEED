from __future__ import annotations

import enum
import typing
from copy import deepcopy
from typing import List, Tuple, Union, Sequence

import numpy as np

if typing.TYPE_CHECKING:
    from .tleed import RefCalc, SiteDeltaAmps, MultiDeltaAmps
    from .structure import AtomicStructure
    from .curves import IVCurveSet

from . import curves


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
         is sampled/changed, the variable at idx BOUND_IDX is changed to match the constraint.
        TODO: Re-do how constraints are handled to allow arbitrary multi-atom constraints
    """

    def __init__(self, atomic_structure: AtomicStructure,
                 search_dims: List[SearchDim],
                 constraints: List[Constraint] = None):
        self.atomic_structure = deepcopy(atomic_structure)
        self.search_params = [(key, idx) for key, idx, _ in search_dims]
        self.search_bounds = [bounds for _, _, bounds in search_dims]
        self.num_params = len(self.search_params)

        if constraints is None:
            constraints = list()

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


# While the search space in the full space is continuous, due to how TensErLEED functions it is necessary (for now)
#  to limit these to search space of a discrete grid of points. In principle, this can be made continuous as well,
#  however that will increase the computational effort greatly since T-matrices will have be to be re-generated on
#  every evaluation. Maybe there is some nice compromise?
# TODO: Make a continuous version for comparison.
# (atom_idx, disps_list, vibs_list) : Displacements should be in un-normalized coordinates!
DeltaSearchDim = Tuple[int, List[Sequence[float]], Sequence[float]]


class DeltaSearchSpace:
    """ This class defines the search space for a TensorLEED perturbative search around a reference calculation.
    """
    def __init__(self, ref_calc: RefCalc, search_dims: List[DeltaSearchDim], constraints: List[Constraint] = None):
        """ Constructor
            ref_calc: A (completed) reference calculation which to perturbatively search around
            search_dims: A list of discrete search dimensions. The list of search values are interpreted as *deltas*
                          to the values of the reference calculation, and are interpreted in un-normalized coordinates
                          (Angstroms).
            constraints: A list of constraints to constrain search dimensions. [Not implemented currently].
            TODO: Implement constraints, and different displacement sets per site.
        """
        self.ref_calc = ref_calc
        self.struct = ref_calc.struct
        self.atoms = [atom for atom, _, _ in search_dims]
        self.search_disps = [disps for _, disps, _ in search_dims]
        self.search_vibs = [vibs for _, _, vibs in search_dims]
        self.num_params = len(search_dims)
        self.search_dims = search_dims
        self.constraints = constraints


def optimize_delta_anneal(search_space: DeltaSearchSpace, multi_delta_amps: MultiDeltaAmps,
                          exp_curves: IVCurveSet, nindivs: int = 25,
                          nepochs: int = 100000, init_gaus: float = 0.5,
                          gaus_decay: float = 0.9999) -> Tuple[Tuple[int, int], float]:
    """ Optimize the TLEED problem using a simulated-annealing type algorithm, similar to
         how the original Fortran implements this.
        Returns (disp, rfactor) of the integer index of the displacement and the corresponding r-factor
    """
    shifts, vibs = search_space.search_disps[0], search_space.search_vibs[0]
    # TODO: Change language globally. Disps are combined geo/shifts + vib.
    nsites, ngeo, nvibs = len(multi_delta_amps), len(shifts), len(vibs)
    ndisps = ngeo * nvibs

    # Join geometric and vibrational displacements into a combined lattice
    # Each site will be displaced according to some point in this lattice
    disps = np.empty((ndisps, 4))
    disps[:, :3] = np.tile(shifts, (nvibs, 1))
    disps[:, 3] = np.repeat(vibs, ngeo)

    # Randomly initialize each individual to a lattice point
    indivs = np.random.randint(0, ndisps, (nindivs, nsites))
    rfactors = np.empty(nindivs)

    # Calculate all of these initial r-factors
    for iindiv in range(nindivs):
        new_curves = multi_delta_amps.compute_curves(indivs[iindiv])
        rfactors[iindiv] = np.min(curves.avg_rfactors(exp_curves, new_curves))

    # Current width of the Gaussian used for sampling moves
    gaus_width = init_gaus

    # TODO: Parallelize across individuals
    for epoch in range(nepochs):
        for iindiv in range(nindivs):
            # Displacement indexes per site of lattice
            curr_grid_pt = indivs[iindiv]
            # Actual displacement vectors
            curr_disp = np.empty((nsites, 4))
            for isite in range(nsites):
                curr_disp[isite] = disps[curr_grid_pt[isite]]

            # Sample a move
            propose_disp = curr_disp + gaus_width * np.random.randn(nsites, 4)

            # Get the nearest grid points in the lattice
            propose_grid_pt = np.empty(nsites, dtype=np.int64)
            for isite in range(nsites):
                propose_grid_pt[isite] = np.argmin(
                    np.sum(np.square(disps - propose_disp[isite]), axis=-1)
                )

            # Calculate new IV curves
            propose_curves = multi_delta_amps.compute_curves(propose_grid_pt)
            # Calculate new rfactor
            propose_rfactor = np.min(curves.avg_rfactors(exp_curves, propose_curves))

            # Check if better than current rfactor; if so, make the move
            if propose_rfactor < rfactors[iindiv]:
                rfactors[iindiv] = propose_rfactor
                indivs[iindiv] = propose_grid_pt

        # Decay the gaussian width
        gaus_width *= gaus_decay

        if epoch % 10000 == 0:
            print("Epoch: {}, Best R-factor: {}".format(epoch, np.min(rfactors)))

    best_idx = np.argmin(rfactors)
    best_indiv, best_rfactor = indivs[best_idx], rfactors[best_idx]
    return divmod(best_indiv, nvibs), best_rfactor
