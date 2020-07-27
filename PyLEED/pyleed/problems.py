""" Contains definitions for example problems for the current data """
# TODO: If anyone else will ever use this, should probably migrate to real configuration files
from .structure import *
from .searchspace import *
from .tleed import *
import numpy as np

FESE_BEAMINFO_TRIMMED = BeamInfo(
    0.0, 0.0,    # Incident theta, phi
    [            # (kx, ky) of beams
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 2),
        (3, 0)
    ],
    30.0, 550.0  # Emin, Emax
)

FESE_20UC = AtomicStructure(
    # Atomic sites
    [
     Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe top layer"),
     Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe 2nd layer"),
     Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe bulk"),
     Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se top layer"),
     Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se 2nd layer"),
     Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se bulk")
    ],
    # Layer definitions (fractional coordinates)
    [
     Layer([
        Atom(1, 0.25, 0.75, 0.25),  # Top Layer Fe
        Atom(1, 0.75, 0.25, 0.25),  # Top Layer Fe
        Atom(2, 0.25, 0.75, 1.25),  # 2nd Layer Fe
        Atom(2, 0.75, 0.25, 1.25),  # 2nd Layer Fe
        Atom(4, 0.25, 0.25, 0.00),  # Top Layer Se
        Atom(4, 0.75, 0.75, 0.50),  # Top Layer Se
        Atom(5, 0.25, 0.25, 1.00),  # 2nd Layer Se
        Atom(5, 0.75, 0.75, 1.50),  # 2nd Layer Se
        ],
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3, 0.25, 0.75, 0.25),  # Bulk Fe
        Atom(3, 0.75, 0.25, 0.25),  # Bulk Fe
        Atom(6, 0.25, 0.25, 0.00),  # Bulk Se
        Atom(6, 0.75, 0.75, 0.50),  # Bulk Se
        ],
        "Bulk"
     )
    ],
    # Unit cell parameters
    [3.7676, 3.7676, 5.5180]
)

FESE_20UC_SINGLEZREGRESSED = AtomicStructure(
    # Atomic sites
    [
        Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe top layer"),
        Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe 2nd layer"),
        Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe bulk"),
        Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se top layer"),
        Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se 2nd layer"),
        Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se bulk")
    ],
    # Layer definitions (fractional coordinates)
    [
        Layer([
            Atom(1, 0.25, 0.75, 0.26255),  # Top Layer Fe
            Atom(1, 0.75, 0.25, 0.26387),  # Top Layer Fe
            Atom(2, 0.25, 0.75, 1.2500),  # 2nd Layer Fe
            Atom(2, 0.75, 0.25, 1.2500),  # 2nd Layer Fe
            Atom(4, 0.25, 0.25, 0.00655),  # Top Layer Se
            Atom(4, 0.75, 0.75, 0.5322),  # Top Layer Se
            Atom(5, 0.25, 0.25, 1.0000),  # 2nd Layer Se
            Atom(5, 0.75, 0.75, 1.5000),  # 2nd Layer Se
        ],
            "Top 2 unit cells"
        ),
        Layer([
            Atom(3, 0.25, 0.75, 0.2500),  # Bulk Fe
            Atom(3, 0.75, 0.25, 0.2500),  # Bulk Fe
            Atom(6, 0.25, 0.25, 0.0000),  # Bulk Se
            Atom(6, 0.75, 0.75, 0.5000),  # Bulk Se
        ],
            "Bulk"
        )
    ],
    # Unit cell parameters
    [3.7659, 3.7659, 5.51547]
)


TEST_FESE_20UC_FOR2D = AtomicStructure(
    # Atomic sites
    [
        Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe top layer"),
        Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe 2nd layer"),
        Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe bulk"),
        Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se top layer"),
        Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se 2nd layer"),
        Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se bulk")
    ],
    # Layer definitions (fractional coordinates)
    [
        Layer([
            Atom(1, 0.25, 0.75, 0.25000),  # Top Layer Fe  <-------- Search over these two
            Atom(1, 0.75, 0.25, 0.25000),  # Top Layer Fe  <----|
            Atom(2, 0.25, 0.75, 1.2500),  # 2nd Layer Fe
            Atom(2, 0.75, 0.25, 1.2500),  # 2nd Layer Fe
            Atom(4, 0.25, 0.25, 0.00655),  # Top Layer Se
            Atom(4, 0.75, 0.75, 0.5322),  # Top Layer Se
            Atom(5, 0.25, 0.25, 1.0000),  # 2nd Layer Se
            Atom(5, 0.75, 0.75, 1.5000),  # 2nd Layer Se
        ],
            "Top 2 unit cells"
        ),
        Layer([
            Atom(3, 0.25, 0.75, 0.2500),  # Bulk Fe
            Atom(3, 0.75, 0.25, 0.2500),  # Bulk Fe
            Atom(6, 0.25, 0.25, 0.0000),  # Bulk Se
            Atom(6, 0.75, 0.75, 0.5000),  # Bulk Se
        ],
            "Bulk"
        )
    ],
    # Unit cell parameters
    [3.7659, 3.7659, 5.51547]
)


FESE_20UC_CLOSE = AtomicStructure(
    # Atomic sites
    [
        Site([1.0, 0.0], 0.1319, ["Fe", "Se"], "Fe top layer"),
        Site([1.0, 0.0], 0.1319, ["Fe", "Se"], "Fe 2nd layer"),
        Site([1.0, 0.0], 0.1319, ["Fe", "Se"], "Fe bulk"),
        Site([0.0, 1.0], 0.1204, ["Fe", "Se"], "Se top layer"),
        Site([0.0, 1.0], 0.1204, ["Fe", "Se"], "Se 2nd layer"),
        Site([0.0, 1.0], 0.1204, ["Fe", "Se"], "Se bulk")
    ],
    # Layer definitions (fractional coordinates)
    [
        Layer([
            Atom(1, 0.25, 0.75, 0.26126),  # Top Layer Fe
            Atom(1, 0.75, 0.25, 0.27046),  # Top Layer Fe
            Atom(2, 0.25, 0.75, 1.24418),  # 2nd Layer Fe
            Atom(2, 0.75, 0.25, 1.27062),  # 2nd Layer Fe
            Atom(4, 0.25, 0.25, 0.00732),  # Top Layer Se
            Atom(4, 0.75, 0.75, 0.53256),  # Top Layer Se
            Atom(5, 0.25, 0.25, 0.99870),  # 2nd Layer Se
            Atom(5, 0.75, 0.75, 1.51332),  # 2nd Layer Se
        ],
            "Top 2 unit cells"
        ),
        Layer([
            Atom(3, 0.25, 0.75, 0.25),  # Bulk Fe
            Atom(3, 0.75, 0.25, 0.25),  # Bulk Fe
            Atom(6, 0.25, 0.25, 0.00),  # Bulk Se
            Atom(6, 0.75, 0.75, 0.50),  # Bulk Se
        ],
            "Bulk"
        )
    ],
    # Unit cell parameters
    [3.7667322, 3.7667322, 5.513444]
)


LANIO3 = AtomicStructure(
    # Atomic sites
    [
     Site([0.0, 0.0, 1.0, 0.0], 0.14,   ["apO", "eqO", "La", "Ni"], "La top layer"),
     Site([0.0, 0.0, 1.0, 0.0], 0.02,   ["apO", "eqO", "La", "Ni"], "La 2nd layer"),
     Site([0.0, 0.0, 1.0, 0.0], 0.0298,  ["apO", "eqO", "La", "Ni"], "La bulk"),
     Site([0.0, 0.0, 0.0, 1.0], 0.02,   ["apO", "eqO", "La", "Ni"], "Ni top layer"),
     Site([0.0, 0.0, 0.0, 1.0], 0.04,   ["apO", "eqO", "La", "Ni"], "Ni 2nd layer"),
     Site([0.0, 0.0, 0.0, 1.0], 0.0298, ["apO", "eqO", "La", "Ni"], "Ni bulk"),
     Site([1.0, 0.0, 0.0, 0.0], 0.14,   ["apO", "eqO", "La", "Ni"], "apO top layer"),
     Site([1.0, 0.0, 0.0, 0.0], 0.14,   ["apO", "eqO", "La", "Ni"], "apO 2nd layer"),
     Site([1.0, 0.0, 0.0, 0.0], 0.0528, ["apO", "eqO", "La", "Ni"], "apO bulk"),
     Site([0.0, 1.0, 0.0, 0.0], 0.20,   ["apO", "eqO", "La", "Ni"], "eqO top layer"),
     Site([0.0, 1.0, 0.0, 0.0], 0.18,   ["apO", "eqO", "La", "Ni"], "eqO 2nd layer"),
     Site([0.0, 1.0, 0.0, 0.0], 0.0528, ["apO", "eqO", "La", "Ni"], "eqO bulk"),
    ],
    # Layer definitions
    [
     Layer([
        Atom(1,  0.5, 0.5, 0.0),
        Atom(7,  0.0, 0.0, 0.0),
        Atom(4,  0.0, 0.0, 0.5),
        Atom(10, 0.5, 0.0, 0.5),
        Atom(10, 0.0, 0.5, 0.5),
        Atom(2,  0.5, 0.5, 1.0),
        Atom(8,  0.0, 0.0, 1.0),
        Atom(5,  0.0, 0.0, 1.5),
        Atom(11, 0.5, 0.0, 1.5),
        Atom(11, 0.0, 0.5, 1.5),
     ],
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3,  0.5, 0.5, 0.0),
        Atom(9,  0.0, 0.0, 0.0),
        Atom(6,  0.0, 0.0, 0.5),
        Atom(12, 0.5, 0.0, 0.5),
        Atom(12, 0.0, 0.5, 0.5)
     ],
        "Bulk"
     )
    ],
    # Unit cell parameters
    [3.7910, 3.7910, 3.9000]
)

LANIO3_PROBLEM = SearchSpace(
    LANIO3,
    [
        (SearchKey.ATOMZ,  1, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  2, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  3, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  4, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  6, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  7, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  8, (-0.066, 0.066)),
        (SearchKey.ATOMZ,  9, (-0.066, 0.066)),
    ],
    constraints=[  # Bind matching eqOs to be equal in z coordinate
        EqualityConstraint(SearchKey.ATOMZ, 4, SearchKey.ATOMZ, 5),
        EqualityConstraint(SearchKey.ATOMZ, 9, SearchKey.ATOMZ, 10),
    ]
)

# Absolute coordinate shifts!
LANIO3_SOLUTION = np.array(
    [0.2200, -0.1800, 0.0000, -0.0500, 0.0900, -0.0800, -0.0100, -0.0100]
)

LANIO3_SOLUTION_RFACTOR = 0.2794


FESE_20UC_PROBLEM = SearchSpace(
    FESE_20UC,
    [
        (SearchKey.ATOMZ, 1, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 2, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 3, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 4, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 5, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 6, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 7, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 8, (-0.1, 0.1)),
        (SearchKey.VIB,   1, (-0.08, 0.1)),
        (SearchKey.VIB,   4, (-0.08, 0.1)),
        (SearchKey.CELLA, -1, (-0.001, 0.001)),
        (SearchKey.CELLC, -1, (-0.010, 0.010)),
    ],
    constraints=[   # Bind cell's a and b axes, and vertical displacement of Se atoms
        EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
        EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 2),
        EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 3),
        EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 5),
        EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 6),
        # (SearchKey.ATOMZ,  5, SearchKey.ATOMZ,  7),
        # (SearchKey.ATOMZ,  6, SearchKey.ATOMZ,  8),
    ]
)

FESE_20UC_SINGLEZ_PROBLEM = SearchSpace(
    FESE_20UC,
    [
        (SearchKey.ATOMZ, 1, (-0.05, 0.05)),
        (SearchKey.ATOMZ, 2, (-0.05, 0.05)),
        (SearchKey.ATOMZ, 5, (-0.05, 0.05)),
        (SearchKey.ATOMZ, 6, (-0.05, 0.05)),
        (SearchKey.VIB,   1, (-0.08, 0.1)),
        (SearchKey.VIB,   4, (-0.08, 0.1)),
        (SearchKey.CELLA, -1, (-0.010, 0.010)),
        (SearchKey.CELLC, -1, (-0.010, 0.010)),
    ],
    constraints=[
        EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
        EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 2),
        EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 3),
        EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 5),
        EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 6),
    ]
)


FESE_20UC_SECOND_SINGLEXY_PROBLEM = SearchSpace(
    FESE_20UC_SINGLEZREGRESSED,
    [
        (SearchKey.ATOMX, 1, (-0.05, 0.05)),
        (SearchKey.ATOMX, 2, (-0.05, 0.05)),
        (SearchKey.ATOMX, 5, (-0.05, 0.05)),
        (SearchKey.ATOMX, 6, (-0.05, 0.05)),
        (SearchKey.ATOMY, 1, (-0.05, 0.05)),
        (SearchKey.ATOMY, 2, (-0.05, 0.05)),
        (SearchKey.ATOMY, 5, (-0.05, 0.05)),
        (SearchKey.ATOMY, 6, (-0.05, 0.05)),
        (SearchKey.CELLA, -1, (-0.010, 0.010)),
        (SearchKey.CELLC, -1, (-0.010, 0.010)),
    ],
    constraints=[
        EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
    ]
)


TEST_FESE_20UC_2D_PROBLEM = SearchSpace(
    TEST_FESE_20UC_FOR2D,
    [
        (SearchKey.ATOMZ, 1, (-0.05, 0.05)),
        (SearchKey.ATOMZ, 2, (-0.05, 0.05)),
    ],
    constraints=[
    ]
)


FESE_20UC_PROBLEM_SECONDXY = SearchSpace(
    FESE_20UC_CLOSE,
    [
        (SearchKey.CELLA, -1, (-0.01, 0.01)),
        (SearchKey.ATOMY, 1, (-0.1, 0.1)),
        (SearchKey.ATOMY, 2, (-0.1, 0.1)),
        (SearchKey.ATOMY, 3, (-0.1, 0.1)),
        (SearchKey.ATOMY, 4, (-0.1, 0.1)),
        (SearchKey.ATOMY, 5, (-0.1, 0.1)),
        (SearchKey.ATOMY, 6, (-0.1, 0.1)),
        (SearchKey.ATOMY, 7, (-0.1, 0.1)),
        (SearchKey.ATOMY, 8, (-0.1, 0.1)),
        (SearchKey.ATOMX, 1, (-0.1, 0.1)),
        (SearchKey.ATOMX, 2, (-0.1, 0.1)),
        (SearchKey.ATOMX, 3, (-0.1, 0.1)),
        (SearchKey.ATOMX, 4, (-0.1, 0.1)),
        (SearchKey.ATOMX, 5, (-0.1, 0.1)),
        (SearchKey.ATOMX, 6, (-0.1, 0.1)),
        (SearchKey.ATOMX, 7, (-0.1, 0.1)),
        (SearchKey.ATOMX, 8, (-0.1, 0.1)),
    ],
    constraints=[   # Bind cell's a and b axes, and vertical displacement of Se atoms
        EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
    ]
)

problems = {
    "LANIO3":    LANIO3_PROBLEM,
    "FESE_20UC": FESE_20UC_PROBLEM,
    "FESE_20UC_SECONDXY": FESE_20UC_PROBLEM_SECONDXY,
    "FESE_20UC_SINGLEZ": FESE_20UC_SINGLEZ_PROBLEM,
    "FESE_20UC_SECOND_SINGLEXY": FESE_20UC_SECOND_SINGLEXY_PROBLEM,
    "TEST_FESE_2D": TEST_FESE_20UC_2D_PROBLEM,
}

beaminfos = {
    "FESE_TRIM": FESE_BEAMINFO_TRIMMED,
}