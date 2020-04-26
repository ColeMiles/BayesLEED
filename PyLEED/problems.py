from tleed import AtomicStructure, Site, Layer, Atom, SearchKey, SearchSpace
import numpy as np

FESE_20UC = AtomicStructure(
    # Atomic sites
    [
     Site([1.0, 0.0], 0.0528, ["Fe", "Se"], "Fe top layer"),
     Site([1.0, 0.0], 0.0528, ["Fe", "Se"], "Fe 2nd layer"),
     Site([1.0, 0.0], 0.0528, ["Fe", "Se"], "Fe bulk"),
     Site([0.0, 1.0], 0.0298, ["Fe", "Se"], "Se top layer"),
     Site([0.0, 1.0], 0.0298, ["Fe", "Se"], "Se 2nd layer"),
     Site([0.0, 1.0], 0.0298, ["Fe", "Se"], "Se bulk")
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
        Atom(4,  0.0, 0.0, 1.0),
        Atom(10, 0.5, 0.0, 1.0),
        Atom(10, 0.0, 0.5, 1.0),
        Atom(2,  0.5, 0.5, 2.0),
        Atom(8,  0.5, 0.5, 2.0),
        Atom(5,  0.0, 0.0, 3.0),
        Atom(11, 0.5, 0.0, 3.0),
        Atom(11, 0.0, 0.5, 3.0),
     ],
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3,  0.5, 0.5, 0.0),
        Atom(9,  0.0, 0.0, 0.0),
        Atom(6,  0.0, 0.0, 1.0),
        Atom(12, 0.5, 0.0, 1.0),
        Atom(12, 0.0, 0.5, 1.0)
     ],
        "Bulk"
     )
    ],
    # Unit cell parameters
    [3.7910, 3.7910, 1.9500]
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
        (SearchKey.ATOMZ, 4, 5),
        (SearchKey.ATOMZ, 9, 10),
    ]
)

LANIO3_SOLUTION = np.array(
    [0.2200, -0.1800, 0.0000, -0.0500, 0.0900, -0.0800, -0.0100, -0.0100]
)

LANIO3_SOLUTION_RFACTOR = 0.2794


FESE_20UC_PROBLEM = SearchSpace(
    FESE_20UC,
    [
        (SearchKey.ATOMZ, 1, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 2, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 3, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 4, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 5, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 6, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 7, (-0.066, 0.066)),
        (SearchKey.ATOMZ, 8, (-0.066, 0.066)),
        # (SearchKey.ATOMY, 1, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 2, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 3, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 4, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 5, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 6, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 7, (-0.25, 0.25)),
        # (SearchKey.ATOMY, 8, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 1, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 2, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 3, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 4, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 5, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 6, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 7, (-0.25, 0.25)),
        # (SearchKey.ATOMX, 8, (-0.25, 0.25)),
    ]
)

problems = {
    "LANIO3":    LANIO3_PROBLEM,
    "FESE_20UC": FESE_20UC_PROBLEM,
}