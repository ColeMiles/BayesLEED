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
    # Layer definitions
    [
     Layer([
        Atom(1, 0.9413, 2.8238, 0.0000),
        Atom(1, 2.8238, 0.9413, 0.0000),
        Atom(2, 0.9413, 2.8238, 5.5180),
        Atom(2, 2.8238, 0.9413, 5.5180),
        Atom(4, 0.9413, 0.9413, 1.3795),
        Atom(4, 2.8238, 2.8238, 4.1385),
        Atom(5, 0.9413, 0.9413, 6.8975),
        Atom(5, 2.8238, 2.8238, 9.6565),
     ],
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3, 0.9413, 2.8238, 0.0000),
        Atom(3, 2.8238, 0.9413, 0.0000),
        Atom(6, 0.9413, 0.9413, 1.3795),
        Atom(6, 2.8238, 2.8238, 4.1385),
     ],
        "Bulk"
     )
    ]
)

LANIO3 = AtomicStructure(
    # Atomic sites
    [
     Site([0.0, 0.0, 1.0, 0.0], 0.14,   ['apO', 'eqO', 'La', 'Ni'], "La top layer"),
     Site([0.0, 0.0, 1.0, 0.0], 0.02,   ['apO', 'eqO', 'La', 'Ni'], "La 2nd layer"),
     Site([0.0, 0.0, 1.0, 0.0], 0.0298,  ['apO', 'eqO', 'La', 'Ni'], "La bulk"),
     Site([0.0, 0.0, 0.0, 1.0], 0.02,   ['apO', 'eqO', 'La', 'Ni'], "Ni top layer"),
     Site([0.0, 0.0, 0.0, 1.0], 0.04,   ['apO', 'eqO', 'La', 'Ni'], "Ni 2nd layer"),
     Site([0.0, 0.0, 0.0, 1.0], 0.0298, ['apO', 'eqO', 'La', 'Ni'], "Ni bulk"),
     Site([1.0, 0.0, 0.0, 0.0], 0.14,   ['apO', 'eqO', 'La', 'Ni'], "apO top layer"),
     Site([1.0, 0.0, 0.0, 0.0], 0.14,   ['apO', 'eqO', 'La', 'Ni'], "apO 2nd layer"),
     Site([1.0, 0.0, 0.0, 0.0], 0.0528, ['apO', 'eqO', 'La', 'Ni'], "apO bulk"),
     Site([0.0, 1.0, 0.0, 0.0], 0.20,   ['apO', 'eqO', 'La', 'Ni'], "eqO top layer"),
     Site([0.0, 1.0, 0.0, 0.0], 0.18,   ['apO', 'eqO', 'La', 'Ni'], "eqO 2nd layer"),
     Site([0.0, 1.0, 0.0, 0.0], 0.0528, ['apO', 'eqO', 'La', 'Ni'], "eqO bulk"),
    ],
    # Layer definitions
    [
     Layer([
        Atom(1,  1.8955, 1.8955, 0.0211),
        Atom(7,  0.0000, 0.0000, 0.1411),
        Atom(4,  0.0000, 0.0000, 1.9610),
        Atom(10, 1.8955, 0.0000, 1.9583),
        Atom(10, 0.0000, 1.8955, 1.9583),
        Atom(2,  1.8955, 1.8955, 3.8549),
        Atom(8,  0.0000, 0.0000, 3.7860),
        Atom(5,  0.0000, 0.0000, 5.7817),
        Atom(11, 1.8955, 0.0000, 5.6159),
        Atom(11, 0.0000, 1.8955, 5.6159),
     ],
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3,  1.8955, 1.8955, 0.0000),
        Atom(9,  0.0000, 0.0000, 0.0000),
        Atom(6,  0.0000, 0.0000, 1.9500),
        Atom(12, 1.8955, 0.0000, 1.9500),
        Atom(12, 0.0000, 1.8955, 1.9500)
     ],
        "Bulk"
     )
    ]
)

LANIO3_PROBLEM = SearchSpace(
    LANIO3,
    [
        (SearchKey.ATOMZ,  1, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  2, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  3, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  4, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  5, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  6, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  7, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  8, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  9, (-0.25, 0.25)),
        (SearchKey.ATOMZ, 10, (-0.25, 0.25)),
    ]
)

LANIO3_SOLUTION = np.array(
    [0.2200, -0.1800, 0.0000, -0.0500, 0.0900, -0.0800, -0.0100, -0.0100]
)

FESE_20UC_PROBLEM = SearchSpace(
    FESE_20UC,
    [
        (SearchKey.ATOMZ,  1, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  2, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  3, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  4, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  5, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  6, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  7, (-0.25, 0.25)),
        (SearchKey.ATOMZ,  8, (-0.25, 0.25)),
    ]
)

problems = {
    "LANIO3":    LANIO3_PROBLEM,
    'FESE_20UC': FESE_20UC_PROBLEM,
}