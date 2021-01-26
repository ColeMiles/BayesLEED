""" Contains definitions for example problems for the current data """
# TODO: If anyone else will ever use this, should probably migrate to real configuration files
from .structure import *
from .searchspace import *
from .tleed import *
import numpy as np

# TODO: All (commented out) structures need explicit interlayer vectors now!

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
        ], [0.0, 0.0, 5.5180 / 2],
        LayerType.SURF,
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3, 0.25, 0.75, 0.25),  # Bulk Fe
        Atom(3, 0.75, 0.25, 0.25),  # Bulk Fe
        Atom(6, 0.25, 0.25, 0.00),  # Bulk Se
        Atom(6, 0.75, 0.75, 0.50),  # Bulk Se
        ], [0.0, 0.0, 5.5180 / 2],
        LayerType.BULK,
        "Bulk"
     )
    ],
    # Unit cell parameters
    [3.7676, 3.7676, 5.5180]
)

STO_1x1 = AtomicStructure(
    # Atomic sites
    [
        Site([1.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "Sr bulk"),
        Site([0.0, 1.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "Ti bulk"),
        Site([0.0, 0.0, 1.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "apO bulk"),
        Site([0.0, 0.0, 0.0, 1.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "eqO bulk"),
    ],
    # Layer definitions (fractional coordinates)
    [
        Layer([
            Atom(2, 0.50, 0.50, 0.00),  # Ti, doubled overlayer, cell 1
            Atom(4, 0.50, 0.00, 0.00),  # O, doubled overlayer, cell 1
            Atom(4, 0.00, 0.50, 0.00),  # O, doubled overlayer, cell 1
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "TiO2 doubled overlayer",
        ),
        Layer([
            Atom(2, 0.50, 0.50, 0.00),  # Ti surface 1, cell 1
            Atom(4, 0.50, 0.00, 0.00),  # O surface 1, cell 1
            Atom(4, 0.00, 0.50, 0.00),  # O surface 1, cell 1
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "SrTiO3 surface 1 -- TiO2",
        ),
        Layer([
            Atom(1, 0.00, 0.00, 0.00),  # Sr surface 1, cell 1
            Atom(3, 0.50, 0.50, 0.00),  # O surface 1, cell 1
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "SrTiO3 surface 1 -- SrO",
        ),
        Layer([
            Atom(2, 0.50, 0.50, 0.00),  # Ti, bulk, cell 1
            Atom(4, 0.50, 0.00, 0.00),  # O, bulk, cell 1
            Atom(4, 0.00, 0.50, 0.00),  # O, bulk, cell 1
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.BULK,
            "SrTiO3 bulk -- TiO2",
        ),
        Layer([
            Atom(1, 0.00, 0.00, 0.00),  # Sr, bulk, cell 1
            Atom(3, 0.50, 0.50, 0.00),  # O, bulk, cell 1
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.BULK,
            "SrTiO3 bulk -- SrO",
        ),
    ],
    # Unit cell parameters
    [3.905, 3.905, 3.905]
)


STO_2x1 = AtomicStructure(
    # Atomic sites
    [
        Site([1.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "Sr bulk"),
        Site([0.0, 1.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "Ti bulk"),
        Site([0.0, 0.0, 1.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "apO bulk"),
        Site([0.0, 0.0, 0.0, 1.0], 0.1, ["Sr", "Ti", "apO", "eqO"],
             "eqO bulk"),
    ],
    # Layer definitions (fractional coordinates)
    [
        Layer([
            Atom(2, 0.25, 0.50, 0.00),  # Ti, doubled overlayer, cell 1
            Atom(4, 0.25, 0.00, 0.00),  # O, doubled overlayer, cell 1
            Atom(4, 0.00, 0.50, 0.00),  # O, doubled overlayer, cell 1
            Atom(2, 0.75, 0.50, 0.00),  # Ti, doubled overlayer, cell 2
            Atom(4, 0.75, 0.00, 0.00),  # O, doubled overlayer, cell 2
            Atom(4, 0.50, 0.50, 0.00),  # O, doubled overlayer, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "TiO2 doubled overlayer",
        ),
        Layer([
            Atom(2, 0.25, 0.50, 0.00),  # Ti, bulk, cell 1
            Atom(4, 0.25, 0.00, 0.00),  # O, bulk, cell 1
            Atom(4, 0.00, 0.50, 0.00),  # O, bulk, cell 1
            Atom(2, 0.75, 0.50, 0.00),  # Ti, bulk, cell 2
            Atom(4, 0.75, 0.00, 0.00),  # O, bulk, cell 2
            Atom(4, 0.50, 0.50, 0.00),  # O, bulk, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "SrTiO3 surface 1 -- TiO2",
        ),
        Layer([
            Atom(1, 0.00, 0.00, 0.00),  # Sr, bulk, cell 1
            Atom(3, 0.25, 0.50, 0.00),  # O, bulk, cell 1
            Atom(1, 0.50, 0.00, 0.00),  # Sr, bulk, cell 2
            Atom(3, 0.75, 0.50, 0.00),  # O, bulk, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "SrTiO3 surface 1 -- SrO",
        ),
        Layer([
            Atom(2, 0.25, 0.50, 0.00),  # Ti, bulk, cell 1
            Atom(4, 0.25, 0.00, 0.00),  # O, bulk, cell 1
            Atom(4, 0.00, 0.50, 0.00),  # O, bulk, cell 1
            Atom(2, 0.75, 0.50, 0.00),  # Ti, bulk, cell 2
            Atom(4, 0.75, 0.00, 0.00),  # O, bulk, cell 2
            Atom(4, 0.50, 0.50, 0.00),  # O, bulk, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.BULK,
            "SrTiO3 bulk -- TiO2",
        ),
        Layer([
            Atom(1, 0.00, 0.00, 0.00),  # Sr, bulk, cell 1
            Atom(3, 0.25, 0.50, 0.00),  # O, bulk, cell 1
            Atom(1, 0.50, 0.00, 0.00),  # Sr, bulk, cell 2
            Atom(3, 0.75, 0.50, 0.00),  # O, bulk, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.BULK,
            "SrTiO3 bulk -- SrO",
        ),
    ],
    # Unit cell parameters
    [7.810, 3.905, 3.905]
)

# TODO: Should I add one more tunable layer of TiO2?
FESE_1UC_2x1 = AtomicStructure(
    # Atomic sites
    [
     Site([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"], "Sr bulk"),
     Site([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"], "Ti bulk"),
     Site([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"], "apO bulk"),
     Site([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"], "eqO bulk"),
     Site([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"], "Fe film"),
     Site([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"], "Se film"),
    ],
    # Layer definitions (fractional coordinates)
    [
     Layer([
         Atom(5, 0.25, 0.50, 0.25),   # Top Layer Fe, cell 1
         Atom(5, 0.50, 0.00, 0.25),   # Top Layer Fe, cell 1
         Atom(6, 0.25, 0.00, 0.00),   # Top Layer Se, cell 1
         Atom(6, 0.50, 0.50, 0.50),   # Top Layer Se, cell 1
         Atom(5, 0.75, 0.50, 0.25),   # Top Layer Fe, cell 2
         Atom(5, 1.00, 0.00, 0.25),   # Top Layer Fe, cell 2
         Atom(6, 0.75, 0.00, 0.00),   # Top Layer Se, cell 2
         Atom(6, 1.00, 0.50, 0.50),   # Top Layer Se, cell 2
         ], [0.0, 0.0, 5.5180 / 2],
         LayerType.SURF,
         "Top FeSe film",
     ),
     Layer([
         Atom(2, 0.75, 0.00, 0.00),   # Ti, doubled overlayer, cell 1
         Atom(2, 0.50, 0.50, 0.00),   # Ti, doubled overlayer, cell 2
         Atom(4, 0.50, 0.00, 0.00),   # O, doubled overlayer, cell 1
         Atom(4, 0.75, 0.50, 0.00),   # O, doubled overlayer, cell 1
         Atom(4, 0.25, 0.50, 0.00),   # O, doubled overlayer, cell 2
         Atom(4, 0.00, 0.00, 0.00),   # O, doubled overlayer, cell 2
         ], [0.0, 0.0, 3.905 / 2],
         LayerType.SURF,
         "TiO2 doubled overlayer",
     ),
     # TODO: For efficiency, split the bulk into two layers and alternate?
     Layer([
         Atom(2, 0.25, 0.50, 0.50),   # Ti, bulk, cell 1
         Atom(2, 0.75, 0.50, 0.50),   # Ti, bulk, cell 2
         Atom(4, 0.00, 0.50, 0.50),   # O, bulk, cell 1
         Atom(4, 0.25, 0.00, 0.50),   # O, bulk, cell 1
         Atom(4, 0.50, 0.00, 0.50),   # O, bulk, cell 2
         Atom(4, 0.75, 0.00, 0.50),   # O, bulk, cell 2
         Atom(1, 0.00, 0.00, 0.00),   # Sr, bulk, cell 1
         Atom(1, 0.50, 0.00, 0.00),   # Sr, bulk, cell 2
         Atom(3, 0.25, 0.50, 0.00),   # O, bulk, cell 1
         Atom(3, 0.75, 0.50, 0.00),   # O, bulk, cell 2
         ], [0.0, 0.0, 3.905 / 2],
         LayerType.BULK,
         "SrTiO3 bulk",
     ),
    ],
    # Unit cell parameters
    [7.810, 3.905, 3.905]
)

TEST_DOUB_FESE_1UC_2x1 = AtomicStructure(
    # Atomic sites
    [
        Site([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"],
             "Sr bulk"),
        Site([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"],
             "Ti bulk"),
        Site([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"],
             "apO bulk"),
        Site([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"],
             "eqO bulk"),
        Site([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"],
             "Fe film"),
        Site([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.1, ["Sr", "Ti", "apO", "eqO", "Fe", "Se"],
             "Se film"),
    ],
    # Layer definitions (fractional coordinates)
    [
        Layer([
            Atom(5, 0.25, 0.50, 0.25),   # Top Layer Fe, cell 1
            Atom(5, 0.50, 0.00, 0.25),   # Top Layer Fe, cell 1
            Atom(6, 0.25, 0.00, 0.00),   # Top Layer Se, cell 1
            Atom(6, 0.50, 0.50, 0.50),   # Top Layer Se, cell 1
            Atom(5, 0.75, 0.50, 0.25),   # Top Layer Fe, cell 2
            Atom(5, 1.00, 0.00, 0.25),   # Top Layer Fe, cell 2
            Atom(6, 0.75, 0.00, 0.00),   # Top Layer Se, cell 2
            Atom(6, 1.00, 0.50, 0.50),   # Top Layer Se, cell 2
        ], [0.0, 0.0, 5.5180 / 2],
            LayerType.SURF,
            "Top FeSe film",
        ),
        Layer([
            Atom(2, 0.75, 0.00, 0.00),   # Ti, doubled overlayer, cell 1
            Atom(2, 0.50, 0.50, 0.00),   # Ti, doubled overlayer, cell 2
            Atom(4, 0.50, 0.00, 0.00),   # O, doubled overlayer, cell 1
            Atom(4, 0.75, 0.50, 0.00),   # O, doubled overlayer, cell 1
            Atom(4, 0.25, 0.50, 0.00),   # O, doubled overlayer, cell 2
            Atom(4, 0.00, 0.00, 0.00),   # O, doubled overlayer, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.SURF,
            "TiO2 doubled overlayer",
        ),
        Layer([
            Atom(2, 0.25, 0.50, 0.00),  # Ti, bulk, cell 1
            Atom(2, 0.75, 0.50, 0.00),  # Ti, bulk, cell 2
            Atom(4, 0.00, 0.50, 0.00),  # O, bulk, cell 1
            Atom(4, 0.25, 0.00, 0.00),  # O, bulk, cell 1
            Atom(4, 0.50, 0.00, 0.00),  # O, bulk, cell 2
            Atom(4, 0.75, 0.00, 0.00),  # O, bulk, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.BULK,
            "SrTiO3 bulk -- TiO2",
        ),
        Layer([
            Atom(1, 0.00, 0.00, 0.00),   # Sr, bulk, cell 1
            Atom(1, 0.50, 0.00, 0.00),   # Sr, bulk, cell 2
            Atom(3, 0.25, 0.50, 0.00),   # O, bulk, cell 1
            Atom(3, 0.75, 0.50, 0.00),   # O, bulk, cell 2
        ], [0.0, 0.0, 3.905 / 2],
            LayerType.BULK,
            "SrTiO3 bulk -- SrO",
        ),
    ],
    # Unit cell parameters
    [7.810, 3.905, 3.905]
)

# FESE_20UC_SINGLEZREGRESSED = AtomicStructure(
#     # Atomic sites
#     [
#         Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe top layer"),
#         Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe 2nd layer"),
#         Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe bulk"),
#         Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se top layer"),
#         Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se 2nd layer"),
#         Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se bulk")
#     ],
#     # Layer definitions (fractional coordinates)
#     [
#         Layer([
#             Atom(1, 0.25, 0.75, 0.26255),  # Top Layer Fe
#             Atom(1, 0.75, 0.25, 0.26387),  # Top Layer Fe
#             Atom(2, 0.25, 0.75, 1.2500),  # 2nd Layer Fe
#             Atom(2, 0.75, 0.25, 1.2500),  # 2nd Layer Fe
#             Atom(4, 0.25, 0.25, 0.00655),  # Top Layer Se
#             Atom(4, 0.75, 0.75, 0.5322),  # Top Layer Se
#             Atom(5, 0.25, 0.25, 1.0000),  # 2nd Layer Se
#             Atom(5, 0.75, 0.75, 1.5000),  # 2nd Layer Se
#         ],
#             "Top 2 unit cells"
#         ),
#         Layer([
#             Atom(3, 0.25, 0.75, 0.2500),  # Bulk Fe
#             Atom(3, 0.75, 0.25, 0.2500),  # Bulk Fe
#             Atom(6, 0.25, 0.25, 0.0000),  # Bulk Se
#             Atom(6, 0.75, 0.75, 0.5000),  # Bulk Se
#         ],
#             "Bulk"
#         )
#     ],
#     # Unit cell parameters
#     [3.7659, 3.7659, 5.51547]
# )
#
#
# TEST_FESE_20UC_FOR2D = AtomicStructure(
#     # Atomic sites
#     [
#         Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe top layer"),
#         Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe 2nd layer"),
#         Site([1.0, 0.0], 0.1000, ["Fe", "Se"], "Fe bulk"),
#         Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se top layer"),
#         Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se 2nd layer"),
#         Site([0.0, 1.0], 0.1000, ["Fe", "Se"], "Se bulk")
#     ],
#     # Layer definitions (fractional coordinates)
#     [
#         Layer([
#             Atom(1, 0.25, 0.75, 0.25000),  # Top Layer Fe  <-------- Search over these two
#             Atom(1, 0.75, 0.25, 0.25000),  # Top Layer Fe  <----|
#             Atom(2, 0.25, 0.75, 1.2500),  # 2nd Layer Fe
#             Atom(2, 0.75, 0.25, 1.2500),  # 2nd Layer Fe
#             Atom(4, 0.25, 0.25, 0.00655),  # Top Layer Se
#             Atom(4, 0.75, 0.75, 0.5322),  # Top Layer Se
#             Atom(5, 0.25, 0.25, 1.0000),  # 2nd Layer Se
#             Atom(5, 0.75, 0.75, 1.5000),  # 2nd Layer Se
#         ],
#             "Top 2 unit cells"
#         ),
#         Layer([
#             Atom(3, 0.25, 0.75, 0.2500),  # Bulk Fe
#             Atom(3, 0.75, 0.25, 0.2500),  # Bulk Fe
#             Atom(6, 0.25, 0.25, 0.0000),  # Bulk Se
#             Atom(6, 0.75, 0.75, 0.5000),  # Bulk Se
#         ],
#             "Bulk"
#         )
#     ],
#     # Unit cell parameters
#     [3.7659, 3.7659, 5.51547]
# )
#
# FESE_20UC_NEW_CELL_PARAMS = FESE_20UC
# FESE_20UC_NEW_CELL_PARAMS.cell_params = np.array([3.7667322, 3.7667322, 5.513444])
#
# FESE_20UC_CLOSE = AtomicStructure(
#     # Atomic sites
#     [
#         Site([1.0, 0.0], 0.1319, ["Fe", "Se"], "Fe top layer"),
#         Site([1.0, 0.0], 0.1319, ["Fe", "Se"], "Fe 2nd layer"),
#         Site([1.0, 0.0], 0.1319, ["Fe", "Se"], "Fe bulk"),
#         Site([0.0, 1.0], 0.1204, ["Fe", "Se"], "Se top layer"),
#         Site([0.0, 1.0], 0.1204, ["Fe", "Se"], "Se 2nd layer"),
#         Site([0.0, 1.0], 0.1204, ["Fe", "Se"], "Se bulk")
#     ],
#     # Layer definitions (fractional coordinates)
#     [
#         Layer([
#             Atom(1, 0.25, 0.75, 0.26126),  # Top Layer Fe
#             Atom(1, 0.75, 0.25, 0.27046),  # Top Layer Fe
#             Atom(2, 0.25, 0.75, 1.24418),  # 2nd Layer Fe
#             Atom(2, 0.75, 0.25, 1.27062),  # 2nd Layer Fe
#             Atom(4, 0.25, 0.25, 0.00732),  # Top Layer Se
#             Atom(4, 0.75, 0.75, 0.53256),  # Top Layer Se
#             Atom(5, 0.25, 0.25, 0.99870),  # 2nd Layer Se
#             Atom(5, 0.75, 0.75, 1.51332),  # 2nd Layer Se
#         ],
#             "Top 2 unit cells"
#         ),
#         Layer([
#             Atom(3, 0.25, 0.75, 0.25),  # Bulk Fe
#             Atom(3, 0.75, 0.25, 0.25),  # Bulk Fe
#             Atom(6, 0.25, 0.25, 0.00),  # Bulk Se
#             Atom(6, 0.75, 0.75, 0.50),  # Bulk Se
#         ],
#             "Bulk"
#         )
#     ],
#     # Unit cell parameters
#     [3.7667322, 3.7667322, 5.513444]
# )


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
     ], [0.0, 0.0, 3.9000 / 2],
        LayerType.SURF,
        "Top 2 unit cells"
     ),
     Layer([
        Atom(3,  0.5, 0.5, 0.0),
        Atom(9,  0.0, 0.0, 0.0),
        Atom(6,  0.0, 0.0, 0.5),
        Atom(12, 0.5, 0.0, 0.5),
        Atom(12, 0.0, 0.5, 0.5)
     ], [0.0, 0.0, 3.9000 / 2],
        LayerType.BULK,
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

# FESE_20UC_DELTA_PROBLEM = SearchSpace(
#     FESE_20UC_NEW_CELL_PARAMS,
#     [
#         (SearchKey.ATOMZ, 1, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 2, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 3, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 4, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 5, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 6, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 7, (-0.1, 0.1)),
#         (SearchKey.ATOMZ, 8, (-0.1, 0.1)),
#         (SearchKey.VIB, 1, (-0.08, 0.1)),
#         (SearchKey.VIB, 4, (-0.08, 0.1)),
#     ],
#     constraints=[  # Bind cell's a and b axes, and vertical displacement of Se atoms
#         EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 2),
#         EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 3),
#         EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 5),
#         EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 6),
#     ]
# )
#
# FESE_20UC_SINGLEZ_PROBLEM = SearchSpace(
#     FESE_20UC,
#     [
#         (SearchKey.ATOMZ, 1, (-0.05, 0.05)),
#         (SearchKey.ATOMZ, 2, (-0.05, 0.05)),
#         (SearchKey.ATOMZ, 5, (-0.05, 0.05)),
#         (SearchKey.ATOMZ, 6, (-0.05, 0.05)),
#         (SearchKey.VIB,   1, (-0.08, 0.1)),
#         (SearchKey.VIB,   4, (-0.08, 0.1)),
#         (SearchKey.CELLA, -1, (-0.010, 0.010)),
#         (SearchKey.CELLC, -1, (-0.010, 0.010)),
#     ],
#     constraints=[
#         EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
#         EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 2),
#         EqualityConstraint(SearchKey.VIB, 1, SearchKey.VIB, 3),
#         EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 5),
#         EqualityConstraint(SearchKey.VIB, 4, SearchKey.VIB, 6),
#     ]
# )
#
#
# FESE_20UC_SECOND_SINGLEXY_PROBLEM = SearchSpace(
#     FESE_20UC_SINGLEZREGRESSED,
#     [
#         (SearchKey.ATOMX, 1, (-0.05, 0.05)),
#         (SearchKey.ATOMX, 2, (-0.05, 0.05)),
#         (SearchKey.ATOMX, 5, (-0.05, 0.05)),
#         (SearchKey.ATOMX, 6, (-0.05, 0.05)),
#         (SearchKey.ATOMY, 1, (-0.05, 0.05)),
#         (SearchKey.ATOMY, 2, (-0.05, 0.05)),
#         (SearchKey.ATOMY, 5, (-0.05, 0.05)),
#         (SearchKey.ATOMY, 6, (-0.05, 0.05)),
#         (SearchKey.CELLA, -1, (-0.010, 0.010)),
#         (SearchKey.CELLC, -1, (-0.010, 0.010)),
#     ],
#     constraints=[
#         EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
#     ]
# )
#
#
# TEST_FESE_20UC_2D_PROBLEM = SearchSpace(
#     TEST_FESE_20UC_FOR2D,
#     [
#         (SearchKey.ATOMZ, 1, (-0.05, 0.05)),
#         (SearchKey.ATOMZ, 2, (-0.05, 0.05)),
#     ],
#     constraints=[
#     ]
# )
#
#
# FESE_20UC_PROBLEM_SECONDXY = SearchSpace(
#     FESE_20UC_CLOSE,
#     [
#         (SearchKey.CELLA, -1, (-0.01, 0.01)),
#         (SearchKey.ATOMY, 1, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 2, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 3, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 4, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 5, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 6, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 7, (-0.1, 0.1)),
#         (SearchKey.ATOMY, 8, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 1, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 2, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 3, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 4, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 5, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 6, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 7, (-0.1, 0.1)),
#         (SearchKey.ATOMX, 8, (-0.1, 0.1)),
#     ],
#     constraints=[   # Bind cell's a and b axes, and vertical displacement of Se atoms
#         EqualityConstraint(SearchKey.CELLA, -1, SearchKey.CELLB, -1),
#     ]
# )

FESE_1UC_PROBLEM = SearchSpace(
    FESE_1UC_2x1,
    [
        (SearchKey.CELLB, -1, (-0.01, 0.01)),
        (SearchKey.CELLC, -1, (-0.01, 0.01)),
        (SearchKey.ATOMZ,  1, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  2, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  3, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  4, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  5, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  6, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  7, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  8, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  9, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 10, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 11, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 12, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 13, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 14, (-0.1, 0.1)),
    ],
    constraints=[   # Bind cell's a axis to be twice the b axis
        LambdaConstraint(SearchKey.CELLB, -1, SearchKey.CELLA, -1, lambda x: 2*x)
    ]
)

TEST_DOUB_FESE_1UC_PROBLEM = SearchSpace(
    TEST_DOUB_FESE_1UC_2x1,
    [
        (SearchKey.CELLB, -1, (-0.01, 0.01)),
        (SearchKey.CELLC, -1, (-0.01, 0.01)),
        (SearchKey.INTZ, 1, (0.0, 0.3)),
        (SearchKey.INTZ, 2, (-0.1, 0.2)),
        (SearchKey.ATOMZ,  1, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  2, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  3, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  4, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  5, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  6, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  7, (-0.1, 0.1)),
        (SearchKey.ATOMZ,  9, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 10, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 11, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 12, (-0.1, 0.1)),
        (SearchKey.ATOMZ, 13, (-0.1, 0.1)),
        (SearchKey.ATOMX, 1, (0.0, 0.5)),
        (SearchKey.ATOMY, 1, (0.0, 0.5)),
    ],
    constraints=[   # Bind cell's a axis to be twice the b axis
        LambdaConstraint(SearchKey.CELLB, -1, SearchKey.CELLA, -1, lambda x: 2*x),
        # Allow an overal registry shift
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 2),
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 3),
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 4),
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 5),
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 6),
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 7),
        EqualShiftConstraint(SearchKey.ATOMX, 1, SearchKey.ATOMX, 8),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 2),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 3),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 4),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 5),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 6),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 7),
        EqualShiftConstraint(SearchKey.ATOMY, 1, SearchKey.ATOMY, 8),
    ]
)

STO_1x1_PROBLEM = SearchSpace(
    STO_1x1,
    [
        # Relax the lattice constants
        (SearchKey.CELLB, -1, (-0.02, 0.02)),
        (SearchKey.CELLC, -1, (-0.02, 0.02)),
        # Surface atomic coordinates
        (SearchKey.INTZ, 1, (0.0, 0.3)),
        (SearchKey.INTZ, 2, (0.0, 0.3)),
        (SearchKey.INTZ, 3, (0.0, 0.3)),
        (SearchKey.ATOMZ, 1, (0.0, 0.3)),
        (SearchKey.ATOMZ, 2, (0.0, 0.3)),
        (SearchKey.ATOMZ, 4, (0.0, 0.3)),
        (SearchKey.ATOMZ, 5, (0.0, 0.3)),
        (SearchKey.ATOMZ, 7, (0.0, 0.3)),
        # Let the doubled TiO2 layer have some nonzero xy motion
        (SearchKey.ATOMX, 1, (-0.3, 0.3)),
        (SearchKey.ATOMX, 2, (-0.3, 0.3)),
        (SearchKey.ATOMX, 3, (-0.3, 0.3)),
        (SearchKey.ATOMY, 1, (-0.3, 0.3)),
        (SearchKey.ATOMY, 2, (-0.3, 0.3)),
        (SearchKey.ATOMY, 3, (-0.3, 0.3)),
        # Vibrational params
        (SearchKey.VIB, 1, (-0.08, 0.1)),
        (SearchKey.VIB, 2, (-0.08, 0.1)),
        (SearchKey.VIB, 3, (-0.08, 0.1)),
        (SearchKey.VIB, 4, (-0.08, 0.1)),
    ],
    constraints=[
        EqualityConstraint(SearchKey.CELLB, -1, SearchKey.CELLA, -1),
    ]
)


STO_2x1_PROBLEM = SearchSpace(
    STO_2x1,
    [
        # Relax the lattice constants
        (SearchKey.CELLB, -1, (-0.02, 0.02)),
        (SearchKey.CELLC, -1, (-0.02, 0.02)),
        # Surface atomic coordinates
        (SearchKey.INTZ, 1, (0.0, 0.3)),
        (SearchKey.INTZ, 2, (0.0, 0.3)),
        (SearchKey.INTZ, 3, (0.0, 0.3)),
        (SearchKey.ATOMZ, 1, (0.0, 0.3)),
        (SearchKey.ATOMZ, 2, (0.0, 0.3)),
        (SearchKey.ATOMZ, 3, (0.0, 0.3)),
        (SearchKey.ATOMZ, 4, (0.0, 0.3)),
        (SearchKey.ATOMZ, 5, (0.0, 0.3)),
        (SearchKey.ATOMZ, 7, (0.0, 0.3)),
        (SearchKey.ATOMZ, 8, (0.0, 0.3)),
        (SearchKey.ATOMZ, 9, (0.0, 0.3)),
        (SearchKey.ATOMZ, 10, (0.0, 0.3)),
        (SearchKey.ATOMZ, 11, (0.0, 0.3)),
        (SearchKey.ATOMZ, 13, (0.0, 0.3)),
        (SearchKey.ATOMZ, 13, (0.0, 0.3)),
        (SearchKey.ATOMZ, 14, (0.0, 0.3)),
        # Let the doubled TiO2 layer have some nonzero xy motion
        (SearchKey.ATOMX, 1, (-0.3, 0.3)),
        (SearchKey.ATOMX, 2, (-0.3, 0.3)),
        (SearchKey.ATOMX, 3, (-0.3, 0.3)),
        (SearchKey.ATOMX, 4, (-0.3, 0.3)),
        (SearchKey.ATOMX, 5, (-0.3, 0.3)),
        (SearchKey.ATOMX, 6, (-0.3, 0.3)),
        (SearchKey.ATOMY, 1, (-0.3, 0.3)),
        (SearchKey.ATOMY, 2, (-0.3, 0.3)),
        (SearchKey.ATOMY, 3, (-0.3, 0.3)),
        (SearchKey.ATOMX, 4, (-0.3, 0.3)),
        (SearchKey.ATOMX, 5, (-0.3, 0.3)),
        (SearchKey.ATOMX, 6, (-0.3, 0.3)),
        # Vibrational params
        (SearchKey.VIB, 1, (-0.08, 0.1)),
        (SearchKey.VIB, 2, (-0.08, 0.1)),
        (SearchKey.VIB, 3, (-0.08, 0.1)),
        (SearchKey.VIB, 4, (-0.08, 0.1)),
    ],
    constraints=[
        EqualityConstraint(SearchKey.CELLB, -1, SearchKey.CELLA, -1),
    ]
)

DEFAULT_DELTA_DISPS = np.zeros((21, 3))
DEFAULT_DELTA_DISPS[:, 2] = np.arange(-0.2, 0.22, step=0.02)
DEFAULT_DELTA_VIBS = np.arange(0.1, 0.3, step=0.04)

FESE_DELTA_SEARCHDIMS = [
    (1, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (2, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (3, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (4, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (5, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (6, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (7, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (8, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
]

FESE_SRTIO_DELTA_SEARCHDIMS = [
    (1, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (2, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (3, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (4, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (5, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (6, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (7, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (8, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (9, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (10, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (11, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
    (12, DEFAULT_DELTA_DISPS, DEFAULT_DELTA_VIBS),
]

problems = {
    "LANIO3":    LANIO3_PROBLEM,
    "FESE_20UC": FESE_20UC_PROBLEM,
    # "FESE_20UC_SECONDXY": FESE_20UC_PROBLEM_SECONDXY,
    # "FESE_20UC_SINGLEZ": FESE_20UC_SINGLEZ_PROBLEM,
    # "FESE_20UC_DELTA": FESE_20UC_DELTA_PROBLEM,
    # "FESE_20UC_SECOND_SINGLEXY": FESE_20UC_SECOND_SINGLEXY_PROBLEM,
    # "TEST_FESE_2D": TEST_FESE_20UC_2D_PROBLEM,
    "FESE_1UC": FESE_1UC_PROBLEM,
    "TEST_FESE_1UC": TEST_DOUB_FESE_1UC_PROBLEM,
    "STO_1x1": STO_1x1_PROBLEM,
    "STO_2x1": STO_2x1_PROBLEM,
}
