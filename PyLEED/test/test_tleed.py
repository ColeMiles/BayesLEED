import os
import shutil
import pytest

import numpy as np

from pyleed import problems, tleed
from pyleed.structure import AtomicStructure, Site, Layer, Atom
from pyleed.searchspace import SearchKey, SearchSpace, EqualityConstraint, EqualShiftConstraint, DeltaSearchSpace
import pyleed.bayessearch as bayessearch


def _isclose(a, b, eps=1e-6):
    return abs(a - b) < eps


def _make_test_manager():
    # TODO: Make this so that you don't have to call from a specific directory
    return bayessearch.create_manager(
        'test/test_files/FeSetest/',
        '/home/cole/ProgScratch/BayesLEED/TLEED/',
        problems.FESE_BEAMINFO_TRIMMED,
        'test/test_files/FeSetest/NBLIST.FeSe-1x1',
        'test/test_files/FeSetest/FeSeBulk.eight.phase', 8,
        executable='ref-calc.FeSe'
    )

TEST_STRUCT = AtomicStructure(
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


def test_to_script():
    struct = TEST_STRUCT

    with open("test/test_files/fese_answer.txt", "r") as f:
        answer = f.read()

    assert struct.to_script() == answer


def test_write_script():
    struct = TEST_STRUCT

    basedir = "test/test_files/FeSetest/"
    exe = os.path.join(basedir, "ref-calc.FeSe")
    rfact = os.path.join(basedir, "rf.x")
    template = os.path.join(basedir, "FIN")

    with open(template, "r") as f:
        answer = f.read()

    refcalc = tleed.RefCalc(struct, exe, answer, basedir)
    refcalc._write_script("test/test_files/FeSetest/comp_FIN")

    with open("test/test_files/FeSetest/comp_FIN", "r") as f:
        output = f.read()

    assert output == answer

    os.remove("test/test_files/FeSetest/comp_FIN")


def test_to_structures():
    struct = problems.FESE_20UC

    search_space = SearchSpace(
        struct,
        [
            (SearchKey.VIB,   3, (-0.025, 0.025)),
            (SearchKey.ATOMX, 6, (-0.2, 0.2)),
            (SearchKey.ATOMY, 2, (-0.2, 0.2)),
            (SearchKey.ATOMZ, 4, (-0.4, 0.4))
        ]
    )

    norm_vec = np.array([0.25, 0.0, 0.5, 1.0])

    new_struct = search_space.to_structures(norm_vec)
    expected_vib = struct.sites[2].vib - 0.025 + 0.25 * 0.05
    expected_atomx = struct.layers[0].xs[5] - 0.2
    expected_atomy = struct.layers[0].ys[1] - 0.2 + 0.5 * 0.4
    expected_atomz = struct.layers[0].zs[3] - 0.4 + 0.8

    assert _isclose(new_struct.sites[2].vib, expected_vib)
    assert _isclose(new_struct.layers[0].xs[5], expected_atomx)
    assert _isclose(new_struct.layers[0].ys[1], expected_atomy)
    assert _isclose(new_struct.layers[0].zs[3], expected_atomz)


def test_compile_delta_program():
    manager = _make_test_manager()
    ref_calc_dir = os.path.join(manager.workdir, "ref-calc-results")
    disps = [
        np.array([z, 0.0, 0.0]) for z in np.arange(-0.05, 0.06, 0.01)
    ]
    search_dim = (3, disps, [0.0])
    subworkdir = os.path.join(ref_calc_dir, "delta-work-dir")
    try:
        os.mkdir(subworkdir)
    except FileExistsError:
        pass
    exepath = os.path.join(subworkdir, "delta.x")
    manager._compile_delta_program(exepath, search_dim)

    assert os.path.isfile(exepath)
    assert os.access(exepath, os.X_OK)


def test_write_delta_script():
    manager = _make_test_manager()
    ref_calc_dir = os.path.join(manager.workdir, "ref-calc-results")
    ref_calc = tleed.RefCalc(problems.FESE_20UC, manager.phaseshifts,
                             manager.beaminfo, manager.beamlist,
                             manager.leed_exe, ref_calc_dir, produce_tensors=True)
    # Manually assert that the ref calc has been done
    ref_calc.completed = True
    ref_calc.tensorfiles = [
        os.path.join(ref_calc_dir, "LAY1{}".format(i+1)) for i in range(len(ref_calc.struct.layers[0]))
    ]
    subworkdir = os.path.join(ref_calc_dir, "delta-work-dir")
    try:
        os.mkdir(subworkdir)
    except FileExistsError:
        pass
    scriptname = os.path.join(subworkdir, "delta.in")

    disps = [
        np.array([z, 0.0, 0.0]) for z in np.arange(-0.2, 0.2, 0.02)
    ]
    vibs = np.arange(0.1, 0.18, 0.02)
    search_dim = (3, disps, vibs)
    manager._write_delta_script(scriptname, ref_calc, search_dim)
    assert os.path.isfile(scriptname)

    shutil.rmtree(subworkdir)


@pytest.mark.slow
def test_produce_delta_amps():
    manager = _make_test_manager()
    ref_calc_dir = os.path.join(manager.workdir, "ref-calc-results")
    ref_calc = tleed.RefCalc(problems.FESE_20UC, manager.leed_exe, manager.input_template,
                             ref_calc_dir, produce_tensors=True)
    # Manually assert that the ref calc has been done
    ref_calc.completed = True
    ref_calc.tensorfiles = [
        os.path.join(ref_calc_dir, "LAY1{}".format(i+1))
        for i in range(len(ref_calc.struct.layers[0]))
    ]

    search_dims = problems.FESE_DELTA_SEARCHDIMS
    delta_space = DeltaSearchSpace(ref_calc, search_dims)
    delta_amps = manager.produce_delta_amps(delta_space)

    assert len(delta_amps) == len(search_dims)
    for delta_amp in delta_amps:
        assert delta_amp.nbeams == len(manager.beaminfo.beams)
        assert delta_amp.nshifts == len(problems.DEFAULT_DELTA_DISPS)
        assert delta_amp.nvibs == len(problems.DEFAULT_DELTA_VIBS)
        assert delta_amp.theta == manager.beaminfo.theta
        assert delta_amp.phi == manager.beaminfo.phi

    return delta_space, delta_amps


def test_to_normalized():
    struct = problems.FESE_20UC

    search_space = SearchSpace(
        struct,
        [
            (SearchKey.VIB,   3, (-0.025, 0.025)),
            (SearchKey.ATOMX, 6, (-0.2, 0.2)),
            (SearchKey.ATOMY, 2, (-0.2, 0.2)),
            (SearchKey.ATOMZ, 4, (-0.4, 0.4))
        ]
    )

    norm_vib = 0.33
    norm_atomx = 0.2
    norm_atomy = 0.43
    norm_atomz = 0.78
    answer = np.array([norm_vib, norm_atomx, norm_atomy, norm_atomz])

    struct.sites[2].vib += -0.025 + norm_vib * 0.05
    struct.layers[0].xs[5] += -0.2 + norm_atomx * 0.4
    struct.layers[0].ys[1] += -0.2 + norm_atomy * 0.4
    struct.layers[0].zs[3] += -0.4 + norm_atomz * 0.8

    norm_vec = search_space.to_normalized(struct)
    assert np.allclose(norm_vec, answer)


def test_random_points():
    struct = problems.FESE_20UC

    search_space = SearchSpace(
        struct,
        [
            (SearchKey.VIB,   3, (-0.025, 0.025)),
            (SearchKey.ATOMX, 6, (-0.2, 0.2)),
            (SearchKey.ATOMY, 2, (-0.2, 0.2)),
            (SearchKey.ATOMZ, 4, (-0.4, 0.4))
        ]
    )
    num_pts = 20
    random_pts, random_structs = search_space.random_points(num_pts)

    # Check that I get num_pts number of points back
    assert len(random_pts) == len(random_pts) == num_pts

    # Check that all of the normalized points are indeed in a unit cube
    assert np.all(random_pts <= 1.0) and np.all(random_pts >= 0.0)

    # Check that converting the structures to the random_pts gets me the same thing
    assert all(
        np.allclose(pt, search_space.to_normalized(st))
        for pt, st in zip(random_pts, random_structs)
    )


def test_struct_getitem():
    struct = problems.FESE_20UC

    assert struct[SearchKey.VIB, 3] == struct.sites[2].vib
    assert struct[SearchKey.ATOMX, 6] == struct.layers[0].xs[5]
    assert struct[SearchKey.ATOMY, 1] == struct.layers[0].ys[0]
    assert struct[SearchKey.ATOMZ, 2] == struct.layers[0].zs[1]


def test_constraints():
    struct = problems.FESE_20UC

    constraints = [
        EqualityConstraint(SearchKey.VIB, 3, SearchKey.VIB, 1),
        EqualityConstraint(SearchKey.ATOMX, 6, SearchKey.ATOMX, 2),
        EqualityConstraint(SearchKey.ATOMZ, 4, SearchKey.ATOMZ, 5),
        EqualShiftConstraint(SearchKey.ATOMZ, 4, SearchKey.ATOMZ, 7),
        EqualShiftConstraint(SearchKey.CELLA, -1, SearchKey.CELLC, -1)
    ]

    search_space = SearchSpace(
        struct,
        [
            (SearchKey.VIB, 3, (-0.025, 0.025)),
            (SearchKey.ATOMX, 6, (-0.2, 0.2)),
            (SearchKey.ATOMY, 2, (-0.2, 0.2)),
            (SearchKey.ATOMZ, 4, (-0.4, 0.4)),
            (SearchKey.CELLA, -1, (-0.2, 0.2)),
        ],
        constraints=constraints
    )

    num_pts = 20
    random_pts, random_structs = search_space.random_points(num_pts)

    # Confirm that variables are bound together correctly
    for r_struct in random_structs:
        for constraint in constraints:
            b_key = constraint.bound_key
            b_idx = constraint.bound_idx
            s_key = constraint.search_key
            s_idx = constraint.search_idx
            if isinstance(constraint, EqualityConstraint):
                assert r_struct[b_key, b_idx] == r_struct[s_key, s_idx]
            if isinstance(constraint, EqualShiftConstraint):
                assert _isclose(r_struct[b_key, b_idx] - struct[b_key, b_idx],
                                r_struct[s_key, s_idx] - struct[s_key, s_idx])


# TODO: Update this to new manager requirements (phaseshifts, beaminfo)
@pytest.mark.slow
def test_refcalc():
    origdir = "test_files/LaNiO3test"
    newdir = "test_files/LaNiO3test_active"
    executable = "ref-calc.LaNiO3"
    if os.path.exists(newdir):
        shutil.rmtree(newdir)
    shutil.copytree(origdir, newdir)

    manager = bayessearch.create_manager(newdir, executable)

    struct = problems.LANIO3
    solution = problems.LANIO3_SOLUTION.tolist()
    solution.insert(4, solution[3])
    solution.insert(len(solution), solution[-1])
    for i, delta in enumerate(solution):
        struct[SearchKey.ATOMZ, i+1] += delta / struct.cell_params[2]

    rfactor = manager.ref_calc_blocking(struct)
    shutil.rmtree(newdir)

    # I handle the surface-to-bulk distance slightly differently than
    #   Jacob did, so I don't get exactly the same rfactor. (Actually better)
    assert _isclose(rfactor, problems.LANIO3_SOLUTION_RFACTOR, eps=0.01)
