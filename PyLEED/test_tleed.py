import pytest
import os
import shutil

import numpy as np

import tleed
from tleed import SearchKey
import bayessearch
import problems


def isclose(a, b, eps=1e-6):
    return abs(a - b) < eps


def test_to_script():
    struct = problems.FESE_20UC

    with open("test_files/fese_answer.txt", "r") as f:
        answer = f.read()

    assert struct.to_script() == answer


def test_write_structure():
    struct = problems.FESE_20UC

    basedir = "test_files/FeSetest/"
    exe = os.path.join(basedir, "ref-calc.FeSe")
    rfact = os.path.join(basedir, "rf.x")
    expfile = os.path.join(basedir, "WEXPEL")
    template = os.path.join(basedir, "FIN")

    with open(template, "r") as f:
        answer = f.read()

    manager = tleed.LEEDManager(basedir, exe, rfact, expfile, template)
    manager._write_structure(struct, "test_files/FeSetest/comp_FIN")

    with open("test_files/FeSetest/comp_FIN", "r") as f:
        output = f.read()

    assert output == answer

    os.remove("test_files/FeSetest/comp_FIN")


def test_to_structures():
    struct = problems.FESE_20UC

    search_space = tleed.SearchSpace(
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

    assert isclose(new_struct.sites[2].vib, expected_vib)
    assert isclose(new_struct.layers[0].xs[5], expected_atomx)
    assert isclose(new_struct.layers[0].ys[1], expected_atomy)
    assert isclose(new_struct.layers[0].zs[3], expected_atomz)


def test_to_normalized():
    struct = problems.FESE_20UC

    search_space = tleed.SearchSpace(
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

    search_space = tleed.SearchSpace(
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
        (SearchKey.VIB, 3, 1),
        (SearchKey.ATOMX, 6, 2),
        (SearchKey.ATOMZ, 4, 5)
    ]

    search_space = tleed.SearchSpace(
        struct,
        [
            (SearchKey.VIB, 3, (-0.025, 0.025)),
            (SearchKey.ATOMX, 6, (-0.2, 0.2)),
            (SearchKey.ATOMY, 2, (-0.2, 0.2)),
            (SearchKey.ATOMZ, 4, (-0.4, 0.4))
        ],
        constraints=constraints
    )

    num_pts = 20
    random_pts, random_structs = search_space.random_points(num_pts)

    # Confirm that variables are bound together correctly
    for struct in random_structs:
        for key, search_idx, bound_idx in constraints:
            assert struct[key, bound_idx] == struct[key, search_idx]


@pytest.mark.slow
def test_refcalc():
    origdir = "test_files/LaNiO2test"
    newdir = "test_files/LaNiO2test_active"
    executable = os.path.join(newdir, "ref-calc.LaNiO2")
    shutil.copytree(origdir, newdir)

    manager = bayessearch.create_manager(newdir, executable)

    struct = problems.LANIO3
    solution = problems.LANIO3_SOLUTION.tolist()
    solution.insert(4, solution[3])
    solution.insert(len(solution), solution[-1])
    for i, delta in enumerate(solution):
        struct[SearchKey.ATOMZ, i+1] += delta

    rfactor = manager.ref_calc(struct)
    shutil.rmtree(newdir)

    assert isclose(rfactor, problems.LANIO3_SOLUTION_RFACTOR)
