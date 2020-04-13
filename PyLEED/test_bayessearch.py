import pytest
import os
import shutil

import numpy as np
import bayessearch


@pytest.mark.slow
def test_lanio2_convergence():
    origdir = "test_files/LaNiO2test"
    newdir = "test_files/LaNiO2test_active"
    executable = os.path.join(newdir, 'ref-calc.LaNiO2')
    shutil.copytree(origdir, newdir)

    _, _, rfactors = bayessearch.main(
        executable, "LANIO2_PROBLEM", 8, 20
    )

    assert np.min(rfactors) < -1.3

    shutil.rmtree(newdir)
