import pytest
import os
import shutil

import numpy as np
from pyleed import bayessearch


# TODO: Update this to the new manager requirements (beaminfo / phaseshifts)
@pytest.mark.slow
def test_lanio3_convergence():
    origdir = "test_files/LaNiO3test"
    newdir = "test_files/LaNiO3test_active"
    executable = os.path.join(newdir, 'ref-calc.LaNiO3')
    if os.path.exists(newdir):
        shutil.rmtree(newdir)
    shutil.copytree(origdir, newdir)

    _, _, rfactors = bayessearch.main(
        executable, "LANIO3", 8, 20, early_stop=0.3
    )

    assert np.min(rfactors) < 0.3

    shutil.rmtree(newdir)


def test_create_model():
    # Some fake points / rfactors
    pts = np.array([[0.2345, 0.1452, 0.2535, 0.7783],
                    [0.6726, 0.5372, 0.4567, 0.1265],
                    [0.7246, 0.7345, 0.2613, 0.76435]])
    rfactors = np.array([0.6, 0.45, 0.7])
    normalized_rfactors = (rfactors - rfactors.mean()) / rfactors.std(ddof=1)

    assert np.allclose(normalized_rfactors.mean(), 0.0)
    assert np.allclose(normalized_rfactors.var(ddof=1), 1.0)

    model, mll = bayessearch.create_model(pts, -normalized_rfactors)

    assert len(model.train_inputs) == 1
    assert np.all(model.train_inputs[0].cpu().numpy() == pts)
    assert np.all(model.train_targets.cpu().numpy() == -normalized_rfactors)
    assert mll.model == model
