import pytest

from pyleed import searchspace

from pyleed.tleed import parse_ivcurves
from test_tleed import test_produce_delta_amps


@pytest.mark.slow
def test_optimize_anneal():
    delta_space, delta_amps = test_produce_delta_amps()
    exp_curves = parse_ivcurves('test/test_files/FeSetest/WEXPEL', format='WEXPEL')

    print("Completed delta creation. Starting optimization...")
    (best_geo, best_vib), best_rfactor = searchspace.optimize_delta_anneal(
        delta_space, delta_amps, exp_curves
    )

    return (best_geo, best_vib), best_rfactor
