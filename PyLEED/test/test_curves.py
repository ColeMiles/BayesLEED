import os
import filecmp

from pyleed import curves


def test_curve_readin():
    fese_curves = curves.parse_ivcurves("test/test_files/FeSetest/WEXPEL", format='TLEED')
    assert type(fese_curves) is curves.IVCurveSet
    assert len(fese_curves) == 5

    # Check some basic things as a sanity check -- should ideally do a full deep check
    labels = [c.label for c in fese_curves]
    assert labels == [(1, 0), (1, 1), (2, 0), (2, 2), (3, 0)]
    min_energies = [c.energies[0] for c in fese_curves]
    assert min_energies == [50., 66., 124., 254., 278.]
    max_energies = [c.energies[-1] for c in fese_curves]
    assert max_energies == [292., 458., 458., 498., 498.]
    final_intens = [c.intensities[-1] for c in fese_curves]
    assert final_intens == [1.086204, 1.546015, 1.623857, 2.687998, 1.385168]
    lens = [len(c) for c in fese_curves]
    assert lens == [122, 197, 168, 123, 111]


def test_curve_writeout():
    orig_filename = 'test/test_files/FeSetest/WEXPEL'
    fese_curves = curves.parse_ivcurves(orig_filename, format='TLEED')
    writeout_filename = 'test/test_files/FeSetest/tst.tmp'
    curves.write_curves(writeout_filename, fese_curves, format='TLEED')

    assert filecmp.cmp(orig_filename, writeout_filename)
    os.remove(writeout_filename)

    curves.write_curves(writeout_filename, fese_curves, format='PLOT')
    reread_curves = curves.parse_ivcurves(writeout_filename, format='PLOT')

    # Round original curve to two decimal places, since PLOT format does
    for curve in fese_curves:
        curve.intensities.round(2, out=curve.intensities)

    for orig_curve, reread_curve in zip(fese_curves, reread_curves):
        assert orig_curve == reread_curve
    os.remove(writeout_filename)


def test_crop_common_energy():
    fese_curves = curves.parse_ivcurves('test/test_files/FeSetest/WEXPEL')
    cropped_curves = curves._crop_common_energy(fese_curves.curves)
    min_en, max_en = cropped_curves[0].energies[0], cropped_curves[0].energies[-1]
    for curve in cropped_curves:
        assert curve.energies[0] == min_en
        assert curve.energies[-1] == max_en


