import re
import numpy as np
from typing import List, Tuple


class IVCurve:
    def __init__(self, energies: np.ndarray, intensities: np.ndarray, label: Tuple[float, float]):
        self.energies: np.ndarray = energies
        self.intensities: np.ndarray = intensities
        self.label: Tuple[float, float] = label


class IVCurveSet:
    def __init__(self):
        self.curves: List[IVCurve] = []


def _parse_experiment_tleed(filename: str) -> IVCurveSet:
    """ Parse IV curves that are in the experimental data format expected by TLEED
    """
    ivcurves = IVCurveSet()

    with open(filename, 'r') as f:
        # Title line
        f.readline()

        # Grouping line: Use to find number of beams to expect
        line = f.readline()
        num_beams = len(line) // 3

        # Format specification line
        line = f.readline()
        pat = r"^\(F(\d)+\.(\d)+,F(\d)+\.(\d)+\)$"
        match = re.match(pat, line)
        if match is None:
            raise RuntimeError("File {} has ill-formed specification line".format(filename))
        # Format specifications for energy, intensity columns - currently not needed
        Ef1, Ef2 = match.group(1), match.group(2)
        If1, If2 = match.group(1), match.group(2)

        for nbeam in range(num_beams):
            # Beam label line
            line = f.readline()
            pat = r"^\((\d+\.\d+),(\d+\.\d+)\)$"
            match = re.match(pat, line)
            if match is None:
                raise RuntimeError("File {} has ill-formed beam label".format(filename))
            label = (float(match.group(1)), float(match.group(2)))

            # Line stating how many energies are to come
            line = f.readline()
            num_energies = int(line.split()[0])
            energies = np.empty(num_energies)
            intensities = np.empty(num_energies)

            # Read in all those energies
            for i in range(num_energies):
                line = f.readline()
                energy, intens = map(float, line.split())
                energies[i] = energy
                intensities[i] = intens

            ivcurves.curves.append(
                IVCurve(energies, intensities, label)
            )

    return ivcurves


def _parse_ivcurves_plotfmt(filename: str) -> IVCurveSet:
    """ Parse IV curves in the format output by TLEED for plotting.
        This format does not provide beam labels: This sets all to (-1, -1).
    """
    ivcurves = IVCurveSet()
    data = np.loadtxt(filename)
    num_beams = data.shape[1] // 2
    for i in range(num_beams):
        energies = np.trim_zeros(data[:, 2*i], 'b')
        intensities = np.trim_zeros(data[:, 2*i+1], 'b')
        ivcurves.curves.append(
            IVCurve(energies, intensities, (-1, -1))
        )
    return ivcurves


def parse_ivcurves(filename: str, format='TLEED') -> IVCurveSet:
    if format.upper() == 'TLEED':
        return _parse_experiment_tleed(filename)
    elif format.upper() == 'PLOT':
        return _parse_ivcurves_plotfmt(filename)
    else:
        raise ValueError("Unknown format: {}".format(format))


def rfactor(true_curve: IVCurve, comp_curve: IVCurve) -> float:
    """ Calculates Pendry's R-factor between the two I(V) curves.
    """
    return 0.0
