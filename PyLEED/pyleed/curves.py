from __future__ import annotations

import re
import numpy as np
from scipy import signal
from typing import List, Tuple
import numba
from numba import njit

from . import plotting


class IVCurve:
    def __init__(self, energies: np.ndarray, intensities: np.ndarray, label: Tuple[float, float]):
        self.energies: np.ndarray = energies
        self.intensities: np.ndarray = intensities
        self.label: Tuple[float, float] = label

    def smooth(self, nsmooth=1) -> IVCurve:
        smooth_intensities = self.intensities
        for _ in range(nsmooth):
            smooth_intensities = _smooth(smooth_intensities)
        return IVCurve(self.energies, smooth_intensities, self.label)

    def plot(self):
        plotting.plot_iv(self)


class IVCurveSet:
    def __init__(self, curves=None):
        self.curves: List[IVCurve] = [] if curves is None else curves
        self.imagV: float = None
        self.interp_dE: float = None
        self.interp_energies: List[np.ndarray] = None
        self.interp_pendries: List[np.ndarray] = None
        self.precomputed = False

    def __len__(self):
        return len(self.curves)

    def __iter__(self):
        return iter(self.curves)

    def __getitem__(self, idx: int):
        return self.curves[idx]

    def smooth(self, nsmooth=1) -> IVCurveSet:
        smoothed_curves = [curve.smooth(nsmooth) for curve in self.curves]
        return IVCurveSet(smoothed_curves)

    def plot(self):
        plotting.plot_iv(self)

    def precompute_pendry(self, imagV: float = 5.0, dE: float = 0.5):
        """ Precomputes the pendry Y functions for this set of IV curves.
            avg_rfactors will check for these and skip recomputation if present.

            imagV: Imaginary part of the crystal potential
            dE : The energy grid step size to interpolate to.

            Both must match intended values to use in avg_rfactors!
        """
        self.imagV = imagV

        self.interp_energies = []
        self.interp_pendries = []
        for curve in self.curves:
            interpE, interpI = _resample_interpolate(curve.energies, curve.intensities, dx=dE)
            interpIp = _deriv(interpI, dx=dE)
            interpY = _pendry_Y(interpI, interpIp, imagV)
            self.interp_energies.append(interpE)
            self.interp_pendries.append(interpY)

        self.precomputed = False


def _parse_experiment_tleed(filename: str, extra_header=True) -> IVCurveSet:
    """ Parse IV curves that are in the experimental data format expected by TLEED
    """
    ivcurves = IVCurveSet()

    with open(filename, 'r') as f:
        if extra_header:
            # File WEXPEL created has 28 extra lines at the top we don't care about at the moment
            for _ in range(28):
                f.readline()

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


def _parse_theory_tleed(filename: str) -> IVCurveSet:
    ivcurves = IVCurveSet()

    with open(filename, 'r') as f:
        # Description line
        f.readline()

        num_beams = int(f.readline())

        beam_labels = []
        for _ in range(num_beams):
            line_s = f.readline().split()
            beam_labels.append((float(line_s[1]), float(line_s[2])))

        # Read all of the data in, accounting for Fortran's weird output formatting
        # TODO: Make a separate utility function which reads Fortran-formatted output
        line = None
        energies = []
        beams = [[] for _ in range(num_beams)]

        line = f.readline()
        while line != "":
            line_s = line.split()
            energies.append(float(line_s[0]))
            num_beams_read = len(line_s) - 2
            for i in range(num_beams_read):
                beams[i].append(float(line_s[2+i]))
            while num_beams_read < num_beams:
                line_s = f.readline().split()
                for i, val in enumerate(line_s):
                    beams[num_beams_read+i].append(float(val))
                num_beams_read += len(line_s)
            line = f.readline()

        energies = np.array(energies)

        for beam, label in zip(beams, beam_labels):
            ivcurves.curves.append(
                IVCurve(energies, np.array(beam), label)
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
        return _parse_experiment_tleed(filename, extra_header=False)
    elif format.upper() == 'WEXPEL':
        return _parse_experiment_tleed(filename, extra_header=True)
    elif format.upper() == 'RCOUT':
        return _parse_theory_tleed(filename)
    elif format.upper() == 'PLOT':
        return _parse_ivcurves_plotfmt(filename)
    else:
        raise ValueError("Unknown format: {}".format(format))


@njit
def _resample_interpolate(x: np.ndarray, y: np.ndarray, dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    resample_x = np.arange(x[0], x[-1]+dx, step=dx)
    return resample_x, _interpolate_fortran(x, y, resample_x)


@njit(fastmath=True)
def _interpolate_fortran(oldx, oldy, newx):
    """ Attempts to match the Fortran interpolation [XNTERP in aux/rf.f] exactly.
        However, due to all arithmetic being done in pure Python, this is probably
         very slow.
    """
    newy = np.empty_like(newx)
    for i, x in enumerate(newx):
        # Index into array of first value >= x
        iright = np.searchsorted(oldx, x, )
        # Index of start of 4-point run used for interpolation
        istart = iright - 2
        # Check if we have/will run off bounds, and correct for this
        istart = max(istart, 0)
        istart = min(istart, len(oldx)-4)
        x0, x1, x2, x3 = oldx[istart], oldx[istart+1], oldx[istart+2], oldx[istart+3]
        y0, y1, y2, y3 = oldy[istart], oldy[istart+1], oldy[istart+2], oldy[istart+3]

        # Don't ask what exactly these equations are doing... copied directly from XNTERP
        # Some sort of cubic interpolation, but not what's done by scipy
        term = y0
        fact1 = x - x0
        fact2 = (y1 - y0) / (x1 - x0)
        term += fact1 * fact2
        fact1 *= (x - x1)
        fact2 = ((y2 - y1) / (x2 - x1) - fact2) / (x2 - x0)
        term += fact1 * fact2
        fact1 *= (x - x2)
        temp = ((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1)
        fact2 = (temp - fact2) / (x3 - x0)
        term += fact1 * fact2

        newy[i] = term

    return newy


@njit(fastmath=True)
def _compute_rfactor(trueE, trueY, compE, compY, shiftE=0.0):
    """ Compute r-factor between two Y(E) curves, with a relative shift of the
            compE axis. Assumes both live on same grid, just different subsets.
        Returns (rfactor, energy_overlap)
    """
    deltaE = trueE[1] - trueE[0]

    lowE = max(compE[0] + shiftE, trueE[0])
    highE = min(compE[-1] + shiftE, trueE[-1])

    i_lo_true = np.searchsorted(trueE, lowE)
    i_hi_true = np.searchsorted(trueE, highE)
    i_lo_comp = np.searchsorted(compE, lowE-shiftE)
    i_hi_comp = np.searchsorted(compE, highE-shiftE)

    assert trueE[i_lo_true] == lowE
    assert trueE[i_hi_true] == highE
    assert compE[i_lo_comp] == lowE - shiftE
    assert compE[i_hi_comp] == highE - shiftE

    slice_trueY = trueY[i_lo_true:i_hi_true+1]
    slice_compY = compY[i_lo_comp:i_hi_comp+1]

    numer_int = np.trapz((slice_compY - slice_trueY) ** 2, dx=deltaE)
    denom_int = np.trapz((slice_compY ** 2 + slice_trueY ** 2), dx=deltaE)

    return numer_int / denom_int, (i_hi_true - i_lo_true) * deltaE


@njit(fastmath=True)
def _pendry_Y(arrI, deriv_arrI, imagV):
    """ Computes Pendry's Y function in the way that rf.f does: account for
         small values of the intensity in a separate branch.
    """
    Y = np.zeros_like(arrI)
    for i, (I, Ip) in enumerate(zip(arrI, deriv_arrI)):
        if I < 1e-7:
            if Ip > 1e-7:
                invL = I / Ip
                Y[i] = invL / (invL**2 + imagV**2)
        else:
            L = Ip / I
            Y[i] = L / (1 + imagV**2 * L**2)
    return Y


# Accurate to Fortran to less than 1% error
def rfactor(
        true_curve: IVCurve, comp_curve: IVCurve,
        realV: float = -10.7, imagV: float = 5.0,
        realV_shifts: np.ndarray = np.arange(-8, 8.5, step=0.5),
        deltaE: float = 0.5
    ) -> List[float]:
    """ Calculates Pendry's R-factor between the two I(V) curves.
        Attempts to match the Fortran as best as possible.
    """
    trueE, trueI = true_curve.energies, true_curve.intensities
    compE, compI = comp_curve.energies, comp_curve.intensities

    trueE, trueI = _resample_interpolate(trueE, trueI, dx=deltaE)
    compE, compI = _resample_interpolate(compE, compI, dx=deltaE)

    # TODO: Avoid memory allocation here?
    deriv_true = _deriv(trueI, dx=deltaE)
    deriv_comp = _deriv(compI, dx=deltaE)

    trueY = _pendry_Y(trueI, deriv_true, imagV)
    compY = _pendry_Y(compI, deriv_comp, imagV)

    rfactors = [_compute_rfactor(trueE, trueY, compE, compY, shiftE=shiftE)[0] for shiftE in realV_shifts]

    return rfactors


# TODO: Make this njit-able. Requires IVCurveSet to become a jitclass
def avg_rfactors(
        true_curves: IVCurveSet, comp_curves: IVCurveSet,
        imagV: float = 5.0, realV_shifts: np.ndarray = np.arange(-8, 8.5, step=0.5),
        deltaE: float = 0.5
    ) -> np.ndarray:
    """ Calculates Pendry's R-factor between the two sets of I(V) curves,
         testing at a range of potential shifts given by realV_shifts.
    """
    if len(true_curves) != len(comp_curves):
        raise ValueError("Number of experimental curves does not equal number of theoretical curves!")

    # Calculate Pendry's Y function for all experimental curves
    if not true_curves.precomputed:
        true_curves.precompute_pendry(imagV, dE=deltaE)
    true_Es = true_curves.interp_energies
    true_Ys = true_curves.interp_pendries

    # Compute r-factors for all beams, for all potential shifts
    rfactor_list = np.empty((len(true_curves), len(realV_shifts)))
    overlap_list = np.empty((len(true_curves), len(realV_shifts)))

    # Calculate Pendry's Y function for all theoretical curves, and compare to experiment
    for i, curve in enumerate(comp_curves):
        compE, compI = curve.energies, curve.intensities
        compE, compI = _resample_interpolate(compE, compI, dx=deltaE)
        compIp = _deriv(compI, dx=deltaE)
        compY = _pendry_Y(compI, compIp, imagV)
        for j, shift in enumerate(realV_shifts):
            rfactor_list[i, j], overlap_list[i, j] = _compute_rfactor(
                true_Es[i], true_Ys[i], compE, compY, shiftE=shift
            )

    weighted_rfactors = rfactor_list * overlap_list / np.sum(overlap_list, axis=0, keepdims=True)
    avg_rfacts = np.sum(weighted_rfactors, axis=0)

    return avg_rfacts


@njit(fastmath=True)
def _deriv(y: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """ Returns derivatives of y, using the same 6-point stencil on the interior and
         4-point stencil on the boundaries that TLEED uses.
    """
    dydx = np.empty_like(y)

    for i in range(3, len(y)-3):
        dydx[i] = (y[i+3] - 9*y[i+2] + 45*y[i+1] - 45*y[i-1] + 9*y[i-2] - y[i-3]) / (60 * dx)
    for i in range(3):
        dydx[i] = (2*y[i+3] - 9*y[i+2] + 18*y[i+1] - 11*y[i]) / (6 * dx)
        ip = len(y) - i - 1
        dydx[ip] = (11*y[i] - 18*y[i-1] + 9*y[i-2] - 2*y[i-3]) / (6 * dx)

    return dydx


def _smooth(data: np.ndarray) -> np.ndarray:
    """ Simple 3-point windowed averaging smoothing, matching TLEED
    """
    filt = np.array([1/4, 1/2, 1/4])
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    smoothed[1:-1] = signal.correlate(data, filt, mode='valid')
    smoothed[-1] = data[-1]
    return smoothed
