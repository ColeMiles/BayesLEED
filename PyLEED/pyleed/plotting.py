""" Functions for plotting various objects created during LEED
"""
from __future__ import annotations

import typing
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from pyleed import tleed

if typing.TYPE_CHECKING:
    from .curves import IVCurve, IVCurveSet


# TODO: Can this be refactored to share code with plot_ref_amps?
def plot_iv(ivcurves: Union[IVCurve, IVCurveSet], show=True):
    fig, ax = plt.subplots()
    ax.set_xlabel("Energy")
    ax.set_ylabel("Intensity")

    shift = 0.0

    # Check type of input, make correct list of beams
    if isinstance(ivcurves, IVCurve):
        plot_curves = [ivcurves]
    elif isinstance(ivcurves, IVCurveSet):
        plot_curves = ivcurves.curves
    else:
        raise ValueError("Argument `ivcurves` must be of typ IVCurve or IVCurveSet")

    for curve in plot_curves:
        curve_label = "({}, {})".format(*curve.label)
        intensities = curve.intensities + shift
        shift = np.max(intensities)
        ax.plot(curve.energies, intensities, color='k', label=curve_label)
        ax.text(curve.energies[-1] * 1.02, intensities[-1] * 1.02, curve_label)

    if show:
        plt.show()
    else:
        return fig, ax


def plot_ref_amps(delta_amps: tleed.SiteDeltaAmps, show=True):
    fig, ax = plt.subplots()
    ax.set_xlabel("Energy")
    ax.set_ylabel("Intensity")

    Es = delta_amps.real_energies_ev

    shift = 0.0

    for ibeam, beam in enumerate(delta_amps.beams):
        beamx, beamy = int(beam[0]), int(beam[1])

        amps = delta_amps.ref_amplitudes[ibeam, :]
        intensities = np.abs(amps)

        # Shift amplitudes up to avoid curve below
        intensities += shift

        # Adjust the shift so the next beam avoids this one
        shift = np.max(intensities)

        ax.plot(Es, intensities, color='k', label="({:d}, {:d})".format(beamx, beamy))
        ax.text(Es[-1] * 1.02, shift * 1.02, "({:d}, {:d})".format(beamx, beamy))

    if show:
        plt.show()
    else:
        return fig, ax


def plot_delta_amps(delta_amps: tleed.SiteDeltaAmps, delta: int, ax: plt.Axes = None, color='r'):
    Es = delta_amps.real_energies_ev

    shift = 0.0

    for ibeam, beam in enumerate(delta_amps.beams):
        beamx, beamy = int(beam[0]), int(beam[1])

        amps = delta_amps.ref_amplitudes[ibeam, :]
        del_amps = delta_amps.delta_amplitudes[ibeam, delta, :]
        intensities = np.abs(amps + del_amps)

        # Shift amplitudes up to avoid curve below
        intensities += shift

        # Adjust the shift to match the shift produced by plot_ref_amps (max of reference curve)
        shift = np.max(np.abs(amps) + shift)

        if ax is not None:
            ax.plot(Es, intensities, color=color)
        else:
            plt.plot(Es, intensities, color=color)

    return ax
