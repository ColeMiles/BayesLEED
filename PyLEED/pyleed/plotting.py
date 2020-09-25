""" Functions for plotting various objects created during LEED
"""
from __future__ import annotations

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

from pyleed import tleed

from .curves import IVCurve, IVCurveSet, _crop_common_energy

_COLOR_CYCLE = ['black', 'red', 'blue', 'green', 'orange', 'purple']


# TODO: Can this be refactored to share code with plot_ref_amps?
def plot_iv(ivcurves: Union[IVCurve, IVCurveSet], show=True, **kwargs):
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
        raise ValueError("Argument `ivcurves` must be of type IVCurve or IVCurveSet")

    for curve in plot_curves:
        curve_label = "({}, {})".format(*map(int, curve.label))
        intensities = curve.intensities + shift
        shift = np.max(intensities)
        ax.plot(curve.energies, intensities, color='k', **kwargs)
        ax.text(curve.energies[-1] * 1.02, intensities[-1] * 1.02, curve_label)

    if show:
        plt.show()
    else:
        return fig, ax


# TODO: Make work when some IVCurveSets have more beams than others
def plot_many_iv(ivcurve_sets: List[IVCurveSet], show=True, kwarg_list=None):
    if kwarg_list is None:
        kwarg_list = [
            {"color": _COLOR_CYCLE[i % len(_COLOR_CYCLE)]} for i in range(len(ivcurve_sets))
        ]
    for i, kwargs in enumerate(kwarg_list):
        if "color" not in kwargs:
            kwargs["color"] = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]

    # For now, assumes all curve sets have the same number of curves
    num_curves = len(ivcurve_sets[0])

    # Preprocessing: Find relative normalizations between curves
    norms = np.zeros(len(ivcurve_sets))
    for i, curveset in enumerate(ivcurve_sets):
        for curve in curveset:
            norms[i] += np.sum(np.square(curve.intensities))
    norms /= norms[0]

    fig, ax = plt.subplots()
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity (arb. units)")

    shift = 0.0

    for icurve in range(num_curves):
        max_intensity = shift
        all_curves = [curveset.curves[icurve] for curveset in ivcurve_sets]
        all_curves = _crop_common_energy(all_curves)
        for iset, (curve, kwargs) in enumerate(zip(all_curves, kwarg_list)):
            intensities = curve.intensities / norms[iset] + shift
            max_intensity = max(max_intensity, np.max(intensities))

            curve_label = ivcurve_sets[iset].set_label if icurve == 0 else None
            ax.plot(curve.energies, intensities, label=curve_label, **kwargs)

        beam_label = "({}, {})".format(*map(int, all_curves[0].label))
        ax.text(
            all_curves[0].energies[-1] * 1.02, (all_curves[0].intensities[-1] + shift) * 1.02,
            beam_label
        )
        shift = max_intensity

    if show:
        plt.legend()
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
