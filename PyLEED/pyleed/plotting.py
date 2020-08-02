import matplotlib.pyplot as plt
import numpy as np

from pyleed import tleed


def plot_ref_amps(delta_amps: tleed.SiteDeltaAmps, show=True):
    fig, ax = plt.subplots()
    ax.set_xlabel("Energy")
    ax.set_ylabel("Intensity")

    Es = delta_amps.crystal_energies

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
        ax.text(Es[-1] + 1.02, shift * 1.02, "({:d}, {:d})".format(beamx, beamy))

    if show:
        plt.show()
    else:
        return fig, ax


def plot_delta_amps(delta_amps: tleed.SiteDeltaAmps, delta: int, ax: plt.Axes = None, color='r'):
    Es = delta_amps.crystal_energies

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
