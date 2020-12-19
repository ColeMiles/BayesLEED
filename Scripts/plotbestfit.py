#!/usr/bin/python3
import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-talk")

def plot_data(exp_data, sim_data=None, title="", rfactors=None, scale=False):
    """ Plots an array of I(E) data, assuming that every two columns
         forms an (E, I) curve to plot.
    """
    if sim_data is None:
        sim_data = np.array([[], []])

    # Check that there is an even number of columns
    if exp_data.shape[1] % 2 != 0 or sim_data.shape[1] % 2 != 0:
        raise ValueError("Not an even number of columns in data array!")

    num_exp_curves = exp_data.shape[1] // 2

    # Check num rfactors matches num beams
    if rfactors is not None and len(rfactors) != num_exp_curves:
        raise ValueError("Number of rfactors does not match number of beams")

    curve_scales = []
    for i in range(num_exp_curves):
        # Data ends in a stream of zeros - find where that is to avoid plotting it
        curve_scales.append(np.max(exp_data[:, 2*i+1]))
        plt.plot(
            np.trim_zeros(exp_data[:, 2*i], 'b'), 
            np.trim_zeros(exp_data[:, 2*i+1], 'b') + 20 * i, 
            label="Exp" if i == 0 else None,
            color="k"
        )

    num_sim_curves = sim_data.shape[1] // 2
    for i in range(num_sim_curves):
        chop = len(np.trim_zeros(exp_data[:, 2*i], 'b'))
        # Data ends in a stream of zeros - find where that is to avoid plotting it
        E_data = np.trim_zeros(sim_data[:chop, 2*i], 'b')
        I_data = np.trim_zeros(sim_data[:chop, 2*i+1], 'b')

        # Scale maxima to match if scale=True
        if scale:
            max_I = np.max(I_data)
            I_data *= curve_scales[i] / max_I

        # Rigid shift to separate curves
        I_data += 20 * i
        plt.plot(
            E_data,
            I_data,
            label="Sim" if i == 0 else None,
            color="r"
        )
        if rfactors is not None:
            plt.text(
                np.max(E_data) + 5,
                np.mean(I_data),
                "Rf = {:.2f}".format(rfactors[i])
            )


    plt.xlabel(r"$E$")
    plt.ylabel(r"$I$")
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I(E) Plotter")
    parser.add_argument("exp_datafile", type=str)
    parser.add_argument("sim_datafile", type=str, nargs='?')
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--rfactors", nargs="+", type=float)
    parser.add_argument("--scale", action="store_true", help="If set, scales curves to match maxima heights")
    args = parser.parse_args()
    exp_data = np.loadtxt(args.exp_datafile)
    if args.sim_datafile is None:
        sim_data = None
    else:
        sim_data = np.loadtxt(args.sim_datafile)
    plot_data(exp_data, sim_data, title=args.title, rfactors=args.rfactors, scale=args.scale)
