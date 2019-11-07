#!/usr/bin/python3
import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-talk")

def plot_data(arr):
    """ Plots an array of I(E) data, assuming that every two columns
         forms an (E, I) curve to plot.
    """
    # Check that there is an even number of columns
    if arr.shape[1] % 2 != 0:
        raise ValueError("Not an even number of columns in data array!")

    num_curves = arr.shape[1] // 2

    for i in range(num_curves):
        # Data ends in a stream of zeros - find where that is to avoid plotting it
        plt.plot(
            np.trim_zeros(arr[:, 2*i], 'b'), 
            np.trim_zeros(arr[:, 2*i+1], 'b'), 
            label="N="+str(i)
        )

    plt.xlabel(r"$E$")
    plt.ylabel(r"$I$")
    #plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I(E) Plotter")
    parser.add_argument("datafile", type=str)
    args = parser.parse_args()
    data = np.loadtxt(args.datafile)
    plot_data(data)
