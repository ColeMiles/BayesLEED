import argparse
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import botorch

from bayessearch import create_model

X1LIM = [-0.25, 0.25]
X2LIM = [-0.25, 0.0]

def emulate_search(pts, rfactors, batch_size, justpoints=False):
    """ Given an array of pts and their corresponding r-factors, steps through training botorch
         models using the given pts, batch_size at a time.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("R-Factor")
    ax.set_xlim(*X1LIM)
    ax.set_ylim(*X2LIM)
    ax.set_zlim(0.0, 1.5)

    if justpoints:
        ax.scatter(pts[:, 0], pts[:, 1], rfactors)
        plt.show()
        sys.exit(0)

    num_pts = len(pts)
    num_batches = num_pts // batch_size

    state_dict = None

    plt.ion()
    plt.show()
    cbar = None
    for i in range(num_batches):
        print("{} points evaluated".format(batch_size))

        idx1, idx2 = batch_size * i, batch_size * (i + 1)
        ax.scatter(pts[:idx2, 0], pts[:idx2, 1], rfactors[:idx2], c="blue", depthshade=False)
        plt.draw()
        input("Fit Model? [Enter]")

        model, mll = create_model(pts[:idx2, :], rfactors[:idx2], state_dict=state_dict)
        botorch.fit.fit_gpytorch_model(mll)
        state_dict = model.state_dict()

        # Plot mean function
        model.eval()
        X, Y = torch.meshgrid(torch.linspace(*X1LIM), torch.linspace(*X2LIM))
        eval_pts = torch.stack((X, Y), axis=-1).type(torch.float64).cuda()
        posterior = model(eval_pts)
        mean_func = -posterior.mean.detach().cpu().numpy()
        surf = ax.plot_surface(X, Y, mean_func, cmap="plasma")
        if cbar is None:
            cbar = fig.colorbar(surf)
        else:
            cbar.mappable.set_clim(vmin=mean_func.min(), vmax=mean_func.max())
            cbar.draw_all()

        input("Calculate Acquisition Function? [Enter]")

        input("\nNext Batch? [Enter]")
        # fig.clear()
        ax.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feeds already-evaluated points into a model and makes a visualization of the process."
    )
    parser.add_argument("pointfile", type=str, help="File containing points output by bayessearch.py")
    parser.add_argument("--batch", type=int, default=8, help="The batch size used")
    parser.add_argument("--justpoints", action="store_true", help="If set, doesn't fit model, just displays points")
    args = parser.parse_args()

    pts_rfactors = np.loadtxt(args.pointfile, skiprows=1)

    pts = pts_rfactors[:, :-1]
    rfactors = pts_rfactors[:, -1]
    if pts.shape[1] != 2:
        raise ValueError("Currently, this script only supports 2D searches")

    print("Beginning emulated optimization, with {} points and batch size {}".format(len(pts), args.batch))
    emulate_search(pts, rfactors, args.batch, justpoints=args.justpoints)