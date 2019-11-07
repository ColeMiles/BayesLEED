import argparse
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import botorch

from bayessearch import create_model, normalize_input, denormalize_input

plt.style.use("seaborn-talk")

X1LIM = [-0.25, 0.25]
X2LIM = [-0.25, 0.25]


def emulate2D(pts, rfactors, batch_size, justpoints=False):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        ax.scatter(pts[:idx2, 0], pts[:idx2, 1], rfactors[:idx2], c="black", linewidths=5, depthshade=False)
        plt.draw()
        input("Fit Model? [Enter]")

        normalized_pts = normalize_input(pts[:idx2, :])
        model, mll = create_model(normalized_pts, rfactors[:idx2], state_dict=state_dict)
        botorch.fit.fit_gpytorch_model(mll)
        state_dict = model.state_dict()

        # Plot mean function
        model.eval()
        X, Y = torch.meshgrid(torch.linspace(*X1LIM), torch.linspace(*X2LIM))
        eval_pts = torch.stack((X, Y), axis=-1).type(torch.float64).to(device=device)
        posterior = model(normalize_input(eval_pts))
        mean_func = -posterior.mean.detach().cpu().numpy()
        surf = ax.plot_surface(X, Y, mean_func, cmap="plasma", alpha=0.75)
#        if cbar is None:
#            cbar = fig.colorbar(surf)
#        else:
#            cbar.mappable.set_clim(vmin=mean_func.min(), vmax=mean_func.max())
#            cbar.draw_all()

        input("Calculate Acquisition Function? [Enter]")
        best_rfactor = np.min(rfactors[:idx2])
        acq = botorch.acquisition.ExpectedImprovement(model, -best_rfactor)
        acq_values = acq(normalize_input(eval_pts)[:, :, None]).cpu().detach().numpy()
        ax.plot_surface(X, Y, acq_values, cmap="seismic")

        # I'm plotting the analytic single-point acquisition function but in reality I optimize qEI
        sampler = botorch.sampling.SobolQMCNormalSampler(num_samples=2500, resample=False)
        qacq = botorch.acquisition.qExpectedImprovement(model, -best_rfactor, sampler)
        new_normalized_pts, _ = botorch.optim.optimize_acqf(
            acq_function=qacq,
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=torch.float64),
            q=batch_size,
            num_restarts=20,
            raw_samples=200,
            options={},
            sequential=True
        )
        new_pts = denormalize_input(new_normalized_pts).cpu().detach().numpy()
        for pt in new_pts:
            ax.plot((pt[0], pt[0]), (pt[1], pt[1]), (0.0, 1.4), linewidth=4, color="green")
        input("\nNext Batch? [Enter]")
        ax.clear()
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel("R-Factor")
        ax.set_xlim(*X1LIM)
        ax.set_ylim(*X2LIM)
        ax.set_zlim(0.0, 1.5)

def emulate1D(pts, rfactors, batch_size, justpoints=False):
    """ Given an array of pts and their corresponding r-factors, steps through training botorch
         models using the given pts, batch_size at a time.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel("R-Factor")
    ax.set_xlim(*X1LIM)
    ax.set_ylim(0.0, 1.2)

    if justpoints:
        Ax.scatter(pts[:, 0], rfactors)
        plt.show()
        sys.exit(0)

    num_pts = len(pts)
    num_batches = num_pts // batch_size

    state_dict = None

    plt.ion()
    plt.show()
    for i in range(num_batches):
        print("{} points evaluated".format(batch_size))

        idx1, idx2 = batch_size * i, batch_size * (i + 1)
        ax.scatter(pts[:idx2, 0], rfactors[:idx2], c="black")
        plt.draw()
        input("Fit Model? [Enter]")

        normalized_pts = normalize_input(pts[:idx2, :])
        model, mll = create_model(normalized_pts, rfactors[:idx2], state_dict=state_dict)
        botorch.fit.fit_gpytorch_model(mll)
        state_dict = model.state_dict()

        # Plot mean function
        model.eval()
        eval_pts = torch.linspace(*X1LIM, dtype=torch.float64, steps=500)[:, None].to(device=device)
        posterior = model(normalize_input(eval_pts))
        mean_func = -posterior.mean.detach().cpu().numpy()
        lower_conf, upper_conf = posterior.confidence_region()

        eval_pts_np = eval_pts.cpu().numpy()[:, 0]
        upper_conf_np = -lower_conf.cpu().detach().numpy()
        lower_conf_np = -upper_conf.cpu().detach().numpy()
        ax.plot(eval_pts_np, mean_func)
        # import ipdb
        # ipdb.set_trace()
        ax.fill_between(eval_pts_np, lower_conf_np, upper_conf_np, alpha=0.5)

        input("Calculate Acquisition Function? [Enter]")
        best_rfactor = np.min(rfactors[:idx2])
        acq = botorch.acquisition.ExpectedImprovement(model, -best_rfactor)
        acq_values = acq(normalize_input(eval_pts)[:, :, None]).cpu().detach().numpy()
        ax.plot(eval_pts_np, acq_values, "r-")

        # I'm plotting the analytic single-point acquisition function but in reality I optimize qEI
        sampler = botorch.sampling.SobolQMCNormalSampler(num_samples=2500, resample=False)
        qacq = botorch.acquisition.qExpectedImprovement(model, -best_rfactor, sampler)
        new_normalized_pts, _ = botorch.optim.optimize_acqf(
            acq_function=qacq,
            bounds=torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float64),
            q=batch_size,
            num_restarts=20,
            raw_samples=200,
            options={},
            sequential=True
        )
        new_pts = denormalize_input(new_normalized_pts).cpu().detach().numpy()
        ax.vlines(new_pts[:, 0], 0, 1, transform=ax.get_xaxis_transform(), colors='g')


        input("\nNext Batch? [Enter]")
        ax.clear()
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel("R-Factor")
        ax.set_xlim(*X1LIM)
        ax.set_ylim(0.0, 1.2)


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
    if pts.shape[1] == 1:
        print("Beginning emulated optimization, with {} points and batch size {}".format(len(pts), args.batch))
        emulate1D(pts, rfactors, args.batch, justpoints=args.justpoints)
    elif pts.shape[1] == 2:
        print("Beginning emulated optimization, with {} points and batch size {}".format(len(pts), args.batch))
        emulate2D(pts, rfactors, args.batch, justpoints=args.justpoints)
    else:
        raise ValueError("Currently, this script only supports 1D or 2D searches")
