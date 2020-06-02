import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import torch
import botorch
import IPython

scriptdir = os.path.dirname(os.path.realpath(__file__))
packagedir = os.path.dirname(scriptdir)
sys.path.insert(0, os.path.join(packagedir, "pyleed"))

from pyleed.bayessearch import create_model

plt.style.use("seaborn-talk")

def fit_model(pts, norm_rfactors, state_dict=None):
    """ Fits a botorch model, given (normalized) points and (normalized) rfactors
    """
    model, mll = create_model(pts, -norm_rfactors, state_dict=state_dict)
    model.train()
    botorch.fit.fit_gpytorch_model(mll)
    return model, mll

def emulateND(pts, rfactors, batch_size):
    """ Given an array of pts and their corresponding r-factors, steps through training botorch
         models using the given pts, batch_size at a time.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_pts = len(pts)
    num_batches = num_pts // batch_size
    state_dict = None

    for i in range(num_batches):
        idx1, idx2 = batch_size * i, batch_size * (i + 1)
        batch_pts = pts[:idx2, :]
        batch_rfactors = rfactors[:idx2]
        batch_normalized_rfactors = (batch_rfactors - batch_rfactors.mean()) / batch_rfactors.std(ddof=1)
        model, mll = fit_model(batch_pts, batch_normalized_rfactors, state_dict=state_dict)
        state_dict = model.state_dict()

        print("Epoch {} completed".format(i))
        IPython.embed()


def emulate2D(pts, rfactors, batch_size, justpoints=False):
    """ Given an array of pts and their corresponding r-factors, steps through training botorch
         models using the given pts, batch_size at a time.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("Normalized R-Factor")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-2.0, 2.0)
    plt.subplots_adjust(bottom=0.2)
    button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
    button = Button(button_ax, "Computing...")
    button.set_active(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if justpoints:
        ax.scatter(pts[:, 0], pts[:, 1], rfactors)
        plt.show()
        sys.exit(0)

    num_pts = len(pts)
    num_batches = num_pts // batch_size
    batch_num = 1
    batch_idx = batch_size * batch_num
    batch_pts = pts[:batch_idx, :]
    batch_rfactors = rfactors[:batch_idx]
    normalized_batch_rfactors = (batch_rfactors - batch_rfactors.mean()) / batch_rfactors.std(ddof=1)

    # Should I plot normalized rfactors? Normalization changes each batch
    ax.scatter(batch_pts[:, 0], batch_pts[:, 1], normalized_batch_rfactors, c="black", linewidths=5, depthshade=False)
    button.set_active(True)
    button.label.set_text("Fit Model")
    plt.draw()

    state = "Sample"
    model = None
    state_dict = None

    def on_clicked(event):
        nonlocal state
        nonlocal model
        nonlocal state_dict
        nonlocal button
        nonlocal ax
        nonlocal batch_idx
        nonlocal batch_pts
        nonlocal batch_size
        nonlocal batch_num
        nonlocal batch_rfactors
        nonlocal normalized_batch_rfactors

        if state == "Sample":
            model, mll = fit_model(batch_pts, normalized_batch_rfactors, state_dict=state_dict)
            state_dict = model.state_dict()

            # Plot mean function
            model.eval()
            X, Y = torch.meshgrid(torch.linspace(0.0, 1.0), torch.linspace(0.0, 1.0))
            eval_pts = torch.stack((X, Y), axis=-1).type(torch.float64).to(device=device)
            posterior = model(eval_pts)
            mean_func = -posterior.mean.detach().cpu().numpy()
            surf = ax.plot_surface(X, Y, mean_func, cmap="plasma", alpha=0.75)
            button.label.set_text("Calc. Acquisition")
            state = "ModelSurface"
            plt.draw()
        elif state == "ModelSurface":
            X, Y = torch.meshgrid(torch.linspace(0.0, 1.0), torch.linspace(0.0, 1.0))
            eval_pts = torch.stack((X, Y), axis=-1).type(torch.float64).to(device=device)
            best_rfactor = np.min(normalized_batch_rfactors)
            acq = botorch.acquisition.ExpectedImprovement(model, -best_rfactor)
            acq_values = acq(eval_pts[:, :, None]).cpu().detach().numpy()
            ax.plot_surface(X, Y, acq_values - 2.0, cmap="seismic")

            new_pts = pts[batch_idx:batch_idx+batch_size]
            for pt in new_pts:
                ax.plot((pt[0], pt[0]), (pt[1], pt[1]), (-2.0, 2.0), linewidth=4, color="green")
            button.label.set_text("Sample Points")
            state = "Acquisition"
            plt.draw()
        elif state == "Acquisition":
            ax.clear()
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            ax.set_zlabel("R-Factor")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_zlim(-2.0, 2.0)

            batch_num += 1
            batch_idx = batch_size * batch_num
            batch_pts = pts[:batch_idx, :]
            batch_rfactors = rfactors[:batch_idx]
            normalized_batch_rfactors = (batch_rfactors - batch_rfactors.mean()) / batch_rfactors.std(ddof=1)

            # Should I plot normalized rfactors? Normalization changes each batch
            ax.scatter(batch_pts[:, 0], batch_pts[:, 1], normalized_batch_rfactors, c="black", linewidths=5, depthshade=False)
            button.set_active(True)
            button.label.set_text("Fit Model")
            state = "Sample"
            plt.draw()

    button.on_clicked(on_clicked)
    plt.show()
#    cbar = None


##        if cbar is None:
##            cbar = fig.colorbar(surf)
##        else:
##            cbar.mappable.set_clim(vmin=mean_func.min(), vmax=mean_func.max())
##            cbar.draw_all()
#
#        input("Calculate Acquisition Function? [Enter]")
#        input("\nNext Batch? [Enter]")
#        ax.clear()

def emulate1D(pts, rfactors, batch_size, justpoints=False):
    """ Given an array of pts and their corresponding r-factors, steps through training botorch
         models using the given pts, batch_size at a time.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel("R-Factor")
    ax.set_xlim(0.0, 1.0)
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
        batch_pts = pts[:idx2, 0]
        batch_rfactors = rfactors[:idx2]
        normalized_batch_rfactors = (batch_rfactors - batch_rfactors.mean()) / batch_rfactors.std(ddof=1)
        ax.scatter(batch_pts, batch_rfactors, c="black")
        plt.draw()
        input("Fit Model? [Enter]")

        model, mll = fit_model(batch_pts, normalized_batch_rfactors, state_dict=state_dict)
        state_dict = model.state_dict()

        # Plot mean function
        model.eval()
        eval_pts = torch.linspace(0.0, 1.0, dtype=torch.float64, steps=500)[:, None].to(device=device)
        posterior = model(eval_pts)
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
        best_rfactor = np.min(normalized_batch_rfactors)
        acq = botorch.acquisition.ExpectedImprovement(model, -best_rfactor)
        acq_values = acq(eval_pts[:, :, None]).cpu().detach().numpy()
        ax.plot(eval_pts_np, acq_values, "r-")

        # I'm plotting the analytic single-point acquisition function but in reality I optimize qEI
        sampler = botorch.sampling.SobolQMCNormalSampler(num_samples=2500, resample=False)
        qacq = botorch.acquisition.qExpectedImprovement(model, -best_rfactor, sampler)
        new_pts, _ = botorch.optim.optimize_acqf(
            acq_function=qacq,
            bounds=torch.tensor([[0.0], [1.0]], device=device, dtype=torch.float64),
            q=batch_size,
            num_restarts=20,
            raw_samples=200,
            options={},
            sequential=True
        )
        new_pts = new_pts.cpu().detach().numpy()
        ax.vlines(new_pts[:, 0], 0, 1, transform=ax.get_xaxis_transform(), colors='g')


        input("\nNext Batch? [Enter]")
        ax.clear()
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel("R-Factor")
        ax.set_xlim(0.0, 1.0)
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
        print("Cannot visualize >2D data, so will drop into REPL to allow for inspection")
        emulateND(pts, rfactors, args.batch)
