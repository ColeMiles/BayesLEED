import argparse
import os
import time
import logging
import multiprocessing as mp
import multiprocessing.dummy as mpdummy

import numpy as np
import torch
import gpytorch
import botorch
import tleed

SEARCH_SPACE = np.array([-0.25, 0.25])
# True, target solution, with symmetric pts taken out
TRUE_SOL = np.array(
    [0.2200, -0.1800, 0.0000, -0.0500, 0.0900, -0.0800, -0.0100, -0.0100]
)


def create_manager(workdir):
    """ Makes a LEEDManager working in the given directory. This should be
         the only function which has hard-programmed constants.
    """
    leed_executable = "../TLEED/work/ref-calc.LaNiO3"
    rfact_executable = "../TLEED/work/rf.x"
    expdatafile = "../TLEED/work/WEXPEL"
    templatefile = "../TLEED/work/FIN"
    return tleed.LEEDManager(
       workdir,
       leed_executable,
       rfact_executable,
       expdatafile,
       templatefile
    )

# The search space is +- some small value, but the default kernel parameters
#   work best for inputs normalized to the unit cube. So, just do that mapping.
def normalize_input(disp):
    return (disp - SEARCH_SPACE[0]) / (SEARCH_SPACE[1] - SEARCH_SPACE[0])

# Just the reverse operation: Unit cube -> search space in each dimension
def denormalize_input(norm_disp):
    return ((SEARCH_SPACE[1] - SEARCH_SPACE[0]) * norm_disp) + SEARCH_SPACE[0]

def create_model(pts, rfactors, state_dict=None):
    """ Create / update a botorch model using the given points
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(pts) is not torch.Tensor:
        pts = torch.tensor(pts, device=device)
    if type(rfactors) is not torch.Tensor:
        rfactors = torch.tensor(rfactors, device=device)
    if len(rfactors.shape) < 2:
        rfactors = rfactors.unsqueeze(1)
    # Botorch assumes a maximization problem, so we will regress the
    #   negative r-factor to minimize it
    # In addition, our LEED evaluations are noiseless, so assert a
    #   noise level of zero
    model = botorch.models.FixedNoiseGP(
        pts, 
        -rfactors, 
        torch.zeros_like(rfactors)
    )

    # Set the prior for the rfactor mean to be at a reasonable level
    model.mean_module.register_parameter(
        name="constant",
        parameter=torch.nn.Parameter(
            torch.tensor([-0.7], device=device, dtype=torch.float64)
        )
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def random_unit_vecs(num_vecs, vec_dim):
    """ Samples num_vecs number of random unit vectors, in vec_dim-dimensional space
    """
    vecs = np.random.random((num_vecs, vec_dim))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def random_sample(npts):
    """ Return npts number of points, randomly sampled across the entire
         search space.
    """
    random_unit_cube = np.random.random((npts, len(TRUE_SOL)))
    return denormalize_input(random_unit_cube)


def warm_start(npts, dist):
    """ Return npts number of points in search space, sampled close to the 
         true solution. The distance from the true solution is given by dist.
    """
    # Sample npts random vectors on the unit sphere, then scale by dist
    rand_disps = dist * random_unit_vecs(npts, len(TRUE_SOL))
    sample_pts = TRUE_SOL + rand_disps
    return sample_pts

def main(workdir, ncores, nepochs, warm=None):
    num_eval = min(ncores, mp.cpu_count())
    manager = create_manager(workdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model with random points in search space to begin
    if warm is None:
        logging.info("Performing random start with {} points".format(num_eval))
        pts = random_start(num_eval)
    else:
        logging.info(
            "Performing warm start with {} points, {} from true solution".format(num_eval, warm)
        )
        pts = warm_start(num_eval, warm)
    normalized_pts = normalize_input(pts)
    rfactors = manager.batch_ref_calcs(pts)
    model, mll = create_model(normalized_pts, rfactors)

    # Find the best out of these initial trial points
    best_idx = np.argmin(rfactors)
    best_pt = pts[best_idx]
    best_rfactor = rfactors[best_idx]
    rfactor_progress = [best_rfactor]
    np.savetxt("bestpt.txt", best_pt)

    normalized_pts = torch.tensor(normalized_pts, device=device, dtype=torch.float64)
    rfactors = torch.tensor(rfactors, device=device, dtype=torch.float64)
    # Main Bayesian Optimization Loop
    for epoch in range(nepochs):
        #import ipdb
        #ipdb.set_trace()
        logging.info("Starting Epoch {}".format(epoch))
        # Fit the kernel hyperparameters
        logging.info("Fitting Kernel hyperparameters...")
        botorch.fit.fit_gpytorch_model(mll)

        sampler = botorch.sampling.SobolQMCNormalSampler(
            num_samples=1000, 
            resample=False
        )
        acquisition = botorch.acquisition.qExpectedImprovement(
            model,
            -best_rfactor,
            sampler
        )
        logging.info("Optimizing acquisition function to generate new test points...")
        new_normalized_pts, _ = botorch.optim.optimize_acqf(
            acq_function=acquisition,
            bounds=torch.tensor([[0.0] * 8, [1.0] * 8], device=device, dtype=torch.float64),
            q=num_eval,
            num_restarts=20,
            raw_samples=100,
            options={},
            sequential=True
        )
        logging.info("New test points generated")
        pts = denormalize_input(new_normalized_pts.cpu().numpy())
        new_rfactors = manager.batch_ref_calcs(pts)

        # Get the new best pt, rfactor
        best_new_idx = np.argmin(new_rfactors)
        best_new_rfactor = new_rfactors[best_new_idx]
        if best_new_rfactor < best_rfactor:
            best_rfactor = best_new_rfactor
            best_pt = pts[best_new_idx]
            np.savetxt("bestpt.txt", best_pt)
        rfactor_progress.append(best_rfactor)
        np.savetxt("rfactor_progress.txt", rfactor_progress)
        logging.info("Current best rfactor = {}".format(best_rfactor))
        
        # Update the model with the new (point, rfactor) values
        new_rfactors_tensor = torch.tensor(new_rfactors, device=device, dtype=torch.float64)
        normalized_pts = torch.cat((normalized_pts, new_normalized_pts))
        rfactors = torch.cat((rfactors, new_rfactors_tensor))
        model, mll = create_model(normalized_pts, rfactors, state_dict=model.state_dict())
        logging.info("Botorch model updated with new evaluated points")
        torch.save(model.state_dict(), "finalmodel.mdl")
        logging.info("Saved model state dictionary to finalmodel.mdl")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(message)s", 
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", 
        help="The directory where calculations will be carried out and stored."
    )
    parser.add_argument("--ncores", type=int, default=5,
        help="The number of cores to use == the number of parallel evaluations each iteration."
    )
    parser.add_argument("--nepochs", type=int, default=100,
        help="The number of epochs to run."
    )
    parser.add_argument("--warm", type=float, default=0.03,
        help="Warm start with points a given distance from the true solution"
    )
    args = parser.parse_args()

    # Check for GPU presence
    if torch.cuda.is_available():
        logging.info("Found CUDA-capable GPU")
    else:
        logging.warning("No CUDA-capable GPU found, continuing on CPU")

    # Set random seeds to get reproducible behavior for now
    np.random.seed(1234)
    torch.manual_seed(seed=1234)

    main(args.workdir, args.ncores, args.nepochs, args.warm)
