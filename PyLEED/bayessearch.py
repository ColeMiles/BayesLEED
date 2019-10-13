import argparse
import os
import time
import multiprocessing as mp
import multiprocessing.dummy as mpdummy

import numpy as np
import torch
import botorch
import tleed


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

# The search space is [-0.2, 0.2] for each atom, but the default kernel parameters
#   work best for inputs normalized to the unit cube. So, just do that mapping.
def normalize_input(disp):
    return (disp + 0.2) * 2.5

# Just the reverse operation
def denormalize_input(norm_disp):
    return (0.4 * norm_disp) - 0.2

def initialize_model(manager, num_pts):
    """ Evaluates LEED at num_pts randomly sampled configurations, then
         creates a botorch model with this initial data
    """
    # For now, deterministically seed the rng for reproducable behavior
    np.random.seed(12345)
    initial_pts = np.random.rand(num_pts, 8)
    denormalized_pts = denormalize_input(initial_pts)

    # Create and start the threads to run each calculation
    print("Evaluating reference calculations on {} initial points...".format(num_pts))
    init_time = time.time()

    num_threads = min(num_pts, mp.cpu_count())
    with mp.Pool(processes=num_threads) as pool:
        rfactors = pool.starmap(
            manager.ref_calc,
            zip(denormalized_pts, range(num_pts)), 
            chunksize=1
        )

    final_time = time.time()
    print("Initial reference calculations completed in {:.1f}s".format(final_time - init_time))
    print("Creating botorch model...")
    model = botorch.models.SingleTaskGP(torch.Tensor(initial_pts), torch.Tensor(rfactors))
    return model
    
# Find out what weights means?
# Outputs a single-output posterior with mean
#   weights.T @ mu and variance weights.T @ sigma @ weights
#objective = botorch.acquisition.objective.ScalarizedObjective([1.0])
#acquisition = botorch.acquisition.ExpectedImprovement(
#    model, best_f, objective (? optional), maximize=True
#)


def main(workdir, ninit):
    manager = create_manager(workdir)
    model = initialize_model(manager, ninit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", 
        help="The directory where calculations will be carried out and stored."
    )
    parser.add_argument("--ninit", type=int, default=5,
        help="The number of initial random points to sample"
    )
    args = parser.parse_args()

    main(args.workdir, args.ninit)
