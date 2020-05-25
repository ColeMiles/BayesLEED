import argparse
import os
import time
import logging
import multiprocessing as mp
import gc

import numpy as np
import torch
import gpytorch
import botorch
import tleed
import problems


def append_arrays_to_file(filename, pts, rfactors):
    assert len(pts.shape) == 2, "append_array_to_file only handles dim-2 pts array"
    assert len(pts) == len(rfactors), "do not have same number of pts and rfactors"
    row_format_string = "{:<8.4f}" * (pts.shape[1] + 1)
    with open(filename, "a") as f:
        for pt, rfactor in zip(pts, rfactors):
            f.write(row_format_string.format(*pt, rfactor) + "\n")


def create_manager(workdir, executable='ref-calc.LaNiO3'):
    """ Makes a LEEDManager working in the given directory
    """
    leed_executable = os.path.join(workdir, executable)
    rfact_executable = os.path.join(workdir, "rf.x")
    expdatafile = os.path.join(workdir, "WEXPEL")
    templatefile = os.path.join(workdir, "FIN")
    return tleed.LEEDManager(
       workdir,
       leed_executable,
       rfact_executable,
       expdatafile,
       templatefile
    )


def create_model(pts, targets, state_dict=None):
    """ Create / update a botorch model using the given points
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(pts) is not torch.Tensor:
        pts = torch.tensor(pts, device=device)
    if type(targets) is not torch.Tensor:
        targets = torch.tensor(targets, device=device)
    if len(targets.shape) < 2:
        targets = targets.unsqueeze(1)
    # Botorch assumes a maximization problem, so we will regress the
    #   negative r-factor to minimize it
    # In addition, our LEED evaluations are noiseless, so assert a
    #   noise level of zero
    model = botorch.models.FixedNoiseGP(
        pts,
        targets,
        torch.zeros_like(targets)
    )
    # model = botorch.models.SingleTaskGP(
    #     pts,
    #     -targets,
    # )

    # Set the prior for the rfactor mean to be at a reasonable level
    model.mean_module.load_state_dict(
        {'constant': torch.tensor([-0.65], device=device, dtype=torch.float64)}
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def optimize_EI(obs_pts, obs_objectives, q=5, state_dict=None, save_model=None, device="cuda"):
    """ Sample q more pts from the search space, by optimizing
         expected improvement using a model and current observations.
        Note: Assumes, like botorch, a maximization problem!
    """
    model, mll = create_model(obs_pts, obs_objectives, state_dict=state_dict)
    logging.info("Botorch model updated with new evaluated points")

    logging.info("Fitting Kernel hyperparameters...")
    model.train()
    botorch.fit.fit_gpytorch_model(mll)
    model.eval()
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
        logging.info("Saved model state dict to " + str(save_model))

    num_params = obs_pts.shape[1]
    sampler = botorch.sampling.SobolQMCNormalSampler(
        num_samples=4096,
        resample=False
    )

    best_objective = torch.max(obs_objectives).item()
    acquisition = botorch.acquisition.qExpectedImprovement(
        model,
        best_objective,
        sampler
    )
    # acquisition = botorch.acquisition.qKnowledgeGradient(
    #     model,
    #     num_fantasies=32,
    #     sampler=sampler,
    # )
    logging.info("Optimizing acquisition function to generate new test points...")

    # The next step uses a lot of GPU memory. Free everything we can
    gc.collect()

    new_normalized_pts, _ = botorch.optim.optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0] * num_params, [1.0] * num_params], device=device, dtype=torch.float64),
        q=q,
        num_restarts=50,
        raw_samples=8192,
        options={},
        sequential=True
    )

    return new_normalized_pts


def random_sample(q, num_params, device="cuda"):
    """ Sample q more pts randomly from the search space
    """
    return torch.rand(q, num_params, dtype=torch.float64, device=device)


def main(leed_executable, problem, ncores, nepochs,
         warm=None, seed=None, start_pts_file=None, early_stop=None, random=False):
    workdir, executable = os.path.split(leed_executable)
    tested_filename = os.path.join(workdir, "tested_point.txt")
    model_filename = os.path.join(workdir, "finalmodel.pt")
    rfactor_filename = os.path.join(workdir, "rfactorprogress.txt")

    num_eval = min(ncores, mp.cpu_count())
    manager = create_manager(workdir, executable=executable)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    search_problem = problems.problems[problem]
    num_params = search_problem.num_params

    if start_pts_file is not None:
        logging.info("Loading points from file: {}".format(start_pts_file))
        data = np.loadtxt(start_pts_file, skiprows=1)
        start_pts = data[:, :-1]
        rfactors = data[:, -1]
    else:
        # Initialize a model with random points in search space to begin
        if warm is None:
            logging.info("Performing random start with {} points".format(num_eval))
            start_pts, random_structs = search_problem.random_points(num_eval)
            # Replace one of the random pts with the "ideal" structure (no perturbations)
            start_pts[0] = search_problem.to_normalized(search_problem.atomic_structure)
            random_structs[0] = search_problem.atomic_structure
        else:
            print("Warm start is unimplemented as of right now")
            return
            logging.info(
                "Performing warm start with {} points, {} from true solution".format(num_eval, warm)
            )
        rfactors = manager.batch_ref_calcs(random_structs)

    # Normalize rfactors to zero mean, unit variance
    normalized_rfactors = (rfactors - rfactors.mean()) / rfactors.std(ddof=1)
    model, mll = create_model(start_pts, normalized_rfactors)

    # Find the best out of these initial trial points
    best_idx = np.argmin(rfactors)
    best_pt = start_pts[best_idx]
    best_rfactor = rfactors[best_idx]
    rfactor_progress = [best_rfactor]

    # Create header for tested points file
    with open(tested_filename, "w") as ptfile:
        header_str = "DISPLACEMENT" + " " * 52 + "RFACTOR"
        if seed is not None:
            header_str += "    SEED: " + str(seed)
        header_str += "\n"
        ptfile.write(header_str)
    append_arrays_to_file(tested_filename, start_pts, rfactors)
    logging.info("Best r-factor from initial set: {:.4f}".format(best_rfactor))

    normalized_pts = torch.tensor(start_pts, device=device, dtype=torch.float64)
    rfactors = torch.tensor(rfactors, device=device, dtype=torch.float64)
    normalized_rfactors = torch.tensor(normalized_rfactors, device=device, dtype=torch.float64)

    # Main Bayesian Optimization Loop
    for epoch in range(nepochs):
        logging.info("Starting Epoch {}".format(epoch))

        # Sample new points from the search space
        if not random:
            new_normalized_pts = optimize_EI(
                normalized_pts, -normalized_rfactors,
                q=ncores, state_dict=model.state_dict(), save_model=model_filename, device=device,
            )
        else:
            new_normalized_pts = random_sample(ncores, num_params, device=device)

        logging.info("New test points generated. Running TLEED.")

        # Perform the reference calculations at those points
        new_normalized_pts_np = new_normalized_pts.cpu().numpy()
        structs = search_problem.to_structures(new_normalized_pts_np)
        new_rfactors = manager.batch_ref_calcs(structs)

        # Get the new best pt, rfactor
        best_new_idx = np.argmin(new_rfactors)
        best_new_rfactor = new_rfactors[best_new_idx]
        if best_new_rfactor < best_rfactor:
            best_rfactor = best_new_rfactor
            best_pt = new_normalized_pts_np[best_new_idx]
        append_arrays_to_file(tested_filename, new_normalized_pts_np, new_rfactors)
        rfactor_progress.append(best_rfactor)
        np.savetxt(rfactor_filename, rfactor_progress)
        logging.info("Current best rfactor = {}".format(best_rfactor))

        # Early stop if we get a "good enough" solution
        if early_stop is not None and best_rfactor < early_stop:
            return model, normalized_pts, rfactors

        # Update the model with the new (point, rfactor) values
        new_rfactors_tensor = torch.tensor(new_rfactors, device=device, dtype=torch.float64)
        normalized_pts = torch.cat((normalized_pts, new_normalized_pts))
        rfactors = torch.cat((rfactors, new_rfactors_tensor))
        normalized_rfactors = (rfactors - rfactors.mean()) / rfactors.std()

    return model, normalized_pts, rfactors


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] %(message)s", 
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("leed_executable", 
        help="Path to LEED executable. Directory containing it treated as work directory."
    )
    parser.add_argument("--problem", type=str, default="LANIO3",
        help="Name of problem to run (from problems.py). [Options: `LANIO3`, `FESE_20UC`]"
    )
    parser.add_argument("--ncores", type=int, default=5,
        help="The number of cores to use == the number of parallel evaluations each iteration."
    )
    parser.add_argument("--nepochs", type=int, default=25,
        help="The number of epochs to run. [Default = 25]"
    )
    parser.add_argument("--warm", type=float,
        help="Warm start with points a given distance from the true solution (in normalized space). (Unimplemented)."
    )
    parser.add_argument("--seed", type=int,
        help="Set the seed for the RNGs"
    )
    parser.add_argument('--start-pts', type=str, default=None,
        help="Given a file of tested points, continues from there"
    )
    parser.add_argument('--random', action='store_true',
        help="If set, performs random search rather than Bayesian Optimization"
    )
    args = parser.parse_args()

    # Check for GPU presence
    if torch.cuda.is_available():
        logging.info("Found CUDA-capable GPU")
    else:
        logging.warning("No CUDA-capable GPU found, continuing on CPU")

    # Set random seeds to get reproducible behavior
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(seed=args.seed)

    main(args.leed_executable, args.problem, args.ncores, args.nepochs, args.warm,
         seed=args.seed, start_pts_file=args.start_pts, random=args.random)
