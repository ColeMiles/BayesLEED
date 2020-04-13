import argparse
import os
import time
import logging
import multiprocessing as mp

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
            torch.tensor([-0.65], device=device, dtype=torch.float64)
        )
    )

    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def main(leed_executable, problem, ncores, nepochs, warm=None, seed=None):
    workdir, executable = os.path.split(leed_executable)
    tested_filename = os.path.join(workdir, "tested_point.txt")
    model_filename = os.path.join(workdir, "finalmodel.pt")
    rfactor_filename = os.path.join(workdir, "rfactorprogress.txt")

    num_eval = min(ncores, mp.cpu_count())
    manager = create_manager(workdir, executable=executable)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    search_problem = problems.problems[problem]
    num_params = search_problem.num_params

    # Initialize a model with random points in search space to begin
    if warm is None:
        logging.info("Performing random start with {} points".format(num_eval))
        random_pts, random_structs = search_problem.random_points(num_eval)
    else:
        print("Warm start is unimplemented as of right now")
        return
        logging.info(
            "Performing warm start with {} points, {} from true solution".format(num_eval, warm)
        )

    rfactors = manager.batch_ref_calcs(random_structs)
    model, mll = create_model(random_pts, rfactors)

    # Find the best out of these initial trial points
    best_idx = np.argmin(rfactors)
    best_pt = random_pts[best_idx]
    best_rfactor = rfactors[best_idx]
    rfactor_progress = [best_rfactor]
    with open(tested_filename, "w") as ptfile:
        header_str = "DISPLACEMENT" + " " * 52 + "RFACTOR"
        if seed is not None:
            header_str += "    SEED: " + str(seed)
        header_str += "\n"
        ptfile.write(header_str)
    append_arrays_to_file(tested_filename, random_pts, rfactors)
    logging.info("Best r-factor from initial set: {:.4f}".format(best_rfactor))

    normalized_pts = torch.tensor(random_pts, device=device, dtype=torch.float64)
    rfactors = torch.tensor(rfactors, device=device, dtype=torch.float64)
    # Main Bayesian Optimization Loop
    for epoch in range(nepochs):
        logging.info("Starting Epoch {}".format(epoch))
        # Fit the kernel hyperparameters
        logging.info("Fitting Kernel hyperparameters...")
        botorch.fit.fit_gpytorch_model(mll)
        torch.save(model.state_dict(), model_filename)
        logging.info("Saved model state dict to " + str(model_filename))

        sampler = botorch.sampling.SobolQMCNormalSampler(
            num_samples=2500, 
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
            bounds=torch.tensor([[0.0] * num_params, [1.0] * num_params], device=device, dtype=torch.float64),
            q=num_eval,
            num_restarts=20,
            raw_samples=200,
            options={},
            sequential=True
        )
        logging.info("New test points generated")
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
        
        # Update the model with the new (point, rfactor) values
        new_rfactors_tensor = torch.tensor(new_rfactors, device=device, dtype=torch.float64)
        normalized_pts = torch.cat((normalized_pts, new_normalized_pts))
        rfactors = torch.cat((rfactors, new_rfactors_tensor))
        model, mll = create_model(normalized_pts, rfactors, state_dict=model.state_dict())
        logging.info("Botorch model updated with new evaluated points")

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
        help="Name of problem to run (from problems.py)"
    )
    parser.add_argument("--ncores", type=int, default=5,
        help="The number of cores to use == the number of parallel evaluations each iteration."
    )
    parser.add_argument("--nepochs", type=int, default=25,
        help="The number of epochs to run."
    )
    parser.add_argument("--warm", type=float,
        help="Warm start with points a given distance from the true solution (in normalized space). (Unimplemented)."
    )
    parser.add_argument("--seed", type=int,
        help="Set the seed for the RNGs"
    )
    parser.add_argument("--restrict", nargs="+", type=int,
        help="Restricts the search space to include only the coordinates provided. (Unimplemented)."
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

    if args.restrict is not None and 1 <= len(args.restrict) <= 7:
        print("Restricted problem not implemented yet, just running full problem")

    main(args.leed_executable, args.problem, args.ncores, args.nepochs, args.warm, seed=args.seed)
