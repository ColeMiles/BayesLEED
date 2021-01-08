import argparse
import os
import sys
import logging
import multiprocessing as mp
import gc
import time
import re
from typing import Tuple, List

import numpy as np
import torch
import gpytorch
import botorch

from pyleed import problems, tleed
from pyleed.structure import AtomicStructure
from pyleed.tleed import RefCalc, parse_ref_calc, CalcState
from pyleed.curves import write_curves


def pretty_print_args(args):
    logging.info("Running BayesSearch with the following parameters:")
    for arg in vars(args):
        logging.info("| {} {}".format(arg, getattr(args, arg) or 'None'))
    logging.info("----------------------------------")


def append_arrays_to_file(filename, pts, rfactors, labels=None):
    assert len(pts) == len(rfactors), "do not have same number of pts and rfactors"
    if labels is None:
        labels = ("" for _ in pts)
    row_format_string = "{:<8.4f}" * (len(pts[0]) + 1)
    label_format_string = "{:<7s}"
    with open(filename, "a") as f:
        for pt, rfactor, label in zip(pts, rfactors, labels):
            f.write(label_format_string.format(label))
            f.write(row_format_string.format(*pt, rfactor) + "\n")


def create_manager(workdir, tleed_dir, exp_curves, beamlist_file, phaseshift_file, lmax, num_el,
                   executable='ref-calc.LaNiO3'):
    """ Makes a LEEDManager working in the given directory, assuming default names for files.
    """
    beamlist = tleed.parse_beamlist(beamlist_file)
    phaseshifts = tleed.parse_phaseshifts(phaseshift_file, num_el, lmax)
    exp_curves = tleed.parse_ivcurves(exp_curves, format='TLEED')
    exp_curves = exp_curves.smooth(2)
    return tleed.LEEDManager(workdir, tleed_dir, exp_curves,
                             phaseshifts, beamlist)


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
    # model = botorch.models.FixedNoiseGP(
    #     pts,
    #     targets,
    #     torch.zeros_like(targets)
    # )
    model = botorch.models.SingleTaskGP(
        pts,
        targets,
    )

    # Set the prior for the rfactor mean to be at a reasonable level
    # model.mean_module.load_state_dict(
    #     {'constant': torch.tensor([-0.65], device=device, dtype=torch.float64)}
    # )

    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def optimize_EI(obs_pts, obs_objectives, q=5, pending_pts=None,
                state_dict=None, save_model=None, device="cuda"):
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

    best_objective = torch.max(obs_objectives)
    acquisition = botorch.acquisition.qExpectedImprovement(
        model,
        best_objective,
        sampler,
        X_pending=pending_pts
    )
    # acquisition = botorch.acquisition.qKnowledgeGradient(
    #     model,
    #     num_fantasies=32,
    #     sampler=sampler,
    #     X_pending=pending_pts
    # )
    logging.info("Optimizing acquisition function to generate new test points...")

    # The next step uses a lot of GPU memory. Free everything we can
    gc.collect()

    new_normalized_pts, _ = botorch.optim.optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.0] * num_params, [1.0] * num_params], device=device, dtype=torch.float64),
        q=q,
        num_restarts=50,
        raw_samples=4096,
        options={},
        sequential=True
    )

    return new_normalized_pts


def random_sample(q, num_params, device="cuda", **kwargs):
    """ Sample q more pts randomly from the search space
    """
    return torch.rand(q, num_params, dtype=torch.float64, device=device)


def acquire_sample_points(
        pts, targets, num_sample, pending_pts=None,
        method='bayes', tleed_radius=0.0, **kwargs
) -> np.ndarray:
    """ Given historically sampled points and target values (pts, targets), as well as the
         number of desired new points (num_sample), performs some sampling routine (either
         Bayesian optimization or random sampling) to produce new points to sample.
    """
    if method == 'random':
        new_normalized_pts = random_sample(num_sample, pts.shape[1], **kwargs)
    elif method == 'bayes':
        new_normalized_pts = optimize_EI(
            pts, -targets, q=num_sample, pending_pts=pending_pts, **kwargs
        )
    else:
        raise ValueError("`method` provided to acquire_sample_points is not valid.")

    logging.info(str(num_sample) + " new sample points generated.")

    return new_normalized_pts.cpu().numpy()


def main(workdir, tleed_dir, phaseshifts, lmax, num_el, exp_curves, beamlist, problem, ncores, ncalcs,
         tleed_radius=0.0, warm=None, seed=None, start_pts_file=None, detect_existing_calcs=None,
         early_stop=None, random=False, save_curves=None):
    if save_curves is None:
        save_curves = os.path.join(workdir, 'bestcurves.data')

    tested_filename = os.path.join(workdir, "tested_point.txt")
    model_filename = os.path.join(workdir, "finalmodel.pt")
    rfactor_filename = os.path.join(workdir, "rfactorprogress.txt")

    num_eval = min(ncores, mp.cpu_count())
    manager = create_manager(workdir, tleed_dir, exp_curves, beamlist, phaseshifts, lmax, num_el)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    search_problem = problems.problems[problem]
    num_params = search_problem.num_params

    # TODO: add this to SearchSpace definition
    delta_search_dims = problems.FESE_DELTA_SEARCHDIMS

    if detect_existing_calcs:
        logging.info("Loading points from reference calculations in directory {}".format(workdir))
        filename_pat = r"ref-calc\d+"
        calcs = [
            parse_ref_calc(os.path.join(workdir, f, "FIN")) for f in os.listdir(workdir)
            if re.match(filename_pat, f) is not None
        ]
        # Keep only the calcs with a discovered fd.out
        calcs = [c for c in calcs if c.state == CalcState.COMPLETED]
        start_pts = search_problem.to_normalized([calc.struct for calc in calcs])
        rfactors = np.array([calc.rfactor(manager.exp_curves) for calc in calcs])
        manager.calc_number = len(start_pts)
        num_ref_calcs = len(start_pts)
        num_delta_calcs = 0
    elif start_pts_file is not None:
        # TODO: Needs more logic to parse refcalcs/deltacalcs
        logging.info("Loading points from file: {}".format(start_pts_file))
        data = np.loadtxt(start_pts_file, skiprows=1)
        start_pts = data[:, :-1]
        rfactors = data[:, -1]
        num_ref_calcs = len(start_pts)
        num_delta_calcs = 0
    else:
        if warm:
            print("Warm start is unimplemented as of right now")
        # Initialize a model with random points in search space to begin
        logging.info("Performing random start with {} points".format(num_eval))
        start_pts, random_structs = search_problem.random_points(num_eval)
        # Replace one of the random pts with the "ideal" structure (no perturbations)
        start_pts[0] = search_problem.to_normalized(search_problem.atomic_structure)
        random_structs[0] = search_problem.atomic_structure
        ref_rfactors, delta_structs, delta_rfactors, best_curves = manager.batch_ref_calc_local_searches(
            random_structs, delta_search_dims
        )
        rfactors = np.array(ref_rfactors + delta_rfactors)
        delta_pts = search_problem.to_normalized(delta_structs)
        start_pts = np.concatenate((start_pts, delta_pts), axis=0)
        num_ref_calcs = num_eval
        num_delta_calcs = num_eval
        write_curves(save_curves, best_curves, format='TLEED')

    # Normalize rfactors to zero mean, unit variance -- maybe don't do this, pick approx values
    normalized_rfactors = (rfactors - rfactors.mean()) / rfactors.std(ddof=1)
    model, mll = create_model(start_pts, normalized_rfactors)

    # Find the best out of these initial trial points
    best_idx = np.argmin(rfactors)
    best_rfactor = rfactors[best_idx]
    rfactor_progress = [best_rfactor]
    ncalcs_completed = len(start_pts) // 2

    # Create header for tested points file
    with open(tested_filename, "w") as ptfile:
        header_str = "LABEL  DISPLACEMENT" + " " * 52 + "RFACTOR"
        if seed is not None:
            header_str += "    SEED: " + str(seed)
        header_str += "\n"
        ptfile.write(header_str)
    labels = ["R"+str(i) for i in range(num_ref_calcs)]
    labels += ["D"+str(i) for i in range(num_delta_calcs)]
    append_arrays_to_file(tested_filename, start_pts, rfactors, labels)
    logging.info("Best r-factor from initial set: {:.4f}".format(best_rfactor))

    normalized_pts = torch.tensor(start_pts, device=device, dtype=torch.float64)
    rfactors = torch.tensor(rfactors, device=device, dtype=torch.float64)
    normalized_rfactors = torch.tensor(normalized_rfactors, device=device, dtype=torch.float64)

    # Main Bayesian Optimization Loop
    num_to_opt = ncores
    while ncalcs_completed < ncalcs:
        # Sample new points from the search space
        new_normalized_pts = acquire_sample_points(
            normalized_pts, normalized_rfactors, num_to_opt,
            method='random' if random else 'bayes', tleed_radius=tleed_radius,
            state_dict=model.state_dict(), save_model=model_filename, device=device
        )

        trial_structs = search_problem.to_structures(new_normalized_pts)

        ref_rfactors, delta_structs, delta_rfactors, best_curves = manager.batch_ref_calc_local_searches(
            trial_structs,
            delta_search_dims
        )
        new_rfactors = ref_rfactors + delta_rfactors
        delta_pts = search_problem.to_normalized(delta_structs)
        new_normalized_pts = np.concatenate((new_normalized_pts, delta_pts), axis=0)

        # Get the new best pt, rfactor
        best_new_idx = np.argmin(new_rfactors).item()
        best_new_rfactor = new_rfactors[best_new_idx]
        if best_new_rfactor < best_rfactor:
            best_rfactor = best_new_rfactor
            best_pt = new_normalized_pts[best_new_idx]
            # Save out the best curves
            write_curves(save_curves, best_curves, format='TLEED')

        # Log some output
        labels = ["R" + str(i) for i in range(num_ref_calcs, num_ref_calcs + num_to_opt)]
        labels += ["D" + str(i) for i in range(num_delta_calcs, num_delta_calcs + num_to_opt)]
        append_arrays_to_file(tested_filename, new_normalized_pts, new_rfactors, labels)
        rfactor_progress.append(best_rfactor)
        np.savetxt(rfactor_filename, rfactor_progress)
        num_ref_calcs += len(ref_rfactors)
        num_delta_calcs += len(delta_rfactors)

        logging.info("{} calculations completed. New best rfactor = {}".format(
            len(new_normalized_pts), best_rfactor
        ))

        # Update the global tensors with the new (point, rfactor) values
        new_normalized_pts = torch.tensor(new_normalized_pts, device=device)
        normalized_pts = torch.cat((normalized_pts, new_normalized_pts))
        new_rfactors = torch.tensor(new_rfactors, device=device, dtype=torch.float64)
        rfactors = torch.cat((rfactors, new_rfactors))
        normalized_rfactors = (rfactors - rfactors.mean()) / rfactors.std()

        ncalcs_completed += num_to_opt

        # Early stop if we get a "good enough" solution
        if early_stop is not None and best_rfactor < early_stop:
            return model, normalized_pts, rfactors

    return model, normalized_pts, rfactors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=str,
        help="Directory to carry out all calculations in."
    )
    parser.add_argument("-p", "--phaseshifts", type=str,
        default="/home/cole/ProgScratch/BayesLEED/TLEED/phaseshifts/FeSeBulk.eight.phase",
        help="Path to phaseshift file to use for calculations."
    )
    parser.add_argument("--num-el", type=int, default=2,
        help="Number of elements present in phaseshift file. [Default = 2]"
    )
    parser.add_argument("--lmax", type=int, default=8,
        help="Highest angular momentum number present in phaseshift file. [Default = 8]"
    )
    parser.add_argument("-t", "--tleed", type=str,
        default="/home/cole/ProgScratch/BayesLEED/TLEED/",
        help="Path to the base directory of the TLEED program."
    )
    parser.add_argument("--problem", type=str, default="FESE_20UC",
        help="Name of problem to run (from problems.py)."
    )
    # parser.add_argument("-b", "--beaminfo", type=str, default="FESE_TRIM",
    #     help="Name of a beam set descriptor (from problems.py)."
    # )
    parser.add_argument("-b", "--beaminfo", type=str,
        default="/home/cole/ProgScratch/BayesLEED/TLEED/exp-data/Data.TrimmedFeSe-20uc",
        help="Path to experimental data, formatted in the TLEED format.",
    )
    parser.add_argument('-bl', '--beamlist', type=str,
        default="/home/cole/ProgScratch/BayesLEED/TLEED/beamlists/NBLIST.FeSe-1x1",
        help="Path to a beamlist file to use for reference calculations."
    )
    parser.add_argument("-n", "--ncores", type=int, default=8,
        help="The number of cores to use == the number of parallel evaluations each iteration. "
             "[Default = 8]"
    )
    parser.add_argument("-N", "--num-calcs", type=int, default=500,
        help="The number of total calculations (reference or TLEED) to run. [Default = 500]"
    )
    parser.add_argument("-r", "--radius-tleed", type=float, default=0.0,
        help="If a trial point falls within this radius of an existing reference calc, perform"
             "a TensorLEED calculation instead of a reference calculation. "
             "By default, set to 0.0 so that no TLEED calcs happen."
             "If you make this parameter nonzero, make sure your search problem does not"
             " search over lattice params, as we cannot perturb these in TLEED."
    )
    parser.add_argument("--warm", type=float,
        help="Warm start with points a given distance from the true solution (in normalized space)."
             "(Unimplemented)."
    )
    parser.add_argument("-s", "--seed", type=int,
        help="Set the seed for the RNGs."
    )
    parser.add_argument('--start-pts', type=str, default=None,
        help="Given a file of tested points, continues from there."
    )
    parser.add_argument('--detect-calcs', action='store_true', default=None,
        help="Attempts to detect exisiting finished reference calculations in the targeted work"
             " directory named ref-calc{i}. Overrides --start-pts."
    )
    parser.add_argument('--random', action='store_true',
        help="If set, performs random search rather than Bayesian Optimization."
    )
    parser.add_argument('--log', type=str, default=None,
        help="Name of outputted log file. [Default: workdir/search.log]"
    )
    parser.add_argument('--save-curves', type=str, default=None,
        help="Name of file to store the best found I-V curves."
             " [Default: workdir/bestcurves.data]"
    )
    args = parser.parse_args()

    # Set default logging locations
    if args.log is None:
        args.log = os.path.join(args.workdir, 'search.log')
    # Set up logger
    logging.basicConfig(
        format="[%(asctime)s] %(message)s",
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set random seeds to get reproducible behavior
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(seed=args.seed)
    else:
        # Is there a better way?
        seed = np.random.get_state()[1][0]
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed=args.seed)

    pretty_print_args(args)

    # Check for GPU presence
    if torch.cuda.is_available():
        logging.info("Found CUDA-capable GPU.")
    else:
        logging.info("No CUDA-capable GPU found, continuing on CPU.")

    # TODO: Do something about this. Config file / class?
    main(args.workdir, args.tleed, args.phaseshifts, args.lmax, args.num_el, args.beaminfo, args.beamlist,
         args.problem, args.ncores, args.num_calcs, tleed_radius=args.radius_tleed, seed=args.seed,
         start_pts_file=args.start_pts, detect_existing_calcs=args.detect_calcs, random=args.random,
         save_curves=args.save_curves)
