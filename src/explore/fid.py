import argparse

from explore.compute_fid import compute_fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment name or path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_ema",
        choices=["latest", "latest_ema", "best", "best_ema"],
        help="Model name to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="Number of samples for FID evaluation",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=50,
        help="Number of function evaluations for sampling.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        help="ODE method to use for sampling, including dpm_solver_2, dpm_solver_3, and all torchdiffeq methods",
    )
    parser.add_argument(
        "--skip-type",
        type=str,
        default="logSNR",
        choices=["logSNR", "time_uniform", "time_quadratic", "edm"],
        help="Skip type for ODE sampling",
    )
    parser.add_argument(
        "--warmth",
        type=float,
        default=1.0,
        help="Initial warmth in warm start diffusion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--ctx-frac",
        type=float,
        default=None,
        help="Fraction of context images to use",
    )
    args = parser.parse_args()

    result = compute_fid(
        experiment=args.experiment,
        model_name=args.model,
        num_samples=args.num_samples,
        nfe=args.nfe,
        solver=args.solver,
        skip_type=args.skip_type,
        warmth=args.warmth,
        seed=args.seed,
        ctx_frac=args.ctx_frac,
    )

    print("Args:", args)
    print(result)
