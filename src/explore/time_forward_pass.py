from click.core import F
import argparse
import time
import numpy as np
from pathlib import Path

import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

from cdnp.util.instantiate import Experiment
from config.config import Config
from cdnp.model.cnp import CNP
from cdnp.model.warm_start_diffusion import WarmStartDiffusion
from cdnp.model.flow_matching.flow_matching import FlowMatching


@torch.no_grad()
def time_forward_pass(
    experiment: str,
    num_passes: int,
    num_repeats: int,
    batch_size: int,
    device: str = "cuda",
):
    exp_path = Path(experiment)
    if not exp_path.exists():
        exp_path = Path("./_weights") / experiment
    if not exp_path.exists():
        exp_path = Path("./_output") / experiment

    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path {experiment} not found.")

    path = ExperimentPath.from_path(exp_path)
    cfg: Config = path.get_config()  # type: ignore
    cfg.data.trainloader.batch_size = batch_size

    # Minimal config for loading
    cfg.data.trainloader.num_workers = 0
    cfg.data.trainloader.persistent_workers = False
    cfg.data.trainloader.prefetch_factor = None

    exp = Experiment.from_config(cfg)
    model = exp.model
    cm = CheckpointManager(path)

    model.to(device)
    model.eval()

    # Get a single batch to use for timing
    batch = next(iter(exp.train_loader))
    ctx, _ = exp.preprocess_fn(batch)
    ctx = ctx.to(device)

    print(f"Running {num_passes} forward passes at batch size {batch_size}")
    print(ctx.image_ctx.shape)


    # Warm up:
    if device == "cuda":
        torch.cuda.synchronize()

    if isinstance(model, CNP):
        print("Detected CNP model.")
        for _ in range(5):
            _ = model.sample(ctx)
    elif isinstance(model, WarmStartDiffusion) or isinstance(model, FlowMatching):
        print("Detected generative model.")
        _ = model.sample(ctx, nfe=10, ode_method="euler", num_samples=batch_size)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    times = []
    for _ in range(num_repeats):
        if isinstance(model, CNP):
            start_time = time.perf_counter()
            for _ in range(num_passes):
                _ = model.sample(ctx)

        if isinstance(model, WarmStartDiffusion) or isinstance(model, FlowMatching):
            start_time = time.perf_counter()
            _ = model.sample(ctx, nfe=num_passes, ode_method="euler", num_samples=batch_size)

        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_passes / batch_size
        avg_time_ms = avg_time * 1000
        print(f"Avg time per fwd: {avg_time_ms:.4f} ms")
        times.append(avg_time_ms)

    print(times)
    times = np.array(times)

    two_std = 2 * times.std()
    print(
        f"Final result over {args.num_repeats} runs: "
        f"{times.mean():.4f} ms ± {two_std:.4f} ms per fwd"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time the forward pass of a model.")
    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment name or path"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of forward passes to run",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=5,
        help="Number of times to repeat the timing (for stds)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the timing on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size to use for timing",
    )
    args = parser.parse_args()

    avg_time_ms = time_forward_pass(
        experiment=args.experiment,
        num_passes=args.n,
        num_repeats=args.num_repeats,
        device=args.device,
        batch_size=args.batch_size,
    )
