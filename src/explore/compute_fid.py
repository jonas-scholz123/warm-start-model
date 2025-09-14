# %%
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.iter import StepIterator
from mlbnb.paths import ExperimentPath

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment


def compute_fid(
    experiment: str,
    model_name: str,
    num_samples: int,
    nfe: int,
    solver: str | None,
    skip_type: str,
    warmth: float = 1.0,
    seed: int = 42,
):
    """Computes the FID score for a given experiment and model."""
    torch.manual_seed(seed)
    exp_path = Path(experiment)
    if not exp_path.exists():
        exp_path = Path("./_weights") / experiment
    if not exp_path.exists():
        exp_path = Path("./_output") / experiment

    path = ExperimentPath.from_path(exp_path)
    cfg = path.get_config()
    exp = Experiment.from_config(cfg)  # type: ignore
    model: CDNP = exp.model  # type: ignore
    cm = CheckpointManager(path)

    if "ema" in model_name:
        model = exp.ema_model.get_shadow()  # type: ignore

    _ = cm.reproduce_model(model, model_name)

    mean = cfg.data.dataset.norm_means
    std = cfg.data.dataset.norm_stds

    metric = FIDMetric(
        num_samples=num_samples,
        device="cuda",
        means=mean,
        stds=std,
        nfe=nfe,
        ode_method=solver,
        skip_type=skip_type,
        warmth=warmth,
    )

    dataloader = StepIterator(
        exp.train_loader,
        steps=num_samples // exp.train_loader.batch_size + 1,  # type: ignore
    )

    result = evaluate(
        model=model,
        dataloader=dataloader,  # type: ignore
        preprocess_fn=exp.preprocess_fn,
        metrics=[metric],
        use_tqdm=True,
    )
    return result


if __name__ == "__main__":
    experiments = [
        # "2025-07-21_22-38_playful_xenon",
        # "2025-07-23_15-24_sassy_unicorn_better_cnp",
        "new_warmth_scaling",
    ]
    nfes = [10, 12, 20, 50, 100]
    model = "latest_ema"
    solvers = ["dpm_solver_3"]
    default_skip_type = "logSNR"
    num_samples = 50_000

    all_args = []

    csv_path = "fid_results.csv"
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(
            columns=[
                "experiment",
                "model",
                "nfe",
                "solver",
                "skip_type",
                "num_samples",
                "fid",
            ]
        )

    for experiment, nfe, solver in product(experiments, nfes, solvers):
        # Log snr is bad for these very low NFEs because you need at least a few points
        # in the middle, that you don't get. Fall back to uniform.
        if nfe <= 5 or solver == "euler" or solver == "midpoint":
            skip_type = "time_uniform"
        else:
            skip_type = default_skip_type

        if nfe == 1 and solver == "midpoint":
            # Can't do midpoint with 1 step
            continue

        # Check if the current combination already exists
        exists = (
            (df["experiment"] == experiment)
            & (df["model"] == model)
            & (df["nfe"] == nfe)
            & (df["solver"] == solver)
            & (df["skip_type"] == skip_type)
            & (df["num_samples"] == num_samples)
        ).any()

        if exists:
            print("FID for this configuration already exists. Skipping.")
        else:
            result = compute_fid(
                experiment=experiment,
                model_name=model,
                num_samples=num_samples,
                nfe=nfe,
                solver=solver,
                skip_type=skip_type,
            )

            fid = result[f"fid_nfe={nfe}"]

            # Add new result and save
            new_row = {
                "experiment": experiment,
                "model": model,
                "nfe": nfe,
                "solver": solver,
                "skip_type": skip_type,
                "num_samples": num_samples,
                "fid": fid,
            }
            print(new_row)
            print(f"Computed FID: {fid}")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
