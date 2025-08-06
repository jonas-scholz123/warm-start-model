import argparse
from pathlib import Path

from mlbnb.checkpoint import CheckpointManager
from mlbnb.iter import StepIterator
from mlbnb.paths import ExperimentPath

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment

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
    "--num_samples",
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
args = parser.parse_args()

exp_path = Path(args.experiment)
if not exp_path.exists():
    exp_path = Path("./_weights") / args.experiment
if not exp_path.exists():
    exp_path = Path("./_output") / args.experiment

path = ExperimentPath.from_path(exp_path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)  # ty: ignore
model: CDNP = exp.model  # ty: ignore
cm = CheckpointManager(path)

if "ema" in args.model:
    model = exp.ema_model.get_shadow()  # ty: ignore
else:
    model = model

_ = cm.reproduce_model(model, args.model)

mean = cfg.data.dataset.norm_means
std = cfg.data.dataset.norm_stds

metric = FIDMetric(
    num_samples=args.num_samples,
    device="cuda",
    means=mean,
    stds=std,
    nfe=args.nfe,
    ode_method=args.solver,
    skip_type=args.skip_type,
)

dataloader = StepIterator(
    exp.train_loader, steps=args.num_samples // exp.train_loader.batch_size + 1
)

result = evaluate(
    model=model,
    dataloader=dataloader,  # ty: ignore
    preprocess_fn=exp.preprocess_fn,
    metrics=[metric],
    use_tqdm=True,
)

print("Args:", args)
print(result)
