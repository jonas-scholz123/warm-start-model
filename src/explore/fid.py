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
    "--steps",
    type=int,
    default=50,
    help="Number of ODE steps for sampling",
)
args = parser.parse_args()

exp_path = Path(args.experiment)
if not exp_path.exists():
    exp_path = Path("./_weights") / args.experiment
if not exp_path.exists():
    exp_path = Path("./_output") / args.experiment

path = ExperimentPath.from_path(exp_path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)
model: CDNP = exp.model
cm = CheckpointManager(path)

if "ema" in args.model:
    model = exp.ema_model.get_shadow()
else:
    model = model

_ = cm.reproduce_model(model, args.model)

mean = cfg.data.dataset.norm_means
std = cfg.data.dataset.norm_stds

metric = FIDMetric(
    num_samples=args.num_samples, device="cuda", means=mean, stds=std, nfe=args.steps
)

if len(exp.val_loader) < args.num_samples:
    print(
        f"Validation set is smaller than {args.num_samples} samples, using the training set for FID evaluation."
    )
    dataloader = exp.train_loader
else:
    dataloader = exp.val_loader

dataloader = StepIterator(
    dataloader, steps=args.num_samples // dataloader.batch_size + 1
)

result = evaluate(
    model=model,
    dataloader=dataloader,
    preprocess_fn=exp.preprocess_fn,
    metrics=[metric],
    use_tqdm=True,
)
print(result)
