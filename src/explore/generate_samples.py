import argparse
from pathlib import Path

import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.iter import StepIterator
from mlbnb.paths import ExperimentPath
from torchvision.utils import save_image
from tqdm import tqdm

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
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

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
mean = torch.tensor(mean, device=device)[None, :, None, None]
std = torch.tensor(std, device=device)[None, :, None, None]


if len(exp.val_loader) < args.num_samples:
    print(
        f"Validation set is smaller than {args.num_samples} samples, using the training set for sample generation."
    )
    dataloader = exp.train_loader
else:
    dataloader = exp.val_loader

batch_size = dataloader.batch_size
dataloader = StepIterator(
    dataloader, steps=args.num_samples // dataloader.batch_size + 1
)

samples_dir = path / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)

bar = tqdm(total=args.num_samples, desc="Generating samples", unit="sample")

count = 0
for batch in dataloader:
    ctx, trg = exp.preprocess_fn(batch)
    ctx = ctx.to(device)
    trg = trg.to(device)
    samples = model.sample(ctx, num_samples=batch_size)
    samples = samples * std + mean
    samples = samples.clamp(0, 1)

    for sample in samples:
        count += 1
        bar.update(1)
        bar.set_postfix({"count": count})
        if count > args.num_samples:
            exit(0)
        save_image(sample, samples_dir / f"samples_{count}.png")
