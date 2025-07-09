import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torchvision.utils import make_grid

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
    model_to_load = exp.ema_model.get_shadow()
else:
    model_to_load = model

_ = cm.reproduce_model(model_to_load, args.model)

mean = cfg.data.dataset.norm_means
std = cfg.data.dataset.norm_stds

metric = FIDMetric(num_samples=50_000, device="cuda", means=mean, stds=std)

if len(exp.val_loader) < 50_000:
    print(
        "Validation set is smaller than 50_000 samples, using the training set for FID evaluation."
    )
    dataloader = exp.train_loader
else:
    dataloader = exp.val_loader

result = evaluate(
    model=model_to_load,
    dataloader=dataloader,
    preprocess_fn=exp.preprocess_fn,
    metrics=[metric],
    use_tqdm=True,
)
print(result)

num_samples = 4
num_repeats = 3

batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to("cuda")
trg = trg.to("cuda")
ctx.image_ctx = ctx.image_ctx[:num_samples]
trg = trg[:num_samples]
masked_x = ctx.image_ctx[:, :-1, :, :]

outs = [trg, masked_x]
for _ in range(num_repeats):
    out = model_to_load.sample(ctx)
    outs.append(out)
out = torch.cat(outs, dim=0)

plt.figure(figsize=(10, 20))
grid = make_grid(out.cpu(), nrow=num_samples, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.savefig("celeba_samples.png", bbox_inches="tight", dpi=300)
plt.axis("off")
plt.show()


def unnormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Unnormalise a tensor with given means and stds."""
    means = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    stds = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    normed = tensor * stds + means
    return torch.clamp(normed, 0.0, 1.0)


num_samples = 1
ctx.image_ctx = ctx.image_ctx[:num_samples]
trg = trg[:num_samples]
plottables = model.make_plot(ctx)

mask = ctx.image_ctx[:, -1:, :, :]
mask = mask.expand(-1, 3, -1, -1)
masked_x = ctx.image_ctx[:, :-1, :, :]

pred_mean = plottables[0]
pred_std = plottables[1]

data_space = plottables[3::2]
noise_space = plottables[2::2]

plottables = data_space + noise_space

plottables = [unnormalise(p) for p in plottables]
plottables = torch.cat(plottables, dim=0)

grid = make_grid(plottables.cpu(), nrow=plottables.shape[0] // 2)

imsize = 5

plt.figure(figsize=(2 * imsize, plottables.shape[0] // 2 * imsize))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.savefig("cdnp_sampling_process.png", bbox_inches="tight", dpi=300)
plt.show()

plottables2 = [trg, masked_x]
plottables2 = [unnormalise(p) for p in plottables2]
plottables2 = torch.cat(plottables2, dim=0)
grid2 = make_grid(plottables2.cpu(), nrow=1)
plt.figure(figsize=(imsize, 2 * imsize))
plt.axis("off")
plt.imshow(grid2.permute(1, 2, 0))
plt.savefig("cdnp_target_and_masked.png", bbox_inches="tight", dpi=300)

plt.figure(figsize=(imsize, imsize))
plt.axis("off")
plt.imshow(unnormalise(trg)[0].cpu().permute(1, 2, 0))

plt.figure(figsize=(imsize, imsize))
plt.axis("off")
plt.imshow(unnormalise(masked_x)[0].cpu().permute(1, 2, 0))

plottables3 = [pred_mean, pred_std]

plottables3 = [unnormalise(p) for p in plottables3]
plottables3 = torch.cat(plottables3, dim=0)
grid3 = make_grid(plottables3.cpu(), nrow=2)
plt.figure(figsize=(2 * imsize, imsize))
plt.axis("off")
plt.imshow(grid3.permute(1, 2, 0))
plt.savefig("cdnp_pred_mean_and_std.png", bbox_inches="tight", dpi=300)
