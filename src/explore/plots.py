# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torchvision.utils import make_grid

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment
from config.config import Config


def unnormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Unnormalise a tensor with given means and stds."""
    means = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    stds = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    normed = tensor * stds + means
    return torch.clamp(normed, 0.0, 1.0)

dpi = 100
def plot_grid_no_pad(images: list[torch.Tensor], num_samples: int, imsize: int, fname: str, title: str="") -> None:
    width = num_samples * imsize / dpi
    height = len(images) * imsize / dpi
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    images = torch.cat(images, dim=0)

    images = unnormalise(images)

    grid = make_grid(images.cpu(), nrow=num_samples, normalize=False, padding=0)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(fname, dpi=dpi, pad_inches=0)

# %%

# parser = argparse.ArgumentParser()
# parser.add_argument(
#    "--experiment", type=str, required=True, help="Experiment name or path"
# )
# parser.add_argument(
#    "--model",
#    type=str,
#    default="best_ema",
#    choices=["latest", "latest_ema", "best", "best_ema"],
#    help="Model name to use",
# )
# args = parser.parse_args()


class Args:
    def __init__(self, experiment: str, model: str):
        self.experiment = experiment
        self.model = model

root = Path("/home/jonas/Documents/code/denoising-np")

#args = Args(experiment="2025-07-07_11-16_witty_narwhal", model="latest_ema")
args = Args(experiment="new_warmth_scaling_end_to_end4", model="latest")  # CIFAR10 WSD E2E
#args = Args(experiment="2025-12-28_22-39_witty_bear", model="latest_ema") # SR WSD
#args = Args(experiment="2025-07-29_22-57_quirky_jaguar", model="latest_ema") # CelebA WSD

exp_path = Path(args.experiment)
if not exp_path.exists():
    exp_path = root / "_weights" / args.experiment
if not exp_path.exists():
    exp_path = root / "_output" / args.experiment
if not exp_path.exists():
    raise ValueError(f"Experiment path {exp_path} does not exist.")
print(f"Using experiment path: {exp_path}")

path = ExperimentPath.from_path(exp_path)
cfg: Config = path.get_config()  # type: ignore
exp = Experiment.from_config(cfg)
model: CDNP = exp.model  # type: ignore
cm = CheckpointManager(path)

if "ema" in args.model:
    model_to_load = exp.ema_model.get_shadow()
else:
    model_to_load = model

_ = cm.reproduce_model(model_to_load, args.model)
model_to_load: CDNP = model_to_load  # type: ignore
mean = cfg.data.dataset.norm_means
std = cfg.data.dataset.norm_stds
# %%
num_samples = 8
num_repeats = 3
offset = 0
fid=8

batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to("cuda")
trg = trg.to("cuda")
assert ctx.image_ctx is not None
ctx.image_ctx = ctx.image_ctx[offset : offset + num_samples]
trg = trg[offset : offset + num_samples]
masked_x = ctx.image_ctx[:, :-1, :, :]

outs = [trg, masked_x]
for _ in range(num_repeats):
    out = model_to_load.sample(ctx, num_samples=num_samples, nfe=fid, solver="midpoint")
    outs.append(out)

plot_grid_no_pad(outs, num_samples=num_samples, imsize=64, fname="celeba_inpainting_samples.png", title=None)
#%%

num_samples = 8
num_repeats = 3
offset = 0
fid=8

batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to("cuda")
trg = trg.to("cuda")
assert ctx.image_ctx is not None
ctx.image_ctx = ctx.image_ctx[offset : offset + num_samples]
trg = trg[offset : offset + num_samples]
masked_x = ctx.image_ctx[:, :-1, :, :]

outs = [trg, masked_x]
for _ in range(num_repeats):
    out = model_to_load.sample(ctx, num_samples=num_samples, nfe=fid, solver="midpoint")
    outs.append(out)

plot_grid_no_pad(outs, num_samples=num_samples, imsize=128, fname="cifar10_inpainting_samples.png", title=None)
#%%
num_samples = 8
num_repeats = 2
nfe = 8

batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to("cuda")
trg = trg.to("cuda")
assert ctx.image_ctx is not None

ctx.image_ctx = ctx.image_ctx[:num_samples]
trg = trg[:num_samples]
lowres = ctx.image_ctx

outs = [trg, lowres]
for _ in range(num_repeats):
    out = model_to_load.sample(ctx, num_samples=num_samples, nfe=nfe, solver="midpoint")
    outs.append(out)
#%%
plot_grid_no_pad(outs, num_samples=num_samples, imsize=256, fname="afhq_samples.jpeg", title=None)
# %%

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
