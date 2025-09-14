# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torch.distributions import Normal
from torchvision.utils import make_grid

from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment

# %%
path = Path("/home/jonas/Documents/code/denoising-np/_weights/new_warmth_scaling")

path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)
model: CDNP = exp.model
cm = CheckpointManager(path)
_ = cm.reproduce_model(model, "best")

# %%

data_loader = exp.val_loader
plotter = exp.plotter
preprocess_fn = exp.preprocess_fn
# %%
batch = next(iter(data_loader))
batch[0] = batch[0][:4]
batch[1] = batch[1][:4]  # type: ignore
# %%
ctx, trg = preprocess_fn(batch)
ctx, trg = ctx.to(model.device), trg.to(model.device)
# %%
with torch.no_grad():
    model.eval()
    cnp_dist = model.warm_start_model.predict(ctx)
    cnp_dist = Normal(cnp_dist.mean, cnp_dist.stddev)
    mean = cnp_dist.mean
    std = cnp_dist.stddev

plottables_list = [trg, mean, std]
plottables_list = [plotter._unnormalize(p) for p in plottables_list]
plottables = torch.cat(plottables_list, dim=0)


nrows = plottables.shape[0] // len(plottables_list)
size = 12
plt.figure(figsize=(nrows * size, size))

grid = make_grid(plottables, nrow=nrows)

plt.imshow(grid.permute(1, 2, 0).cpu())

# %%

normed = (trg - mean) / std

import numpy as np


def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


plt.hist(normed.flatten().cpu(), bins=100, density=True, alpha=0.5, label="normed")
x = np.linspace(-3, 3, 1000)
plt.plot(x, gaussian(x, 0, 1), label="standard normal")


# %%
smallest = normed.min().item()
biggest = normed.max().item()

print(normed.shape)
normed_90 = (normed > 0.8 * biggest).to(int).to(float)
print(normed_90.sum())
grid = make_grid(normed_90, nrow=1)
plt.figure(figsize=(1 * size, size))
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.show()
# %%

trg1 = plotter._unnormalize(trg)
mean1 = plotter._unnormalize(mean)
grid = make_grid(
    torch.cat(
        [
            trg1,
            mean1,
            (trg1 - mean1).abs(),
            (trg - mean).abs(),
            (trg - mean).abs() / std,
        ]
    ),
    nrow=4,
)
plt.figure(figsize=(1 * size, size))
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.show()

# %%
mask = normed > 0.8 * biggest
print(mean[mask])
print(trg[mask])
print(trg[mask] - mean[mask])
print(std[mask])
print((trg[mask] - mean[mask]) / std[mask])
# %%
print(normed.mean())
print(normed.std())
# %%

# Show the mask:
plt.figure(figsize=(1 * size, size))
grid = make_grid((mask.to(float)) * -1 + 1, nrow=1)
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.show()
# %%
