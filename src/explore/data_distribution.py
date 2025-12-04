# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

exp_path = Path("/home/jonas/Documents/code/denoising-np/_weights/celeba_cnp")

path = ExperimentPath.from_path(exp_path)
cfg = path.get_config()
cfg.data.trainloader.num_workers = 0
cfg.data.trainloader.prefetch_factor = None
cfg.data.trainloader.persistent_workers = False

exp = Experiment.from_config(cfg)
model: CDNP = exp.model
cm = CheckpointManager(path)
_ = cm.reproduce_model(model, "latest")

mean = cfg.data.dataset.norm_means
std = cfg.data.dataset.norm_stds
mean = torch.tensor(mean, device=device)[None, :, None, None]
std = torch.tensor(std, device=device)[None, :, None, None]

dataloader = exp.val_loader
batch_size = dataloader.batch_size

# %%
batch = next(iter(dataloader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to(device)
trg = trg.to(device)
# %%

with torch.no_grad():
    prd_dist = model.predict(ctx)
    prd_std = prd_dist.stddev
    prd_mean = prd_dist.mean
    # imshow the std and the unmasked pixels:
    unmasked_pixels = ctx.image_ctx[:, 0:3, :, :]
    unmasked_pixels = unmasked_pixels * std + mean
    unmasked_pixels = unmasked_pixels.clamp(0, 1)
    unmasked_pixels = unmasked_pixels[0].permute(1, 2, 0)
    prd_mean = prd_mean * std + mean
    prd_mean = prd_mean.clamp(0, 1)
    prd_mean = prd_mean[0].permute(1, 2, 0)

fig, ax = plt.subplots(figsize=(4, 4))
ax.axis("off")
ax.imshow(unmasked_pixels.detach().cpu(), cmap="gray")
plt.show()
# no spacing
fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=2, constrained_layout=True)
axs[0].axis("off")
axs[1].axis("off")
axs[0].imshow(prd_mean.detach().cpu(), cmap="viridis")
axs[1].imshow(prd_std[0, 0].detach().cpu(), cmap="viridis")
plt.show()

# %%
from tqdm import tqdm

all_trg_vals = []
all_norm_vals = []

limit_batches = 100
current_batch = 0

pbar = tqdm(total=limit_batches, desc="Processing Batches")

with torch.no_grad():
    for batch in dataloader:
        if current_batch >= limit_batches:
            break

        ctx, trg = exp.preprocess_fn(batch)
        ctx = ctx.to(device)
        trg = trg.to(device)

        prd_dist = model.predict(ctx)
        prd_mean = prd_dist.mean
        prd_std = prd_dist.stddev
        prd_std = torch.clamp(prd_std, min=0.1)

        # The warm-start normalization trick
        normalised_trg = (trg - prd_mean) / prd_std

        # Keep data on CPU numpy to save GPU memory
        all_trg_vals.append(trg.flatten().cpu().numpy())
        all_norm_vals.append(normalised_trg.flatten().cpu().numpy())

        current_batch += 1
        pbar.update(1)

# Concatenate all into one massive array
import numpy as np

pixel_values_norm = np.concatenate(all_norm_vals)
pixel_values = np.concatenate(all_trg_vals)

# %% --- PLOTTING & ANALYSIS ---
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm

k_norm = kurtosis(pixel_values_norm)
k = kurtosis(pixel_values)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].hist(
    pixel_values,
    bins=100,
    density=True,
    label="Unnormalised Data",
    log=True,
)

axs[1].hist(
    pixel_values_norm,
    bins=100,
    density=True,
    label="Normalised Data",
    log=True,
)

x = np.linspace(-10, 10, 1000)

p = norm.pdf(x, 0, 1)
axs[0].plot(x, p, "r", linewidth=2, label="Standard Normal")

norm_mean = np.mean(pixel_values_norm)
norm_std = np.std(pixel_values_norm)
p = norm.pdf(x, 0, 1)
axs[0].set_xlim(-1.05, 1.05)
axs[0].set_ylim(0.2, 1.5)
axs[1].set_ylim(1e-7, None)
axs[1].plot(x, p, "r", linewidth=2, label="Standard Normal")
axs[0].set_title(f"Data-Space Pixel Distribution (Kurtosis={k:.1f})")
axs[1].set_title(f"Normalised-Space Pixel Distribution (Kurtosis={k_norm:.1f})")
axs[0].set_ylabel("Density (Log Scale)")

for ax in axs:
    ax.legend()
    ax.set_xlabel("Pixel Value")
# Save or show
fig.savefig("data_distribution.pdf", bbox_inches="tight")

# %%

# %%
