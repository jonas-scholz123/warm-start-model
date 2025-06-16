# %%
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torchvision.utils import make_grid

from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment
#%%

path = Path(
    "/home/jonas/Documents/code/denoising-np/_weights/2025-06-03_17-53_fantastic_jaguar"
)
path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)
model: CDNP = exp.model
cm = CheckpointManager(path)
_ = cm.reproduce_model(model, "best")
# %%
device = "cuda"

dataloader = exp.val_loader
batch = next(iter(dataloader))
ctx, trg = exp.preprocess_fn(batch)
ctx, trg = ctx.to(device), trg.to(device)
# %%
batch_size = 4
ctx.image_ctx = ctx.image_ctx[:batch_size]
samples = model.sample(ctx)

stds = [0.5, 0.5, 0.5]
means = [0.5, 0.5, 0.5]
stds = torch.tensor(stds, device=device).view(1, 3, 1, 1)
means = torch.tensor(means, device=device).view(1, 3, 1, 1)


def unnormalise(x):
    x = x * stds + means
    return x.clamp(0, 1)


samples = unnormalise(samples)

grid = make_grid(samples, nrow=4)

fig, ax = plt.subplots(figsize=(16, 16))
ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.show()
