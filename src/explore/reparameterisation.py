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
path = Path(
    "/home/jonas/Documents/code/denoising-np/_weights/2025-06-15_11-04_sassy_unicorn"
)

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
    cnp_dist = model.cnp.predict(ctx)
    cnp_dist = Normal(cnp_dist.mean, cnp_dist.stddev)
    mean = cnp_dist.mean
    std = cnp_dist.stddev

plottables_list = [trg, mean, std, trg - mean, (trg - mean) / std]
plottables_list = [plotter._unnormalize(p) for p in plottables_list]
plottables = torch.cat(plottables_list, dim=0)


nrows = plottables.shape[0] // len(plottables_list)
size = 12
plt.figure(figsize=(nrows * size, size))

grid = make_grid(plottables, nrow=nrows)

plt.imshow(grid.permute(1, 2, 0).cpu())
