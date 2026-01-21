#%%
from pathlib import Path

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torchvision.utils import make_grid

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment
from config.config import Config

class Args:
    def __init__(self, experiment: str, model: str):
        self.experiment = experiment
        self.model = model


experiments = [
    "new_warmth_scaling",
    #"new_warmth_scaling_end_to_end2",
    #"new_warmth_scaling_end_to_end3",
    "new_warmth_scaling_end_to_end4",
    #"2025-12-23_16-17_witty_octopus"
    #"2025-12-23_17-55_lucky_dog"
    #"2025-12-23_19-36_quirky_kangaroo"
]

exp_path = Path(experiments[0])
if not exp_path.exists():
    exp_path = Path("../../_weights") / experiments[0]
if not exp_path.exists():
    exp_path = Path("../../_output") / experiments[0]
path = ExperimentPath.from_path(exp_path)
cfg: Config = path.get_config()  # type: ignore
cfg.data.testloader.batch_size = 16
exp = Experiment.from_config(cfg)
#%%
batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to("cuda")
trg = trg.to("cuda")
#%%

def get_mean_and_std(experiment_name: str, ctx):
    exp_path = Path(experiment_name)
    if not exp_path.exists():
        exp_path = Path("../../_weights") / experiment_name
    if not exp_path.exists():
        exp_path = Path("../../_output") / experiment_name

    path = ExperimentPath.from_path(exp_path)
    cfg: Config = path.get_config()  # type: ignore
    exp = Experiment.from_config(cfg)
    model: CDNP = exp.model  # type: ignore
    cm = CheckpointManager(path)

    _ = cm.reproduce_model(model, "latest")
    model: CDNP = model  # type: ignore
    cnp = model.warm_start_model

    prd_dist = cnp.predict(ctx)
    mean = prd_dist.mean.cpu()
    std = prd_dist.stddev.cpu()
    mean = mean * 0.5 + 0.5
    std = std * 0.5 + 0.5
    return mean, std

means = {}
stds = {}
for exp_name in experiments:
    mean, std = get_mean_and_std(exp_name, ctx)
    means[exp_name] = mean
    stds[exp_name] = std
#%%
offset = 0
num_elements = 8

means_combined = []
stds_combined = []
for exp_name in experiments:
    means_combined.append(means[exp_name][offset:offset+num_elements])
    stds_combined.append(stds[exp_name][offset:offset+num_elements])

mean_combined = torch.cat(means_combined, dim=0)
std_combined = torch.cat(stds_combined, dim=0)
print(f"Max mean: {mean_combined.max().item():.4f}, Min mean: {mean_combined.min().item():.4f}")
print(f"Max std: {std_combined.max().item():.4f}, Min std: {std_combined.min().item():.4f}")

plt.figure(figsize=(num_elements * 4, len(experiments) * 4))
grid_mean = make_grid(mean_combined, nrow=num_elements, normalize=True)
plt.imshow(grid_mean.permute(1, 2, 0))
plt.axis('off')
plt.figure(figsize=(num_elements * 4, len(experiments) * 4))
grid_std = make_grid(std_combined, nrow=num_elements, normalize=False)
grid_std = grid_std.mean(dim=0, keepdim=True)
plt.imshow(grid_std.permute(1, 2, 0), cmap='viridis')
plt.axis('off')

# %%
