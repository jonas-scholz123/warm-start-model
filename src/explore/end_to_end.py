# %%
from pathlib import Path
import matplotlib.pyplot as plt

from mlbnb.paths import ExperimentPath
from mlbnb.checkpoint import CheckpointManager

from cdnp.util.instantiate import Experiment
#%%

e2e_path = Path(
    "/home/jonas/Documents/code/denoising-np/_output/new_warmth_scaling_end_to_end"
)
path = Path(
    "/home/jonas/Documents/code/denoising-np/_weights/new_warmth_scaling"
)
exp_path = ExperimentPath.from_path(path)
e2e_exp_path = ExperimentPath.from_path(e2e_path)
cfg = exp_path.get_config()
e2e_cfg = e2e_exp_path.get_config()
device = "cuda"

exp = Experiment.from_config(cfg)  # ty: ignore
e2e_exp = Experiment.from_config(e2e_cfg)  # ty: ignore


cm = CheckpointManager(path)
model = exp.model
_ = cm.reproduce_model(model, "latest")

e2e_cm = CheckpointManager(e2e_path)
e2e_model = e2e_exp.model
_ = e2e_cm.reproduce_model(e2e_model, "latest")

# %%
batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx, trg = ctx.to(device), trg.to(device)
#%%

model = exp.model.to(device)
data_std = 0.5
data_mean = 0.5

prd_dist = model.warm_start_model.predict(ctx)
prd_dist_e2e = e2e_model.warm_start_model.predict(ctx)
means = prd_dist.mean
means_e2e = prd_dist_e2e.mean
stds = prd_dist.stddev
stds_e2e = prd_dist_e2e.stddev
# Unnormalise:
means = means * data_std + data_mean
means_e2e = means_e2e * data_std + data_mean

#%%
num_ims = 5
fig, axs = plt.subplots(num_ims, 4, figsize=(10, 3 * num_ims))

for i in range(5):
    mean = means[i].permute(1, 2, 0).cpu().detach()
    mean_e2e = means_e2e[i].permute(1, 2, 0).cpu().detach()

    std = stds[i].mean(axis=0).cpu().detach()
    std_e2e = stds_e2e[i].mean(axis=0).cpu().detach()
    axs[i, 0].imshow(mean)
    axs[i, 1].imshow(mean_e2e)
    # Remove ticks, numbers etc:
    axs[i, 2].imshow(std, cmap="viridis")
    axs[i, 3].imshow(std_e2e, cmap="viridis")
    axs[i, 0].axis("off")
    axs[i, 1].axis("off")
    axs[i, 2].axis("off")
    axs[i, 3].axis("off")
axs[0, 0].set_title("Two-stage model")
axs[0, 1].set_title("End-to-end model")
axs[0, 2].set_title("Two-stage model stddev")
axs[0, 3].set_title("End-to-end model stddev")
plt.show()
# %%
