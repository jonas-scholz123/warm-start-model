# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torchvision.utils import make_grid

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment

# %%

# exp_name = "2025-07-01_09-58_xenial_rabbit"
# exp_name = "2025-07-03_11-31_unique_yak"
exp_name = "2025-07-07_11-16_witty_narwhal"
path = Path("/home/jonas/Documents/code/denoising-np/_weights") / exp_name
path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)
model: CDNP = exp.model
cm = CheckpointManager(path)
_ = cm.reproduce_model(model, "best")

ema_model = exp.ema_model.get_shadow()
_ = cm.reproduce_model(ema_model, "best_ema")
# %%
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# %%
metric = FIDMetric(num_samples=50_000, device="cuda", means=mean, stds=std)

result = evaluate(
    # model=ema_model,
    model=model,
    dataloader=exp.train_loader,
    preprocess_fn=exp.preprocess_fn,
    metrics=[metric],
)
print(result)

# %%

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
    out = ema_model.sample(ctx)
    outs.append(out)
out = torch.cat(outs, dim=0)

plt.figure(figsize=(10, 20))
grid = make_grid(out.cpu(), nrow=num_samples, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.savefig("celeba_samples.png", bbox_inches="tight", dpi=300)
plt.axis("off")
plt.show()


# %%
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
# grid = make_grid(plottables.cpu(), nrow=1)

imsize = 5

plt.figure(figsize=(2 * imsize, plottables.shape[0] // 2 * imsize))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.savefig("cdnp_sampling_process.png", bbox_inches="tight", dpi=300)
plt.show()
# %%

plottables2 = [trg, masked_x]
plottables2 = [unnormalise(p) for p in plottables2]
plottables2 = torch.cat(plottables2, dim=0)
grid2 = make_grid(plottables2.cpu(), nrow=1)
plt.figure(figsize=(imsize, 2 * imsize))
plt.axis("off")
plt.imshow(grid2.permute(1, 2, 0))
plt.savefig("cdnp_target_and_masked.png", bbox_inches="tight", dpi=300)
# %%
plt.figure(figsize=(imsize, imsize))
plt.axis("off")
plt.imshow(unnormalise(trg)[0].cpu().permute(1, 2, 0))
# %%
plt.figure(figsize=(imsize, imsize))
plt.axis("off")
plt.imshow(unnormalise(masked_x)[0].cpu().permute(1, 2, 0))

# %%

plottables3 = [pred_mean, pred_std]

plottables3 = [unnormalise(p) for p in plottables3]
plottables3 = torch.cat(plottables3, dim=0)
grid3 = make_grid(plottables3.cpu(), nrow=2)
plt.figure(figsize=(2 * imsize, imsize))
plt.axis("off")
plt.imshow(grid3.permute(1, 2, 0))
plt.savefig("cdnp_pred_mean_and_std.png", bbox_inches="tight", dpi=300)
