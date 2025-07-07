# %%
from pathlib import Path

import matplotlib.pyplot as plt
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torchvision.utils import make_grid

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment

# %%

# exp_name = "2025-07-01_09-58_xenial_rabbit"
exp_name = "2025-07-03_11-31_unique_yak"
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
batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx = ctx.to("cuda")

out = model.sample(ctx, num_samples=16)


plt.figure(figsize=(10, 20))
out = out.cpu()
trg = trg.cpu()
grid = make_grid(out, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.show()

# %%
