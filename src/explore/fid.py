# %%
from pathlib import Path

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

from cdnp.evaluate import FIDMetric, evaluate
from cdnp.model.cdnp import CDNP
from cdnp.util.instantiate import Experiment

# %%
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
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

metric = FIDMetric(num_samples=1000, device="cuda", means=mean, stds=std)

result = evaluate(
    model=model,
    dataloader=exp.val_loader,
    preprocess_fn=exp.preprocess_fn,
    metrics=[metric],
)
print(result)
