#%%
from pathlib import Path
from time import time

from mlbnb.paths import ExperimentPath

from cdnp.util.instantiate import Experiment

path = Path("[redacted]")
exp_path = ExperimentPath.from_path(path)
cfg = exp_path.get_config()

exp = Experiment.from_config(cfg)  # ty: ignore

def num_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(path)
print(num_params(exp.model) / 1_000_000, "M parameters")
#%%

device = "cuda"

batch = next(iter(exp.val_loader))
ctx, trg = exp.preprocess_fn(batch)
ctx, trg = ctx.to(device), trg.to(device)
model = exp.model.to(device)

start = time()
_ = model(ctx, trg)
end = time()

print("Time taken:", end - start)

# %%
