#%%
from mlbnb.paths import ExperimentPath
from pathlib import Path
from cdnp.util.instantiate import Experiment
from time import time

path = Path("/home/jonas/Documents/code/denoising-np/_weights/cifar10_cnp_001")
#path = Path("/home/jonas/Documents/code/denoising-np/_weights/2025-07-21_22-38_playful_xenon")
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
