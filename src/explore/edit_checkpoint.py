# %%
from pathlib import Path

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

path = Path(
    "/home/jonas/Documents/code/denoising-np/_weights/2025-06-15_11-04_sassy_unicorn"
)
path = ExperimentPath.from_path(path)
cm = CheckpointManager(path)

# %%
checkpoint = cm.load_checkpoint("latest")
# %%
# %%
checkpoint.scheduler_state["_last_lr"] = 0.003 / 20 * 100
checkpoint.scheduler_state["total_iters"] = 100
import torch

path = cm.dir / "latest.pt"
torch.save(checkpoint, path)
# %%
