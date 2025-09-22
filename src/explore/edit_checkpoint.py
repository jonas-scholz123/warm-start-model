# %%
from pathlib import Path

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

path = Path(
    "[redacted]"
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
