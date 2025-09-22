# %%
from pathlib import Path

from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

from cdnp.util.instantiate import Experiment

# %%


fpath = Path(
    "[redacted]"
)
path = ExperimentPath.from_path(fpath)
checkpoint_manager = CheckpointManager(path)
# %%

cp = checkpoint_manager.load_checkpoint("latest")

cfg = path.get_config()
exp = Experiment.from_config(cfg)

# %%
model_dict = exp.model.state_dict()
# %%
next(iter(model_dict))
# %%
next(iter(cp.model_state))
# %%

model = checkpoint_manager.reproduce_model(exp.model, "latest")
# %%
exp.model.warm_start_model
# %%
count = 0
#for k, v in model_dict.items():
for k, v in cp.model_state.items():
    print(k, v.shape)
    count += 1
    if count > 10:
        break
# %%


new = {}
for k, v in cp.model_state.items():
    if k.startswith("warm_start_model"):
        continue
    new_key = ".".join(k.split(".")[1:])
    new[new_key] = v
# %%
