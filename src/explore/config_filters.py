# %%
from hydra import compose

from cdnp.util.explore import load_best_weights
from cdnp.util.instantiate import Experiment
from config.config import Config, init_configs

cs = init_configs()

query_cfg: Config = compose(  # type: ignore
    config_name="train", overrides=["mode=prod", "data.testloader.batch_size=1"]
)

d = Experiment.from_config(query_cfg)
load_best_weights(d.model, query_cfg)
