import pytest
from hydra import compose, initialize

from cdnp.util.instantiate import Experiment
from config.config import Config, init_configs


@pytest.mark.parametrize(
    "cfg_name,overrides",
    [
        ("base", ["mode=prod", "data=mnist"]),
        ("base", ["mode=dev", "data=cifar10"]),
    ],
)
def test_experiment_init(cfg_name: str, overrides: list[str]) -> None:
    init_configs()
    with initialize(version_base=None, config_path="../src/config"):
        cfg: Config = compose(  # type: ignore
            config_name=cfg_name, overrides=overrides
        )
        Experiment.from_config(cfg)
