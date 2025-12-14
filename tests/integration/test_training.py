"""Checks that determinstic config initializes as expected."""

from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from cdnp.train import Trainer
from config.config import Config, init_configs


def get_config_names() -> list[str]:
    """Returns all config names in src/config, excluding base.yaml."""
    config_dir = Path(__file__).parents[2] / "src" / "config"
    return [f.stem for f in config_dir.glob("*.yaml") if f.name != "base.yaml"]


@pytest.mark.parametrize("config_name", get_config_names())
def test_training_works(tmpdir: Path, config_name: str) -> None:
    """Checks that the config initializes as expected."""

    GlobalHydra.instance().clear()  # Clear any previous hydra state
    init_configs()
    initialize(config_path="../../src/config", version_base=None)

    cfg: Config = compose(  # type: ignore
        config_name=config_name,
        overrides=[
            "mode=dev",
            "execution.train_steps=20",
            "data.trainloader.batch_size=2",
            "data.trainloader.num_workers=0",
            "data.trainloader.persistent_workers=false",
            "data.trainloader.prefetch_factor=null",
            "data.testloader.batch_size=2",
            "data.testloader.num_workers=0",
            "data.testloader.persistent_workers=false",
            "data.testloader.prefetch_factor=null",
            f"paths.output={tmpdir / 'output'}",
        ],
    )

    trainer = Trainer.from_config(cfg)

    trainer.train_loop()
