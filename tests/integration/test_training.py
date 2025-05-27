"""Checks that determinstic config initializes as expected."""

from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from cdnp.train import Trainer
from config.config import Config, init_configs


@pytest.mark.parametrize(
    "config_name",
    [
        "cifar10_ccgen",
        "mnist_ccgen",
        "mnist_inpaint",
        "cifar10_inpaint",
    ],
)
def test_training_works(tmpdir: Path, config_name: str) -> None:
    """Checks that the config initializes as expected."""

    GlobalHydra.instance().clear()  # Clear any previous hydra state
    init_configs()
    initialize(config_path="../../src/config", version_base=None)

    cfg: Config = compose(  # type: ignore
        config_name=config_name,
        overrides=[
            "mode=dev",
            "execution.epochs=1",
            f"runtime.root={tmpdir}",
        ],
    )

    trainer = Trainer.from_config(cfg)

    trainer.train_loop()
