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


def _is_era5_config(config_name: str) -> bool:
    return config_name.startswith("era5_")


def _is_warm_start_config(config_name: str) -> bool:
    return "fm_warm" in config_name


@pytest.mark.parametrize("config_name", get_config_names())
def test_training_works(tmpdir: Path, config_name: str) -> None:
    """Checks that the config initializes as expected."""

    if _is_warm_start_config(config_name):
        pytest.skip("Warm-start configs require pre-trained checkpoints")

    GlobalHydra.instance().clear()  # Clear any previous hydra state
    init_configs()
    initialize(config_path="../../src/config", version_base=None)

    overrides = [
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
    ]

    # Local zarr data covers 2019-12-01 to 2021-01-30, so override the
    # date ranges for ERA5 configs to fit within the available data.
    if _is_era5_config(config_name):
        overrides += [
            "data.dataset.start_date=2020-01-01",
            "data.dataset.end_date=2020-10-31",
            "data.dataset.val_start_date=2020-11-01",
            "data.dataset.val_end_date=2021-01-28",
        ]

    cfg: Config = compose(  # type: ignore
        config_name=config_name,
        overrides=overrides,
    )

    trainer = Trainer.from_config(cfg)

    trainer.train_loop()
