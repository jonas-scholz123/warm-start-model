from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import MISSING

# This isn't perfect, the type annotation approach is nicer but doesn't work with omegaconf
SKIP_KEYS = {
    "output",
    "start_from",
    "_partial_",
    "root",
    "testloader",
    "paths",
}


@dataclass
class Paths:
    root: Path
    data: Path
    raw_data: Path
    output: Path
    weights: Path
    normalisation: Path


@dataclass
class DataConfig:
    cache: bool
    in_channels: int
    num_classes: Optional[int]
    width: int
    height: int
    trainloader: dict
    testloader: dict
    dataset: dict
    preprocess_fn: dict


class CheckpointOccasion(Enum):
    BEST = "best"
    LATEST = "latest"


@dataclass
class ExecutionConfig:
    dry_run: bool
    train_steps: int
    seed: int
    gradient_clip_norm: float
    ema_rate: float
    start_from: Optional[CheckpointOccasion]
    start_weights: Optional[str]
    accumulate_steps: int
    resume: Optional[str]


@dataclass
class OutputConfig:
    save_checkpoints: bool
    out_dir: Path
    use_wandb: bool
    wandb_project: str
    log_gradients: bool
    gradient_log_freq: int
    use_tqdm: bool
    log_level: str
    metrics: list[dict]
    final_metrics: list[dict]
    plotter: Optional[dict]
    save_freq: int
    eval_freq: int
    plot_freq: int


@dataclass
class RngConfig:
    generator: dict
    cpu_generator: dict


@dataclass
class RuntimeConfig:
    device: str
    root: str


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = MISSING
    rng: RngConfig = MISSING
    model: dict = MISSING
    optimizer: dict = MISSING
    output: OutputConfig = MISSING
    scheduler: Optional[dict] = MISSING
    paths: Paths = MISSING
    execution: ExecutionConfig = MISSING


def _get_runtime_cfg() -> RuntimeConfig:
    """
    Get the runtime configuration, containing values that the yaml config needs that are
    only available at runtime.
    """
    root = str(Path(__file__).resolve().parent.parent.parent)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return RuntimeConfig(device=device, root=root)


def init_configs() -> ConfigStore:
    cs = ConfigStore.instance()

    cs.store(name="base_cfg", node=Config(runtime=_get_runtime_cfg()))

    return cs
