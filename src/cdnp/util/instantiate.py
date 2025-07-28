from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from loguru import logger
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from mlbnb.types import Split
from omegaconf import OmegaConf
from torch import Generator
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from cdnp.data.data import make_dataset
from cdnp.evaluate import Metric
from cdnp.model.ddpm import ModelCtx
from cdnp.model.ema import ExponentialMovingAverage
from cdnp.plot.plotter import CcgenPlotter
from config.config import Config, init_configs


@dataclass
class Experiment:
    model: Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]
    generator: Generator
    experiment_path: ExperimentPath
    checkpoint_manager: CheckpointManager
    plotter: Optional[CcgenPlotter]
    preprocess_fn: Callable[[Any], ModelCtx]
    metrics: list[Metric]
    final_metrics: list[Metric]
    ema_model: Optional[ExponentialMovingAverage] = None

    @staticmethod
    def from_config(cfg: Config) -> "Experiment":
        """
        Instantiates all dependencies for the training loop.

        This is useful for exploration where you want to have easy access to the
        instantiated objects used for training and evaluation.
        """
        logger.info("Instantiating dependencies")

        exp = instantiate(cfg)
        rng = exp.rng.generator.manual_seed(cfg.execution.seed)
        cpu_rng = exp.rng.cpu_generator.manual_seed(cfg.execution.seed + 1)

        trainset = make_dataset(exp.data, Split.TRAIN, cpu_rng)
        valset = make_dataset(exp.data, Split.VAL, cpu_rng)
        testset = make_dataset(exp.data, Split.TEST, cpu_rng)

        train_loader: DataLoader = exp.data.trainloader(trainset, generator=cpu_rng)
        val_loader: DataLoader = exp.data.testloader(valset, generator=cpu_rng)
        test_loader: DataLoader = exp.data.testloader(testset, generator=cpu_rng)

        model: Module = exp.model.to(cfg.runtime.device)
        _log_num_params(model)
        optimizer = exp.optimizer(model.parameters())

        if exp.execution.ema_rate > 0:
            ema_model = ExponentialMovingAverage(model, exp.execution.ema_rate)
        else:
            ema_model = None

        scheduler: Optional[LRScheduler] = (
            exp.scheduler(optimizer) if exp.scheduler else None
        )

        if not cfg.execution.resume:
            experiment_path = ExperimentPath.from_config(cfg, Path(cfg.paths.output))
        else:
            path = Path(cfg.paths.output) / cfg.execution.resume
            experiment_path = ExperimentPath.from_path(path)

        logger.info("Experiment path: {}", str(experiment_path))
        checkpoint_manager = CheckpointManager(experiment_path)

        plotter: Optional[CcgenPlotter] = (
            exp.output.plotter(test_data=valset, save_to=experiment_path)
            if exp.output.plotter
            else None
        )

        logger.info("Finished instantiating dependencies")

        return Experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=rng,
            experiment_path=experiment_path,
            checkpoint_manager=checkpoint_manager,
            plotter=plotter,
            preprocess_fn=exp.data.preprocess_fn,
            metrics=exp.output.metrics,
            final_metrics=exp.output.final_metrics,
            ema_model=ema_model,
        )


def _log_num_params(model: Module) -> None:
    """
    Log the number of parameters in the model.
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_params_million = num_params / 1_000_000
    logger.info("Number of parameters: {:.2f}M", num_params_million)


def load_config(
    config_name: str = "mnist_ccgen",
    mode: str = "dev",
    overrides: Optional[list[str]] = None,
    config_path: str = "../../config",
) -> Config:
    """
    Load the configuration from the given config name and path.
    """
    init_configs()

    all_overrides = [f"mode={mode}"] + (overrides or [])

    if not GlobalHydra.instance().is_initialized():
        with initialize(config_path=config_path):
            cfg: Config = compose(  # type: ignore
                config_name=config_name, overrides=all_overrides
            )
    else:
        cfg: Config = compose(  # type: ignore
            config_name=config_name, overrides=all_overrides
        )

    return cfg


def load_model_from_path(
    path: Path | ExperimentPath | str,
    checkpoint: str = "latest",
    freeze: bool = False,
    device: str = "cuda",
) -> Module:
    if isinstance(path, str):
        path = Path(path)
    if isinstance(path, Path):
        path = ExperimentPath.from_path(path)

    base_cfg = load_config(config_name="base")
    cfg: Config = path.get_config()
    # For backward compatibility, merge base config, which contains default values
    cfg = OmegaConf.merge(cfg, base_cfg)
    cfg.runtime = base_cfg.runtime
    cfg.runtime.device = device
    print("CFG: ", cfg)
    exp: Experiment = Experiment.from_config(cfg)

    cm = CheckpointManager(path)
    _ = cm.reproduce_model(exp.model, checkpoint)
    exp.model = exp.model.to(device)

    if freeze:
        for param in exp.model.parameters():
            param.requires_grad = False
        logger.info("Loaded model parameters are frozen.")

    return exp.model
