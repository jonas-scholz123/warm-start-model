import warnings
from typing import Callable, Optional

import hydra
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from mlbnb.checkpoint import CheckpointManager, TrainerState
from mlbnb.iter import StepIterator
from mlbnb.metric_logger import WandbLogger
from mlbnb.paths import ExperimentPath
from mlbnb.profiler import WandbProfiler
from mlbnb.rand import seed_everything
from omegaconf import OmegaConf
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from cdnp.evaluate import Metric, evaluate
from cdnp.model.ema import ExponentialMovingAverage
from cdnp.plot.plotter import CcgenPlotter
from cdnp.task import PreprocessFn
from cdnp.util.instantiate import Experiment
from config.config import (
    CheckpointOccasion,
    Config,
    init_configs,
)

init_configs()

TaskType = tuple[torch.Tensor, torch.Tensor]


@hydra.main(version_base=None, config_name="base", config_path="../config")
def main(cfg: Config) -> float:
    _configure_outputs()

    logger.debug(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.execution.seed)

    trainer = Trainer.from_config(cfg)
    trainer.train_loop()
    if cfg.output.use_wandb:
        wandb.finish()
    return trainer.state.best_val_loss


def _configure_outputs():
    load_dotenv()

    # These are all due to deprecation warnings raised within dependencies.
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="google.protobuf"
    )


class Trainer:
    cfg: Config
    model: torch.nn.Module
    optimizer: Optimizer
    train_loader: DataLoader[TaskType]
    val_loader: DataLoader[TaskType]
    test_data: DataLoader[TaskType]
    generator: torch.Generator
    experiment_path: ExperimentPath
    checkpoint_manager: CheckpointManager
    scheduler: Optional[LRScheduler]
    plotter: Optional[CcgenPlotter]
    state: TrainerState
    preprocess_fn: PreprocessFn
    metrics: list[Callable]

    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        generator: torch.Generator,
        experiment_path: ExperimentPath,
        checkpoint_manager: CheckpointManager,
        scheduler: Optional[LRScheduler],
        plotter: Optional[CcgenPlotter],
        preprocess_fn: PreprocessFn,
        metrics: list[Metric],
        final_metrics: list[Metric],
        ema: Optional[ExponentialMovingAverage],
    ):
        self.cfg = cfg
        self.model = model
        self.ema = ema
        self.inference_model = ema.get_shadow() if ema else model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.generator = generator
        self.experiment_path = experiment_path
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler
        self.plotter = plotter
        self.preprocess_fn = preprocess_fn
        self.metrics = metrics
        self.final_metrics = final_metrics
        self.grad_scaler = GradScaler()

        self.state = self._load_initial_state()

        self._init_wandb()
        self.metric_logger = WandbLogger(
            self.cfg.output.use_wandb,
            self.state,
        )
        self.profiler = WandbProfiler(self.metric_logger)

    def _load_initial_state(self) -> TrainerState:
        start_from = self.cfg.execution.start_from

        initial_state = TrainerState(
            step=0, epoch=0, best_val_loss=np.inf, val_loss=np.inf
        )

        weights_name = self.cfg.execution.start_weights
        if weights_name:
            weights_path = self.cfg.paths.weights / weights_name
            CheckpointManager.reproduce_model_from_path(self.model, weights_path)
            logger.info(
                "Pretrained model loaded from path {}, starting from pretrained.",
                weights_path,
            )
        elif start_from and self.checkpoint_manager.checkpoint_exists(start_from.value):
            self.checkpoint_manager.reproduce(
                start_from.value,
                self.model,
                self.optimizer,
                self.generator,
                self.scheduler,
                initial_state,
            )
            if self.ema:
                self.checkpoint_manager.reproduce(
                    start_from.value + "_ema",
                    self.ema.get_shadow(),
                    self.optimizer,
                    self.generator,
                    self.scheduler,
                    initial_state,
                )

            logger.info(
                "Checkpoint loaded, val loss: {}, step: {}",
                initial_state.val_loss,
                initial_state.step,
            )
        else:
            logger.info("Starting from scratch")

        if initial_state.step >= self.cfg.execution.train_steps:
            logger.info(
                "Run has concluded (step {} / {})",
                initial_state.step,
                self.cfg.execution.train_steps,
            )
        return initial_state

    def _init_wandb(self) -> None:
        if self.cfg.output.use_wandb:
            wandb.init(
                project=self.cfg.output.wandb_project,
                config=OmegaConf.to_container(self.cfg),  # type: ignore
                dir=self.cfg.output.out_dir,
                name=self.experiment_path.name,
                id=self.experiment_path.name,
            )

    @staticmethod
    def from_config(cfg: Config) -> "Trainer":
        exp = Experiment.from_config(cfg)

        return Trainer(
            cfg,
            exp.model,
            exp.optimizer,
            exp.train_loader,
            exp.val_loader,
            exp.generator,
            exp.experiment_path,
            exp.checkpoint_manager,
            exp.scheduler,
            exp.plotter,
            exp.preprocess_fn,
            exp.metrics,
            exp.final_metrics,
            exp.ema_model,
        )

    def train_loop(self):
        s = self.state
        self._save_config()

        if self.cfg.output.use_wandb and self.cfg.output.log_gradients:
            wandb.watch(
                self.model,
                log="all",
                log_freq=self.cfg.output.gradient_log_freq,
            )

        logger.info("Starting training")

        train_iter = tqdm(
            StepIterator(self.train_loader, self.cfg.execution.train_steps),
            total=self.cfg.execution.train_steps,
        )

        dry_run = self.cfg.execution.dry_run

        for batch in train_iter:
            if self.state.step % self.cfg.output.eval_freq == 0 or dry_run:
                logger.info("Evaluating on validation set")
                val_metrics = evaluate(
                    self.inference_model,
                    self.val_loader,
                    self.preprocess_fn,
                    self.metrics,
                    self.cfg.execution.dry_run,
                )
                self._log_wandb(val_metrics, prefix="val")
                s.val_loss = val_metrics["loss"]
                if s.val_loss < s.best_val_loss:
                    logger.success("New best val loss: {}", s.val_loss)
                    s.best_val_loss = s.val_loss
                    self.save_checkpoint(CheckpointOccasion.BEST)

            if self.state.step % self.cfg.output.plot_freq == 0 or dry_run:
                logger.info("Plotting predictions")
                if self.plotter:
                    self.plotter.plot_prediction(self.inference_model, self.state.step)

            if self.state.step % self.cfg.output.save_freq == 0 or dry_run:
                logger.info("Saving checkpoint")
                self.save_checkpoint(CheckpointOccasion.LATEST)

            if self.state.step % 500 == 0 and hasattr(self.model, "set_steps"):
                self.model.set_steps(self.state.step)

            self.model.train()
            self.train_step(batch)
            self.model.eval()

            if dry_run:
                break

        logger.success("Finished training")
        if self.final_metrics:
            eval_loader = self.val_loader
            if len(self.val_loader.dataset) < 50_000:
                logger.warning(
                    "Validation set is small, using training set for final evaluation."
                )
                eval_loader = self.train_loader

            logger.info("Evaluating final metrics")
            final_metrics = evaluate(
                self.inference_model,
                eval_loader,
                self.preprocess_fn,
                self.final_metrics,
                self.cfg.execution.dry_run,
            )
            logger.info("Final metrics: {}", final_metrics)
            self._log_wandb(final_metrics, prefix="final")

    def _save_config(self) -> None:
        with open(self.experiment_path / "cfg.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def _log_wandb(self, metrics: dict[str, float], prefix: str = "misc") -> None:
        self.metric_logger.log({f"{prefix}/{k}": v for k, v in metrics.items()})

    def train_step(self, batch: TaskType) -> None:
        device = next(self.model.parameters()).device

        p = self.profiler

        with p.profile("preprocess"):
            ctx, trg = self.preprocess_fn(batch)

        with p.profile("data.to"):
            ctx = ctx.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)

        with p.profile("forward"):
            with autocast(device_type=device.type, dtype=torch.float16):
                loss = self.model(ctx, trg)

        with p.profile("backward"):
            self.grad_scaler.scale(loss).backward()
            self._log_wandb({"loss": loss.item()}, prefix="train")

        with p.profile("optimizer.step"):
            clip_norm: float = self.cfg.execution.gradient_clip_norm
            if clip_norm > 0:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                self._log_wandb({"grad_norm": grad_norm})

            self.grad_scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.grad_scaler.update()
            if self.scheduler:
                self._log_wandb({"lr": self.scheduler.get_last_lr()[0]})

        if self.ema:
            self.ema.update()

        if self.scheduler:
            self.scheduler.step()

        self.state.step += 1

    def save_checkpoint(self, occasion: CheckpointOccasion):
        if self.cfg.output.save_checkpoints:
            self.checkpoint_manager.save_checkpoint(
                occasion.value,
                self.model,
                self.optimizer,
                self.generator,
                self.scheduler,
                self.state,
            )
            if self.ema:
                self.checkpoint_manager.save_checkpoint(
                    occasion.value + "_ema",
                    self.ema.get_shadow(),
                    self.optimizer,
                    self.generator,
                    self.scheduler,
                    self.state,
                )


if __name__ == "__main__":
    main()
