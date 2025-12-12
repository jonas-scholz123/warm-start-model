# %%
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import torch
import wandb
from loguru import logger
from mlbnb.paths import ExperimentPath
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.utils import make_grid, save_image

from cdnp.data.era5 import GriddedWeatherTask
from cdnp.model.cdnp import CDNP
from cdnp.model.cnp import CNP
from cdnp.model.ddpm import DDPM, ModelCtx
from cdnp.plot.geoplot import GeoPlotter


class BasePlotter(ABC):
    def __init__(
        self,
        device: str | torch.device,
        norm_means: tuple[float, ...],
        norm_stds: tuple[float, ...],
        save_to: Optional[ExperimentPath] = None,
    ):
        self._device = device
        self._dir = save_to
        self._norm_means = torch.tensor(norm_means).to(device)[None, :, None, None]
        self._norm_stds = torch.tensor(norm_stds).to(device)[None, :, None, None]

    @abstractmethod
    def plot_prediction(self, model: DDPM, step: int = 0) -> None:
        """
        Generate and save prediction plots using the provided model.

        :param model: The DDPM model to use for generating predictions.
        :param step: The current step number, used for naming the saved plot.
        """
        pass

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the image tensor using the provided means and stds.
        """
        x = x * self._norm_stds + self._norm_means
        return x.clamp(0, 1)

    def _get_path(self, step: int, filename: str = "image") -> Path:
        """
        Generate a path for saving the plot image.
        """
        return self._dir / f"{filename}_ep{step}.png"

    def _log_to_wandb(self, data: Any, step: int, name: str) -> None:
        """Logs plots to wandb if available."""
        if wandb.run:
            wandb.log(
                {f"plots/{name}": wandb.Image(data, caption=f"Step {step}")}, step=step
            )


class CcgenPlotter(BasePlotter):
    def __init__(
        self,
        device: str | torch.device,
        num_samples: int,
        num_classes: int,
        norm_means: tuple[float, ...],
        norm_stds: tuple[float, ...],
        test_data: Dataset,
        save_to: Optional[ExperimentPath] = None,
    ):
        super().__init__(device, norm_means, norm_stds, save_to)
        self._num_samples = num_samples
        self._num_classes = num_classes

    @torch.no_grad()
    def plot_prediction(self, model: DDPM, step: int = 0) -> None:
        logger.info("Making and saving prediction plots")
        class_labels = (
            torch.arange(self._num_classes)
            .repeat(self._num_samples, 1)
            .flatten()
            .to(self._device)
        )

        ctx = ModelCtx(label_ctx=class_labels)

        total_samples = self._num_samples * self._num_classes
        x_gen = model.sample(ctx, total_samples)
        x_gen = self._unnormalize(x_gen)

        grid = make_grid(x_gen, nrow=self._num_classes)
        if self._dir:
            save_image(grid, self._get_path(step))
        self._log_to_wandb(grid, step, "ccgen")


class ImgenPlotter(BasePlotter):
    def __init__(
        self,
        device: str | torch.device,
        num_samples: int,
        norm_means: tuple[float, ...],
        norm_stds: tuple[float, ...],
        test_data: Optional[Dataset] = None,
        save_to: Optional[ExperimentPath] = None,
    ):
        super().__init__(device, norm_means, norm_stds, save_to)
        self._num_samples = num_samples

    @torch.no_grad()
    def plot_prediction(self, model: DDPM, step: int = 0) -> None:
        logger.info("Making and saving prediction plots")

        ctx = ModelCtx()

        x_gen = model.sample(ctx, self._num_samples)
        x_gen = self._unnormalize(x_gen)

        grid = make_grid(x_gen, nrow=int(self._num_samples**0.5))
        if self._dir:
            save_image(grid, self._get_path(step))
        self._log_to_wandb(grid, step, "imgen")


class InpaintPlotter(BasePlotter):
    def __init__(
        self,
        device: str | torch.device,
        num_samples: int,
        norm_means: tuple[float, ...],
        norm_stds: tuple[float, ...],
        test_data: Dataset,
        preprocess_fn: Callable[[Any], tuple[ModelCtx, torch.Tensor]],
        save_to: Optional[ExperimentPath] = None,
    ):
        super().__init__(device, norm_means, norm_stds, save_to)
        self._num_samples = num_samples
        self._dataset = test_data
        self._preprocess_fn = preprocess_fn

        test_elements = []
        for i in range(self._num_samples):
            test_elements.append(self._dataset[i])
        batch = default_collate(test_elements)
        self.ctx, self.trg = self._preprocess_fn(batch)
        self.ctx = self.ctx.to(self._device)
        self.trg = self.trg.to(self._device)

    @torch.no_grad()
    def plot_prediction(self, model: DDPM | CNP | CDNP, step: int = 0) -> None:
        plottables = []
        x_gen = model.make_plot(self.ctx)
        for x in x_gen:
            x = self._unnormalize(x)
            plottables.append(x)
        num_channels = x_gen[0].shape[1]

        mask = self.ctx.image_ctx[:, -1:, :, :]
        masked_x = self.ctx.image_ctx[:, :-1, :, :]

        # Ensure mask has the same number of channels as the other tensors
        mask = mask.expand(-1, num_channels, -1, -1)
        masked_x = self._unnormalize(masked_x)
        trg = self._unnormalize(self.trg)

        plottables = [trg, mask, masked_x] + plottables

        x_gen = torch.cat(plottables, dim=0)
        grid = make_grid(x_gen, nrow=self._num_samples)
        if self._dir:
            save_image(grid, self._get_path(step))
        self._log_to_wandb(grid, step, "inpaint")


class ColourisationPlotter(BasePlotter):
    def __init__(
        self,
        device: str | torch.device,
        num_samples: int,
        norm_means: tuple[float, ...],
        norm_stds: tuple[float, ...],
        test_data: Dataset,
        preprocess_fn: Callable[[Any], tuple[ModelCtx, torch.Tensor]],
        save_to: Optional[ExperimentPath] = None,
    ):
        super().__init__(device, norm_means, norm_stds, save_to)
        self._num_samples = num_samples
        self._dataset = test_data
        self._preprocess_fn = preprocess_fn

        test_elements = []
        for i in range(self._num_samples):
            test_elements.append(self._dataset[i])
        batch = default_collate(test_elements)
        self.ctx, self.trg = self._preprocess_fn(batch)
        self.ctx = self.ctx.to(self._device)
        self.trg = self.trg.to(self._device)

    @torch.no_grad()
    def plot_prediction(self, model: DDPM | CNP | CDNP, step: int = 0) -> None:
        plottables = []
        x_gen = model.make_plot(self.ctx)
        for x in x_gen:
            x = self._unnormalize(x)
            plottables.append(x)

        grayscale = self.ctx.image_ctx
        grayscale = self._unnormalize(grayscale)
        trg = self._unnormalize(self.trg)

        plottables = [trg, grayscale] + plottables

        x_gen = torch.cat(plottables, dim=0)
        grid = make_grid(x_gen, nrow=self._num_samples)
        if self._dir:
            save_image(grid, self._get_path(step))
        self._log_to_wandb(grid, step, "colourisation")


class ForecastPlotter(BasePlotter):
    def __init__(
        self,
        device: str | torch.device,
        num_samples: int,
        test_data: GriddedWeatherTask,
        preprocess_fn: Callable[[Any], tuple[ModelCtx, torch.Tensor]],
        save_to: Optional[ExperimentPath] = None,
    ):
        # TODO: normalisation params
        means = (0.0, 0.0, 0.0)  # Placeholder means
        stds = (1.0, 1.0, 1.0)  # Placeholder stds
        super().__init__(device, means, stds, save_to)
        self._num_samples = num_samples
        self._dataset = test_data
        self._preprocess_fn = preprocess_fn
        self._geoplotter = GeoPlotter()

        test_elements = []
        for i in range(self._num_samples):
            test_elements.append(self._dataset[i])
        batch = default_collate(test_elements)
        self.ctx, self.trg = self._preprocess_fn(batch)
        self.ctx = self.ctx.to(self._device)
        self.trg = self.trg.to(self._device)

    @torch.no_grad()
    def plot_prediction(self, model: DDPM | CNP | CDNP, step: int = 0) -> None:
        plottables = model.make_plot(self.ctx)  # B, C, H, W
        plottables = [self.trg] + plottables
        # Take the first sample from each batch
        plottables = [p[0, ...] for p in plottables]  # C, H, W
        plottables = [p.permute(1, 2, 0) for p in plottables]  # H, W, C

        plottables_flat = []
        for p in plottables:
            for c in range(p.shape[-1]):
                single_channel = p[..., c : c + 1]
                if single_channel.std().item() == 0:
                    # Skip constant channels like time embeddings
                    continue

                if len(plottables_flat) > 20:
                    break

                plottables_flat.append(single_channel)  # H, W, 1

        grid = torch.stack(plottables_flat, dim=-1)
        fig = self._geoplotter.plot_grid(
            data=grid,
            col_titles=[f"Sample {i + 1}" for i in range(grid.shape[-2])],
            row_titles=[f"Channel {i + 1}" for i in range(grid.shape[-1])],
            share_cmap="none",
        )
        if self._dir:
            fig.savefig(self._get_path(step, "forecast_plot"), bbox_inches="tight")
        self._log_to_wandb(fig, step, "forecast")
        plt.close(fig)
