# %%
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from loguru import logger
from mlbnb.paths import ExperimentPath
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.utils import make_grid, save_image

from cdnp.model.cdnp import CDNP
from cdnp.model.cnp import CNP
from cdnp.model.ddpm import DDPM, ModelCtx


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
    def plot_prediction(self, model: DDPM, epoch: int = 0) -> None:
        """
        Generate and save prediction plots using the provided model.

        :param model: The DDPM model to use for generating predictions.
        :param epoch: The current epoch number, used for naming the saved plot.
        """
        pass

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the image tensor using the provided means and stds.
        """
        x = x * self._norm_stds + self._norm_means
        return x.clamp(0, 1)

    def _get_path(self, epoch: int, filename: str = "image") -> Path:
        """
        Generate a path for saving the plot image.
        """
        return self._dir.at(f"{filename}_ep{epoch}.png")


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
    def plot_prediction(self, model: DDPM, epoch: int = 0) -> None:
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
        save_image(grid, self._get_path(epoch))


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
        self.trg = self._unnormalize(self.trg)

    @torch.no_grad()
    def plot_prediction(self, model: DDPM | CNP | CDNP, epoch: int = 0) -> None:
        x_gen = model.sample(self.ctx, self._num_samples)

        mask = self.ctx.image_ctx[:, -1:, :, :]
        masked_x = self.ctx.image_ctx[:, :-1, :, :]

        num_channels = x_gen.shape[1]

        # Ensure mask has the same number of channels as the other tensors
        mask = mask.expand(-1, num_channels, -1, -1)

        x_gen = self._unnormalize(x_gen)
        masked_x = self._unnormalize(masked_x)

        x_gen = torch.cat([self.trg, mask, masked_x, x_gen], dim=0)
        grid = make_grid(x_gen, nrow=self._num_samples)
        save_image(grid, self._get_path(epoch))
