# %%
from typing import Optional

import torch
from loguru import logger
from matplotlib.figure import Figure
from mlbnb.paths import ExperimentPath
from torch.utils.data import Dataset
from torchvision.utils import make_grid, save_image

from cdnp.model.ddpm import DDPM


class Plotter:
    def __init__(
        self,
        device: str | torch.device,
        test_data: Dataset,
        num_samples: int,
        num_classes: int,
        num_channels: int,
        sidelength: int,
        save_to: Optional[ExperimentPath] = None,
    ):
        self._device = device
        self._test_data = test_data
        self._dir = save_to
        self._num_samples = num_samples
        self._num_classes = num_classes
        self._num_channels = num_channels
        self._sidelength = sidelength

    @torch.no_grad()
    def plot_prediction(self, model: DDPM, epoch: int = 0) -> Optional[Figure]:
        logger.info("Making and saving prediction plots")
        class_labels = (
            torch.arange(self._num_classes)
            .repeat(self._num_samples, 1)
            .flatten()
            .to(self._device)
        )

        total_samples = self._num_samples * self._num_classes
        x_gen = model.sample(
            (total_samples, self._num_channels, self._sidelength, self._sidelength),
            class_labels,
        )

        grid = make_grid(x_gen * -1 + 1, nrow=self._num_classes)
        path = self._dir.at(f"image_ep{epoch}.png")
        save_image(grid, path)
