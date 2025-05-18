# %%
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import torch
from loguru import logger
from matplotlib.animation import FuncAnimation, PillowWriter
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
        ws_test: Sequence[float],
        num_samples: int,
        num_classes: int,
        save_to: Optional[ExperimentPath] = None,
    ):
        self._device = device
        self._test_data = test_data
        self._dir = save_to
        self._ws_test = ws_test
        self._num_samples = num_samples
        self._num_classes = num_classes

    @torch.no_grad()
    def plot_prediction(self, model: DDPM, epoch: int = 0) -> Optional[Figure]:
        logger.info("Making and saving prediction plots")
        num_samples = self._num_samples * self._num_classes
        for w_i, w in enumerate(self._ws_test):
            x_gen, x_gen_store = model.sample(num_samples, (1, 28, 28), guide_w=w)

            x_all = torch.cat([x_gen])
            grid = make_grid(x_all * -1 + 1, nrow=10)
            path = self._dir.at(f"image_ep{epoch}_w{w}.png")
            save_image(grid, path)

            if epoch % 5 == 0:
                self._save_gif(epoch, w, x_gen_store)

    def _save_gif(self, epoch: int, w: float, x_gen_store) -> None:
        # create gif of images evolving over time, based on x_gen_store
        fig, axs = plt.subplots(
            nrows=int(self._num_samples),
            ncols=self._num_classes,
            sharex=True,
            sharey=True,
            figsize=(8, 3),
        )

        def animate_diff(i, x_gen_store):
            plots = []
            for row in range(int(self._num_samples)):
                for col in range(self._num_classes):
                    axs[row, col].clear()
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                    plots.append(
                        axs[row, col].imshow(
                            -x_gen_store[i, (row * self._num_classes) + col, 0],
                            cmap="gray",
                            vmin=(-x_gen_store[i]).min(),
                            vmax=(-x_gen_store[i]).max(),
                        )
                    )
            return plots

        ani = FuncAnimation(
            fig,
            animate_diff,
            fargs=[x_gen_store],
            interval=200,
            blit=False,
            repeat=True,
            frames=x_gen_store.shape[0],
        )

        path = self._dir.at(f"gif_ep{epoch}_w{w}.gif")
        ani.save(
            path,
            dpi=100,
            writer=PillowWriter(fps=5),
        )
