# %%
import torch
from mlbnb.types import Split
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from config.config import Paths


class Cifar10Dataset(Dataset):
    def __init__(
        self,
        paths: Paths,
        val_fraction: float,
        split: Split,
        norm_means: tuple[float, float, float],
        norm_stds: tuple[float, float, float],
        generator: torch.Generator,
    ):
        self.device = generator.device
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=norm_means,
                    std=norm_stds,
                ),
            ]
        )

        if split == Split.TEST:
            self.dataset = datasets.CIFAR10(
                root=paths.data,
                train=False,
                download=True,
                transform=transform,
            )
        else:
            dataset = datasets.CIFAR10(
                root=paths.data,
                train=True,
                download=True,
                transform=transform,
            )

            n_val = int(val_fraction * len(dataset))
            train, val = random_split(
                dataset, [len(dataset) - n_val, n_val], generator=generator
            )
            self.dataset = train if split == Split.TRAIN else val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
