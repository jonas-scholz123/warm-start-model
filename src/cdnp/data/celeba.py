import torch
from mlbnb.types import Split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from config.config import Paths


class CelebADataset(Dataset):
    def __init__(
        self,
        paths: Paths,
        val_fraction: float,
        split: Split,
        generator: torch.Generator,
        norm_means: tuple[float],
        norm_stds: tuple[float],
    ):
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_means, std=norm_stds),
            ]
        )

        match split:
            case Split.TRAIN:
                split_str = "train"
            case Split.VAL:
                split_str = "valid"
            case Split.TEST:
                split_str = "test"

        self.dataset = datasets.CelebA(
            root=paths.data,
            split=split_str,
            download=True,
            transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
