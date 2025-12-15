import torch
from datasets import load_dataset
from mlbnb.types import Split
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from config.config import Paths

DataType = tuple[torch.Tensor, int]


class AfhqDataset(Dataset):
    HF_REPO_ID = "huggan/AFHQ"

    def __init__(
        self,
        paths: Paths,
        split: Split,
        generator: torch.Generator,
        val_fraction: float = 0.1,
        norm_means: tuple[float, ...] = (0.5, 0.5, 0.5),
        norm_stds: tuple[float, ...] = (0.5, 0.5, 0.5),
    ):
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_means, std=norm_stds),
            ]
        )

        self.dataset: Dataset = load_dataset(
            self.HF_REPO_ID, split="train", cache_dir=str(paths.data)
        )

        ds_len = len(self.dataset)  # type: ignore
        n_val = int(val_fraction * ds_len)
        train, val = random_split(
            self.dataset, [ds_len - n_val, n_val], generator=generator
        )
        self.dataset = train if split == Split.TRAIN else val

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> DataType:  # type: ignore
        item = self.dataset[idx]  # type: ignore
        image = item["image"]
        label = item["label"]
        image = self.transform(image)

        return image, label  # type: ignore
