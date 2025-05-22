import torch
from torch.utils.data import Dataset

from cdnp.task.batch import ModelInput

ClassificationSample = tuple[torch.Tensor, torch.Tensor]


class InpaintingDataset(Dataset):
    def __init__(
        self,
        delegate: Dataset[ClassificationSample],
        min_frac: float,
        max_frac: float,
        gen: torch.Generator,
    ):
        self.delegate = delegate
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.gen = gen

    def __len__(self):
        return len(self.delegate)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = self.delegate[idx]

        frac = torch.rand(1, generator=self.gen).item()
        frac = frac * (self.max_frac - self.min_frac) + self.min_frac

        # Generates an "image" of the same shape as x, with values between 0 and 1,
        # and then compares it to frac to create a mask.
        mask = torch.empty_like(x).uniform_(generator=self.gen) < frac

        x_masked = x * mask
        return x_masked, mask, x


def preprocess_inpainting_batch(
    x_masked: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
) -> ModelInput:
    # Concat along the channel dimension
    ctx = torch.cat([x_masked, mask], dim=1)
    return ModelInput(trg=x, image_ctx=ctx)
