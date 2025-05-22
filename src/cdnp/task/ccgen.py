import torch

from cdnp.model.ddpm import ModelInput


def preprocess(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> ModelInput:
    x, y = batch
    return ModelInput(trg=x, label_ctx=y)
