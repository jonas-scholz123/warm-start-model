import torch

from cdnp.model.ddpm import ModelCtx


def preprocess_ccgen(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> tuple[ModelCtx, torch.Tensor]:
    x, y = batch
    return ModelCtx(label_ctx=y), x


def preprocess_inpaint(
    batch: tuple[torch.Tensor, torch.Tensor],
    gen: torch.Generator,
    min_frac: float,
    max_frac: float,
) -> tuple[ModelCtx, torch.Tensor]:
    x, _ = batch

    frac = torch.rand(1, generator=gen).item()
    frac = frac * (max_frac - min_frac) + min_frac

    # Generates an "image" of the same shape as x, with values between 0 and 1,
    # and then compares it to frac to create a mask.
    mask = torch.empty_like(x).uniform_(generator=gen) < frac
    x_masked = x * mask
    # Concat along the channel dimension
    ctx = torch.cat([x_masked, mask], dim=1)
    return ModelCtx(image_ctx=ctx), x
