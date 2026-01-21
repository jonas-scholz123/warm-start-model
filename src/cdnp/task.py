from typing import Callable

import torch
from torch.nn.functional import interpolate

from cdnp.model.ddpm import ModelCtx

PreprocessFn = Callable[
    [tuple[torch.Tensor, torch.Tensor]], tuple[ModelCtx, torch.Tensor]
]


def preprocess_ccgen(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> tuple[ModelCtx, torch.Tensor]:
    x, y = batch
    return ModelCtx(label_ctx=y), x


def preprocess_imgen(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> tuple[ModelCtx, torch.Tensor]:
    x, _ = batch
    return ModelCtx(), x


def preprocess_inpaint(
    batch: tuple[torch.Tensor, torch.Tensor],
    gen: torch.Generator,
    min_frac: float,
    max_frac: float,
    seed: int | None = None,
) -> tuple[ModelCtx, torch.Tensor]:
    x, _ = batch

    if seed is not None:
        gen = torch.Generator().manual_seed(seed)

    frac = torch.rand(1, generator=gen).item()
    frac = frac * (max_frac - min_frac) + min_frac

    # Generates an "image" of the same shape as x, with values between 0 and 1,
    # and then compares it to frac to create a mask.
    single_channel_x = x[:, 0:1, :, :]
    mask = torch.empty_like(single_channel_x).uniform_(generator=gen) < frac
    x_masked = x * mask
    # Concat along the channel dimension
    ctx = torch.cat([x_masked, mask], dim=1)
    return ModelCtx(image_ctx=ctx), x


def preprocess_colourisation(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> tuple[ModelCtx, torch.Tensor]:
    x, _ = batch
    # Convert to grayscale by averaging the color channels
    grayscale = x.mean(dim=1, keepdim=True)
    return ModelCtx(image_ctx=grayscale), x


def preprocess_superresolution(
    batch: tuple[torch.Tensor, torch.Tensor],
    scale_factor: int,
) -> tuple[ModelCtx, torch.Tensor]:
    x, _ = batch
    B, C, H, W = x.shape

    low_res_H = H // scale_factor
    low_res_W = W // scale_factor

    low_res = interpolate(x, size=(low_res_H, low_res_W), mode="bicubic")
    low_res_upsampled = interpolate(low_res, size=(H, W), mode="bicubic")

    return ModelCtx(image_ctx=low_res_upsampled), x


def preprocess_weather_forecast(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[ModelCtx, torch.Tensor]:
    zero_time, static, dyn, trg = batch

    B, lat, lon, time, var = dyn.shape

    # For now (no time embeddings), just fold the time dimension into the var dimension.
    dyn = dyn.reshape(B, lat, lon, time * var)

    B, lat, lon, time, var = trg.shape
    if time > 1:
        raise NotImplementedError("Only single target time step is supported for now.")
    trg = trg.reshape(B, lat, lon, time * var)

    # Match convention of other datasets to have channels first.

    # Convert to (B, time*var, lat, lon)
    dyn = dyn.permute(0, 3, 1, 2)
    trg = trg.permute(0, 3, 1, 2)

    # Have dyn at the end, because it's used as the residual.

    ctx = torch.cat([static, dyn], dim=1)  # (B, static+dyn, lat, lon)

    return ModelCtx(image_ctx=ctx), trg


def preprocess_weather_inpaint(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    gen: torch.Generator,
    min_frac: float,
    max_frac: float,
    seed: int | None = None,
) -> tuple[ModelCtx, torch.Tensor]:
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)

    zero_time, static, dyn, _ = batch

    B, lat, lon, time, var = dyn.shape

    # For now (no time embeddings), just fold the time dimension into the var dimension.
    dyn = dyn.reshape(B, lat, lon, time * var)

    # Match convention of other datasets to have channels first.

    # Convert to (B, time*var, lat, lon)
    dyn = dyn.permute(0, 3, 1, 2)

    frac = torch.rand(1, generator=gen).item()
    frac = frac * (max_frac - min_frac) + min_frac

    # Generates an "image" of the same shape as x, with values between 0 and 1,
    # and then compares it to frac to create a mask.
    single_channel_x = dyn[:, 0:1, :, :]
    mask = torch.empty_like(single_channel_x).uniform_(generator=gen) < frac
    dyn_masked = dyn * mask

    # Have dyn at the end, because it's used as the residual.

    ctx = torch.cat(
        [mask, static, dyn_masked], dim=1
    )  # (B, mask+static+dyn+mask, lat, lon)

    return ModelCtx(image_ctx=ctx), dyn
