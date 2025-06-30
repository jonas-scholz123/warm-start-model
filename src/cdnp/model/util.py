import torch
from diffusers import UNet2DModel

from cdnp.model.swin.utils import geopad


def round_up_to_power_of_two(n):
    if n < 1:
        raise ValueError("Input must be a positive integer.")
    if (n & (n - 1)) == 0:
        return n  # Already a power of two
    power = 1
    while power < n:
        power <<= 1
    return power


def padded_forward(
    model: UNet2DModel, x: torch.Tensor, *args, **kwargs
) -> torch.Tensor:
    height, width = x.shape[2], x.shape[3]
    padded_height = round_up_to_power_of_two(height)
    padded_width = round_up_to_power_of_two(width)
    if padded_height == height and padded_width == width:
        return model(x, *args, **kwargs).sample  # ty: ignore
    padded_x, pad_left, pad_bottom = geopad(
        x.permute(0, 2, 3, 1), padded_height, padded_width
    )
    padded_x = padded_x.permute(0, 3, 1, 2)
    out = model(padded_x, *args, **kwargs).sample  # ty: ignore
    out = out[:, :, pad_bottom : pad_bottom + height, pad_left : pad_left + width]
    return out
