import torch
import torch.nn.functional as F
from torch import nn


def conv_channel_last(conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return out


def pad_bottom_right(
    x: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    """
    Pads a channel-last tensor (B, H, W, C) to target_height and target_width
    by adding padding only to the bottom and right sides.
    """
    input_height: int = x.shape[1]
    input_width: int = x.shape[2]

    pad_h: int = target_height - input_height
    pad_w: int = target_width - input_width

    # F.pad expects padding in reverse order of dimensions:
    # (..., pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
    # We only pad right (W) and bottom (H).
    # Input is (B, H, W, C) -> Dims (0, 1, 2, 3)
    # Padding order for F.pad: (pad_C_left, pad_C_right,
    # pad_W_left, pad_W_right, pad_H_top, pad_H_bottom)
    padding_tuple: tuple[int, ...] = (
        0,
        0,  # Pad C (dim 3) - No padding
        0,
        pad_w,  # Pad W (dim 2) - Pad only right
        0,
        pad_h,  # Pad H (dim 1) - Pad only bottom
    )

    out: torch.Tensor = F.pad(x, pad=padding_tuple, mode="constant", value=0.0)
    return out


def geopad(x: torch.Tensor, target_height: int, target_width: int) -> tuple[torch.Tensor, int, int]:
    """
    Pads a channel-last tensor (B, H, W, C) to target_height and target_width by adding
    wrap-around padding to the left and right, and reflective padding to the bottom.
    """

    x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W) for padding

    _, _, input_height, input_width = x.shape

    # Validate inputs
    if target_height < input_height or target_width < input_width:
        raise ValueError(
            "Target dimensions must be greater than or equal to input dimensions."
        )

    # 1. Pad the width dimension with wrap-around padding
    pad_w = target_width - input_width
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_left < 1 or pad_right < 1:
        raise ValueError("Width padding must be at least 1 pixel on both sides.")

    # left_chunk = x[:, :, -pad_left:, :]  # Last `pad_left` columns
    # right_chunk = x[:, :, :pad_right, :]  # First `pad_right` columns
    # x_padded_width = torch.cat((left_chunk, x, right_chunk), dim=2)

    pad = (pad_left, pad_right, 0, 0)
    x = F.pad(x, pad=pad, mode="circular")

    # 2. Pad the height dimension with reflective padding at the bottom/top
    # TODO: This isn't quite right, e.g. Argentina should wrap to reflected Australia.
    pad_h = target_height - input_height
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    if pad_top < 1 or pad_bottom < 1:
        raise ValueError("Height padding must be at least 1 pixel on both sides.")

    pad = (0, 0, pad_top, pad_bottom)  # 0, 0 => no more padding on width
    x = F.pad(x, pad=pad, mode="reflect")

    return x.permute(0, 2, 3, 1), pad_left, pad_bottom
