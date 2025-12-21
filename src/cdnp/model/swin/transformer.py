from collections.abc import Callable
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DOutput
from torch import nn

from cdnp.model.swin.embeddings import (
    PositionEmbedding,
    SpatialEmbedding,
)
from cdnp.model.swin.tokeniser import ImageTokeniser
from cdnp.model.swin.transformer_blocks import make_swin_stage
from cdnp.model.swin.utils import geopad
from cdnp.plot.geoplot import GeoPlotter


def interpolate_bilinear_channel_last(
    x: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    out: torch.Tensor = torch.nn.functional.interpolate(
        x.permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return out


class SwinTransformer(nn.Module):
    def __init__(
        self,
        token_dimensions: Sequence[int] | int,
        final_swin_token_dim: int,
        output_dimension: int,
        window_size: int,
        num_heads: int,
        in_channels: int,
        height: int,
        width: int,
        patch_size: int,
        num_blocks_per_stage: Sequence[int],
        feedforward_network: Callable[[int, int], nn.Module],
        pos_embedding: PositionEmbedding | SpatialEmbedding,
        use_efficient_attention: bool = True,
        pad_dont_interpolate: bool = False,
    ):
        super().__init__()

        assert len(num_blocks_per_stage) > 0

        if isinstance(token_dimensions, int):
            first_token_dim = token_dimensions
        else:
            first_token_dim = token_dimensions[0]

        if isinstance(token_dimensions, int):
            num_stages = len(num_blocks_per_stage)
            token_dimensions = [token_dimensions] * (num_stages + 1)

        down_swin_stages = []
        up_swin_stages = []
        patch_downsampling_layers = []
        patch_upsampling_layers = []
        token_in_out_dims = list(zip(token_dimensions[:-1], token_dimensions[1:]))

        for num_blocks, (t1, t2) in zip(
            num_blocks_per_stage,
            token_in_out_dims,
        ):
            patch_downsampling_layers.append(
                nn.Conv2d(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=2,
                    stride=2,
                )
            )

            down_swin_stages.append(
                make_swin_stage(
                    token_dim=t2,
                    feedforward_network=feedforward_network,
                    num_heads=num_heads,
                    num_swin_blocks=num_blocks,
                    window_size=window_size,
                    use_efficient_attention=use_efficient_attention,
                )
            )

        for num_blocks, (t2, t1) in zip(
            num_blocks_per_stage[::-1],
            token_in_out_dims[::-1],
        ):
            up_swin_stages.append(
                make_swin_stage(
                    token_dim=t1,
                    feedforward_network=feedforward_network,
                    num_heads=num_heads,
                    num_swin_blocks=num_blocks,
                    window_size=window_size,
                    use_efficient_attention=use_efficient_attention,
                )
            )
            patch_upsampling_layers.append(
                nn.ConvTranspose2d(
                    in_channels=t1,
                    out_channels=t2,
                    kernel_size=2,
                    stride=2,
                )
            )

        self.down_swin_stages = torch.nn.ModuleList(down_swin_stages)
        self.up_swin_stages = torch.nn.ModuleList(up_swin_stages)
        self.patch_downsampling_layers = torch.nn.ModuleList(patch_downsampling_layers)
        self.patch_upsampling_layers = torch.nn.ModuleList(patch_upsampling_layers)

        self.tokeniser = ImageTokeniser(
            in_channels=in_channels,
            token_dimension=first_token_dim,
            patch_size=patch_size,
        )

        self.embedding = pos_embedding

        self.unpatch_conv = nn.ConvTranspose2d(
            in_channels=token_dimensions[-1],
            out_channels=final_swin_token_dim,
            stride=(patch_size, patch_size),
            kernel_size=(patch_size, patch_size),
        )

        # At the high res, need bigger window size, which somehow lower memory.
        self.final_swin_stage = make_swin_stage(
            token_dim=final_swin_token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            num_swin_blocks=1,
            window_size=2 * window_size,
            use_efficient_attention=use_efficient_attention,
        )

        self.final_conv = nn.Conv2d(
            in_channels=final_swin_token_dim,
            out_channels=output_dimension,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pad_dont_interpolate = pad_dont_interpolate
        self.width = width
        self.height = height
        self.plot_count = 0

    def forward(
        self,
        x: torch.Tensor,
        # TODO: Use these inputs in the future
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> UNet2DOutput:
        """Apply vision transformer to batch of images.

        Arguments:
            x: input image tensor of shape (B, C, H, W)

        Returns:
            output logits tensor of shape (B, C, H, W)
        """
        # Convert from B, C, H, W -> B, H, W, C
        # which is the Otter convention
        x = x.permute(0, 2, 3, 1)

        _, original_height, original_width, _ = x.shape

        if self.pad_dont_interpolate:
            # Pad input to target dimensions
            x, _, _ = geopad(x, target_height=self.height, target_width=self.width)
        else:
            x = interpolate_bilinear_channel_last(
                x, height=self.height, width=self.width
            )

        x = self.tokeniser(x)
        x = self.embedding(x)

        skips = []

        for downsampling_conv, down_swin_stage in zip(
            self.patch_downsampling_layers,
            self.down_swin_stages,
        ):
            skips.append(x)
            x = self.apply_geoconv(downsampling_conv, x)
            for swin_transformer_block in down_swin_stage:
                x, _ = swin_transformer_block(x)

        for upsampling_conv, up_swin_stage, skip in zip(
            self.patch_upsampling_layers,
            self.up_swin_stages,
            skips[::-1],
        ):
            for swin_transformer_block in up_swin_stage:
                x, _ = swin_transformer_block(x)
            x = self.apply_geoupsample(upsampling_conv, x)
            x = x + skip

        x = self.apply_geoconv(self.unpatch_conv, x)
        for swin_transformer_block in self.final_swin_stage:
            x, _ = swin_transformer_block(x)
        x = self.apply_geoconv(self.final_conv, x)

        self.debug_plot(x)

        if self.pad_dont_interpolate:
            # Crop the padding to restore original dimensions
            x = x[:, :original_height, :original_width, :]
        else:
            x = interpolate_bilinear_channel_last(
                x, height=original_height, width=original_width
            )
        self.debug_plot(x)

        x = x.permute(0, 3, 1, 2)  # Convert back to B, C, H, W
        self.plot_count = 0

        return UNet2DOutput(sample=x)

    def apply_geoconv_old(self, conv_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        k_lat, k_lon = conv_layer.kernel_size  # ty: ignore
        pad_lat = k_lat - 1
        pad_lon = k_lon - 1

        pad_lat_top = pad_lat // 2
        pad_lat_bottom = pad_lat - pad_lat_top
        pad_lon_left = pad_lon // 2
        pad_lon_right = pad_lon - pad_lon_left

        self.debug_plot(x.permute(0, 2, 3, 1))

        if pad_lon > 0:
            if pad_lon_left < 1 or pad_lon_right < 1:
                raise ValueError(
                    "Longitude padding must be at least 1 pixel on both sides."
                )
            x_padded_lon = torch.cat(
                [x[..., -pad_lon_left:], x, x[..., :pad_lon_right]], dim=3
            )
        else:
            x_padded_lon = x

        self.debug_plot(x_padded_lon.permute(0, 2, 3, 1))
        if pad_lat > 0:
            x_padded = F.pad(
                x_padded_lon,
                (0, 0, pad_lat_top, pad_lat_bottom),
                mode="constant",
                value=0,
            )
        else:
            x_padded = x_padded_lon

        # The provided conv_layer must have padding=0
        self.debug_plot(x_padded.permute(0, 2, 3, 1))
        output = conv_layer(x_padded)

        return output.permute(0, 2, 3, 1)

    def apply_geoconv2(self, conv_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)

        k_lat, k_lon = conv_layer.kernel_size  # ty: ignore

        # Padding required to center the kernel. For even kernels, we pad more
        # on the 'before' side (top/left) to counteract the kernel's spatial shift.
        pad_lat_top = (k_lat - 1) // 2 + (k_lat % 2 == 0)
        pad_lat_bottom = (k_lat - 1) // 2

        pad_lon_left = (k_lon - 1) // 2 + (k_lon % 2 == 0)
        pad_lon_right = (k_lon - 1) // 2

        # Apply longitude padding (circular)
        if pad_lon_left > 0 or pad_lon_right > 0:
            x = torch.cat([x[..., -pad_lon_left:], x, x[..., :pad_lon_right]], dim=3)

        # Apply latitude padding (zero-padding)
        x = F.pad(x, (0, 0, pad_lat_top, pad_lat_bottom), mode="constant", value=0)

        output = conv_layer(x)

        return output.permute(0, 2, 3, 1)

    def apply_geoconv(self, conv_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        output = conv_layer(x)
        return output.permute(0, 2, 3, 1)

    def apply_geoupsample(
        self, upsampling_layer: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        output = upsampling_layer(x)
        return output.permute(0, 2, 3, 1)

    def apply_geoupsample2(
        self, upsampling_layer: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        self.debug_plot(x.permute(0, 2, 3, 1))
        target_lat = x.shape[2] * upsampling_layer.stride[0]  # ty: ignore
        target_lon = x.shape[3] * upsampling_layer.stride[1]  # ty: ignore

        output = upsampling_layer(x)
        self.debug_plot(output.permute(0, 2, 3, 1))

        current_lat = output.shape[2]
        crop_lat = current_lat - target_lat
        if crop_lat > 0:
            crop_top = crop_lat // 2
            crop_bottom = crop_lat - crop_top
            output = output[:, :, crop_top : current_lat - crop_bottom, :]

        # --- Fix Longitude (Wrapping) ---
        # Symmetrically crop the extra pixels, but add them back to the opposite side.
        current_lon = output.shape[3]
        crop_lon = current_lon - target_lon
        if crop_lon > 0:
            crop_left = crop_lon // 2
            crop_right = crop_lon - crop_left

            # Extract the central part and the spillover from both sides
            center = output[:, :, :, crop_left : current_lon - crop_right]
            left_spill = output[:, :, :, :crop_left]
            right_spill = output[:, :, :, -crop_right:]

            # Add the spillover to the opposite ends of the central part
            center[:, :, :, :crop_right] += right_spill
            center[:, :, :, -crop_left:] += left_spill
            output = center
        self.debug_plot(output.permute(0, 2, 3, 1))

        return output.permute(0, 2, 3, 1)

    def debug_plot(self, x: torch.Tensor) -> None:
        return
        """Plot the input tensor x using GeoPlotter."""
        gp = GeoPlotter()
        fig = gp.plot_single(x[0, :, :, -1])
        fig.savefig(f"debug_{self.plot_count}_input.png")
        plt.close(fig)

        self.plot_count += 1


def make_debug_input(x: torch.Tensor) -> torch.Tensor:
    """Create a debug input tensor with a gradient pattern."""
    return x
    lat, lon = x.shape[1], x.shape[2]

    grid_x = torch.linspace(-1, 1, steps=lon)
    grid_y = torch.linspace(-1, 1, steps=lat)

    # Create 2D gradient by averaging x and y using broadcasting
    grid = (grid_x[None, :] + grid_y[:, None]) / 2  # Shape: (height, width)
    grid = grid[:, :, None]

    grid[...] = 0
    # grid[:, -1, :] = -5
    grid[:, 0, :] = -5
    x[0, :, :, :] = grid

    return x
