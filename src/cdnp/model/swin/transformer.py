from collections.abc import Callable
from typing import Optional, Sequence, Union

import torch
from diffusers.models.unets.unet_2d import UNet2DOutput
from torch import nn

from cdnp.model.swin.embeddings import (
    PositionEmbedding,
    SpatialEmbedding,
)
from cdnp.model.swin.tokeniser import ImageTokeniser
from cdnp.model.swin.transformer_blocks import make_swin_stage
from cdnp.model.swin.utils import (
    conv_channel_last,
    pad_bottom_right,
)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        token_dimensions: Sequence[int] | int,
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

        self.final_conv = nn.ConvTranspose2d(
            in_channels=token_dimensions[-1],
            out_channels=output_dimension,
            stride=(patch_size, patch_size),
            kernel_size=(patch_size, patch_size),
        )

        self.width = width
        self.height = height

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

        # Pad input to target dimensions
        x = pad_bottom_right(x, target_height=self.height, target_width=self.width)

        x = self.tokeniser(x)
        x = self.embedding(x)

        skips = []

        for downsampling_conv, down_swin_stage in zip(
            self.patch_downsampling_layers,
            self.down_swin_stages,
        ):
            skips.append(x)
            x = downsampling_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            for swin_transformer_block in down_swin_stage:
                x, _ = swin_transformer_block(x)

        for upsampling_conv, up_swin_stage, skip in zip(
            self.patch_upsampling_layers,
            self.up_swin_stages,
            skips[::-1],
        ):
            for swin_transformer_block in up_swin_stage:
                x, _ = swin_transformer_block(x)
            x = upsampling_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = x + skip

        x = conv_channel_last(self.final_conv, x)

        # Crop the padding to restore original dimensions
        x = x[:, :original_height, :original_width, :]

        x = x.permute(0, 3, 1, 2)  # Convert back to B, C, H, W

        return UNet2DOutput(sample=x)
