from typing import (
    Callable,
    List,
    Optional,
    Tuple,
)

import torch
from torch import nn

from cdnp.model.swin.moe import LoadBalancingLosses


def extract_subimages(x: torch.Tensor, subimage_size: int) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x`.

    Arguments:
        x: input tensor of shape
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H//S, S, W//S, S, D)
    """
    B, H, W, D = x.shape
    S = subimage_size

    assert H % S == 0, f"Height {H} must be divisible by window size {S} ({x.shape=})"
    assert W % S == 0, f"Width {W} must be divisible by window size {S} ({x.shape=})"

    return torch.reshape(x, [B, H // S, S, W // S, S, D])


def extract_subimages_and_fold_into_batch_dim(
    x: torch.Tensor,
    subimage_size: int,
) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x` and fold the
    spatial dimensions into the batch dimension.

    Arguments:
        x: input tensor of shape (B, H, W, D)
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B*(H//S)*(W//S), S**2, D)
    """
    B, H, W, D = x.shape
    S = subimage_size
    x = extract_subimages(x, subimage_size)  # (B, H//S, S, W//S, S, D)
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, W//S, S, S, D)
    return torch.reshape(x, [B * (H // S) * (W // S), S**2, D])


def extract_subimages_and_fold_into_token_dim(
    x: torch.Tensor,
    subimage_size: int,
) -> torch.Tensor:
    """Extract non-overlapping windows from input tensor `x` and fold the
    spatial dimensions into the token dimension.

    Arguments:
        x: input tensor of shape (B, H, W, D)
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """
    B, H, W, D = x.shape
    S = subimage_size
    x = extract_subimages(x, subimage_size)  # (B, H//S, S, W//S, S, D)
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, W//S, S, S, D)
    return torch.reshape(x, [B, H // S, W // S, S**2 * D])


def combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Combine subimages extracted from input tensor `x`.

    Arguments:
        x: input tensor of shape (B, H//S, S, W//S, S, D)
        original_shape: shape of original tensor

    Returns:
        output tensor of shape (B, H, W, D)
    """
    B, H, W, D = original_shape
    S = subimage_size

    assert H % S == 0, f"Height {H} must be divisible by window size {S}"
    assert W % S == 0, f"Width {W} must be divisible by window size {S}"

    return torch.reshape(x, [B, H, W, D])


def unfold_from_batch_dim_and_combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Unfold the batch dimension of input tensor `x` and combine
    the subimages.

    Arguments:
        x: input tensor of shape (B*(H//S)*(W//S), S**2, D)
        original_shape: shape of original tensor
        subimage_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """

    B, H, W, D = original_shape
    S = subimage_size

    x = torch.reshape(x, [B, H // S, W // S, S, S, D])
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, S, W//S, S, D)
    return combine_subimages(x, original_shape, S)


def unfold_from_token_dim_and_combine_subimages(
    x: torch.Tensor,
    original_shape: torch.Size | Tuple[int, int, int, int],
    subimage_size: int,
) -> torch.Tensor:
    """Unfold the token dimension of input tensor `x` and combine
    the subimages.

    Arguments:
        x: input tensor of shape (B, H // S, W // S, S**2 * D)
        original_shape: shape of original tensor
        window_size: size of the extracted subimages

    Returns:
        output tensor of shape (B, H, W, D)
    """

    B, H, W, D = original_shape
    S = subimage_size

    x = torch.reshape(x, [B, H // S, W // S, S, S, D])
    x = torch.permute(x, [0, 1, 3, 2, 4, 5])  # (B, H//S, S, W//S, S, D)
    return combine_subimages(x, original_shape, S)


def shift_horizontally_and_vertically(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift windows in the input tensor `x` by shift along its width and
    height. For example, using shift == 1 (and fixing an index for the B and D
    dimensions), the corresponding image would change as follows:

                      Original                Shifted
                 -----------------       -----------------
                |  x   x   x   o  |     |  *   +   +   +  |
                |  x   x   x   o  |     |  o   x   x   x  |
                |  x   x   x   o  |     |  o   x   x   x  |
                |  +   +   +   *  |     |  o   x   x   x  |
                 -----------------       -----------------

    Arguments:
        x: input tensor of shape (B, H, W, D)
        shift: amount of shift to apply

    Returns:
        output tensor of shape (B, H, W, D)
    """
    if len(x.shape) != 4:
        raise ValueError("Input tensor must have shape (B, H, W, D)")
    return torch.roll(torch.roll(x, shift, dims=1), shift, dims=2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        feedforward_network: Callable[[int, int], nn.Module],
        num_heads: int,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        self.mhsa: Callable[[torch.Tensor], torch.Tensor]

        self.mha_layer = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
        )
        self.mhsa = lambda x: (self.mha_layer(x, x, x)[0])

        self.ffn = feedforward_network(token_dim, token_dim)

        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[LoadBalancingLosses]]:
        """Apply the transformer block to input tokens `x`.

        Args:
            x: input tensor of shape (B, N, D)

        Returns:
            output tensor of shape (B, N, D)
        """

        x = x + self.mhsa(self.ln1(x))
        x_ffn, load_balancing_losses = self.ffn(self.ln2(x))
        x = x + x_ffn

        return x, load_balancing_losses


class SwinTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        feedforward_network: Callable[[int, int], nn.Module],
        num_heads: int,
        window_size: int,
        use_efficient_attention: bool = True,
    ):
        super().__init__()

        # Transformer block applied before the shift operation.
        self.first_block = TransformerBlock(
            token_dim=token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
        )

        # Transformer block applied after the shift operation.
        self.second_block = TransformerBlock(
            token_dim=token_dim,
            feedforward_network=feedforward_network,
            num_heads=num_heads,
            use_efficient_attention=use_efficient_attention,
        )

        # Window size for extracting subimages.
        self.window_size = window_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, List[LoadBalancingLosses]]:
        """Apply the Swin Transformer block to input tokens `x`.

        Arguments:
            x: input tensor of shape (B, H, W, D)

        Returns:
            output tensor of shape (B, H, W, D)
        """
        original_shape = x.shape
        S = self.window_size

        load_balancing_losses_list: List[LoadBalancingLosses] = list()
        # Apply first transformer block, extracting windows, applying
        # the transformer block to them, and re-combining them to the
        # original image.
        x = extract_subimages_and_fold_into_batch_dim(x, S)  # (B*H//S*W//S, S**2, D)
        x, load_balancing_losses = self.first_block(x)  # (B*H//S*W//S, S**2, D)

        if load_balancing_losses is not None:
            load_balancing_losses_list.append(load_balancing_losses)

        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)

        # Apply second transformer block same as the first block, but
        # shifting the windows before and after the block.
        x = shift_horizontally_and_vertically(x, S // 2)  # (B, H, W, D)
        x = extract_subimages_and_fold_into_batch_dim(x, S)  # (B*H//S*W//S, S**2, D)

        x, load_balancing_losses = self.second_block(x)  # (B*H//S*W//S, S**2, D)

        if load_balancing_losses is not None:
            load_balancing_losses_list.append(load_balancing_losses)

        x = unfold_from_batch_dim_and_combine_subimages(
            x, original_shape, S
        )  # (B, H, W, D)
        x = shift_horizontally_and_vertically(x, -(S // 2))  # (B, H, W, D)

        return x, load_balancing_losses_list


def make_swin_stage(
    token_dim: int,
    feedforward_network: Callable[[int, int], nn.Module],
    num_heads: int,
    num_swin_blocks: int,
    window_size: int,
    use_efficient_attention: bool = True,
) -> torch.nn.ModuleList:
    return torch.nn.ModuleList(
        [
            SwinTransformerBlock(
                token_dim=token_dim,
                feedforward_network=feedforward_network,
                num_heads=num_heads,
                window_size=window_size,
                use_efficient_attention=use_efficient_attention,
            )
            for _ in range(num_swin_blocks)
        ]
    )
