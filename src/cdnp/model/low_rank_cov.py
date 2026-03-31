"""Low-rank normalization utilities.

Defines a normalizing transform L = diag(d) + V Vᵀ that operates jointly
across channels and spatial dimensions. Uses the Woodbury matrix identity
for efficient inversion.
"""

import torch
import torch.amp
from torch import Tensor


def normalize(
    x: Tensor,
    mean: Tensor,
    d: Tensor,
    V: Tensor,
) -> Tensor:
    """Normalize: z = L⁻¹(x - mean) where L = diag(d) + V Vᵀ.

    Uses the Woodbury identity for efficient inversion:
        L⁻¹ = D⁻¹ − D⁻¹V(I + VᵀD⁻¹V)⁻¹VᵀD⁻¹

    Args:
        x: Data to normalize, shape (B, C, H, W).
        mean: Per-pixel mean, shape (B, C, H, W) or broadcastable.
        d: Diagonal factor (positive), shape (B, C, H, W) or broadcastable.
        V: Low-rank factors, shape (B, rank, C, H, W).

    Returns:
        Normalized data z, shape (B, C, H, W).
    """
    # Run in float32 — linalg.solve does not support float16.
    input_dtype = x.dtype
    with torch.amp.autocast("cuda", enabled=False):
        x = x.float()
        mean = mean.float()
        d = d.float()
        V = V.float()

        B = x.shape[0]
        rank = V.shape[1]
        N = x[0].numel()  # C*H*W

        # Flatten (C, H, W) -> (N,) where N = C*H*W
        x_flat = (x - mean).reshape(B, N)  # (B, N)
        d_flat = d.reshape(B, N)  # (B, N) or broadcastable
        V_flat = V.reshape(B, rank, N)  # (B, rank, N)

        # D⁻¹ x
        d_inv = 1.0 / d_flat  # (B, N)
        D_inv_x = d_inv * x_flat  # (B, N)

        # D⁻¹ V  -> (B, N, rank)
        D_inv_V = d_inv.unsqueeze(-1) * V_flat.permute(0, 2, 1)  # (B, N, rank)

        # Inner matrix: I + Vᵀ D⁻¹ V  -> (B, rank, rank)
        inner = torch.eye(rank, device=x.device).unsqueeze(0)
        inner = inner + torch.bmm(V_flat, D_inv_V)  # (B, rank, rank)

        # (I + Vᵀ D⁻¹ V)⁻¹ Vᵀ D⁻¹ x  -> (B, rank)
        Vt_Dinv_x = torch.bmm(V_flat, D_inv_x.unsqueeze(-1)).squeeze(-1)  # (B, rank)
        correction_coeff = torch.linalg.solve(inner, Vt_Dinv_x)  # (B, rank)

        # D⁻¹ V @ correction_coeff  -> (B, N)
        correction = torch.bmm(D_inv_V, correction_coeff.unsqueeze(-1)).squeeze(-1)

        z_flat = D_inv_x - correction  # (B, N)
        return z_flat.reshape_as(x).to(input_dtype)


def log_det_L(
    d: Tensor,
    V: Tensor,
) -> Tensor:
    """Log-determinant of L = diag(d) + V Vᵀ via the matrix determinant lemma.

    log det(L) = log det(diag(d)) + log det(I + Vᵀ D⁻¹ V)
               = Σ log(dᵢ) + log det(I + Vᵀ D⁻¹ V)

    The inner matrix is (rank × rank), so this is cheap.

    Args:
        d: Diagonal factor (positive), shape (B, C, H, W).
        V: Low-rank factors, shape (B, rank, C, H, W).

    Returns:
        Log-determinant per sample, shape (B,).
    """
    # Run in float32 — slogdet does not support float16.
    input_dtype = d.dtype
    with torch.amp.autocast("cuda", enabled=False):
        d = d.float()
        V = V.float()

        B = d.shape[0]
        rank = V.shape[1]
        N = d[0].numel()

        d_flat = d.reshape(B, N)  # (B, N)
        V_flat = V.reshape(B, rank, N)  # (B, rank, N)

        # Σ log(dᵢ)
        log_det_diag = d_flat.log().sum(dim=-1)  # (B,)

        if rank == 0:
            return log_det_diag.to(input_dtype)

        # I + Vᵀ D⁻¹ V  -> (B, rank, rank)
        d_inv = 1.0 / d_flat  # (B, N)
        D_inv_V = d_inv.unsqueeze(-1) * V_flat.permute(0, 2, 1)  # (B, N, rank)
        inner = torch.eye(rank, device=d.device).unsqueeze(0)
        inner = inner + torch.bmm(V_flat, D_inv_V)  # (B, rank, rank)

        # log det of (rank × rank) matrix
        log_det_inner = torch.linalg.slogdet(inner).logabsdet  # (B,)

        return (log_det_diag + log_det_inner).to(input_dtype)


def nll(
    x: Tensor,
    mean: Tensor,
    d: Tensor,
    V: Tensor,
) -> Tensor:
    """Negative log-likelihood under N(mean, LLᵀ) where L = diag(d) + VVᵀ.

    NLL = ½(‖z‖² + 2·log det(L) + N·log(2π))
    where z = L⁻¹(x - mean).

    Args:
        x: Data, shape (B, C, H, W).
        mean: Per-pixel mean, shape (B, C, H, W).
        d: Diagonal factor (positive), shape (B, C, H, W).
        V: Low-rank factors, shape (B, rank, C, H, W).

    Returns:
        Scalar mean NLL across batch and dimensions.
    """
    N = x[0].numel()

    z = normalize(x, mean, d, V)
    z_sq = (z * z).reshape(x.shape[0], -1).sum(dim=-1)  # (B,)
    log_det = log_det_L(d, V)  # (B,)

    # Per-sample NLL, then mean over batch
    nll_per_sample = 0.5 * (z_sq + 2.0 * log_det + N * torch.log(torch.tensor(2.0 * torch.pi)))
    # Normalize by N to match scale of diagonal NLL (which averages over dimensions)
    return (nll_per_sample / N).mean()


def unnormalize(
    z: Tensor,
    mean: Tensor,
    d: Tensor,
    V: Tensor,
) -> Tensor:
    """Unnormalize: x = L z + mean where L = diag(d) + V Vᵀ.

    Direct computation (no inversion needed):
        L z = d ⊙ z + V(Vᵀz)

    Args:
        z: Normalized data, shape (B, C, H, W).
        mean: Per-pixel mean, shape (B, C, H, W) or broadcastable.
        d: Diagonal factor (positive), shape (B, C, H, W) or broadcastable.
        V: Low-rank factors, shape (B, rank, C, H, W).

    Returns:
        Unnormalized data x, shape (B, C, H, W).
    """
    B = z.shape[0]
    rank = V.shape[1]
    N = z[0].numel()  # C*H*W

    z_flat = z.reshape(B, N)  # (B, N)
    d_flat = d.reshape(B, N)  # (B, N) or broadcastable
    V_flat = V.reshape(B, rank, N)  # (B, rank, N)

    # d ⊙ z
    dz = d_flat * z_flat  # (B, N)

    # V(Vᵀz) = V @ (V @ z)
    Vt_z = torch.bmm(V_flat, z_flat.unsqueeze(-1)).squeeze(-1)  # (B, rank)
    VVt_z = torch.bmm(V_flat.permute(0, 2, 1), Vt_z.unsqueeze(-1)).squeeze(-1)  # (B, N)

    x_flat = dz + VVt_z  # (B, N)
    return x_flat.reshape_as(z) + mean
