"""Unit tests for low-rank covariance normalization."""

import torch
import pytest

from cdnp.model.low_rank_cov import log_det_L, nll, normalize, unnormalize


class TestLowRankNormalizer:
    """Tests for the low-rank normalizer functions."""

    @pytest.fixture
    def shapes(self):
        """Standard test shapes."""
        return {"B": 2, "C": 3, "H": 8, "W": 10, "rank": 4}

    @pytest.fixture
    def data(self, shapes):
        """Generate test data with standard shapes."""
        B, C, H, W, rank = shapes["B"], shapes["C"], shapes["H"], shapes["W"], shapes["rank"]
        torch.manual_seed(42)
        return {
            "x": torch.randn(B, C, H, W),
            "mean": torch.randn(B, C, H, W),
            "d": torch.rand(B, C, H, W) + 0.5,  # positive diagonal
            "V": torch.randn(B, rank, C, H, W) * 0.1,  # small low-rank factors
        }

    def test_round_trip_identity(self, data):
        """unnormalize(normalize(x)) should recover x."""
        x, mean, d, V = data["x"], data["mean"], data["d"], data["V"]

        z = normalize(x, mean, d, V)
        x_reconstructed = unnormalize(z, mean, d, V)

        torch.testing.assert_close(x_reconstructed, x, atol=1e-4, rtol=1e-4)

    def test_round_trip_identity_large_v(self):
        """Round-trip should work even with larger V factors."""
        B, C, H, W, rank = 2, 2, 4, 6, 3
        torch.manual_seed(123)
        x = torch.randn(B, C, H, W)
        mean = torch.randn(B, C, H, W)
        d = torch.rand(B, C, H, W) + 1.0
        V = torch.randn(B, rank, C, H, W) * 0.5

        z = normalize(x, mean, d, V)
        x_reconstructed = unnormalize(z, mean, d, V)

        torch.testing.assert_close(x_reconstructed, x, atol=1e-4, rtol=1e-4)

    def test_rank_zero_fallback_normalize(self):
        """With V of rank 0 shape, normalize should behave like diagonal."""
        B, C, H, W = 2, 3, 8, 10
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W)
        mean = torch.randn(B, C, H, W)
        d = torch.rand(B, C, H, W) + 0.5
        V = torch.zeros(B, 0, C, H, W)  # rank=0

        # With V empty, should not crash but won't be identical to (x-mean)/d
        # because V@V^T = 0, so L = diag(d), L^{-1} = diag(1/d)
        # Actually it should work: I + V^T D^{-1} V = I (0x0), correction = 0
        # Let's verify via unnormalize round-trip instead
        z_diagonal = (x - mean) / d
        z_low_rank = normalize(x, mean, d, V)
        torch.testing.assert_close(z_low_rank, z_diagonal, atol=1e-5, rtol=1e-5)

    def test_rank_zero_fallback_unnormalize(self):
        """With V of rank 0, unnormalize should behave like diagonal."""
        B, C, H, W = 2, 3, 8, 10
        torch.manual_seed(42)
        z = torch.randn(B, C, H, W)
        mean = torch.randn(B, C, H, W)
        d = torch.rand(B, C, H, W) + 0.5
        V = torch.zeros(B, 0, C, H, W)

        x_diagonal = z * d + mean
        x_low_rank = unnormalize(z, mean, d, V)
        torch.testing.assert_close(x_low_rank, x_diagonal, atol=1e-5, rtol=1e-5)

    def test_output_shapes(self, data):
        """Output shapes should match input shapes."""
        x, mean, d, V = data["x"], data["mean"], data["d"], data["V"]

        z = normalize(x, mean, d, V)
        assert z.shape == x.shape

        x_back = unnormalize(z, mean, d, V)
        assert x_back.shape == x.shape

    def test_normalize_gradient_flows(self, data):
        """Gradients should flow through normalize."""
        x = data["x"].requires_grad_(True)
        mean = data["mean"]
        d = data["d"].requires_grad_(True)
        V = data["V"].requires_grad_(True)

        z = normalize(x, mean, d, V)
        loss = z.sum()
        loss.backward()

        assert x.grad is not None
        assert d.grad is not None
        assert V.grad is not None

    def test_unnormalize_gradient_flows(self, data):
        """Gradients should flow through unnormalize."""
        z = data["x"].requires_grad_(True)
        mean = data["mean"]
        d = data["d"].requires_grad_(True)
        V = data["V"].requires_grad_(True)

        x = unnormalize(z, mean, d, V)
        loss = x.sum()
        loss.backward()

        assert z.grad is not None
        assert d.grad is not None
        assert V.grad is not None

    def test_manual_small_example(self):
        """Verify against manual computation with a tiny example."""
        # 1 batch, 1 channel, 1x2 spatial (N=2)
        B, C, H, W, rank = 1, 1, 1, 2, 1
        x = torch.tensor([[[[3.0, 5.0]]]])  # (1, 1, 1, 2)
        mean = torch.tensor([[[[1.0, 2.0]]]])
        d = torch.tensor([[[[2.0, 3.0]]]])
        V = torch.tensor([[[[[0.5, 0.3]]]]]).reshape(B, rank, C, H, W)

        # L = diag([2, 3]) + [0.5; 0.3] @ [0.5, 0.3]
        # L = [[2.25, 0.15], [0.15, 3.09]]
        # x - mean = [2, 3]
        # L^{-1} (x-mean) via Woodbury:
        # D^{-1} = diag([0.5, 1/3])
        # D^{-1}(x-mean) = [1.0, 1.0]
        # D^{-1}V = [0.25; 0.1]
        # inner = 1 + V^T D^{-1} V = 1 + 0.5*0.25 + 0.3*0.1 = 1.155
        # V^T D^{-1} (x-mean) = 0.5*1.0 + 0.3*1.0 = 0.8  (wait, V is (rank, N) = (1, 2))

        # Let me compute manually:
        centered = x - mean  # [2, 3]
        d_flat = d.reshape(-1)  # [2, 3]
        v_flat = V.reshape(1, -1)  # [0.5, 0.3]

        d_inv = 1.0 / d_flat  # [0.5, 1/3]
        d_inv_x = d_inv * centered.reshape(-1)  # [1.0, 1.0]
        d_inv_v = d_inv * v_flat.reshape(-1)  # [0.25, 0.1]
        inner = 1.0 + (v_flat.reshape(-1) * d_inv_v).sum()  # 1 + 0.125 + 0.0333 = 1.1583
        vt_dinv_x = (v_flat.reshape(-1) * d_inv_x).sum()  # 0.5 + 0.3333 = 0.8333
        correction = d_inv_v * (vt_dinv_x / inner)
        expected_z = d_inv_x - correction

        z = normalize(x, mean, d, V)
        torch.testing.assert_close(z.reshape(-1), expected_z, atol=1e-5, rtol=1e-5)

        # Round-trip
        x_back = unnormalize(z, mean, d, V)
        torch.testing.assert_close(x_back, x, atol=1e-5, rtol=1e-5)


class TestLogDetL:
    """Tests for log_det_L."""

    def test_diagonal_only(self):
        """With rank=0, log det(L) = Σ log(dᵢ)."""
        B, C, H, W = 2, 3, 4, 5
        torch.manual_seed(42)
        d = torch.rand(B, C, H, W) + 0.5
        V = torch.zeros(B, 0, C, H, W)

        result = log_det_L(d, V)
        expected = d.reshape(B, -1).log().sum(dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_matches_dense_logdet(self):
        """log det(L) should match torch.linalg.slogdet on the dense L matrix."""
        B, C, H, W, rank = 1, 1, 2, 3, 2
        N = C * H * W
        torch.manual_seed(42)
        d = torch.rand(B, C, H, W) + 0.5
        V = torch.randn(B, rank, C, H, W) * 0.3

        # Build dense L = diag(d) + V Vᵀ
        d_flat = d.reshape(B, N)
        V_flat = V.reshape(B, rank, N)
        L_dense = torch.diag_embed(d_flat) + torch.bmm(
            V_flat.permute(0, 2, 1), V_flat
        )
        expected = torch.linalg.slogdet(L_dense).logabsdet

        result = log_det_L(d, V)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        """Gradients should flow through log_det_L."""
        B, C, H, W, rank = 2, 2, 4, 4, 3
        torch.manual_seed(42)
        d = (torch.rand(B, C, H, W) + 0.5).requires_grad_(True)
        V = (torch.randn(B, rank, C, H, W) * 0.1).requires_grad_(True)

        result = log_det_L(d, V).sum()
        result.backward()

        assert d.grad is not None
        assert V.grad is not None


class TestNLL:
    """Tests for the full NLL function."""

    def test_diagonal_matches_normal_nll(self):
        """With rank=0, NLL should match diagonal Normal NLL."""
        B, C, H, W = 2, 3, 4, 5
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W)
        mean = torch.randn(B, C, H, W)
        d = torch.rand(B, C, H, W) + 0.5
        V = torch.zeros(B, 0, C, H, W)

        result = nll(x, mean, d, V)

        from torch.distributions import Normal
        expected = -Normal(mean, d).log_prob(x).mean()
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_matches_dense_mvn(self):
        """NLL should match MultivariateNormal on dense covariance LLᵀ."""
        B, C, H, W, rank = 1, 1, 2, 3, 2
        N = C * H * W
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W)
        mean = torch.randn(B, C, H, W)
        d = torch.rand(B, C, H, W) + 0.5
        V = torch.randn(B, rank, C, H, W) * 0.3

        result = nll(x, mean, d, V)

        # Build dense L and covariance Σ = LLᵀ
        d_flat = d.reshape(B, N)
        V_flat = V.reshape(B, rank, N)
        L_dense = torch.diag_embed(d_flat) + torch.bmm(
            V_flat.permute(0, 2, 1), V_flat
        )
        cov = L_dense @ L_dense.transpose(-1, -2)

        from torch.distributions import MultivariateNormal
        mvn = MultivariateNormal(mean.reshape(B, N), covariance_matrix=cov)
        expected = -mvn.log_prob(x.reshape(B, N)).mean() / N

        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        """Gradients should flow through nll to all parameters."""
        B, C, H, W, rank = 2, 2, 4, 4, 3
        torch.manual_seed(42)
        x = torch.randn(B, C, H, W)
        mean = torch.randn(B, C, H, W).requires_grad_(True)
        d = (torch.rand(B, C, H, W) + 0.5).requires_grad_(True)
        V = (torch.randn(B, rank, C, H, W) * 0.1).requires_grad_(True)

        loss = nll(x, mean, d, V)
        loss.backward()

        assert mean.grad is not None
        assert d.grad is not None
        assert V.grad is not None
