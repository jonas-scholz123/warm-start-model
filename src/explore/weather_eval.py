# %%
from pathlib import Path

import numpy as np
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from cdnp.model.flow_matching.flow_matching import FlowMatching
from cdnp.plot.geoplot import GeoPlotter
from cdnp.util.instantiate import Experiment


def generate_forecasts(n_timesteps: int) -> list[torch.Tensor]:
    """Generate forecasts for a given model."""
    ctx, _ = exp.preprocess_fn(batch)  # type: ignore
    ctx = ctx.to(device)

    forecasts = []
    pbar_desc = "Generating forecasts"
    pbar = tqdm(range(n_timesteps), desc=pbar_desc)

    for _ in pbar:
        ctx.image_ctx = ctx.image_ctx.to(dtype=torch.float32)
        forecast = model.sample(ctx, num_samples=-1, nfe=nfe, solver=solver)
        im_ctx = ctx.image_ctx
        static_ctx = im_ctx[:, :num_static_dims, ...]
        dyn_prev = im_ctx[:, num_static_dims : num_static_dims + num_dyn_dims, ...]

        new_ctx = torch.cat(
            [
                static_ctx,
                forecast[:, :, ...],
                dyn_prev,
            ],
            dim=1,
        )
        ctx.image_ctx = new_ctx
        forecasts.append(forecast)
    return forecasts


def generate_ensemble(n_timesteps: int, n_members: int) -> torch.Tensor:
    """Returns (B, C, H, W, T, N)"""
    ctx, _ = exp.preprocess_fn(batch)  # type: ignore
    ctx = ctx.to(device)

    all_forecasts = []
    pbar_desc = "Generating ensembles"
    pbar = tqdm(range(n_members), desc=pbar_desc)

    for _ in pbar:
        trajectory = generate_forecasts(n_timesteps)
        trajectory = torch.stack(trajectory, dim=-1)  # (B, C, H, W, T)
        all_forecasts.append(trajectory)

    all_forecasts = torch.stack(all_forecasts, dim=-1)  # (B, C, H, W, T, N)
    return all_forecasts


class CRPS:
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        trg = trg.expand_as(samples[..., 0])

        # Term A: E|X - trg|  (Monte-Carlo mean over samples)
        a = (samples - trg.unsqueeze(-1)).abs().mean(dim=-1)

        # Pairwise absolute differences
        # Expand: (..., N, 1) and (..., 1, N)
        diffs = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs()
        b = 0.5 * diffs.mean(dim=(-1, -2))

        crps = a - b
        return crps.mean(dim=(2, 3))

    def name(self) -> str:
        return "crps"


class EnsembleRmse:
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        pred = samples.mean(dim=-1)  # (B, C, H, W)
        rmse = torch.sqrt(torch.mean((pred - trg) ** 2, dim=(2, 3)))  # ty: ignore
        return rmse

    def name(self) -> str:
        return "ensemble_rmse"


@torch.no_grad()
def compute_results(
    experiment_name: str,
    n_t0s: int,
    n_members: int,
    nfe: int,
    n_days: int | None = None,
    solver: str = "dpm_solver_3",
    skip_type: str = "logSNR",
):
    path = Path("/home/jonas/Documents/code/denoising-np/_output") / exp_name
    path = ExperimentPath.from_path(path)
    cfg = path.get_config()
    exp = Experiment.from_config(cfg)  # ty: ignore

    ds = exp.val_loader.dataset

    if n_days is not None:
        n_timesteps = 4 * n_days
        ds.target_timedeltas = ds.temporal_resolution * (np.arange(n_timesteps) + 1)  # ty: ignore

    cm = CheckpointManager(path)
    model: FlowMatching = exp.model  # ty: ignore
    _ = cm.reproduce_model(model, "latest_ema")

    metrics = [
        EnsembleRmse(),
        CRPS(),
    ]
    results = {}

    count = 0

    for batch in exp.val_loader:
        ctx, trg = exp.preprocess_fn(batch)  # ty: ignore
        ctx, trg = ctx.to(device), trg.to(device)
        ensemble = generate_ensemble(n_timesteps=n_timesteps, n_members=4)

        for metric in metrics:
            result = metric.compute(ensemble, trg)
            if metric.name() in results:
                results[metric.name()] += result
            results[metric.name()] = result

        count += trg.shape[0]
        print(f"Processed {count} samples / {n_t0s}")
        if count >= 1:
            # if count >= num_t0s:
            break

    results = {k: v / count for k, v in results.items()}
    return results


# %%
# exp_name = "2025-07-07_21-13_noble_iguana"
# exp_name = "2025-09-01_10-06_lucky_dog"
exp_name = "2025-09-01_16-29_jolly_whale"
num_static_dims = 5
device = "cuda"
num_dyn_dims = 2

n_days = 5
n_t0s = 8
n_members = 4
nfe = 20

solver = "dpm_solver_3"

results = compute_results(
    experiment_name=exp_name,
    n_t0s=n_t0s,
    n_members=n_members,
    nfe=nfe,
    n_days=n_days,
    solver=solver,
)


# %%

path = Path("/home/jonas/Documents/code/denoising-np/_output") / exp_name
path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)  # ty: ignore

ds = exp.val_loader.dataset

n_timesteps = 4 * n_days
ds.target_timedeltas = ds.temporal_resolution * (np.arange(n_timesteps) + 1)  # ty: ignore

gp = GeoPlotter()

batch = default_collate([ds[0]])
ctx, trg = exp.preprocess_fn(batch)  # ty: ignore
ctx, trg = ctx.to(device), trg.to(device)
# %%

cm = CheckpointManager(path)
model: FlowMatching = exp.model  # ty: ignore
_ = cm.reproduce_model(model, "latest_ema")
# %%

# %%
ensemble = generate_ensemble(
    n_timesteps=n_timesteps, n_members=20
)  # (B, C, H, W, T, N)
# %%
metrics = [
    EnsembleRmse(),
    CRPS(),
]
results = {}

for metric in metrics:
    result = metric.compute(ensemble, trg)
    if metric.name() in results:
        results[metric.name()] += result
    results[metric.name()] = result
# %%

n_plot_trajectories = 4
time_skip = 4  # plot every 24h
plot_ensemble = ensemble[:, :, :, :, ::time_skip, :n_plot_trajectories]
_ = gp.plot_grid(
    plot_ensemble[0, 0, :, :, :, :].cpu(),
    col_titles=[
        f"T + {(6 * (i + 1) * time_skip) // 24} days"
        for i in range(plot_ensemble.shape[4])
    ],
    row_titles=[f"Member {i + 1}" for i in range(n_plot_trajectories)],
)


# %%

metrics = [
    EnsembleRmse(),
    CRPS(),
]
results = {}

count = 0

for batch in exp.val_loader:
    ctx, trg = exp.preprocess_fn(batch)  # ty: ignore
    ctx, trg = ctx.to(device), trg.to(device)
    ensemble = generate_ensemble(n_timesteps=n_timesteps, n_members=4)

    for metric in metrics:
        result = metric.compute(ensemble, trg)
        if metric.name() in results:
            results[metric.name()] += result
        results[metric.name()] = result

    count += trg.shape[0]
    print(f"Processed {count} samples / {n_t0s}")
    if count >= 1:
        # if count >= num_t0s:
        break
# %%
import matplotlib.pyplot as plt

variable_idx = 0
variable_name = ds.trg_variables_and_levels[variable_idx]  # ty: ignore

times_hrs = np.arange(n_timesteps) * 6 + 6
times_days = times_hrs / 24
for name, result in results.items():
    result = result * count
    plt.figure()
    plt.title(f"{name} - {variable_name}")
    plt.plot(times_hrs, result[0, variable_idx, :].cpu())
    ticks = times_hrs[3::4]
    plt.xticks(ticks, labels=ticks // 24)
    plt.xlabel("Lead times (days)")
    plt.ylabel(name)
    plt.show()
# %%
ds.target_timedeltas


# %%
def _power_spectrum_density_mse(pred: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D PSD for a batch of 2D tensors.
    """
    assert pred.ndim == 3, "Input must be a batch of 2D images: [B, H, W]"
    assert trg.ndim == 3, "Input must be a batch of 2D images: [B, H, W]"
    B, H, W = pred.shape
    device = pred.device

    # 2D FFT and power spectrum
    fft2_pred = torch.fft.fft2(pred)
    fft2_shifted_pred = torch.fft.fftshift(fft2_pred, dim=(-2, -1))
    psd_pred = torch.abs(fft2_shifted_pred) ** 2 / (H * W)

    # 2D FFT and power spectrum
    fft2_trg = torch.fft.fft2(trg)
    fft2_shifted_trg = torch.fft.fftshift(fft2_trg, dim=(-2, -1))
    psd_trg = torch.abs(fft2_shifted_trg) ** 2 / (H * W)

    psd_mse = (psd_pred - psd_trg) ** 2

    # Create frequency radius map
    y = torch.arange(H, device=device).reshape(-1, 1)
    x = torch.arange(W, device=device).reshape(1, -1)
    cy, cx = H // 2, W // 2
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r.round().long()
    r_max = int(r.max().item())

    # Flatten the radius map
    r_flat = r.flatten().unsqueeze(0).repeat(B, 1)
    psd_mse_flat = psd_mse.reshape(B, -1)

    psd_1d = torch.zeros(B, r_max + 1, dtype=torch.float32, device=device)
    counts = torch.zeros(B, r_max + 1, dtype=torch.float32, device=device)

    counts.scatter_add_(
        dim=1,
        index=r_flat,
        src=torch.ones_like(psd_mse_flat, dtype=torch.float32, device=device),
    )
    counts = counts.clamp(min=1)

    psd_1d.scatter_add_(dim=1, index=r_flat, src=psd_mse_flat)

    psd_1d /= counts

    return psd_1d


# %%
# Ensemble: (B, C, H, W, T, N)
ens_r = ensemble[0, 0, :, :, :, 0].permute(2, 0, 1).to(torch.float32)
trg_r = trg[0, 0, :, :, :].permute(2, 0, 1).to(torch.float32)
ps = _power_spectrum_density_mse(ens_r, trg_r).mean(dim=0)
plt.plot(np.arange(len(ps)), ps.cpu())
plt.xscale("log")
plt.yscale("log")
# %%
mean = ensemble.mean(dim=-1)
mean = mean[0, 0, :, :, :].permute(2, 0, 1).to(torch.float32)

ps2 = _power_spectrum_density_mse(mean, trg_r).mean(dim=0)
plt.plot(np.arange(len(ps2)), ps2.cpu())
plt.xscale("log")
plt.yscale("log")
# %%
