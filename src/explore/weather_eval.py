# %%
import pickle
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from cdnp.model.ctx import ModelCtx
from cdnp.model.flow_matching.flow_matching import FlowMatching
from cdnp.util.instantiate import Experiment

NUM_STATIC_DIMS = 5
NUM_DYN_DIMS = 2


def generate_forecasts(
    ctx: ModelCtx, model: FlowMatching, n_timesteps: int, **kwargs
) -> list[torch.Tensor]:
    """Generate forecasts for a given model."""
    ctx = deepcopy(ctx)
    forecasts = []
    assert ctx.image_ctx is not None

    for _ in range(n_timesteps):
        ctx.image_ctx = ctx.image_ctx.to(dtype=torch.float32)
        forecast = model.sample(ctx, num_samples=-1, **kwargs)
        im_ctx = ctx.image_ctx
        static_ctx = im_ctx[:, :NUM_STATIC_DIMS, ...]
        dyn_prev = im_ctx[:, NUM_STATIC_DIMS : NUM_STATIC_DIMS + NUM_DYN_DIMS, ...]

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


def generate_ensemble(
    ctx: ModelCtx, model: FlowMatching, n_timesteps: int, n_members: int, **kwargs
) -> torch.Tensor:
    """Returns (B, C, H, W, T, N)"""
    all_forecasts = []

    if n_members > 1:
        pbar = tqdm(range(n_members), desc="Generating ensemble")
    else:
        pbar = range(n_members)
    for _ in pbar:
        trajectory = generate_forecasts(ctx, model, n_timesteps, **kwargs)
        trajectory = torch.stack(trajectory, dim=-1)  # (B, C, H, W, T)
        all_forecasts.append(trajectory)

    all_forecasts = torch.stack(all_forecasts, dim=-1)  # (B, C, H, W, T, N)
    return all_forecasts


class CRPS:
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # Term A: E|X - y|
        a = (samples - trg.unsqueeze(-1)).abs().mean(dim=-1)

        # Term B: 0.5 * E|X - X'|
        samples_sorted = torch.sort(samples, dim=-1)[0]
        n = samples.shape[-1]

        # Create weights (2*i - N + 1) for 0-based index i
        i = torch.arange(n, device=samples.device, dtype=samples.dtype)
        weights = (2 * i - n + 1).view([1] * (samples.dim() - 1) + [-1])

        # Calculate the unbiased expectation of pairwise differences
        b_sum = (samples_sorted * weights).sum(dim=-1)
        # b = 0.5 * b_sum / (n * (n - 1) + 1e-6)  # Add epsilon for n=1 case
        b = b_sum / (n * (n - 1) + 1e-6)  # Add epsilon for n=1 case

        crps = a - b
        return crps.mean(dim=(0, 2, 3))

    def name(self) -> str:
        return "crps"


class EnsembleRmse:
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        pred = samples.mean(dim=-1)  # (B, C, H, W)
        rmse = torch.sqrt(torch.mean((pred - trg) ** 2, dim=(2, 3)))  # ty: ignore
        return rmse.mean(dim=0)  # (C,)

    def name(self) -> str:
        return "ensemble_rmse"


@torch.no_grad()
def compute_results(
    exp: Experiment,
    n_t0s: int,
    n_members: int,
    n_days: int | None = None,
    **kwargs,
):
    ds = exp.val_loader.dataset

    mean = ds.trg_mean.squeeze()[None, :, None, None, None]  # ty: ignore
    std = ds.trg_std.squeeze()[None, :, None, None, None]  # ty: ignore
    mean = mean.to("cuda")
    std = std.to("cuda")

    if n_days is not None:
        n_timesteps = 4 * n_days
        ds.target_timedeltas = ds.temporal_resolution * (np.arange(n_timesteps) + 1)  # ty: ignore

    metrics = [
        EnsembleRmse(),
        CRPS(),
    ]
    results = {}

    count = 0

    for batch in exp.val_loader:
        ctx, trg = exp.preprocess_fn(batch)  # ty: ignore
        ctx, trg = ctx.to(device), trg.to(device)
        ensemble = generate_ensemble(
            ctx, exp.model, n_timesteps=n_timesteps, n_members=n_members, **kwargs
        )  # (B, C, H, W, T, N)

        trg = trg * std + mean
        ensemble = ensemble * std.unsqueeze(-1) + mean.unsqueeze(-1)

        for metric in metrics:
            result = metric.compute(ensemble, trg)
            if metric.name() in results:
                results[metric.name()] += result
            results[metric.name()] = result

        count += trg.shape[0]
        print(f"Processed {count} samples out of {n_t0s}")

        # if count >= 1:
        if count >= n_t0s:
            break

    return results


@torch.no_grad()
def compute_ensemble(
    experiment_name: str,
    n_members: int,
    n_days: int | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    batch = default_collate([ds[0]])

    ctx, trg = exp.preprocess_fn(batch)  # ty: ignore
    ctx, trg = ctx.to(device), trg.to(device)
    ensemble = generate_ensemble(
        ctx, model, n_timesteps=n_timesteps, n_members=n_members, **kwargs
    )
    return ensemble, trg


def load_experiment(exp_name: str) -> Experiment:
    path = Path("/home/jonas/Documents/code/denoising-np/_output") / exp_name
    path = ExperimentPath.from_path(path)
    cfg = path.get_config()
    exp = Experiment.from_config(cfg)  # ty: ignore
    cm = CheckpointManager(path)
    _ = cm.reproduce_model(exp.model, "latest_ema")
    return exp


def get_power_spectrum_1d(
    data_tensor: torch.Tensor, num_bins: int = 60
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the radially averaged 1D power spectrum for a 2D global dataset.

    Args:
        data_tensor (torch.Tensor): A 2D tensor of shape [lat, lon] representing
                                    global data. Assumes lat dimension covers 180
                                    degrees and lon covers 360 degrees.
        num_bins (int): The number of logarithmic bins to use for the spectrum.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - A tensor containing the relative power for each wavelength bin.
        - A tensor containing the central wavelength (in km) for each bin.
    """
    # --- 1. Define Physical Constants and Grid Properties ---
    # Earth's approximate equatorial circumference in km
    EARTH_CIRCUMFERENCE_KM = 40075.0
    # Pole-to-pole distance (half circumference)
    POLE_TO_POLE_KM = EARTH_CIRCUMFERENCE_KM / 2.0

    # Get grid dimensions
    n_lat, n_lon = data_tensor.shape
    device = data_tensor.device

    # Calculate the physical distance per pixel/grid cell
    # dx: distance per longitude grid cell (at the equator)
    dx = EARTH_CIRCUMFERENCE_KM / n_lon
    # dy: distance per latitude grid cell
    dy = POLE_TO_POLE_KM / n_lat

    # --- 2. Perform 2D FFT and Calculate Power Spectrum ---
    # Apply a window function (e.g., Hann) to reduce spectral leakage, optional but good practice
    window_lat = torch.hann_window(n_lat, device=device)
    window_lon = torch.hann_window(n_lon, device=device)
    window_2d = window_lat.unsqueeze(1) * window_lon.unsqueeze(0)

    # Detrend the data (subtract the mean) before FFT
    data_detrended = data_tensor - data_tensor.mean()

    # Apply window and perform FFT
    fft_data = torch.fft.fft2(data_detrended * window_2d)

    # Calculate the 2D power spectrum (squared magnitude)
    # The result is shifted to have the zero-frequency component in the center
    power_spectrum_2d = torch.abs(torch.fft.fftshift(fft_data)) ** 2

    # --- 3. Create Wavenumber Grid in physical units (cycles/km) ---
    # Get the frequency bins for each axis
    freq_lat = torch.fft.fftshift(torch.fft.fftfreq(n_lat, d=dy, device=device))
    freq_lon = torch.fft.fftshift(torch.fft.fftfreq(n_lon, d=dx, device=device))

    # Create a 2D grid of wavenumbers
    kx, ky = torch.meshgrid(freq_lon, freq_lat, indexing="xy")

    # --- 4. Calculate Radial Wavenumber ---
    # This gives the distance from the center for each point in the 2D spectrum
    k_radial = torch.sqrt(kx**2 + ky**2)

    # --- 5. Perform Logarithmic Binning ---
    # Flatten the 2D arrays to 1D for easier processing
    k_flat = k_radial.flatten()
    power_flat = power_spectrum_2d.flatten()

    # Exclude the zero-frequency (DC) component, which corresponds to the mean
    # It has k=0 and infinite wavelength, which messes up log plots.
    dc_mask = k_flat > 0
    k_flat = k_flat[dc_mask]
    power_flat = power_flat[dc_mask]

    # Define the edges for our logarithmic bins
    k_min = k_flat.min()
    k_max = k_flat.max()
    # torch.logspace is perfect for this
    bin_edges = torch.logspace(
        torch.log10(k_min), torch.log10(k_max), steps=num_bins + 1, device=device
    )

    # Use torch.bucketize to find which bin each k value falls into
    bin_indices = torch.bucketize(k_flat, bin_edges) - 1  # -1 to make it 0-indexed

    bin_indices.clamp_(0, num_bins - 1)

    # Sum the power in each bin
    # We create a tensor to hold the binned power
    binned_power = torch.zeros(num_bins, device=device, dtype=power_flat.dtype)
    # .scatter_add_ is an efficient way to sum values based on an index tensor
    binned_power.scatter_add_(0, bin_indices, power_flat)

    # Also count how many items are in each bin to average later if desired,
    # but summing is more common for total power.
    bin_counts = torch.zeros(num_bins, device=device, dtype=torch.int32)
    bin_counts.scatter_add_(
        0, bin_indices, torch.ones_like(bin_indices, dtype=torch.int32)
    )

    # --- 6. Convert Bins to Wavelengths (km) ---
    # Calculate the geometric mean of the bin edges for the representative wavenumber
    k_bin_centers = torch.sqrt(bin_edges[:-1] * bin_edges[1:])
    # Convert wavenumber (cycles/km) to wavelength (km/cycle)
    wavelength_bins_km = 1.0 / k_bin_centers

    # --- 7. Finalize and Normalize ---
    # Filter out any empty bins that might have occurred
    non_empty_mask = binned_power > 0
    final_power = binned_power[non_empty_mask]
    final_wavelengths_km = wavelength_bins_km[non_empty_mask]

    # Normalize to get relative power (sums to 1)
    # relative_power = final_power / final_power.sum()

    # The result is typically plotted with wavelength on x-axis, so we reverse
    # the order to go from small to large wavelengths.
    return torch.flip(final_power, dims=[0]), torch.flip(final_wavelengths_km, dims=[0])


if __name__ == "__main__":
    device = "cuda"

    n_t0s = 30
    num_bins = 30
    exp_names = [
        # "2025-09-05_13-18_radiant_hippo",
        # "2025-09-01_16-29_jolly_whale",
    ]
    nfes = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30, 50, 100]

    samplers = [
        {
            "solver": "midpoint",
            "skip_type": "time_uniform",
        },
        {
            "solver": "dpm_solver_3",
            "skip_type": "time_uniform",
        },
    ]

    # %%

    for exp_name in exp_names:
        exp = load_experiment(exp_name)
        ds = exp.val_loader.dataset

        plt.figure()

        sampler_results = {}
        with torch.no_grad():
            for sampler in samplers:
                results = []
                for nfe in nfes:
                    ratios = []
                    for i in tqdm(range(n_t0s)):
                        batch = default_collate([ds[i]])

                        ctx, trg = exp.preprocess_fn(batch)  # ty: ignore
                        ctx, trg = ctx.to(device), trg.to(device)

                        ens = generate_ensemble(
                            ctx,
                            exp.model,  # ty: ignore
                            n_timesteps=1,
                            n_members=1,
                            nfe=nfe,
                            ode_method=sampler["solver"],
                            skip_type=sampler["skip_type"],
                        )
                        ens_slice = ens[0, 0, :, :, 0, 0]
                        trg_slice = trg[0, 0, :, :]
                        ps, wl = get_power_spectrum_1d(
                            ens_slice.cpu(), num_bins=num_bins
                        )
                        ps_trg, wl_trg = get_power_spectrum_1d(
                            trg_slice.cpu(), num_bins=num_bins
                        )

                        ratio = ps / ps_trg
                        ratios.append(ratio)
                    ratio = torch.stack(ratios, dim=0).mean(dim=0)
                    results.append(ratio)
                res = torch.stack(results, dim=0).cpu().numpy()  # (num_nfes, num_bins)
                sampler_results[sampler["solver"]] = res

        best = None
        for res in sampler_results.values():
            if best is None:
                best = res.copy()
            else:
                for i in range(res.shape[0]):
                    total_abs_err = np.abs(res[i, :] - 1.0).sum()
                    current_abs_err = np.abs(best[i, :] - 1.0).sum()
                    if total_abs_err < current_abs_err:
                        best[i, :] = res[i, :]
        sampler_results["best"] = best
        sampler_results["wl"] = wl.cpu().numpy()

        with open(f"frequency_results_{exp_name}.pkl", "wb") as f:
            pickle.dump(sampler_results, f)

    # %%
    # exp_name = "2025-07-07_21-13_noble_iguana"
    # exp_name = "2025-09-01_10-06_lucky_dog"
    exp_name = "2025-09-01_16-29_jolly_whale"
    # exp_name = "2025-09-05_13-18_radiant_hippo"

    n_days = 5
    n_t0s = 40
    n_members = 50
    nfes = [2, 6, 10, 16, 20, 30]
    solver = "dpm_solver_3"
    skip_type = "logSNR"

    for nfe in nfes:
        exp = load_experiment(exp_name)
        results = compute_results(
            exp=exp,
            n_t0s=n_t0s,
            n_members=n_members,
            nfe=nfe,
            n_days=n_days,
            ode_method=solver,
            skip_type=skip_type,
        )

        ds = exp.val_loader.dataset
        trg_vars = ds.trg_variables_and_levels  # ty: ignore

        csv_path = f"wind_results_{exp_name}.csv"
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(
                columns=[
                    "exp_name",
                    "nfe",
                    "solver",
                    "skip_type",
                    "n_members",
                    "n_0_times",
                    "variable",
                    "metric",
                    "time_delta_hrs",
                    "value",
                ]
            )

        rows = []
        for var_idx, var in enumerate(trg_vars):
            for metric, result in results.items():
                for time_idx in range(result.shape[1]):
                    time_delta = 6 * (time_idx + 1)
                    row = {
                        "exp_name": exp_name,
                        "nfe": nfe,
                        "solver": solver,
                        "skip_type": skip_type,
                        "n_members": n_members,
                        "n_0_times": n_t0s,
                        "variable": var,
                        "metric": metric,
                        "time_delta_hrs": time_delta,
                        "value": float(result[var_idx, time_idx]),
                    }
                    rows.append(row)

        print(rows)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        print("Saving to ", csv_path)
        df.to_csv(csv_path, index=False)

    # %%

    # %%

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    yticklabels = [
        f"{w:.0e}" if idx % 5 == 0 else "" for idx, w in enumerate(wl.numpy())
    ]

    plottable = sampler_results["midpoint"]
    sns.heatmap(
        plottable.T,
        xticklabels=nfes,
        yticklabels=yticklabels,
        cmap="vlag",
        center=1.0,
        ax=axs[0],
    )

    plottable = sampler_results["dpm_solver_3"]
    sns.heatmap(
        plottable.T,
        xticklabels=nfes,
        yticklabels=yticklabels,
        cmap="vlag",
        center=1.0,
        ax=axs[1],
    )

    axs[0].set_xlabel("NFE")
    axs[1].set_xlabel("NFE")
    axs[0].set_ylabel("Wavelength (km)")
    axs[0].set_title("Midpoint")
    axs[1].set_title("DPM-Solver")
    axs[1].set_yticks([], labels=[])

    plottable = sampler_results["best"]
    abs_diff = np.abs(plottable - 1.0).sum(axis=1)
    axs[2].plot(nfes, abs_diff, marker="x")
    axs[2].set_xscale("log")
    axs[2].set_title("Power Spectrum Deviation (Best)")
    plt.xlabel("NFE")
    plot_nfes = [2, 4, 6, 8, 10, 14, 20, 30, 50]
    plt.xticks(plot_nfes, labels=plot_nfes, rotation=90)
    plt.show()

    # %%

    time_skip = 4  # plot every 24h
    trg_and_ensemble = torch.cat([trg.unsqueeze(-1), ensemble], dim=-1)
    plot_ensemble = trg_and_ensemble[:, :, :, :, ::time_skip, : n_plot_trajectories + 1]

    col_titles = [
        f"T + {(6 * (i + 1) * time_skip) // 24} days"
        for i in range(plot_ensemble.shape[4])
    ]

    row_titles = [f"Member {i + 1}" for i in range(n_plot_trajectories)]
    row_titles = ["Ground Truth"] + row_titles

    _ = gp.plot_grid(
        plot_ensemble[0, 1, :, :, :, :].cpu(),
        col_titles=col_titles,
        row_titles=row_titles,
    )

    # %%

    variable_idx = 0
    # variable_name = ds.trg_variables_and_levels[variable_idx]  # ty: ignore
    # n_timesteps = trg.shape[4]
    n_timesteps = n_days * 4

    times_hrs = np.arange(n_timesteps) * 6 + 6
    times_days = times_hrs / 24
    for name, result in results.items():
        plt.figure()
        # plt.title(f"{name} - {variable_name}")
        # x with dots and lines:
        plt.plot(times_hrs, result[variable_idx, :].cpu(), marker="x")
        ticks = times_hrs[3::4]
        plt.xticks(ticks, labels=ticks // 24)
        plt.xlabel("Lead times (days)")
        plt.ylabel(name)
        plt.show()

    # %%
