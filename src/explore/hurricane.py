# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cartopy import crs as ccrs
from matplotlib.animation import FuncAnimation
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from mlbnb.types import Split
from torch import Generator
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from cdnp.data.era5 import ZarrDatasource, make_gridded_weather_task
from cdnp.plot.geoplot import GeoPlotter
from cdnp.util.instantiate import Experiment
from explore.weather_eval import generate_ensemble

# %%

ds = make_gridded_weather_task(
    data_source=ZarrDatasource.from_path(
        path="[redacted]"
    ),
    norm_path="[redacted]",
    start_date="2010-01-01",
    end_date="2011-01-01",
    val_start_date="2015-10-19",
    val_end_date="2015-12-31",
    num_context_frames=2,
    num_target_frames=1,
    temporal_resolution_hours=6,
    ctx_variables_to_exclude=[
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
    ],
    trg_variables_to_exclude=[
        "geopotential",
        "specific_humidity",
        "temperature",
        "2m_temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
    ],
    split=Split.VAL,
    generator=Generator().manual_seed(42),
)
# %%

# exp_name = "2025-07-07_21-13_noble_iguana"
# exp_name = "2025-07-08_16-32_jolly_narwhal"
exp_name = "2025-09-05_13-18_radiant_hippo"

path = Path("[redacted]") / exp_name
path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)
device = "cuda"
batch = default_collate([ds[0]])
ctx, trg = exp.preprocess_fn(batch)
ctx, trg = ctx.to(device), trg.to(device)

# %%

# %%

plot = [None]  # container for the plot handle


def init():
    """Initialize the plot."""
    _, _, _, trg = ds[0]
    data = trg[..., 0, 0]
    plot[0] = gp.plot_single(data, ax=ax, show_cbar=False)
    return plot


def update(frame):
    """Update the plot for each frame."""
    ax.clear()
    _, _, _, trg = ds[frame]
    data = trg[..., 0, 0]
    gp.plot_single(data, ax=ax, show_cbar=False)
    ax.set_title(f"T + {frame * 6} hours")
    return (ax,)


gp = GeoPlotter()
fig, ax = plt.subplots(figsize=(15, 7), subplot_kw={"projection": ccrs.PlateCarree()})

ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=False)
ani.save("./era5.gif", fps=3)
# %%
# %%

cm = CheckpointManager(path)
_ = cm.reproduce_model(exp.model, "best")
# %%
N_timesteps = 50
num_static_dims = 37
num_dyn_dims = 2

mean_model = False

ctx, trg = exp.preprocess_fn(batch)
ctx, trg = ctx.to(device), trg.to(device)

forecasts = []

for i in tqdm(range(N_timesteps)):
    if mean_model:
        forecast = model.cnp.predict(ctx).mean
    else:
        forecast = model.sample(ctx)
    im_ctx = ctx.image_ctx
    static_ctx = im_ctx[:, :num_static_dims, ...]
    dyn_prev = im_ctx[:, num_static_dims : num_static_dims + num_dyn_dims, ...]
    dyn_prev2 = im_ctx[
        :, num_static_dims + num_dyn_dims : num_static_dims + 2 * num_dyn_dims, ...
    ]

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


plot = [None]  # container for the plot handle


def init():
    """Initialize the plot."""
    forecast = forecasts[0]
    data = forecast[0, 0, ...]
    plot[0] = gp.plot_single(data, ax=ax, show_cbar=False)
    return plot


def update(frame):
    """Update the plot for each frame."""
    ax.clear()
    forecast = forecasts[frame]
    data = forecast[0, 0, ...]
    gp.plot_single(data, ax=ax, show_cbar=False)
    ax.set_title(f"T + {(frame + 1) * 6} hours")
    return (ax,)


gp = GeoPlotter()
fig, ax = plt.subplots(figsize=(15, 7), subplot_kw={"projection": ccrs.PlateCarree()})

ani = FuncAnimation(fig, update, frames=len(forecasts), init_func=init, blit=False)

name = "mean_model" if mean_model else "sampled_model"
ani.save(f"./forecast_rollout_{name}.gif", fps=3)
# %%

dyn2 = ctx.image_ctx[:, num_static_dims : num_static_dims + num_dyn_dims, ...]
dyn1 = ctx.image_ctx[
    :, num_static_dims + num_dyn_dims : num_static_dims + 2 * num_dyn_dims, ...
]
gp.plot_single(
    dyn2[0, 0, ...],
    ax=ax,
    show_cbar=False,
)
# %%
gp.plot_single(
    dyn1[0, 0, ...],
    ax=ax,
    show_cbar=False,
)

# %%


# %%

num_static_dims = 37
num_dyn_dims = 2


def generate_forecasts(mean_model: bool, n_timesteps: int) -> list[torch.Tensor]:
    """Generate forecasts for a given model."""
    ctx, _ = exp.preprocess_fn(batch)
    ctx = ctx.to(device)

    forecasts = []
    pbar_desc = f"Generating {'mean' if mean_model else 'sampled'} forecasts"
    pbar = tqdm(range(n_timesteps), desc=pbar_desc)

    for _ in pbar:
        if mean_model:
            forecast = model.cnp.predict(ctx).mean
        else:
            forecast = model.sample(ctx)

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


# %%

intervals = 6
num_timesteps_to_plot = 8
N_timesteps = num_timesteps_to_plot * intervals
num_samples = 4

ground_truth = [ds[i][3] for i in range(N_timesteps)]
sampled_forecasts_list = [
    generate_forecasts(mean_model=False, n_timesteps=N_timesteps)
    for _ in range(num_samples)
]


# --- Plotting ---
timesteps_hours = [i * intervals for i in range(1, N_timesteps // intervals + 1)]
timesteps_indices = [t // 6 - 1 for t in timesteps_hours]

# --- Determine global min/max for color scaling ---
all_forecasts_to_plot = []
for time_idx in timesteps_indices:
    all_forecasts_to_plot.append(ground_truth[time_idx][0, 0, ...].cpu())
    for sampled_forecasts in sampled_forecasts_list:
        all_forecasts_to_plot.append(sampled_forecasts[time_idx][0, 0, ...].cpu())

# %%
all_data = torch.stack(all_forecasts_to_plot)
vmin = all_data.min().item()
vmax = all_data.max().item()


fig, axes = plt.subplots(
    num_samples + 1,
    num_timesteps_to_plot,
    figsize=(40, 5 * (num_samples + 1)),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"hspace": 0.3, "wspace": 0.1},
)

fig.suptitle("Forecast Rollout Comparison", fontsize=20)

# Plot ground truth
for i, (ax, time_idx) in enumerate(zip(axes[0], timesteps_indices)):
    data = ground_truth[time_idx][0, 0, ...].cpu()
    gp.plot_single(data, ax=ax, show_cbar=False, vmin=vmin, vmax=vmax)
    ax.set_title(f"Ground Truth\nT + {timesteps_hours[i]}h")

# Plot sampled forecasts
for sample_idx in range(num_samples):
    for i, (ax, time_idx) in enumerate(zip(axes[sample_idx + 1], timesteps_indices)):
        forecast = sampled_forecasts_list[sample_idx][time_idx]
        data = forecast[0, 0, ...].cpu()
        gp.plot_single(data, ax=ax, show_cbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"Sample {sample_idx + 1}\nT + {timesteps_hours[i]}h")


# Add a single colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes(rect=[0.92, 0.15, 0.02, 0.7])  # type: ignore
mappable = plt.cm.ScalarMappable(cmap=gp.cmap)
mappable.set_array([])
mappable.set_clim(vmin, vmax)
fig.colorbar(mappable, cax=cbar_ax)


plt.show()

# %%

N_timesteps = 6
N_samples = 4

all_forecasts = []
for _ in range(N_samples):
    forecasts = generate_forecasts(mean_model=False, n_timesteps=N_timesteps)
    forecasts = [f[0, 0, ...].cpu() for f in forecasts]
    all_forecasts.append(torch.stack(forecasts, dim=-1))
all_forecasts = torch.stack(all_forecasts, dim=-1)
ground_truth = torch.stack([ds[i][3][..., 0, 0] for i in range(N_timesteps)], dim=-1)
ground_truth = ground_truth.unsqueeze(-1)
plottable = torch.cat([ground_truth, all_forecasts], dim=-1)
# %%
n_members = 3
n_days = 5
n_timesteps = 4 * n_days
ode_method = "midpoint"
nfe = 10
ensemble = generate_ensemble(
    ctx,
    exp.model,
    n_timesteps=n_timesteps,
    n_members=n_members,
    ode_method=ode_method,
    nfe=nfe,
)
# %%

var_idx = 0

width = 3.5

gp = GeoPlotter(map_width=width, map_height=width / 2)

ground_truth = torch.Tensor(np.array([ds[i][3] for i in range(n_timesteps)])).to(device)
ground_truth = ground_truth[:, :, :, :, var_idx]
ground_truth = ground_truth.permute(1, 2, 0, 3)
ens = ensemble[0, var_idx]

plottable = torch.cat([ground_truth, torch.Tensor(ens)], dim=-1)
plottable = plottable[:, :, 3::4, :]

col_titles = ["Lead Time = 1 day"] + [f"{i} days" for i in range(2, n_days + 1)]

fig = gp.plot_grid(
    data=plottable,
    row_titles=["Ground Truth"] + [f"Sample {i + 1}" for i in range(n_members)],
    col_titles=col_titles,
    share_cmap="global",
    show_cbar=False,
)
fig.subplots_adjust(wspace=0.00, hspace=0.05)

fig.savefig(
    "forecast_samples.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
