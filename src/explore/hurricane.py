# %%
from pathlib import Path

import matplotlib.pyplot as plt
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
from cdnp.model.cdnp import CDNP
from cdnp.plot.geoplot import GeoPlotter
from cdnp.util.instantiate import Experiment

# %%

ds = make_gridded_weather_task(
    data_source=ZarrDatasource.from_path(
        path="/home/jonas/Documents/code/otter/_data/era5_240x121.zarr"
    ),
    norm_path="/home/jonas/Documents/code/otter/otter/data/normalisation/stats",
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
exp_name = "2025-07-08_16-32_jolly_narwhal"

# path = Path("/home/jonas/Documents/code/denoising-np/_weights") / exp_name
path = Path("/home/jonas/Documents/code/denoising-np/_output") / exp_name
path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)

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
device = "cuda"
batch = default_collate([ds[0]])
ctx, trg = exp.preprocess_fn(batch)
ctx, trg = ctx.to(device), trg.to(device)
# %%

cm = CheckpointManager(path)
model: CDNP = exp.model
_ = cm.reproduce_model(model, "best")
# %%
model.set_steps(100000)
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


intervals = 24
N_timesteps = 8 * intervals

ground_truth = [ds[i][3] for i in range(N_timesteps)]
mean_forecasts = generate_forecasts(mean_model=True, n_timesteps=N_timesteps)
sampled_forecasts = generate_forecasts(mean_model=False, n_timesteps=N_timesteps)


# --- Plotting ---
timesteps_hours = [i * intervals for i in range(1, N_timesteps // intervals + 1)]
timesteps_indices = [t // 6 - 1 for t in timesteps_hours]

# --- Determine global min/max for color scaling ---
all_forecasts_to_plot = []
for time_idx in timesteps_indices:
    all_forecasts_to_plot.append(ground_truth[time_idx][0, 0, ...].cpu())
    all_forecasts_to_plot.append(mean_forecasts[time_idx][0, 0, ...].cpu())
    all_forecasts_to_plot.append(sampled_forecasts[time_idx][0, 0, ...].cpu())

all_data = torch.stack(all_forecasts_to_plot)
vmin = all_data.min().item()
vmax = all_data.max().item()


fig, axes = plt.subplots(
    2,
    8,
    figsize=(40, 10),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"hspace": 0.3, "wspace": 0.1},
)

fig.suptitle("Forecast Rollout", fontsize=20)

# Plot mean forecasts
for i, (ax, time_idx) in enumerate(zip(axes[0], timesteps_indices)):
    forecast = mean_forecasts[time_idx]
    data = forecast[0, 0, ...].cpu()
    gp.plot_single(data, ax=ax, show_cbar=False, vmin=vmin, vmax=vmax)
    ax.set_title(f"Mean model\nT + {timesteps_hours[i]}h")

# Plot sampled forecasts
for i, (ax, time_idx) in enumerate(zip(axes[1], timesteps_indices)):
    forecast = sampled_forecasts[time_idx]
    data = forecast[0, 0, ...].cpu()
    gp.plot_single(data, ax=ax, show_cbar=False, vmin=vmin, vmax=vmax)
    ax.set_title(f"Sampled model\nT + {timesteps_hours[i]}h")

# Add a single colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes(rect=[0.92, 0.15, 0.02, 0.7])  # type: ignore
mappable = plt.cm.ScalarMappable(cmap=gp.cmap)
mappable.set_array([])
mappable.set_clim(vmin, vmax)
fig.colorbar(mappable, cax=cbar_ax)


plt.show()
