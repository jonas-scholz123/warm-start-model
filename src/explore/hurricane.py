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

exp_name = "2025-07-07_21-13_noble_iguana"

path = Path("/home/jonas/Documents/code/denoising-np/_weights") / exp_name
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
N_timesteps = 200
num_static_dims = 37
num_dyn_dims = 2

mean_model = True

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
