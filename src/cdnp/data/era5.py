# %%
from __future__ import annotations

from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import torch
import xarray as xr
from mlbnb.types import Split
from torch.utils.data import Dataset

from cdnp.model.swin.embeddings import _get_hours_from_reference_time

ERA5_START_DATE = "1979-01-01"
ERA5_END_DATE = "2023-12-31"
BASE_TEMPORAL_RESOLUTION = np.timedelta64(6, "h")


ALL_GRID_SURFACE_DYNAMIC_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
]

ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

ALL_GRID_STATIC_VARIABLES = [
    "angle_of_sub_gridscale_orography",
    "anisotropy_of_sub_gridscale_orography",
    "land_sea_mask",
    "slope_of_sub_gridscale_orography",
    "geopotential_at_surface",
]

ALL_GRID_VARIABLE_NAMES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
    + ALL_GRID_STATIC_VARIABLES
)

ALL_GRID_VARIABLE_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

VARIABLES_WITH_SCALING_FACTORS = {
    "geopotential": 1e-3,
    "geopotential_at_surface": 1e-3,
    "mean_sea_level_pressure": 1e-3,
}

ALL_CTX_VARIABLES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
    + ALL_GRID_STATIC_VARIABLES
)

ALL_TRG_VARIABLES = (
    ALL_GRID_SURFACE_DYNAMIC_VARIABLES + ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
)


def split_into_different_variables_along_dim(
    dataset: xr.Dataset,
    dim: str,
) -> xr.Dataset:
    """Splits variables in a dataset to different sub-variables
    along the specified dimension.

    For example, if the dataset contains a variable `temperature`
    with the dimensions `time`, `lat`, `lon`, `level` where
    dataset.level == (1000, 850, 700), then calling this function
    with dim="level" will return a new dataset where the
    `temperature` variable has been split into the sub-variables

        `temperature/level_1000`,
        `temperature/level_850`,
        `temperature/level_700`.

    The function join_into_single_variable_along_dim can be used
    to reverse this operation.

    Args:
        dataset: xarray dataset to split
        dim: dimension to split the variables along

    Returns:
        xarray dataset with variables split along the dimension
    """

    dataset = dataset.copy()

    assert dim in dataset.dims, f"Dimension {dim} not found in zarr"

    dim_values = dataset[dim].values

    for variable in dataset.variables:
        if (dim in dataset[variable].dims) and variable not in dataset.coords:
            for dim_value in dim_values:
                dataset[f"{variable}/{dim}_{dim_value}"] = dataset[variable].sel(
                    {dim: dim_value}
                )
            del dataset[variable]

    del dataset[dim]

    return dataset


def _get_normalisation_mean_and_std(
    norm_path: str,
    variables_and_levels: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_ds = _load_statistic(norm_path, "mean")
    mean_ds = split_into_different_variables_along_dim(mean_ds, dim="level")
    mean = stack_dataset_variable_and_levels(mean_ds)
    mean = mean.sel(variable_and_level=variables_and_levels)

    std_ds = _load_statistic(norm_path, "std")
    std_ds = split_into_different_variables_along_dim(std_ds, dim="level")
    std = stack_dataset_variable_and_levels(std_ds)
    std = std.sel(variable_and_level=variables_and_levels)

    mean = torch.from_numpy(mean.values).float()
    std = torch.from_numpy(std.values).float()

    return mean, std


def _load_statistic(
    base_path: str,
    statistic: str,
    zarr_name: str = "era5_240x121",
) -> xr.Dataset:
    return xr.open_dataset(f"{base_path}/{zarr_name}_{statistic}.nc")


def stack_dataset_variable_and_levels(
    dataset: xr.Dataset,
    name: str = "stacked",
    vars_to_drop: Optional[List[str]] = None,
) -> xr.DataArray:
    if vars_to_drop:
        dataset = dataset.drop_vars(vars_to_drop)

    # Stack all variables and levels into a single dimension. By default,
    # xarray stacks the variables across the first dimension, so we then
    # reshape the dimensions to have the variables and levels as the last
    # dimension.
    data_array = dataset.to_dataarray(dim="variable_and_level", name=name)
    return data_array.transpose(*data_array.dims[1:], data_array.dims[0])


def filter_variables(variables: Sequence[str], exclude: Sequence[str]) -> List[str]:
    return [var for var in variables if var not in exclude]


class ZarrDatasource:
    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds

    @staticmethod
    def from_path(path: str) -> ZarrDatasource:
        ds = xr.open_zarr(path, consolidated=True, chunks=None)
        ds = split_into_different_variables_along_dim(ds, dim="level")
        return ZarrDatasource(ds)

    def sel(self, **index_coords: Any) -> ZarrDatasource:
        # The variable_and_level dimension is not a coordinate, so we need to
        # instead select the correct variables and remove the index_coord.
        ds = self.ds
        if "variable_and_level" in index_coords:
            ds = ds[index_coords["variable_and_level"]]
            del index_coords["variable_and_level"]

        ds: xr.Dataset = ds.sel(**index_coords)  # type: ignore
        return ZarrDatasource(ds)

    def to_numpy(self) -> npt.NDArray[Any]:
        scaled_ds = self.ds.copy()  # Create a temporary copy of the dataset
        for var_and_level in scaled_ds.variables:
            var: str = var_and_level.split("/")[0]  # type: ignore
            factor = VARIABLES_WITH_SCALING_FACTORS.get(var)
            if factor:
                scaled_ds[var_and_level] = scaled_ds[var_and_level] * factor

        stacked = stack_dataset_variable_and_levels(scaled_ds)
        return stacked.values


def _combine_variables_and_levels(
    variables: Sequence[str],
    levels: Sequence[int],
) -> List[str]:
    return [f"{var}/level_{level}" for var in variables for level in levels]


class GriddedWeatherTask(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        data_source: ZarrDatasource,
        start_date: str,
        end_date: str,
        val_start_date: str,
        val_end_date: str,
        num_context_frames: int,
        num_target_frames: int,
        temporal_resolution_hours: int,
        norm_path: str,
        split: Split,
        ctx_surface_dynamic_variables: List[str] = ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        ctx_multilevel_dynamic_variables: List[
            str
        ] = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        ctx_static_variables: List[str] = ALL_GRID_STATIC_VARIABLES,
        trg_surface_dynamic_variables: List[str] = ALL_GRID_SURFACE_DYNAMIC_VARIABLES,
        trg_multilevel_dynamic_variables: List[
            str
        ] = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
        levels: List[int] = ALL_GRID_VARIABLE_LEVELS,
        task_sub_sampling_factor: int = 1,
    ):
        super().__init__()
        assert num_context_frames > 0

        # Start and end dates that will be used to form training tasks,
        # i.e. no tasks will contain data outside of this range.

        if split == Split.VAL or split == Split.TEST:
            self.start_date = np.datetime64(val_start_date)
            self.end_date = np.datetime64(val_end_date)
            # TODO test split separate
        else:
            # Default to training split
            self.start_date = np.datetime64(start_date)
            self.end_date = np.datetime64(end_date)

        # Temporal resolution of the tasks in hours. This is used as the
        # lead time resolution as well as the context frame resolution.
        self.temporal_resolution = np.timedelta64(temporal_resolution_hours, "h")

        # First zero-time of the collection of tasks. Since the tasks may have
        # more than one context frame, the first zero-time is not the same as
        # the start time, as we need to account for the first context frame
        self.first_zero_time = self.start_date + self.temporal_resolution * (
            num_context_frames - 1
        )

        # Time window spanned by a single task, including context and
        # target frames. For example, if the temporal resolution is 6 hours,
        # the number of context frames is 3 and the number of target frames
        # is 2, the single task time window will be 30 hours.
        num_frames = num_context_frames + num_target_frames
        self.single_task_time_window = np.timedelta64(
            temporal_resolution_hours * (num_frames - 1),
            "h",
        )

        # Timedeltas between each context frame and zero time.
        self.context_timedeltas = self.temporal_resolution * (
            -np.arange(num_context_frames)
        )

        # Timedeltas between each target frame and zero time.
        self.target_timedeltas = self.temporal_resolution * (
            np.arange(num_target_frames) + 1
        )
        self.dataset = data_source

        # Context and target variable names
        ctx_multilevel_dynamic_variables = _combine_variables_and_levels(
            ctx_multilevel_dynamic_variables, levels
        )
        trg_multilevel_dynamic_variables = _combine_variables_and_levels(
            trg_multilevel_dynamic_variables, levels
        )

        self.trg_variables_and_levels = sorted(
            trg_surface_dynamic_variables + trg_multilevel_dynamic_variables
        )

        self.dyn_ctx_variables_and_levels = sorted(
            ctx_surface_dynamic_variables + ctx_multilevel_dynamic_variables
        )

        if self.trg_variables_and_levels != self.dyn_ctx_variables_and_levels:
            raise ValueError(
                "Target and dynamic context variables must be the same for now, "
                f"got {self.trg_variables_and_levels} and "
                f"{self.dyn_ctx_variables_and_levels}."
            )

        self.static_ctx_variables = sorted(ctx_static_variables)

        self.trg_mean, self.trg_std = _get_normalisation_mean_and_std(
            norm_path, self.trg_variables_and_levels
        )
        self.trg_mean = self.trg_mean.view(1, 1, 1, -1)

        self.dyn_ctx_mean, self.dyn_ctx_std = _get_normalisation_mean_and_std(
            norm_path, self.dyn_ctx_variables_and_levels
        )
        self.dyn_ctx_mean = self.dyn_ctx_mean.view(1, 1, 1, -1)

        static_ctx = self.dataset.sel(variable_and_level=self.static_ctx_variables)
        static_ctx = torch.from_numpy(static_ctx.to_numpy())
        static_mean, static_std = _get_normalisation_mean_and_std(
            norm_path, self.static_ctx_variables
        )
        static_mean = static_mean.view(1, 1, -1)
        static_std = static_std.view(1, 1, -1)

        # Load once and keep normalised static context
        static_ctx = (static_ctx - static_mean) / static_std
        self.static_ctx = static_ctx.permute(2, 1, 0)

        self.task_sub_sampling_factor = task_sub_sampling_factor

    def __len__(self) -> int:
        # Time interval for the specified dates from which we subtract the time needed to form a full task.
        total_time_interval = (
            self.end_date - self.start_date - self.single_task_time_window
        )

        # Number of datapoints within this time interval given the
        # resolution of the data.
        # NOTE: we cast to int as mypy infers np.int64
        num_datapoints = int(total_time_interval // BASE_TEMPORAL_RESOLUTION)

        # Number of subsampled tasks according to task_sub_sampling_factor.
        # NOTE: We add 1 because we start the numbering from 0. For example,
        # if num_datapoints is 9 and task_sub_sampling factor=2, we will use
        # datapoints [0, 2, 4, 6, 8], hence 9 // 2 + 1.
        length = num_datapoints // self.task_sub_sampling_factor + 1

        return length

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = index * self.task_sub_sampling_factor
        zero_time = self.first_zero_time + index * BASE_TEMPORAL_RESOLUTION
        context_times = zero_time + self.context_timedeltas
        target_times = zero_time + self.target_timedeltas

        assert all(ctx_time >= self.start_date for ctx_time in context_times)
        assert all(trg_time <= self.end_date for trg_time in target_times)

        dyn_ctx = self.dataset.sel(
            time=context_times,
            variable_and_level=self.dyn_ctx_variables_and_levels,
        )
        trg = self.dataset.sel(
            time=target_times, variable_and_level=self.trg_variables_and_levels
        )

        # (lat, lon, time, var_and_level) -> (var_and_level, time, lat, lon)
        dyn_ctx = torch.from_numpy(dyn_ctx.to_numpy()).permute(2, 1, 0, 3)
        trg_tensor = torch.from_numpy(trg.to_numpy()).permute(2, 1, 0, 3)

        ctx_tensor = self.normalise_ctx(dyn_ctx)
        trg_tensor = self.normalise_trg(trg_tensor)

        zero_time = _get_hours_from_reference_time(zero_time)  # type: ignore

        return torch.tensor(zero_time), self.static_ctx, ctx_tensor, trg_tensor

    def normalise_trg(self, trg: torch.Tensor) -> torch.Tensor:
        return (trg - self.trg_mean) / self.trg_std

    def normalise_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        return (ctx - self.dyn_ctx_mean) / self.dyn_ctx_std

    def unnormalise_trg(self, trg: torch.Tensor) -> torch.Tensor:
        return trg * self.trg_std + self.trg_mean

    def unnormalise_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        return ctx * self.dyn_ctx_std + self.dyn_ctx_mean


def make_gridded_weather_task(
    data_source: ZarrDatasource,
    start_date: str,
    end_date: str,
    val_start_date: str,
    val_end_date: str,
    num_context_frames: int,
    num_target_frames: int,
    temporal_resolution_hours: int,
    ctx_variables_to_exclude: List[str],
    trg_variables_to_exclude: List[str],
    norm_path: str,
    split: Split,
    generator: torch.Generator,
) -> GriddedWeatherTask:
    ctx_static_variables = ALL_GRID_STATIC_VARIABLES
    ctx_surface_dynamic_variables = ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    ctx_multilevel_dynamic_variables = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
    trg_surface_dynamic_variables = ALL_GRID_SURFACE_DYNAMIC_VARIABLES
    trg_multilevel_dynamic_variables = ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES

    # TODO: Do something with the split/gen

    return GriddedWeatherTask(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        val_start_date=val_start_date,
        val_end_date=val_end_date,
        num_context_frames=num_context_frames,
        num_target_frames=num_target_frames,
        temporal_resolution_hours=temporal_resolution_hours,
        ctx_static_variables=filter_variables(
            ctx_static_variables, ctx_variables_to_exclude
        ),
        ctx_surface_dynamic_variables=filter_variables(
            ctx_surface_dynamic_variables,
            ctx_variables_to_exclude,
        ),
        ctx_multilevel_dynamic_variables=filter_variables(
            ctx_multilevel_dynamic_variables,
            ctx_variables_to_exclude,
        ),
        trg_surface_dynamic_variables=filter_variables(
            trg_surface_dynamic_variables,
            trg_variables_to_exclude,
        ),
        trg_multilevel_dynamic_variables=filter_variables(
            trg_multilevel_dynamic_variables,
            trg_variables_to_exclude,
        ),
        norm_path=norm_path,
        split=split,
    )
