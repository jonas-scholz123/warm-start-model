# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of "Warm Starts Accelerate Generative Modelling" — a two-step conditional generation pipeline:
1. Train a **CNP** (Conditional Neural Process) as a warm-start model that predicts mean/std of the target.
2. Train a **WarmStartDiffusion** (or FlowMatching) model that uses the CNP's predictions to accelerate generation.

## Commands

### Development

```bash
# Install dependencies
pdm install

# Run linting (ruff + ty type checker) - must be run via pdm so executables are on PATH
pdm run ruff check src tests --fix
pdm run ty check src tests
pre-commit run --all-files
```

### Training

```bash
# Train a model — mode is required (dev or prod)
pdm run python src/cdnp/train.py -cn cifar10_inpaint_cnp mode=prod

# Quick dev mode (dry_run=true, no wandb)
pdm run python src/cdnp/train.py -cn cifar10_inpaint_cnp mode=dev

# Override config values
pdm run python src/cdnp/train.py -cn cifar10_inpaint_fm_warm mode=prod execution.train_steps=10000
```

### Testing

```bash
# Run all integration tests
pytest tests/integration/test_training.py -v

# Run a specific config test
pytest "tests/integration/test_training.py::test_training_works[cifar10_inpaint_cnp]"
```

## Config System (Hydra)

All configs live in `src/config/`. Experiment configs compose multiple sub-configs via Hydra defaults lists. `mode` is always required at the CLI.

### Composition axes

| Group | Options |
|---|---|
| `data` | `celeba`, `cifar10`, `afhq`, `era5_wind`, `era5_full` |
| `task` | `imgen`, `inpaint`, `colourisation`, `superresolution`, `forecast`, `era5_inpaint` |
| `model` | `cnp`, `flow_matching`, `flow_matching_no_attention`, `warm_start_fm`, `warm_start_fm_no_attention` |
| `model/backbone` | `small`, `big`, `no_attention`, `small_no_attention`, `tiny_no_attention`, `swin` |
| `model_type` | `deterministic` (CNP defaults), `generative` (FM defaults) |
| `mode` | `dev`, `prod` |

### Resolution order

Within an experiment config's defaults list, later entries override earlier ones. The typical order is:

```
base → data/<dataset> → task/<task> → model/<model> → model/backbone/<backbone> → model_type/<type> → _self_
```

`_self_` (the experiment file itself) goes last, so it wins over all sub-configs.

### model_type configs

`model_type/deterministic.yaml` and `model_type/generative.yaml` set model-family defaults that individual experiment files can override:
- `deterministic`: `train_steps: 6M`, `gradient_clip_norm: 10`, no EMA, LossMetric only
- `generative`: `train_steps: 5M`, `gradient_clip_norm: 3`, `ema_rate: 0.999`, `accumulate_steps: 4`, FID metrics

### Host configs for ERA5 data paths

ERA5 data configs (`data/era5_wind.yaml`, `data/era5_full.yaml`) include a `host` sub-config that injects machine-specific paths into `data.dataset.data_source.path` and `data.dataset.norm_path`. The host defaults to `desktop` but can be overridden at the CLI:

```bash
# Run on CBL cluster
pdm run python src/cdnp/train.py -cn era5_full_forecast_cnp mode=prod data/host=cbl

# Run on HPC
pdm run python src/cdnp/train.py -cn era5_full_forecast_cnp mode=prod data/host=hpc
```

Host configs use `@package _global_` so they merge at the config root. Available hosts and their data roots:
- `desktop`: `/home/jonas/Documents/code/otter/`
- `cbl`: `/scratches/peano_2/em626/otter/`
- `hpc`: `/rds/project/rds-dHudQI6NhiE/otter/data/`

### Input/output channel counts

Experiment configs compute channel counts explicitly (with comments). The pattern for warm-start FM models:
- `in_channels` = (context frames × variables) + static channels + noise channels + CNP mean channels + CNP std channels
- Example `era5_full_forecast_fm_warm.yaml`: `2*83 + 5 + 83 + 83 + 83 = 420`

Backbone configs use `@package _global_` so they override `model.backbone` directly.

### Warm-start model loading

FM experiment configs that use a pre-trained CNP set `model.warm_start_model` to load from `${paths.weights}/<name>`, e.g. `${paths.weights}/cifar10_cnp`. The weights directory is `_weights/` at the project root.

### Config dataclasses

`src/config/config.py` defines typed dataclasses (`Config`, `DataConfig`, `ExecutionConfig`, `OutputConfig`, `Paths`, `RuntimeConfig`) that serve as the structured config schema. Hydra validates composed configs against these.

## Architecture

### Training Pipeline

`src/cdnp/train.py` → `Trainer` (from `mlbnb`) → `train_loop()`

`src/cdnp/util/instantiate.py` contains `Experiment.from_config()` which creates all training dependencies (model, dataloaders, optimizer, scheduler, EMA) from a config object.

### Models

All models in `src/cdnp/model/`:

- **`CNP`** (`cnp.py`): Warm-start model. Takes context and returns a `Normal` distribution (mean + std). Trained with NLL loss.
- **`WarmStartDiffusion`** (`warm_start_diffusion.py`): Wraps CNP + a generative model. Key params: `warm_start_model`, `min_warmth`/`max_warmth`, `end_to_end`.
- **`DDPM`** (`ddpm.py`): Standard diffusion using HuggingFace `DDPMScheduler`.
- **`FlowMatching`** (`model/flow_matching/flow_matching.py`): ODE-based generative model.

`src/cdnp/model/meta/` and `src/cdnp/model/flow_matching/` are excluded from linting and type checking.

UNet backbones come from HuggingFace `diffusers` (`UNet2DModel`) or the custom Swin Transformer in `model/swin/`.

### Task Preprocessing

`src/cdnp/task.py` defines `PreprocessFn` callables that transform raw batches into `(context, target)` pairs:
- `preprocess_imgen`: Unconditional generation
- `preprocess_inpaint`: Random masking
- `preprocess_ccgen`: Class-conditional generation
- `preprocess_weather_forecast`: Temporal forecasting (ERA5)
- `preprocess_weather_inpaint`: Weather inpainting

### Data

`src/cdnp/data/` has dataset classes per domain. ERA5 weather data (`era5.py`) uses `xarray`/`zarr` via `ZarrDatasource`. Image datasets wrap torchvision.

### Evaluation

`src/cdnp/evaluate.py`: `LossMetric`, `FIDMetric`, `CnpRmseMetric`. Metrics are computed at intervals and logged to W&B.

## Key Paths

- Training outputs/checkpoints: `_output/`
- Pre-trained weights (warm-start loading): `_weights/`
- Normalisation stats: `_normalisation/`
- Data: `_data/` (symlink)
