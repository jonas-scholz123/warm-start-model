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

# Run linting (ruff + ty type checker)
pre-commit run --all-files
ruff check src tests --fix
ty check src tests
```

### Training

```bash
# Train a model (Hydra config with -cn flag)
python src/cdnp/train.py -cn cifar10_inpaint_cnp

# Quick dev mode (fewer steps, debugging)
python src/cdnp/train.py -cn cifar10_inpaint_cnp mode=dev

# Override config values
python src/cdnp/train.py -cn cifar10_inpaint_fm_warm execution.train_steps=10000
```

### Testing

```bash
# Run all integration tests
pytest tests/integration/test_training.py -v

# Run a specific config test
pytest "tests/integration/test_training.py::test_training_works[cifar10_inpaint_cnp]"
```

## Architecture

### Config System (Hydra)

All configs live in `src/config/`. The base config is `base.yaml`; experiment configs compose from it using Hydra defaults. Key composition axes:
- `data`: celeba, cifar10, afhq, mnist, era5_wind, era5_full
- `task`: imgen, inpaint, colorization, forecast
- `model`: cnp, warm_start_fm, flow_matching
- `model/backbone`: small, big, swin, tiny_no_attention, no_attention
- `mode`: dev (quick run), prod

Config dataclasses are in `src/config/config.py` and define the typed schema for all config fields.

### Training Pipeline

`src/cdnp/train.py` → `Trainer` (from `mlbnb`) → `train_loop()`

`src/cdnp/util/instantiate.py` contains `Experiment.from_config()` which creates all training dependencies (model, dataloaders, optimizer, scheduler, EMA) from a config object.

### Models

All models in `src/cdnp/model/`:

- **`CNP`** (`cnp.py`): Warm-start model. Takes context (e.g., observed pixels) and returns a `Normal` distribution (mean + std) over the target. Used for NLL training loss.
- **`WarmStartDiffusion`** (`warm_start_diffusion.py`): Main generative model. Wraps a CNP + a generative model (DDPM or FlowMatching). Key params: `warm_start_model`, `min_warmth`/`max_warmth` (scale warm-start contribution), `end_to_end` (fine-tune CNP jointly).
- **`DDPM`** (`ddpm.py`): Standard diffusion using HuggingFace `DDPMScheduler`.
- **`FlowMatching`** (`model/flow_matching/flow_matching.py`): ODE-based generative model. `src/cdnp/model/meta/` and `src/cdnp/model/flow_matching/` are excluded from linting and type checking.

UNet backbones come from HuggingFace `diffusers` (`UNet2DModel`) or the custom Swin Transformer in `model/swin/`.

### Task Preprocessing

`src/cdnp/task.py` defines `PreprocessFn` callables that transform raw batches into `(context, target)` pairs:
- `preprocess_imgen`: Unconditional generation
- `preprocess_inpaint`: Random masking for inpainting
- `preprocess_ccgen`: Class-conditional generation
- `preprocess_forecast`: Temporal forecasting (ERA5)

### Data

`src/cdnp/data/` has dataset classes per domain. ERA5 weather data uses `xarray`/`zarr`. Image datasets wrap standard torchvision datasets.

### Evaluation

`src/cdnp/evaluate.py`: `LossMetric` (NLL/MSE) and `FIDMetric`. Metrics are computed at intervals defined in config and logged to W&B.

### Sampling

`src/cdnp/sampler/dpm_solver.py`: DPM-Solver for efficient inference. `src/cdnp/sampler/` also contains wrappers for DDPM and FlowMatching samplers.

## Key Paths

- Training outputs/checkpoints: `_output/`
- Pre-trained weights (for warm-start loading): `_weights/`
- Normalisation stats: `_normalisation/`
- Data: `_data/` (symlink)
