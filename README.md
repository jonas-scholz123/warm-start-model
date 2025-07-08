# Shortcut Models Accelerate Generative Modelling

This repository contains the official implementation of the paper "Shortcut Models Accelerate Generative Modelling".

## Installation

It is recommended to use [pdm](https://pdm.fming.dev/latest/) for dependency management.

1.  **Install PDM:**
    ```bash
    pip install --user pdm
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    pdm venv create
    pdm use
    ```

3.  **Install dependencies:**
    ```bash
    pdm install
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory with your Weights & Biases API key:
    ```
    WANDB_API_KEY=<Your API Key>
    ```

## Running the code

This project uses [Hydra](https://hydra.cc/) for configuration management. You can specify the configuration for a run from the command line.

The main entry point for training is `src/cdnp/train.py`.

### Configuration

The training runs are configured using YAML files located in `src/config`. The main configuration for a specific experiment is selected using the `-cn` flag.

For example, to use the `celeba_inpaint_cdnp.yaml` configuration, you would use `-cn celeba_inpaint_cdnp`.

You can override specific configuration settings from the command line. For debugging purposes, you can use `mode=dev`. This will perform a dry run of the training script without executing the actual training loop, which is useful for catching configuration errors.

### Training CDNP Models (Two-Step Process)

Training a Conditional Denoising Neural Process (CDNP) model is a two-step process:

**Step 1: Train a Conditional Neural Process (CNP) model.**

First, you need to train a standard CNP model for the desired task. The weights from this model will be used as a "shortcut" for the CDNP model.

*Example: Training a CNP for CelebA inpainting:*
```bash
python src/cdnp/train.py -cn celeba_inpaint_cnp mode=prod
```
The weights will be saved to the directory specified in the configuration (by default in `_weights/<experiment_name>`). For the `CDNP` run, the experiment name from the `CNP` run needs to be provided. For `celeba_inpaint_cnp` this will be `celeba_cnp`.

**Step 2: Train the CDNP model.**

Next, train the CDNP model. The configuration for the CDNP model is set up to load the weights from the corresponding pre-trained CNP model. Make sure the `path` in the CDNP config file points to the correct CNP experiment weights.

For example, in `src/config/celeba_inpaint_cdnp.yaml`:
```yaml
model:
  cnp:
    _target_: cdnp.util.instantiate.load_model_from_path
    path: ${paths.weights}/celeba_cnp # Points to the CNP weights
    freeze: true
```

*Example: Training a CDNP for CelebA inpainting:*
```bash
python src/cdnp/train.py -cn celeba_inpaint_cdnp mode=prod
```

### Debugging Example

To debug the configuration for the CIFAR-10 inpainting task, you can use `mode=dev`. This will initialize the models and data loaders but will not run the training loop.

First, for the CNP:
```bash
python src/cdnp/train.py -cn cifar10_inpaint_cnp mode=dev
```
Then, for the CDNP:
```bash
python src/cdnp/train.py -cn cifar10_inpaint_cdnp mode=dev
```

## Citation

If you find this work useful, please consider citing the paper:

```bibtex
@article{scholz2024shortcut,
  title={Shortcut Models Accelerate Generative Modelling},
  author={Jonas Scholz and Richard E. Turner},
  year={2025},
  eprint={TODO},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
