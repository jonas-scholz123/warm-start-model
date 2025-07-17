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

### Training Warm-Start Generative Models (Two-Step Process)

Training a Warm-Start Generative Model is a two-step process:

**Step 1: Train a Warm-Start Model.**

We initially called warm-start models "cnp" (conditional neural processes), hence the naming.

*Example: Training a warm-start model for CelebA inpainting:*
```bash
python src/cdnp/train.py -cn celeba_inpaint_cnp mode=prod
```
The weights will be saved to the directory specified in the configuration (by default in `_output/<experiment_name>/checkpoints`). For the full warm-start diffusion run, you will need to provide the experiment path to the pre-trained warm-start model.

**Step 2: Train the Warm-Start Diffusion model.**

Next, train the full generative model. The configuration for the generative model (we call it CDNP in earlier drafts) is set up to load the weights from the corresponding pre-trained warm-start model. Make sure the `path` in the CDNP config file points to the correct warm-start experiment weights.

For example, in `src/config/celeba_inpaint_cdnp.yaml`:
```yaml
model:
  cnp:
    _target_: cdnp.util.instantiate.load_model_from_path
    path: ${paths.weights}/celeba_cnp # Points to the CNP weights
    freeze: true
```

*Example: Training a warm-start diffusion model for CelebA inpainting:*
```bash
python src/cdnp/train.py -cn celeba_inpaint_cdnp mode=prod
```

### Debugging Example

To debug, you can use `mode=dev`. This will fast-forward through the training loop to reveal bugs.

E.g.:
```bash
python src/cdnp/train.py -cn cifar10_inpaint_cdnp mode=dev
```

## Citation

If you find this work useful, please cite the paper:

```bibtex
@misc{scholz2025warmstarts,
      title={Warm Starts Accelerate Generative Modelling}, 
      author={Jonas Scholz and Richard E. Turner},
      year={2025},
      eprint={2507.09212},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.09212}, 
}
