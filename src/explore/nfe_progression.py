# %%
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath
from tqdm import tqdm

from cdnp.model.flow_matching.flow_matching import FlowMatching
from cdnp.util.instantiate import Experiment

torch.manual_seed(42)  # Set a seed for reproducibility.

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": "STIXGeneral",
        "mathtext.fontset": "stix",
    }
)


@dataclass
class Args:
    exp_name: str
    title: str
    model: str = "latest_ema"
    solver: str = "heun2"
    skip_type: str = "logSNR"


# args = Args(experiment="2025-07-23_15-24_sassy_unicorn_better_cnp", model="latest_ema")
args = [
    Args(
        exp_name="2025-07-21_22-38_playful_xenon",
        title="FM",
        solver="heun2",
    ),
    Args(
        exp_name="2025-07-23_15-24_sassy_unicorn_better_cnp",
        title="Warm FM",
        solver="heun2",
    ),
    Args(
        exp_name="2025-07-21_22-38_playful_xenon",
        title="FM + DPM Solver++",
        solver="dpm_solver_3",
    ),
    Args(
        exp_name="2025-07-23_15-24_sassy_unicorn_better_cnp",
        title="Warm FM + DPM Solver++",
        solver="dpm_solver_3",
    ),
]
nfes = [1, 2, 3, 4, 5, 8, 12, 20, 50]

root = Path("/home/jonas/Documents/code/denoising-np/_weights")


path = root / args[0].exp_name
path = ExperimentPath.from_path(path)
cfg = path.get_config()
exp = Experiment.from_config(cfg)  # ty: ignore
dataloader = exp.train_loader
batch = next(iter(dataloader))
ctx, x = exp.preprocess_fn(batch)  # ty: ignore
ctx = ctx.to("cuda")


def load_model(args: Args):
    path = Path("/home/jonas/Documents/code/denoising-np/_weights") / args.exp_name
    path = ExperimentPath.from_path(path)
    cfg = path.get_config()
    exp = Experiment.from_config(cfg)  # ty: ignore
    model: FlowMatching = exp.model  # ty: ignore
    cm = CheckpointManager(path)
    if "ema" in args.model:
        model = exp.ema_model.get_shadow()  # ty: ignore
    else:
        model = model
    _ = cm.reproduce_model(model, args.model)
    return model, exp.val_loader


def generate_samples(args: Args):
    model, _ = load_model(args)
    gen = torch.Generator(device="cpu")
    all_samples = []
    for nfe in tqdm(nfes):
        gen.manual_seed(42)  # Set a seed for same noise across iterations.
        samples = model.sample(
            ctx=ctx,
            x_T=x,
            nfe=nfe,
            ode_method=args.solver,
            skip_type=args.skip_type,
            num_samples=0,  # ignored
            gen=gen,
        )
        all_samples.append(samples)

    samples = torch.stack(all_samples, dim=1)
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    return samples


# %%
all_samples = []
for i, arg in enumerate(args):
    samples = generate_samples(arg)
    all_samples.append(samples)
# %%
fontsize = 12
idx = 3

for idx in range(samples.shape[0]):
    fig, axs = plt.subplots(
        len(args),
        len(nfes),
        figsize=(2 * len(nfes), 2 * len(args)),
        constrained_layout=True,
    )

    for i, (arg, samples) in enumerate(zip(args, all_samples)):
        for j, nfe in enumerate(nfes):
            axs[i, j].imshow(samples[idx, j].permute(1, 2, 0).cpu().numpy())
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i == 0:
                axs[0, j].set_title(f"NFE: {nfe}", fontsize=fontsize)
        axs[i, 0].set_ylabel(arg.title, rotation=90, fontsize=fontsize)
    plt.show()
# %%
