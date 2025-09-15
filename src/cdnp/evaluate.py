from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import torch
from torch.amp import autocast
from torch.utils.data.dataloader import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from cdnp.model.cdnp import CDNP
from cdnp.model.cnp import CNP
from cdnp.model.ddpm import DDPM
from cdnp.model.flow_matching.flow_matching import FlowMatching
from cdnp.model.warm_start_diffusion import WarmStartDiffusion
from cdnp.task import ModelCtx, PreprocessFn


def unnormalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = x * std + mean
    return x.clamp(0, 1)


class Metric(ABC):
    @abstractmethod
    def update(
        self, model: CDNP | DDPM | CNP, ctx: ModelCtx, trg: torch.Tensor
    ) -> None:
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class LossMetric(Metric):
    def __init__(self):
        self.loss = 0.0
        self.count = 0

    def update(
        self, model: CDNP | DDPM | CNP, ctx: ModelCtx, trg: torch.Tensor
    ) -> None:
        self.loss += model(ctx, trg).item()
        self.count += 1

    def compute(self) -> float:
        result = self.loss / self.count
        self.loss = 0.0
        self.count = 0
        return result

    def name(self) -> str:
        return "loss"


class FIDMetric(Metric):
    def __init__(
        self,
        num_samples: int,
        means: list[int],
        stds: list[int],
        device: str,
        nfe: Optional[int] = None,
        **sample_kwargs,
    ):
        self.fid = FrechetInceptionDistance(normalize=True).to(
            device, non_blocking=True
        )
        self.num_samples = num_samples
        self.count = 0
        self.device = device
        self.means = torch.tensor(means).view(1, 3, 1, 1).to(device)
        self.stds = torch.tensor(stds).view(1, 3, 1, 1).to(device)
        self.nfe = nfe
        self.sample_kwargs = sample_kwargs

    def update(
        self, model: CDNP | DDPM | CNP, ctx: ModelCtx, trg: torch.Tensor
    ) -> None:
        if self.count > self.num_samples:
            return

        num_samples = trg.shape[0]
        fake_images = model.sample(
            ctx, num_samples=num_samples, nfe=self.nfe, **self.sample_kwargs
        )
        fake_images = unnormalise(fake_images, self.means, self.stds)
        self.fid.update(fake_images, real=False)

        real_images = unnormalise(trg, self.means, self.stds)
        self.fid.update(real_images, real=True)
        self.count += real_images.shape[0]

    def compute(self) -> float:
        result = self.fid.compute().item()
        self.fid.reset()
        self.count = 0
        return result

    def name(self) -> str:
        if self.nfe is not None:
            return f"fid_nfe={self.nfe}"
        return "fid"


class CnpRmseMetric(Metric):
    def __init__(self):
        self.rmse = 0.0
        self.count = 0

    def update(
        self, model: CDNP | DDPM | CNP, ctx: ModelCtx, trg: torch.Tensor
    ) -> None:
        assert isinstance(model, CNP), "Model must be a CNP"
        pred = model.predict(ctx).mean
        self.rmse += torch.sqrt(torch.mean((pred - trg) ** 2)).item()  # ty:ignore
        self.count += 1

    def compute(self) -> float:
        result = self.rmse / self.count
        self.rmse = 0.0
        self.count = 0
        return result

    def name(self) -> str:
        return "cnp_rmse"


class EnsembleMetric(ABC):
    @abstractmethod
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class CRPS(EnsembleMetric):
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> float:
        trg = trg.expand_as(samples[..., 0])

        # Term A: E|X - trg|  (Monte-Carlo mean over samples)
        a = (samples - trg.unsqueeze(-1)).abs().mean(dim=-1)

        # Pairwise absolute differences
        # Expand: (..., N, 1) and (..., 1, N)
        diffs = (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs()
        b = 0.5 * diffs.mean(dim=(-1, -2))

        crps = a - b
        return crps.mean().item()

    def name(self) -> str:
        return "crps"


class EnsembleRmse(EnsembleMetric):
    def compute(self, samples: torch.Tensor, trg: torch.Tensor) -> float:
        pred = samples.mean(dim=-1)
        rmse = torch.sqrt(torch.mean((pred - trg) ** 2))
        return rmse.item()

    def name(self) -> str:
        return "ensemble_rmse"


@torch.no_grad()
def evaluate(
    model: CDNP | DDPM | CNP,
    dataloader: DataLoader,
    preprocess_fn: PreprocessFn,
    metrics: list[Metric],
    use_tqdm: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    # TODO: make sure we don't repeat forward passes
    with autocast(device_type=device.type, dtype=torch.float16):
        for batch in tqdm(dataloader, disable=not use_tqdm):
            ctx, trg = preprocess_fn(batch)
            ctx = ctx.to(device)
            trg = trg.to(device)

            for metric in metrics:
                metric.update(model, ctx, trg)

            if dry_run:
                break

    return {metric.name(): metric.compute() for metric in metrics}


@torch.no_grad()
def evaluate_ensemble(
    model: FlowMatching | WarmStartDiffusion,
    dataloader: DataLoader,
    preprocess_fn: PreprocessFn,
    metrics: list[EnsembleMetric],
    num_samples: int,
    use_tqdm: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    sums = defaultdict(int)
    count = 0

    # TODO: make sure we don't repeat forward passes
    with autocast(device_type=device.type, dtype=torch.float16):
        for batch in tqdm(dataloader, disable=not use_tqdm):
            ctx, trg = preprocess_fn(batch)
            ctx = ctx.to(device)
            trg = trg.to(device)  # B, C, H, W

            samples = []
            for _ in range(num_samples):
                samples.append(model.sample(ctx, num_samples=-1))
            samples = torch.stack(samples, dim=-1)  # B, C, H, W, N

            for metric in metrics:
                result = metric.compute(samples, trg)
                sums[metric.name()] += result
            count += 1

            if dry_run:
                break

    return {name: total / count for name, total in sums.items()}
