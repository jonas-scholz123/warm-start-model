import torch
from torch import nn
from torch.amp import autocast
from torch.utils.data.dataloader import DataLoader

from cdnp.task import PreprocessFn


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    preprocess_fn: PreprocessFn,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    val_loss = 0
    device = next(model.parameters()).device

    with autocast(device_type=device.type, dtype=torch.float16):
        for batch in val_loader:
            ctx, trg = preprocess_fn(batch)
            ctx = ctx.to(device)
            trg = trg.to(device)
            val_loss += model(ctx, trg).item()

            if dry_run:
                break

    val_loss /= len(val_loader)

    return {"val_loss": val_loss}
