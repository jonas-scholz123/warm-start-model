import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from cdnp.task import PreprocessFn


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    preprocess_fn: PreprocessFn,
    dry_run: bool = False,
) -> dict[str, float]:
    model.eval()
    val_loss = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in val_loader:
            ctx, trg = preprocess_fn(batch)
            ctx = ctx.to(device)
            trg = trg.to(device)
            val_loss += model(ctx, trg).item()

            if dry_run:
                break

    return {"val_loss": val_loss}
