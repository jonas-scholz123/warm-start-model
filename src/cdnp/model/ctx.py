from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelCtx:
    image_ctx: Optional[torch.Tensor] = None
    label_ctx: Optional[torch.Tensor] = None

    def to(self, device: str, non_blocking: bool = False) -> "ModelCtx":
        return ModelCtx(
            image_ctx=self.image_ctx.to(device, non_blocking=non_blocking)
            if self.image_ctx is not None
            else None,
            label_ctx=self.label_ctx.to(device, non_blocking=non_blocking)
            if self.label_ctx is not None
            else None,
        )
