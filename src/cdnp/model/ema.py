from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow_model = deepcopy(model)
        self.backup = {}

    @torch.no_grad()
    def update(self):
        model_params = self.model.parameters()
        shadow_params = self.shadow_model.parameters()
        for shadow_p, model_p in zip(shadow_params, model_params):
            shadow_p.lerp_(model_p, 1.0 - self.decay)

        model_buffers = self.model.buffers()
        shadow_buffers = self.shadow_model.buffers()
        for shadow_b, model_b in zip(shadow_buffers, model_buffers):
            shadow_b.copy_(model_b)

    def get_shadow(self) -> nn.Module:
        return self.shadow_model
