import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in model.parameters()]
        self.shadow_buffers = [b.clone().detach() for b in model.buffers()]
        self.backup = {}

    @torch.no_grad()
    def update(self):
        model_params = self.model.parameters()
        for shadow_p, model_p in zip(self.shadow_params, model_params):
            shadow_p.lerp_(model_p, 1.0 - self.decay)

        model_buffers = self.model.buffers()
        for shadow_b, model_b in zip(self.shadow_buffers, model_buffers):
            shadow_b.copy_(model_b)

    @torch.no_grad()
    def __enter__(self):
        self.backup["params"] = [p.clone() for p in self.model.parameters()]
        self.backup["buffers"] = [b.clone() for b in self.model.buffers()]

        for model_p, shadow_p in zip(self.model.parameters(), self.shadow_params):
            model_p.copy_(shadow_p)

        for model_b, shadow_b in zip(self.model.buffers(), self.shadow_buffers):
            model_b.copy_(shadow_b)

    @torch.no_grad()
    def __exit__(self, *args):
        for model_p, backup_p in zip(self.model.parameters(), self.backup["params"]):
            model_p.copy_(backup_p)

        for model_b, backup_b in zip(self.model.buffers(), self.backup["buffers"]):
            model_b.copy_(backup_b)

        self.backup = {}
