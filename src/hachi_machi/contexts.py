import torch
import torch.nn as nn


class AdaptiveNoise:

    def __init__(self,
                 model: nn.Module,
                 std: float = 0.01,
                 decay: float = 1.0):
        self.model = model
        self.std = std
        self.decay = min(1, max(0, decay))
        self.params = {}
        self.epoch = 0

    @torch.no_grad()
    def __enter__(self):
        std = self.std * self.decay ** self.epoch
        for name, param in self.model.named_parameters():
            self.params[name] = param.data.clone()
            param.data.add_(torch.randn_like(param) * std)
        return self

    @torch.no_grad()
    def __exit__(self, *_):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.params[name])
        self.params.clear()

    def __call__(self, epoch: int):
        self.epoch = epoch
        return self


class NullContext:

    def __enter__(self):
        return self

    def __exit__(self, *_): ...

    def __call__(self, *_):
        return self
