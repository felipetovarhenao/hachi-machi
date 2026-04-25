import torch
import torch.nn as nn
from .onehot import OneHot


class FeatureScaler(nn.Module):

    def __init__(self, data: torch.Tensor, time_dims: list[int] = [0, 1], voice_dim: int = 2, num_voices: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = data.clone()
        self.register_buffer('time_dims', torch.tensor(time_dims))

        with torch.no_grad():
            self.oh = OneHot(data=data, dim=voice_dim, size=num_voices)
            data = self.log_time(data)
            mean = data.mean(0)
            std = data.std(0)
            mean[voice_dim] = 0
            std[voice_dim] = 1
            std[torch.where(std == 0)] = 1.0

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def log_time(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            x[..., self.time_dims] = torch.expm1(
                x[..., self.time_dims]) * 1000
        else:
            x[..., self.time_dims] = torch.log1p(
                (x[..., self.time_dims]) / 1000)
        return x

    def forward(self, x: torch.Tensor, inverse: bool = False):
        x = x.clone()
        if inverse:
            y = self.oh(x, inverse=True)
            y = y * self.std + self.mean
            y = self.log_time(y, inverse)
        else:
            y = x
            y = self.log_time(x, inverse)
            y = (y - self.mean) / self.std
            y = self.oh(y)
        return y
