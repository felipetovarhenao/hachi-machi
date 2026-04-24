import torch
import torch.nn as nn


class FeatureScaler(nn.Module):

    def __init__(self, data: torch.Tensor, time_dims: list[int] = [0, 1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = data.clone()
        self.e = 1
        self.time_dims = time_dims
        data = self.log_time(data)
        mean = data.mean(0)
        std = data.std(0)
        std[torch.where(std == 0)] = 1.0
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def log_time(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            x[..., self.time_dims] = torch.exp2(
                x[..., self.time_dims]) * 1000 - self.e
        else:
            x[..., self.time_dims] = torch.log2(
                (x[..., self.time_dims] + self.e) / 1000)
        return x

    def forward(self, x: torch.Tensor, inverse: bool = False):
        x = x.clone()
        if inverse:
            y = x * self.std + self.mean
            y = self.log_time(y, inverse)
        else:
            y = self.log_time(x, inverse)
            y = (y - self.mean) / self.std
        return y
