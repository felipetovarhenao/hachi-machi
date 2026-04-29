import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, size: int = 2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('std', torch.ones(size))

    def fit(self, x: torch.Tensor) -> None:
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        return x * self.std + self.mean if inverse else (x - self.mean) / self.std


class Categorical(nn.Module):

    def __init__(self, dim: int = 0, size: int = 2):
        super().__init__()
        self.register_buffer('dim', torch.tensor(dim, dtype=torch.int))
        self.register_buffer('size', torch.tensor(size, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        i = self.dim
        s = self.size
        if not inverse:
            l, m, r = x[..., :i], x[..., i:i+1], x[..., i+1:]
            m = F.one_hot(m.to(torch.long), num_classes=s).squeeze(-2)
        else:
            l, m, r = x[..., :i], x[..., i:i+s], x[..., i+s:]
            m = torch.argmax(m, dim=-1).unsqueeze(-1)
        return torch.cat([l, m, r], dim=-1)


class TimePhase(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.register_buffer('dim', torch.tensor(dim, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            return x[..., :-2]
        i = self.dim
        l, c, r = x[..., :i], x[..., i:i+1], x[..., i+1:]
        t = torch.zeros_like(c)
        mask = c > 0
        t[mask] = (torch.log2(c[mask] / 1000.0) % 1) * torch.pi * 2
        return torch.cat([l, c, r, torch.cos(t), torch.sin(t)], dim=-1)


class LogSpace(nn.Module):

    def __init__(self, dims: list[int] = [0]):
        super().__init__()
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        x[..., self.dims] = torch.log1p(
            x[..., self.dims]) if not inverse else torch.expm1(x[..., self.dims])
        return x


class Transform(nn.Module):

    def __init__(self, layers: list[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList(layers)
        self.output_size = 0

    def forward(self, x: torch.Tensor, inverse: bool = False):
        layers = reversed(self.layers) if inverse else self.layers
        for layer in layers:
            x = layer(x, inverse)
        return x

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        x = data.clone()
        for layer in self.layers:
            if hasattr(layer, 'fit'):
                layer.fit(x)
            else:
                x = layer(x)
        self.output_size = x.size(-1)
