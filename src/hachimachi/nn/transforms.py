import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class TransformLayer(nn.Module, ABC):

    def fit(self, *args, **kwargs): ...

    def forward(self, x: torch.Tensor, inverse: bool = False): ...


class Normalize(TransformLayer):
    def __init__(self, size: int = 2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('std', torch.ones(size))

    def fit(self, x: torch.Tensor) -> None:
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        return x * self.std + self.mean if inverse else (x - self.mean) / self.std


class Categorical(TransformLayer):

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


class TimePhase(TransformLayer):
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


class LogSpace(TransformLayer):

    def __init__(self, dims: list[int] = [0]):
        super().__init__()
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        x[..., self.dims] = torch.log1p(
            x[..., self.dims]) if not inverse else torch.expm1(x[..., self.dims])
        return x


class CategorySum(TransformLayer):

    def __init__(self, key_dim: int = 0, value_dim: int = 0, num_voices: int = 3):
        super().__init__()
        self.register_buffer('cache', torch.zeros(num_voices))
        self.register_buffer('key_dim', torch.tensor(key_dim, dtype=torch.int))
        self.register_buffer('value_dim', torch.tensor(
            value_dim, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            return x[..., :-1]
        keys = x[..., self.key_dim]
        values = x[..., self.value_dim]
        same_key = keys.unsqueeze(2) == keys.unsqueeze(1)
        causal = torch.ones(x.shape[1], x.shape[1]).tril().bool()
        mask = same_key & causal
        cumsum = (mask * values.unsqueeze(1)).sum(dim=2)
        y = torch.cat([x, cumsum.unsqueeze(-1)], dim=-1)
        return y


class Transform(TransformLayer):

    def __init__(self, layers: list[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList(layers)
        self.input_size = 0
        self.output_size = 0

    def forward(self, x: torch.Tensor, inverse: bool = False):
        layers = reversed(self.layers) if inverse else self.layers
        for layer in layers:
            x = layer(x, inverse)
        return x

    @torch.no_grad()
    def fit(self, data: torch.Tensor):
        x = data.clone()
        self.input_size = data.size(-1)
        for layer in self.layers:
            layer.fit(x)
            x = layer(x)
        self.output_size = x.size(-1)


if __name__ == '__main__':
    vd = CategorySum()
    t = torch.randint(0, 5, size=(3, 4, 1)) * 150
    c = torch.randint_like(t, 2)
    x = torch.cat([t, c], dim=-1).to(torch.float32)
    y = vd(x)
    x_hat = vd(y, True)
    print(x == x_hat)
