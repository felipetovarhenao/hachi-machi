import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


class TransformLayer(nn.Module, ABC):

    BIJECTIVE = True

    @classmethod
    def bijective(cls):
        return cls.BIJECTIVE

    def fit(self, *args, **kwargs): ...

    def forward(self, x: torch.Tensor, inverse: bool = False): ...


class Normalize(TransformLayer):
    def __init__(self, size: int = 2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('std', torch.ones(size))

    def fit(self, x: torch.Tensor) -> None:
        std = x.std(0)
        std[std == 0] = 1
        self.mean.copy_(x.mean(0))
        self.std.copy_(std)

    def forward(self, x: torch.Tensor, inverse: bool = False):
        return x * self.std + self.mean if inverse else (x - self.mean) / self.std


class Categorical(TransformLayer):

    def __init__(self, dims: list[int] = [0]):
        super().__init__()
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.int))
        self.register_buffer('sizes', torch.zeros(len(dims), dtype=torch.int))

    def fit(self, data: torch.Tensor):
        for i, dim in enumerate(self.dims):
            is_class = data[..., dim].frac().eq(0).all()
            if not is_class:
                raise ValueError(
                    f"All values along feature dimension {dim} must be integers to be handled as categorical.")
            classes = data[..., dim].unique()
            self.register_buffer(f'classes_{i}', classes)
            self.get_buffer
            self.sizes[i] = len(classes)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        offset = 0
        for i, (dim, size) in enumerate(zip(self.dims, self.sizes)):
            dim += offset
            classes = self.get_buffer(f'classes_{i}')
            if not inverse:
                l, m, r = x[..., :dim], x[..., dim:dim+1], x[..., dim+1:]
                m = m.squeeze(-1).reshape(-1)
                all_classes = torch.cat([classes, m])
                m = torch.unique(all_classes, return_inverse=True)[1]
                m = F.one_hot(m[size:], num_classes=size).to(
                    x.dtype).reshape(*x.shape[:-1], size)
                offset += (size - 1)
            else:
                l, m, r = x[..., :dim], x[..., dim:dim+size], x[..., dim+size:]
                m = classes[torch.argmax(m, dim=-1)].unsqueeze(-1)
                offset -= (size - 1)
            x = torch.cat([l, m, r], dim=-1)
        return x


class TimePhase(TransformLayer):

    BIJECTIVE = False

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

    BIJECTIVE = False

    def __init__(self, key_dim: int = 0, value_dim: int = 0, num_voices: int = 3):
        super().__init__()
        self.register_buffer('cache', torch.zeros(num_voices))
        self.register_buffer('key_dim', torch.tensor(key_dim, dtype=torch.int))
        self.register_buffer('value_dim', torch.tensor(
            value_dim, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            return x[..., :-1]
        if x.size(0) > 1:
            return self._batch_forward(x)
        return self._step_forward(x)

    def _step_forward(self, x: torch.Tensor):
        k = x[..., self.key_dim].to(torch.long)
        v = x[..., self.value_dim].item()
        r = self.cache[k:k+1].clone().unsqueeze(0).unsqueeze(0)
        mask = F.one_hot(k, self.cache.size(0)).flatten().bool()
        self.cache[~mask] += v
        self.cache[mask] = v
        return torch.cat([x, r], dim=-1)

    def _batch_forward(self, x: torch.Tensor) -> torch.Tensor:
        keys = x[..., self.key_dim]
        values = x[..., self.value_dim]
        same_key = keys.unsqueeze(-1) == keys.unsqueeze(-2)
        causal = torch.ones(x.shape[-2], x.shape[-2]
                            ).to(x.device).tril().bool()
        mask = same_key & causal
        cumsum = (mask * values.unsqueeze(-1)).sum(dim=-2)
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


class TransformFactory:

    @classmethod
    def options(cls):
        return list(cls(0, 0, 0)._options.keys())

    def __init__(self, voice_dim: int, time_dim: int, num_voices: int):
        self._size = 0
        self._options = {
            "category-sum": {
                "cls": CategorySum,
                "kwargs": {"key_dim": voice_dim, "value_dim": time_dim, "num_voices": num_voices},
                "output_size": lambda: self._size + 1
            },
            "time-phase": {
                "cls": TimePhase,
                "kwargs": {"dim": time_dim},
                "output_size": lambda: self._size + 2
            },
            "log-space": {
                "cls": LogSpace,
                "kwargs": {"dims": [time_dim]},
            },
            'categorical': {
                "cls": Categorical,
                "kwargs": {
                    "dims": [voice_dim,],
                },
                "output_size": lambda: self._size + (num_voices - 1)
            },
            "normalize": {
                "cls": Normalize,
                "kwargs": lambda: {"size": self._size},
            }
        }

    @torch.no_grad()
    def make(self,
             input_data: torch.Tensor,
             output_data: torch.Tensor,
             transforms: list[str]) -> tuple[Transform, Transform]:
        transform_layers = []
        for i, data in enumerate([input_data, output_data]):
            self._size = data.size(-1)
            layers = []
            for k in self._options.keys():
                if k not in transforms:
                    continue
                opt = self._options[k]
                cls: TransformLayer = opt['cls']
                if i == 1 and not cls.bijective():
                    continue
                kwargs = opt['kwargs']
                if 'output_size' in opt:
                    self._size = opt['output_size']()
                if callable(kwargs):
                    kwargs = kwargs()
                layer = cls(**kwargs)
                layers.append(layer)
            t = Transform(layers).to(data.device)
            t.fit(data)
            transform_layers.append(t)
        return tuple(transform_layers)
