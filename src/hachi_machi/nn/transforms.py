import torch
import torch.nn as nn
import torch.nn.functional as F
from ..features import FeatureMap
from abc import ABC, abstractmethod


class BaseTransform(nn.Module, ABC):

    BIJECTIVE = True

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def bijective(cls):
        return cls.BIJECTIVE

    def fit(self, data: torch.Tensor):
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, inverse: bool = False):
        ...


class Normalize(BaseTransform):

    mean: torch.Tensor
    std: torch.Tensor

    def fit(self, data: torch.Tensor) -> None:
        std = data.std(0)
        std[std == 0] = 1
        self.register_buffer('mean', data.mean(0))
        self.register_buffer('std', std)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        return x * self.std + self.mean if inverse else (x - self.mean) / self.std


class Categorical(BaseTransform):

    dims: torch.Tensor
    sizes: torch.Tensor

    def __init__(self, dims: list[int]):
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
            dim = dim.item() + offset
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
            x = torch.cat([l, m, r], dim=-1)
        return x


class TimePhase(BaseTransform):

    dims: torch.Tensor
    BIJECTIVE = False

    def __init__(self, dims: list[int]):
        super().__init__()
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return x[..., :len(self.dims) * -2]
        c = x[..., self.dims]
        t = torch.zeros_like(c)
        mask = c > 0
        t[mask] = (torch.log2(c[mask] / 1000.0) % 1) * torch.pi * 2
        return torch.cat([x, torch.cos(t), torch.sin(t)], dim=-1)


class LogSpace(BaseTransform):

    dims: torch.Tensor

    def __init__(self, dims: list[int]):
        super().__init__()
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.int))

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        x[..., self.dims] = torch.log1p(
            x[..., self.dims]) if not inverse else torch.expm1(x[..., self.dims])
        return x


class Transform(BaseTransform):

    def __init__(self, layers: list[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList(layers)
        self.input_size = 0
        self.output_size = 0

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
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


class HarmonicScore(BaseTransform):

    def __init__(self, dims: list[int], n: int = 4, sigma: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        i = torch.arange(1, n + 1, dtype=torch.float32)
        ii, jj = torch.meshgrid(i, i, indexing='ij')
        self.register_buffer('i', ii.unsqueeze(-1))
        self.register_buffer('j', jj.unsqueeze(-1))
        self.register_buffer('w', 1 / (self.i * self.j))
        self.register_buffer('dims', torch.tensor(dims, dtype=torch.int))
        self.sigma = sigma

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return x[..., :-len(self.dims)]
        xd = x[..., self.dims].unsqueeze(-2).unsqueeze(-2) / 1000
        diff = torch.abs(self.i - self.j * xd)
        score = self.w * torch.exp(-0.5 * (diff / self.sigma) ** 2)
        score = score.transpose(-3, -1).sum((-2, -1))
        return torch.cat([x, score], dim=-1)


class TransformFactory:

    REQUIRED = ['categorical', 'normalize']

    OPTIONS = {
        "time-phase": {
            "cls": TimePhase,
        },
        "harmonic-score": {
            "cls": HarmonicScore,
        },
        "log-space": {
            "cls": LogSpace,
        },
        'categorical': {
            "cls": Categorical,
            "type": "categorical"
        },
        "normalize": {
            "cls": Normalize,
        },

    }

    @classmethod
    def options(cls) -> list:
        return list(set(cls.OPTIONS.keys()).difference(cls.REQUIRED))

    def __init__(self, feature_map: FeatureMap):
        self.feature_map = feature_map

    @torch.no_grad()
    def make(self,
             data: torch.Tensor,
             transforms: list[str]) -> tuple[Transform, Transform]:
        transforms = list(set([*transforms, *self.REQUIRED]))
        transform_layers = []
        for i, io in enumerate(['input', 'output']):
            layers = []
            fn = getattr(self.feature_map, f"{io}_dims")
            dims = fn()
            data_subset = data[..., dims]
            for k in self.OPTIONS.keys():
                if k not in transforms:
                    continue
                opt: dict = self.OPTIONS[k]
                cls: BaseTransform = opt['cls']
                if i == 1 and not cls.bijective():
                    continue
                type = opt.get('type', None)
                dims: list = fn(type)
                layer = cls(dims)
                layers.append(layer)
            t = Transform(layers).to(data.device)
            t.fit(data_subset)
            transform_layers.append(t)
        return tuple(transform_layers)
