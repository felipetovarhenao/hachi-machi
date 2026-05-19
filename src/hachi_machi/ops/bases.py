import abc
import torch
from typing import Callable


class Operation(abc.ABC):

    def __init__(self, *, dims: list[int] | None = None, p: float = 1.0):
        self.p = max(0.0, min(float(p), 1.0))
        self.dims = dims if dims is not None else slice(None, None)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        x[..., self.dims] = self.forward(x[..., self.dims])
        return x

    @classmethod
    def name(cls):
        return cls.__name__.lower()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def fit(self, _: torch.Tensor): ...


_SCOPES = {
    'global': None,
    'time': 1,
    'feature': 2,
}


class BinaryOperation(Operation):

    def __init__(self, *, value: float | int, scope: str = 'global', **kwargs):
        super().__init__(**kwargs)
        if isinstance(value, str) and value not in ['mean', 'std']:
            raise ValueError(f"{self.name()}: Invalid value: {value}")
        self.value = value
        if scope not in _SCOPES:
            raise ValueError(f"{self.name()}: Invalid scope: {scope}")
        self.dim = _SCOPES[scope]

    def fit(self, x: torch.Tensor):
        if not isinstance(self.value, str):
            return
        attr = self.value
        func = getattr(torch, attr)
        dim = tuple(range(1, x.ndim)) if self.dim is None else self.dim
        self.value = func(x, dim=dim, keepdim=True)
        if attr == 'std':
            self.value += 1e-8


_RAND_SCOPE = {
    'global': lambda x: (x.shape[0], 1, 1),
    'time': lambda x: (x.shape[0], x.shape[1], 1),
    'feature': lambda x: (x.shape[0], 1, x.shape[2]),
    'all': lambda x: x.shape,
}


class RandomOperation(Operation):

    def __init__(self, *, value: tuple[float | int, float | int], scope: str = 'global', dist: str = 'normal', **kwargs):
        super().__init__(**kwargs)
        self.func = self.get_func(value, dist)
        self.scope = scope
        self.shape = None

    def get_func(self, value, dist) -> Callable[[torch.Tensor], torch.Tensor]:
        match dist:
            case 'normal':
                def func(shape): return torch.randn(
                    shape) * value[1] + value[0]
            case 'uniform':
                def func(shape): return torch.rand(shape) * \
                    (value[1] - value[0]) + value[0]
        return lambda x: func(_RAND_SCOPE[self.scope](x))
