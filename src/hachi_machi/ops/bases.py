import abc
import torch
from typing import Callable
import inspect
from collections import OrderedDict


class Operation(abc.ABC):

    DOCS = {
        'dims': 'Optional feature dimensions to apply operation to. If none are provided, all feature dimensions are used.',
        'p': 'Probability for operation to be applied to data sequence'
    }

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

    @classmethod
    def docs(cls):
        docs = OrderedDict()
        for base in cls.__mro__:
            if not hasattr(base, 'DOCS'):
                continue
            docs.update(**base.DOCS)

        return docs

    @classmethod
    def get_signature(cls):
        params = OrderedDict()
        for klass in cls.__mro__:
            if klass is object:
                continue
            for name, param in inspect.signature(klass.__init__).parameters.items():
                if name in ("self", "args", "kwargs"):
                    continue
                params[name] = param
        return params


_SCOPES = {
    'global': None,
    'time': -2,
    'feature': -1,
}


class BinaryOperation(Operation):

    DOCS = {
        'value': ("Numeric constant, or one of the following keywords for referencing data-derived properties."
                  "\n\t- `mean`\n\t- `std`\n"),
        'scope': ("Reduction axis for data-derived values. Ignored when `value` is a constant:\n"
                  "\n\t- `global`: data-derived value is based on all dimensions and time steps."
                  "\n\t- `time`: data-derived value is computed along the time-step dimension."
                  "\n\t- `feature`: data-derived value is computed for each step along the feature dimension.")
    }

    def __init__(self, *, value: float | int = 0, scope: str = 'global', **kwargs):
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
    'global': lambda _: (1, 1),
    'time': lambda x: (x.shape[1], 1),
    'feature': lambda x: (1, x.shape[2]),
    'all': lambda x: x.shape,
}


class RandomOperation(Operation):

    DOCS = {
        'value': "Two values representing the random range, based on `dist`.",
        'scope': ("Shape of the random tensor"
                  "\n\t- `global`: a single random value is applied globally"
                  "\n\t- `time`: random values are applied, one for each time step, but constant for all feature `dims`"
                  "\n\t- `feature`: random values are applied, one for each feature, but constant for all steps"
                  "\n\t- `both`: random values are applied, one for each feature and time steps"),
        'dist': ("Type of random distribution to sample from."
                 "\n\t- `uniform`: Even distribution. `value` denotes the minimum and maximum value."
                 "\n\t- `normal`: Gaussian distribution. `value` denotes the mean and standard deviation."),
    }

    def __init__(self,
                 *,
                 value: tuple[float | int, float | int] = (0, 1),
                 scope: str = 'global',
                 dist: str = 'uniform',
                 log: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        rand = self.get_func(value, dist, scope)
        if not log:
            self.func = rand
        else:
            self.func = lambda x: 2 ** rand(x)

    def get_func(self, value, dist, scope) -> Callable[[torch.Tensor], torch.Tensor]:
        match dist:
            case 'normal':
                def rand(shape): return torch.randn(
                    shape) * value[1] + value[0]
            case 'uniform':
                def rand(shape): return torch.rand(shape) * \
                    (value[1] - value[0]) + value[0]
        return lambda x: rand(_RAND_SCOPE[scope](x)).to(x.device)
