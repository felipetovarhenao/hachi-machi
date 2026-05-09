import ast
import re
import abc
import torch
import inspect
from .features import FeatureMap


_DISTS = {
    "uniform": lambda shape, a, b: torch.rand(shape) * (b - a) + a,
    "normal": lambda shape, a, b: torch.randn(shape) * b + a,
}

_SCOPES = {
    "global": lambda _: (1, 1),
    "time": lambda x: (x.shape[-2], 1),
    "feature": lambda x: (1, x.shape[-1]),
    "both": lambda x: x.shape[-2:],
}

_VALUES = {
    "mean": {
        "global": lambda x: x.mean((-2, -1), keepdim=True),
        "time": lambda x: x.mean(-2, keepdim=True),
        "feature": lambda x: x.mean(-1, keepdim=True),
    },
    "std": {
        "global": lambda x: x.std((-2, -1), keepdim=True),
        "time": lambda x: x.std(-2, keepdim=True),
        "feature": lambda x: x.std(-1, keepdim=True),
    },
}


class Operation(abc.ABC):

    def __init__(self, *dims: int, p: float = 1.0):
        self.p = max(0.0, min(float(p), 1.0))
        self.dims = dims

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        x[..., self.dims] = self.forward(x[..., self.dims])
        return x


class DeterministicOperation(Operation):
    """Base for operations that apply a constant or data-derived scalar.

    Args:
        value: Numeric constant or one of 'mean', 'std'.
        scope: Reduction axis for data-derived values — 'global', 'time', or 'feature'.
    """

    def __init__(self, *dims, value: int | float | str, scope: str = "global", **kwargs):
        super().__init__(*dims, **kwargs)
        if isinstance(value, str):
            if value not in _VALUES:
                raise ValueError(
                    f"Invalid value: {value!r}. Expected one of {list(_VALUES)} or a numeric constant")
            if scope not in _VALUES[value]:
                raise ValueError(
                    f"Invalid scope: {scope!r}. Expected one of {list(_VALUES[value])}")
            self._value_fn = _VALUES[value][scope]
        else:
            self._value_fn = lambda _: float(value)


class Add(DeterministicOperation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._value_fn(x)


class Sub(DeterministicOperation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self._value_fn(x)


class Mul(DeterministicOperation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._value_fn(x)


class Div(DeterministicOperation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._value_fn(x)


class RandAdd(Operation):
    """Add a random value sampled from the specified distribution.

    Args:
        a: Distribution parameter — lower bound for 'uniform', mean for 'normal'.
        b: Distribution parameter — upper bound for 'uniform', std for 'normal'.
        scope: Shape of the random tensor — 'global', 'time', 'feature', or 'both'.
        dist: Distribution — 'uniform' or 'normal'.
    """

    def __init__(self, *dims, a: int | float = 0, b: int | float = 1,
                 scope: str = "global", dist: str = "uniform", **kwargs):
        super().__init__(*dims, **kwargs)
        if dist not in _DISTS:
            raise ValueError(
                f"Invalid dist: {dist!r}. Expected one of {list(_DISTS)}")
        if scope not in _SCOPES:
            raise ValueError(
                f"Invalid scope: {scope!r}. Expected one of {list(_SCOPES)}")
        if dist == "uniform" and a >= b:
            raise ValueError(
                f"For dist='uniform', a must be less than b, got a={a}, b={b}")
        self.a = a
        self.b = b
        self.scope = scope
        self.dist = dist

    def random(self, x: torch.Tensor) -> torch.Tensor:
        shape = _SCOPES[self.scope](x)
        return _DISTS[self.dist](shape, self.a, self.b).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.random(x)


class RandMul(RandAdd):
    """Multiply x by a random value sampled from the specified distribution.

    Args:
        a: Distribution parameter — lower bound for 'uniform', mean for 'normal'.
        b: Distribution parameter — upper bound for 'uniform', std for 'normal'.
        scope: Shape of the random tensor — 'global', 'time', 'feature', or 'both'.
        dist: Distribution — 'uniform' or 'normal'.
        space: 'linear' for x * r, 'log' for x * 2**r.
    """

    def __init__(self, *dims, space: str = "linear", **kwargs):
        if space not in ("linear", "log"):
            raise ValueError(
                f"Invalid space: {space!r}. Expected 'linear' or 'log'")
        super().__init__(*dims, **kwargs)
        self.space = space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.random(x)
        return x * (2 ** r if self.space == "log" else r)


class Rotate2D(RandAdd):
    """Rotate two dimensions by a random angle.

    Args:
        a: Lower bound of the rotation angle range (in turns, i.e. 1.0 = 2π).
        b: Upper bound of the rotation angle range (in turns).
    """

    def __init__(self, *dims, **kwargs):
        if len(dims) != 2:
            raise ValueError(
                f"Rotate2D requires exactly 2 dimensions, got {len(dims)}")
        super().__init__(*dims, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = (self.random(x).squeeze() * torch.pi)
        cos_t, sin_t = theta.cos(), theta.sin()
        R = torch.tensor(
            [[cos_t, -sin_t],
             [sin_t,  cos_t]],
            dtype=x.dtype, device=x.device,
        )
        mean = x.mean(dim=-2, keepdim=True)
        return (x - mean) @ R.T + mean


class Permute(Operation):
    """Randomly permute the discrete values within each selected dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = x.clone()
        for dim in range(x.size(-1)):
            channels = result[..., dim].unique()
            swap = channels[torch.randperm(len(channels))]
            for v, s in zip(channels, swap):
                mask = x[..., dim] == v
                result[..., dim] = torch.where(mask, s, result[..., dim])
        return result


class DataAugmentator:

    OPERATIONS = {
        cls.__name__.lower(): cls
        for cls in [Add,
                    Sub,
                    Mul,
                    Div,
                    RandAdd,
                    RandMul,
                    Rotate2D,
                    Permute,]
    }

    def __init__(self, operations: list[str], feature_map: FeatureMap):
        self.operations = self.parse(operations, feature_map)

    def __len__(self):
        return len(self.operations)

    def get_signature(self, cls):
        params = {}
        for klass in cls.__mro__:
            if klass is object:
                continue
            for name, param in inspect.signature(klass.__init__).parameters.items():
                if name in ("self", "args", "kwargs"):
                    continue
                params.setdefault(name, param)
        return params

    def from_str(self, s: str) -> tuple[str, list, dict]:
        s = re.sub(
            r'\b(time|global|feature|both|normal|uniform|linear|log|mean|std)\b',
            r'"\g<1>"', s)
        tree = ast.parse(s, mode='eval')
        call = tree.body
        assert isinstance(call, ast.Call)

        name: str = call.func.id
        args: list = [ast.literal_eval(a) for a in call.args]
        kwargs: dict = {kw.arg: ast.literal_eval(
            kw.value) for kw in call.keywords}

        return name.lower(), args, kwargs

    def parse(self, cmds: list[str], feature_map: FeatureMap):
        dim_offset = int(feature_map.temporal())
        ops: list[Operation] = []
        for cmd in cmds:
            name, args, kwargs = self.from_str(cmd)
            if name not in self.OPERATIONS:
                raise NameError(f"Invalid operation name: '{name}'")
            dims = []
            if len(args) == 0:
                raise KeyError(
                    f"{name}: You must provide at least one dimension")
            for i, dim in enumerate(args):
                if dim == 't' and dim_offset == 0:
                    raise ValueError(
                        "Time dimension 't' cannot be used in non-temporal datasets")
                if dim == 't':
                    dim = -1
                if not isinstance(dim, int):
                    raise ValueError(
                        f"Invalid dimension type at index {i} in '{name}(...)': {dim}. Expected int")
                if not 0 <= (dim + dim_offset) < len(feature_map):
                    raise ValueError(
                        f"Outside of range dimension for '{name}' at index {i}: {dim}. "
                        f"Must be 0 <= dim < {len(feature_map) - dim_offset}")
                dims.append(dim + dim_offset)

            op_cls = self.OPERATIONS[name]
            op_params = self.get_signature(op_cls)
            op_keys = op_params.keys()
            for k, v in kwargs.items():
                if k not in op_keys:
                    raise KeyError(
                        f"Invalid keyword argument in '{name}(...)': {k}")
                ann = op_params[k].annotation
                if ann is not inspect.Parameter.empty and not isinstance(v, ann):
                    raise ValueError(
                        f"Invalid value type for '{k}' in '{name}(...)': "
                        f"{type(v).__name__}. Expected: {ann}")
            ops.append(op_cls(*dims, **kwargs))
        return ops

    def __getitem__(self, key):
        return self.operations[key]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        with torch.no_grad():
            for fn in self.operations:
                y = fn(y)
        return y
