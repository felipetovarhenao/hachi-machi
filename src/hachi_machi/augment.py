import ast
import re
import abc
import torch
import inspect
from .features import FeatureMap

_DISTS = {
    "normal": torch.randn,
    "uniform": torch.rand,
}

_SCOPES = {
    "global": lambda fn, x: fn(1),
    "time": lambda fn, x: fn(x.shape[-2], 1),
    "feature": lambda fn, x: fn(1, x.shape[-1]),
    "both": lambda fn, x: fn(*x.shape[-2:]),
}


class Operation(abc.ABC):

    def __init__(self, *dims: int, p: int | float = 0.5):
        self.p = max(0, min(p, 1))
        self.dims = dims

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        x[..., self.dims] = self.forward(x[..., self.dims])
        return x


class RandomOperation(Operation):

    def __init__(self, *dims, var: int | float = 0, scope: str = "global", dist: str = "normal"):
        super().__init__(*dims)
        self.var = var
        if dist not in _DISTS:
            raise ValueError(
                f"Invalid dist: {dist!r}. Expected one of {list(_DISTS)}")
        if scope not in _SCOPES:
            raise ValueError(
                f"Invalid scope: {scope!r}. Expected one of {list(_SCOPES)}")
        dist_fn = _DISTS[dist]
        self._rand_fn = lambda x: _SCOPES[scope](dist_fn, x)

    def random(self, x):
        return self._rand_fn(x).to(x.device)


class Shift(RandomOperation):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.random(x) * self.var


class Scale(RandomOperation):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 ** (self.random(x) * self.var)


class Mirror(Operation):

    def forward(self, x):
        return x.mean(-2) * 2 - x


class Rotate2D(RandomOperation):

    def __init__(self, *dims,  **kwargs):
        if len(dims) != 2:
            raise ValueError(
                f"Rotate2D requires exactly 2 dimensions, got {len(dims)}")
        super().__init__(*dims,  **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = (self.random(x) * self.var * torch.pi).squeeze()
        cos_t, sin_t = theta.cos(), theta.sin()
        R = torch.tensor(
            [[cos_t, -sin_t],
             [sin_t,  cos_t]],
            dtype=x.dtype, device=x.device,
        )
        mean = x.mean(dim=-2, keepdim=True)
        return (x - mean) @ R.T + mean


class Permute(Operation):

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
        for cls in [Scale,
                    Mirror,
                    Shift,
                    Permute,
                    Rotate2D,]
    }

    def __init__(self, operations: list[str], feature_map: FeatureMap):
        self.operations = self.parse(operations, feature_map)

    def __len__(self):
        return len(self.operations)

    def get_signature(self, cls: Operation):
        options = {}
        bases = [cls]
        while True:
            children = []
            stop = True
            for cls in bases:
                options = {
                    **inspect.signature(cls).parameters,
                    **options
                }
                stop = False
                children.extend(cls.__bases__)

            bases = children
            if stop:
                break
        return options

    def from_str(self, s: str) -> tuple[str, list, dict]:
        s = re.sub(
            r'\b(t|time|global|feature|both|normal|uniform)\b', r'"\g<1>"', s)
        tree = ast.parse(s, mode='eval')
        call = tree.body
        assert isinstance(call, ast.Call)

        name: str = call.func.id
        args: list[int] = [ast.literal_eval(a) for a in call.args]
        kwargs: dict = {kw.arg: ast.literal_eval(
            kw.value) for kw in call.keywords}

        return name.lower(), args, kwargs

    def parse(self, cmds: list[str], feature_map: FeatureMap):
        dim_offset = int(feature_map.temporal())
        ops: list[Operation] = []
        for cmd in cmds:
            name, args, kwargs = self.from_str(cmd)
            dims = []
            if len(args) == 0:
                raise KeyError(
                    f"{name}: You must provide at least one dimension")
            else:
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
                            f"Outside of range dimension for '{name}' operation at index {i}: {dim}. Must be 0 <= dim < {len(feature_map) - dim_offset}")
                    dims.append(dim + dim_offset)
            if name not in self.OPERATIONS:
                raise NameError(f"Invalid operation name: '{name}'")
            op_cls = self.OPERATIONS[name]
            op_params = self.get_signature(op_cls)
            op_keys = op_params.keys()
            for k, v in kwargs.items():
                if k not in op_keys:
                    raise KeyError(
                        f"Invalid keyword argument in '{name}(...)': {k}")
                arg_type = op_params[k].annotation
                if not isinstance(v, arg_type):
                    raise ValueError(
                        f"Invalid value type for '{k}' argument in '{name}(...)': {type(v)}. Expected: {arg_type}")
            op = op_cls(*dims, **kwargs)
            ops.append(op)
        return ops

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        n = len(self)
        i = torch.randint(1, n + 1, size=(1,)).item()
        order = torch.randperm(n=n)
        with torch.no_grad():
            for i in order:
                fn = self.operations[i]
                y = fn(y)
        return y
