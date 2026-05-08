import ast
import re
import abc
import torch
import inspect
from .features import FeatureMap


class Operation(abc.ABC):

    def __init__(self, *dims: int):
        self.dims = dims

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x[..., self.dims] = self.forward(x[..., self.dims])
        return x

    def random(self, x):
        return torch.randn(1).to(x.device)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class Shift(Operation):

    def __init__(self, *dims: int | str, factor: float | int = 0):
        super().__init__(dims)
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.random(x) * self.factor


class Scale(Operation):

    def __init__(self, *dims: int, factor: float | int = 0):
        super().__init__(dims)
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 ** (self.random(x) * self.factor)


class Mirror(Operation):

    def forward(self, x):
        return x.mean(-2) * 2 - x


class DataAugmentator:

    OPERATIONS = {
        cls.__name__.lower(): cls
        for cls in [Scale,
                    Mirror,
                    Shift]
    }

    def __init__(self, operations: list[str], feature_map: FeatureMap):
        self.operations = self.parse(operations, feature_map)

    def __len__(self):
        return len(self.operations)

    def from_str(self, s: str) -> tuple[str, list, dict]:
        s = re.sub(r'\bt\b', '"t"', s)
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
                            "Time dimension t cannot be used in non-temporal datasets.")

                    if dim == 't':
                        dim = -1

                    if not isinstance(dim, int):
                        raise ValueError(
                            f"{name} operation: Invalid dimension type at index {i}: {dim}.")

                    if not 0 <= (dim + dim_offset) < len(feature_map):
                        raise ValueError(
                            f"Outside of range dimension for '{name}' operation at index {i}: {dim}. Must be 0 <= dim < {len(feature_map) - dim_offset}")
                    dims.append(dim + dim_offset)
            if name not in self.OPERATIONS:
                raise NameError(f'Invalid operation name: {name}')
            op_cls = self.OPERATIONS[name]
            op_params = inspect.signature(op_cls).parameters
            op_keys = op_params.keys()
            for k, v in kwargs.items():
                if k not in op_keys:
                    raise KeyError(
                        f"Invalid keyword argument for {name}: {k}")
                arg_type = op_params[k].annotation
                if not isinstance(v, arg_type):
                    raise ValueError(
                        f'Invalid type for argument: {k}: {type(v)}. Expected: {arg_type}')
            op = op_cls(*dims, **kwargs)
            ops.append(op)
        return ops

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        n = len(self)
        i = torch.randint(1, n + 1, size=(1,)).item()
        order = torch.randperm(n=n)
        order = order[:i]
        with torch.no_grad():
            for i in order:
                fn = self.operations[i]
                y = fn(y)
        return y
