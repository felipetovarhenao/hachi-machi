import ast
import re
import abc
import torch
import inspect
from .features import FeatureMap, FeatureType


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


def _make_value_fn(reduce_fn):
    return {
        "global": lambda x: reduce_fn(x, (-2, -1), keepdim=True),
        "time": lambda x: reduce_fn(x, -2, keepdim=True),
        "feature": lambda x: reduce_fn(x, -1, keepdim=True),
    }


_VALUES = {
    "mean": _make_value_fn(torch.Tensor.mean),
    "std":  _make_value_fn(torch.Tensor.std),
    "min":  _make_value_fn(lambda x, dim, keepdim: x.amin(dim, keepdim=keepdim)),
    "max":  _make_value_fn(lambda x, dim, keepdim: x.amax(dim, keepdim=keepdim)),
}


class Operation(abc.ABC):

    TYPE = FeatureType.CONTINUOUS

    DOCS = {
        '*dims': 'Feature dimensions to apply operation to. If none are provided, all feature dimensions are used.',
        'p': 'Probability for operation to be applied to data sequence'
    }

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

    @classmethod
    def type(self):
        return self.TYPE

    @classmethod
    def docs(cls):
        docs = {}
        for base in reversed(cls.__bases__):
            if not hasattr(base, 'docs'):
                continue
            docs = {
                **docs,
                **base.docs()
            }

        return {**docs, **cls.DOCS}


class DeterministicOperation(Operation):
    """
    Base for operations that apply a constant or data-derived scalar.
    """

    DOCS = {
        'value': ("Numeric constant, or one of the following keywords for referencing data-derived properties."
                  "\n\t- `mean`\n\t- `std`\n\t- `min`\n\t- `max`"),
        'scope': ("Reduction axis for data-derived values. Ignored when `value` is a constant:\n"
                  "\n\t- `global`: data-derived value is based on all dimensions and time steps."
                  "\n\t- `time`: data-derived value is computed along the time-step dimension."
                  "\n\t- `feature`: data-derived value is computed for each step along the feature dimension.")
    }

    def __init__(self, *dims, value: int | float | str = 0, scope: str = "global", **kwargs):
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
    """
    Performs addition on the sequence `dims`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._value_fn(x)


class Sub(DeterministicOperation):
    """
    Performs subtraction on the sequence `dims`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self._value_fn(x)


class Mul(DeterministicOperation):
    """
    Performs multiplication on the sequence `dims`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._value_fn(x)


class Div(DeterministicOperation):
    """
    Performs division on the sequence `dims`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._value_fn(x)


class RandAdd(Operation):
    """Performs addition with value(s) sampled from a specified random distribution.
    """

    DOCS = {
        'a': "Distribution parameter. Lower bound for `scope=uniform`, mean for `scope=normal`.",
        'b': "Distribution parameter.\nUpper bound `scope=uniform`, standard deviation for `scope=normal`.",
        'scope': ("Shape of the random tensor"
                  "\n\t- `global`: a single random value is applied globally"
                  "\n\t- `time`: random values are applied, one for each time step, but constant for all feature `dims`"
                  "\n\t- `feature`: random values are applied, one for each feature, but constant for all steps"
                  "\n\t- `both`: random values are applied, one for each feature and time steps"),
        'dist': ("Type of random distribution to sample from."
                 "\n\t- `uniform`: Even distribution."
                 "\n\t- `normal`: Gaussian distribution"),
    }

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
    """Performs multiplication with value(s) sampled from a specified random distribution.
    """

    DOCS = {
        'space': ('Determines how random number is interpreted.'
                  "\n\t- `linear`: `x * r`"
                  "\n\t- `log`: `x * 2 ** r`")
    }

    def __init__(self, *dims, space: str = "linear", **kwargs):
        if space not in ("linear", "log"):
            raise ValueError(
                f"Invalid space: {space!r}. Expected 'linear' or 'log'")
        super().__init__(*dims, **kwargs)
        self.space = space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.random(x)
        return x * (2 ** r if self.space == "log" else r)


class CumSum(Operation):
    """
    Replaces each time step with the cumulative sum along the `time` axis.
    """

    DOCS = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(x, dim=-2)


class Diff(Operation):
    """
    Replaces each time step with the _n_-th discrete difference along the time axis.

    To preserve the original sequence length a constant `prepend_value` is
    inserted before differencing, so the output shape is always identical to
    the input shape.
    """

    DOCS = {
        'prepend_value': ('Scalar prepended to the sequence before differencing to keep '
                          'the time dimension length unchanged.'),
    }

    def __init__(self, *dims, prepend_value: float = 0.0, **kwargs):
        super().__init__(*dims, **kwargs)
        self.prepend_value = prepend_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prepend = torch.full(
            (*x.shape[:-2], 1, x.shape[-1]),
            self.prepend_value,
            dtype=x.dtype,
            device=x.device,
        )
        return torch.diff(x, n=self.n, dim=-2, prepend=prepend)


class Clip(DeterministicOperation):
    """
    Clips values to the range `[min, max]` along the specified dims.

    `value` sets the lower bound and `value_max` sets the upper bound.
    Either may be omitted (`none`) to clip only one side.
    """

    DOCS = {
        'value': ("Lower bound for clipping. Accepts a numeric constant or a data-derived "
                  "keyword (`mean`, `std`, `min`, `max`). Pass `none` to skip."),
        'value_max': ("Upper bound for clipping. Accepts a numeric constant or a data-derived "
                      "keyword (`mean`, `std`, `min`, `max`). Pass `none` to skip."),
    }

    def __init__(self, *dims, value: int | float | str | None = None,
                 value_max: int | float | str | None = None, **kwargs):
        # We don't want DeterministicOperation's __init__ to process value_max,
        # so we call Operation.__init__ directly and handle both bounds manually.
        Operation.__init__(self, *dims, **kwargs)
        self._min_fn = self._resolve(value, kwargs.get('scope', 'global'))
        self._max_fn = self._resolve(value_max, kwargs.get('scope', 'global'))

    @staticmethod
    def _resolve(value, scope):
        if value is None:
            return None
        if isinstance(value, str):
            if value not in _VALUES:
                raise ValueError(
                    f"Invalid value: {value!r}. Expected one of {list(_VALUES)} or a numeric constant")
            if scope not in _VALUES[value]:
                raise ValueError(
                    f"Invalid scope: {scope!r}. Expected one of {list(_VALUES[value])}")
            return _VALUES[value][scope]
        return lambda _: float(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = self._min_fn(x) if self._min_fn is not None else None
        hi = self._max_fn(x) if self._max_fn is not None else None
        return torch.clamp(x, min=lo, max=hi)


class Quantize(Operation):
    """
    Rounds values to the nearest multiple of `step`.
    """

    DOCS = {
        'step': 'Quantization resolution. Values are rounded to the nearest multiple of this.',
    }

    def __init__(self, *dims, step: int | float = 1.0, **kwargs):
        super().__init__(*dims, **kwargs)
        if step <= 0:
            raise ValueError(f"step must be > 0, got {step}")
        self.step = float(step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x / self.step) * self.step


class Rot2(Operation):
    """
    Applies a 2D rotation to pairs of feature dimensions.

    Each pair of dims is treated as (x, y) coordinates and rotated by `angle` radians.
    If a constant numeric angle is given, the same rotation is applied to every step.
    If a data-derived keyword is used, the angle is computed from the data via `scope`.
    """

    DOCS = {
        'angle': ("Rotation angle in radians. Accepts a numeric constant or a data-derived "
                  "keyword (`mean`, `std`, `min`, `max`)."),
        'scope': ("Reduction axis when `angle` is data-derived:\n"
                  "\n\t- `global`: single angle derived from all dims and steps"
                  "\n\t- `time`: one angle per time step"
                  "\n\t- `feature`: one angle per feature"),
    }

    def __init__(self,
                 *dims,
                 angle: int | float | str = 0.0,
                 scope: str = "global",
                 **kwargs):
        if len(dims) % 2 != 0:
            raise ValueError(
                f"Rot2 requires an even number of dims (paired as x, y), got {len(dims)}")
        super().__init__(*dims, **kwargs)
        if isinstance(angle, str):
            if angle not in _VALUES:
                raise ValueError(
                    f"Invalid angle: {angle!r}. Expected one of {list(_VALUES)} or a numeric constant")
            if scope not in _VALUES[angle]:
                raise ValueError(
                    f"Invalid scope: {scope!r}. Expected one of {list(_VALUES[angle])}")
            self._angle_fn = _VALUES[angle][scope]
        else:
            self._angle_fn = lambda _: float(angle)

    def angle(self, pair: torch.Tensor) -> torch.Tensor:
        return self._angle_fn(pair)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        for i in range(0, x.shape[-1], 2):
            pair = x[..., i:i+2]
            a = self.angle(pair)
            cos_a, sin_a = torch.cos(a), torch.sin(a)
            out[..., i] = pair[..., 0] * cos_a - pair[..., 1] * sin_a
            out[..., i+1] = pair[..., 0] * sin_a + pair[..., 1] * cos_a
        return out


class RandRot2(Rot2):
    """
    Applies a 2D rotation with angle(s) sampled from a specified random distribution.
    """

    DOCS = {
        # 'a': "Distribution parameter. Lower bound for `dist=uniform`, mean for `dist=normal`.",
        # 'b': "Distribution parameter. Upper bound for `dist=uniform`, standard deviation for `dist=normal`.",
        **RandMul.DOCS,
        # 'scope': ("Shape of the sampled angle tensor:\n"
        #           "\n\t- `global`: a single angle applied to all pairs and steps"
        #           "\n\t- `time`: one angle per time step, constant across feature pairs"
        #           "\n\t- `both`: one angle per time step per feature pair"),
        # 'dist': ("Type of random distribution to sample from."
        #          "\n\t- `uniform`: even distribution."
        #          "\n\t- `normal`: Gaussian distribution"),
    }

    def __init__(self,
                 *dims,
                 a: int | float = 0,
                 b: int | float = 1,
                 scope: str = "global",
                 dist: str = "uniform",
                 **kwargs):
        if dist not in _DISTS:
            raise ValueError(
                f"Invalid dist: {dist!r}. Expected one of {list(_DISTS)}")
        if scope not in ("global", "time", "both"):
            raise ValueError(
                f"Invalid scope: {scope!r}. Expected one of ['global', 'time', 'both']")
        if dist == "uniform" and a >= b:
            raise ValueError(
                f"For dist='uniform', a must be less than b, got a={a}, b={b}")
        # Pass a dummy angle=0 to satisfy Rot2.__init__; angle() is overridden anyway
        super().__init__(*dims, angle=0.0, **kwargs)
        self.a = a
        self.b = b
        self.scope = scope
        self.dist = dist

    def angle(self, pair: torch.Tensor) -> torch.Tensor:
        # pair shape: (..., T, 2) — scope 'feature' is excluded since a pair is
        # already a single logical unit; 'both' gives one angle per (step, pair)
        shape = {"global": (1, 1), "time": (
            pair.shape[-2], 1), "both": pair.shape[-2:]}[self.scope]
        return _DISTS[self.dist](shape, self.a, self.b).to(pair.device)


class DataAugmenter:

    OPERATIONS = {
        cls.__name__.lower(): cls
        for cls in [Add,
                    Sub,
                    Mul,
                    Div,
                    RandAdd,
                    RandMul,
                    CumSum,
                    Diff,
                    Clip,
                    Quantize,
                    Rot2,
                    RandRot2]
    }

    def __init__(self, operations: list[str], feature_map: FeatureMap):
        self.operations = self.parse(operations, feature_map)

    def __len__(self):
        return len(self.operations)

    @staticmethod
    def get_signature(cls):
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
        raw = s
        s = re.sub(r"\bnone\b", 'None', s)
        s = re.sub(
            r'\b(t|time|global|feature|both|normal|uniform|linear|log|mean|std|min|max)\b',
            r'"\g<1>"', s)
        tree = ast.parse(s, mode='eval')
        call = tree.body
        assert isinstance(call, ast.Call)

        try:
            name: str = call.func.id
            args: list = [ast.literal_eval(a) for a in call.args]
            kwargs: dict = {kw.arg: ast.literal_eval(
                kw.value) for kw in call.keywords}
        except:
            raise SyntaxError(f"Invalid operation syntax: {raw!r}")

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
                dims.extend(feature_map.dims.tolist())
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

            op_cls: Operation = self.OPERATIONS[name]

            if not torch.all(op_cls.type().value == feature_map.types[dims]):
                raise TypeError(
                    f"{name!r} operation can only be applied to dims of type {op_cls.type().name.lower()!r}")
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
