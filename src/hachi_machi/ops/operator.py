import torch
import re
import inspect
import ast
from ..features import FeatureMap
from typing_extensions import Self
from .binary import (Add, Sub, Div, Mul,)
from .random import (AddRand, SubRand, DivRand, MulRand,)
from .unary import (CumSum, Diff, Abs, Rev,)
from .bases import Operation


class DataOperator:

    OPS = {cls.name(): cls for cls in [Add, Sub, Div, Mul,
                                       AddRand, SubRand, DivRand, MulRand,
                                       CumSum, Diff, Abs, Rev,]}

    def __init__(self, ops: list[Operation]):
        self.ops = ops

    def __call__(self, x: torch.Tensor):
        y = x.clone()
        with torch.no_grad():
            for op in self.ops:
                op.fit(x)
                y = op(y)
        return y

    def __iter__(self):
        return iter(self.ops)

    @classmethod
    def _parse_literal(cls, literal: str) -> tuple[str, dict]:
        literal = cls._replace_keywords(literal)
        call = ast.parse(literal, mode='eval').body
        if not isinstance(call, ast.Call):
            raise SyntaxError(f"Invalid operation syntax: {literal!r}")
        name: str = call.func.id.lower()
        if name not in cls.OPS:
            raise NameError(f"Invalid operation name: {name!r}")
        kwargs = {}
        for kw in call.keywords:
            try:
                val = ast.literal_eval(kw.value)
            except:
                raise SyntaxError(
                    f"Invalid value for {kw.arg!r} in {name!r}: {literal!r}")
            kwargs[kw.arg] = val

        return name, kwargs

    @classmethod
    def _replace_keywords(cls, literal: str):
        literal = literal.lower()
        literal = re.sub(
            r"\b(t|time|feature|element|mean|std|uniform|normal)\b", r"'\g<1>'", literal)
        literal = literal.replace("true", "True").replace(
            "false", "False").replace("none", "None")
        return literal

    @classmethod
    def _format_dims(cls, dims: str | int | list[int | str], feature_map: FeatureMap):
        if isinstance(dims, int):
            dims = [dims]
        else:
            dims = list(dims)
        temporal = feature_map.temporal()
        dim_offset = int(temporal)
        dims_post = []
        for i, dim in enumerate(dims):
            if dim == 't' and not temporal:
                raise ValueError(
                    "Time dimension 't' cannot be used in atemporal datasets")
            if dim == 't':
                dim = -1
            if not isinstance(dim, int):
                raise TypeError(
                    f"Invalid dimension type at index {i}': {dim}. Expected int")
            if not 0 <= (dim + dim_offset) < len(feature_map):
                raise ValueError(
                    f"Outside of range dimension at index {i}: {dim}. "
                    f"Must be 0 <= dim < {len(feature_map) - dim_offset}")
            dims_post.append(dim + dim_offset)
        return dims_post

    @classmethod
    def from_callbacks(cls, callbacks: list[str], feature_map: FeatureMap) -> Self:
        ops = []
        for cb in callbacks:
            name, kwargs = cls._parse_literal(cb)
            if 'dims' in kwargs:
                kwargs['dims'] = cls._format_dims(kwargs['dims'], feature_map)
            op_class = cls.OPS[name]
            try:
                op = op_class(**kwargs)
            except Exception as e:
                msg = e.args[0]
                msg = re.sub(r"[A-Za-z]+\.__init__\(\)", f"{name}:", msg)
                raise AttributeError(msg)
            ops.append(op)
        return DataOperator(ops)
