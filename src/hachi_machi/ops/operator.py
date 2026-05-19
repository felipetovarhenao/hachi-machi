import torch
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
