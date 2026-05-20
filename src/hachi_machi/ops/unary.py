import torch
from .bases import Operation


class Round(Operation):
    """Rounds data values to the nearest integer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.round()


class Diff(Operation):
    """Computes the difference between consecutive feature values along the `time` axis."""

    def forward(self, x):
        return x.diff(dim=-2, prepend=x[:1])


class CumSum(Operation):
    """Computes the cumulative sum of consecutive feature values along the `time` axis.
    """

    def forward(self, x):
        return x.cumsum(dim=-2)


class Rev(Operation):
    """Reverses feature values along the `time` axis."""

    def forward(self, x):
        idx = torch.arange(x.size(0), 0, -1) - 1
        return x[idx]


class Abs(Operation):
    """Computes the absolute value of input features."""

    def forward(self, x):
        return x.abs()
