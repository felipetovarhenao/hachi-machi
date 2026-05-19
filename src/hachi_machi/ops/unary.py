import torch
from .bases import Operation


class Round(Operation):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.round()


class Diff(Operation):

    def __init__(self, dims=None, p=1):
        super().__init__(dims, p)
        self.prepend = None

    def fit(self, x):
        self.prepend = torch.full(
            (*x.shape[:-2], 1, x.shape[-1]),
            0.0,
            dtype=x.dtype,
            device=x.device,
        )

    def forward(self, x):
        return x.diff(dim=-2, prepend=self.prepend)


class CumSum(Operation):

    def forward(self, x):
        return x.cumsum(dim=-2)
