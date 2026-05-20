import torch
from .bases import Operation, _STATS, _AXES


class Clip(Operation):

    """Given a `min` and/or `max` value, clips the range of values in input data. Use `none` to prevent either end to remain unclipped."""

    DOCS = {
        'min': "Minimum clipping value.",
        'max': "Maximum clipping value.",
    }

    def __init__(self, *, min: int | float | str | None = 0, max: int | float | None = None, axis: str | None = None,  **kwargs):
        super().__init__(**kwargs)
        self.min = None
        self.max = None
        for value, attr in zip([min, max], ['min', 'max']):
            if isinstance(value, str) and value not in _STATS:
                raise ValueError(
                    f"{self.name()}: Invalid value for {attr}: {value}")
            setattr(self, attr, value)
            if axis not in _AXES:
                raise ValueError(f"{self.name()}: Invalid axis: {axis}")
            self.dim = _AXES[axis]

    def fit(self, x: torch.Tensor):
        for param in ['min', 'max']:
            attr = getattr(self, param)
            if not isinstance(attr, str):
                if attr is not None:
                    setattr(self, param, torch.Tensor(attr).to(x.device))
                continue
            attr = getattr(self, param)
            func = getattr(torch, attr)
            value = func(x, dim=self.dim, keepdim=True)
            if attr == 'std':
                value += 1e-8
            setattr(self, param, torch.Tensor(value).to(x.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=self.min, max=self.max)
