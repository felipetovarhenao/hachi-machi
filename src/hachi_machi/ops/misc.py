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
                    setattr(self, param, torch.tensor([attr], device=x.device))
                continue
            attr = getattr(self, param)
            func = getattr(torch, attr)
            value = func(x, dim=self.dim, keepdim=True)
            if attr == 'std':
                value += 1e-8
            setattr(self, param, torch.Tensor(value).to(x.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=self.min, max=self.max)


class Sort(Operation):
    """Sorts input sequence data by the feature index `by`. If `dims` are provided, `by` must be an index relative to `dims`."""

    def __init__(self, *, by: int = 0, descending: bool = False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(self.dims, slice):
            self.by = by
        elif not 0 <= by < len(self.dims):
            raise IndexError(
                f"Outside of range index in {self.name()}: When 'dims' is not 'none', 'by' must be an index relative to 'dims'.")
        else:
            self.by = self.dims[by]
        self.desc = descending

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(input=x[..., self.by],
                              descending=self.desc,
                              stable=True)
        return x[order]
