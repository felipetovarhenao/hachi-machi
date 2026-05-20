import torch
from torch.utils.data import Dataset, DataLoader
from .ops import DataOperator
from .console import Console


EventLoader = DataLoader


class EventDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 input_dims: list[int],
                 output_dims: list[int],
                 context_length: int = 16,
                 augmenter: DataOperator | None = None):
        with torch.no_grad():
            self.mean = data.mean(0)
            context_length = self._clamp_context(context_length, len(data))
            self.dims = data.size(-1)
            data = data.unfold(dimension=0,
                               size=context_length + 1,
                               step=1)
            self.size = len(data)
            data = data.transpose(2, 1)
            data = data[torch.randperm(n=self.size)]

        self.data = data

        self._in_dims = input_dims
        self._out_dims = output_dims

        self.input_size = len(self._in_dims)
        self.output_size = len(self._out_dims)

        self.augmenter = augmenter

    def _clamp_context(self, context_length: int, data_size: int):
        y = max(2, min(context_length, data_size // 2))
        if y != context_length:
            Console.warning(
                f'Adjusting context length due to insufficient data samples: {context_length} -> {y}')
        return y

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        data = self.data
        item = data[index]
        if self.augmenter is not None:
            item = self.augmenter(item)
        x, y = item[..., :-1, self._in_dims], item[..., 1:, self._out_dims]
        return x, y
