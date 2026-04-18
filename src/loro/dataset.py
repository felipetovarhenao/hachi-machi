import torch
from torch.utils.data import Dataset, DataLoader
from .utils import echo
from .augment import Augmentator


EventLoader = DataLoader


class EventDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 context_length: int = 16,
                 split: float = 0.6,
                 augmentator: Augmentator | None = None):
        self.mean = data.mean(0)
        context_length = self._clamp_context(context_length, len(data))
        self.dims = data.shape[-1]
        data = data.unfold(dimension=0,
                           size=context_length + 1,
                           step=1)
        self.size = len(data)
        split = int(split * self.size)
        data = data.transpose(2, 1)
        data = data[torch.randperm(n=self.size)]
        self.train_set, self.eval_set = data[:split], data[split:]
        self.training = True
        self._in_dims = list(range(self.dims))
        self._out_dims = list(range(self.dims))
        self.augmentator = augmentator

    def _clamp_context(self, context_length: int, data_size: int):
        y = max(2, min(context_length, data_size // 2))
        if y != context_length:
            echo(text=f'Adjusting context length due to insufficient data samples: {context_length} -> {y}',
                 type='info')
        return y

    def get_split(self) -> torch.Tensor:
        return self.train_set if self.training else self.eval_set

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self) -> None:
        self.training = False

    def __len__(self) -> int:
        data = self.get_split()
        return len(data)

    def augment(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        if not self.training or not self.augmentator:
            return y
        n = len(self.augmentator)
        i = torch.randint(0, n + 1, size=(1,)).item()
        order = torch.randperm(n=n)
        order = order[:i]
        for i in order:
            fn = self.augmentator[i]
            y = fn(y)
        return y

    def __getitem__(self, index) -> torch.Tensor:
        data = self.get_split()
        item = self.augment(data[index])
        x, y = item[..., :-1, self._in_dims], item[...,
                                                   1:, self._out_dims]
        return x, y
