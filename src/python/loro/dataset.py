import torch
from torch.utils.data import Dataset, DataLoader
from loro.midi import MidiParser


EventLoader = DataLoader


class EventDataset(Dataset):

    def __init__(self,
                 data: torch.Tensor,
                 context_length: int = 16,
                 split: float = 0.6):
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
        self.dim_map = {
            'voice': 0,
            'pitch': 1,
            'velocity': 2,
            'ioi': 3,
            'ioi_voice': 4,
            'duration': 5
        }
        self._augmentators = [
            self._augment_time,
            self._augment_pitch,
            self._augment_velocity,
            self._augment_voice,
            # self._augment_inversion,
        ]

    def _dims(self, *labels) -> list[int]:
        return [self.dim_map[key] for key in labels]

    def _clamp_context(self, context_length: int, data_size: int):
        y = max(2, min(context_length, data_size // 2))
        if y != context_length:
            print(
                f'Adjusting context length due to insufficient data samples: {context_length} -> {y}')
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

    def _augment_time(self, x: torch.Tensor) -> torch.Tensor:
        dims = self._dims('ioi', 'ioi_voice', 'duration')
        s = (torch.rand(1).item() * 2 - 1) * 0.25
        x[..., dims] *= 2 ** s
        return x

    def _augment_inversion(self, x: torch.Tensor) -> torch.Tensor:
        dim = self._dims('pitch', 'velocity')
        x[..., dim] = self.mean[dim] * 2 - x[..., dim]
        return x

    def _augment_pitch(self, x: torch.Tensor) -> torch.Tensor:
        dim = self._dims('pitch')
        s = torch.rand(1).item() * 2 - 1
        x[..., dim] += s * 700
        return x

    def _augment_velocity(self, x: torch.Tensor) -> torch.Tensor:
        max_vel = 127
        dim = self._dims('velocity')
        x[..., dim] = x[..., dim] / max_vel
        s = torch.randn(1,).item() * 0.5
        theta = (x[..., dim] + s).clip(0, 1) * torch.pi
        x[..., dim] = 0.5 - 0.5 * torch.cos(theta)
        x[..., dim] *= max_vel
        return x

    def _augment_voice(self, x: torch.Tensor) -> torch.Tensor:
        dim = self._dims('voice')
        classes = x[..., dim].unique()
        perm = classes[torch.randperm(len(classes))]
        result = x.clone()
        for c, p in zip(classes, perm):
            mask = x[..., dim] == c
            result[..., dim] = torch.where(mask, p, result[..., dim])
        return result

    def _apply_data_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        i = torch.randint(0, len(self._augmentators) + 1, size=(1,)).item()
        order = torch.randperm(n=len(self._augmentators))
        order = order[:i]
        for i in order:
            fn = self._augmentators[i]
            y = fn(y)
        return y

    def __getitem__(self, index) -> torch.Tensor:
        data = self.get_split()
        item = data[index]
        if self.training:
            item = self._apply_data_augmentation(item)
        x, y = item[..., :-1, :], item[..., 1:, :]
        return x, y
