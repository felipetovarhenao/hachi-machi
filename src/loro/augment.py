import torch
from abc import ABC


class Augmentator(ABC):

    def __init__(self):
        super().__init__()
        self.augmentators = []
        for attr in dir(self):
            if attr.startswith('use_'):
                self.augmentators.append(getattr(self, attr))
        if len(self.augmentators) == 0:
            raise RuntimeError(
                "You must define at least one augmentator method, following the naming convention `use_*`")


class MidiAugmentator(Augmentator):

    def __init__(self):
        super().__init__()
        self._feature_to_dim_map = {
            k: i for i, k in enumerate(
                [
                    'ioi',
                    'ioi_voice',
                    'voice',
                    'pitch',
                    'velocity',
                    # 'duration',
                ]
            )
        }

    def _dims(self, *labels) -> list[int]:
        return [self._feature_to_dim_map[key] for key in labels if key in self._feature_to_dim_map]

    def __len__(self):
        return len(self.augmentators)

    def __getitem__(self, key):
        return self.augmentators[key]

    def use_timestretch(self, x: torch.Tensor) -> torch.Tensor:
        dims = self._dims('ioi', 'ioi_voice', 'duration')
        s = (torch.rand(1).item() * 2 - 1) * 0.25
        x[..., dims] *= 2 ** s
        return x

    # def use_inversion(self, x: torch.Tensor) -> torch.Tensor:
    #     dim = self._dims('pitch', 'velocity')
    #     x[..., dim] = self.mean[dim] * 2 - x[..., dim]
    #     return x

    def use_pitchshift(self, x: torch.Tensor) -> torch.Tensor:
        dim = self._dims('pitch')
        s = torch.rand(1).item() * 2 - 1
        x[..., dim] += s * 700
        return x

    def use_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        max_vel = 127
        dim = self._dims('velocity')
        x[..., dim] = x[..., dim] / max_vel
        s = torch.randn(1,).item() * 0.5
        theta = (x[..., dim] + s).clip(0, 1) * torch.pi
        x[..., dim] = 0.5 - 0.5 * torch.cos(theta)
        x[..., dim] *= max_vel
        return x

    def use_voice_swap(self, x: torch.Tensor) -> torch.Tensor:
        dim = self._dims('voice')
        classes = x[..., dim].unique()
        perm = classes[torch.randperm(len(classes))]
        result = x.clone()
        for c, p in zip(classes, perm):
            mask = x[..., dim] == c
            result[..., dim] = torch.where(mask, p, result[..., dim])
        return result
