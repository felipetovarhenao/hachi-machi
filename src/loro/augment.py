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

    def __init__(self, num_voices: int = 2):
        super().__init__()
        self.num_voices = num_voices
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

    def use_inversion(self, x: torch.Tensor) -> torch.Tensor:
        p_dim = self._dims('pitch')
        v_dim = self._dims('voice')
        for v in range(self.num_voices):
            mask = x[..., v_dim] == v
            pitch = x[..., p_dim]
            mean = (pitch * mask).sum(-1, keepdim=True) / \
                mask.sum(-1, keepdim=True)
            inverted = mean * 2 - pitch
            x[..., p_dim] = torch.where(mask, inverted, pitch)
        return x

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
        voices = x[..., dim].unique()
        swap = voices[torch.randperm(self.num_voices)]
        result = x.clone()
        for v, s in zip(voices, swap):
            mask = x[..., dim] == v
            result[..., dim] = torch.where(mask, s, result[..., dim])
        return result
