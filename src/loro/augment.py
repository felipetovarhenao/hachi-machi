import torch
from typing import Callable
from abc import ABC


class Augmentator(ABC):

    def __init__(self, transforms: None | tuple | list = None):
        super().__init__()
        self.augmentators = []
        for attr in dir(self):
            if attr.startswith('use_'):
                name = attr[4:].replace('_', '-')
                if transforms is not None and name not in transforms:
                    continue
                self.augmentators.append(getattr(self, attr))
        if len(self.augmentators) == 0:
            raise RuntimeError(
                "You must define at least one augmentator method, following the naming convention `use_*`")

    def __len__(self):
        return len(self.augmentators)

    def __getitem__(self, key):
        return self.augmentators[key]

    @classmethod
    def options(cls):
        return [k[4:].replace('_', '-') for k in dir(cls) if k.startswith('use_')]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        n = len(self)
        i = torch.randint(1, n + 1, size=(1,)).item()
        order = torch.randperm(n=n)
        order = order[:i]
        for i in order:
            fn = self.augmentators[i]
            y = fn(y)
        return y


class MidiAugmentator(Augmentator):

    def __init__(self, num_voices: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_voices = num_voices
        self._feature_to_dim_map = {
            k: i for i, k in enumerate(
                [
                    'ioi',
                    'ioi_voice',
                    'voice',
                    'pitch',
                    'velocity',
                    'duration',
                ]
            )
        }

    def get(self, *labels) -> list[int]:
        return [self._feature_to_dim_map[key] for key in labels if key in self._feature_to_dim_map]

    def use_time_stretch(self, x: torch.Tensor) -> torch.Tensor:
        dims = self.get('ioi', 'ioi_voice', 'duration')
        s = (torch.rand(1).item() * 2 - 1) * 0.66
        x[..., dims] *= 2 ** s
        return x

    def use_pitch_inversion(self, x: torch.Tensor) -> torch.Tensor:
        p_dim = self.get('pitch')
        v_dim = self.get('voice')
        for v in range(self.num_voices):
            mask = x[..., v_dim] == v
            pitch = x[..., p_dim]
            inverted = pitch[mask].mean(0) * 2 - pitch
            x[..., p_dim] = torch.where(mask, inverted, pitch)
        return x

    def use_pitch_shift(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.get('pitch')
        s = torch.rand(1).item() * 2 - 1
        x[..., dim] += s * 700
        return x

    def use_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        max_vel = 127
        dim = self.get('velocity')
        x[..., dim] = x[..., dim] / max_vel
        s = torch.randn(1,).item() * 0.5
        theta = (x[..., dim] + s).clip(0, 1) * torch.pi
        x[..., dim] = 0.5 - 0.5 * torch.cos(theta)
        x[..., dim] *= max_vel
        return x

    def use_voice_swap(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.get('voice')
        voices = x[..., dim].unique()
        swap = voices[torch.randperm(self.num_voices)]
        result = x.clone()
        for v, s in zip(voices, swap):
            mask = x[..., dim] == v
            result[..., dim] = torch.where(mask, s, result[..., dim])
        return result

    def __handle_onsets(self, x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        y = x.clone()
        voice_dim = self.get('voice')
        voice_ioi_dim = self.get('ioi_voice')
        ioi_dim = self.get('ioi')
        y[:, ioi_dim] = torch.cumsum(y[:, ioi_dim], dim=0)
        y = func(y)
        order = torch.sort(y[:, ioi_dim], dim=0, stable=True).indices.flatten()
        y = y[order]

        for v in range(self.num_voices):
            voice_mask = (y[:, voice_dim] == v).flatten()
            voice_onsets = y[voice_mask, ioi_dim]
            voice_iois = torch.diff(voice_onsets, dim=0)
            y[voice_mask][1:, voice_ioi_dim] = voice_iois.unsqueeze(-1)
        y[1:, ..., ioi_dim] = torch.diff(y[..., ioi_dim], dim=-2)
        return y

    def use_chord_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        return self.__handle_onsets(x, lambda y: y[torch.randperm(n=len(y))])

    def use_time_noise(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.get('ioi')

        def func(y: torch.Tensor) -> torch.Tensor:
            y[..., dim] += torch.randn_like(y[..., dim]) * 5
            return y.clip(0)
        return self.__handle_onsets(x, func)

    def use_time_rubato(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.get('ioi')
        st, end = 2 ** (torch.randn(2) * 0.125)
        warp = torch.linspace(start=st,
                              end=end,
                              steps=len(x)).unsqueeze(-1).to(x.device)

        x[..., dim] *= warp
        return self.__handle_onsets(x, lambda y: y)
