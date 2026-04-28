
import torch
import torch.nn as nn
from .scaler import FeatureScaler
from .rnn import RecurrentMDN


class MultiplayerAgent(nn.Module):

    def __init__(self,
                 model: RecurrentMDN,
                 x_scaler:  FeatureScaler,
                 y_scaler:  FeatureScaler,
                 player_voices: list[int] = [0],
                 device: str = 'mps',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = device
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.input_size = self.model.input_size
        self.player_voices = player_voices
        self.temp = 1.0
        self.next_event: torch.Tensor | None = None
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self.alpha = 1.25
        self.weights: torch.Tensor | None = None
        self.set_weights(torch.ones(self.model.input_size))

    def set_weights(self, x: torch.Tensor) -> None:
        self.weights: torch.Tensor = x.to(self.device)
        self.weights /= self.weights.sum()

    def set_temp(self, x: float) -> None:
        self.temp = x

    def clear_hidden(self):
        self.hidden_state = None

    def set_alpha(self, x: float) -> None:
        self.alpha = max(1, min(2, x))

    def get_confidence(self, x: torch.Tensor) -> float:
        y: torch.Tensor = (x - self.next_event[..., :-1]) ** 2 * self.weights
        y = y.sum().sqrt()
        return torch.exp(-self.alpha * y).item()

    def forward(self, x: torch.Tensor) -> None | torch.Tensor:
        x: torch.Tensor = self.x_scaler(x)
        if self.hidden_state is not None:
            conf = self.get_confidence(x)
            (hn, cn) = self.hidden_state
            hn = hn.clone()
            cn = cn.clone()
            hn[-1] *= conf
            cn[-1] *= conf
            self.hidden_state = (hn, cn)
        self.next_event, self.hidden_state = self.model.step(x=x,
                                                             hidden=self.hidden_state,
                                                             temp=self.temp)
        y: torch.Tensor = self.y_scaler(
            self.next_event.clone(), inverse=True)
        y = y.clip(0).squeeze().round().int()
        if y[2] in self.player_voices:
            return
        return y
