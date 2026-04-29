
import torch
import torch.nn as nn
from .rnn import RecurrentMDN
from . import transforms as T


class MultiplayerAgent(nn.Module):

    def __init__(self,
                 model: RecurrentMDN,
                 input_layer: T.Transform,
                 output_layer: T.Transform,
                 player_voices: tuple[int] = (0, ),
                 device: str = 'mps',
                 voice_dim: int = 1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = device
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.voice_dim = voice_dim
        self.input_size = self.input_layer.input_size
        self.player_voices = player_voices
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self):
        self.hidden_state = None

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        return self.model.forward(x=x)

    def step(self, x: torch.Tensor) -> None | torch.Tensor:
        x: torch.Tensor = self.input_layer(x)
        y, self.hidden_state = self.model.step(x=x,
                                               hidden=self.hidden_state)
        y: torch.Tensor = self.output_layer(y, True)
        
        if y[..., self.voice_dim].item() in self.player_voices:
            return
        return y.clip(0)
