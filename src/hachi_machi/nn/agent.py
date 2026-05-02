
import torch
import torch.nn as nn
from .rnn import RecurrentMDN
from . import transforms as T


class MultiplayerAgent(nn.Module):

    input_mask: torch.Tensor

    def __init__(self,
                 model: RecurrentMDN,
                 input_layer: T.Transform,
                 output_layer: T.Transform,
                 input_mask: list[int],
                 players: tuple[int] | None = None,
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
        self.register_buffer(name='input_mask',
                             tensor=torch.tensor(input_mask,
                                                 dtype=torch.int).to(device))
        self.input_size = self.input_layer.input_size
        self.players = players if players is not None else []
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self.hidden_state = None

    def set_players(self, indices: list[int]) -> None:
        indices = list(set(indices))
        self.players = indices

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        return self.model.forward(x=x)

    def step(self, x: torch.Tensor) -> None | torch.Tensor:
        x: torch.Tensor = self.input_layer(x)
        y, self.hidden_state = self.model.step(x=x,
                                               hidden=self.hidden_state)
        y: torch.Tensor = self.output_layer(y, inverse=True)

        if y[..., self.voice_dim].item() in self.players:
            return
        y[..., 0] = y[..., 0].clip(0)
        return y
