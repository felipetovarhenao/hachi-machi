import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """Multi-layer LSTM with residual skip connections between layers."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            ) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.input_proj = (
            nn.Linear(input_size, hidden_size)
            if input_size != hidden_size else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if hidden is None:
            hidden = [None] * self.num_layers

        new_hidden = []
        out = x
        for i, lstm in enumerate(self.layers):
            residual = self.input_proj(out) if i == 0 else out
            out, h = lstm(out, hidden[i])
            new_hidden.append(h)
            if self.num_layers > 1:
                out = out + residual
            if i < self.num_layers - 1:
                out = self.dropout(out)

        return out, new_hidden
