from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
from .mdn import MixtureDensityNetwork


class RecurrentMDN(nn.Module):

    def __init__(self,
                 k: int = 2,
                 input_size: int = 2,
                 output_size: int = 2,
                 num_layers: int = 1,
                 hidden_size: int = 100,
                 dropout: float = 0.25,
                 slope: float = 0.01,
                 device: str = 'mps',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=0 if num_layers == 1 else dropout,
                            batch_first=True).to(device)
        self.proj = MixtureDensityNetwork(k=k,
                                          num_features=hidden_size,
                                          out_features=self.output_size,
                                          dropout=dropout,
                                          slope=slope).to(device)

    def forward(self, x: torch.Tensor, hidden=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, hidden = self.lstm(x, hidden)
        pi, mu, L = self.proj(y)
        return pi, mu, L, hidden

    def step(self,
             x: torch.Tensor,
             hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        pi, mu, L, hidden = self.forward(x, hidden)
        k = torch.multinomial(pi.squeeze(0).squeeze(0), num_samples=1).item()
        dist = MultivariateNormal(loc=mu[0, 0, k], scale_tril=L[0, 0, k])
        y = dist.rsample().unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        return y, hidden
