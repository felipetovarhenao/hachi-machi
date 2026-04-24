import torch
import torch.nn as nn
from . import MixtureDensityNetwork


class RecurrentMDN(nn.Module):

    def __init__(self,
                 k: int = 2,
                 input_size: int = 2,
                 output_size: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.25,
                 slope: float = 0.01,
                 device: str = 'mps',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=hidden_size,
                            batch_first=True).to(device)
        self.proj = MixtureDensityNetwork(k=k,
                                          num_features=hidden_size,
                                          out_features=self.output_size,
                                          dropout=dropout,
                                          slope=slope).to(device)

    def forward(self, x: torch.Tensor, hidden=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, hidden = self.lstm(x, hidden)
        pi, mu, sigma = self.proj(y)
        return pi, mu, sigma, hidden

    def step(self,
             x: torch.Tensor,
             hidden: torch.Tensor | None,
             temp: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        pi, mu, sigma, hidden = self.forward(x, hidden)
        log = torch.log(pi.squeeze())
        k = torch.multinomial(
            torch.exp((log / temp)), num_samples=1).item()
        y = torch.normal(mean=mu[0, 0, :, k],
                         std=sigma[0, 0, :, k]).unsqueeze(0).unsqueeze(0)
        return y, hidden
