import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityNetwork(nn.Module):

    def __init__(self,
                 k: int = 2,
                 num_features: int = 100,
                 out_features: int = 2,
                 dropout: float = 0.25,
                 slope: float = 0.01,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Sequential(
            nn.Linear(in_features=num_features,
                      out_features=num_features,),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=slope,
                         inplace=True)
        )
        self.out_features = out_features
        self.net = nn.ModuleList(modules=[self.block(in_features=num_features,
                                                     out_features=out_features,
                                                     dropout=dropout,
                                                     slope=slope) for _ in range(k)])

    def block(self,
              in_features: int = 2,
              out_features: int = 2,
              dropout: float = 0.25,
              slope: float = 0.01):
        out_size = out_features * 2 + 1
        return nn.Sequential(
            nn.Linear(in_features,
                      in_features * 4),
            nn.Dropout(dropout),
            nn.LeakyReLU(slope, True),
            nn.Linear(in_features * 4,
                      in_features * 2),
            nn.Dropout(dropout),
            nn.LeakyReLU(slope, True),
            nn.Linear(in_features * 2,
                      out_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.proj(x)
        y = torch.stack([layer(y) for layer in self.net], dim=-1)
        pi = F.softmax(y[..., 0, :], dim=-1)
        mu = y[..., 1:self.out_features + 1, :]
        sigma = F.softplus(y[..., self.out_features + 1:, :]) + 1e-8
        return pi, mu, sigma
