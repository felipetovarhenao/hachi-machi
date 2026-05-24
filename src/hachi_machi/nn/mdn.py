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
        self.out_features = out_features
        self.tril_size = out_features * (out_features + 1) // 2
        self.proj = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )
        self.net = nn.ModuleList([
            self.block(num_features,
                       out_features,
                       dropout,
                       slope) for _ in range(k)
        ])

    def block(self,
              in_features,
              out_features,
              dropout,
              slope):
        out_size = 1 + out_features + self.tril_size
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
        y = torch.stack([layer(x) for layer in self.net], dim=-1)
        pi = F.softmax(y[..., 0, :], dim=-1)
        mu = y[..., 1:self.out_features + 1, :]
        tril_vals = y[..., self.out_features + 1:, :]
        *batch, _, K = mu.shape
        D = self.out_features
        L = torch.zeros(*batch, K, D, D, device=x.device, dtype=x.dtype)
        rows, cols = torch.tril_indices(D, D)
        L[..., rows, cols] = tril_vals.permute(*range(len(batch)), -1, -2)
        diag = torch.arange(D, device=x.device)
        L[..., diag, diag] = F.softplus(L[..., diag, diag]) + 1e-6
        mu = mu.permute(*range(len(batch)), -1, -2)
        return pi, mu, L
