import torch
import torch.nn.functional as F
import torch.nn as nn


class OneHot(nn.Module):
    def __init__(self, data: torch.Tensor, dim: int, size: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = dim
        self.size = size
        with torch.no_grad():
            data = data[..., dim].clone()
            data = F.one_hot(data.to(torch.long), num_classes=size).to(torch.float)
            self.mean = data.mean(0)
            self.std = data.std(0)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        i = self.i
        s = self.size
        if not inverse:
            l, m, r = x[..., :i], x[..., i:i+1], x[..., i+1:]
            m = F.one_hot(m.to(torch.long), num_classes=s).squeeze(-2)
            m = (m - self.mean) / self.std
        else:
            l, m, r = x[..., :i], x[..., i:i+s], x[..., i+s:]
            m = m * self.std + self.mean
            m = torch.argmax(m, dim=-1).unsqueeze(-1)
        return torch.cat([l, m, r], dim=-1)


if __name__ == '__main__':
    x = torch.randn(10, 7, 5)
    dim = 4
    x[..., dim] = 2
    oh = OneHot(dim=dim, size=4)
    x_hat = oh(oh(x), inverse=True)
    print(x_hat.shape)
