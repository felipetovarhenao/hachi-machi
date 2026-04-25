import torch
import torch.nn.functional as F
import torch.nn as nn


class OneHot(nn.Module):
    def __init__(self, data: torch.Tensor, dim: int, size: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('i', torch.tensor(dim))
        self.register_buffer('size', torch.tensor(size))
        with torch.no_grad():
            data = data[..., dim].clone()
            data = F.one_hot(data.to(torch.long),
                             num_classes=size).to(torch.float)
            mean = data.mean(0)
            std = data.std(0)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

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
