import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureScaler(nn.Module):

    def __init__(self, data: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = data.clone()
        self.e = 1
        data = self.log_time(data)
        mean = data.mean(0)
        std = data.std(0)
        std[torch.where(std == 0)] = 1.0
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def log_time(self, x: torch.Tensor, inverse: bool = False):
        if inverse:
            x[..., :2] = torch.exp2(x[..., :2]) * 1000 - self.e
        else:
            x[..., :2] = torch.log2((x[..., :2] + self.e) / 1000)
        return x

    def forward(self, x: torch.Tensor, inverse: bool = False):
        x = x.clone()
        if inverse:
            y = x * self.std + self.mean
            y = self.log_time(y, inverse)
        else:
            y = self.log_time(x, inverse)
            y = (y - self.mean) / self.std
        return y


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


class RecurrentMDN(nn.Module):

    def __init__(self,
                 k: int = 2,
                 input_size: int = 2,
                 hidden_size: int = 100,
                 dropout: float = 0.25,
                 slope: float = 0.01,
                 device: str = 'mps',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            batch_first=True).to(device)
        self.proj = MixtureDensityNetwork(k=k,
                                          num_features=hidden_size,
                                          out_features=input_size,
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
            torch.exp((log / temp).clip(-88, 88)), num_samples=1).item()
        y = torch.normal(mean=mu[0, 0, :, k],
                         std=sigma[0, 0, :, k]).unsqueeze(0).unsqueeze(0)
        return y, hidden


class MusicAgent(nn.Module):

    def __init__(self,
                 model: RecurrentMDN,
                 scaler:  FeatureScaler,
                 player_voices: list[int] = [0],
                 device: str = 'mps',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.device = device
        self.scaler = scaler
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
        y: torch.Tensor = (x - self.next_event) ** 2 * self.weights
        y = y.sum().sqrt()
        return torch.exp(-self.alpha * y).item()

    def forward(self, x: torch.Tensor) -> None | torch.Tensor:
        x: torch.Tensor = self.scaler(x)
        if self.hidden_state is not None:
            conf = self.get_confidence(x)
            (hn, cn) = self.hidden_state
            hn = hn.clone() * conf
            cn = cn.clone() * conf
            self.hidden_state = (hn, cn)
        self.next_event, self.hidden_state = self.model.step(x=x,
                                                             hidden=self.hidden_state,
                                                             temp=self.temp)
        y: torch.Tensor = self.scaler(
            self.next_event.clone(), inverse=True)
        y = y.clip(0).squeeze().round().int()
        if y[2] in self.player_voices:
            return
        return y
