import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class NLLLoss(nn.Module):
    def forward(self,
                pi: torch.Tensor,
                mu: torch.Tensor,
                L: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(-2).expand_as(mu)
        dist = MultivariateNormal(loc=mu, scale_tril=L)
        log_prob = dist.log_prob(target)
        log_pi = torch.log(pi + 1e-8)
        return -torch.logsumexp(log_prob + log_pi, dim=-1).mean()
