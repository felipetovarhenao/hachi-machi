import torch
import torch.nn as nn
from torch.distributions import Normal
from .model import EventModel


class NLLLoss(nn.Module):

    def forward(self,
                pi: torch.Tensor,
                mu: torch.Tensor,
                sigma: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # repeat each dimension for every gaussian mixture
        target = target.unsqueeze(-1).expand_as(mu)
        dist = Normal(loc=mu, scale=sigma)
        # add log prob across dimensions
        log_prob = dist.log_prob(target).sum(dim=-2)
        log_pi = torch.log(pi + 1e-4)
        joint_prob = log_prob + log_pi
        return -torch.logsumexp(joint_prob, dim=-1).mean()
