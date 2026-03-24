import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""
# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class FixedMultiCategorical:
    def __init__(self, dists):
        self.dists = dists

    def sample(self):
        return torch.cat([dist.sample().unsqueeze(-1) for dist in self.dists], dim=-1)

    def log_probs(self, actions):
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        parts = []
        for idx, dist in enumerate(self.dists):
            parts.append(dist.log_prob(actions[:, idx]).unsqueeze(-1))
        return torch.cat(parts, dim=-1).sum(dim=-1, keepdim=True)

    def entropy(self):
        entropies = [dist.entropy() for dist in self.dists]
        return torch.stack(entropies, dim=-1).sum(dim=-1)

    def mode(self):
        modes = [dist.probs.argmax(dim=-1, keepdim=True) for dist in self.dists]
        return torch.cat(modes, dim=-1)


class MultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linears = nn.ModuleList(
            [init_(nn.Linear(num_inputs, int(out_dim))) for out_dim in num_outputs]
        )

    def forward(self, x):
        dists = [torch.distributions.Categorical(logits=linear(x)) for linear in self.linears]
        return FixedMultiCategorical(dists)
