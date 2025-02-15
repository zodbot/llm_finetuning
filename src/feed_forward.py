import math

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), )

    def forward(self, x):
        return self.layers(x)


def GELU(x):
    """
    Gaussian Error Linear Unit (GELU)
    GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal distribution
    """
    # Exact formula
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
