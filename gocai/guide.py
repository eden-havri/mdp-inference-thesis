from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Simple MLP mapping state (x,y) -> action logits."""

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: int = 2, hidden_width: int = 64):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(d, hidden_width))
            layers.append(nn.Tanh())
            d = hidden_width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Guide:
    policy: MLPPolicy

    def probs(self, state: Tuple[float, float]) -> torch.Tensor:
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1,2]
        logits = self.policy(x)[0]
        return torch.softmax(logits, dim=-1)

    def logits(self, states_flat: torch.Tensor) -> torch.Tensor:
        """Vectorized logits for many states, shape [B, n_actions]."""
        return self.policy(states_flat)
