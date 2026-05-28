"""
models/heads.py — Downstream prediction heads for the CS2 Transformer.

Heads operate on the final hidden states:
  - NextTokenHead: predicts next token for autoregressive pretraining
  - DeathHead: predicts P(death in [t, t+Δt] | history)
  - ValueHead: predicts round impact / win probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NextTokenHead(nn.Module):
    """
    Predicts the next token from hidden states.
    Autoregressive objective: cross-entropy on predicted vs actual next token.

    Architecture: Linear(d_model, d_model) → the output is a vector in d_model,
    which is compared against the actual next token embedding.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, S, d_model] hidden states
        Returns:
            predictions: [B, S, d_model] predicted next token embeddings
        """
        return self.ln(self.proj(hidden))

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Cosine similarity loss between predicted and actual next token.
        """
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        return 1.0 - (pred_n * target_n).sum(dim=-1).mean()


class DeathHead(nn.Module):
    """
    Predicts probability of death in the near future.
    Binary classification with Focal Loss for class imbalance.

    Input: hidden state at time t
    Output: P(death in [t, t+Δt] | history up to t)
    """

    def __init__(self, d_model: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, S, d_model]
        Returns:
            death_prob: [B, S, 1]
        """
        return self.net(hidden)

    @staticmethod
    def focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal loss for imbalanced death prediction (< 5% positive).
        """
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        ce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** gamma
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        return (alpha_weight * focal_weight * ce).mean()


class ValueHead(nn.Module):
    """
    Predicts round impact / win probability delta.
    Output is in [-1, 1] via tanh.
    """

    def __init__(self, d_model: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, S, d_model]
        Returns:
            value: [B, S, 1] in range [-1, 1]
        """
        return self.net(hidden)

    @staticmethod
    def huber_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        delta: float = 1.0,
    ) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=delta)
