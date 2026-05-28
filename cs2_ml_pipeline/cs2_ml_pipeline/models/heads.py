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
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, hidden):
        return self.ln(self.proj(hidden))

    def compute_loss(self, pred, target):
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        return 1.0 - (pred_n * target_n).sum(dim=-1).mean()


class DeathHead(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, hidden):
        return self.net(hidden)

    @staticmethod
    def focal_loss(pred, target, alpha=0.25, gamma=2.0):
        eps = 1e-7
        pred = pred.clamp(eps, 1 - eps)
        ce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** gamma
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        return (alpha_weight * focal_weight * ce).mean()


class ValueHead(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1), nn.Tanh(),
        )

    def forward(self, hidden):
        return self.net(hidden)

    @staticmethod
    def huber_loss(pred, target, delta=1.0):
        return F.huber_loss(pred, target, delta=delta)
