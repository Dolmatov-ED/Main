"""
training/contrastive.py — Contrastive learning for separating high/low skill behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TripletLoss(nn.Module):
    """Triplet margin loss for skill-aware contrastive learning."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        return torch.clamp(dist_pos - dist_neg + self.margin, min=0.0).mean()


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for skill clustering."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        emb = F.normalize(embeddings, dim=-1)
        sim = torch.matmul(emb, emb.T) / self.temperature
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask = pos_mask - torch.eye(len(embeddings), device=embeddings.device)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(dim=-1)
        all_sum = exp_sim.sum(dim=-1) - torch.exp(sim.diag())
        all_sum = all_sum.clamp(min=1e-9)
        return -torch.log(pos_sum.clamp(min=1e-9) / all_sum).mean()


class SkillContrastiveHead(nn.Module):
    """Projects pooled window embeddings to contrastive space."""

    def __init__(self, d_model: int = 512, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        return self.proj(x)

    def compute_triplet_loss(self, anchor, positive, negative, margin=0.5):
        a = self.forward(anchor)
        p = self.forward(positive)
        n = self.forward(negative)
        return TripletLoss(margin=margin)(a, p, n)
