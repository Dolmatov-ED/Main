"""
training/contrastive.py — Contrastive learning for separating high/low skill behavior.

Key insight: high likelihood ≠ optimality. A player can consistently do suboptimal
things with high probability. Contrastive loss pushes high-impact and low-impact
phases apart in latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TripletLoss(nn.Module):
    """
    Triplet margin loss for skill-aware contrastive learning.

    Pushes anchor (current) closer to positive (high-skill similar) than
    negative (low-skill or low-impact).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   [B, d] pooled window embeddings
            positive: [B, d] high-skill or high-impact embeddings
            negative: [B, d] low-skill or low-impact embeddings
        Returns:
            scalar loss
        """
        dist_pos = F.pairwise_distance(anchor, positive, p=2)
        dist_neg = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for skill clustering.
    Positives = high-skill phases, negatives = low-skill phases.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [N, d] normalized embeddings
            labels: [N] binary: 1 = high skill, 0 = low skill
        Returns:
            scalar loss
        """
        # Normalize
        emb = F.normalize(embeddings, dim=-1)

        # Similarity matrix
        sim = torch.matmul(emb, emb.T) / self.temperature

        # Positive mask: same label
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        # Remove self
        pos_mask = pos_mask - torch.eye(len(embeddings), device=embeddings.device)

        # InfoNCE: -log(sum_pos / sum_all)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(dim=-1)
        all_sum = exp_sim.sum(dim=-1) - torch.exp(sim.diag())  # exclude self
        all_sum = all_sum.clamp(min=1e-9)

        loss = -torch.log(pos_sum.clamp(min=1e-9) / all_sum)
        return loss.mean()


class SkillContrastiveHead(nn.Module):
    """
    Projects pooled window embeddings to contrastive space.
    Used during contrastive fine-tuning phase.
    """

    def __init__(self, d_model: int = 512, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] → [B, proj_dim]"""
        return self.proj(x)

    def compute_triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
        """Convenience wrapper for triplet loss on projected embeddings."""
        a = self.forward(anchor)
        p = self.forward(positive)
        n = self.forward(negative)
        loss_fn = TripletLoss(margin=margin)
        return loss_fn(a, p, n)
