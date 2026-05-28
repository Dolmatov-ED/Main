"""
Tests for training/contrastive.py
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.training.contrastive import (
    TripletLoss, InfoNCELoss, SkillContrastiveHead,
)


class TestTripletLoss:
    def test_loss_positive(self):
        loss_fn = TripletLoss(margin=0.5)
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.1, 0.1]])  # close to anchor
        negative = torch.tensor([[5.0, 5.0]])  # far from anchor
        loss = loss_fn(anchor, positive, negative)
        assert loss >= 0
        # Distance to positive < distance to negative → small loss
        assert loss < 1.0

    def test_loss_negative(self):
        loss_fn = TripletLoss(margin=0.5)
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[5.0, 5.0]])  # far
        negative = torch.tensor([[1.1, 0.1]])  # close
        loss = loss_fn(anchor, positive, negative)
        assert loss > 0

    def test_loss_zero_when_satisfied(self):
        loss_fn = TripletLoss(margin=0.5)
        anchor = torch.tensor([[0.0, 0.0]])
        positive = torch.tensor([[0.0, 0.0]])  # same as anchor
        negative = torch.tensor([[10.0, 10.0]])  # very far
        loss = loss_fn(anchor, positive, negative)
        assert loss == 0.0  # margin satisfied

    def test_batched(self):
        loss_fn = TripletLoss(margin=0.5)
        anchor = torch.randn(4, 16)
        positive = torch.randn(4, 16)
        negative = torch.randn(4, 16)
        loss = loss_fn(anchor, positive, negative)
        assert loss >= 0


class TestInfoNCELoss:
    def test_loss_scalar(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        emb = torch.randn(8, 32)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(emb, labels)
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_loss_perfect_separation(self):
        """Two clusters far apart → low loss."""
        loss_fn = InfoNCELoss(temperature=0.1)
        # High skill cluster
        high = torch.tensor([[1.0, 0.0], [1.1, 0.1]])
        # Low skill cluster
        low = torch.tensor([[-1.0, 0.0], [-1.1, 0.1]])
        emb = torch.cat([high, low], dim=0)
        labels = torch.tensor([1, 1, 0, 0])
        loss = loss_fn(emb, labels)
        assert loss < 2.0  # should be low


class TestSkillContrastiveHead:
    @pytest.fixture
    def head(self):
        return SkillContrastiveHead(d_model=64, proj_dim=32)

    def test_output_shape(self, head):
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 32)

    def test_triplet_loss(self, head):
        anchor = torch.randn(4, 64)
        positive = torch.randn(4, 64)
        negative = torch.randn(4, 64)
        loss = head.compute_triplet_loss(anchor, positive, negative)
        assert loss >= 0
