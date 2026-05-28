"""
Tests for models/heads.py and models/map_conditioning.py.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.models.heads import (
    NextTokenHead, DeathHead, ValueHead,
)
from cs2_ml_pipeline.models.map_conditioning import (
    MapTokenInjector, CrossAttentionBlock,
)
from cs2_ml_pipeline.mocks.mock_models import (
    make_token_batch, make_z_map_batch,
    make_death_targets, make_value_targets,
    set_seed, TEST_D_MODEL, TEST_Z_DIM,
)


class TestNextTokenHead:
    @pytest.fixture
    def head(self):
        return NextTokenHead(d_model=TEST_D_MODEL)

    def test_shape(self, head):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        pred = head(x)
        assert pred.shape == (2, 8, TEST_D_MODEL)

    def test_compute_loss(self, head):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        pred = head(x)
        # Target = shifted input (simulate next-token)
        target = torch.cat([x[:, 1:, :], torch.zeros(2, 1, TEST_D_MODEL)], dim=1)
        loss = head.compute_loss(pred, target)
        assert loss >= 0
        assert not torch.isnan(loss)


class TestDeathHead:
    @pytest.fixture
    def head(self):
        return DeathHead(d_model=TEST_D_MODEL, hidden_dim=32)

    def test_shape(self, head):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        prob = head(x)
        assert prob.shape == (2, 8, 1)

    def test_output_range(self, head):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        prob = head(x)
        assert prob.min() >= 0
        assert prob.max() <= 1

    def test_focal_loss(self):
        pred = torch.tensor([[0.1, 0.9, 0.05]]).unsqueeze(-1)
        target = torch.tensor([[0.0, 1.0, 0.0]]).unsqueeze(-1)
        loss = DeathHead.focal_loss(pred, target)
        assert loss > 0
        assert not torch.isnan(loss)

    def test_focal_loss_imbalance(self):
        """Focal loss penalizes confident wrong predictions less for negatives."""
        pred = torch.tensor([[0.01, 0.99, 0.01]]).unsqueeze(-1)
        target = torch.tensor([[0.0, 1.0, 0.0]]).unsqueeze(-1)
        loss = DeathHead.focal_loss(pred, target)
        # Loss should be low (confident and correct)
        assert loss < 1.0

    def test_gradient(self, head):
        x = make_token_batch(batch_size=2, seq_len=4, d_model=TEST_D_MODEL)
        targets = make_death_targets(batch_size=2, seq_len=4)
        prob = head(x)
        loss = DeathHead.focal_loss(prob, targets)
        loss.backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


class TestValueHead:
    @pytest.fixture
    def head(self):
        return ValueHead(d_model=TEST_D_MODEL, hidden_dim=32)

    def test_shape(self, head):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        val = head(x)
        assert val.shape == (2, 8, 1)

    def test_output_range(self, head):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        val = head(x)
        assert val.min() >= -1.0
        assert val.max() <= 1.0

    def test_huber_loss(self, head):
        x = make_token_batch(batch_size=2, seq_len=4, d_model=TEST_D_MODEL)
        targets = make_value_targets(batch_size=2, seq_len=4)
        pred = head(x)
        loss = ValueHead.huber_loss(pred, targets)
        assert loss >= 0
        assert not torch.isnan(loss)


class TestMapTokenInjector:
    @pytest.fixture
    def injector(self):
        return MapTokenInjector(z_dim=TEST_Z_DIM, d_model=TEST_D_MODEL)

    def test_shape(self, injector):
        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        z_map = make_z_map_batch(batch_size=2, z_dim=TEST_Z_DIM)
        out = injector(tokens, z_map)
        # [MAP] token prepended → S+1
        assert out.shape == (2, 9, TEST_D_MODEL)

    def test_map_token_is_first(self, injector):
        tokens = torch.zeros(2, 4, TEST_D_MODEL)
        z_map = torch.ones(2, TEST_Z_DIM)
        out = injector(tokens, z_map)
        # First token should be non-zero (map token)
        assert not torch.allclose(out[:, 0, :], torch.zeros_like(out[:, 0, :]))

    def test_gradient(self, injector):
        tokens = make_token_batch(batch_size=2, seq_len=4, d_model=TEST_D_MODEL)
        z_map = make_z_map_batch(batch_size=2, z_dim=TEST_Z_DIM)
        z_map = z_map.clone().requires_grad_(True)
        out = injector(tokens, z_map)
        out.sum().backward()
        assert z_map.grad is not None


class TestCrossAttentionBlock:
    @pytest.fixture
    def cross_attn(self):
        return CrossAttentionBlock(z_dim=TEST_Z_DIM, d_model=TEST_D_MODEL, n_heads=4)

    def test_shape(self, cross_attn):
        hidden = make_token_batch(batch_size=2, seq_len=8, d_model=TEST_D_MODEL)
        z_map = make_z_map_batch(batch_size=2, z_dim=TEST_Z_DIM)
        out = cross_attn(hidden, z_map)
        assert out.shape == (2, 8, TEST_D_MODEL)

    def test_gradient(self, cross_attn):
        hidden = make_token_batch(batch_size=2, seq_len=4, d_model=TEST_D_MODEL)
        hidden = hidden.clone().requires_grad_(True)
        z_map = make_z_map_batch(batch_size=2, z_dim=TEST_Z_DIM)
        out = cross_attn(hidden, z_map)
        out.sum().backward()
        assert hidden.grad is not None
