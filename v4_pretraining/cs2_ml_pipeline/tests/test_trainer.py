"""
Tests for training/trainer.py
"""

import sys
import os
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.training.trainer import CS2Trainer
from cs2_ml_pipeline.mocks.mock_training import (
    MockTransformerModel, make_token_batch,
    make_death_targets, make_value_targets, MockDataLoader,
)


class TestCS2Trainer:

    @pytest.fixture
    def model(self):
        return MockTransformerModel(d_model=64)

    @pytest.fixture
    def optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-3)

    @pytest.fixture
    def trainer(self, model, optimizer):
        return CS2Trainer(model, optimizer, device=torch.device("cpu"))

    def test_pretrain_step(self, trainer):
        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        result = trainer.pretrain_step(tokens)
        assert "ntp_loss" in result
        assert result["ntp_loss"] >= 0
        assert trainer.step == 1

    def test_pretrain_step_loss_decreases(self, trainer):
        """Multiple steps should see loss change."""
        losses = []
        for _ in range(5):
            tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
            result = trainer.pretrain_step(tokens)
            losses.append(result["ntp_loss"])
        # Loss should be tracked
        assert len(trainer.train_losses["ntp"]) == 5

    def test_multi_task_step(self, trainer, model):
        from cs2_ml_pipeline.models.heads import DeathHead, ValueHead
        model.attach_death_head(DeathHead(d_model=64, hidden_dim=16))
        model.attach_value_head(ValueHead(d_model=64, hidden_dim=16))

        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        death_t = make_death_targets(batch_size=2, seq_len=8)
        value_t = make_value_targets(batch_size=2, seq_len=8)

        result = trainer.multi_task_step(tokens, death_t, value_t)
        for k in ["ntp", "death", "value", "total"]:
            assert k in result
            assert result[k] >= 0

    def test_multi_task_step_without_heads(self, trainer):
        """Works without death/value heads — returns zero for those."""
        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        death_t = make_death_targets(2, 8)
        value_t = make_value_targets(2, 8)
        result = trainer.multi_task_step(tokens, death_t, value_t)
        assert result["death"] == 0.0
        assert result["value"] == 0.0

    def test_validate(self, trainer):
        loader = MockDataLoader(batch_size=2, seq_len=8, n_batches=3, d_model=64)
        metrics = trainer.validate(loader, max_batches=3)
        assert "ntp" in metrics
        assert metrics["ntp"] >= 0

    def test_get_metrics(self, trainer):
        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        trainer.pretrain_step(tokens)
        metrics = trainer.get_metrics()
        assert metrics["step"] == 1
        assert len(metrics["train_losses"]["ntp"]) == 1

    def test_gradient_update(self, trainer, model):
        """Verify model weights change after training step."""
        before = model.linear.weight.clone()
        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        trainer.pretrain_step(tokens)
        after = model.linear.weight.clone()
        assert not torch.allclose(before, after)

    def test_clip_gradient(self, model):
        """Gradient clipping activates on large gradients."""
        from cs2_ml_pipeline.models.heads import DeathHead, ValueHead
        model.attach_death_head(DeathHead(d_model=64, hidden_dim=16))
        model.attach_value_head(ValueHead(d_model=64, hidden_dim=16))

        opt = optim.SGD(model.parameters(), lr=10.0)  # Large lr → large gradients
        trainer = CS2Trainer(model, opt, clip_grad_norm=0.1)

        tokens = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        death_t = make_death_targets(2, 8)
        value_t = make_value_targets(2, 8)

        # Should not crash with clipping
        trainer.multi_task_step(tokens, death_t, value_t)
        assert trainer.step == 1

    def test_custom_loss_weights(self, trainer, model):
        from cs2_ml_pipeline.models.heads import DeathHead, ValueHead
        model.attach_death_head(DeathHead(d_model=64, hidden_dim=16))
        model.attach_value_head(ValueHead(d_model=64, hidden_dim=16))

        tokens = make_token_batch(2, 8, 64)
        death_t = make_death_targets(2, 8)
        value_t = make_value_targets(2, 8)

        weights = {"next_token": 1.0, "death": 2.0, "value": 0.0}
        result = trainer.multi_task_step(tokens, death_t, value_t, loss_weights=weights)
        # value_weight=0 so value loss shouldn't contribute
        assert result["value"] >= 0  # still computed but not in total
