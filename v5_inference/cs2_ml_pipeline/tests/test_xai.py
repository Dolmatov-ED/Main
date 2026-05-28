"""
Tests for inference/xai.py
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.inference.xai import XAIModule
from cs2_ml_pipeline.mocks.mock_inference import (
    MockTransformerForStreaming, make_log_probs,
)


class TestXAIModule:

    @pytest.fixture
    def model(self):
        return MockTransformerForStreaming(d_model=32)

    @pytest.fixture
    def xai(self, model):
        return XAIModule(model)

    def test_attention_rollout_shape(self, xai):
        hidden = torch.randn(2, 16, 32)
        rollout = xai.attention_rollout(hidden)
        assert rollout.shape == (2, 16)

    def test_attention_rollout_sums_to_one(self, xai):
        """Fallback uniform weights sum to 1."""
        hidden = torch.randn(2, 16, 32)
        rollout = xai.attention_rollout(hidden)
        assert torch.allclose(rollout.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_gradient_saliency(self, xai):
        tokens = torch.randn(1, 4, 32)
        saliency = xai.gradient_saliency(tokens, target_dim=0)
        assert saliency.shape == (1, 4, 32)
        assert (saliency >= 0).all()

    def test_log_likelihood_decay(self, xai):
        log_probs = make_log_probs(64)
        decay = xai.log_likelihood_decay(log_probs, window=8)
        assert decay.shape == (64,)

    def test_score_to_color(self):
        green = XAIModule.score_to_color(0.9)
        assert green[0] == "green"
        yellow = XAIModule.score_to_color(0.6)
        assert yellow[0] == "yellow"
        red = XAIModule.score_to_color(0.3)
        assert red[0] == "red"

    def test_generate_hint(self):
        hint = XAIModule.generate_hint(0.9, 0.1, "connector")
        assert "Optimal" in hint
        hint2 = XAIModule.generate_hint(0.3, 0.8, "connector")
        assert "Risk" in hint2
