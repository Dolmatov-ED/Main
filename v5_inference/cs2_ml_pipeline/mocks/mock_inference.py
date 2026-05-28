"""
mocks/mock_inference.py — Synthetic data and mock models for v5 tests.
"""

import torch
import torch.nn as nn


class MockTransformerForStreaming(nn.Module):
    """Minimal transformer for streaming inference tests."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, **kwargs):
        hidden = self.linear(x)
        return {
            "hidden_states": hidden,
            "kv_caches": [],
        }

    def generate_one_step(self, token, kv_caches, offset):
        hidden = self.linear(token)
        return hidden, kv_caches or []


def make_stream_tokens(
    seq_len: int = 64,
    d_model: int = 32,
) -> torch.Tensor:
    """Generate streaming token sequence [1, S, d_model]."""
    return torch.randn(1, seq_len, d_model, dtype=torch.float32)


def make_log_probs(seq_len: int = 64) -> torch.Tensor:
    """Generate synthetic log probabilities."""
    probs = torch.randn(seq_len)
    probs[seq_len // 2:] += torch.linspace(-2, 0, seq_len // 2)
    return probs
