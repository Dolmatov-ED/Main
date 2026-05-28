"""
mocks/mock_training.py — Synthetic data and mock model for v4 training tests.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MockTransformerModel(nn.Module):
    """
    Minimal mock model that returns hidden_states dict.
    For testing CS2Trainer without real Transformer.
    """

    def __init__(self, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.linear = nn.Linear(d_model, d_model)
        # Optional heads (attached in tests)
        self.death_head: Optional[nn.Module] = None
        self.value_head: Optional[nn.Module] = None

    def forward(self, x: torch.Tensor, **kwargs):
        """Simple linear projection + empty KV caches."""
        hidden = self.linear(x)
        # Return dict matching CS2Transformer interface
        return {"hidden_states": hidden, "kv_caches": []}

    def attach_death_head(self, head: nn.Module):
        self.death_head = head

    def attach_value_head(self, head: nn.Module):
        self.value_head = head


def make_token_batch(
    batch_size: int = 4,
    seq_len: int = 16,
    d_model: int = 64,
) -> torch.Tensor:
    """Synthetic token batch [B, S, d_model]."""
    return torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)


def make_death_targets(batch_size: int = 4, seq_len: int = 16) -> torch.Tensor:
    """Synthetic death labels with ~5% positive."""
    targets = torch.zeros(batch_size, seq_len, 1)
    mask = torch.rand(batch_size, seq_len, 1) > 0.95
    targets[mask] = 1.0
    return targets


def make_value_targets(batch_size: int = 4, seq_len: int = 16) -> torch.Tensor:
    """Synthetic value targets in [-1, 1]."""
    return torch.rand(batch_size, seq_len, 1) * 2 - 1


def make_proxy_batch(
    batch_size: int = 4,
    seq_len: int = 16,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic proxy metric batch."""
    return {
        "kills": torch.randint(0, 2, (batch_size, seq_len)).float(),
        "assists": torch.randint(0, 2, (batch_size, seq_len)).float(),
        "survived": torch.randint(0, 2, (batch_size, seq_len)).float(),
        "traded": torch.randint(0, 2, (batch_size, seq_len)).float(),
        "cover_score": torch.rand(batch_size, seq_len),
        "exposure_time": torch.rand(batch_size, seq_len) * 30,
        "win_prob_before": torch.rand(batch_size, seq_len),
        "win_prob_after": torch.rand(batch_size, seq_len),
        "teammate_death_tick": torch.randint(0, seq_len * 10, (batch_size, seq_len)).float(),
        "revenge_kill_tick": torch.randint(0, seq_len * 10, (batch_size, seq_len)).float(),
    }


class MockDataLoader:
    """Mock DataLoader that yields synthetic batches."""

    def __init__(self, batch_size: int = 4, seq_len: int = 16,
                 n_batches: int = 5, d_model: int = 64):
        self.batches = [
            {"tokens": make_token_batch(batch_size, seq_len, d_model)}
            for _ in range(n_batches)
        ]
        self._iter = iter(self.batches)

    def __iter__(self):
        self._iter = iter(self.batches)
        return self

    def __next__(self):
        return next(self._iter)
