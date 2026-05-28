"""
mocks/mock_models.py — Small model factories and synthetic data for v3 tests.

Provides lightweight model instances (d_model=64, n_layers=2) for fast
testing of Map-AE, Transformer, conditioning, and heads.
"""

import torch
import numpy as np
from typing import Dict


# ── Constants for small test models ───────────────────────────────────

TEST_D_MODEL = 64
TEST_N_HEADS = 4
TEST_N_LAYERS = 2
TEST_Z_DIM = 32


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_map_batch(
    batch_size: int = 2,
    channels: int = 3,
    size: int = 32,
) -> torch.Tensor:
    """Generate synthetic MapTensor batch [B, C, H, W]."""
    return torch.rand(batch_size, channels, size, size, dtype=torch.float32)


def make_token_batch(
    batch_size: int = 2,
    seq_len: int = 16,
    d_model: int = TEST_D_MODEL,
) -> torch.Tensor:
    """Generate synthetic token batch [B, S, d_model]."""
    return torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)


def make_z_map_batch(batch_size: int = 2, z_dim: int = TEST_Z_DIM) -> torch.Tensor:
    """Generate synthetic z_map batch [B, z_dim]."""
    return torch.randn(batch_size, z_dim, dtype=torch.float32)


def make_death_targets(batch_size: int = 2, seq_len: int = 16) -> torch.Tensor:
    """Generate death labels [B, S, 1] with ~5% positive."""
    targets = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32)
    mask = torch.rand(batch_size, seq_len, 1) > 0.95
    targets[mask] = 1.0
    return targets


def make_value_targets(batch_size: int = 2, seq_len: int = 16) -> torch.Tensor:
    """Generate value targets [B, S, 1] in [-1, 1]."""
    return torch.rand(batch_size, seq_len, 1) * 2 - 1
