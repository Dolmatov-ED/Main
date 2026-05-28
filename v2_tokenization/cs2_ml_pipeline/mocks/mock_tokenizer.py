"""
mocks/mock_tokenizer.py — Synthetic tensor generators for testing tokenizer.

All generators produce float32 tensors compatible with HybridTokenizer.
No external dependencies (no awpy, no .dem files needed).
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


SEED = 42


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class BatchGenerator:
    """
    Generates synthetic batches for tokenizer testing.

    Produces dict of tensors with keys expected by HybridTokenizer:
      pos, orient, state, cover, events
    """

    def __init__(
        self,
        batch_size: int = 4,
        seq_len: int = 128,
        d_model: int = 512,
        seed: int = SEED,
    ):
        set_seed(seed)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model

    def generate(
        self,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate a full batch with all feature groups."""
        B = batch_size or self.batch_size
        S = seq_len or self.seq_len

        return {
            "pos": self._make_pos(B, S),
            "orient": self._make_orient(B, S),
            "state": self._make_state(B, S),
            "cover": self._make_cover(B, S),
            "events": self._make_events(B, S),
        }

    def generate_no_batch(self, seq_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Generate a single sequence (no batch dim)."""
        S = seq_len or self.seq_len
        return {
            "pos": self._make_pos(1, S).squeeze(0),
            "orient": self._make_orient(1, S).squeeze(0),
            "state": self._make_state(1, S).squeeze(0),
            "cover": self._make_cover(1, S).squeeze(0),
            "events": self._make_events(1, S).squeeze(0),
        }

    @staticmethod
    def _make_pos(B: int, S: int) -> torch.Tensor:
        """Position: [dx_c4, dy_c4, dz, vel_x, vel_y] — 5 features."""
        return torch.randn(B, S, 5, dtype=torch.float32) * 0.5

    @staticmethod
    def _make_orient(B: int, S: int) -> torch.Tensor:
        """Orientation: [yaw_cos, yaw_sin, pitch] — 3 features, cos/sin in [-1,1]."""
        yaw = torch.rand(B, S, 1) * 2 * np.pi
        return torch.cat([
            torch.cos(yaw),
            torch.sin(yaw),
            torch.randn(B, S, 1) * 0.2,  # pitch
        ], dim=-1).to(torch.float32)

    @staticmethod
    def _make_state(B: int, S: int) -> torch.Tensor:
        """State: [health_norm, armor_norm, money_norm] — 3 features, range [0,1]."""
        return torch.rand(B, S, 3, dtype=torch.float32)

    @staticmethod
    def _make_cover(B: int, S: int) -> torch.Tensor:
        """Cover: [cover_score] — 1 feature, range [0,1]."""
        return torch.rand(B, S, 1, dtype=torch.float32)

    @staticmethod
    def _make_events(B: int, S: int) -> torch.Tensor:
        """Events: integer IDs, mostly NULL=0 with some events."""
        # 80% NULL, 20% random events
        events = torch.zeros(B, S, dtype=torch.long)
        mask = torch.rand(B, S) > 0.8
        events[mask] = torch.randint(1, 30, (mask.sum(),)).long()
        return events


def generate_edge_case_batch() -> Dict[str, torch.Tensor]:
    """Generate edge cases: all zeros, all ones, mixed."""
    B, S = 2, 4

    all_zeros = {
        "pos": torch.zeros(B, S, 5),
        "orient": torch.zeros(B, S, 3),
        "state": torch.zeros(B, S, 3),
        "cover": torch.zeros(B, S, 1),
        "events": torch.zeros(B, S, dtype=torch.long),
    }

    all_ones = {
        "pos": torch.ones(B, S, 5),
        "orient": torch.ones(B, S, 3),
        "state": torch.ones(B, S, 3),
        "cover": torch.ones(B, S, 1),
        "events": torch.ones(B, S, dtype=torch.long),
    }

    dead_players = {
        "pos": torch.randn(B, S, 5) * 0.3,
        "orient": torch.randn(B, S, 3) * 0.1,
        "state": torch.zeros(B, S, 3),     # health=0, armor=0, money=0
        "cover": torch.rand(B, S, 1),
        "events": torch.tensor([0, 2, 0, 0] * B, dtype=torch.long).reshape(B, S),
    }

    return all_zeros, all_ones, dead_players
