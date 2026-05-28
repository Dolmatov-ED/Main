"""
tokenizer/hybrid.py — HybridTokenizer: combines continuous MLP projectors
with discrete event embeddings into a unified d_model token.

Architecture:
    Token_t = LayerNorm(Emb_Pos + Emb_Orient + Emb_State + Emb_Event + Emb_Cover)

Input: structured dict with tensors
Output: [B, S, d_model] ready for Transformer with RoPE
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .projectors import (
    PositionProjector, OrientProjector,
    StateProjector, CoverProjector,
)
from .events import EventEmbedder, VOCAB_SIZE


class HybridTokenizer(nn.Module):
    """
    Converts raw per-tick feature vectors into unified d_model tokens.

    Input features (per tick, per player):
      pos:    [dx_c4, dy_c4, dz, vel_x, vel_y]  (5d)
      orient: [yaw_cos, yaw_sin, pitch]          (3d)
      state:  [health_norm, armor_norm, money_norm] (3d)
      cover:  [cover_score]                       (1d)
      event:  int event ID                        (scalar)

    All features are projected to d_model, summed, and normalized.
    """

    def __init__(
        self,
        d_model: int = 512,
        event_vocab_size: int = VOCAB_SIZE,
    ):
        super().__init__()
        self.d_model = d_model

        # Continuous projectors
        self.proj_pos = PositionProjector(d_model)
        self.proj_orient = OrientProjector(d_model)
        self.proj_state = StateProjector(d_model)
        self.proj_cover = CoverProjector(d_model)

        # Discrete embedder
        self.emb_event = EventEmbedder(
            vocab_size=event_vocab_size, d_model=d_model
        )

        # Final normalization
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        pos: torch.Tensor,
        orient: torch.Tensor,
        state: torch.Tensor,
        cover: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pos:    [B, S, 5] or [S, 5] — position features
            orient: [B, S, 3] or [S, 3] — orientation features
            state:  [B, S, 3] or [S, 3] — state features
            cover:  [B, S, 1] or [S, 1] — cover score
            events: [B, S] or [S]         — event IDs (long)

        Returns:
            tokens: [B, S, d_model] or [S, d_model]
        """
        h_pos = self.proj_pos(pos)
        h_orient = self.proj_orient(orient)
        h_state = self.proj_state(state)
        h_cover = self.proj_cover(cover)
        h_event = self.emb_event(events)

        # Summation + LayerNorm (standard for multimodal transformers)
        tokens = self.ln(h_pos + h_orient + h_state + h_cover + h_event)
        return tokens

    def forward_dict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convenience method that unpacks a dictionary.

        Expected keys:
            "pos", "orient", "state", "cover", "events"
        """
        return self.forward(
            pos=batch["pos"],
            orient=batch["orient"],
            state=batch["state"],
            cover=batch["cover"],
            events=batch["events"],
        )

    @staticmethod
    def get_default_dtype() -> torch.dtype:
        return torch.float32

    @staticmethod
    def get_feature_shapes() -> Dict[str, Tuple[int, ...]]:
        """Return expected feature shapes (last dim only)."""
        return {
            "pos": (5,),
            "orient": (3,),
            "state": (3,),
            "cover": (1,),
            "events": (),
        }
