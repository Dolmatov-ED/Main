"""
tokenizer/projectors.py — MLP projectors for continuous features.

Each projector maps a small continuous vector → d_model embedding.
Using GELU activation for smooth gradients (better than ReLU for
noisy game data with micro-movements).
"""

import torch
import torch.nn as nn


class ContinuousProjector(nn.Module):
    """Generic MLP projector: input_dim → hidden → d_model."""

    def __init__(self, in_dim: int, hidden_dim: int, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, in_dim] or [S, in_dim]
        Returns:
            [B, S, d_model] or [S, d_model]
        """
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class PositionProjector(ContinuousProjector):
    """
    Projects spatial features: dx_to_c4, dy_to_c4, dz, vel_x, vel_y.
    Input: 5 dimensions.
    """

    def __init__(self, d_model: int = 512):
        super().__init__(in_dim=5, hidden_dim=64, d_model=d_model)


class OrientProjector(ContinuousProjector):
    """
    Projects orientation: yaw_cos, yaw_sin, pitch.
    Input: 3 dimensions.
    """

    def __init__(self, d_model: int = 512):
        super().__init__(in_dim=3, hidden_dim=32, d_model=d_model)


class StateProjector(ContinuousProjector):
    """
    Projects player state: health_norm, armor_norm, money_norm.
    Input: 3 dimensions.
    """

    def __init__(self, d_model: int = 512):
        super().__init__(in_dim=3, hidden_dim=32, d_model=d_model)


class CoverProjector(ContinuousProjector):
    """
    Projects cover score from map layers.
    Input: 1 dimension.
    """

    def __init__(self, d_model: int = 512):
        super().__init__(in_dim=1, hidden_dim=16, d_model=d_model)
