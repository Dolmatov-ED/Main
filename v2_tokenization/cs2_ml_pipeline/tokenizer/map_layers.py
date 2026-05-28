"""
tokenizer/map_layers.py — Map layer tensor generation.

Produces MapTensor [C, H, W] with tactical channels:
  Channel 0: Height map (voxelized from .bsp)
  Channel 1: Walkability (nav mesh projection)
  Channel 2: Cover Score (raycast density)

In production, .bsp files are parsed. For v2, we provide
the interface and a mock generator for testing.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path


MAP_SIZE = 256  # H x W resolution


class MapLayerGenerator:
    """
    Generates tactical map layers from .bsp geometry.
    Mocked for v2; real .bsp parsing comes in v3 (Map-AE).

    Output: MapTensor [C=3, H=256, W=256] in float32.
    """

    def __init__(
        self,
        map_name: str = "de_mirage",
        resolution: int = MAP_SIZE,
        output_dir: str = "map_tensors",
    ):
        self.map_name = map_name
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self._tensor: Optional[np.ndarray] = None

    def generate(self) -> np.ndarray:
        """
        Generate synthetic map layers.
        In production, this would parse .bsp and do raycasting.
        Uses resolution-aware scaling so it works for any size.
        """
        rng = np.random.RandomState(hash(self.map_name) % (2**31))
        H = W = self.resolution
        s = H / 256.0  # scale factor from base 256 resolution

        # Height map: smooth hills (simulated via radial basis)
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        height = (
            0.3 * np.exp(-((X - 0.2)**2 + (Y - 0.3)**2) / 0.1) +
            0.5 * np.exp(-((X + 0.3)**2 + (Y - 0.1)**2) / 0.15) +
            0.2 * np.exp(-((X)**2 + (Y + 0.4)**2) / 0.2) +
            rng.randn(H, W) * 0.02
        )
        height = np.clip(height, 0.0, 1.0)

        # Walkability: 0 = blocked, 1 = walkable
        walk = np.ones((H, W), dtype=np.float32)
        # Wall segments scaled to resolution
        w1_h = max(2, int(20 * s))
        w1_w = max(4, int(60 * s))
        w1_y = int(60 * s)
        w1_x = int(130 * s)
        if w1_y + w1_h <= H and w1_x + w1_w <= W:
            walk[w1_y:w1_y + w1_h, w1_x:w1_x + w1_w] = 0.0

        w2_h = max(4, int(100 * s))
        w2_w = max(1, int(5 * s))
        w2_y = int(140 * s)
        w2_x = int(125 * s)
        if w2_y + w2_h <= H and w2_x + w2_w <= W:
            walk[w2_y:w2_y + w2_h, w2_x:w2_x + w2_w] = 0.0

        w3_h = max(2, int(20 * s))
        w3_w = max(3, int(140 * s))
        w3_y = int(200 * s)
        w3_x = int(100 * s)
        if w3_y + w3_h <= H and w3_x + w3_w <= W:
            walk[w3_y:w3_y + w3_h, w3_x:w3_x + w3_w] = 0.0

        walk += rng.randn(H, W) * 0.05
        walk = np.clip(walk, 0.0, 1.0)

        # Cover score: 0 = exposed, 1 = full cover
        cover = np.zeros((H, W), dtype=np.float32)
        # Wall-like cover patterns scaled by resolution
        cover_y = int(60 * s)
        spread = max(1, int(10 * s))
        for i in range(max(0, cover_y - spread), min(H, cover_y + spread)):
            j_start = max(0, int(130 * s))
            j_end = min(W, int(190 * s))
            dist = abs(i - cover_y)
            cover[i, j_start:j_end] += np.exp(-dist**2 / max(1, spread**2)) * 0.7

        cover += rng.rand(H, W) * 0.1
        cover = np.clip(cover, 0.0, 1.0)

        self._tensor = np.stack([height, walk, cover], axis=0).astype(np.float32)
        return self._tensor

    def save(self, filepath: Optional[str] = None) -> Path:
        """Save MapTensor as .npy file."""
        if self._tensor is None:
            self.generate()
        if filepath is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / f"{self.map_name}_map.npy"
        else:
            filepath = Path(filepath)
        np.save(str(filepath), self._tensor)
        return filepath

    def load(self, filepath: str) -> np.ndarray:
        """Load MapTensor from .npy file."""
        self._tensor = np.load(filepath)
        return self._tensor

    @property
    def tensor(self) -> Optional[np.ndarray]:
        return self._tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._tensor is None:
            return (3, self.resolution, self.resolution)
        return self._tensor.shape


# Pre-defined map configurations for later BSP parsing
MAP_LIST = [
    "de_dust2", "de_mirage", "de_inferno", "de_nuke",
    "de_anubis", "de_ancient", "de_vertigo", "de_overpass",
]
