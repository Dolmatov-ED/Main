"""
tokenizer/map_layers.py — Map layer tensor generation.

Produces MapTensor [C, H, W] with tactical channels.
In production, .bsp files are parsed. For v2, provides
the interface and a mock generator for testing.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path


MAP_SIZE = 256


class MapLayerGenerator:
    """Generates tactical map layers from .bsp geometry."""

    def __init__(self, map_name: str = "de_mirage", resolution: int = MAP_SIZE,
                 output_dir: str = "map_tensors"):
        self.map_name = map_name
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self._tensor: Optional[np.ndarray] = None

    def generate(self) -> np.ndarray:
        """Generate synthetic map layers (mock for v2; real BSP parsing in v3)."""
        rng = np.random.RandomState(hash(self.map_name) % (2**31))
        H = W = self.resolution
        s = H / 256.0

        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        height = (
            0.3 * np.exp(-((X - 0.2)**2 + (Y - 0.3)**2) / 0.1) +
            0.5 * np.exp(-((X + 0.3)**2 + (Y - 0.1)**2) / 0.15) +
            0.2 * np.exp(-((X)**2 + (Y + 0.4)**2) / 0.2) +
            rng.randn(H, W) * 0.02
        )
        height = np.clip(height, 0.0, 1.0)

        walk = np.ones((H, W), dtype=np.float32)
        # Wall segments
        for (wy, wh, wx, ww) in [(60, 20, 130, 60), (140, 100, 125, 5), (200, 20, 100, 140)]:
            y1, x1 = int(wy * s), int(wx * s)
            h1, w1 = max(2, int(wh * s)), max(1, int(ww * s))
            if y1 + h1 <= H and x1 + w1 <= W:
                walk[y1:y1 + h1, x1:x1 + w1] = 0.0
        walk += rng.randn(H, W) * 0.05
        walk = np.clip(walk, 0.0, 1.0)

        cover = np.zeros((H, W), dtype=np.float32)
        cy = int(60 * s)
        spread = max(1, int(10 * s))
        for i in range(max(0, cy - spread), min(H, cy + spread)):
            j_start = max(0, int(130 * s))
            j_end = min(W, int(190 * s))
            dist = abs(i - cy)
            cover[i, j_start:j_end] += np.exp(-dist**2 / max(1, spread**2)) * 0.7
        cover += rng.rand(H, W) * 0.1
        cover = np.clip(cover, 0.0, 1.0)

        self._tensor = np.stack([height, walk, cover], axis=0).astype(np.float32)
        return self._tensor

    def save(self, filepath: Optional[str] = None) -> Path:
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


MAP_LIST = [
    "de_dust2", "de_mirage", "de_inferno", "de_nuke",
    "de_anubis", "de_ancient", "de_vertigo", "de_overpass",
]
