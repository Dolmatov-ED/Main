"""
etl/exporter.py — Downsampling, normalization, and Parquet export.
Transforms per-round DataFrames into ML-ready tensors and serializes them.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


# Default feature schema
DEFAULT_SCHEMA = {
    "continuous": ["x", "y", "z", "yaw", "health", "armor"],
    "discrete": ["has_helmet", "has_defuser", "team"],
    "derived": ["yaw_cos", "yaw_sin", "dx_to_c4", "dy_to_c4", "t_round"],
    "meta": ["tick", "round_id", "player_name", "is_alive"],
}


class TickExporter:
    """
    Downsamples, normalizes, and exports round data to Parquet.
    """

    def __init__(
        self,
        rounds: Dict[int, pd.DataFrame],
        tick_rate: float = 64.0,
        target_hz: int = 8,
        output_dir: str = "dataset",
        map_name: str = "unknown",
    ):
        self.rounds = rounds
        self.tick_rate = tick_rate
        self.target_hz = target_hz
        self.output_dir = Path(output_dir)
        self.map_name = map_name
        self._downsampled: Dict[int, pd.DataFrame] = {}
        self._step = max(1, int(tick_rate / target_hz))

    def downsample(self) -> Dict[int, pd.DataFrame]:
        """Downsample all rounds to target Hz."""
        for rnum, rdf in self.rounds.items():
            if rdf.empty:
                continue
            # Sort by tick and take every Nth
            rdf = rdf.sort_values("tick").reset_index(drop=True)
            down = rdf.iloc[::self._step].copy()
            self._downsampled[rnum] = down
        return self._downsampled

    def normalize(self) -> Dict[int, pd.DataFrame]:
        """Apply normalization: angular encoding, relative coordinates."""
        for rnum, rdf in self._downsampled.items():
            rdf = self._normalize_frame(rdf)
            self._downsampled[rnum] = rdf
        return self._downsampled

    def _normalize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a single round DataFrame."""
        df = df.copy()

        # Angular encoding: yaw → cos/sin (in radians)
        if "yaw" in df.columns:
            yaw_rad = np.radians(df["yaw"].fillna(0).astype(float))
            df["yaw_cos"] = np.cos(yaw_rad)
            df["yaw_sin"] = np.sin(yaw_rad)
        else:
            # yaw not available — fill with zero vector
            df["yaw_cos"] = 1.0
            df["yaw_sin"] = 0.0

        # Relative coordinates: dx_to_c4, dy_to_c4
        # C4_ENTITY rows have player_name == "C4_ENTITY"
        has_c4 = "player_name" in df.columns and (df["player_name"] == "C4_ENTITY").any()
        if has_c4 and "x" in df.columns and "y" in df.columns:
            c4_rows = df[df["player_name"] == "C4_ENTITY"]
            c4_x = float(c4_rows["x"].iloc[0])
            c4_y = float(c4_rows["y"].iloc[0])
            df["dx_to_c4"] = df["x"].astype(float) - c4_x
            df["dy_to_c4"] = df["y"].astype(float) - c4_y
        else:
            # Relative to map center (or zero if no coords)
            df["dx_to_c4"] = df.get("x", 0.0) if "x" in df.columns else 0.0
            df["dy_to_c4"] = df.get("y", 0.0) if "y" in df.columns else 0.0

        # State mask
        if "health" in df.columns:
            df["state_mask"] = (df["health"] > 0).astype(int)
        else:
            df["state_mask"] = 1

        return df

    def export(self) -> List[Path]:
        """Export all rounds as Parquet files, return file paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        map_dir = self.output_dir / self.map_name
        map_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for rnum, rdf in self._downsampled.items():
            fpath = map_dir / f"round_{rnum:03d}.parquet"
            rdf.to_parquet(fpath, index=False, compression="snappy")
            paths.append(fpath)
        return paths

    def export_metadata(self) -> Path:
        """Write metadata.json with dataset statistics."""
        meta = {
            "map_name": self.map_name,
            "tick_rate": self.tick_rate,
            "target_hz": self.target_hz,
            "downsample_step": self._step,
            "num_rounds": len(self._downsampled),
            "round_stats": {},
            "feature_schema": DEFAULT_SCHEMA,
        }

        for rnum, rdf in self._downsampled.items():
            meta["round_stats"][str(rnum)] = {
                "num_ticks": len(rdf),
                "duration_sec": float(rdf["t_round"].max()) if "t_round" in rdf.columns else 0,
                "players": list(rdf["player_name"].unique()) if "player_name" in rdf.columns else [],
            }

        fpath = self.output_dir / self.map_name / "metadata.json"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return fpath

    def export_all(self) -> Dict[str, Any]:
        """Run full export pipeline: downsample → normalize → export."""
        self.downsample()
        self.normalize()
        parquet_paths = self.export()
        meta_path = self.export_metadata()
        return {
            "parquet_paths": parquet_paths,
            "metadata_path": meta_path,
            "num_rounds": len(self._downsampled),
        }
