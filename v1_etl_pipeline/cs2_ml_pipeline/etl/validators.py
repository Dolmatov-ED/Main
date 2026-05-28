"""
etl/validators.py — Sanity checks for exported Parquet data.
Ensures data integrity before feeding into Stage 2 (tokenization).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple


class DataValidator:
    """Validates exported Parquet data for correctness."""

    # CS2 map coordinate bounds (±5000 to be conservative)
    COORD_BOUNDS = (-5000, 5000)
    MAX_ROUND_DURATION_SEC = 300  # 5 minutes max

    def __init__(self, parquet_paths: List[Path]):
        self.parquet_paths = parquet_paths
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validations. Returns True if no errors."""
        self.errors.clear()
        self.warnings.clear()

        for fpath in self.parquet_paths:
            if not fpath.exists():
                self.errors.append(f"File not found: {fpath}")
                continue
            df = pd.read_parquet(fpath)
            self._check_monotonic_ticks(df, fpath)
            self._check_t_round_bounds(df, fpath)
            self._check_coordinate_bounds(df, fpath)
            self._check_nan_in_required(df, fpath)
            self._check_state_integrity(df, fpath)

        return len(self.errors) == 0

    def _check_monotonic_ticks(self, df: pd.DataFrame, fpath: Path) -> None:
        """Verify ticks are monotonically increasing within a round."""
        if "tick" not in df.columns:
            self.warnings.append(f"{fpath.name}: no 'tick' column")
            return
        diffs = df["tick"].diff().dropna()
        if (diffs < 0).any():
            self.errors.append(
                f"{fpath.name}: ticks are not monotonically increasing"
            )

    def _check_t_round_bounds(self, df: pd.DataFrame, fpath: Path) -> None:
        """Verify t_round is within valid range."""
        if "t_round" not in df.columns:
            self.warnings.append(f"{fpath.name}: no 't_round' column")
            return
        t_round = df["t_round"].dropna()
        if t_round.empty:
            self.errors.append(f"{fpath.name}: all t_round values are NaN")
            return
        if t_round.min() < 0:
            self.errors.append(f"{fpath.name}: negative t_round values")
        if t_round.max() > self.MAX_ROUND_DURATION_SEC:
            self.errors.append(
                f"{fpath.name}: t_round exceeds {self.MAX_ROUND_DURATION_SEC}s "
                f"(max={t_round.max():.1f})"
            )

    def _check_coordinate_bounds(self, df: pd.DataFrame, fpath: Path) -> None:
        """Verify x, y coordinates are within CS2 map bounds."""
        for col in ["x", "y", "dx_to_c4", "dy_to_c4"]:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if vals.empty:
                continue
            mn, mx = self.COORD_BOUNDS
            out_of_bounds = (vals < mn) | (vals > mx)
            if out_of_bounds.any():
                self.warnings.append(
                    f"{fpath.name}: {col} has {out_of_bounds.sum()} values "
                    f"outside [{mn}, {mx}]"
                )

    def _check_nan_in_required(self, df: pd.DataFrame, fpath: Path) -> None:
        """Verify no NaN in required columns (after forward-fill)."""
        required = ["tick", "x", "y", "health", "state_mask", "t_round"]
        for col in required:
            if col not in df.columns:
                continue
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                self.errors.append(
                    f"{fpath.name}: {nan_count} NaN values in required column '{col}'"
                )

    def _check_state_integrity(self, df: pd.DataFrame, fpath: Path) -> None:
        """Verify logical consistency of state data."""
        # state_mask should match health > 0
        if "state_mask" in df.columns and "health" in df.columns:
            mask_alive = (df["health"] > 0).astype(int)
            mismatch = (df["state_mask"] != mask_alive).sum()
            if mismatch > 0:
                self.warnings.append(
                    f"{fpath.name}: state_mask mismatches health in {mismatch} rows"
                )

    def get_report(self) -> Dict[str, Any]:
        """Return validation report."""
        return {
            "total_files": len(self.parquet_paths),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "error_details": self.errors[:20],
            "warning_details": self.warnings[:20],
            "passed": len(self.errors) == 0,
        }
