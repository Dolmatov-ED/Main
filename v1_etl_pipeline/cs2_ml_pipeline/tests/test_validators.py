"""
Tests for etl/validators.py — Sanity checks on Parquet data.
"""

import sys
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.mocks.mock_demo import generate_synthetic_demo
from cs2_ml_pipeline.etl.aligner import TickAligner
from cs2_ml_pipeline.etl.segmenter import RoundSegmenter
from cs2_ml_pipeline.etl.exporter import TickExporter
from cs2_ml_pipeline.etl.validators import DataValidator


class TestDataValidator:
    """Test validator with synthetic exported data."""

    @pytest.fixture
    def exported_paths(self):
        """Export synthetic data and return Parquet paths."""
        demo = generate_synthetic_demo()
        aligner = TickAligner(
            ticks_df=demo.ticks,
            rounds_df=demo.rounds,
            events=demo.events,
            kills_df=demo.kills,
            tick_rate=64.0,
        )
        aligned = aligner.get_aligned()
        events = aligner.extract_game_events()
        segmenter = RoundSegmenter(aligned, demo.rounds, events, tick_rate=64.0)
        rounds = segmenter.segment()

        with tempfile.TemporaryDirectory() as tmp:
            exporter = TickExporter(
                rounds, tick_rate=64, target_hz=8,
                output_dir=tmp, map_name="de_test"
            )
            result = exporter.export_all()
            yield result["parquet_paths"]

    @pytest.fixture
    def temp_parquet_dir(self):
        """Temporary directory for manual Parquet creation."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_validate_valid_data_passes(self, exported_paths):
        """Validator passes on well-formed synthetic data."""
        validator = DataValidator(exported_paths)
        assert validator.validate_all(), f"Validation failed: {validator.errors}"

    def test_validate_nonexistent_file(self, temp_parquet_dir):
        """Validator reports error for missing file."""
        fake_path = temp_parquet_dir / "nonexistent.parquet"
        validator = DataValidator([fake_path])
        assert not validator.validate_all()
        report = validator.get_report()
        assert report["errors"] > 0

    def test_validate_non_monotonic(self, temp_parquet_dir):
        """Validator catches non-monotonic ticks."""
        df = pd.DataFrame({
            "tick": [10, 8, 12],  # Not monotonic
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "health": [100, 100, 100],
            "t_round": [0.1, 0.2, 0.3],
            "state_mask": [1, 1, 1],
        })
        fpath = temp_parquet_dir / "bad.parquet"
        df.to_parquet(fpath, index=False)

        validator = DataValidator([fpath])
        assert not validator.validate_all()
        assert any("monoton" in e.lower() for e in validator.errors)

    def test_validate_negative_t_round(self, temp_parquet_dir):
        """Validator catches negative t_round."""
        df = pd.DataFrame({
            "tick": [1, 2, 3],
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "health": [100, 100, 100],
            "t_round": [-0.5, 0.1, 0.2],
            "state_mask": [1, 1, 1],
        })
        fpath = temp_parquet_dir / "neg_time.parquet"
        df.to_parquet(fpath, index=False)

        validator = DataValidator([fpath])
        assert not validator.validate_all()
        assert any("negative" in e.lower() for e in validator.errors)

    def test_validate_nan_in_required(self, temp_parquet_dir):
        """Validator catches NaN in required columns."""
        df = pd.DataFrame({
            "tick": [1, 2, 3],
            "x": [0, 1, np.nan],
            "y": [0, 1, 2],
            "health": [100, 100, 100],
            "t_round": [0.1, 0.2, 0.3],
            "state_mask": [1, 1, 1],
        })
        fpath = temp_parquet_dir / "nan_data.parquet"
        df.to_parquet(fpath, index=False)

        validator = DataValidator([fpath])
        validator.validate_all()
        assert validator.errors  # Should have NaN errors

    def test_get_report_structure(self, exported_paths):
        """Validation report has expected structure."""
        validator = DataValidator(exported_paths)
        validator.validate_all()
        report = validator.get_report()
        assert "total_files" in report
        assert "errors" in report
        assert "warnings" in report
        assert "passed" in report
        assert "error_details" in report

    def test_validate_state_mask_mismatch(self, temp_parquet_dir):
        """Validator warns on state_mask vs health mismatch."""
        df = pd.DataFrame({
            "tick": [1, 2, 3],
            "x": [0, 1, 2],
            "y": [0, 1, 2],
            "health": [0, 100, 100],  # dead in first tick
            "t_round": [0.1, 0.2, 0.3],
            "state_mask": [1, 1, 1],  # but mask says alive
        })
        fpath = temp_parquet_dir / "mask_mismatch.parquet"
        df.to_parquet(fpath, index=False)

        validator = DataValidator([fpath])
        validator.validate_all()
        # state_mask mismatch is a warning, not error
        assert any("mismatch" in w.lower() for w in validator.warnings)

    def test_validate_empty_paths(self):
        """Validator with empty path list returns True."""
        validator = DataValidator([])
        assert validator.validate_all()
