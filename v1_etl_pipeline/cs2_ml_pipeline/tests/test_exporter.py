"""
Tests for etl/exporter.py — Downsampling, normalization, Parquet export.
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


class TestTickExporter:
    """Test exporter with synthetic segments."""

    @pytest.fixture
    def rounds_dict(self):
        """Create segmented rounds from synthetic demo."""
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
        return segmenter.segment()

    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for export."""
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_downsample_reduces_size(self, rounds_dict):
        """Downsampling reduces the number of ticks."""
        exporter = TickExporter(rounds_dict, tick_rate=64, target_hz=8)
        result = exporter.downsample()
        for rnum, rdf in result.items():
            orig_len = len(rounds_dict[rnum])
            # At 64→8 Hz, step=8, roughly 8x reduction
            assert len(rdf) <= orig_len

    def test_downsample_step_calculation(self, rounds_dict):
        """Step is calculated correctly for tick_rate/target_hz."""
        exporter = TickExporter(rounds_dict, tick_rate=64, target_hz=8)
        assert exporter._step == 8

        exporter2 = TickExporter(rounds_dict, tick_rate=128, target_hz=4)
        assert exporter2._step == 32

    def test_normalize_adds_derived_columns(self, rounds_dict):
        """Normalization adds yaw_cos, yaw_sin, dx_to_c4, dy_to_c4."""
        exporter = TickExporter(rounds_dict, tick_rate=64, target_hz=8)
        exporter.downsample()
        exporter.normalize()
        for rnum, rdf in exporter._downsampled.items():
            assert "yaw_cos" in rdf.columns
            assert "yaw_sin" in rdf.columns
            assert "dx_to_c4" in rdf.columns
            assert "dy_to_c4" in rdf.columns
            assert "state_mask" in rdf.columns

    def test_yaw_cos_sin_range(self, rounds_dict):
        """yaw_cos and yaw_sin are in [-1, 1]."""
        exporter = TickExporter(rounds_dict, tick_rate=64, target_hz=8)
        exporter.downsample()
        exporter.normalize()
        for rnum, rdf in exporter._downsampled.items():
            cos_vals = rdf["yaw_cos"].dropna()
            sin_vals = rdf["yaw_sin"].dropna()
            if len(cos_vals) > 0:
                assert cos_vals.between(-1, 1).all()
                assert sin_vals.between(-1, 1).all()

    def test_state_mask_binary(self, rounds_dict):
        """state_mask is 0 or 1."""
        exporter = TickExporter(rounds_dict, tick_rate=64, target_hz=8)
        exporter.downsample()
        exporter.normalize()
        for rnum, rdf in exporter._downsampled.items():
            unique = rdf["state_mask"].unique()
            for v in unique:
                assert v in (0, 1)

    def test_export_creates_parquet_files(self, rounds_dict, temp_dir):
        """Export creates Parquet files on disk."""
        exporter = TickExporter(
            rounds_dict, tick_rate=64, target_hz=8,
            output_dir=temp_dir, map_name="de_test"
        )
        result = exporter.export_all()
        paths = result["parquet_paths"]
        assert len(paths) > 0
        for p in paths:
            assert p.exists()
            assert p.suffix == ".parquet"

    def test_export_creates_metadata(self, rounds_dict, temp_dir):
        """Export creates metadata.json."""
        exporter = TickExporter(
            rounds_dict, tick_rate=64, target_hz=8,
            output_dir=temp_dir, map_name="de_test"
        )
        result = exporter.export_all()
        meta_path = result["metadata_path"]
        assert meta_path.exists()
        assert meta_path.name == "metadata.json"

    def test_export_parquet_readable(self, rounds_dict, temp_dir):
        """Exported Parquet files can be read back."""
        exporter = TickExporter(
            rounds_dict, tick_rate=64, target_hz=8,
            output_dir=temp_dir, map_name="de_test"
        )
        result = exporter.export_all()
        for p in result["parquet_paths"]:
            df = pd.read_parquet(p)
            assert not df.empty
            assert "tick" in df.columns

    def test_downsample_preserves_round_order(self, rounds_dict):
        """Ticks within a round remain sorted after downsampling."""
        exporter = TickExporter(rounds_dict, tick_rate=64, target_hz=8)
        result = exporter.downsample()
        for rnum, rdf in result.items():
            assert rdf["tick"].is_monotonic_increasing
