"""
Tests for etl/segmenter.py — Round segmentation with synthetic data.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.mocks.mock_demo import generate_synthetic_demo
from cs2_ml_pipeline.etl.aligner import TickAligner
from cs2_ml_pipeline.etl.segmenter import RoundSegmenter


class TestRoundSegmenter:
    """Test round segmentation."""

    @pytest.fixture
    def demo_data(self):
        """Create fully aligned demo data."""
        demo = generate_synthetic_demo()
        aligner = TickAligner(
            ticks_df=demo.ticks,
            rounds_df=demo.rounds,
            events=demo.events,
            kills_df=demo.kills,
            tick_rate=64.0,
        )
        aligned = aligner.get_aligned()
        game_events = aligner.extract_game_events()
        return demo.rounds, aligned, game_events

    @pytest.fixture
    def segmenter(self, demo_data):
        """Create RoundSegmenter."""
        rounds_df, aligned, game_events = demo_data
        return RoundSegmenter(aligned, rounds_df, game_events, tick_rate=64.0)

    def test_segment_returns_dict(self, segmenter):
        """Segment returns dict of round DataFrames."""
        rounds = segmenter.segment()
        assert isinstance(rounds, dict)
        assert len(rounds) > 0, "Expected at least one round"

    def test_segment_has_t_round(self, segmenter):
        """Each round has t_round column."""
        rounds = segmenter.segment()
        for rnum, rdf in rounds.items():
            assert "t_round" in rdf.columns, f"Round {rnum} missing t_round"
            assert rdf["t_round"].min() >= 0, f"Round {rnum} has negative t_round"

    def test_segment_has_round_metadata(self, segmenter):
        """Each round has round_id, winner, win_reason."""
        rounds = segmenter.segment()
        for rnum, rdf in rounds.items():
            assert "round_id" in rdf.columns
            assert "winner" in rdf.columns
            assert "win_reason" in rdf.columns
            # round_id should match round number
            assert (rdf["round_id"] == rnum).all()

    def test_segment_attrs_preserved(self, segmenter):
        """Round attrs contain events and boundaries."""
        rounds = segmenter.segment()
        for rnum, rdf in rounds.items():
            assert hasattr(rdf, "attrs")
            assert "events" in rdf.attrs
            assert "round_num" in rdf.attrs
            assert "t_start" in rdf.attrs
            assert "t_end" in rdf.attrs
            assert rdf.attrs["t_start"] <= rdf.attrs["t_end"]

    def test_segment_with_empty_ticks(self):
        """Segmenter handles empty aligned DataFrame."""
        seg = RoundSegmenter(
            pd.DataFrame(), pd.DataFrame(), [], tick_rate=64.0
        )
        rounds = seg.segment()
        assert rounds == {}

    def test_segment_with_missing_cols(self):
        """Segmenter handles rounds_df without expected columns."""
        seg = RoundSegmenter(
            pd.DataFrame({"tick": [1, 2], "x": [0, 1]}),
            pd.DataFrame({"bad_col": [1]}),
            [],
        )
        rounds = seg.segment()
        assert rounds == {}  # No valid rounds

    def test_economy_phase_classification(self):
        """Economy phase classifier returns correct phases."""
        assert RoundSegmenter.get_economy_phase(5.0, 100.0) == "freezetime"
        assert RoundSegmenter.get_economy_phase(20.0, 100.0) == "buy"
        assert RoundSegmenter.get_economy_phase(50.0, 100.0) == "midplay"
        assert RoundSegmenter.get_economy_phase(90.0, 100.0) == "endgame"

    def test_segment_ticks_within_range(self, segmenter):
        """All ticks in a round are within the round boundaries."""
        rounds = segmenter.segment()
        for rnum, rdf in rounds.items():
            t_start = rdf.attrs["t_start"]
            t_end = rdf.attrs["t_end"]
            assert rdf["tick"].min() >= t_start
            assert rdf["tick"].max() <= t_end
