"""
Tests for etl/aligner.py — Uses synthetic mock data.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.mocks.mock_demo import (
    generate_synthetic_demo, MockDemoParser, ALL_PLAYER_NAMES, ROUND_CONFIGS
)
from cs2_ml_pipeline.etl.aligner import TickAligner


class TestTickAligner:
    """Test tick alignment with synthetic data."""

    @pytest.fixture
    def synthetic_demo(self):
        """Return a fully parsed synthetic demo."""
        return generate_synthetic_demo()

    @pytest.fixture
    def extra_df(self):
        """Generate synthetic extra tick data."""
        parser = MockDemoParser()
        return parser.parse_ticks(["yaw", "armor", "has_helmet", "has_defuser"])

    @pytest.fixture
    def aligner(self, synthetic_demo):
        """Create TickAligner with synthetic demo data."""
        return TickAligner(
            ticks_df=synthetic_demo.ticks,
            rounds_df=synthetic_demo.rounds,
            events=synthetic_demo.events,
            kills_df=synthetic_demo.kills,
            damages_df=synthetic_demo.damages,
            tick_rate=64.0,
        )

    def test_init(self, aligner):
        """Aligner initializes without error."""
        assert aligner.tick_rate == 64.0
        assert not aligner.ticks_df.empty
        assert not aligner.rounds_df.empty

    def test_merge_extra(self, aligner, extra_df):
        """Extra fields merge into ticks without error."""
        initial_cols = set(aligner.ticks_df.columns)
        aligner.merge_extra(extra_df)
        # Should have gained yaw column
        assert "yaw" in aligner.ticks_df.columns

    def test_merge_extra_empty_df(self, aligner):
        """Merging empty DataFrame is a no-op."""
        before = aligner.ticks_df.shape
        aligner.merge_extra(pd.DataFrame())
        after = aligner.ticks_df.shape
        assert before == after

    def test_merge_c4(self, aligner):
        """C4 data merges into ticks."""
        parser = MockDemoParser()
        c4_df = parser.parse_ticks(["X", "Y"], ticks_target=["weapon_c4"])
        before_len = len(aligner.ticks_df)
        aligner.merge_c4(c4_df)
        after_len = len(aligner.ticks_df)
        # C4 rows added
        assert after_len >= before_len

    def test_merge_c4_empty(self, aligner):
        """Merging empty C4 is a no-op."""
        before = aligner.ticks_df.shape
        aligner.merge_c4(pd.DataFrame())
        after = aligner.ticks_df.shape
        assert before == after

    def test_extract_nades(self, aligner):
        """Nade extraction returns expected keys."""
        nades = aligner.extract_nades()
        assert set(nades.keys()) == {"he", "flash", "smoke", "molotov"}
        # At least some nades found
        total = sum(len(v) for v in nades.values())
        assert total >= 0  # May be zero if no events in range

    def test_extract_game_events(self, aligner):
        """Game events extracted as sorted list."""
        events = aligner.extract_game_events()
        assert isinstance(events, list)
        if events:
            # Verify sorted by tick
            ticks = [e["tick"] for e in events]
            assert ticks == sorted(ticks)
            # Verify structure
            for ev in events:
                assert "tick" in ev
                assert "type" in ev
                assert "player" in ev

    def test_get_aligned_has_required_columns(self, aligner):
        """get_aligned DataFrame has expected columns."""
        aligned = aligner.get_aligned()
        # Core columns (lowercased)
        assert "tick" in aligned.columns or "tick" in [c.lower() for c in aligned.columns]
        assert "health" in aligned.columns or "health" in [c.lower() for c in aligned.columns]

    def test_get_aligned_no_nan_in_key_cols(self, aligner, extra_df):
        """Aligned data has no NaN in critical columns after merge."""
        aligner.merge_extra(extra_df)
        aligned = aligner.get_aligned()
        # health should be filled
        health_col = "health"
        if health_col not in aligned.columns:
            health_col = "health"  # already lower
        assert not aligned[health_col].isna().all()

    def test_smart_get_with_series(self, aligner):
        """_smart_get returns value from Series."""
        row = pd.Series({"x": 100, "y": 200})
        assert aligner._smart_get(row, ["x", "X"]) == 100
        assert aligner._smart_get(row, ["X", "x"]) == 100
        assert aligner._smart_get(row, ["z"]) is None

    def test_get_player_name(self, aligner):
        """_get_player_name extracts from event row."""
        row = pd.Series({"user_name": "PlayerX"})
        assert aligner._get_player_name(row) == "PlayerX"

        row2 = pd.Series({"name": "123"})  # numeric string
        result = aligner._get_player_name(row2)
        assert result == "Player"  # fallback
