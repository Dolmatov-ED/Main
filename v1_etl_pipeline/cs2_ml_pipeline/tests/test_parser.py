"""
Tests for etl/parser.py — Uses mocks for awpy and demoparser2.
"""

import sys
import os
import pytest
import pandas as pd

# Make mocks importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.mocks.mock_demo import (
    MockDemo, MockDemoParser, generate_synthetic_demo
)
from cs2_ml_pipeline.etl.parser import (
    CS2DemoParser, DemoParseError, TICK_FEATURES, EVENT_NAMES
)


class TestCS2DemoParser:
    """Test parser with mocked dependencies."""

    def test_init_with_empty_path(self):
        """Parser raises on empty demo_path."""
        with pytest.raises(ValueError, match="demo_path must not be empty"):
            CS2DemoParser("")

    def test_init_with_valid_path(self):
        """Parser accepts non-empty path."""
        parser = CS2DemoParser("test.dem")
        assert parser.demo_path == "test.dem"
        assert not parser.is_parsed

    def test_parse_with_mock(self, monkeypatch):
        """Parser uses mocked Demo and completes without error."""
        # Patch awpy.Demo to MockDemo
        import cs2_ml_pipeline.etl.parser as parser_mod
        monkeypatch.setattr(parser_mod, "Demo", MockDemo)
        monkeypatch.setattr(parser_mod, "DemoParser", MockDemoParser)

        p = CS2DemoParser("fake.dem")
        p.parse()
        assert p.is_parsed
        assert p.map_name is not None
        assert p.map_name != "unknown"

    def test_parse_extracts_map_name(self, monkeypatch):
        """Map name extracted from mock header."""
        import cs2_ml_pipeline.etl.parser as parser_mod
        monkeypatch.setattr(parser_mod, "Demo", MockDemo)
        monkeypatch.setattr(parser_mod, "DemoParser", MockDemoParser)

        p = CS2DemoParser("fake.dem")
        p.parse()
        # MockDemo uses MAP_NAME = "de_mirage"
        assert p.map_name == "de_mirage"

    def test_parse_without_awpy_raises(self, monkeypatch):
        """Parser raises DemoParseError when awpy is None."""
        import cs2_ml_pipeline.etl.parser as parser_mod
        monkeypatch.setattr(parser_mod, "Demo", None)

        p = CS2DemoParser("fake.dem")
        with pytest.raises(DemoParseError, match="awpy not installed"):
            p.parse()

    def test_tick_features_constant(self):
        """Verify TICK_FEATURES contains expected fields."""
        assert "tick" in TICK_FEATURES
        assert "x" in TICK_FEATURES
        assert "y" in TICK_FEATURES
        assert "health" in TICK_FEATURES
        assert "yaw" in TICK_FEATURES

    def test_event_names_coverage(self):
        """Verify EVENT_NAMES covers critical events."""
        critical = [
            "round_start", "round_end", "bomb_planted", "bomb_defused",
            "player_death", "player_hurt", "hegrenade_detonate",
        ]
        for evt in critical:
            assert evt in EVENT_NAMES, f"Missing critical event: {evt}"

    def test_get_extra_fields_without_parser(self, monkeypatch):
        """get_extra_fields returns empty DataFrame when no extra parser."""
        import cs2_ml_pipeline.etl.parser as parser_mod
        monkeypatch.setattr(parser_mod, "Demo", MockDemo)

        p = CS2DemoParser("fake.dem")
        p._demo = MockDemo()
        p._demo.parse()
        p._extra_parser = None
        result = p.get_extra_fields(["yaw"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty
