"""
etl/parser.py — Low-level .dem file parser.
Wraps awpy.Demo and demoparser2.DemoParser.
All actual parsing is mocked in tests; this module is the production interface.
"""

import sys
from typing import Optional, List, Dict, Any
import pandas as pd


# External dependencies (mocked in tests)
try:
    from awpy import Demo
except ImportError:
    Demo = None  # type: ignore

try:
    from demoparser2 import DemoParser
except ImportError:
    DemoParser = None  # type: ignore


TICK_FEATURES = [
    "tick", "name", "team", "is_alive",
    "x", "y", "z",
    "yaw", "pitch",
    "health", "armor",
    "has_helmet", "has_defuser",
    "money", "active_weapon",
]

EVENT_NAMES = [
    "round_start", "round_end", "round_officially_ended", "round_freeze_end",
    "bomb_planted", "bomb_defused", "bomb_exploded", "bomb_pickup", "bomb_dropped",
    "bomb_begindefuse", "bomb_abortdefuse", "bomb_beginplant", "bomb_abortplant",
    "hegrenade_detonate", "flashbang_detonate", "smokegrenade_detonate",
    "smokegrenade_expired", "inferno_startburn", "inferno_expire",
    "player_death", "player_hurt", "player_blind", "player_spawn",
    "weapon_fire", "item_pickup",
]


class DemoParseError(Exception):
    """Raised when demo parsing fails."""
    pass


class CS2DemoParser:
    """
    High-level parser for CS2 .dem files.
    Uses awpy for event graph, demoparser2 for raw tick data.
    """

    def __init__(self, demo_path: str):
        if not demo_path:
            raise ValueError("demo_path must not be empty")
        self.demo_path = demo_path
        self._demo = None          # awpy.Demo instance
        self._extra_parser = None  # demoparser2.DemoParser
        self.tick_rate: float = 64.0
        self.map_name: Optional[str] = None

    def parse(self) -> None:
        """Execute full parsing of the demo file."""
        self._parse_with_awpy()
        self._parse_extra_fields()

    def _parse_with_awpy(self) -> None:
        """Parse demo using awpy for event graph and basic ticks."""
        if Demo is None:
            raise DemoParseError(
                "awpy not installed. Install with: pip install awpy"
            )
        try:
            self._demo = Demo(self.demo_path, verbose=False)
            self._demo.parse(events=EVENT_NAMES)
        except Exception as e:
            raise DemoParseError(f"awpy parsing failed: {e}") from e

        # Extract map_name
        try:
            if hasattr(self._demo, "header"):
                if isinstance(self._demo.header, dict):
                    raw = self._demo.header.get("map_name", "")
                else:
                    raw = getattr(self._demo.header, "map_name", "")
                name = raw.split("/")[-1] if raw else ""
                # Strip .bsp extension
                if name.endswith(".bsp"):
                    name = name[:-4]
                self.map_name = name or "unknown"
        except Exception:
            self.map_name = "unknown"

    def _parse_extra_fields(self) -> None:
        """Extract additional fields via demoparser2 (yaw, armor, helmet, defuser)."""
        if DemoParser is None:
            return  # Soft-fail: extra fields optional
        try:
            self._extra_parser = DemoParser(self.demo_path)
        except Exception:
            self._extra_parser = None

    @property
    def demo(self):
        """Access the underlying awpy.Demo object."""
        if self._demo is None:
            raise DemoParseError("Demo not parsed. Call .parse() first.")
        return self._demo

    @property
    def extra_parser(self):
        """Access the underlying demoparser2.DemoParser."""
        return self._extra_parser

    @property
    def is_parsed(self) -> bool:
        return self._demo is not None

    def get_extra_fields(self, fields: List[str],
                         ticks_target: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract extra tick data via demoparser2."""
        if self._extra_parser is None:
            return pd.DataFrame()
        try:
            return self._extra_parser.parse_ticks(fields, ticks_target=ticks_target)
        except Exception:
            return pd.DataFrame()
