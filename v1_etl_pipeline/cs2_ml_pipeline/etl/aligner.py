"""
etl/aligner.py — Merge ticks, C4 positions, nade events, and kills
into a unified per-tick-per-player DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


# Event type constants
EVENT_TYPES = {
    "round_start": "round_start",
    "round_end": "round_end",
    "bomb_planted": "plant",
    "bomb_defused": "defuse",
    "bomb_exploded": "explode",
    "bomb_pickup": "bomb_pickup",
    "bomb_dropped": "bomb_drop",
    "player_death": "death",
    "weapon_fire": "weapon_fire",
    "item_pickup": "item_pickup",
}

NADE_KEYS = {
    "hegrenade_detonate": "he",
    "flashbang_detonate": "flash",
    "smokegrenade_detonate": "smoke",
    "inferno_startburn": "molotov",
}


class TickAligner:
    """
    Aligns ticks from awpy with extra fields from demoparser2,
    C4 positions, nade events, and kill data.
    """

    def __init__(
        self,
        ticks_df,
        rounds_df,
        events: Dict[str, Any],
        kills_df = None,
        damages_df = None,
        tick_rate: float = 64.0,
    ):
        self.ticks_df = ticks_df.copy() if isinstance(ticks_df, pd.DataFrame) else pd.DataFrame(ticks_df).copy()
        self.rounds_df = rounds_df.copy() if isinstance(rounds_df, pd.DataFrame) else pd.DataFrame(rounds_df).copy()
        self.events = events
        self.kills_df = kills_df.copy() if isinstance(kills_df, pd.DataFrame) else pd.DataFrame(kills_df).copy() if kills_df is not None else pd.DataFrame()
        self.damages_df = damages_df.copy() if isinstance(damages_df, pd.DataFrame) else pd.DataFrame(damages_df).copy() if damages_df is not None else pd.DataFrame()
        self.tick_rate = tick_rate
        self._aligned: Optional[pd.DataFrame] = None

    def merge_extra(
        self, extra_df: pd.DataFrame
    ) -> None:
        """Merge extra tick fields (yaw, armor, helmet, defuser) into ticks."""
        if extra_df.empty:
            return

        merge_cols = ["tick", "name"]
        extra_cols = [c for c in ["yaw", "armor", "has_helmet", "has_defuser"]
                      if c in extra_df.columns]
        if not extra_cols:
            return

        self.ticks_df = pd.merge(
            self.ticks_df,
            extra_df[merge_cols + extra_cols],
            on=merge_cols,
            how="left",
        )

    def merge_c4(self, c4_df: pd.DataFrame) -> None:
        """Attach C4 position to ticks (broadcast or join)."""
        if c4_df.empty:
            return

        # Ensure standard column names
        c4_clean = pd.DataFrame()
        c4_clean["tick"] = c4_df["tick"]
        for col, fallback in [("X", "x"), ("Y", "y"), ("Z", "z")]:
            if col in c4_df.columns:
                c4_clean[col.lower()] = c4_df[col]
            elif fallback in c4_df.columns:
                c4_clean[col.lower()] = c4_df[fallback]
            else:
                c4_clean[col.lower()] = 0.0

        c4_clean["player_name"] = "C4_ENTITY"
        c4_clean["team"] = "c4"
        c4_clean["health"] = 1

        # Forward-fill C4 position to all ticks
        self.ticks_df = pd.concat([self.ticks_df, c4_clean], ignore_index=True)

    def extract_nades(self) -> Dict[str, List[Tuple[float, float]]]:
        """Extract nade detonation positions from events."""
        nades: Dict[str, List[Tuple[float, float]]] = {
            "he": [], "flash": [], "smoke": [], "molotov": []
        }

        for event_name, nade_key in NADE_KEYS.items():
            if event_name in self.events:
                df = pd.DataFrame(self.events[event_name])
                if not df.empty:
                    for _, row in df.iterrows():
                        x = self._smart_get(row, ["x", "X", "entity_x"])
                        y = self._smart_get(row, ["y", "Y", "entity_y"])
                        if x is not None and y is not None:
                            try:
                                nades[nade_key].append((float(x), float(y)))
                            except (ValueError, TypeError):
                                pass
        return nades

    def extract_game_events(self) -> List[Dict]:
        """Convert raw events into structured game event list."""
        game_events: List[Dict] = []

        if isinstance(self.events, dict):
            for event_name, ev_type in EVENT_TYPES.items():
                if event_name in self.events:
                    df = pd.DataFrame(self.events[event_name])
                    if df.empty:
                        continue
                    for _, row in df.iterrows():
                        tick = self._smart_get(row, ["tick"])
                        name = self._get_player_name(row)
                        if tick is not None:
                            game_events.append({
                                "tick": int(tick),
                                "type": ev_type,
                                "player": name,
                                "text": f"[{ev_type.upper()}] {name}",
                                "duration": 0,
                            })
        return sorted(game_events, key=lambda e: e["tick"])

    def get_aligned(self) -> pd.DataFrame:
        """Return the aligned ticks DataFrame."""
        df = self.ticks_df.copy()

        # Standardize column names to lowercase strings
        # (awpy may return int or mixed-type column names)
        df.columns = [str(c).lower() for c in df.columns]

        # Fill missing standard columns
        defaults = {
            "armor": 0,
            "health": 100,
            "has_helmet": False,
            "has_defuser": False,
            "yaw": 0.0,
            "team": "unknown",
            "is_alive": True,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        # Ensure "tick" column exists (awpy may use various names)
        if "tick" not in df.columns:
            for alt in ["ticks", "game_tick", "framenumber", "tick_number",
                        "time", "frame", "tick_id", "ticknum", "ingame_tick"]:
                if alt in df.columns:
                    df.rename(columns={alt: "tick"}, inplace=True)
                    break
        if "tick" not in df.columns:
            df["tick"] = df.index  # fallback: row index as tick

        # Ensure numeric types
        for col in ["health", "armor", "yaw"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Convert bool columns
        for col in ["has_helmet", "has_defuser"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        return df

    @staticmethod
    def _smart_get(row, candidates: List[str], default=None):
        """Get value from row using first available column name."""
        if isinstance(row, pd.Series):
            for cand in candidates:
                if cand in row.index:
                    val = row[cand]
                    if pd.notna(val):
                        return val
        return default

    @staticmethod
    def _get_player_name(row) -> str:
        """Extract player name from event row."""
        for cand in ["user_name", "name", "attacker_name", "victim_name"]:
            if cand in row.index:
                val = row[cand]
                if isinstance(val, str) and val and not val.isdigit():
                    return val
        return "Player"