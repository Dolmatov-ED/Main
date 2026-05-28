"""
etl/segmenter.py — Segment aligned tick data into rounds.
Computes t_round, attaches round metadata, binds events to ticks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class RoundSegmenter:
    """
    Splits continuous tick stream into per-round DataFrames
    with relative time, metadata, and event bindings.
    """

    def __init__(
        self,
        aligned_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
        game_events: List[Dict],
        tick_rate: float = 64.0,
    ):
        self.aligned_df = aligned_df
        self.rounds_df = rounds_df
        self.game_events = game_events
        self.tick_rate = tick_rate

    def segment(self) -> Dict[int, pd.DataFrame]:
        """Return dict of round_number → round DataFrame."""
        rounds: Dict[int, pd.DataFrame] = {}

        # Standardize rounds columns
        rdf = self.rounds_df.copy()
        rdf.columns = [str(c).lower() for c in rdf.columns]

        start_col = self._find_col(rdf, ["freeze_end", "start", "round_start",
                                         "tick_start", "start_tick"])
        end_col = self._find_col(rdf, ["official_end", "end_official", "round_end",
                                       "tick_end", "end_tick", "game_end"])
        win_col = self._find_col(rdf, ["winner", "winning_team", "team_winner",
                                       "winner_side"])
        reason_col = self._find_col(rdf, ["reason", "win_reason", "end_reason",
                                          "round_end_reason"])

        if not start_col or not end_col:
            print(f"[!] RoundSegmenter: cannot find start/end columns in {list(rdf.columns)}")
            return rounds

        for idx, row in rdf.iterrows():
            round_num = idx + 1
            t_start = row.get(start_col)
            t_end = row.get(end_col)

            if pd.isna(t_start) or pd.isna(t_end):
                continue

            round_slice = self.aligned_df[
                (self.aligned_df["tick"] >= t_start) &
                (self.aligned_df["tick"] <= t_end)
            ].copy()

            if round_slice.empty:
                continue

            min_tick = round_slice["tick"].min()
            round_slice["t_round"] = (
                (round_slice["tick"] - min_tick) / self.tick_rate
            ).astype(float)

            round_slice["round_id"] = round_num
            round_slice["winner"] = str(row.get(win_col, "unknown")) if win_col else "unknown"
            round_slice["win_reason"] = str(row.get(reason_col, "unknown")) if reason_col else "unknown"

            round_events = [
                ev for ev in self.game_events
                if t_start <= ev["tick"] <= t_end
            ]
            round_slice.attrs["events"] = round_events
            round_slice.attrs["round_num"] = round_num
            round_slice.attrs["t_start"] = int(t_start)
            round_slice.attrs["t_end"] = int(t_end)

            rounds[round_num] = round_slice

        return rounds

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def get_economy_phase(t_round: float, round_duration: float) -> str:
        """Classify tick into economy phase."""
        ratio = t_round / max(round_duration, 0.1)
        if ratio < 0.15:
            return "freezetime"
        elif ratio < 0.30:
            return "buy"
        elif ratio < 0.85:
            return "midplay"
        else:
            return "endgame"
