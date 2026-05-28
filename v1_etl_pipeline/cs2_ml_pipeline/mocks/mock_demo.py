"""
Mock objects for external dependencies (awpy, demoparser2).
Generates synthetic but realistic CS2 demo data for isolated testing.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ── Constants ──────────────────────────────────────────────────────────

TICK_RATE = 64
MAP_NAME = "de_mirage"

PLAYER_NAMES_CT = [
    "CT_Player1", "CT_Player2", "CT_Player3", "CT_Player4", "CT_Player5"
]
PLAYER_NAMES_T = [
    "T_Player1", "T_Player2", "T_Player3", "T_Player4", "T_Player5"
]
ALL_PLAYER_NAMES = PLAYER_NAMES_CT + PLAYER_NAMES_T

MAP_BOUNDS = {"x_min": -3230, "x_max": 3230, "y_min": -1713, "y_max": 1713}

# Round configuration: (freeze_end_tick, official_end_tick, winner, reason)
ROUND_CONFIGS = [
    {"freeze_end": 500, "end_official": 3500, "game_end": 3550, "winner": "CT", "reason": "ct_killed"},
    {"freeze_end": 4000, "end_official": 7000, "game_end": 7050, "winner": "T", "reason": "bomb_exploded"},
    {"freeze_end": 7500, "end_official": 10500, "game_end": 10550, "winner": "CT", "reason": "bomb_defused"},
]


# ── Generator helpers ──────────────────────────────────────────────────

def _generate_player_trajectory(
    player_name: str,
    team: str,
    start_tick: int,
    end_tick: int,
    tick_step: int = 1,
) -> pd.DataFrame:
    """Generate synthetic player movement across ticks."""
    np.random.seed(hash(player_name) % (2**31))
    ticks = np.arange(start_tick, end_tick + 1, tick_step)
    n = len(ticks)

    # Start position per team side
    if team == "CT":
        base_x, base_y = -1500, 0
    else:
        base_x, base_y = 1500, 0

    # Random walk with drift
    x = base_x + np.cumsum(np.random.randn(n) * 15)
    y = base_y + np.cumsum(np.random.randn(n) * 15)
    z = np.full(n, 64.0) + np.cumsum(np.random.randn(n) * 3)
    yaw = np.cumsum(np.random.randn(n) * 5) % 360

    # Health starts at 100, some players die
    health = np.full(n, 100)
    armor = np.full(n, 100)
    has_helmet = np.full(n, True)
    has_defuser = np.full(n, False)
    if team == "CT":
        # One CT gets defuser
        if "Player1" in player_name:
            has_defuser = np.full(n, True)

    # Kill a random player mid-round
    kill_tick_idx = n // 2 + np.random.randint(-n//4, n//4)
    kill_tick_idx = max(0, min(n-1, kill_tick_idx))
    health[kill_tick_idx:] = 0

    return pd.DataFrame({
        "tick": ticks,
        "name": player_name,
        "X": x,
        "Y": y,
        "Z": z,
        "health": health,
        "armor": armor,
        "has_helmet": has_helmet,
        "has_defuser": has_defuser,
        "team": team,
        "is_alive": health > 0,
    })


def _generate_extra_ticks(players: List[str], start_tick: int, end_tick: int,
                          tick_step: int = 1) -> pd.DataFrame:
    """Generate extra tick data (yaw, armor, helmet, defuser)."""
    np.random.seed(42)
    ticks = np.arange(start_tick, end_tick + 1, tick_step)
    rows = []
    for t in ticks:
        for p in players:
            rows.append({
                "tick": t,
                "name": p,
                "yaw": np.random.uniform(0, 360),
                "armor": 100,
                "has_helmet": True,
                "has_defuser": "CT" in p and "Player1" in p,
            })
    return pd.DataFrame(rows)


def _generate_c4_ticks(start_tick: int, end_tick: int,
                       tick_step: int = 1) -> pd.DataFrame:
    """Generate C4 position data."""
    np.random.seed(99)
    ticks = np.arange(start_tick, end_tick + 1, tick_step)
    n = len(ticks)
    return pd.DataFrame({
        "tick": ticks,
        "X": np.full(n, 200.0),
        "Y": np.full(n, -100.0),
        "Z": np.zeros(n),
    })


def _generate_kills(round_configs: List[dict]) -> pd.DataFrame:
    """Generate kill events."""
    np.random.seed(77)
    rows = []
    kill_id = 1
    for rnd in round_configs:
        start = rnd["freeze_end"]
        end = rnd["end_official"]
        # 2-3 kills per round
        n_kills = np.random.randint(2, 4)
        kill_ticks = sorted(np.random.randint(start + 100, end - 100, n_kills))
        for kt in kill_ticks:
            attacker = np.random.choice(ALL_PLAYER_NAMES)
            victim = np.random.choice([p for p in ALL_PLAYER_NAMES if p != attacker])
            rows.append({
                "tick": kt,
                "attacker_name": attacker,
                "victim_name": victim,
                "attacker_x": np.random.uniform(-2000, 2000),
                "attacker_y": np.random.uniform(-1000, 1000),
                "victim_x": np.random.uniform(-2000, 2000),
                "victim_y": np.random.uniform(-1000, 1000),
                "weapon": np.random.choice(["ak47", "m4a1", "awp", "deagle"]),
                "headshot": np.random.choice([True, False]),
                "assister_name": "",
                "kill_id": kill_id,
            })
            kill_id += 1
    return pd.DataFrame(rows)


def _generate_damages(round_configs: List[dict]) -> pd.DataFrame:
    """Generate damage events."""
    np.random.seed(55)
    rows = []
    for rnd in round_configs:
        start = rnd["freeze_end"]
        end = rnd["end_official"]
        n_hits = np.random.randint(5, 12)
        for _ in range(n_hits):
            rows.append({
                "tick": np.random.randint(start, end),
                "attacker_name": np.random.choice(ALL_PLAYER_NAMES),
                "victim_name": np.random.choice(ALL_PLAYER_NAMES),
                "dmg_health": np.random.randint(20, 99),
                "weapon": np.random.choice(["ak47", "m4a1", "awp", "deagle"]),
            })
    return pd.DataFrame(rows)


def _generate_events(round_configs: List[dict]) -> Dict[str, pd.DataFrame]:
    """Generate game events dict."""
    np.random.seed(33)
    events = {}

    # Round events
    round_starts = []
    round_ends = []
    for rnd in round_configs:
        round_starts.append({"tick": rnd["freeze_end"] - 100})  # approximate
        round_ends.append({"tick": rnd["end_official"], "winner": rnd["winner"], "reason": rnd["reason"]})
    events["round_start"] = pd.DataFrame(round_starts)
    events["round_end"] = pd.DataFrame(round_ends)

    # Player deaths
    deaths = []
    for rnd in round_configs:
        start = rnd["freeze_end"]
        end = rnd["end_official"]
        death_tick = start + (end - start) // 2
        deaths.append({"tick": death_tick, "user_name": "T_Player3", "weapon": "awp"})
    events["player_death"] = pd.DataFrame(deaths)

    # Bomb events
    for rnd in round_configs:
        if "bomb" in rnd["reason"]:
            mid = (rnd["freeze_end"] + rnd["end_official"]) // 2
            events["bomb_planted"] = pd.DataFrame([{
                "tick": mid, "user_name": "T_Player1",
                "x": 200.0, "y": -100.0, "site": "A"
            }])
            if "defused" in rnd["reason"]:
                events["bomb_defused"] = pd.DataFrame([{
                    "tick": mid + 1000, "user_name": "CT_Player1"
                }])
            break

    # Nade events
    for evt_name, key in [("hegrenade_detonate", "he"),
                           ("flashbang_detonate", "flash"),
                           ("smokegrenade_detonate", "smoke"),
                           ("inferno_startburn", "molotov")]:
        rows = []
        for rnd in round_configs:
            start = rnd["freeze_end"]
            end = rnd["end_official"]
            for _ in range(np.random.randint(1, 3)):
                rows.append({
                    "tick": np.random.randint(start, end),
                    "x": np.random.uniform(-2000, 2000),
                    "y": np.random.uniform(-1000, 1000),
                })
        events[evt_name] = pd.DataFrame(rows)

    # Blind events
    blinds = []
    for rnd in round_configs:
        start = rnd["freeze_end"]
        end = rnd["end_official"]
        for _ in range(np.random.randint(1, 4)):
            blinds.append({
                "tick": np.random.randint(start, end),
                "user_name": np.random.choice(ALL_PLAYER_NAMES),
                "duration": np.random.uniform(0.5, 3.0),
            })
    events["player_blind"] = pd.DataFrame(blinds)

    return events


def _generate_rounds_df(round_configs: List[dict]) -> pd.DataFrame:
    """Generate rounds DataFrame matching awpy format."""
    rows = []
    for i, rnd in enumerate(round_configs):
        rows.append({
            "freeze_end": rnd["freeze_end"],
            "start": rnd["freeze_end"],
            "end": rnd["game_end"],
            "end_tick": rnd["game_end"],
            "official_end": rnd["end_official"],
            "winner": rnd["winner"],
            "winning_team": rnd["winner"],
            "reason": rnd["reason"],
            "win_reason": rnd["reason"],
            "round_num": i + 1,
        })
    return pd.DataFrame(rows)


# ── Mock Demo class ────────────────────────────────────────────────────

class MockDemo:
    """Mock of awpy.Demo that returns synthetic data."""

    def __init__(self, demo_path: str = "", verbose: bool = False):
        self.demo_path = demo_path
        self.map_name = MAP_NAME
        self.tick_rate = TICK_RATE
        self.header = MockHeader(self.map_name)
        self._ticks_df: Optional[pd.DataFrame] = None
        self._rounds_df: Optional[pd.DataFrame] = None
        self._events: Dict[str, pd.DataFrame] = {}
        self._kills_df: Optional[pd.DataFrame] = None
        self._damages_df: Optional[pd.DataFrame] = None
        self._flashes_df: Optional[pd.DataFrame] = None
        self._parsed = False

    def parse(self, events: Optional[List[str]] = None) -> None:
        """Simulate parsing a demo file."""
        all_ticks = []
        for rnd in ROUND_CONFIGS:
            for p in ALL_PLAYER_NAMES:
                team = "CT" if "CT" in p else "T"
                traj = _generate_player_trajectory(
                    p, team, rnd["freeze_end"], rnd["game_end"], tick_step=2
                )
                all_ticks.append(traj)
        self._ticks_df = pd.concat(all_ticks, ignore_index=True)
        self._rounds_df = _generate_rounds_df(ROUND_CONFIGS)
        self._events = _generate_events(ROUND_CONFIGS)
        self._kills_df = _generate_kills(ROUND_CONFIGS)
        self._damages_df = _generate_damages(ROUND_CONFIGS)
        self._parsed = True

    @property
    def ticks(self) -> pd.DataFrame:
        if self._ticks_df is None:
            raise ValueError("Demo not parsed. Call .parse() first.")
        return self._ticks_df

    @property
    def rounds(self) -> pd.DataFrame:
        if self._rounds_df is None:
            raise ValueError("Demo not parsed.")
        return self._rounds_df

    @property
    def events(self) -> Dict[str, pd.DataFrame]:
        if not self._parsed:
            raise ValueError("Demo not parsed.")
        return self._events

    @property
    def kills(self) -> pd.DataFrame:
        if self._kills_df is None:
            raise ValueError("Demo not parsed.")
        return self._kills_df

    @property
    def damages(self) -> pd.DataFrame:
        if self._damages_df is None:
            return pd.DataFrame()
        return self._damages_df

    @property
    def flashes(self) -> pd.DataFrame:
        if "player_blind" in self._events:
            return self._events["player_blind"]
        return pd.DataFrame()


class MockHeader:
    """Mock of demo header."""
    def __init__(self, map_name: str):
        self.map_name = f"maps/{map_name}.bsp"


# ── Mock DemoParser class ──────────────────────────────────────────────

class MockDemoParser:
    """Mock of demoparser2.DemoParser."""

    def __init__(self, demo_path: str = ""):
        self.demo_path = demo_path

    def parse_ticks(self, fields: List[str],
                    ticks_target: Optional[List[str]] = None) -> pd.DataFrame:
        """Return synthetic tick data for requested fields."""
        all_frames = []
        for rnd in ROUND_CONFIGS:
            extra = _generate_extra_ticks(
                ALL_PLAYER_NAMES, rnd["freeze_end"], rnd["game_end"],
                tick_step=2
            )
            all_frames.append(extra)

        df = pd.concat(all_frames, ignore_index=True)

        # If C4 requested
        if ticks_target and any("c4" in str(t).lower() or "weapon_c4" in str(t).lower()
                               for t in ticks_target):
            c4_frames = []
            for rnd in ROUND_CONFIGS:
                c4_frames.append(_generate_c4_ticks(
                    rnd["freeze_end"], rnd["game_end"], tick_step=2
                ))
            c4_df = pd.concat(c4_frames, ignore_index=True)
            # Merge
            c4_df["name"] = "C4_ENTITY"
            c4_df["team"] = "c4"
            c4_df["health"] = 1
            # Return combined or just C4 — here we return C4 if it's the target
            if ticks_target and any("c4" in str(t).lower() or "weapon_c4" in str(t).lower()
                                   for t in ticks_target):
                return c4_df

        # Filter to requested fields
        available = [c for c in fields if c in df.columns or c in ["X", "Y", "Z"]]
        if not available:
            available = list(df.columns)
        return df[["tick", "name"] + [c for c in available if c in df.columns]]


# ── Factory function ───────────────────────────────────────────────────

def generate_synthetic_demo() -> MockDemo:
    """Create and parse a complete synthetic demo for testing."""
    demo = MockDemo()
    demo.parse()
    return demo


def generate_synthetic_round_configs() -> List[dict]:
    """Return round configs used in synthetic data."""
    return ROUND_CONFIGS.copy()
