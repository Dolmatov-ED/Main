"""
inference/dashboard.py — WebSocket dashboard for real-time coaching.
"""

import json
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class CoachingPayload:
    tick: int
    score: float
    color: str
    hint: str
    death_prob: float
    value: float
    attention_map: Optional[List[List[float]]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "tick": self.tick, "score": self.score, "color": self.color,
            "hint": self.hint, "death_prob": self.death_prob, "value": self.value,
            "attention_map": self.attention_map, "timestamp": self.timestamp,
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class CoachingDashboard:
    """Coaching dashboard that aggregates model outputs into player-facing UI."""

    def __init__(self):
        self.history: List[CoachingPayload] = []
        self.feedback_log: List[Dict] = []

    def process_tick(self, tick, score, death_prob, value, hint="", attention_map=None):
        if score >= 0.8:
            color = "green"
        elif score >= 0.5:
            color = "yellow"
        else:
            color = "red"

        if not hint:
            if death_prob > 0.7:
                hint = f"High risk: {death_prob*100:.0f}% death probability"
            elif score < 0.5:
                hint = "Suboptimal positioning detected"
            else:
                hint = "Good position"

        payload = CoachingPayload(tick=tick, score=score, color=color, hint=hint,
                                  death_prob=death_prob, value=value, attention_map=attention_map)
        self.history.append(payload)
        return payload

    def collect_feedback(self, tick, user_rating, comment=""):
        entry = {
            "tick": tick, "rating": user_rating, "comment": comment,
            "timestamp": time.time(),
            "score_at_tick": self.history[-1].score if self.history else None,
        }
        self.feedback_log.append(entry)
        return entry

    def get_feedback_stats(self):
        ratings = [f["rating"] for f in self.feedback_log]
        if not ratings:
            return {"total": 0, "agree_pct": 0, "disagree_pct": 0}
        agree = sum(1 for r in ratings if r > 0)
        disagree = sum(1 for r in ratings if r < 0)
        total = len(ratings)
        return {"total": total, "agree_pct": agree / total * 100,
                "disagree_pct": disagree / total * 100}

    def get_history(self, last_n=50):
        return [p.to_dict() for p in self.history[-last_n:]]


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.sent_messages: List[Dict] = []
        self.connected: bool = False

    def accept(self):
        self.connected = True

    def send_json(self, data):
        self.sent_messages.append(data)

    def close(self):
        self.connected = False
