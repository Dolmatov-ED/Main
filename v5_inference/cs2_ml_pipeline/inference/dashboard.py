"""
inference/dashboard.py — WebSocket dashboard for real-time coaching.

Provides:
  - FastAPI app with WebSocket endpoint
  - Score streaming with color coding
  - Attention map overlay (stub)
  - Feedback collection endpoint

All network I/O is mocked in tests.
"""

import json
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field


@dataclass
class CoachingPayload:
    """Payload sent to dashboard client."""
    tick: int
    score: float
    color: str
    hint: str
    death_prob: float
    value: float
    attention_map: Optional[List[List[float]]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "score": self.score,
            "color": self.color,
            "hint": self.hint,
            "death_prob": self.death_prob,
            "value": self.value,
            "attention_map": self.attention_map,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class CoachingDashboard:
    """
    Coaching dashboard that aggregates model outputs into player-facing UI.

    In production, this connects to a FastAPI WebSocket.
    For v5, it provides the data transformation layer.
    """

    def __init__(self):
        self.history: List[CoachingPayload] = []
        self.feedback_log: List[Dict] = []

    def process_tick(
        self,
        tick: int,
        score: float,
        death_prob: float,
        value: float,
        hint: str = "",
        attention_map: Optional[List[List[float]]] = None,
    ) -> CoachingPayload:
        """Process a single tick and create dashboard payload."""
        # Color coding
        if score >= 0.8:
            color = "green"
        elif score >= 0.5:
            color = "yellow"
        else:
            color = "red"

        # Auto-generate hint if empty
        if not hint:
            if death_prob > 0.7:
                hint = f"High risk: {death_prob*100:.0f}% death probability"
            elif score < 0.5:
                hint = "Suboptimal positioning detected"
            else:
                hint = "Good position"

        payload = CoachingPayload(
            tick=tick,
            score=score,
            color=color,
            hint=hint,
            death_prob=death_prob,
            value=value,
            attention_map=attention_map,
        )
        self.history.append(payload)
        return payload

    def collect_feedback(
        self,
        tick: int,
        user_rating: int,  # +1 = agree, -1 = disagree
        comment: str = "",
    ) -> Dict:
        """Log user feedback for future DPO/RLHF."""
        entry = {
            "tick": tick,
            "rating": user_rating,
            "comment": comment,
            "timestamp": time.time(),
            "score_at_tick": self.history[-1].score if self.history else None,
        }
        self.feedback_log.append(entry)
        return entry

    def get_feedback_stats(self) -> Dict:
        """Aggregate feedback statistics."""
        ratings = [f["rating"] for f in self.feedback_log]
        if not ratings:
            return {"total": 0, "agree_pct": 0, "disagree_pct": 0}
        agree = sum(1 for r in ratings if r > 0)
        disagree = sum(1 for r in ratings if r < 0)
        total = len(ratings)
        return {
            "total": total,
            "agree_pct": agree / total * 100,
            "disagree_pct": disagree / total * 100,
        }

    def get_history(self, last_n: int = 50) -> List[Dict]:
        """Get recent history for client display."""
        return [p.to_dict() for p in self.history[-last_n:]]


# Mock WebSocket endpoint (FastAPI-based)
# In production, this would be a real WebSocket:
#
# @app.websocket("/ws/coach")
# async def coach_stream(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         payload = await inference_queue.get()
#         await websocket.send_json(payload.to_dict())


class MockWebSocket:
    """Mock WebSocket for testing dashboard communication."""

    def __init__(self):
        self.sent_messages: List[Dict] = []
        self.connected: bool = False

    def accept(self):
        self.connected = True

    def send_json(self, data: Dict):
        self.sent_messages.append(data)

    def close(self):
        self.connected = False
