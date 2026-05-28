"""
Tests for inference/dashboard.py
"""

import sys
import os
import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.inference.dashboard import (
    CoachingDashboard, CoachingPayload, MockWebSocket,
)


class TestCoachingPayload:

    def test_creation(self):
        p = CoachingPayload(tick=100, score=0.7, color="yellow",
                           hint="test", death_prob=0.3, value=0.1)
        assert p.tick == 100
        assert p.color == "yellow"

    def test_to_dict(self):
        p = CoachingPayload(tick=1, score=0.5, color="yellow",
                           hint="ok", death_prob=0.2, value=0.0)
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["tick"] == 1
        assert d["score"] == 0.5

    def test_to_json(self):
        p = CoachingPayload(tick=1, score=0.5, color="green",
                           hint="good", death_prob=0.1, value=0.5)
        j = p.to_json()
        data = json.loads(j)
        assert data["color"] == "green"


class TestCoachingDashboard:

    @pytest.fixture
    def dash(self):
        return CoachingDashboard()

    def test_process_tick(self, dash):
        p = dash.process_tick(tick=0, score=0.85, death_prob=0.1, value=0.3)
        assert p.color == "green"
        assert "Good" in p.hint

    def test_process_tick_high_risk(self, dash):
        p = dash.process_tick(tick=1, score=0.3, death_prob=0.85, value=-0.5)
        assert p.color == "red"
        assert "High risk" in p.hint or "death" in p.hint.lower()

    def test_collect_feedback(self, dash):
        dash.process_tick(tick=0, score=0.8, death_prob=0.1, value=0.5)
        fb = dash.collect_feedback(tick=0, user_rating=1, comment="agreed")
        assert fb["rating"] == 1
        assert len(dash.feedback_log) == 1

    def test_feedback_stats(self, dash):
        dash.process_tick(0, 0.8, 0.1, 0.5)
        dash.process_tick(1, 0.3, 0.8, -0.5)
        dash.collect_feedback(0, 1)
        dash.collect_feedback(1, -1)
        stats = dash.get_feedback_stats()
        assert stats["total"] == 2
        assert stats["agree_pct"] == 50.0
        assert stats["disagree_pct"] == 50.0

    def test_feedback_stats_empty(self, dash):
        stats = dash.get_feedback_stats()
        assert stats["total"] == 0

    def test_get_history(self, dash):
        for i in range(10):
            dash.process_tick(tick=i, score=0.5, death_prob=0.1, value=0.0)
        hist = dash.get_history(last_n=5)
        assert len(hist) == 5

    def test_custom_hint(self, dash):
        p = dash.process_tick(tick=5, score=0.4, death_prob=0.2, value=0.0,
                              hint="Custom hint text")
        assert p.hint == "Custom hint text"


class TestMockWebSocket:

    def test_connect(self):
        ws = MockWebSocket()
        assert not ws.connected
        ws.accept()
        assert ws.connected

    def test_send_json(self):
        ws = MockWebSocket()
        ws.send_json({"key": "value"})
        assert len(ws.sent_messages) == 1
        assert ws.sent_messages[0]["key"] == "value"

    def test_close(self):
        ws = MockWebSocket()
        ws.accept()
        ws.close()
        assert not ws.connected
