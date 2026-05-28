"""
Tests for training/curriculum.py
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.training.curriculum import (
    CurriculumScheduler, CurriculumPhase,
)


class TestCurriculumScheduler:

    @pytest.fixture
    def scheduler(self):
        return CurriculumScheduler(
            total_steps=1000,
            seq_len_start=64,
            seq_len_end=1024,
        )

    def test_initial_phase(self, scheduler):
        assert scheduler.get_phase(0) == CurriculumPhase.FREEZETIME

    def test_mid_phase(self, scheduler):
        # step 500 of 1000 = 50% → MIDPLAY
        phase = scheduler.get_phase(500)
        assert phase == CurriculumPhase.MIDPLAY

    def test_final_phase(self, scheduler):
        assert scheduler.get_phase(950) == CurriculumPhase.FULL

    def test_seq_len_progression(self, scheduler):
        assert scheduler.get_seq_len(0) == 64
        assert scheduler.get_seq_len(500) > 64
        assert scheduler.get_seq_len(1000) == 1024

    def test_loss_weights_evolution(self, scheduler):
        early = scheduler.get_loss_weights(0)
        assert early["next_token"] == 1.0
        assert early["death"] == 0.0

        mid = scheduler.get_loss_weights(500)
        assert mid["next_token"] < early["next_token"]
        assert mid["death"] > early["death"]

    def test_teacher_forcing_decay(self, scheduler):
        assert scheduler.get_teacher_forcing_ratio(0) == 1.0
        assert scheduler.get_teacher_forcing_ratio(1000) == 0.9

    def test_dropout_increase(self, scheduler):
        assert scheduler.get_dropout(0) == 0.1
        assert scheduler.get_dropout(1000) == 0.2

    def test_step_increments(self, scheduler):
        assert scheduler.current_step == 0
        scheduler.step()
        assert scheduler.current_step == 1

    def test_phase_boundaries(self, scheduler):
        """Test at exact boundaries."""
        # 15% of 1000 = 150 → BUY starts at 150
        assert scheduler.get_phase(149) == CurriculumPhase.FREEZETIME
        assert scheduler.get_phase(150) == CurriculumPhase.BUY

        # 30% = 300 → MIDPLAY
        assert scheduler.get_phase(299) == CurriculumPhase.BUY
        assert scheduler.get_phase(300) == CurriculumPhase.MIDPLAY

        # 70% = 700 → ENDGAME
        assert scheduler.get_phase(699) == CurriculumPhase.MIDPLAY
        assert scheduler.get_phase(700) == CurriculumPhase.ENDGAME

        # 90% = 900 → FULL
        assert scheduler.get_phase(899) == CurriculumPhase.ENDGAME
        assert scheduler.get_phase(900) == CurriculumPhase.FULL

    def test_small_total_steps(self):
        """Scheduler works with very few steps."""
        s = CurriculumScheduler(total_steps=1)
        assert s.get_seq_len(0) == 64
        assert s.get_seq_len(1) == 1024
