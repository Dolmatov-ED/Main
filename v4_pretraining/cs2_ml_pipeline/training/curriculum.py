"""
training/curriculum.py — Curriculum learning scheduler for progressive difficulty.

Phases:
  1. Freezetime + Buy (short windows, predictable)
  2. Midplay + Skirmishes (random encounters)
  3. Endgame + Clutch (high pressure, long dependencies)
  4. Full rounds (overlapping windows)

Scheduler controls: seq_len, dropout, loss weights.
"""

import torch
from enum import Enum
from typing import Dict, Optional


class CurriculumPhase(Enum):
    FREEZETIME = 0
    BUY = 1
    MIDPLAY = 2
    ENDGAME = 3
    FULL = 4


class CurriculumScheduler:
    """
    Progressive curriculum for CS2 training.

    Gradually increases sequence length, reduces teacher forcing ratio,
    and shifts loss weights from pretraining to task-specific.
    """

    def __init__(
        self,
        total_steps: int = 10000,
        seq_len_start: int = 64,
        seq_len_end: int = 1024,
        warmup_steps: int = 1000,
    ):
        self.total_steps = total_steps
        self.seq_len_start = seq_len_start
        self.seq_len_end = seq_len_end
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1

    def get_phase(self, step: Optional[int] = None) -> CurriculumPhase:
        """Get current curriculum phase."""
        s = step if step is not None else self.current_step
        progress = s / max(self.total_steps, 1)

        if progress < 0.15:
            return CurriculumPhase.FREEZETIME
        elif progress < 0.30:
            return CurriculumPhase.BUY
        elif progress < 0.70:
            return CurriculumPhase.MIDPLAY
        elif progress < 0.90:
            return CurriculumPhase.ENDGAME
        else:
            return CurriculumPhase.FULL

    def get_seq_len(self, step: Optional[int] = None) -> int:
        """Get current sequence length (linear ramp)."""
        s = step if step is not None else self.current_step
        progress = min(s / max(self.total_steps, 1), 1.0)
        seq = self.seq_len_start + int(
            (self.seq_len_end - self.seq_len_start) * progress
        )
        return seq

    def get_loss_weights(self, step: Optional[int] = None) -> Dict[str, float]:
        """Get loss weights for current phase."""
        phase = self.get_phase(step)

        base = {
            "next_token": 1.0,
            "death": 0.1,
            "value": 0.1,
            "contrastive": 0.0,
        }

        if phase == CurriculumPhase.FREEZETIME:
            base.update(next_token=1.0, death=0.0, value=0.0)
        elif phase == CurriculumPhase.BUY:
            base.update(next_token=1.0, death=0.1, value=0.1)
        elif phase == CurriculumPhase.MIDPLAY:
            base.update(next_token=0.7, death=0.5, value=0.3, contrastive=0.1)
        elif phase == CurriculumPhase.ENDGAME:
            base.update(next_token=0.5, death=0.8, value=0.5, contrastive=0.2)
        elif phase == CurriculumPhase.FULL:
            base.update(next_token=0.3, death=0.5, value=0.3, contrastive=0.2)

        return base

    def get_teacher_forcing_ratio(self, step: Optional[int] = None) -> float:
        """Teacher forcing ratio: 1.0 early, decays to 0.9."""
        s = step if step is not None else self.current_step
        progress = min(s / max(self.total_steps, 1), 1.0)
        return 1.0 - 0.1 * progress  # 1.0 → 0.9

    def get_dropout(self, step: Optional[int] = None) -> float:
        """Dropout rate: starts at 0.1, increases to 0.2 for regularization."""
        s = step if step is not None else self.current_step
        progress = min(s / max(self.total_steps, 1), 1.0)
        return 0.1 + 0.1 * progress  # 0.1 → 0.2
