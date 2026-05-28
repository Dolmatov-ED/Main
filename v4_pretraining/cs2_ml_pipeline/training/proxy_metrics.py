"""
training/proxy_metrics.py — Generate proxy labels for "optimality" from raw logs.

No hand-labeling needed. Metrics are derived from event graph and telemetry:
  - KAST: Kill/Assist/Survive/Trade per round
  - Trade Efficiency: time between teammate death and revenge kill
  - Positional Safety: cover_score weighted by exposure time
  - DeltaWinProb: win probability change attributed to tick
  - ValueProxy: combined [-1, 1] score for training ValueHead
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class ProxyMetricGenerator:
    """
    Generates proxy labels for training without human annotation.

    All metrics are in [-1, 1] or [0, 1] range.
    """

    def __init__(self, tick_rate: float = 64.0, window_sec: float = 3.0):
        self.tick_rate = tick_rate
        self.window_ticks = int(tick_rate * window_sec)

    def compute_kast_proxy(
        self,
        kills: torch.Tensor,
        assists: torch.Tensor,
        survived: torch.Tensor,
        traded: torch.Tensor,
    ) -> torch.Tensor:
        """
        KAST proxy: binary per-player per-round.
        Args:
            kills:    [..., S] binary kill mask
            assists:  [..., S] binary assist mask
            survived: [..., S] binary survive-at-tick mask
            traded:   [..., S] binary trade mask
        Returns:
            kast: [..., S] floating in [0, 1]
        """
        kast = 0.25 * (kills.float() + assists.float()
                       + survived.float() + traded.float())
        return kast

    def compute_trade_efficiency(
        self,
        teammate_death_tick: torch.Tensor,
        revenge_kill_tick: torch.Tensor,
    ) -> torch.Tensor:
        """
        Trade efficiency: +1 if kill within 3s of teammate death.
        Args:
            teammate_death_tick: [..., S] tick of teammate death
            revenge_kill_tick:   [..., S] tick of revenge kill
        Returns:
            efficiency: [..., S] in [0, 1]
        """
        time_diff = (revenge_kill_tick - teammate_death_tick).abs()
        max_trade_ticks = int(3.0 * self.tick_rate)  # 3 seconds
        return (time_diff <= max_trade_ticks).float()

    def compute_safety_score(
        self,
        cover_score: torch.Tensor,
        exposure_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Positional safety: cover_score weighted by inverse exposure.
        Args:
            cover_score:   [..., S] in [0, 1]
            exposure_time: [..., S] time exposed (higher = worse)
        Returns:
            safety: [..., S] in [0, 1]
        """
        safety = cover_score * (1.0 - torch.tanh(exposure_time * 0.1))
        return safety.clamp(0.0, 1.0)

    def compute_delta_win_prob(
        self,
        win_prob_before: torch.Tensor,
        win_prob_after: torch.Tensor,
    ) -> torch.Tensor:
        """
        Change in estimated win probability.
        Args:
            win_prob_before: [..., S] estimated before action
            win_prob_after:  [..., S] estimated after action
        Returns:
            delta: [..., S] in [-1, 1]
        """
        return (win_prob_after - win_prob_before).clamp(-1.0, 1.0)

    def compute_value_proxy(
        self,
        kast_proxy: torch.Tensor,
        delta_win: torch.Tensor,
        safety: torch.Tensor,
        trade_eff: torch.Tensor,
        weights: Tuple[float, ...] = (0.3, 0.3, 0.2, 0.2),
    ) -> torch.Tensor:
        """
        Combined value proxy for ValueHead training.
        Returns:
            value_proxy: [..., S] in [-1, 1]
        """
        w_kast, w_dwin, w_safety, w_trade = weights
        value = (
            w_kast * (kast_proxy * 2 - 1) +     # map [0,1] → [-1,1]
            w_dwin * delta_win +
            w_safety * (safety * 2 - 1) +
            w_trade * (trade_eff * 2 - 1)
        )
        return value.clamp(-1.0, 1.0)

    def temporal_smooth(
        self,
        signal: torch.Tensor,
        kernel_size: int = 4,
    ) -> torch.Tensor:
        """
        Apply temporal smoothing (EMA) to proxy signals.
        """
        if kernel_size <= 1 or signal.shape[-1] < kernel_size:
            return signal
        kernel = torch.ones(1, 1, kernel_size, device=signal.device) / kernel_size
        # Apply along last dimension
        original_shape = signal.shape
        signal_flat = signal.view(-1, 1, original_shape[-1])
        smoothed = torch.nn.functional.conv1d(
            signal_flat, kernel, padding=kernel_size // 2
        )
        return smoothed.view(original_shape)
