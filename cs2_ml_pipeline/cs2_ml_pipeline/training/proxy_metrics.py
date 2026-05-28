"""
training/proxy_metrics.py — Generate proxy labels for "optimality" from raw logs.

Metrics derived from event graph and telemetry:
  - KAST: Kill/Assist/Survive/Trade per round
  - Trade Efficiency, Positional Safety, DeltaWinProb, ValueProxy
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


class ProxyMetricGenerator:
    """Generates proxy labels for training without human annotation."""

    def __init__(self, tick_rate: float = 64.0, window_sec: float = 3.0):
        self.tick_rate = tick_rate
        self.window_ticks = int(tick_rate * window_sec)

    def compute_kast_proxy(self, kills, assists, survived, traded):
        return 0.25 * (kills.float() + assists.float() + survived.float() + traded.float())

    def compute_trade_efficiency(self, teammate_death_tick, revenge_kill_tick):
        time_diff = (revenge_kill_tick - teammate_death_tick).abs()
        max_trade_ticks = int(3.0 * self.tick_rate)
        return (time_diff <= max_trade_ticks).float()

    def compute_safety_score(self, cover_score, exposure_time):
        return (cover_score * (1.0 - torch.tanh(exposure_time * 0.1))).clamp(0.0, 1.0)

    def compute_delta_win_prob(self, win_prob_before, win_prob_after):
        return (win_prob_after - win_prob_before).clamp(-1.0, 1.0)

    def compute_value_proxy(self, kast_proxy, delta_win, safety, trade_eff,
                            weights=(0.3, 0.3, 0.2, 0.2)):
        w_kast, w_dwin, w_safety, w_trade = weights
        value = (w_kast * (kast_proxy * 2 - 1) +
                 w_dwin * delta_win +
                 w_safety * (safety * 2 - 1) +
                 w_trade * (trade_eff * 2 - 1))
        return value.clamp(-1.0, 1.0)

    def temporal_smooth(self, signal, kernel_size=4):
        if kernel_size <= 1 or signal.shape[-1] < kernel_size:
            return signal
        kernel = torch.ones(1, 1, kernel_size, device=signal.device) / kernel_size
        original_shape = signal.shape
        signal_flat = signal.view(-1, 1, original_shape[-1])
        smoothed = torch.nn.functional.conv1d(signal_flat, kernel, padding=kernel_size // 2)
        return smoothed.view(original_shape)
