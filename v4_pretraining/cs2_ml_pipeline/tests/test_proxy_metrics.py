"""
Tests for training/proxy_metrics.py
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.training.proxy_metrics import ProxyMetricGenerator
from cs2_ml_pipeline.mocks.mock_training import make_proxy_batch


class TestProxyMetricGenerator:

    @pytest.fixture
    def gen(self):
        return ProxyMetricGenerator(tick_rate=64.0, window_sec=3.0)

    @pytest.fixture
    def batch(self):
        return make_proxy_batch(batch_size=2, seq_len=8)

    def test_kast_proxy_range(self, gen):
        k = torch.ones(4, 8)
        kast = gen.compute_kast_proxy(k, k, k, k)
        assert kast.shape == (4, 8)
        assert kast.min() >= 0 and kast.max() <= 1

    def test_kast_proxy_all_zero(self, gen):
        z = torch.zeros(4, 8)
        kast = gen.compute_kast_proxy(z, z, z, z)
        assert torch.allclose(kast, torch.zeros_like(kast))

    def test_trade_efficiency(self, gen):
        death = torch.tensor([100.0, 200.0, 300.0])
        kill = torch.tensor([102.0, 205.0, 500.0])
        eff = gen.compute_trade_efficiency(death, kill)
        # 100→102 diff=2 < 192 ticks (3s at 64Hz) → 1.0
        # 200→205 diff=5 → 1.0
        # 300→500 diff=200 > 192 → 0.0
        assert eff[0] == 1.0
        assert eff[1] == 1.0
        assert eff[2] == 0.0

    def test_safety_score_range(self, gen):
        cover = torch.rand(4, 8)
        exp = torch.rand(4, 8) * 30
        safety = gen.compute_safety_score(cover, exp)
        assert safety.shape == (4, 8)
        assert safety.min() >= 0 and safety.max() <= 1

    def test_delta_win_prob(self, gen):
        before = torch.tensor([0.3, 0.5, 0.9])
        after = torch.tensor([0.8, 0.5, 0.1])
        delta = gen.compute_delta_win_prob(before, after)
        assert delta[0] > 0  # win prob increased
        assert delta[1] == 0.0  # no change
        assert delta[2] < 0  # win prob decreased

    def test_value_proxy_range(self, gen):
        kast = torch.rand(4, 8)
        dwin = torch.rand(4, 8) * 2 - 1
        safety = torch.rand(4, 8)
        trade = torch.rand(4, 8)
        val = gen.compute_value_proxy(kast, dwin, safety, trade)
        assert val.shape == (4, 8)
        assert val.min() >= -1.0 and val.max() <= 1.0

    def test_temporal_smooth(self, gen):
        signal = torch.zeros(1, 20)
        signal[0, 10] = 1.0  # spike
        smoothed = gen.temporal_smooth(signal, kernel_size=5)
        # Smoothed should spread the spike
        assert smoothed[0, 10] < 1.0
        assert smoothed[0, 9] > 0.0  # neighbor affected

    def test_temporal_smooth_noop(self, gen):
        signal = torch.randn(1, 3)
        smoothed = gen.temporal_smooth(signal, kernel_size=1)
        assert torch.allclose(signal, smoothed)
