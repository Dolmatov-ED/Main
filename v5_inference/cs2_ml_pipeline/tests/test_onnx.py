"""
Tests for inference/onnx_export.py
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.inference.onnx_export import ONNXExporter, estimate_latency
from cs2_ml_pipeline.mocks.mock_inference import MockTransformerForStreaming


class TestONNXExporter:

    @pytest.fixture
    def model(self):
        return MockTransformerForStreaming(d_model=32)

    def test_export_mocked(self, model, tmp_path):
        """ONNX export via torch.onnx (mocked by model simplicity)."""
        exporter = ONNXExporter(model, d_model=32)
        fpath = tmp_path / "test.onnx"
        # Simple linear model should export cleanly
        result = exporter.export(str(fpath), batch_size=1, seq_len=8)
        assert os.path.exists(result)

    def test_validate_stub(self):
        """Validate stub returns success when onnxruntime missing."""
        result = ONNXExporter.validate_onnx(
            "fake.onnx",
            torch.randn(1, 4, 32),
            torch.randn(1, 4, 32),
        )
        assert result["passed"] is True


class TestEstimateLatency:

    def test_cpu_latency(self):
        model = MockTransformerForStreaming(d_model=32)
        x = torch.randn(1, 8, 32)
        stats = estimate_latency(model, x, num_warmup=2, num_runs=10)
        assert "mean_ms" in stats
        assert stats["mean_ms"] >= 0
        assert "p99_ms" in stats
