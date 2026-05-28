"""
Tests for inference/streamer.py
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.inference.streamer import StreamingInferenceEngine
from cs2_ml_pipeline.mocks.mock_inference import (
    MockTransformerForStreaming, make_stream_tokens,
)


class TestStreamingInferenceEngine:

    @pytest.fixture
    def model(self):
        return MockTransformerForStreaming(d_model=32)

    @pytest.fixture
    def engine(self, model):
        return StreamingInferenceEngine(model, window_size=64)

    def test_init_state(self, engine):
        assert not engine.is_ready
        assert engine.buffer_size == 0
        assert engine.position == 0

    def test_prefill(self, engine):
        tokens = make_stream_tokens(seq_len=32, d_model=32)
        hidden = engine.prefill(tokens)
        assert hidden.shape == (1, 32, 32)
        assert engine.is_ready
        assert engine.position == 32

    def test_step_after_prefill(self, engine):
        tokens = make_stream_tokens(seq_len=8, d_model=32)
        engine.prefill(tokens[:, :4, :])
        tok = tokens[:, 4:5, :]
        hidden = engine.step(tok)
        assert hidden.shape == (1, 1, 32)

    def test_step_before_prefill_raises(self, engine):
        tok = torch.randn(1, 1, 32)
        with pytest.raises(RuntimeError, match="Must call prefill"):
            engine.step(tok)

    def test_reset(self, engine):
        tokens = make_stream_tokens(seq_len=8, d_model=32)
        engine.prefill(tokens)
        engine.reset()
        assert not engine.is_ready
        assert engine.buffer_size == 0

    def test_predict_stream(self, engine):
        tokens = make_stream_tokens(seq_len=40, d_model=32)
        outputs = engine.predict_stream(tokens, chunk_size=8)
        assert len(outputs) > 1

    def test_buffer_size_increases(self, engine):
        tokens = make_stream_tokens(seq_len=16, d_model=32)
        engine.prefill(tokens[:, :4, :])  # 4 tokens
        for t in range(4, 16):
            engine.step(tokens[:, t:t+1, :])
        assert engine.buffer_size == 12  # 16 - 4 prefill + stepping
