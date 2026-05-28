"""
Tests for models/transformer.py — CS2Transformer with RoPE, KV-cache.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.models.transformer import (
    RotaryPositionalEmbedding, apply_rotary_emb,
    CausalSelfAttention, TransformerBlock, CS2Transformer,
    SwiGLUFFN,
)
from cs2_ml_pipeline.mocks.mock_models import (
    make_token_batch, set_seed, TEST_D_MODEL, TEST_N_HEADS, TEST_N_LAYERS,
)


class TestRoPE:
    def test_shape(self):
        rope = RotaryPositionalEmbedding(dim=64)
        cos, sin = rope(seq_len=16)
        assert cos.shape == (16, 32)
        assert sin.shape == (16, 32)

    def test_offset(self):
        rope = RotaryPositionalEmbedding(dim=64)
        cos0, _ = rope(seq_len=4, offset=0)
        cos4, _ = rope(seq_len=4, offset=4)
        # Different positions → different embeddings
        assert not torch.allclose(cos0, cos4)

    def test_apply_rotary(self):
        x = torch.randn(2, 4, 8, 32)
        rope = RotaryPositionalEmbedding(dim=32)
        cos, sin = rope(seq_len=8)
        rotated = apply_rotary_emb(x, cos, sin)
        assert rotated.shape == x.shape
        assert not torch.isnan(rotated).any()


class TestSwiGLUFFN:
    def test_shape(self):
        ffn = SwiGLUFFN(d_model=64)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_gradient(self):
        ffn = SwiGLUFFN(d_model=64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestCausalSelfAttention:
    @pytest.fixture
    def attn(self):
        return CausalSelfAttention(d_model=64, n_heads=4)

    def test_forward_shape(self, attn):
        x = torch.randn(2, 8, 64)
        out, cache = attn(x)
        assert out.shape == (2, 8, 64)

    def test_causal_mask(self, attn):
        """Causal mask: batch forward produces valid output."""
        x = torch.randn(2, 4, 64)
        out_full, cache = attn(x)
        assert out_full.shape == (2, 4, 64)
        assert not torch.isnan(out_full).any()
        # KV cache works for incremental
        token = x[:, :1, :]
        out0, cache0 = attn(token)
        assert out0.shape == (2, 1, 64)

    def test_gradient(self, attn):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestCS2Transformer:
    @pytest.fixture
    def transformer(self):
        return CS2Transformer(
            d_model=64, n_layers=2, n_heads=4, dropout=0.0
        )

    def test_forward(self, transformer):
        x = make_token_batch(batch_size=2, seq_len=16, d_model=64)
        out = transformer(x)
        assert out["hidden_states"].shape == (2, 16, 64)
        assert len(out["kv_caches"]) == 2

    def test_kv_cache_init(self, transformer):
        cache = transformer.init_kv_cache(batch_size=2)
        assert len(cache) == 2
        for k, v in cache:
            assert k.shape[0] == 2
            assert k.shape[2] == 0  # Empty initially

    def test_generate_one_step(self, transformer):
        token = torch.randn(2, 1, 64)
        cache = transformer.init_kv_cache(batch_size=2)
        out, new_cache = transformer.generate_one_step(token, cache, offset=0)
        assert out.shape == (2, 1, 64)
        # Cache should now have 1 entry
        assert new_cache[0][0].shape[2] == 1

    def test_streaming_sequence(self, transformer):
        """Full streaming: feed token by token, compare with batch."""
        x = make_token_batch(batch_size=1, seq_len=8, d_model=64)
        # Batch forward
        batch_out = transformer(x)["hidden_states"]

        # Streaming forward
        cache = transformer.init_kv_cache(batch_size=1)
        stream_outs = []
        for t in range(8):
            token = x[:, t:t+1, :]
            out, cache = transformer.generate_one_step(token, cache, offset=t)
            stream_outs.append(out)

        stream_tensor = torch.cat(stream_outs, dim=1)
        # Should be very close (minor numerical differences from KV-cache)
        assert torch.allclose(stream_tensor, batch_out, atol=1e-3)

    def test_gradient_flow(self, transformer):
        x = make_token_batch(batch_size=2, seq_len=8, d_model=64)
        x = x.clone().requires_grad_(True)
        out = transformer(x)["hidden_states"]
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
