"""
Tests for tokenizer modules: projectors, events, hybrid, map_layers.
Uses mock tensor generators — no real .dem or Parquet files needed.
"""

import sys
import os
import tempfile
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.tokenizer.projectors import (
    ContinuousProjector,
    PositionProjector,
    OrientProjector,
    StateProjector,
    CoverProjector,
)
from cs2_ml_pipeline.tokenizer.events import (
    EventEmbedder,
    EVENT_VOCAB,
    ID_TO_EVENT,
    VOCAB_SIZE,
    EVENT_PRIORITY,
)
from cs2_ml_pipeline.tokenizer.hybrid import HybridTokenizer
from cs2_ml_pipeline.tokenizer.map_layers import MapLayerGenerator, MAP_SIZE, MAP_LIST
from cs2_ml_pipeline.mocks.mock_tokenizer import (
    BatchGenerator, generate_edge_case_batch, set_seed
)


# ── Projectors ────────────────────────────────────────────────────────

class TestProjectors:

    def test_continuous_projector_output_shape(self):
        """Output shape matches [B, S, d_model]."""
        proj = ContinuousProjector(in_dim=5, hidden_dim=16, d_model=32)
        x = torch.randn(2, 10, 5)
        out = proj(x)
        assert out.shape == (2, 10, 32)

    def test_continuous_projector_no_batch(self):
        """Works without batch dimension."""
        proj = ContinuousProjector(in_dim=3, hidden_dim=8, d_model=16)
        x = torch.randn(10, 3)
        out = proj(x)
        assert out.shape == (10, 16)

    def test_position_projector(self):
        proj = PositionProjector(d_model=64)
        x = torch.randn(4, 8, 5)
        out = proj(x)
        assert out.shape == (4, 8, 64)
        assert not torch.isnan(out).any()

    def test_orient_projector(self):
        proj = OrientProjector(d_model=64)
        x = torch.randn(4, 8, 3)
        out = proj(x)
        assert out.shape == (4, 8, 64)
        assert not torch.isinf(out).any()

    def test_state_projector(self):
        proj = StateProjector(d_model=64)
        x = torch.randn(4, 8, 3)
        out = proj(x)
        assert out.shape == (4, 8, 64)

    def test_cover_projector(self):
        proj = CoverProjector(d_model=64)
        x = torch.randn(4, 8, 1)
        out = proj(x)
        assert out.shape == (4, 8, 64)

    def test_projector_gradient_flow(self):
        """Gradients propagate through projector."""
        proj = ContinuousProjector(in_dim=5, hidden_dim=16, d_model=32)
        x = torch.randn(2, 10, 5, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_projector_weights_initialized(self):
        """Weights are not all zeros after init."""
        proj = PositionProjector(d_model=64)
        w = list(proj.parameters())[0]
        assert not torch.allclose(w, torch.zeros_like(w))


# ── Events ────────────────────────────────────────────────────────────

class TestEventEmbedder:

    def test_vocab_size(self):
        assert VOCAB_SIZE == 30
        assert len(EVENT_VOCAB) == 30

    def test_vocab_has_critical_events(self):
        for evt in ["KILL", "DEATH", "NULL", "PLANT_START", "DEFUSE_START",
                      "HE_THROW", "FLASH_THROW", "BUY_AWP", "ROUND_START"]:
            assert evt in EVENT_VOCAB, f"Missing event: {evt}"

    def test_vocab_roundtrip(self):
        """ID_TO_EVENT reverses EVENT_VOCAB."""
        for name, idx in EVENT_VOCAB.items():
            assert ID_TO_EVENT[idx] == name

    def test_embedder_output_shape(self):
        embedder = EventEmbedder(vocab_size=VOCAB_SIZE, d_model=64)
        ids = torch.randint(0, VOCAB_SIZE, (2, 10))
        out = embedder(ids)
        assert out.shape == (2, 10, 64)

    def test_embedder_padding_is_zero(self):
        embedder = EventEmbedder(vocab_size=VOCAB_SIZE, d_model=64)
        ids = torch.zeros(2, 10, dtype=torch.long)
        out = embedder(ids)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_embedder_no_batch(self):
        embedder = EventEmbedder(vocab_size=VOCAB_SIZE, d_model=64)
        ids = torch.randint(0, VOCAB_SIZE, (10,))
        out = embedder(ids)
        assert out.shape == (10, 64)

    def test_resolve_single_event(self):
        eid = EventEmbedder.resolve_event_id(["KILL"])
        assert eid == EVENT_VOCAB["KILL"]

    def test_resolve_empty_returns_null(self):
        eid = EventEmbedder.resolve_event_id([])
        assert eid == EVENT_VOCAB["NULL"]

    def test_resolve_composite(self):
        """FLASH + KILL → FLASH_AND_KILL composite token."""
        eid = EventEmbedder.resolve_event_id(["FLASH_THROW", "KILL"])
        # Composite should exist
        assert eid == EVENT_VOCAB["FLASH_AND_KILL"]

    def test_resolve_priority(self):
        """Higher priority event wins in conflict."""
        # KILL has prio 8, DAMAGE has prio 4
        eid = EventEmbedder.resolve_event_id(["DAMAGE", "KILL"])
        assert eid == EVENT_VOCAB["KILL"]

    def test_resolve_unknown_event(self):
        eid = EventEmbedder.resolve_event_id(["NONEXISTENT"])
        assert eid == EVENT_VOCAB["NULL"]

    def test_priority_coverage(self):
        """All vocab events have a priority."""
        for name in EVENT_VOCAB:
            assert name in EVENT_PRIORITY, f"Missing priority for {name}"


# ── Hybrid Tokenizer ──────────────────────────────────────────────────

class TestHybridTokenizer:

    @pytest.fixture
    def tokenizer(self):
        return HybridTokenizer(d_model=64, event_vocab_size=VOCAB_SIZE)

    @pytest.fixture
    def batch(self):
        gen = BatchGenerator(batch_size=4, seq_len=32)
        return gen.generate()

    def test_forward_output_shape(self, tokenizer, batch):
        out = tokenizer.forward(
            pos=batch["pos"],
            orient=batch["orient"],
            state=batch["state"],
            cover=batch["cover"],
            events=batch["events"],
        )
        assert out.shape == (4, 32, 64)

    def test_forward_no_batch(self, tokenizer):
        gen = BatchGenerator(batch_size=1, seq_len=16)
        b = gen.generate_no_batch()  # dict of [S, ...] tensors
        out = tokenizer.forward(
            pos=b["pos"].unsqueeze(0),
            orient=b["orient"].unsqueeze(0),
            state=b["state"].unsqueeze(0),
            cover=b["cover"].unsqueeze(0),
            events=b["events"].unsqueeze(0),
        )
        assert out.shape == (1, 16, 64)

    def test_forward_dict(self, tokenizer, batch):
        out = tokenizer.forward_dict(batch)
        assert out.shape == (4, 32, 64)

    def test_output_no_nan(self, tokenizer, batch):
        out = tokenizer.forward_dict(batch)
        assert not torch.isnan(out).any()

    def test_output_finite(self, tokenizer, batch):
        out = tokenizer.forward_dict(batch)
        assert torch.isfinite(out).all()

    def test_reproducibility(self, tokenizer):
        """Same input → same output (deterministic)."""
        set_seed(42)
        gen = BatchGenerator(seed=42)
        b1 = gen.generate(batch_size=2, seq_len=8)
        set_seed(42)
        gen2 = BatchGenerator(seed=42)
        b2 = gen2.generate(batch_size=2, seq_len=8)
        out1 = tokenizer.forward_dict(b1)
        out2 = tokenizer.forward_dict(b2)
        assert torch.allclose(out1, out2)

    def test_gradient_flow(self, tokenizer, batch):
        """Loss gradients propagate through all submodules."""
        batch = {k: v.clone().requires_grad_(v.dtype == torch.float32)
                 for k, v in batch.items()}
        out = tokenizer.forward_dict(batch)
        loss = out.sum()
        loss.backward()
        assert batch["pos"].grad is not None

    def test_edge_case_all_zeros(self, tokenizer):
        all_zeros, _, _ = generate_edge_case_batch()
        out = tokenizer.forward_dict(all_zeros)
        assert not torch.isnan(out).any()

    def test_edge_case_all_ones(self, tokenizer):
        _, all_ones, _ = generate_edge_case_batch()
        out = tokenizer.forward_dict(all_ones)
        assert torch.isfinite(out).all()

    def test_edge_case_dead_players(self, tokenizer):
        _, _, dead = generate_edge_case_batch()
        out = tokenizer.forward_dict(dead)
        assert out.shape == (2, 4, 64)

    def test_feature_shapes(self):
        shapes = HybridTokenizer.get_feature_shapes()
        assert shapes["pos"] == (5,)
        assert shapes["orient"] == (3,)
        assert shapes["state"] == (3,)
        assert shapes["cover"] == (1,)
        assert shapes["events"] == ()

    def test_custom_d_model(self):
        tokenizer = HybridTokenizer(d_model=128)
        gen = BatchGenerator(batch_size=1, seq_len=8)
        batch = gen.generate(batch_size=1, seq_len=8)
        out = tokenizer.forward_dict(batch)
        assert out.shape == (1, 8, 128)


# ── Map Layers ────────────────────────────────────────────────────────

class TestMapLayerGenerator:

    def test_generate_shape(self):
        gen = MapLayerGenerator(map_name="de_mirage")
        tensor = gen.generate()
        assert tensor.shape == (3, MAP_SIZE, MAP_SIZE)
        assert tensor.dtype == np.float32

    def test_generate_values_in_range(self):
        gen = MapLayerGenerator()
        tensor = gen.generate()
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_deterministic_per_map(self):
        """Same map name → same tensor."""
        gen1 = MapLayerGenerator(map_name="de_dust2")
        gen2 = MapLayerGenerator(map_name="de_dust2")
        t1 = gen1.generate()
        t2 = gen2.generate()
        assert np.allclose(t1, t2)

    def test_different_maps_different(self):
        """Different maps → different tensors."""
        gen1 = MapLayerGenerator(map_name="de_dust2")
        gen2 = MapLayerGenerator(map_name="de_nuke")
        t1 = gen1.generate()
        t2 = gen2.generate()
        assert not np.allclose(t1, t2)

    def test_save_and_load(self):
        gen = MapLayerGenerator(map_name="de_test")
        gen.generate()
        with tempfile.TemporaryDirectory() as tmp:
            fpath = os.path.join(tmp, "test_map.npy")
            gen.save(fpath)
            assert os.path.exists(fpath)

            gen2 = MapLayerGenerator(map_name="de_test")
            loaded = gen2.load(fpath)
            assert np.allclose(gen.tensor, loaded)

    def test_shape_property(self):
        gen = MapLayerGenerator(map_name="de_mirage", resolution=128)
        assert gen.shape == (3, 128, 128)
        gen.generate()
        assert gen.shape == (3, 128, 128)

    def test_map_list_has_expected(self):
        assert "de_mirage" in MAP_LIST
        assert "de_dust2" in MAP_LIST
        assert "de_nuke" in MAP_LIST
        assert len(MAP_LIST) == 8


# ── Integration: Tokenizer + Map Layers ───────────────────────────────

class TestIntegration:

    def test_tokenizer_with_map_tensor_as_cover(self):
        """MapLayerGenerator → cover channel → HybridTokenizer."""
        map_gen = MapLayerGenerator(map_name="de_mirage", resolution=8)
        map_tensor = map_gen.generate()  # [3, 8, 8]
        cover_channel = map_tensor[2]     # Cover: [8, 8]

        # Simulate tokenizer input
        tokenizer = HybridTokenizer(d_model=64)
        B, S = 2, 16

        pos = torch.randn(B, S, 5)
        orient = torch.randn(B, S, 3)
        state = torch.rand(B, S, 3)
        # Map cover to sequence (sample positions)
        cover_flat = torch.from_numpy(cover_channel.flatten()).float()
        cover_indices = torch.randint(0, 64, (B, S))
        cover = cover_flat[cover_indices].unsqueeze(-1)  # [B, S, 1]
        events = torch.zeros(B, S, dtype=torch.long)

        out = tokenizer.forward(pos, orient, state, cover, events)
        assert out.shape == (B, S, 64)
        assert not torch.isnan(out).any()
