"""
Tests for models/map_ae.py — Map Autoencoder.
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cs2_ml_pipeline.models.map_ae import (
    MapEncoder, MapDecoder, MapAutoencoder,
)
from cs2_ml_pipeline.mocks.mock_models import (
    make_map_batch, set_seed, TEST_Z_DIM,
)


class TestMapEncoder:
    @pytest.fixture
    def encoder(self):
        return MapEncoder(in_channels=3, base_channels=8, z_dim=TEST_Z_DIM,
                          input_size=32)

    def test_output_shape(self, encoder):
        x = make_map_batch(batch_size=2, size=32)
        z_mu, z_logvar = encoder(x)
        assert z_mu.shape == (2, TEST_Z_DIM)
        assert z_logvar.shape == (2, TEST_Z_DIM)

    def test_gradient_flow(self, encoder):
        x = make_map_batch(batch_size=2, size=32)
        z_mu, z_logvar = encoder(x)
        loss = z_mu.sum() + z_logvar.sum()
        loss.backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None, f"No grad for {name}"


class TestMapDecoder:
    @pytest.fixture
    def decoder(self):
        return MapDecoder(out_channels=3, base_channels=8, z_dim=TEST_Z_DIM,
                          input_size=32)

    def test_output_shape(self, decoder):
        z = torch.randn(2, TEST_Z_DIM)
        recon = decoder(z)
        assert recon.shape == (2, 3, 32, 32)

    def test_output_range(self, decoder):
        z = torch.randn(2, TEST_Z_DIM)
        recon = decoder(z)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0

    def test_gradient_flow(self, decoder):
        z = torch.randn(2, TEST_Z_DIM, requires_grad=True)
        recon = decoder(z)
        loss = recon.sum()
        loss.backward()
        assert z.grad is not None


class TestMapAutoencoder:
    @pytest.fixture
    def ae(self):
        return MapAutoencoder(in_channels=3, base_channels=8, z_dim=TEST_Z_DIM,
                              input_size=32, beta=0.1)

    def test_forward(self, ae):
        x = make_map_batch(batch_size=2, size=32)
        out = ae.forward(x)
        assert "recon" in out
        assert "z_mu" in out
        assert "z_logvar" in out
        assert "z" in out
        assert out["recon"].shape == x.shape
        assert out["z"].shape == (2, TEST_Z_DIM)

    def test_encode(self, ae):
        ae.eval()  # disable batch norm training mode
        x = make_map_batch(batch_size=2, size=32)
        z_map = ae.encode(x)
        assert z_map.shape == (2, TEST_Z_DIM)

    def test_compute_loss(self, ae):
        x = make_map_batch(batch_size=2, size=32)
        out = ae.forward(x)
        losses = ae.compute_loss(x, out)
        assert "mse" in losses
        assert "kl" in losses
        assert "total" in losses
        assert losses["mse"] >= 0
        assert losses["total"] >= 0

    def test_training_step(self, ae):
        """Verify gradients flow through full AE."""
        x = make_map_batch(batch_size=2, size=32)
        out = ae.forward(x)
        losses = ae.compute_loss(x, out)
        losses["total"].backward()
        for name, p in ae.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_reproducibility(self, ae):
        ae.eval()
        set_seed(42)
        x1 = make_map_batch(batch_size=2, size=32)
        z1 = ae.encode(x1)
        set_seed(42)
        x2 = make_map_batch(batch_size=2, size=32)
        z2 = ae.encode(x2)
        assert torch.allclose(z1, z2)

    def test_deterministic_encode(self, ae):
        """encode() returns same result for same input."""
        ae.eval()
        x = make_map_batch(batch_size=2, size=32)
        z1 = ae.encode(x)
        z2 = ae.encode(x)
        assert torch.allclose(z1, z2)
