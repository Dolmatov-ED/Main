"""
models/map_ae.py — Map Autoencoder for spatial encoding of CS2 maps.

Compresses [C=3, H=256, W=256] MapTensor → z_map ∈ ℝ^{z_dim} → reconstruct.
Uses β-VAE loss + optional auxiliary tasks (cover_pred, rotation_time).

The encoder produces a latent vector that encodes tactical geometry:
height profile, walkability, and cover density — without memorizing map names.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MapEncoder(nn.Module):
    """CNN encoder: MapTensor [C, H, W] → z_map [batch, z_dim]."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        z_dim: int = 256,
        input_size: int = 256,
    ):
        super().__init__()
        self.z_dim = z_dim

        # 256 → 128 → 64 → 32 → 16 → 8
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),  # 128
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 64
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 32
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),  # 16
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),  # 8
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU(),
        )

        # Flatten → z_mean, z_logvar
        conv_out_size = input_size // 32  # after 5 stride-2 layers
        self.flat_dim = base_channels * 8 * conv_out_size * conv_out_size

        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] map tensor
        Returns:
            z_mean: [B, z_dim], z_logvar: [B, z_dim]
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z_mean = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar


class MapDecoder(nn.Module):
    """CNN decoder: z_map [B, z_dim] → reconstructed MapTensor [B, C, H, W]."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 32,
        z_dim: int = 256,
        input_size: int = 256,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.input_size = input_size
        self.init_size = input_size // 32

        # z_dim → flat → reshape
        self.flat_dim = base_channels * 8 * self.init_size * self.init_size
        self.fc = nn.Linear(z_dim, self.flat_dim)

        # 8 → 16 → 32 → 64 → 128 → 256
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 4, 2, 1),  # 16
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),  # 32
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # 64
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # 128
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, out_channels, 4, 2, 1),  # 256
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim] latent
        Returns:
            reconstruction: [B, C, H, W]
        """
        h = self.fc(z)
        h = h.view(h.size(0), self.base_channels * 8, self.init_size, self.init_size)
        return self.deconv(h)


class MapAutoencoder(nn.Module):
    """
    β-VAE for map layer compression.

    Loss:
      L = MSE(recon, input) + β * KL(N(z_mean, z_logvar) || N(0,1))
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        z_dim: int = 256,
        input_size: int = 256,
        beta: float = 0.1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta

        self.encoder = MapEncoder(in_channels, base_channels, z_dim, input_size)
        self.decoder = MapDecoder(in_channels, base_channels, z_dim, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode map → z_map (deterministic, no sampling)."""
        z_mu, _ = self.encoder(x)
        return z_mu

    def reparameterize(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from N(z_mu, z_logvar) using reparameterization trick."""
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] map tensor
        Returns:
            dict with recon, z_mu, z_logvar, z (sampled)
        """
        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)
        recon = self.decoder(z)
        return {
            "recon": recon,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        """
        # Reconstruction loss
        mse = F.mse_loss(output["recon"], x, reduction="mean")

        # KL divergence
        kl = -0.5 * torch.mean(1 + output["z_logvar"]
                               - output["z_mu"].pow(2)
                               - output["z_logvar"].exp())

        total = mse + self.beta * kl
        return {"mse": mse, "kl": kl, "total": total}
