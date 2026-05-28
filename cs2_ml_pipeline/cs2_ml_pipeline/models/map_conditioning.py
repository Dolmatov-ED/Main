"""
models/map_conditioning.py — Inject map geometry (z_map) into the Transformer.

Two strategies:
  1. [MAP] token — prepend z_map projected to d_model as first token
  2. Cross-Attention — query temporal tokens against map features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MapTokenInjector(nn.Module):
    """Projects z_map to d_model and prepends as [MAP] token."""

    def __init__(self, z_dim: int = 256, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(z_dim, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tokens, z_map):
        """tokens: [B, S, d_model], z_map: [B, z_dim] → [B, 1+S, d_model]"""
        map_token = self.proj(z_map)
        if map_token.shape[0] == 1 and tokens.shape[0] > 1:
            map_token = map_token.expand(tokens.shape[0], -1)
        elif map_token.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"Batch mismatch: z_map batch={map_token.shape[0]}, "
                f"tokens batch={tokens.shape[0]}"
            )
        map_token = map_token.unsqueeze(1)
        map_token = self.ln(map_token)
        return torch.cat([map_token, tokens], dim=1)


class CrossAttentionBlock(nn.Module):
    """Cross-attention: temporal hidden states query map features."""

    def __init__(self, z_dim: int = 256, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.map_kv_proj = nn.Linear(z_dim, d_model * 2, bias=False)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model * 2)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, hidden, z_map):
        B, S, _ = hidden.shape
        kv = self.map_kv_proj(z_map)
        kv = self.ln_kv(kv)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1).expand(-1, S, -1)
        v = v.unsqueeze(1).expand(-1, S, -1)
        q = self.ln_q(hidden)
        out, _ = self.attn(q, k, v)
        return out
