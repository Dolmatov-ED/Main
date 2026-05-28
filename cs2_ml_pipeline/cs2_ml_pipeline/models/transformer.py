"""
models/transformer.py — Autoregressive CS2 Transformer with RoPE, KV-cache, SwiGLU.

Core temporal model for predicting game dynamics.
Uses Rotary Position Embeddings (RoPE), causal masking,
SwiGLU feed-forward, and incremental KV-cache for streaming inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# ── RoPE ──────────────────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence positions."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, seq_len: int, offset: int = 0,
                device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(offset, offset + seq_len, device=device).float()
        freqs = self.freqs.to(device)
        theta = torch.outer(t, freqs)
        return torch.cos(theta), torch.sin(theta)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding. x: [B, n_heads, S, head_dim]."""
    d = x.shape[-1] // 2
    x_rot = x[..., :d]
    x_pass = x[..., d:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    x1 = x_rot * cos - torch.cat([-x_rot[..., d//2:], x_rot[..., :d//2]], dim=-1) * sin
    return torch.cat([x1, x_pass], dim=-1)


# ── SwiGLU FFN ────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ── Causal Attention ──────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and KV-cache support."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1,
                 max_seq_len: int = 4096):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None, offset=0):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(S, offset=offset, device=x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if S > 1 and kv_cache is None:
            causal = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            attn_weights = attn_weights + causal
        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out), new_cache


# ── Transformer Block ─────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single Transformer block: Attention + SwiGLU FFN with pre-layer-norm."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout=dropout)

    def forward(self, x, mask=None, kv_cache=None, offset=0):
        attn_out, new_cache = self.attn(self.ln1(x), mask, kv_cache, offset)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


# ── Full Transformer ──────────────────────────────────────────────────

class CS2Transformer(nn.Module):
    """Autoregressive Transformer for CS2 game sequences."""

    def __init__(self, d_model: int = 512, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, kv_caches=None, offset=0):
        new_caches = []
        hidden = x
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            hidden, new_cache = layer(hidden, mask, cache, offset)
            new_caches.append(new_cache)
        hidden = self.ln_final(hidden)
        return {"hidden_states": hidden, "kv_caches": new_caches}

    def init_kv_cache(self, batch_size=1, device=None):
        caches = []
        for _ in range(self.n_layers):
            k = torch.empty(batch_size, self.layers[0].attn.n_heads, 0,
                           self.layers[0].attn.head_dim, device=device)
            v = k.clone()
            caches.append((k, v))
        return caches

    def generate_one_step(self, token, kv_caches, offset):
        result = self.forward(token, kv_caches=kv_caches, offset=offset)
        return result["hidden_states"], result["kv_caches"]
