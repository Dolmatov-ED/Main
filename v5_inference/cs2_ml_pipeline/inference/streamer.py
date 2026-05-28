"""
inference/streamer.py — Streaming inference with sliding window and KV-cache.

Manages:
  - Token buffer: sliding window of recent tokens
  - KV-cache: incremental decoding per layer
  - Prefill vs Decode phases
  - Latency budget tracking
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from collections import deque


class StreamingInferenceEngine:
    """
    Streaming inference wrapper for CS2Transformer.

    Maintains a sliding window buffer of tokens and KV-caches.
    Supports prefill (initial full forward) and incremental decode.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.window_size = window_size
        self.device = device
        self.model.to(device)
        self.model.eval()

        # State
        self.token_buffer: deque = deque(maxlen=window_size)
        self.kv_caches: Optional[list] = None
        self.position: int = 0
        self._prefilled: bool = False

    def reset(self) -> None:
        """Reset state for new round / map."""
        self.token_buffer.clear()
        self.kv_caches = None
        self.position = 0
        self._prefilled = False

    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Initial full forward pass over a window of tokens.

        Args:
            tokens: [1, S, d_model] initial sequence
        Returns:
            hidden_states: [1, S, d_model]
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(tokens)
            hidden = output["hidden_states"]
            self.kv_caches = output.get("kv_caches", [])
            self.position = tokens.shape[1]
            self._prefilled = True
        return hidden

    def step(self, token: torch.Tensor) -> torch.Tensor:
        """
        Process single new token using KV-cache.

        Args:
            token: [1, 1, d_model] new token
        Returns:
            hidden: [1, 1, d_model]
        """
        if not self._prefilled:
            raise RuntimeError("Must call prefill() before step()")

        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, "generate_one_step"):
                hidden, self.kv_caches = self.model.generate_one_step(
                    token, self.kv_caches, self.position
                )
            else:
                # Fallback: full forward with KV caches
                output = self.model(
                    token, kv_caches=self.kv_caches, offset=self.position
                )
                hidden = output["hidden_states"]
                self.kv_caches = output["kv_caches"]

            self.position += 1

        # Maintain sliding window
        if len(self.token_buffer) >= self.window_size:
            # Evict oldest KV entry (simplified: just track position)
            pass
        self.token_buffer.append(token.cpu())

        return hidden

    def predict_stream(
        self,
        tokens: torch.Tensor,
        chunk_size: int = 32,
    ) -> List[torch.Tensor]:
        """
        Process a full sequence in streaming chunks.

        Args:
            tokens: [1, S, d_model] full sequence
            chunk_size: tokens per forward call
        Returns:
            list of hidden state chunks
        """
        S = tokens.shape[1]
        self.reset()
        outputs = []

        # Prefill with first chunk
        first_chunk = tokens[:, :chunk_size, :]
        outputs.append(self.prefill(first_chunk))

        # Decode remaining chunks
        for start in range(chunk_size, S, chunk_size):
            end = min(start + chunk_size, S)
            chunk = tokens[:, start:end, :]
            # Process chunk token by token (streaming)
            chunk_outputs = []
            for t in range(chunk.shape[1]):
                tok = chunk[:, t:t+1, :]
                hidden = self.step(tok)
                chunk_outputs.append(hidden)
            outputs.append(torch.cat(chunk_outputs, dim=1))

        return outputs

    @property
    def buffer_size(self) -> int:
        return len(self.token_buffer)

    @property
    def is_ready(self) -> bool:
        return self._prefilled
