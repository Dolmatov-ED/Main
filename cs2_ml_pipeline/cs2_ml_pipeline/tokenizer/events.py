"""
tokenizer/events.py — Discrete event vocabulary and embedding.

Maps game events (kill, death, plant, nade, buy, etc.) to integer IDs
and learns d_model-dimensional embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict


EVENT_VOCAB: Dict[str, int] = {
    "NULL": 0,
    "KILL": 1, "DEATH": 2, "DAMAGE": 3,
    "FLASH_AND_KILL": 4,
    "PLANT_START": 5, "PLANT_ABORT": 6,
    "DEFUSE_START": 7, "DEFUSE_ABORT": 8,
    "BOMB_PICKUP": 9, "BOMB_DROP": 10,
    "BOMB_EXPLODED": 11, "BOMB_DEFUSED_SUCC": 12,
    "HE_THROW": 13, "FLASH_THROW": 14,
    "SMOKE_THROW": 15, "MOLOTOV_THROW": 16,
    "NADE_DETONATE": 17,
    "JUMP": 18, "CROUCH": 19, "WALK": 20,
    "BUY_RIFLE": 21, "BUY_SMG": 22, "BUY_AWP": 23,
    "BUY_PISTOL": 24, "BUY_NADES": 25, "BUY_ARMOR": 26,
    "ROUND_START": 27, "ROUND_END": 28, "FREEZE_END": 29,
}

ID_TO_EVENT: Dict[int, str] = {v: k for k, v in EVENT_VOCAB.items()}
VOCAB_SIZE = max(EVENT_VOCAB.values()) + 1

EVENT_PRIORITY: Dict[str, int] = {
    "NULL": 0,
    "WALK": 1, "CROUCH": 1, "JUMP": 1,
    "BUY_RIFLE": 2, "BUY_SMG": 2, "BUY_AWP": 2, "BUY_PISTOL": 2,
    "BUY_NADES": 2, "BUY_ARMOR": 2,
    "HE_THROW": 3, "FLASH_THROW": 3, "SMOKE_THROW": 3, "MOLOTOV_THROW": 3,
    "NADE_DETONATE": 3,
    "DAMAGE": 4,
    "BOMB_DROP": 5, "BOMB_PICKUP": 5, "BOMB_EXPLODED": 5,
    "BOMB_DEFUSED_SUCC": 5,
    "PLANT_START": 6, "PLANT_ABORT": 6,
    "DEFUSE_START": 6, "DEFUSE_ABORT": 6,
    "FLASH_AND_KILL": 7,
    "KILL": 8, "DEATH": 8,
    "ROUND_START": 9, "ROUND_END": 9, "FREEZE_END": 9,
}


class EventEmbedder(nn.Module):
    """Learned embedding for discrete events."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 512, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self._init_weights()

    def forward(self, event_ids: torch.Tensor) -> torch.Tensor:
        """event_ids: [B, S] or [S] → [B, S, d_model] or [S, d_model]"""
        return self.embedding(event_ids.long())

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx] = 0.0

    @staticmethod
    def resolve_event_id(event_names: List[str]) -> int:
        """Resolve one or more event names to a single event ID."""
        if not event_names:
            return EVENT_VOCAB["NULL"]

        if len(event_names) > 1:
            composite = "_AND_".join(sorted(event_names))
            if composite in EVENT_VOCAB:
                return EVENT_VOCAB[composite]
            stripped = []
            for name in event_names:
                short = name
                for suffix in ["_THROW", "_DETONATE", "_SUCC"]:
                    if short.endswith(suffix):
                        short = short[:-len(suffix)]
                stripped.append(short)
            composite_stripped = "_AND_".join(sorted(stripped))
            if composite_stripped in EVENT_VOCAB:
                return EVENT_VOCAB[composite_stripped]

        best = "NULL"
        best_prio = -1
        for name in event_names:
            if name in EVENT_VOCAB:
                prio = EVENT_PRIORITY.get(name, 0)
                if prio > best_prio:
                    best = name
                    best_prio = prio
        return EVENT_VOCAB.get(best, 0)
