"""
inference/xai.py — Explainable AI for coaching dashboard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class XAIModule:
    """Explainability module for the CS2 coaching indicator."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def attention_rollout(self, hidden, layer_idx=-1):
        B, S, _ = hidden.shape
        attn_weights = None
        if hasattr(self.model, "layers"):
            layer = self.model.layers[layer_idx]
            if hasattr(layer, "attn") and hasattr(layer.attn, "last_attn_weights"):
                attn_weights = layer.attn.last_attn_weights
        if attn_weights is not None:
            return attn_weights.mean(dim=1).mean(dim=1)
        return torch.ones(B, S, device=hidden.device) / S

    def gradient_saliency(self, tokens, target_dim=0):
        tokens = tokens.clone().requires_grad_(True)
        self.model.train()
        output = self.model(tokens)
        hidden = output["hidden_states"]
        target = hidden[:, :, target_dim].sum()
        target.backward()
        saliency = tokens.grad.abs()
        self.model.eval()
        return saliency

    def log_likelihood_decay(self, log_probs, window=32):
        S = log_probs.shape[-1]
        if window >= S:
            window = max(1, S // 2)
        kernel = torch.ones(1, 1, window, device=log_probs.device) / window
        log_probs_1d = log_probs.view(1, 1, -1)
        smoothed = F.conv1d(log_probs_1d, kernel, padding=window // 2)
        return smoothed.view(-1)[:S]

    @staticmethod
    def score_to_color(score):
        if score >= 0.8:    return "green", "#00FF00"
        elif score >= 0.5:  return "yellow", "#FFFF00"
        else:               return "red", "#FF0000"

    @staticmethod
    def generate_hint(score, risk, map_zone="unknown"):
        if risk > 0.35:
            return f"Высокий риск гибели {risk*100:.0f}%. Смените позицию на более безопасную."
        elif risk > 0.20:
            return f"Умеренный риск {risk*100:.0f}%. Держитесь ближе к укрытию."
        elif score > 0.55:
            return f"Хорошая позиция. Контроль карты стабилен."
        elif score > 0.45:
            return f"Нейтральная фаза. Удерживайте позицию."
        else:
            return f"Неоптимально. {risk*100:.0f}% вероятность гибели, смените позицию."
