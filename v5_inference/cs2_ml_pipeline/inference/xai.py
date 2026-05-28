"""
inference/xai.py — Explainable AI for coaching dashboard.

Provides:
  - Attention Rollout: which past positions influenced current score
  - Gradient Saliency: which features drove score change
  - Score Decay: log-likelihood tracking over time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class XAIModule:
    """
    Explainability module for the CS2 coaching indicator.

    Attaches to a trained Transformer model and extracts
    interpretable signals for the dashboard.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def attention_rollout(
        self,
        hidden: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Compute attention rollout: average attention weights across heads.

        Simplified: uses the last layer's attention if available,
        otherwise returns uniform weights.

        Args:
            hidden: [B, S, d_model] hidden states
            layer_idx: which layer to extract from
        Returns:
            rollout: [B, S] attention weight per position
        """
        B, S, _ = hidden.shape

        # Try to extract attention weights from model
        attn_weights = None
        if hasattr(self.model, "layers"):
            layer = self.model.layers[layer_idx]
            if hasattr(layer, "attn") and hasattr(layer.attn, "last_attn_weights"):
                attn_weights = layer.attn.last_attn_weights  # [B, H, S, S]

        if attn_weights is not None:
            # Average over heads, sum over query positions
            rollout = attn_weights.mean(dim=1).mean(dim=1)  # [B, S]
        else:
            # Fallback: uniform attention (all positions equally important)
            rollout = torch.ones(B, S, device=hidden.device) / S

        return rollout

    def gradient_saliency(
        self,
        tokens: torch.Tensor,
        target_dim: int = 0,
    ) -> torch.Tensor:
        """
        Compute input gradient saliency for a target output dimension.

        Args:
            tokens: [1, S, d_model] input tokens
            target_dim: which output dimension to explain
        Returns:
            saliency: [1, S, d_model] gradient magnitudes
        """
        tokens = tokens.clone().requires_grad_(True)

        self.model.train()  # enable grad
        output = self.model(tokens)
        hidden = output["hidden_states"]

        # Backprop from target
        target = hidden[:, :, target_dim].sum()
        target.backward()

        saliency = tokens.grad.abs()
        self.model.eval()
        return saliency

    def log_likelihood_decay(
        self,
        log_probs: torch.Tensor,
        window: int = 32,
    ) -> torch.Tensor:
        """
        Track log-likelihood decay over time.

        Args:
            log_probs: [S] log probabilities per tick
            window: EMA window size
        Returns:
            decay: [S] smoothed log-likelihood
        """
        S = log_probs.shape[-1]
        if window >= S:
            window = max(1, S // 2)

        kernel = torch.ones(1, 1, window, device=log_probs.device) / window
        log_probs_1d = log_probs.view(1, 1, -1)
        smoothed = F.conv1d(log_probs_1d, kernel, padding=window // 2)
        return smoothed.view(-1)[:S]

    @staticmethod
    def score_to_color(score: float) -> Tuple[str, str]:
        """
        Convert optimality score to dashboard color.

        Returns:
            (color_name, hex_code)
        """
        if score >= 0.8:
            return "green", "#00FF00"
        elif score >= 0.5:
            return "yellow", "#FFFF00"
        else:
            return "red", "#FF0000"

    @staticmethod
    def generate_hint(
        score: float,
        risk: float,
        map_zone: str = "unknown",
    ) -> str:
        """
        Generate textual coaching hint based on death risk and score.
        """
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