"""
training/trainer.py — CS2 Trainer for pretraining and multi-task fine-tuning.

Three training phases:
  1. Autoregressive pretraining (next-token prediction)
  2. Multi-task fine-tuning (death, value heads)
  3. Contrastive fine-tuning (skill separation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any
from collections import defaultdict

try:
    from cs2_ml_pipeline.models.heads import DeathHead
except ImportError:
    DeathHead = None


class CS2Trainer:
    """Trainer for CS2 foundation model."""

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,
                 device: torch.device = torch.device("cpu"),
                 clip_grad_norm: float = 1.0, label_smoothing: float = 0.0,
                 contrastive_head: Optional[nn.Module] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.label_smoothing = label_smoothing
        self.contrastive_head = contrastive_head
        self.model.to(device)
        self.train_losses: Dict[str, List[float]] = defaultdict(list)
        self.val_losses: Dict[str, List[float]] = defaultdict(list)
        self.step = 0

    def pretrain_step(self, tokens, mask=None):
        """Single pretraining step: next-token prediction with teacher forcing."""
        self.model.train()
        self.optimizer.zero_grad()
        B, S, D = tokens.shape
        input_tokens = tokens[:, :-1, :]
        target_tokens = tokens[:, 1:, :]
        output = self.model(input_tokens)
        hidden = output["hidden_states"]
        pred_n = torch.nn.functional.normalize(hidden, dim=-1)
        target_n = torch.nn.functional.normalize(target_tokens, dim=-1)
        ntp_loss = 1.0 - (pred_n * target_n).sum(dim=-1).mean()
        ntp_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.step += 1
        self.train_losses["ntp"].append(ntp_loss.item())
        return {"ntp_loss": ntp_loss.item()}

    def multi_task_step(self, tokens, death_targets, value_targets, loss_weights=None):
        """Multi-task fine-tuning step with death and value heads."""
        if loss_weights is None:
            loss_weights = {"next_token": 0.5, "death": 0.5, "value": 0.3}
        self.model.train()
        self.optimizer.zero_grad()
        B, S, D = tokens.shape
        input_tokens = tokens[:, :-1, :]
        output = self.model(input_tokens)
        hidden = output["hidden_states"]
        target_tokens = tokens[:, 1:, :]
        pred_n = torch.nn.functional.normalize(hidden, dim=-1)
        target_n = torch.nn.functional.normalize(target_tokens, dim=-1)
        loss_ntp = 1.0 - (pred_n * target_n).sum(dim=-1).mean()

        death_head = getattr(self.model, "death_head", None)
        if death_head is not None:
            death_pred = death_head(hidden)
            death_t = death_targets[:, 1:, :]
            loss_death = DeathHead.focal_loss(death_pred, death_t) if DeathHead else \
                torch.nn.functional.binary_cross_entropy(death_pred, death_t)
        else:
            loss_death = torch.tensor(0.0, device=self.device)

        value_head = getattr(self.model, "value_head", None)
        if value_head is not None:
            value_pred = value_head(hidden)
            value_t = value_targets[:, 1:, :]
            loss_value = torch.nn.functional.huber_loss(value_pred, value_t)
        else:
            loss_value = torch.tensor(0.0, device=self.device)

        total = (loss_weights.get("next_token", 0.5) * loss_ntp +
                 loss_weights.get("death", 0.5) * loss_death +
                 loss_weights.get("value", 0.3) * loss_value)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.step += 1
        losses = {"ntp": loss_ntp.item(), "death": loss_death.item(),
                  "value": loss_value.item(), "total": total.item()}
        for k, v in losses.items():
            self.train_losses[k].append(v)
        return losses

    def contrastive_step(self, tokens, value_targets, window_size=16,
                         contrastive_weight=0.1, temperature=0.07):
        """Contrastive step: pool hidden states, compare high/low skill windows."""
        if self.contrastive_head is None:
            return {"contrastive": 0.0}
        self.model.train()
        B, S, _ = tokens.shape
        w = window_size
        n_windows = S // w
        if n_windows < 2:
            return {"contrastive": 0.0}
        tokens_trimmed = tokens[:, :n_windows * w, :]
        tokens_win = tokens_trimmed.view(B * n_windows, w, -1)
        with torch.no_grad():
            hidden = self.model(tokens_win)["hidden_states"]
            pooled = hidden.mean(dim=1)
        projected = self.contrastive_head(pooled)
        value_trimmed = value_targets[:, :n_windows * w, :]
        value_win = value_trimmed.view(B * n_windows, w, 1).mean(dim=1)
        labels = (value_win.squeeze(-1) > 0).long()
        if labels.sum() < 2 or (1 - labels).sum() < 2:
            return {"contrastive": 0.0}
        emb = torch.nn.functional.normalize(projected, dim=-1)
        sim = torch.matmul(emb, emb.T) / temperature
        labels_2d = labels.view(-1, 1)
        pos_mask = (labels_2d == labels_2d.T).float()
        pos_mask = pos_mask - torch.eye(len(projected), device=projected.device)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(dim=-1)
        all_sum = exp_sim.sum(dim=-1) - torch.exp(sim.diag())
        all_sum = all_sum.clamp(min=1e-9)
        loss = -torch.log(pos_sum.clamp(min=1e-9) / all_sum).mean()
        self.train_losses["contrastive"].append(loss.item())
        return {"contrastive": loss.item()}
