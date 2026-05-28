"""
training/trainer.py — CS2 Trainer for pretraining and multi-task fine-tuning.

Implements three training phases:
  1. Autoregressive pretraining (next-token prediction)
  2. Multi-task fine-tuning (death, value heads)
  3. Contrastive fine-tuning (skill separation)

Uses teacher forcing, gradient clipping, mixed precision stub.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Any
from collections import defaultdict

# Import DeathHead for focal_loss in multi_task_step
try:
    from cs2_ml_pipeline.models.heads import DeathHead
except ImportError:
    DeathHead = None


class CS2Trainer:
    """
    Trainer for CS2 foundation model.

    Wraps a trainable model (e.g., CS2Transformer + heads) and
    provides training/validation loops with curriculum support.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device = torch.device("cpu"),
        clip_grad_norm: float = 1.0,
        label_smoothing: float = 0.0,
        contrastive_head: Optional[nn.Module] = None,
    ):
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

    def pretrain_step(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single pretraining step: next-token prediction with teacher forcing.

        Args:
            tokens: [B, S, d_model] input sequence
            mask:   [B, S] attention mask (1 = attend, 0 = ignore)
        Returns:
            dict with loss components
        """
        self.model.train()
        self.optimizer.zero_grad()

        B, S, D = tokens.shape

        # Teacher forcing: input is tokens[:, :-1, :], target is tokens[:, 1:, :]
        input_tokens = tokens[:, :-1, :]
        target_tokens = tokens[:, 1:, :]

        # Forward
        output = self.model(input_tokens)
        hidden = output["hidden_states"]  # [B, S-1, D]

        # Next-token prediction loss: cosine similarity
        pred = hidden  # model's head predicts next token
        pred_n = torch.nn.functional.normalize(pred, dim=-1)
        target_n = torch.nn.functional.normalize(target_tokens, dim=-1)
        ntp_loss = 1.0 - (pred_n * target_n).sum(dim=-1).mean()

        ntp_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.step += 1
        self.train_losses["ntp"].append(ntp_loss.item())
        return {"ntp_loss": ntp_loss.item()}

    def multi_task_step(
        self,
        tokens: torch.Tensor,
        death_targets: torch.Tensor,
        value_targets: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Multi-task fine-tuning step with death and value heads.

        Args:
            tokens:         [B, S, d_model]
            death_targets:  [B, S, 1] binary
            value_targets:  [B, S, 1] in [-1, 1]
            loss_weights:   dict of per-task weights
        Returns:
            dict with loss components
        """
        if loss_weights is None:
            loss_weights = {"next_token": 0.5, "death": 0.5, "value": 0.3}

        self.model.train()
        self.optimizer.zero_grad()

        B, S, D = tokens.shape

        # Forward (full sequence)
        input_tokens = tokens[:, :-1, :]
        output = self.model(input_tokens)
        hidden = output["hidden_states"]  # [B, S-1, D]

        # Next-token loss
        target_tokens = tokens[:, 1:, :]
        pred_n = torch.nn.functional.normalize(hidden, dim=-1)
        target_n = torch.nn.functional.normalize(target_tokens, dim=-1)
        loss_ntp = (1.0 - (pred_n * target_n).sum(dim=-1).mean())

        # Death loss (Focal) — apply to sequence
        death_head = getattr(self.model, "death_head", None)
        if death_head is not None:
            death_pred = death_head(hidden)
            death_t = death_targets[:, 1:, :]
            if DeathHead is not None:
                loss_death = DeathHead.focal_loss(death_pred, death_t)
            else:
                loss_death = torch.nn.functional.binary_cross_entropy(death_pred, death_t)
        else:
            loss_death = torch.tensor(0.0, device=self.device)

        # Value loss (Huber)
        value_head = getattr(self.model, "value_head", None)
        if value_head is not None:
            value_pred = value_head(hidden)
            value_t = value_targets[:, 1:, :]
            loss_value = torch.nn.functional.huber_loss(value_pred, value_t)
        else:
            loss_value = torch.tensor(0.0, device=self.device)

        total = (
            loss_weights.get("next_token", 0.5) * loss_ntp +
            loss_weights.get("death", 0.5) * loss_death +
            loss_weights.get("value", 0.3) * loss_value
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.step += 1
        losses = {
            "ntp": loss_ntp.item(),
            "death": loss_death.item(),
            "value": loss_value.item(),
            "total": total.item(),
        }
        for k, v in losses.items():
            self.train_losses[k].append(v)
        return losses

    def contrastive_step(
        self,
        tokens: torch.Tensor,
        value_targets: torch.Tensor,
        window_size: int = 16,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
    ) -> Dict[str, float]:
        """
        Contrastive step: pool hidden states into windows,
        classify as high-skill / low-skill based on value targets,
        apply InfoNCE loss.

        Args:
            tokens:         [B, S, d_model]
            value_targets:  [B, S, 1] in [-1, 1]
            window_size:    tokens per pooled window
            contrastive_weight: multiplier for the loss
            temperature:    InfoNCE temperature
        Returns:
            dict with 'contrastive' loss
        """
        if self.contrastive_head is None:
            return {"contrastive": 0.0}

        self.model.train()
        self.optimizer.zero_grad()

        B, S, D = tokens.shape
        input_tokens = tokens[:, :-1, :]
        output = self.model(input_tokens)
        hidden = output["hidden_states"]  # [B, S-1, D]

        # Split sequence into non-overlapping windows, pool each
        effective_S = hidden.shape[1]  # S-1
        n_windows = max(1, effective_S // window_size)
        win_hidden = hidden[:, :n_windows * window_size, :]
        win_hidden = win_hidden.view(B, n_windows, window_size, D)
        pooled = win_hidden.mean(dim=2)  # [B, n_windows, D]

        # Get corresponding value targets per window
        vt = value_targets[:, 1:, :]  # [B, S-1, 1]
        vt = vt[:, :n_windows * window_size, :]
        vt = vt.view(B, n_windows, window_size, 1).mean(dim=2).squeeze(-1)  # [B, n_windows]

        # Classify windows: avg value > 0.2 → high (1), < -0.2 → low (0)
        # Flatten all windows
        pooled_flat = pooled.reshape(B * n_windows, D)  # [N, D]
        vt_flat = vt.reshape(-1)                          # [N]

        # Filter windows with clear signal
        high_mask = vt_flat > 0.2
        low_mask = vt_flat < -0.2
        valid = high_mask | low_mask
        if valid.sum() < 4:
            return {"contrastive": 0.0}

        emb = pooled_flat[valid]
        labels = high_mask[valid].float()

        # Project through contrastive head
        emb_proj = self.contrastive_head(emb)  # [N_valid, proj_dim]

        # InfoNCE loss
        from cs2_ml_pipeline.training.contrastive import InfoNCELoss
        infonce = InfoNCELoss(temperature=temperature)
        loss_contrast = infonce(emb_proj, labels)

        total = contrastive_weight * loss_contrast
        if total.item() > 0:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        self.step += 1
        self.train_losses["contrastive"].append(loss_contrast.item())
        return {"contrastive": loss_contrast.item()}

    def validate(
        self,
        dataloader: DataLoader,
        max_batches: int = 10,
    ) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        total_losses = defaultdict(float)
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                tokens = batch["tokens"].to(self.device)
                B, S, D = tokens.shape

                input_tokens = tokens[:, :-1, :]
                output = self.model(input_tokens)
                hidden = output["hidden_states"]

                target_tokens = tokens[:, 1:, :]
                pred_n = torch.nn.functional.normalize(hidden, dim=-1)
                target_n = torch.nn.functional.normalize(target_tokens, dim=-1)
                ntp_loss = 1.0 - (pred_n * target_n).sum(dim=-1).mean()

                total_losses["ntp"] += ntp_loss.item()
                count += 1

        for k in total_losses:
            total_losses[k] /= max(count, 1)
            self.val_losses[k].append(total_losses[k])

        return dict(total_losses)

    def get_metrics(self) -> Dict[str, Any]:
        """Return training metrics summary."""
        return {
            "step": self.step,
            "train_losses": {k: v[-100:] for k, v in self.train_losses.items()},
            "val_losses": {k: v[-10:] for k, v in self.val_losses.items()},
        }
