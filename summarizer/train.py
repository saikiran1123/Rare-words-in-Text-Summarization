"""Training: loss, LR warmup and the training loop.

Full-precision (fp32) training with gradient clipping, per-step LR warmup, and
validation-loss early stopping with best-checkpoint saving.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import Config


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def compute_loss(final_dist, tgt_out, pad_idx: int, unk_idx: int) -> torch.Tensor:
    """Masked NLL averaged over non-pad target tokens.

    Targets the distribution cannot represent (extended-OOV ids when the pointer
    is off, or ids >= EV) fall back to ``<unk>``.
    """
    B, T, EV = final_dist.shape
    log_dist = torch.log(final_dist.clamp(min=1e-10))

    tgt = tgt_out.clone()
    tgt[tgt >= EV] = unk_idx

    nll = -log_dist.gather(2, tgt.unsqueeze(2)).squeeze(2)
    pad_mask = tgt_out == pad_idx
    nll = nll.masked_fill(pad_mask, 0.0)
    denom = (~pad_mask).sum().float().clamp(min=1)
    return nll.sum() / denom


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def make_warmup_scheduler(optimizer, warmup_steps: int):
    """Linear warmup then constant LR (``LambdaLR``)."""

    def lr_lambda(step: int) -> float:
        return min((step + 1) / float(max(1, warmup_steps)), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Drives training, validation, early stopping and checkpointing."""

    def __init__(self, model, config: Config, pad_idx: int, unk_idx: int, device):
        self.model = model.to(device)
        self.config = config
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.device = device

        tcfg = config.training
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=tcfg.lr, betas=(0.9, 0.98), eps=1e-9
        )
        self.scheduler = make_warmup_scheduler(self.optimizer, tcfg.warmup_steps)

        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(tcfg.checkpoint_dir, config.checkpoint_name)

    def _move_batch(self, batch):
        src, src_ext, tgt_in, tgt_out, src_freq, max_oov = batch
        return (
            src.to(self.device),
            src_ext.to(self.device),
            tgt_in.to(self.device),
            tgt_out.to(self.device),
            src_freq.to(self.device),
            max_oov,
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        clip = self.config.training.grad_clip
        pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=True)
        for batch in pbar:
            src, src_ext, tgt_in, tgt_out, src_freq, max_oov = self._move_batch(batch)
            self.optimizer.zero_grad()
            final_dist = self.model(src, src_ext, tgt_in, src_freq, max_oov)
            loss = compute_loss(final_dist, tgt_out, self.pad_idx, self.unk_idx)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Val loss", leave=False)
        for batch in pbar:
            src, src_ext, tgt_in, tgt_out, src_freq, max_oov = self._move_batch(batch)
            final_dist = self.model(src, src_ext, tgt_in, src_freq, max_oov)
            loss = compute_loss(final_dist, tgt_out, self.pad_idx, self.unk_idx)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(1, len(loader))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, float]]:
        tcfg = self.config.training
        best_val = float("inf")
        patience_counter = 0
        history: List[Dict[str, float]] = []

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params:,}")
        print(f"=== Training (use_pointer={self.config.model.use_pointer}) -> {self.checkpoint_path} ===")

        for epoch in range(1, tcfg.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            current_lr = self.scheduler.get_last_lr()[0]
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(
                f"Epoch {epoch:02d}/{tcfg.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print("  -> best val loss improved; checkpoint saved")
            else:
                patience_counter += 1
                print(f"  -> no improvement ({patience_counter}/{tcfg.patience})")
                if patience_counter >= tcfg.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Best validation loss: {best_val:.4f}")
        return history

    def load_best(self) -> None:
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
