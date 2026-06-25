"""Exponential Moving Average of trainable weights.

Only ``requires_grad`` parameters are shadowed, so a frozen backbone (e.g. the
ChemBERTa encoder in Model 2) costs no extra memory. Uses the timm-style warmup
schedule so the EMA tracks fast early and settles to ``decay`` later.
"""
from __future__ import annotations

import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        base = model.module if hasattr(model, "module") else model
        self.shadow = {
            n: p.data.detach().clone()
            for n, p in base.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        base = model.module if hasattr(model, "module") else model
        for n, p in base.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.data, alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, model) -> dict:
        base = model.module if hasattr(model, "module") else model
        backup = {}
        for n, p in base.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    @torch.no_grad()
    def swap_out(self, model, backup: dict) -> None:
        base = model.module if hasattr(model, "module") else model
        for n, p in base.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = sd["shadow"]
        self.num_updates = sd["num_updates"]
        self.decay = sd["decay"]
