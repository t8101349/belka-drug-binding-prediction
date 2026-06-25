"""Focal loss for the extreme class imbalance.

Standard formulation: label smoothing is applied only to the BCE target, while
the focal modulating factor ``p_t`` and the ``alpha`` weighting use the *hard*
labels (they measure "confidence on the true class", which should not be softened).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1, 1).float()
        inputs = inputs.view(-1, 1).float()

        if self.label_smoothing > 0:
            smoothed = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed = targets

        bce = self.bce(inputs, smoothed)
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)            # hard labels
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # hard labels
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()
