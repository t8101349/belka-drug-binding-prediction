"""BELKA small-molecule / protein binding prediction.

Two interchangeable model families that share the same data pipeline,
trainer, EMA, loss and inference code:

    * Model 1 — Multi-Scale CNN + FiLM Peri-LN  (from-scratch, generalises well to novel chemistry)
    * Model 2 — Frozen ChemBERTa + Cross-Attention fusion  (high-capacity, pretrained)

See README.md for the design rationale and the public/private leaderboard lesson.
"""

__version__ = "0.1.0"

from .losses import FocalLoss
from .ema import ModelEMA

__all__ = ["FocalLoss", "ModelEMA", "__version__"]
