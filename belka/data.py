"""Data splitting, datasets and collate functions.

Important BELKA-specific notes
------------------------------
* The private leaderboard is dominated by *nonshare* building blocks (novel
  chemistry). A random split massively over-estimates generalisation, so we
  group by building block. ``bb_aware_split`` uses ``StratifiedGroupKFold`` on
  BB1 as a pragmatic default; for the strictest nonshare simulation hold out all
  three building blocks (see README).
* Always split *before* any RUS sub-sampling and keep the validation set at the
  natural class distribution, otherwise the validation AP is inflated.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


def bb_aware_split(
    parquet_path: str,
    n_splits: int = 5,
    fold: int = 0,
    seed: int = 42,
    group_col: str = "buildingblock1_smiles",
    label_col: str = "binds",
):
    """Group-aware stratified split. Returns (train_df, val_df) as pandas frames."""
    from sklearn.model_selection import StratifiedGroupKFold

    df = pl.read_parquet(parquet_path)
    n_null = df[group_col].null_count()
    if n_null:
        print(f"[warn] {group_col} has {n_null} nulls (external rows without BBs?) — "
              "they will all land in one group.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()
    X = np.zeros(len(y))

    splits = list(sgkf.split(X, y, groups))
    tr_idx, va_idx = splits[fold]
    train_df = df[tr_idx.tolist()].to_pandas()
    val_df = df[va_idx.tolist()].to_pandas()

    leak = set(train_df[group_col]) & set(val_df[group_col])
    print(f"[split] train={len(train_df):,} val={len(val_df):,} | {group_col} leakage={len(leak)}")
    print(f"[split] val positive ratio = {val_df[label_col].mean():.4%} "
          "(~6% => data already RUS'd, AP will be optimistic vs LB)")

    train_df = train_df.drop(columns=[group_col])
    val_df = val_df.drop(columns=[group_col])
    return train_df, val_df


class BELKAMoleculeDataset(Dataset):
    """Train/val dataset: returns raw SMILES + protein vector + label."""

    def __init__(self, df, prot_map: dict):
        self.smiles = df["molecule_smiles"].values
        self.prot_names = df["protein_name"].values
        self.labels = df["binds"].values
        self.prot_map = prot_map

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {
            "smiles": self.smiles[i],
            "prot_emb": torch.tensor(self.prot_map[self.prot_names[i]], dtype=torch.float32),
            "label": torch.tensor([self.labels[i]], dtype=torch.float32),
        }


class BELKATestDataset(Dataset):
    """Test dataset: carries the row id for the submission file."""

    def __init__(self, df, prot_map: dict):
        self.ids = df["id"].to_numpy()
        self.smiles = df["molecule_smiles"].to_numpy()
        self.prot_names = df["protein_name"].to_numpy()
        self.prot_map = prot_map

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, i: int):
        return {
            "id": self.ids[i],
            "smiles": self.smiles[i],
            "prot_emb": torch.tensor(self.prot_map[self.prot_names[i]], dtype=torch.float32),
        }


def make_collate(tokenizer, max_length: int = 160):
    """Collate for training/validation. Produces input_ids/attention_mask/prot_emb/labels."""

    def collate(batch):
        enc = tokenizer([b["smiles"] for b in batch], max_length=max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prot_emb": torch.stack([b["prot_emb"] for b in batch]),
            "labels": torch.stack([b["label"] for b in batch]),
        }

    return collate


def make_test_collate(tokenizer, max_length: int = 160):
    """Collate for inference. Keeps ids, drops labels."""

    def collate(batch):
        enc = tokenizer([b["smiles"] for b in batch], max_length=max_length)
        return {
            "ids": [b["id"] for b in batch],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prot_emb": torch.stack([b["prot_emb"] for b in batch]),
        }

    return collate
