"""Molecule tokenizers with a uniform interface.

Both tokenizers are callable as ``tok(smiles_list, max_length) -> dict`` returning
``{"input_ids": LongTensor, "attention_mask": LongTensor}`` so the data pipeline
and the two models can be swapped freely.

    * SmartSmilesTokenizer  — character-level for Model 1 (CNN). Fixes the classic
      bugs: <UNK> never collapses into <PAD>, and multi-char atoms (Cl, Br, [Dy])
      stay as single tokens.
    * HFMoleculeTokenizer    — wraps a HuggingFace BPE tokenizer for Model 2 (ChemBERTa).
"""
from __future__ import annotations

import re

import torch


class SmartSmilesTokenizer:
    """Regex-based character-level SMILES tokenizer (id 0 = PAD, 1 = UNK)."""

    PATTERN = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Se|se|%\d{2}|.)")

    def __init__(self) -> None:
        specials = ["<PAD>", "<UNK>"]
        organic = ["C", "N", "O", "S", "P", "F", "I", "B", "Cl", "Br", "Si", "Se"]
        aromatic = ["c", "n", "o", "s", "p", "se"]
        bonds = ["-", "=", "#", "/", "\\", ".", ":"]
        rings = list("0123456789") + [f"%{i:02d}" for i in range(10, 100)]
        branch = ["(", ")"]
        bracketed = [
            "[Dy]", "[nH]", "[N+]", "[N-]", "[O-]", "[O+]", "[NH+]", "[NH2+]", "[NH3+]",
            "[C@H]", "[C@@H]", "[C@]", "[C@@]", "[S+]", "[s+]", "[CH]", "[CH2]", "[CH-]",
            "[c-]", "[n+]", "[n-]", "[se]", "[B-]", "[P+]", "[P-]",
        ]
        vocab = list(dict.fromkeys(specials + organic + aromatic + bonds + rings + branch + bracketed))
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.pad_id = 0
        self.unk_id = 1
        self.vocab_size = len(vocab)

    def encode(self, smiles: str, max_length: int) -> list[int]:
        ids = [self.vocab.get(t, self.unk_id) for t in self.PATTERN.findall(smiles)]
        return ids[:max_length]

    def __call__(self, smiles_list, max_length: int = 160) -> dict[str, torch.Tensor]:
        batch = [self.encode(s, max_length) for s in smiles_list]
        L = max(len(t) for t in batch)
        ids = torch.tensor([t + [self.pad_id] * (L - len(t)) for t in batch], dtype=torch.long)
        mask = (ids != self.pad_id).long()
        return {"input_ids": ids, "attention_mask": mask}


class LegacySmilesCharTokenizer:
    """Original 34-token char tokenizer used to train the released ``model_1.pth``.

    It reproduces the original behaviour EXACTLY — unknown characters map to id 0
    (PAD) and are therefore masked out — so inference stays consistent with how the
    released weights were trained. For *fresh* training prefer SmartSmilesTokenizer,
    which fixes the UNK/multi-char-atom issues.
    """

    CHARS = [
        "<PAD>", "C", "N", "O", "F", "S", "c", "n", "o", "s", "(", ")", "[", "]",
        "=", "#", "@", "+", "-", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "l", "r", "B", "D", "y", "H",
    ]

    def __init__(self) -> None:
        self.vocab = {c: i for i, c in enumerate(self.CHARS)}
        self.pad_id = 0
        self.vocab_size = len(self.CHARS)

    def __call__(self, smiles_list, max_length: int = 256) -> dict[str, torch.Tensor]:
        batch = [[self.vocab.get(c, 0) for c in s][:max_length] for s in smiles_list]
        L = max(len(t) for t in batch)
        ids = torch.tensor([t + [0] * (L - len(t)) for t in batch], dtype=torch.long)
        mask = (ids != 0).long()
        return {"input_ids": ids, "attention_mask": mask}


class HFMoleculeTokenizer:
    """Adapter around a HuggingFace tokenizer (e.g. ChemBERTa)."""

    def __init__(self, hf_name: str) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(hf_name)
        self.vocab_size = self.tok.vocab_size

    def __call__(self, smiles_list, max_length: int = 160) -> dict[str, torch.Tensor]:
        enc = self.tok(
            list(smiles_list),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def build_tokenizer(cfg: dict):
    """Build a tokenizer from the ``tokenizer`` config block."""
    kind = cfg["kind"]
    if kind == "char":
        return SmartSmilesTokenizer()
    if kind == "char_legacy":
        return LegacySmilesCharTokenizer()
    if kind == "hf":
        return HFMoleculeTokenizer(cfg["hf_name"])
    raise ValueError(f"unknown tokenizer kind: {kind}")
