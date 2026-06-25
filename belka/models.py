"""The two model families. Both share the forward signature

    forward(input_ids, attention_mask, prot_emb) -> logits (B, 1)

so the trainer / predictor are identical for both. Each model declares
``EXCLUDE_PREFIXES`` listing parameter-name prefixes that should NOT be saved
(used to drop the frozen ChemBERTa backbone from checkpoints).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    CrossAttnFusionLayer,
    FiLMPeriLNGatedLayer,
    MultiScaleCNN,
    QuerySelfAttnLayer,
)


class CNNFiLMModel(nn.Module):
    """Model 1 — Multi-Scale CNN + FiLM Peri-LN, trained from scratch.

    Local functional-group features (CNN) + protein conditioning via FiLM. Tends
    to generalise better to *novel* building blocks (private leaderboard).
    """

    EXCLUDE_PREFIXES: tuple[str, ...] = ()

    def __init__(self, vocab_size: int, embed_dim: int = 128, d_model: int = 256,
                 prot_dim: int = 480, max_len: int = 256, num_layers: int = 4,
                 nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        self.cnn = MultiScaleCNN(embed_dim, d_model)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)
        self.prot_proj = nn.Sequential(
            nn.LayerNorm(prot_dim), nn.Linear(prot_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            FiLMPeriLNGatedLayer(d_model, nhead, d_model * 4, d_model, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, prot_emb):
        B, L = input_ids.size()
        x = self.embedding(input_ids)
        x = self.cnn(x)
        x = x * attention_mask.unsqueeze(-1).float()
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(L + 1, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        prot_cond = self.prot_proj(prot_emb)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_pad = torch.cat([cls_pad, attention_mask == 0], dim=1)
        for layer in self.layers:
            x = layer(x, prot_cond, pad_mask=full_pad)
        x = self.final_norm(x)
        return self.head(x[:, 0, :])


class ChemBERTaCrossAttnModel(nn.Module):
    """Model 2 — frozen ChemBERTa + protein-query cross-attention fusion.

    Protein-conditioned query probes cross-attend molecule tokens, then
    self-attend to specialise; an attention-pooled readout feeds the head.
    High public-LB capacity; weight it modestly in an ensemble (it overfits
    *seen* chemistry — see README).
    """

    EXCLUDE_PREFIXES: tuple[str, ...] = ("chemberta",)

    def __init__(self, chem_model_name: str = "DeepChem/ChemBERTa-77M-MLM",
                 d_model: int = 384, prot_dim: int = 480, num_queries: int = 8,
                 num_fusion_layers: int = 4, nhead: int = 6, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.chemberta = AutoModel.from_pretrained(chem_model_name)
        for p in self.chemberta.parameters():
            p.requires_grad = False
        self.chemberta.eval()
        chem_dim = self.chemberta.config.hidden_size

        self.mol_proj = nn.Sequential(nn.LayerNorm(chem_dim), nn.Linear(chem_dim, d_model))
        self.prot_proj = nn.Sequential(
            nn.LayerNorm(prot_dim), nn.Linear(prot_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.num_queries = num_queries
        self.query_seeds = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.query_pos = nn.Embedding(num_queries, d_model)

        self.cross_layers = nn.ModuleList([
            CrossAttnFusionLayer(d_model, nhead, d_model * 4, dropout) for _ in range(num_fusion_layers)
        ])
        self.query_layers = nn.ModuleList([
            QuerySelfAttnLayer(d_model, nhead, d_model * 4, dropout) for _ in range(num_fusion_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 2),
            nn.Linear(128, 1),
        )

    def train(self, mode: bool = True):  # keep ChemBERTa frozen in eval mode always
        super().train(mode)
        self.chemberta.eval()
        return self

    def forward(self, input_ids, attention_mask, prot_emb):
        B = input_ids.size(0)
        with torch.no_grad():
            chem_out = self.chemberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mol = self.mol_proj(chem_out)
        mol_pad = attention_mask == 0

        prot_vec = self.prot_proj(prot_emb).unsqueeze(1)
        seeds = self.query_seeds.unsqueeze(0).expand(B, -1, -1)
        qpos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        q = seeds + qpos + prot_vec

        for ca, sa in zip(self.cross_layers, self.query_layers):
            q = ca(q, mol, kv_pad_mask=mol_pad)
            q = sa(q)
        q = self.final_norm(q)

        pq = self.pool_query.expand(B, -1, -1)
        fused, _ = self.pool_attn(pq, q, q, need_weights=False)
        return self.head(fused.squeeze(1))


class BELKAHybridModel(nn.Module):
    """Original Model-1 architecture: single 1-D CNN + Transformer encoder +
    masked mean-pool + BatchNorm head (late fusion with the protein vector).

    Kept so the released ``model_weights/model_1.pth`` loads. It was trained with
    the legacy 34-token char tokenizer (``tokenizer.kind: char_legacy``) and
    ``nhead=4`` must match training. For new work prefer ``CNNFiLMModel``.
    """

    EXCLUDE_PREFIXES: tuple[str, ...] = ()

    def __init__(self, vocab_size: int = 34, embed_dim: int = 128, cnn_dim: int = 256,
                 prot_dim: int = 480, max_len: int = 256, num_layers: int = 4,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, cnn_dim, kernel_size=3, padding=1)
        self.cnn_activation = nn.GELU()
        self.pos_encoder = nn.Embedding(max_len, cnn_dim)
        layer = nn.TransformerEncoderLayer(
            cnn_dim, nhead, dim_feedforward=cnn_dim * 4, batch_first=True,
            dropout=dropout, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(cnn_dim + prot_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask, prot_emb):
        B, L = input_ids.size()
        x = self.embedding(input_ids)
        x = self.cnn_activation(self.conv1d(x.permute(0, 2, 1))).permute(0, 2, 1)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_encoder(pos)
        pad = attention_mask == 0
        x = self.transformer(x, src_key_padding_mask=pad)
        tok = attention_mask.unsqueeze(-1).float()
        pooled = (x * tok).sum(dim=1) / tok.sum(dim=1).clamp(min=1e-9)  # masked mean
        return self.head(torch.cat([pooled, prot_emb], dim=1))


def build_model(cfg: dict, prot_dim: int, vocab_size: int | None = None):
    """Construct a model from the ``model`` config block."""
    m = cfg["model"]
    name = m["name"]
    if name == "cnn_film":
        assert vocab_size is not None, "cnn_film needs the tokenizer vocab_size"
        return CNNFiLMModel(
            vocab_size=vocab_size, embed_dim=m.get("embed_dim", 128),
            d_model=m.get("d_model", 256), prot_dim=prot_dim,
            num_layers=m.get("num_layers", 4), nhead=m.get("nhead", 8),
            dropout=m.get("dropout", 0.1),
        )
    if name == "cnn_hybrid":
        assert vocab_size is not None, "cnn_hybrid needs the tokenizer vocab_size"
        return BELKAHybridModel(
            vocab_size=vocab_size, embed_dim=m.get("embed_dim", 128),
            cnn_dim=m.get("cnn_dim", 256), prot_dim=prot_dim,
            num_layers=m.get("num_layers", 4), nhead=m.get("nhead", 4),
            dropout=m.get("dropout", 0.1),
        )
    if name == "chemberta_crossattn":
        return ChemBERTaCrossAttnModel(
            chem_model_name=m.get("chem_model_name", "DeepChem/ChemBERTa-77M-MLM"),
            d_model=m.get("d_model", 384), prot_dim=prot_dim,
            num_queries=m.get("num_queries", 8), num_fusion_layers=m.get("num_fusion_layers", 4),
            nhead=m.get("nhead", 6), dropout=m.get("dropout", 0.1),
        )
    raise ValueError(f"unknown model name: {name}")
