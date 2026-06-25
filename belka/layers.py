"""Shared neural building blocks.

All transformer-style blocks use the same stable recipe:
    Peri-LN (LayerNorm on the sublayer *input and output*) + GEGLU FFN +
    ReZero-style learnable residual gate (alpha initialised to 0, so each block
    starts as identity and the network is well-conditioned at init).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """Parallel k=3/5/7 1-D convolutions to capture functional groups at
    different scales (double bonds, rings, hetero-cycle motifs)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        per = out_dim // 3
        rem = out_dim - 2 * per
        self.c3 = nn.Conv1d(in_dim, per, kernel_size=3, padding=1)
        self.c5 = nn.Conv1d(in_dim, per, kernel_size=5, padding=2)
        self.c7 = nn.Conv1d(in_dim, rem, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, in) -> (B, L, out)
        x = x.transpose(1, 2)
        x = torch.cat([self.c3(x), self.c5(x), self.c7(x)], dim=1).transpose(1, 2)
        return self.norm(self.act(x))


class FiLMPeriLNGatedLayer(nn.Module):
    """Self-attention block whose FFN output is FiLM-modulated by the protein
    vector (multiplicative conditioning). The FiLM generator is zero-initialised
    so it starts as identity (gamma = beta = 0)."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, prot_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff * 2)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.n_ai = nn.LayerNorm(d_model)
        self.n_ao = nn.LayerNorm(d_model)
        self.n_fi = nn.LayerNorm(d_model)
        self.n_fo = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.a_attn = nn.Parameter(torch.zeros(1))
        self.a_ffn = nn.Parameter(torch.zeros(1))
        self.film = nn.Sequential(
            nn.Linear(prot_dim, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model * 2)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, src, prot_cond, pad_mask=None):
        nx = self.n_ai(src)
        a, _ = self.self_attn(nx, nx, nx, key_padding_mask=pad_mask, need_weights=False)
        src = src + self.a_attn * self.drop(self.n_ao(a))

        nx = self.n_fi(src)
        h, g = self.lin1(nx).chunk(2, dim=-1)
        ffn = self.n_fo(self.lin2(h * F.gelu(g)))
        gamma, beta = self.film(prot_cond).chunk(2, dim=-1)
        ffn = ffn * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return src + self.a_ffn * self.drop(ffn)


class CrossAttnFusionLayer(nn.Module):
    """Cross-attention: protein-derived queries read molecule tokens (K=V=mol)."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff * 2)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.n_q = nn.LayerNorm(d_model)
        self.n_kv = nn.LayerNorm(d_model)
        self.n_ao = nn.LayerNorm(d_model)
        self.n_fi = nn.LayerNorm(d_model)
        self.n_fo = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.a_attn = nn.Parameter(torch.zeros(1))
        self.a_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, q, kv, kv_pad_mask=None):
        nq, nkv = self.n_q(q), self.n_kv(kv)
        a, _ = self.cross_attn(nq, nkv, nkv, key_padding_mask=kv_pad_mask, need_weights=False)
        q = q + self.a_attn * self.drop(self.n_ao(a))

        nq = self.n_fi(q)
        h, g = self.lin1(nq).chunk(2, dim=-1)
        ffn = self.lin2(h * F.gelu(g))
        q = q + self.a_ffn * self.drop(self.n_fo(ffn))
        return q


class QuerySelfAttnLayer(nn.Module):
    """Self-attention among the protein query probes so they specialise."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff * 2)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.n_ai = nn.LayerNorm(d_model)
        self.n_ao = nn.LayerNorm(d_model)
        self.n_fi = nn.LayerNorm(d_model)
        self.n_fo = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.a_attn = nn.Parameter(torch.zeros(1))
        self.a_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        nx = self.n_ai(x)
        a, _ = self.self_attn(nx, nx, nx, need_weights=False)
        x = x + self.a_attn * self.drop(self.n_ao(a))

        nx = self.n_fi(x)
        h, g = self.lin1(nx).chunk(2, dim=-1)
        ffn = self.lin2(h * F.gelu(g))
        x = x + self.a_ffn * self.drop(self.n_fo(ffn))
        return x
