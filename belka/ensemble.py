"""Rank-based ensembling for an Average-Precision metric.

AP only cares about *ordering*, so we blend on per-protein percentile ranks
(robust to each model's probability scale). Within-family means first, then a
weighted cross-family blend.

WARNING: do NOT pick the cross-family weight from the public leaderboard. In
BELKA the public LB is mostly *shared* building blocks while the private LB is
mostly *novel* ones — a model that wins public can lose private. Tune the weight
on a nonshare validation split, or just use equal weights.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def load_aligned(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read submissions, sort by id, assert alignment. Returns (ids, matrix[n_models, n_rows])."""
    import polars as pl  # lazy: keeps the pure-numpy helpers import-light

    base = pl.read_csv(files[0]).sort("id")
    ids = base["id"].to_numpy()
    mat = []
    for f in files:
        s = pl.read_csv(f).sort("id")
        assert (s["id"].to_numpy() == ids).all(), f"{f}: ids not aligned"
        mat.append(s["binds"].to_numpy())
    return ids, np.vstack(mat)


def rank_per_protein(x: np.ndarray, prot: np.ndarray) -> np.ndarray:
    """Percentile-rank within each protein group (removes cross-protein scale bias)."""
    out = np.empty(len(x), dtype=float)
    for p in np.unique(prot):
        m = prot == p
        out[m] = rankdata(x[m]) / m.sum()
    return out


def lb_to_weights(scores, temperature: float = 0.03, floor: float = 0.25) -> np.ndarray:
    """Temperature-softmax of family scores with a guaranteed diversity floor.

    Each weight is >= ``floor`` and the weights sum to 1. We allocate ``floor`` to
    every family first, then distribute the remainder by the softmax — clamping +
    renormalising would *not* preserve the floor. Use with care: don't tune on the
    public LB (see module docstring).
    """
    s = np.asarray(scores, dtype=float) / temperature
    n = len(s)
    if n * floor > 1.0:
        raise ValueError(f"floor={floor} too high for {n} families (n*floor must be <= 1)")
    w = np.exp(s - s.max())
    w = w / w.sum()
    return floor + (1.0 - n * floor) * w


def blend(
    family_files: dict[str, list[str]],
    protein_names: np.ndarray,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-level rank blend.

    family_files : {family_name: [submission.csv, ...]}
    weights      : {family_name: weight}; defaults to equal weights.
    Returns (ids, blended_scores).
    """
    ids_ref = None
    family_rank = {}
    for fam, files in family_files.items():
        ids, mat = load_aligned(files)
        if ids_ref is None:
            ids_ref = ids
        else:
            assert (ids == ids_ref).all(), f"{fam}: ids differ from other families"
        ranks = [rank_per_protein(mat[i], protein_names) for i in range(len(mat))]
        family_rank[fam] = np.mean(ranks, axis=0)

    if weights is None:
        weights = {fam: 1.0 for fam in family_files}
    wsum = sum(weights[f] for f in family_rank)
    final = sum(weights[f] / wsum * family_rank[f] for f in family_rank)
    final = rank_per_protein(final, protein_names)
    return ids_ref, final
