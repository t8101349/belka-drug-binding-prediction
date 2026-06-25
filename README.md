# BELKA Binding Prediction — Two Model Families

Reusable PyTorch pipeline for the **NeurIPS 2024 – Predict New Medicines with BELKA**
competition (predicting whether a small molecule binds one of three protein
targets: **BRD4, HSA, sEH**), on extremely imbalanced data.

It ships **two interchangeable model families** behind one shared data /
training / inference pipeline, plus a rank-based ensembler:

| | Model 1 — `cnn_film` | Model 2 — `chemberta_crossattn` |
|---|---|---|
| Molecule encoder | char-level + multi-scale 1-D CNN (from scratch) | **frozen** ChemBERTa-77M (pretrained) |
| Protein fusion | **FiLM** (multiplicative conditioning, per layer) | **cross-attention** (protein query-probes read atoms) |
| Readout | learnable `[CLS]` | attention-pooled query probes |
| Params (trainable) | ~4–5M | ~10M (+ frozen 77M) |
| Strength | generalises to **novel** chemistry (private LB) | high capacity on **seen** chemistry (public LB) |

Both share: `SmartSmilesTokenizer`/ChemBERTa tokenizer, BB-aware split,
`FocalLoss`, EMA, OneCycle + AMP trainer, and per-protein rank blending.

---

## Strategy — RUS EasyEnsemble

BELKA is extremely imbalanced (~0.5% positives over hundreds of millions of
rows). This repo is built around an **EasyEnsemble with Random Under-Sampling
(RUS)**, not a single big model:

1. **Under-sample** negatives to ~**1:15** (pos:neg) and merge in external
   actives — keep every positive, train on a manageable slice of negatives.
2. **Bag** — train *several* models, each on a **different** random negative
   subset (vary the RUS sample / `--fold` / seed). Every bag sees all positives
   but different negatives, so their errors **decorrelate**. That is the
   EasyEnsemble idea: many balanced learners over disjoint negative draws.
3. **Diversify architecture** — two decorrelated families so the ensemble isn't
   just one model averaged with itself:
   * **CNN family** — `cnn_hybrid` (released) and the new `cnn_film` (CNN + GEGLU
     FiLM Peri-LN). Local functional-group features → strong on **novel**
     building blocks (private LB).
   * **ChemBERTa family** — frozen ChemBERTa + cross-attention. Pretrained
     capacity → strong on **seen** chemistry (public LB).
4. **Aggregate** — per-protein **rank** blend across every bag and family
   (`scripts/blend.py`). Rank averaging fits the Average-Precision metric and
   cancels each model's probability-scale differences.

The files in `model_weights/` are example bags. The whole workflow is: add more
bags (more folds/seeds, the new CNN+GEGLU model), then rank-blend them.

---

## ⚠️ The single most important BELKA lesson

The **public** leaderboard is dominated by *shared* building blocks (chemistry
seen in training); the **private** leaderboard is dominated by *nonshare* /
novel building blocks. They measure different things and can **disagree** — a
model that wins public can lose private.

Consequences baked into this repo:

1. **Split by building block** (`bb_aware_split`) — never a random split.
2. **Validate at the natural class distribution** — split *before* any RUS
   sub-sampling, or your validation AP is inflated.
3. **Never tune ensemble weights on the public LB.** Tune on a nonshare
   validation split, or use equal weights. `ensemble.lb_to_weights` exists but
   keeps a diversity floor and is documented as risky.

In practice, Model 1 (CNN+FiLM) is the more robust private-LB base; Model 2
(ChemBERTa) adds *decorrelated* diversity and should be blended at a modest
weight, not used alone because it scored high on public.

---

## Install

```bash
git clone <your-repo-url> belka-binding-prediction
cd belka-binding-prediction
git lfs install && git lfs pull        # fetch model_weights/*.pth (see below)
pip install -e .                       # or: pip install -r requirements.txt
```

## Pretrained weights (`model_weights/`)

Two trained checkpoints are bundled and tracked with **Git LFS**:

| File | Loads into | Tokenizer | Config | Status |
|------|-----------|-----------|--------|--------|
| `model_weights/model_1.pth` (15 MB) | `BELKAHybridModel` (`cnn_hybrid`) | `char_legacy` (34-token) | `configs/model1_cnn_hybrid.yaml` | ✅ included |
| `model_weights/model_2.pth` (80 MB) | `ChemBERTaCrossAttnModel` | ChemBERTa HF | `configs/model2_chemberta.yaml` | ✅ included |
| `model_weights/model_3.pth` | `CNNFiLMModel` (`cnn_film`, CNN+GEGLU) | `char` (Smart) | `configs/model3_cnn_film.yaml` | ⏳ **reserved** — drop in after training |

When the new CNN+GEGLU (`cnn_film`) model finishes training, drop it into the
reserved `model_3` slot and it joins the ensemble automatically:

```bash
python scripts/train.py --config configs/model3_cnn_film.yaml --fold 0
cp checkpoints/model_3_fold0.pth model_weights/model_3.pth   # fill the slot
python scripts/predict.py --config configs/model3_cnn_film.yaml --out submissions/model_3.csv
```

> **Note** `model_1.pth` is the *original* hybrid architecture (single CNN +
> Transformer + BatchNorm head) trained with the legacy 34-token tokenizer — **not**
> the newer `cnn_film` model. Use `configs/model1_cnn_hybrid.yaml` (which pins
> `nhead=4` and `char_legacy`) to reproduce it; `cnn_film` is for fresh training.

Predict straight from the bundled weights (no `--checkpoint` needed — `predict.py`
falls back to the config's `pretrained:` field):

```bash
python scripts/predict.py --config configs/model1_cnn_hybrid.yaml --out submissions/model_1.csv
python scripts/predict.py --config configs/model2_chemberta.yaml  --out submissions/model_2.csv
python scripts/blend.py --cnn submissions/model_1.csv --chemberta submissions/model_2.csv \
    --test /kaggle/input/competitions/leash-BELKA/test.parquet --out submissions/blend.csv
```

If you don't want Git LFS, host the `.pth` files on a GitHub Release or the
HuggingFace Hub and download them into `model_weights/` instead.

## Project layout

```
belka/
  proteins.py     ESM-2 sequences + one-time embedding pre-compute (3 vectors)
  tokenizer.py    SmartSmilesTokenizer (char) + HF tokenizer adapter
  data.py         BB-aware split, datasets, collates
  losses.py       FocalLoss (label-smoothed BCE + hard-label focal modulation)
  ema.py          ModelEMA (shadows only trainable params)
  layers.py       MultiScaleCNN, FiLM/Peri-LN, Cross-Attn, Query-Self-Attn
  models.py       CNNFiLMModel (M1 new), BELKAHybridModel (M1 released),
                  ChemBERTaCrossAttnModel (M2), build_model
  trainer.py      shared OneCycle+AMP+EMA loop, safe checkpoint load/save
  ensemble.py     per-protein rank blend, LB->weight helper
configs/          model1_cnn_hybrid / model2_chemberta / model3_cnn_film YAMLs
scripts/          train.py / predict.py / blend.py (CLI)
model_weights/    bundled pretrained checkpoints (Git LFS)
```

## Usage

Edit the `data.train_path` / `data.test_path` in the chosen config first.

### 1. Train

```bash
# Model 3 — new CNN+GEGLU (recommended base for fresh training)
python scripts/train.py --config configs/model3_cnn_film.yaml --fold 0

# Model 2 — ChemBERTa
python scripts/train.py --config configs/model2_chemberta.yaml --fold 0
```

Train several **RUS bags** by varying `--fold` (and/or the seed) to build an
EasyEnsemble; each bag sees different negatives.

### 2. Predict

```bash
python scripts/predict.py --config configs/model3_cnn_film.yaml \
    --checkpoint checkpoints/model_3_fold0.pth \
    --out submissions/model_3.csv
```

`predict.py` asserts the trainable weights actually loaded (guards against the
silent `strict=False` / DataParallel-prefix failure that yields a random-weight
submission).

### 3. Blend

```bash
# equal-weight (safest)
python scripts/blend.py \
    --cnn submissions/cnn_film_fold*.csv \
    --chemberta submissions/chemberta_fold*.csv \
    --test /kaggle/input/competitions/leash-BELKA/test.parquet \
    --out submissions/blend.csv

# tilt toward the CNN family (better private generalisation)
python scripts/blend.py --cnn submissions/cnn_*.csv --chemberta submissions/chemberta_*.csv \
    --test .../test.parquet --w-cnn 0.7 --w-chemberta 0.3 --out submissions/blend.csv
```

Blending is **rank-based and per-protein** because Average Precision only cares
about ordering and the two families output different probability scales.

## Library API (notebook use)

```python
import torch
from belka.proteins import precompute_protein_embeddings
from belka.tokenizer import SmartSmilesTokenizer
from belka.models import CNNFiLMModel

device = "cuda"
prot_map = precompute_protein_embeddings(device=device)         # {name: vector}
tok = SmartSmilesTokenizer()
model = CNNFiLMModel(vocab_size=tok.vocab_size, prot_dim=480).to(device)
```

## Data expectations

A single parquet with columns:
`buildingblock1_smiles, molecule_smiles, protein_name, binds` (+ `id` for test).
External rows merged in should still carry `buildingblock1_smiles`; otherwise the
group split warns and lumps them into one group.

## Notes & licence

* ESM-2 (MIT) is used only to embed 3 fixed proteins; with so few targets the
  protein encoder choice barely matters (a 3-entry lookup would do).
* Optional speed-up: ChemBERTa is frozen, so molecule features are constant
  across epochs and can be pre-computed once to disk — see comments in
  `models.py` / project issues.
* Code: MIT (see `LICENSE`). Model weights and competition data follow their own
  licences — check the competition rules before using external models.

This repository is educational; it reconstructs a clean, debugged version of a
BELKA solution pipeline.
