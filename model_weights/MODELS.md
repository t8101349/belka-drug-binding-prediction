# Model weights — RUS EasyEnsemble bags

These are trained "bags" of the ensemble (see the Strategy section in the root
README). Large `.pth` files are tracked with **Git LFS** (`.gitattributes`).

| File | Architecture | Tokenizer | Config | Status |
|------|--------------|-----------|--------|--------|
| `model_1.pth` | `BELKAHybridModel` — CNN + Transformer + BatchNorm head | `char_legacy` (34-token) | `configs/model1_cnn_hybrid.yaml` | ✅ included |
| `model_2.pth` | `ChemBERTaCrossAttnModel` — frozen ChemBERTa + cross-attention | ChemBERTa HF | `configs/model2_chemberta.yaml` | ✅ included |
| `model_3.pth` | `CNNFiLMModel` — multi-scale CNN + **GEGLU** FiLM Peri-LN | `char` (SmartSmiles) | `configs/model3_cnn_film.yaml` | ⏳ reserved (train, then drop in) |

## Adding the new CNN+GEGLU model (model_3)

```bash
# 1. train (vary --fold / seed to add more bags)
python scripts/train.py --config configs/model3_cnn_film.yaml --fold 0

# 2. fill the reserved slot
cp checkpoints/model_3_fold0.pth model_weights/model_3.pth

# 3. predict + blend into the ensemble
python scripts/predict.py --config configs/model3_cnn_film.yaml --out submissions/model_3.csv
python scripts/blend.py \
    --cnn submissions/model_1.csv submissions/model_3.csv \
    --chemberta submissions/model_2.csv \
    --test /kaggle/input/competitions/leash-BELKA/test.parquet \
    --out submissions/blend.csv
```

`model_1` and `model_3` are both the CNN family (pass them together to `--cnn`);
`model_2` is the ChemBERTa family.

## Adding more bags

Each extra bag = a model trained on a different RUS negative draw (different
pre-sampled parquet, `--fold`, or seed). Drop each `submission_*.csv` into the
matching `--cnn` / `--chemberta` list in `scripts/blend.py`. More decorrelated
bags → better, more shake-up-resistant private score.

> Reminder: tune blend weights on a **nonshare** validation split, never the
> public leaderboard (it measures *seen* chemistry; the private LB measures
> *novel* chemistry).
