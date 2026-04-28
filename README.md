# Motion-S: Hierarchical Text-to-Sign Motion Generation

Solution scaffold for the [Kaggle competition](https://www.kaggle.com/competitions/motion-s-hierarchical-text-to-motion-generation-for-sign-language).

## Task
Given an English `sentence` and `gloss` (sign-language gloss), generate **6 RVQ token layers** (`base_tokens`, `residual_1..residual_5`), each token ∈ [0, 511], total length per sequence ∈ [40, 800], for 3000 test rows. The provided `rvq_vae_best.pth` is the frozen decoder — do not modify or replace.

**Score** = 0.30·FID + 0.50·R-Precision + 0.20·Diversity (in a learned feature space).

## Strategy
Train two complementary text→tokens generators on Colab Pro/Pro+ and ensemble at inference time on Kaggle (T4/P100, 9 h, internet disabled):
1. **Model A — MoMask-style**: masked transformer for the base layer + residual transformer for layers 1–5. Iterative parallel decoding with classifier-free guidance.
2. **Model B — T2M-GPT-style**: causal autoregressive transformer for the base layer + small AR residual heads. Nucleus sampling + CFG.
3. **Ensemble**: per-row, generate K=4 candidates and rerank by a local R-Precision proxy (best-of-N).

R-Precision is 50% of the score, so text–motion alignment is the dominant lever.

## Layout
```
configs/                # YAML hyperparameters per model/run
data/                   # Kaggle dataset (gitignored). Place rvq_vae_best.pth, train.csv, test.csv here
notebooks/              # exploration + final Kaggle inference notebook
src/
  data/                 # loaders, train/val split, inspection
  eval/                 # local proxy scorer + submission validator
  infer/                # ensemble + rerank
  models/               # momask.py, t2mgpt.py
  rvq.py                # frozen RVQ-VAE wrapper (encode/decode only)
  length.py             # text-to-length predictor
scripts/                # CLI entrypoints (train, predict, validate, ensemble)
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place competition files under `data/`:
```
data/
  train.csv
  test.csv
  sample_submission.csv
  rvq_vae_best.pth
  length_estimator.pth
  evaluation_script.py
  baseline_notebook.ipynb
```

## Quick commands
```powershell
# 1. Validate a submission file (works without the model)
python -m scripts.validate_submission data\sample_submission.csv

# 2. Inspect dataset (run once)
python -m scripts.inspect_data

# 3. Build a frozen 90/10 train/val split
python -m scripts.make_split
```

## Status
Phase 0 — scaffolding done; awaiting Kaggle data download.
