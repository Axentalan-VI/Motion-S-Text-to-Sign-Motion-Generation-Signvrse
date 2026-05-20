# Motion-S: Hierarchical Text-to-Sign Motion Generation

This repository implements a discrete-token pipeline for the [Kaggle Motion-S competition](https://www.kaggle.com/competitions/motion-s-hierarchical-text-to-motion-generation-for-sign-language). The goal is to map English text and gloss into a 6-layer RVQ token grid that can be decoded into sign motion by the organizer-provided frozen RVQ-VAE.

## Project overview

The project uses a 3-stage architecture:

1. **Frozen RVQ tokenizer**
   - Encodes motion into 6 token layers with codebook size 512.
   - Decodes predicted tokens back into motion.
   - The checkpoint is fixed and is used only for encode/decode.

2. **Length predictor**
   - A DistilBERT-based classifier that predicts one of 32 sequence-length bins.
   - Provides the target motion duration before token generation.

3. **MoMask token generator**
   - **Base transformer** predicts layer 0 with iterative masked-token decoding.
   - **Residual transformer** predicts layers 1 through 5 conditioned on the lower layers.

This decomposition makes the task easier than generating raw motion frames directly: the model first predicts duration, then a coarse discrete motion plan, then progressively refines it.

## Competition task

For each test row, generate:

- `base_tokens`
- `residual_1`
- `residual_2`
- `residual_3`
- `residual_4`
- `residual_5`

Each token is an integer in `[0, 511]`, and each sequence length must stay within `[40, 800]`.

The competition score is:

- `0.30 * FID`
- `0.50 * R-Precision`
- `0.20 * Diversity`

Text-motion alignment is the main optimization target because R-Precision has the highest weight.

## Repository structure

```text
configs/                YAML hyperparameters
data/                   competition data and frozen checkpoints
scripts/                training, validation, and debugging entrypoints
src/
  data/                 dataset loading and split helpers
  eval/                 local scoring and submission validation
  infer/                inference utilities
  models/               MoMask models, text conditioning, token datasets
  constants.py          shared paths and constants
  length.py             length predictor model
  rvq.py                frozen RVQ-VAE wrapper
```

## Required data files

Place the competition assets under `data/`:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  rvq_vae_best.pth
  evaluation_script.py
```

Optional local artifacts created by training include:

```text
data/
  length_estimator.pth
checkpoints/
  momask_base.pth
  momask_residual.pth
```

## Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Typical workflow

Inspect and prepare the dataset:

```powershell
python -m scripts.inspect_data
python -m scripts.make_split
```

Train the length predictor:

```powershell
python -m scripts.train_length --epochs 8 --batch-size 64
```

Train the MoMask models:

```powershell
python -m scripts.train_momask_base --epochs 30 --batch-size 32
python -m scripts.train_momask_residual --epochs 10 --batch-size 32
```

Validate checkpoints and submissions:

```powershell
python -m scripts.test_rvq
python -m scripts.test_momask
python -m scripts.validate_submission data\sample_submission.csv
```

## Current scope

This codebase is centered on the MoMask-style hierarchical generator and its supporting tooling for training, validation, and submission checks. The RVQ model remains frozen throughout.
