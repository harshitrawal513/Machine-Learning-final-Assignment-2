# Bagging and Boosting on Olivetti Faces

**Machine Learning — Assignment 2**

- **Name:** Harshid Bhupendra Rawal
- **Roll number:** M26ECM504

## Overview

This project compares a single decision tree, bagging and SAMME boosting on the Olivetti Faces dataset. Both ensembles are implemented from scratch using `DecisionTreeClassifier` as the base learner.

## Experimental Setup

- **Features:** All 4096 raw pixels; no dimensionality reduction.
- **Seed:** `26504`, taken from the numeric characters in the roll number.
- **Split:** 6 training, 2 validation and 2 test images per identity, giving 240/80/80 images.
- **Masked test:** Rows and columns `24:40` are set to zero.
- Models are trained on clean images only.
- Hyperparameters are selected using validation accuracy and frozen before test evaluation.
- The saved split is reused in every experiment.

## Results

| Model | Train | Validation | Clean Test | Masked Test |
|---|---:|---:|---:|---:|
| Single tree (`max_depth=None`, `min_samples_leaf=2`) | 0.9958 | 0.5875 | 0.6250 | 0.1875 |
| Bagging (200 trees) | 1.0000 | 0.8875 | 0.8875 | 0.7750 |
| SAMME (depth 4, 100 rounds) | 0.9917 | 0.3500 | 0.4875 | 0.4250 |

Bagging performed best on both test sets. Across 30 resampled training sets, it reduced measured prediction variance by about 92–96%, depending on tree depth. Boosting showed a large gap between training and validation accuracy.

Each test set contains 80 images, so one prediction changes accuracy by 1.25 percentage points.

## Installation and Usage

Tested with **Python 3.8.10**.

```bash
git clone https://github.com/harshitrawal513/Machine-Learning-final-Assignment-2.git
cd Machine-Learning-final-Assignment-2

python3 -m venv venv
source venv/bin/activate

python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"

bash run_all.sh
```

To run the steps separately:

```bash
python scripts/00_make_split.py
python scripts/01_q1_bagging.py
python scripts/02_q2_boosting.py
python scripts/03_q3_bias_variance.py
python scripts/04_make_figures.py
```

Run the scripts in this order. The first run downloads the dataset. The bias–variance experiment may take an hour or more on a laptop.

## Repository Structure

```text
.
├── src/          # Data loading, metrics, bagging and SAMME
├── scripts/      # Experiment and plotting scripts
├── configs/      # Settings and frozen hyperparameters
├── data/         # Saved train/validation/test split
├── logs/         # Execution logs
├── results/      # CSV results
├── figures/      # Generated plots
├── report/       # Final PDF
├── requirements.txt
├── requirements-lock.txt
└── run_all.sh
```

## Notes

- Dependencies are pinned in `requirements.txt`; the full environment is recorded in `requirements-lock.txt`.
- Bias–variance calculations use one-hot predictions for the tree and vote fractions for the ensembles. Compare these values across model types with care.
