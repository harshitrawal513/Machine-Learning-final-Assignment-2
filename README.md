# Bagging and Boosting on Olivetti Faces

Machine Learning, Assignment 2

- **Name:** Harshid Bhupendra Rawal
- **Roll number:** M26ECM504
- **Final code:** git tag `final-code` (the commit hash is on page 1 of the report)

## What this project is about

A decision tree trained on raw face pixels fits the training images very well, but it changes a lot when the training set changes. In this assignment I check how much bagging and boosting help with that.

Both ensembles are written by me from scratch. The only scikit-learn model I use is `DecisionTreeClassifier`. The features are the raw 4096 pixels of each image. I do not use PCA or any other reduction.

## Main results

All choices (tree depth, number of trees, boosting depth and rounds) were made on validation accuracy only. After that they were frozen, and the test sets were used only to report.

| Model | Train | Validation | Clean test | Masked test |
|---|---|---|---|---|
| Single tree (depth None, min leaf 2) | 0.9958 | 0.5875 | 0.6250 | 0.1875 |
| Bagging (200 trees) | 1.0000 | 0.8875 | 0.8875 | 0.7750 |
| Boosting (depth 4, 100 rounds) | 0.9917 | 0.3500 | 0.4875 | 0.4250 |

Bias and variance, measured over 30 resampled training sets (clean test set):

| Model | Bias squared | Variance | Test accuracy (mean ± std) |
|---|---|---|---|
| Single tree | 0.3747 | 0.6195 | 0.5029 ± 0.0634 |
| Bagging (200 trees) | 0.5094 | 0.0200 | 0.7796 ± 0.0371 |
| Boosting | 0.8742 | 0.0195 | 0.3371 ± 0.0868 |

Single tree against 100-tree bagging at different depths:

| Max depth | Tree variance | Bagging variance | Variance removed | Tree acc. | Bagging acc. |
|---|---|---|---|---|---|
| 2 | 0.1384 | 0.0109 | 92.1% | 0.0654 | 0.1625 |
| 8 | 0.3416 | 0.0172 | 95.0% | 0.1804 | 0.4854 |
| None | 0.6195 | 0.0238 | 96.2% | 0.5029 | 0.7663 |

## What I found

- Bagging lifts clean test accuracy from 0.6250 to 0.8875. With the centre block masked it still gets 0.7750, while the single tree drops to 0.1875.
- Bagging removes 92% to 96% of the tree's variance at every depth. The biggest accuracy gain is at depth 8 (+0.305), not at the deepest tree (+0.263).
- Boosting has almost the same variance as bagging, but the highest bias and the lowest accuracy. So the lowest variance is not the same as the best accuracy.
- Boosting overfits on raw pixels: train accuracy is 0.9917 but validation accuracy is only 0.35.

Each test set has only 80 images, so one image is 1.25 accuracy points. Small differences should not be over-read.

## Folder layout

```
src/          shared code: config, data and split, metrics, bagging and SAMME
scripts/      one script per step, run in number order
configs/      config.json and the frozen choices from Q1 and Q2
data/         split.npz, the saved train / validation / test split
logs/         final log of every script
results/      CSV tables
figures/      plots
report/       the final PDF
run_all.sh    runs all scripts in order
```

## How the scripts connect

```
00 split -> 01 Q1 tree + bagging -> 02 Q2 boosting -> 03 Q3 bias/variance -> 04 figures
                   |                       |
         configs/frozen_q1.json   configs/frozen_q2.json
```

Each script reads the choices saved by the one before it, so nothing is tuned twice. This is why the order matters.

## Setup rules from the assignment

- Seed is `26504`, the numbers in the roll number M26ECM504 in order. It is passed to `numpy.random.default_rng`.
- For each of the 40 identities, the 10 images are split into 6 train, 2 validation and 2 test. That gives 240, 80 and 80 images. The split is saved in `data/split.npz` and every script loads that same file.
- Masked test set: the centre 16 x 16 block (rows and columns 24 to 39) is set to zero. Models are trained on clean images only.

## How to run

Python 3.8.10 was used.

```bash
git clone https://github.com/harshitrawal513/Machine-Learning-final-Assignment-2.git
cd Machine-Learning-final-Assignment-2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
./run_all.sh
```

Or one step at a time, in this order:

```bash
python scripts/00_make_split.py
python scripts/01_q1_bagging.py
python scripts/02_q2_boosting.py
python scripts/03_q3_bias_variance.py
python scripts/04_make_figures.py
```

Run time: Q1 and Q2 take a few minutes each. Q3 trains several thousand trees and takes about an hour or more on a laptop. The first run also downloads the Olivetti data.

## Which script answers which question

| Part | Script | Output |
|---|---|---|
| Q1(a) single tree grid and frozen tree | `01_q1_bagging.py` | `results/q1a_grid.csv`, `logs/01_q1.log` |
| Q1(b), (c) bagging, OOB, disagreement, margin | `01_q1_bagging.py` | `results/q1bc_bagging.csv`, `figures/q1c_plots.png` |
| Q1(d) clean vs masked, example faces | `01_q1_bagging.py` | `figures/q1d_cases_clean.png`, `figures/q1d_cases_masked.png` |
| Q2(a) toy SAMME round | `02_q2_boosting.py` | `logs/02_q2.log` |
| Q2(b) depth and rounds, weight plots | `02_q2_boosting.py` | `results/q2b_grid.csv`, `figures/q2b_plots.png` |
| Q2(c) boosting vs bagging, heavy sample | `02_q2_boosting.py` | `results/q2c_compare.csv`, `figures/q2c_heavy_sample.png` |
| Q2(d) clean vs masked, one differing face | `02_q2_boosting.py` | `logs/02_q2.log` |
| Q3(a) bias, variance, accuracy spread | `03_q3_bias_variance.py` | `results/q3a.csv` |
| Q3(b), (c) depth study | `03_q3_bias_variance.py` | `results/q3bc.csv`, `figures/fig_bias_variance_depth.png` |
| Q3(d) final table | `03_q3_bias_variance.py` | `results/q3d_final_table.csv` |

## Things to know

- Everything is seeded, so accuracy numbers repeat exactly. Only the timings in `q2c_compare.csv` change from machine to machine.
- A fully grown tree has almost pure leaves, so its `predict_proba` confidence is always 1.0. This is why tree confidence says nothing about being right.
- For the single tree, bias squared uses one-hot predictions. For bagging and boosting it uses vote fractions. A spread-out vote sits far from the one-hot target even when its top class is right, so compare bias squared across model types with care.
- In boosting, if a weak learner is no better than random guessing (error at or above 1 - 1/K), it is dropped and boosting stops. This never happened in the final runs.

## Environment

Python 3.8.10, numpy 1.24.4, scikit-learn 1.3.2, pandas 2.0.3, matplotlib 3.7.5, joblib 1.4.2. These are pinned in `requirements.txt`. The full list is in `requirements-lock.txt`.
