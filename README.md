# Bagging and Boosting on Olivetti Faces

Machine Learning, Assignment 2

**Author:** Harshid Bhupendra Rawal  
**Roll number:** M26ECM504  
**Final code commit:** git tag `final-code` (the commit hash is printed on page 1 of the report)

A decision tree on raw face pixels has low bias but very high variance. This project measures what bagging and SAMME boosting do about that. Both ensembles are written from scratch. The only scikit-learn model used is `DecisionTreeClassifier`. Features are raw flattened pixels (4096 per image). No PCA or other dimensionality reduction is used.

## Results at a glance

Choices were made on validation accuracy only and then frozen.

| Model | Train | Validation | Clean test | Masked test |
|---|---|---|---|---|
| Single tree (depth None, min leaf 2) | 0.9958 | 0.5875 | 0.6250 | 0.1875 |
| Bagging (B = 200) | 1.0000 | 0.8875 | 0.8875 | 0.7750 |
| Boosting (depth 4, 100 rounds) | 0.9917 | 0.3500 | 0.4875 | 0.4250 |

Bias and variance over 30 resampled training sets (clean test set):

| Model | Bias squared | Variance | Test accuracy (mean ± std) |
|---|---|---|---|
| Single tree | 0.3747 | 0.6195 | 0.5029 ± 0.0634 |
| Bagging (B = 200) | 0.5094 | 0.0200 | 0.7796 ± 0.0371 |
| Boosting | 0.8742 | 0.0195 | 0.3371 ± 0.0868 |

Effect of depth, single tree against 100-tree bagging:

| Max depth | Tree variance | Bagging variance | Variance removed | Tree acc. | Bagging acc. |
|---|---|---|---|---|---|
| 2 | 0.1384 | 0.0109 | 92.1% | 0.0654 | 0.1625 |
| 8 | 0.3416 | 0.0172 | 95.0% | 0.1804 | 0.4854 |
| None | 0.6195 | 0.0238 | 96.2% | 0.5029 | 0.7663 |

What the numbers show:

- Bagging raises clean test accuracy from 0.6250 to 0.8875 and keeps most of its accuracy under the centre mask (0.7750), while the single tree falls to 0.1875.
- Bagging removes 92% to 96% of the tree variance at every depth. The largest accuracy gain from bagging is at depth 8 (+0.305), not at the deepest tree (+0.263).
- Boosting has about the same variance as bagging but the highest bias squared and the lowest accuracy. Low variance does not mean best accuracy.
- Boosting overfits raw pixels: train accuracy 0.9917 but validation accuracy 0.35.

## How the project is organised

```
src/          reusable code: config, data and split, metrics, bagging and SAMME
scripts/      one script per step, run in numeric order
configs/      config.json (settings) and the frozen choices from Q1 and Q2
data/         split.npz, the saved train / validation / test split
logs/         final logs of every script
results/      CSV tables
figures/      plots
report/       the final PDF
run_all.sh    runs everything in order
```

## Pipeline

```
00 split ──> 01 Q1 tree + bagging ──> 02 Q2 boosting ──> 03 Q3 bias/variance ──> 04 figures
                    │                      │
          configs/frozen_q1.json   configs/frozen_q2.json
```

Each step reads the choices frozen by the earlier step, so nothing is tuned twice.

## Setup rules from the assignment

- Seed `26504` (the numeric characters of M26ECM504, in order) for `numpy.random.default_rng`.
- Per identity, 10 images are split into 6 train, 2 validation and 2 test. The split is saved in `data/split.npz` and used everywhere.
- Masked test set: the centred 16 x 16 block (rows and columns 24 to 39) is set to zero. Models are trained on clean images only.

## Run it

Fresh machine (Python 3.8.10 was used):

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
./run_all.sh
```

One step at a time (run them in this order, because later steps read the frozen settings):

```bash
python scripts/00_make_split.py
python scripts/01_q1_bagging.py
python scripts/02_q2_boosting.py
python scripts/03_q3_bias_variance.py
python scripts/04_make_figures.py
```

Run time: Q1 and Q2 take a few minutes each. Q3 fits several thousand trees and takes about an hour or more on a laptop. The first run downloads the Olivetti data.

## Which file answers which part

| Part | Script | Output |
|---|---|---|
| Q1(a) single tree grid and frozen tree | `01_q1_bagging.py` | `results/q1a_grid.csv`, `logs/01_q1.log` |
| Q1(b), (c) bagging, OOB, disagreement, margin | `01_q1_bagging.py` | `results/q1bc_bagging.csv`, `figures/q1c_plots.png` |
| Q1(d) clean and masked comparison, example faces | `01_q1_bagging.py` | `figures/q1d_cases_clean.png`, `figures/q1d_cases_masked.png`, log |
| Q2(a) toy SAMME round | `02_q2_boosting.py` | `logs/02_q2.log` |
| Q2(b) depth and rounds selection, weight plots | `02_q2_boosting.py` | `results/q2b_grid.csv`, `figures/q2b_plots.png` |
| Q2(c) boosting against bagging, heavy sample | `02_q2_boosting.py` | `results/q2c_compare.csv`, `figures/q2c_heavy_sample.png` |
| Q2(d) clean and masked, one differing face | `02_q2_boosting.py` | `logs/02_q2.log` |
| Q3(a) bias, variance, accuracy spread | `03_q3_bias_variance.py` | `results/q3a.csv` |
| Q3(b), (c) depth study | `03_q3_bias_variance.py` | `results/q3bc.csv`, `figures/fig_bias_variance_depth.png` |
| Q3(d) final table | `03_q3_bias_variance.py` | `results/q3d_final_table.csv` |

## Notes

- All randomness is seeded, so the accuracy numbers repeat exactly. Only the timings in `q2c_compare.csv` change from machine to machine.
- A fully grown tree has almost pure leaves, so its `predict_proba` confidence is 1.0.
- Bias squared for the single tree uses one-hot predictions. For bagging and boosting it uses vote fractions. A spread-out vote vector sits far from the one-hot target even when its top class is right, so compare bias squared across model types with care.
- If a boosting weak learner is no better than random multiclass guessing (error at or above 1 - 1/K), it is discarded and boosting stops. This never happened in the final runs.

## Environment

Python 3.8.10, numpy 1.24.4, scikit-learn 1.3.2, pandas 2.0.3, matplotlib 3.7.5, joblib 1.4.2. Pinned in `requirements.txt`; the full list is in `requirements-lock.txt`.