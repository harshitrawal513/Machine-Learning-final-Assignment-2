# Bagging and Boosting on Olivetti Faces

**Machine Learning — Assignment 2**  
**Name:** Harshid Bhupendra Rawal  
**Roll number:** M26ECM504

## Overview

This project compares a single decision tree, bagging and SAMME boosting on Olivetti Faces. Both ensembles are implemented from scratch using `DecisionTreeClassifier` as the base learner.

The models use all 4096 raw pixels without dimensionality reduction. Each identity is split into 6 training, 2 validation and 2 test images using `numpy.random.default_rng(26504)`. The saved split is reused throughout.

Hyperparameters are selected using validation accuracy and frozen before test evaluation. For the masked test, the central 16 × 16 block (`24:40, 24:40`) is set to zero. Training uses clean images only.

## Results

### Classification Accuracy

| Model | Train | Validation | Clean test | Masked test |
|---|---:|---:|---:|---:|
| Single tree (depth=None, min leaf=2) | 0.9958 | 0.5875 | 0.6250 | 0.1875 |
| Bagging (200 trees) | 1.0000 | 0.8875 | 0.8875 | 0.7750 |
| Boosting (depth=4, 100 rounds) | 0.9917 | 0.3500 | 0.4875 | 0.4250 |

### Bias and Variance

Measured over 30 resampled training sets on the clean test set.

| Model | Bias squared | Variance | Test accuracy (mean ± std) |
|---|---:|---:|---:|
| Single tree | 0.3747 | 0.6195 | 0.5029 ± 0.0634 |
| Bagging (200 trees) | 0.5094 | 0.0200 | 0.7796 ± 0.0371 |
| Boosting | 0.8742 | 0.0195 | 0.3371 ± 0.0868 |

The tree uses one-hot predictions, while ensembles use vote fractions, so cross-model bias and variance comparisons need care.

### Effect of Tree Depth

Single tree compared with 100-tree bagging.

| Max depth | Tree variance | Bagging variance | Variance removed | Tree accuracy | Bagging accuracy |
|---|---:|---:|---:|---:|---:|
| 2 | 0.1384 | 0.0109 | 92.1% | 0.0654 | 0.1625 |
| 8 | 0.3416 | 0.0172 | 95.0% | 0.1804 | 0.4854 |
| None | 0.6195 | 0.0238 | 96.2% | 0.5029 | 0.7663 |

Bagging performs best on both test sets and reduces measured variance at every tested depth. Boosting shows a large training–validation gap. Each test set has 80 images, so one prediction changes accuracy by 1.25 percentage points.

## Reproduction

Tested with Python 3.8.10. Dependencies are pinned in `requirements.txt`; the full environment is in `requirements-lock.txt`.

```bash
git clone https://github.com/harshitrawal513/Machine-Learning-final-Assignment-2.git
cd Machine-Learning-final-Assignment-2

python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

export PYTHONPATH="$PWD"
bash run_all.sh
```

The pipeline runs: saved split → tree and bagging → boosting → bias–variance experiments → figures. The first run downloads the dataset. The bias–variance experiment may take an hour or more on a laptop.

| Files | Contents |
|---|---|
| `src/`, `scripts/` | Implementations and experiment scripts |
| `configs/` | Settings and frozen model choices |
| `data/split.npz` | Shared train/validation/test split |
| `logs/`, `results/` | Execution logs and result tables |
| `figures/`, `report/` | Plots and final PDF |
| `run_all.sh` | Complete reproduction pipeline |
