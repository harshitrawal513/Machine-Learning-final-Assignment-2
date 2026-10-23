Bagging and Boosting on Olivetti Faces
Machine Learning — Assignment 2
Harshid Bhupendra Rawal | M26ECM504

This project compares a single decision tree with bagging and SAMME boosting on Olivetti Faces. Both ensembles are implemented from scratch using DecisionTreeClassifier as the base learner.
Dataset and setup
- 400 images across 40 identities, using all 4096 raw pixels without dimensionality reduction.
- Seed: 26504, taken from the numeric characters in the roll number.
- Per identity: 6 training, 2 validation and 2 test images. The saved split is reused throughout.
- Masked test images have rows and columns 24:40 set to zero. Training uses clean images only.
- Hyperparameters are selected on validation accuracy and frozen before test evaluation.
Results
Model	Train	Validation	Clean test	Masked test
Single tree (depth=None, min leaf=2)	0.9958	0.5875	0.6250	0.1875
Bagging (200 trees)	1.0000	0.8875	0.8875	0.7750
SAMME (depth=4, 100 rounds)	0.9917	0.3500	0.4875	0.4250


Bagging performed best on both test sets. Across 30 resampled training sets, it reduced the measured prediction variance by about 92–96%, depending on tree depth. Boosting showed a large gap between training and validation accuracy.

Each test set contains 80 images, so one correct prediction changes accuracy by 1.25 percentage points.
Run
Tested with Python 3.8.10. Dependencies are pinned in requirements.txt.
git clone https://github.com/harshitrawal513/Machine-Learning-final-Assignment-2.git
cd Machine-Learning-final-Assignment-2
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
bash run_all.sh

The first run downloads the dataset. The bias–variance experiment may take an hour or more on a laptop.
Project files
Folder	Contents
src/	Data loading, metrics, bagging and SAMME
scripts/	Split → Q1 bagging → Q2 boosting → Q3 bias–variance → figures
configs/	Experiment settings and frozen choices
data/	Saved split
logs/, results/	Execution logs and CSV results
figures/, report/	Plots and final PDF


Bias–variance note: The tree uses one-hot predictions, while the ensembles use vote fractions. Their bias and variance values therefore need care when comparing across model types.
