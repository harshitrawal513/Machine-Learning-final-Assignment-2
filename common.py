import os, time
import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

SEED = 26504          # numeric characters of M26ECM504, in order
K = 40


def mask_block(X):
    M = X.reshape(-1, 64, 64).copy()
    M[:, 24:40, 24:40] = 0.0
    return M.reshape(len(X), -1)


def load_data():
    d = fetch_olivetti_faces(shuffle=False)
    X = d.data.astype(np.float64)
    y = d.target.astype(int)
    if os.path.exists("split.npz"):
        s = np.load("split.npz")
        tr, va, te = s["train"], s["val"], s["test"]
    else:
        rng = np.random.default_rng(SEED)
        tr, va, te = [], [], []
        for c in range(K):
            p = rng.permutation(np.where(y == c)[0])
            tr += list(p[:6]); va += list(p[6:8]); te += list(p[8:10])
        tr, va, te = np.array(tr), np.array(va), np.array(te)
        np.savez("split.npz", train=tr, val=va, test=te)
    return X, y, tr, va, te


def get_sets():
    X, y, tr, va, te = load_data()
    D = dict(Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va], Xte=X[te], yte=y[te])
    D["Xte_m"] = mask_block(D["Xte"])
    return D


# ---------- vote helpers ----------
def votes_from_preds(P):
    B, n = P.shape
    V = np.zeros((n, K))
    for b in range(B):
        V[np.arange(n), P[b]] += 1
    return V


def disagree_each(V):
    B = V.sum(1)
    if B[0] < 2:
        return np.full(len(V), np.nan)
    return 1 - (V * (V - 1)).sum(1) / (B * (B - 1))


def mean_margin(V):
    F = np.sort(V / V.sum(1, keepdims=True), axis=1)
    return float(np.mean(F[:, -1] - F[:, -2]))


def leaf_stats(tree, Xtr):
    leaves = tree.apply(Xtr)
    _, counts = np.unique(leaves, return_counts=True)
    return len(counts), float(counts.mean()), float((counts == 1).mean())


# ---------- bagging from scratch ----------
def _fit_tree(Xb, yb, params, seed):
    return DecisionTreeClassifier(random_state=seed, **params).fit(Xb, yb)


class Bagging:
    def __init__(self, B, params, seed):
        self.B, self.params, self.seed = B, params, seed

    def fit(self, X, y, n_jobs=-1):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        idxs, seeds = [], []
        for _ in range(self.B):                      # first B trees do not depend on B
            idxs.append(rng.integers(0, n, n))
            seeds.append(int(rng.integers(0, 2**31 - 1)))
        self.trees = Parallel(n_jobs=n_jobs)(
            delayed(_fit_tree)(X[i], y[i], self.params, s) for i, s in zip(idxs, seeds))
        self.inbag = np.zeros((self.B, n), dtype=bool)
        for b, i in enumerate(idxs):
            self.inbag[b, i] = True
        self.distinct = self.inbag.sum(1)
        return self

    def preds(self, X, B=None):
        B = self.B if B is None else B
        return np.array([t.predict(X) for t in self.trees[:B]]).astype(int)

    def votes(self, X, B=None):
        return votes_from_preds(self.preds(X, B))


# ---------- SAMME from scratch ----------
class SAMME:
    def __init__(self, T, depth, K=K, seed=0, verbose=False):
        self.T, self.depth, self.K, self.seed, self.verbose = T, depth, K, seed, verbose

    def fit(self, X, y):
        n, Kc = len(X), self.K
        w = np.full(n, 1.0 / n)
        self.learners, self.alphas = [], []
        self.hist = dict(err=[], alpha=[], maxw=[], top10=[], W=[], pred=[])
        self.stopped = None
        if self.verbose:
            print("initial weights:", np.round(w, 4))
        for t in range(self.T):
            h = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed + t)
            h.fit(X, y, sample_weight=w)
            p = h.predict(X)
            miss = (p != y)
            err = w[miss].sum()
            if self.verbose:
                print(f"--- round {t+1} ---")
                print("true labels      :", y)
                print("weak predictions :", p)
                print("misclassified    :", miss.astype(int))
                print(f"weighted error   : {err:.4f}  (random level = {1-1/Kc:.4f})")
            if err >= 1 - 1.0 / Kc:                  # not better than random guessing
                self.stopped = (f"round {t+1}: error {err:.3f} >= {1-1/Kc:.3f}, "
                                "learner discarded and boosting stopped")
                if self.verbose:
                    print(self.stopped)
                break
            eps = max(err, 1e-10)                    # avoid log(0) for a perfect learner
            a = np.log((1 - eps) / eps) + np.log(Kc - 1)
            self.learners.append(h); self.alphas.append(a)
            self.hist["err"].append(err); self.hist["alpha"].append(a)
            self.hist["W"].append(w.copy()); self.hist["pred"].append(p)
            self.hist["maxw"].append(w.max())
            self.hist["top10"].append(np.sort(w)[-10:].sum())
            w = w * np.exp(a * miss)
            w = w / w.sum()
            if self.verbose:
                print(f"alpha            : {a:.4f}")
                print("updated weights  :", np.round(w, 4), " sum =", round(w.sum(), 6))
            if err == 0:
                self.stopped = f"round {t+1}: perfect training fit, stopped"
                break
        return self

    def _T(self, T):
        return len(self.alphas) if T is None else min(T, len(self.alphas))

    def scores(self, X, T=None):
        T = self._T(T)
        S = np.zeros((len(X), self.K))
        for t in range(T):
            S[np.arange(len(X)), self.learners[t].predict(X)] += self.alphas[t]
        return S

    def proba(self, X, T=None):
        T = self._T(T)
        return self.scores(X, T) / np.sum(self.alphas[:T])

    def predict(self, X, T=None):
        return self.proba(X, T).argmax(1)