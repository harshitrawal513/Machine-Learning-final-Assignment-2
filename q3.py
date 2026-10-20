import json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from common import *

fz = json.load(open("frozen.json")); fz2 = json.load(open("frozen2.json"))
sel = fz["depth"]; mleaf = fz["min_leaf"]; Bf = fz["B_final"]
dsel, Tsel = fz2["depth"], fz2["rounds"]
D = get_sets()
Xtr, ytr, Xte, yte, Xte_m = (D[k] for k in ["Xtr", "ytr", "Xte", "yte", "Xte_m"])
N, R = len(Xte), 30
E = np.eye(K)[yte]


def P_tree(depth):
    return dict(max_depth=depth, min_samples_leaf=mleaf)


if sel is None:
    depth_list = [2, 8, None]          # selected is already the deepest, 8 is the middle one
elif sel == 2:
    depth_list = [1, 2, None]
else:
    depth_list = [2, sel, None]
print("depths used in (b):", depth_list)

# 30 resampled training sets: within each identity, 6 faces drawn with replacement from its 6
rng = np.random.default_rng(SEED)
by_id = [np.where(ytr == c)[0] for c in range(K)]
sets = [np.concatenate([rng.choice(p, 6, replace=True) for p in by_id]) for _ in range(R)]


def onehot(pred):
    M = np.zeros((len(pred), K)); M[np.arange(len(pred)), pred] = 1; return M


out = {"tree_sel": [], "bag_final": [], "boost": []}
for d in depth_list:
    out[f"tree_{d}"] = []; out[f"bag100_{d}"] = []
dis = {d: [] for d in depth_list}

for r, idx in enumerate(sets):
    Xr, yr = Xtr[idx], ytr[idx]
    out["tree_sel"].append(onehot(DecisionTreeClassifier(random_state=SEED + r, **P_tree(sel)).fit(Xr, yr).predict(Xte)))
    V = Bagging(Bf, P_tree(sel), SEED + r).fit(Xr, yr).votes(Xte); out["bag_final"].append(V / Bf)
    out["boost"].append(SAMME(Tsel, dsel, seed=SEED + r).fit(Xr, yr).proba(Xte))
    for d in depth_list:
        out[f"tree_{d}"].append(onehot(DecisionTreeClassifier(random_state=SEED + r, **P_tree(d)).fit(Xr, yr).predict(Xte)))
        V = Bagging(100, P_tree(d), SEED + r).fit(Xr, yr).votes(Xte)
        out[f"bag100_{d}"].append(V / 100); dis[d].append(np.mean(disagree_each(V)))
    print(f"repetition {r+1}/{R} done", flush=True)

out = {k: np.array(v) for k, v in out.items()}      # (R, N, K)


def bv(Pr):
    pbar = Pr.mean(0)
    bias2 = ((pbar - E) ** 2).sum(1).mean()
    var = ((Pr - pbar) ** 2).sum(2).mean()
    accs = (Pr.argmax(2) == yte).mean(1)
    return bias2, var, accs.mean(), accs.std(ddof=1)


# ---------------- (a) ----------------
print("\n(a)")
ra = []
for name, key in [("single tree", "tree_sel"), (f"bagging B={Bf}", "bag_final"), ("boosting", "boost")]:
    b, v, am, asd = bv(out[key]); ra.append(dict(model=name, bias2=b, variance=v, acc_mean=am, acc_std=asd))
dfa = pd.DataFrame(ra); print(dfa.round(4)); dfa.to_csv("q3a.csv", index=False)

# ---------------- (b) + (c) ----------------
print("\n(b)/(c)")
rb = []
for d in depth_list:
    bt, vt, at, st = bv(out[f"tree_{d}"]); bb, vb, ab, sb = bv(out[f"bag100_{d}"])
    t = DecisionTreeClassifier(random_state=SEED, **P_tree(d)).fit(Xtr, ytr)
    nl, mps, f1 = leaf_stats(t, Xtr)
    rb.append(dict(depth=str(d), tree_bias2=bt, tree_var=vt, bag_bias2=bb, bag_var=vb,
                   var_removed=vt - vb, var_removed_pct=100 * (vt - vb) / vt,
                   tree_acc=at, bag_acc=ab, acc_gain=ab - at,
                   leaves=nl, mean_per_leaf=mps, frac_single_leaf=f1,
                   bag_disagreement=np.mean(dis[d])))
dfb = pd.DataFrame(rb); pd.set_option("display.width", 250); print(dfb.round(4).T)
dfb.to_csv("q3bc.csv", index=False)

# ---------------- (d) ----------------
print("\n(d) final table")
tree = DecisionTreeClassifier(random_state=SEED, **P_tree(sel)).fit(Xtr, ytr)
bag = Bagging(Bf, P_tree(sel), SEED).fit(Xtr, ytr)
boost = SAMME(Tsel, dsel, seed=SEED).fit(Xtr, ytr)
clean = [np.mean(tree.predict(Xte) == yte), np.mean(bag.votes(Xte).argmax(1) == yte), np.mean(boost.predict(Xte) == yte)]
masked = [np.mean(tree.predict(Xte_m) == yte), np.mean(bag.votes(Xte_m).argmax(1) == yte), np.mean(boost.predict(Xte_m) == yte)]
dfd = pd.DataFrame(dict(model=["single tree", f"bagging B={Bf}", "boosting"], clean_acc=clean, masked_acc=masked,
                        bias2=dfa.bias2.values, variance=dfa.variance.values))
dfd["acc_drop"] = dfd.clean_acc - dfd.masked_acc
print(dfd.round(4)); dfd.to_csv("q3d_final_table.csv", index=False)