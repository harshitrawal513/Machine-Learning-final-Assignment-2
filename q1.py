import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import *

D = get_sets()
Xtr, ytr, Xva, yva, Xte, yte, Xte_m = (D[k] for k in
    ["Xtr", "ytr", "Xva", "yva", "Xte", "yte", "Xte_m"])
n = len(Xtr)

# ---------------- (a) single tree grid ----------------
rows = []
for d in [2, 4, 8, 16, None]:
    for m in [1, 2, 4]:
        t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=SEED).fit(Xtr, ytr)
        nl, mps, f1 = leaf_stats(t, Xtr)
        rows.append(dict(max_depth=str(d), min_leaf=m, train=t.score(Xtr, ytr),
                         val=t.score(Xva, yva), test=t.score(Xte, yte),
                         leaves=nl, mean_per_leaf=mps, frac_single=f1,
                         actual_depth=t.get_depth()))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n(a) grid\n", df.round(4))
df.to_csv("q1a_grid.csv", index=False)

best = df.sort_values(["val", "leaves"], ascending=[False, True]).iloc[0]   # validation only
depth = None if best.max_depth == "None" else int(best.max_depth)
params = dict(max_depth=depth, min_samples_leaf=int(best.min_leaf))
print("\nFROZEN single tree:", params)
print(best)

# ---------------- (b) bagging ----------------
BS = [1, 5, 10, 25, 50, 100, 200]
bag = Bagging(200, params, SEED).fit(Xtr, ytr)
P = {"tr": bag.preds(Xtr), "va": bag.preds(Xva), "te": bag.preds(Xte), "te_m": bag.preds(Xte_m)}
theory = n * (1 - (1 - 1 / n) ** n)

rows = []
for B in BS:
    Vtr = votes_from_preds(P["tr"][:B]); Vva = votes_from_preds(P["va"][:B])
    Vte = votes_from_preds(P["te"][:B])
    # OOB
    Vo = np.zeros((n, K))
    for b in range(B):
        m = ~bag.inbag[b]
        Vo[np.where(m)[0], P["tr"][b][m]] += 1
    cov = Vo.sum(1) > 0
    oob = (Vo.argmax(1) == ytr)[cov].mean()
    rows.append(dict(B=B,
                     train=(Vtr.argmax(1) == ytr).mean(),
                     val=(Vva.argmax(1) == yva).mean(),
                     test=(Vte.argmax(1) == yte).mean(),
                     distinct=bag.distinct[:B].mean(), theory=theory,
                     oob_acc=oob, oob_coverage=cov.mean(),
                     disagreement=np.nanmean(disagree_each(Vva)),   # measured on validation set
                     margin=mean_margin(Vva)))                      # measured on validation set
db = pd.DataFrame(rows)
print("\n(b),(c) bagging table\n", db.round(4))
db.to_csv("q1bc_bagging.csv", index=False)

# ---------------- (c) plots + final B ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col, ttl in zip(ax, ["oob_acc", "disagreement", "margin"],
                       ["OOB accuracy", "Avg pairwise disagreement (val)", "Mean vote margin (val)"]):
    a.plot(db.B, db[col], "o-"); a.set_xscale("log"); a.set_xlabel("B"); a.set_title(ttl); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("q1c_plots.png", dpi=200); plt.close()

Bf = int(db.sort_values(["val", "B"], ascending=[False, True]).iloc[0].B)
print("\nFINAL ensemble size (validation):", Bf)
json.dump(dict(depth=depth, min_leaf=params["min_samples_leaf"], B_final=Bf), open("frozen.json", "w"))

# ---------------- (d) tree vs bagging, clean vs masked ----------------
tree = DecisionTreeClassifier(random_state=SEED, **params).fit(Xtr, ytr)


def top3(V, i):
    return ", ".join(f"class {c}: {int(V[i, c])}" for c in np.argsort(-V[i])[:3])


for tag, X, key in [("clean", Xte, "te"), ("masked", Xte_m, "te_m")]:
    tp = tree.predict(X)
    V = votes_from_preds(P[key][:Bf]); bp = V.argmax(1)
    print(f"\n(d) {tag} test: tree acc={np.mean(tp==yte):.4f} conf={tree.predict_proba(X).max(1).mean():.4f} | "
          f"bagging acc={np.mean(bp==yte):.4f} conf(mean winning vote frac)={np.mean(V.max(1)/Bf):.4f}")
    fixed = np.where((tp != yte) & (bp == yte))[0][:2]
    both = np.where((tp != yte) & (bp != yte))[0][:2]
    cases = [("bagging corrects tree", i) for i in fixed] + [("both wrong", i) for i in both]
    fig, ax = plt.subplots(1, max(len(cases), 2), figsize=(3.2 * max(len(cases), 2), 3.6))
    for a, (lab, i) in zip(np.ravel(ax), cases):
        print(f"  [{lab}] test idx {i}: true={yte[i]}, tree={tp[i]}, bagging votes -> {top3(V, i)}")
        a.imshow(X[i].reshape(64, 64), cmap="gray"); a.axis("off")
        a.set_title(f"{lab}\ntrue {yte[i]} tree {tp[i]} bag {bp[i]}", fontsize=8)
    plt.tight_layout(); plt.savefig(f"q1d_cases_{tag}.png", dpi=200); plt.close()