import json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from common import *

fz = json.load(open("frozen.json"))
params = dict(max_depth=fz["depth"], min_samples_leaf=fz["min_leaf"]); Bf = fz["B_final"]
D = get_sets()
Xtr, ytr, Xva, yva, Xte, yte, Xte_m = (D[k] for k in
    ["Xtr", "ytr", "Xva", "yva", "Xte", "yte", "Xte_m"])
n = len(Xtr)

# ---------------- (a) toy 3-class SAMME round ----------------
print("=========== (a) toy example ===========")
Xt = np.array([[1, 1], [1, 2], [2, 1], [5, 5], [5, 6], [6, 5], [9, 1], [9, 2], [8, 1]], float)
yt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
SAMME(T=1, depth=1, K=3, seed=SEED, verbose=True).fit(Xt, yt)

# ---------------- (b) depth x rounds ----------------
print("\n=========== (b) selection ===========")
depths, rounds = [1, 2, 3, 4], [10, 25, 50, 100]
models, rows = {}, []
for d in depths:
    m = SAMME(100, d, seed=SEED).fit(Xtr, ytr)
    models[d] = m
    if m.stopped: print(f"depth {d}: {m.stopped}")
    for T in rounds:
        rows.append(dict(depth=d, rounds=T, used=min(T, len(m.alphas)),
                         train=np.mean(m.predict(Xtr, T) == ytr),
                         val=np.mean(m.predict(Xva, T) == yva)))
dfb = pd.DataFrame(rows); print(dfb.round(4)); dfb.to_csv("q2b_grid.csv", index=False)
bb = dfb.sort_values(["val", "rounds", "depth"], ascending=[False, True, True]).iloc[0]
dsel, Tsel = int(bb.depth), int(bb.rounds)
print("SELECTED boosting: depth", dsel, "rounds", Tsel)
json.dump(dict(depth=dsel, rounds=Tsel), open("frozen2.json", "w"))
boost = models[dsel]
Tused = min(Tsel, len(boost.alphas))
H = boost.hist
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for a, key, ttl in zip(ax.ravel(), ["err", "alpha", "maxw", "top10"],
        ["Weighted training error", "Learner weight alpha", "Max sample weight", "Weight of 10 heaviest samples"]):
    a.plot(range(1, Tused + 1), H[key][:Tused]); a.set_title(ttl); a.set_xlabel("round"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("q2b_plots.png", dpi=200); plt.close()

# ---------------- (c) boosting vs bagging ----------------
print("\n=========== (c) boosting vs bagging ===========")
rows = []
for T in [25, 50, 100]:
    t0 = time.perf_counter()
    bg = Bagging(T, params, SEED).fit(Xtr, ytr, n_jobs=1)       # n_jobs=1 for a fair timing
    tb = time.perf_counter() - t0
    Vtr, Vva, Vte = (bg.votes(X) for X in (Xtr, Xva, Xte))
    rows.append(dict(model="bagging", T=T, train=np.mean(Vtr.argmax(1) == ytr),
                     val=np.mean(Vva.argmax(1) == yva), test=np.mean(Vte.argmax(1) == yte), seconds=tb))
    t0 = time.perf_counter()
    bo = SAMME(T, dsel, seed=SEED).fit(Xtr, ytr)
    tb = time.perf_counter() - t0
    rows.append(dict(model="boosting", T=T, train=np.mean(bo.predict(Xtr) == ytr),
                     val=np.mean(bo.predict(Xva) == yva), test=np.mean(bo.predict(Xte) == yte), seconds=tb))
dfc = pd.DataFrame(rows); print(dfc.round(4)); dfc.to_csv("q2c_compare.csv", index=False)

Wm = np.array(H["W"])[:Tused]; Pm = np.array(H["pred"])[:Tused]
i_star = int(Wm.max(0).argmax())
wrong = Pm[:, i_star] != ytr[i_star]
print(f"\nHeavy sample: train idx {i_star}, identity {ytr[i_star]}")
print(f"max weight {Wm[:, i_star].max():.4f} (uniform = {1/n:.4f}) at round {Wm[:, i_star].argmax()+1}")
print(f"misclassified in {wrong.sum()} of {Tused} rounds")
print("first 30 predictions:", Pm[:30, i_star].tolist())
vals, cnt = np.unique(Pm[wrong, i_star], return_counts=True)
print("most frequent wrong labels:", dict(zip(vals[np.argsort(-cnt)][:3].tolist(), np.sort(cnt)[::-1][:3].tolist())))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].imshow(Xtr[i_star].reshape(64, 64), cmap="gray"); ax[0].axis("off"); ax[0].set_title(f"train idx {i_star}, id {ytr[i_star]}")
ax[1].plot(range(1, Tused + 1), Wm[:, i_star]); r = np.where(wrong)[0] + 1
ax[1].scatter(r, Wm[wrong, i_star], c="r", s=14, label="misclassified"); ax[1].legend()
ax[1].set_xlabel("round"); ax[1].set_ylabel("sample weight"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("q2c_heavy_sample.png", dpi=200); plt.close()

# ---------------- (d) clean vs masked ----------------
print("\n=========== (d) clean vs masked ===========")
tree = DecisionTreeClassifier(random_state=SEED, **params).fit(Xtr, ytr)
bag = Bagging(Bf, params, SEED).fit(Xtr, ytr)
res = {}
for tag, X in [("clean", Xte), ("masked", Xte_m)]:
    V = bag.votes(X); q = boost.proba(X, Tsel)
    res[tag] = dict(V=V, q=q, tp=tree.predict(X), bp=V.argmax(1), qp=q.argmax(1))
    print(f"{tag}: tree acc {np.mean(res[tag]['tp']==yte):.4f} conf {tree.predict_proba(X).max(1).mean():.4f} | "
          f"bagging acc {np.mean(res[tag]['bp']==yte):.4f} conf {np.mean(V.max(1)/Bf):.4f} | "
          f"boosting acc {np.mean(res[tag]['qp']==yte):.4f} conf {q.max(1).mean():.4f}")

c, m = res["clean"], res["masked"]
cand = np.where((c["bp"] == yte) & (c["qp"] == yte) & (m["bp"] == yte) & (m["qp"] != yte))[0]
kind = "bagging stays right, boosting breaks"
if len(cand) == 0:
    cand = np.where((c["bp"] == yte) & (c["qp"] == yte) & (m["bp"] != yte) & (m["qp"] == yte))[0]
    kind = "boosting stays right, bagging breaks"
if len(cand) == 0:
    cand = np.where(m["bp"] != m["qp"])[0]; kind = "bagging and boosting disagree (masked)"
if len(cand) == 0:
    print("No differing face found."); raise SystemExit
i = int(cand[0]); print(f"\nChosen face: test idx {i}, true {yte[i]}  ({kind})")


def t3(v, f):
    return ", ".join(f"class {k}: {f(v[k])}" for k in np.argsort(-v)[:3])


print("bagging votes clean :", t3(c["V"][i], lambda x: int(x)))
print("bagging votes masked:", t3(m["V"][i], lambda x: int(x)))
print(f"tree disagreement clean {disagree_each(c['V'])[i]:.3f} -> masked {disagree_each(m['V'])[i]:.3f}")
print("boosting q clean :", t3(c["q"][i], lambda x: round(float(x), 3)))
print("boosting q masked:", t3(m["q"][i], lambda x: round(float(x), 3)))
al = np.array(boost.alphas)[:Tsel]
changed = [t for t, h in enumerate(boost.learners[:Tsel])
           if h.predict(Xte[i:i+1])[0] != h.predict(Xte_m[i:i+1])[0]]
print(f"{len(changed)} of {Tsel} boosting learners change their vote; they carry "
      f"{al[changed].sum()/al.sum():.3f} of total alpha")
ids = np.where(ytr == yte[i])[0]
Wall = np.array(H["W"])[:Tsel]
print("avg boosting weight of this identity's train faces (x uniform):", np.round(Wall[:, ids].mean(0) * n, 2))