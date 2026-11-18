"""Ensemble rules evaluated honestly on val: choose/fit on fixed half A (same split as train.py sel_half), report half B
(and the reverse, averaged). Inputs: experiments/*/val_logits_tta8.npy of finished runs. Labels/keys: val rows only.
Rules: best single (picked on A), mean of top-k (ranked on A), greedy forward selection with replacement (on A),
logistic stacking (L2, fit on A). Output: results/ensemble/ensemble_val.json + printed table.
Full metrics (AUC, per pipeline, TPR@FPR, ...): each rule's predictions are cross-fitted (half B scored by the rule
chosen on A, half A by the rule chosen on B), then concatenated over all val rows -> results/ensemble/ensemble_val_metrics.json.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(W, "src"))
from lagn.metrics import all_metrics  # noqa: E402

keys = pd.read_parquet(os.path.join(W, "data/keys_trainval.parquet"))
y_all = np.load(os.path.join(W, "data/y_trainval.npy"))
va = np.where((keys.split == "val").values)[0]
y = y_all[va].astype(np.float64)
meta = keys.iloc[va].reset_index(drop=True)
perm = np.random.RandomState(20261008).permutation(len(va))
A, B = np.sort(perm[:len(va) // 2]), np.sort(perm[len(va) // 2:])


def bce(z, idx):
    z, t = z[idx].astype(np.float64), y[idx]
    return float(np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))))


runs = {}
for f in sorted(glob.glob(os.path.join(W, "experiments/*/val_logits_tta8.npy"))):
    r = f.split("/")[-2]
    if not os.path.exists(os.path.join(os.path.dirname(f), "metrics.json")) or r.startswith("diag") or r.startswith("sb_"):
        continue
    z = np.load(f).astype(np.float64)
    if len(z) == len(va) and np.isfinite(z).all():
        runs[r] = z
names = list(runs)
Z = np.stack([runs[n] for n in names], 1)


def rules(S, T):
    out = {}
    ls = np.array([bce(Z[:, i], S) for i in range(len(names))])
    order = np.argsort(ls)
    out["best_single"] = (bce(Z[:, order[0]], T), [names[order[0]]], Z[:, order[0]])
    for k in (3, 5, 7, 10):
        out[f"top{k}_mean"] = (bce(Z[:, order[:k]].mean(1), T), [names[i] for i in order[:k]], Z[:, order[:k]].mean(1))
    cur, pick = np.zeros(len(y)), []
    for _ in range(25):                                   # greedy with replacement on S
        cand = [bce((cur * len(pick) + Z[:, i]) / (len(pick) + 1), S) for i in range(len(names))]
        i = int(np.argmin(cand))
        if pick and cand[i] >= bce(cur, S) - 1e-6:
            break
        cur = (cur * len(pick) + Z[:, i]) / (len(pick) + 1)
        pick.append(i)
    out["greedy"] = (bce(Z[:, pick].mean(1), T), sorted({names[i] for i in pick}), Z[:, pick].mean(1))
    for C in (0.001, 0.003, 0.01, 0.03):
        lr = LogisticRegression(C=C, max_iter=2000).fit(Z[S], y[S])
        zt = lr.decision_function(Z)
        out[f"stack_C{C}"] = (bce(zt, T), None, zt)
    return out


ab, ba = rules(A, B), rules(B, A)
res = {k: {"B_given_A": ab[k][0], "A_given_B": ba[k][0], "mean": (ab[k][0] + ba[k][0]) / 2, "members_A": ab[k][1]}
       for k in ab}
res["_n_runs"] = len(names)
res["_optimistic_full_val_greedy_note"] = "selection on full val is optimistic; use 'mean' column"
json.dump(res, open(os.path.join(W, "results/ensemble/ensemble_val.json"), "w"), indent=1)
full = {}
for k in ab:
    zc = np.empty(len(y))
    zc[B], zc[A] = ab[k][2][B], ba[k][2][A]                 # cross-fitted: never scored by a rule chosen on its own half
    full[k] = all_metrics(y, 1 / (1 + np.exp(-zc)), meta)
    np.save(os.path.join(W, f"results/ensemble/cf_logits_{k}.npy"), zc.astype(np.float32))
json.dump(full, open(os.path.join(W, "results/ensemble/ensemble_val_metrics.json"), "w"), indent=1)
print(f"{len(names)} runs")
for k, v in res.items():
    if not k.startswith("_"):
        print(f"{k:14s} B|A {v['B_given_A']:.4f}  A|B {v['A_given_B']:.4f}  mean {v['mean']:.4f}  {v['members_A'] if v['members_A'] and len(v['members_A'])<12 else ''}")
print(f"\n{'rule':14s} logloss    AUC    AUC_D1   AUC_D2  tpr@1e-2 tpr@1e-3")
for k, m in full.items():
    print(f"{k:14s} {m['logloss']:.4f}  {m['auc']:.5f} {m['per_pipeline']['D1']:.5f} {m['per_pipeline']['D2']:.5f}  "
          f"{m['tpr@1e-2']:.3f}    {m['tpr@1e-3']:.3f}")
