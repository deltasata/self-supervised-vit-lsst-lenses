"""Compare robustness of base vs augmented models (val only). Pairs as base:aug. Thresholds fixed per pipeline on each
model's clean scores at FPR 1e-3. Usage: python scripts/robust_compare.py base1:aug1 base2:aug2 ... [> out.md]"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "scripts")
from robustness_eval import CONDS, OUT  # noqa: E402

k = pd.read_parquet("data/keys_trainval.parquet")
y = np.load("data/y_trainval.npy")
va = (k.split == "val").values
yv, p = y[va].astype(int), k.pipeline.values[va]
bce = lambda z, t: float(np.mean(np.logaddexp(0, z) - t * z))


def stats(r, c):
    z, zc = np.load(f"{OUT}/{r}/{c}.npy"), np.load(f"{OUT}/{r}/clean.npy")
    thr = {g: np.quantile(zc[(p == g) & (yv == 0)], 1 - 1e-3) for g in ("D1", "D2")}
    m1, m2 = p == "D1", p == "D2"
    return (bce(z, yv), roc_auc_score(yv[m2], z[m2]), np.mean(z[m1 & (yv == 0)] > thr["D1"]),
            np.mean(z[m2 & (yv == 1)] > thr["D2"]))


for pair in sys.argv[1:]:
    b, a = pair.split(":")
    print(f"\n## {b} (base) vs {a} (aug)\n")
    print("| condition | loss base | loss aug | SIMCT AUC base | aug | SLSim FPR base | aug | SIMCT TPR base | aug |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in CONDS:
        x, w = stats(b, c), stats(a, c)
        print(f"| {c} | {x[0]:.4f} | {w[0]:.4f} | {x[1]:.5f} | {w[1]:.5f} | {x[2]:.4f} | {w[2]:.4f} | {x[3]:.3f} | {w[3]:.3f} |")
