"""Current val leaderboard (snapshot-ensemble TTA8 logits when present, else TTA8): log loss, cross-fit TS loss, AUC,
SIMCT AUC, TPR@FPR=1e-3 (all / SLSim / SIMCT), params from train.log. Val only. Usage: python scripts/leaderboard_now.py"""
import os
import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

k = pd.read_parquet("data/keys_trainval.parquet")
y = np.load("data/y_trainval.npy")
va = (k.split == "val").values
yv = y[va].astype(float)
d2 = (k.pipeline.values[va] == "D2") if "pipeline" in k else np.array([s.startswith("D2") for s in k.simset.values[va]])
p = np.random.RandomState(20261008).permutation(len(yv))
H = [p[:len(p) // 2], p[len(p) // 2:]]
VA_ROWS = np.where(va)[0]


def labels(r):
    """(labels, is_SIMCT) of the rows run r's val logits refer to. k-fold runs (train.fold) validate on their own fold
    (experiments/<r>/val_rows.npy), which has the same size as the standard val split (17,000) but different rows
"""
    fp = f"experiments/{r}/val_rows.npy"
    if os.path.exists(fp):
        rows = np.load(fp)
        if not np.array_equal(rows, VA_ROWS):
            return y[rows].astype(float), k.pipeline.values[rows] == "D2"
    return yv, d2


def val_set(r):
    """'val' (standard split) or 'fold<k>' (k-fold out-of-fold rows)."""
    t, _ = labels(r)
    if t is yv:
        return "val"
    import json
    return f"fold{json.load(open(f'experiments/{r}/config.json'))['train'].get('fold')}"


def bce(z, t):
    return float(np.mean(np.logaddexp(0, z) - t * z))


def ts(z, t=None):
    t = yv if t is None else t
    pp = np.random.RandomState(20261008).permutation(len(t))
    hh = [pp[:len(pp) // 2], pp[len(pp) // 2:]]
    o = np.empty_like(z)
    for a, b in [(0, 1), (1, 0)]:
        s = minimize_scalar(lambda s: bce(z[hh[a]] * s, t[hh[a]]), bounds=(0.3, 3), method="bounded").x
        o[hh[b]] = z[hh[b]] * s
    return bce(o, t)


def tpr(z, t, fpr=1e-3):
    thr = np.quantile(z[t == 0], 1 - fpr)
    return float(np.mean(z[t == 1] > thr))


def logits(r):
    for f in ("val_logits_snapens_tta8.npy", "val_logits_tta8.npy"):
        fp = f"experiments/{r}/{f}"
        if os.path.exists(fp):
            return np.load(fp).ravel()
    raise FileNotFoundError(r)


def params(r):
    """'params N.NNM' printed by train.py; *_ft runs of pretrain_mae / pretrain_mim_cnn log into the parent run."""
    for f in (f"experiments/{r}/train.log", f"experiments/{r[:-3]}/train.log" if r.endswith("_ft") else None):
        if f and os.path.exists(f):
            m = re.findall(r"params ([0-9.]+)M", open(f, errors="replace").read())
            if m:
                return f"{float(m[-1]):.1f}"
    return "?"


def row(name, z, par, t=None, dd=None):
    """Metrics of logits z against labels t / SIMCT mask dd (default: the standard val split)."""
    t, dd = (yv, d2) if t is None else (t, dd)
    return dict(model=name, params_M=par, loss=round(bce(z, t), 4), loss_TS=round(ts(z, t), 4),
                AUC=round(roc_auc_score(t, z), 5), AUC_SIMCT=round(roc_auc_score(t[dd], z[dd]), 5),
                TPR3_all=round(tpr(z, t), 3), TPR3_SLSim=round(tpr(z[~dd], t[~dd]), 3),
                TPR3_SIMCT=round(tpr(z[dd], t[dd]), 3))


if __name__ == "__main__":
    import sys
    rows = []
    for r in sys.argv[1:]:
        try:
            rows.append(row(r, logits(r), params(r), *labels(r)))
        except FileNotFoundError:
            print("no logits:", r)
    ens = ["mae_vitb_p3_42_in1kmae_e800_ft", "mae_vitb_p3_42_in1kmae_e400_ft", "mae_in1kB_e400_s1_ft",
           "mae_in1kB_e800_s1_ft", "mimv2b_llrd75"]
    rows.append(row("ENSEMBLE best5", np.mean([logits(r) for r in ens], 0), "-"))
    print(pd.DataFrame(rows).to_string(index=False))
