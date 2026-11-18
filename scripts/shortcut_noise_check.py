"""SIMCT shortcut check (val only): how well do simple outer-region statistics (r >= 16 px, sources clipped) separate
lenses from non-lenses, and does the ensemble score track them within each class?
Features per band: lag-1 x/y autocorrelation of clipped outer pixels, robust sigma, median level.
Usage: python scripts/shortcut_noise_check.py run1 run2 ...  (ensemble = mean snapshot-ens TTA8 logits)"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
k = pd.read_parquet(os.path.join(W, "data/keys_trainval.parquet"))
y = np.load(os.path.join(W, "data/y_trainval.npy"))
vidx = np.where((k.split == "val").values)[0]
kv = k.iloc[vidx].reset_index(drop=True)
d2 = (kv.pipeline == "D2").values
X = np.load(os.path.join(W, "data/X_trainval.npy"), mmap_mode="r")
yy, xx = np.mgrid[:41, :41]
outer = np.hypot(yy - 20, xx - 20) >= 16

rows = np.sort(vidx[d2])
xs = np.asarray(X[rows], np.float32)                       # (n,5,41,41) ~ 8.5k stamps, ~290 MB
feat, names = [], []
for b in range(5):
    im = xs[:, b]
    o = im[:, outer]
    med = np.median(o, 1)
    sig = 1.4826 * np.median(np.abs(o - med[:, None]), 1)
    z = (im - med[:, None, None]) / sig[:, None, None]
    m = (np.abs(z) < 3) & outer[None]                       # clip sources, keep outer region
    mx = m[:, :, 1:] & m[:, :, :-1]
    my = m[:, 1:, :] & m[:, :-1, :]
    ax_ = (z[:, :, 1:] * z[:, :, :-1] * mx).sum((1, 2)) / np.maximum((z[:, :, 1:] ** 2 * mx).sum((1, 2)), 1e-9)
    ay_ = (z[:, 1:, :] * z[:, :-1, :] * my).sum((1, 2)) / np.maximum((z[:, 1:, :] ** 2 * my).sum((1, 2)), 1e-9)
    frac = 1 - m[:, outer].mean(1)                          # fraction of outer pixels with |z| >= 3 (other sources)
    feat += [ax_, ay_, np.log(sig), med / sig, frac]
    names += [f"acx_{'grizy'[b]}", f"acy_{'grizy'[b]}", f"logsig_{'grizy'[b]}", f"bkg_{'grizy'[b]}",
              f"srcfrac_{'grizy'[b]}"]
F = np.stack(feat, 1)
pos = np.searchsorted(vidx, rows)                           # positions in the val arrays
yv = y[rows].astype(int)

print("SIMCT val, outer region r >= 16 px, single-feature AUC (|AUC-0.5| sorted):")
au = [(roc_auc_score(yv, F[:, j]), n) for j, n in enumerate(names)]
for a, n in sorted(au, key=lambda t: -abs(t[0] - 0.5))[:12]:
    print(f"  {n:10s} {a:.3f}")
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
groups = {"autocorr only": [n.startswith("ac") for n in names],
          "noise sigma + bkg level": [n.startswith(("logsig", "bkg")) for n in names],
          "other-source fraction": [n.startswith("srcfrac") for n in names],
          "all": [True] * len(names)}
for g, sel in groups.items():
    p = cross_val_predict(clf, F[:, np.array(sel)], yv, cv=5, method="predict_proba")[:, 1]
    print(f"5-fold logistic on {g:24s}: AUC {roc_auc_score(yv, p):.3f}")

if len(sys.argv) > 1:
    zl = np.mean([np.load(os.path.join(W, "experiments", r, "val_logits_snapens_tta8.npy")).ravel()
                  for r in sys.argv[1:]], 0)[pos]
    pa = cross_val_predict(clf, F[:, np.array(groups["autocorr only"])], yv, cv=5, method="predict_proba")[:, 1]
    for c, nm in ((1, "lenses"), (0, "non-lenses")):
        s = yv == c
        print(f"Spearman(ensemble logit, autocorr-only shortcut score) within {nm}: {spearmanr(zl[s], pa[s])[0]:+.3f}")
