"""k-fold out-of-fold (OOF) report, val only. For each recipe: per-fold loss on its own fold,
OOF loss / TS / AUC / TPR over all finished folds, and on the OLD val rows inside finished folds the k-fold model vs
the original frozen run of the same recipe (same rows; the k-fold model trained on 153k incl. old-train+old-val rows,
the original on old train only). Writes results/kfold/oof_<recipe>.npy (NaN where the fold is not finished).
Usage: python scripts/kfold_report.py"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = sys.argv[:1]
from leaderboard_now import bce, tpr  # noqa: E402

REC = {"vit_plain": "mae_vitb_p3_42_in1kmae_e400_ft", "vit_rob": "e400_robaug", "cnn_plain": "mimv2b_llrd75",
       "cnn_rob": "mimv2b_robaug",
       "cnn400_plain": "mimF_v2b_e400_ft", "cnn400_rob": "mimF_v2b_e400_ft"}
k = pd.read_parquet("data/keys_trainval.parquet")
y = np.load("data/y_trainval.npy").astype(float)
d2 = k.pipeline.values == "D2"
old_va = np.where((k.split == "val").values)[0]
pos_in_va = {r: i for i, r in enumerate(old_va)}
os.makedirs("results/kfold", exist_ok=True)
rows = []
for rec, orig in REC.items():
    oof = np.full(len(y), np.nan)
    zo = np.load(f"experiments/{orig}/val_logits_tta8.npy").ravel()
    fl = []
    for f in range(10):
        d = f"experiments/kf_{rec}_f{f}"
        if not os.path.exists(f"{d}/metrics.json"):
            continue
        vr = np.load(f"{d}/val_rows.npy")
        z = np.load(f"{d}/val_logits_tta8.npy").ravel()
        oof[vr] = z
        fl.append(f"f{f} {bce(z, y[vr]):.4f}")
    np.save(f"results/kfold/oof_{rec}.npy", oof.astype(np.float32))
    m = ~np.isnan(oof)
    if not m.any():
        continue
    mo = m[old_va]                                           # old-val rows already covered by finished folds
    zk, zov, yo = oof[old_va][mo], zo[mo], y[old_va][mo]
    rows.append(dict(recipe=rec, folds=len(fl), n_oof=int(m.sum()), oof_loss=round(bce(oof[m], y[m]), 4),
                     oof_auc=round(roc_auc_score(y[m], oof[m]), 5),
                     oof_auc_SIMCT=round(roc_auc_score(y[m & d2], oof[m & d2]), 5),
                     oof_TPR3_SIMCT=round(tpr(oof[m & d2], y[m & d2]), 3),
                     oldval_n=int(mo.sum()), oldval_kfold=round(bce(zk, yo), 4), oldval_orig=round(bce(zov, yo), 4),
                     per_fold=" ".join(fl)))
print(pd.DataFrame(rows).to_string(index=False))
