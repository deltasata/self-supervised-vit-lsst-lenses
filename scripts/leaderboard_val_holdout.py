"""Leaderboard: val and holdout rows per model (best.pt, TTA8 — the predictor scored on the holdout)
and for the four frozen final ensembles (k-fold ensembles: val = out-of-fold over all 170k train+val rows; k-fold
members: val = their own fold, column val_set). Reads only stored scores: experiments/<run>/val_logits_tta8.npy and
results/holdout_predictions/ (written by eval_holdout.py). REPORT ONLY: nothing here is used for model selection.
Writes results/holdout_predictions/leaderboard_val_holdout.csv (all runs). Usage: python scripts/leaderboard_val_holdout.py"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leaderboard_now as lb  # noqa: E402

HP = "results/holdout_predictions"
yh = np.load(f"{HP}/holdout_y.npy").astype(float)
d2h = pd.read_parquet(f"{HP}/holdout_keys.parquet").pipeline.values == "D2"
yv, d2v = lb.yv, lb.d2
ENS = ["mae_vitb_p3_42_in1kmae_e800_ft", "mae_vitb_p3_42_in1kmae_e400_ft", "mae_in1kB_e400_s1_ft",
       "mae_in1kB_e800_s1_ft", "mimv2b_llrd75"]


def met(z, y, d2):
    return dict(loss=round(lb.bce(z, y), 4), AUC=round(roc_auc_score(y, z), 5),
                AUC_SIMCT=round(roc_auc_score(y[d2], z[d2]), 5), TPR3_all=round(lb.tpr(z, y), 3),
                TPR3_SIMCT=round(lb.tpr(z[d2], y[d2]), 3))


def kind(r):
    if r.startswith("dual_"):
        return "hybrid ViT+CNN"
    a = __import__("json").load(open(f"experiments/{r}/config.json"))["model"].get("arch", "")
    return "transformer" if any(s in a for s in ("vit", "deit", "eva", "beit", "swin", "hiera", "maxvit", "coat",
                                                     "davit", "gcvit", "xcit", "caformer", "aion", "clip", "dino",
                                                     "band", "point")) else "CNN"


rows = []
for r in sorted(os.listdir(HP)):
    fh, fv = f"{HP}/{r}/holdout_logits_tta8.npy", f"experiments/{r}/val_logits_tta8.npy"
    if not (os.path.exists(fh) and os.path.exists(fv)):
        continue
    zv, zh = np.load(fv).ravel(), np.load(fh).ravel()
    tv, dv = lb.labels(r)                                  # k-fold members: their own fold rows
    if len(zv) != len(tv) or not np.isfinite(zv).all() or not np.isfinite(zh).all():
        continue
    vs = lb.val_set(r)
    rows.append(dict(model=r, type=kind(r), params_M=lb.params(r), set="val", val_set=vs, **met(zv, tv, dv)))
    rows.append(dict(model=r, type=kind(r), params_M=lb.params(r), set="holdout", val_set="", **met(zh, yh, d2h)))
from kfold_ens import oof  # noqa: E402
yall = lb.y.astype(float)
d2all = lb.k.pipeline.values == "D2"
ROB6 = ["e800_robaug", "e400_robaug", "e400_robaug_strong", "e400_robaug_s1", "e800_robaug_s1", "mimv2b_robaug"]
for name, kind_, val in (
        ("final_kfold_plain", "kfold", ["vit_plain", "cnn400_plain"]), ("final_kfold_robust", "kfold", ["vit_rob", "cnn400_rob"]),
        ("final_best5", "std", ENS), ("final_robust6", "std", ROB6)):
    if kind_ == "kfold":
        z = np.mean([oof(r, len(yall)) for r in val], 0)
        m = np.isfinite(z)
        mv, vs = met(z[m], yall[m], d2all[m]), f"OOF {int(m.sum())}"
    else:
        mv, vs = met(np.mean([np.load(f"experiments/{r}/val_logits_tta8.npy").ravel() for r in val], 0), yv, d2v), "val"
    zh = np.load(f"{HP}/ENS_{name}/holdout_logits.npy").ravel()
    tag = f"ENSEMBLE {name} (frozen)" + (" = FINAL" if name == "final_kfold_plain" else "")
    rows.append(dict(model=tag, type="ensemble", params_M="-", set="val", val_set=vs, **mv))
    rows.append(dict(model=tag, type="ensemble", params_M="-", set="holdout", val_set="", **met(zh, yh, d2h)))
df = pd.DataFrame(rows)
vl = df[df.set == "val"].set_index("model").loss
df["val_rank_loss"] = df.model.map(vl)
df = df.sort_values(["val_rank_loss", "model", "set"], ascending=[True, True, False]).drop(columns="val_rank_loss")
df.to_csv(f"{HP}/leaderboard_val_holdout.csv", index=False)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for t in ("ensemble", "transformer", "hybrid ViT+CNN", "CNN"):
    s = df[df.type == t]
    keep = list(dict.fromkeys(s.model))[:n]
    print(f"\n## {t}\n" + s[s.model.isin(keep)].drop(columns="type").to_string(index=False))
print("\nNote: val_set foldK = a k-fold member scored on its own fold (17k rows, not the standard val split): fold losses"
      " are not comparable with standard-val losses or across folds.")
print(f"\n{df.model.nunique()} models; full table: {HP}/leaderboard_val_holdout.csv")
