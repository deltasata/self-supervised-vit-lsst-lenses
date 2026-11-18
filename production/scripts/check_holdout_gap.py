"""Diagnose why the former-holdout rows score worse in the partial production OOF. CPU, light."""
import glob, os, sys
import numpy as np, pandas as pd
import data_all
P = data_all.PROD; W = data_all.W
X, y, keys = data_all.load_all()
F = pd.read_parquet(os.path.join(P, "results", "folds.parquet"))

def bce_v(z, t):
    z = z.astype(np.float64); return np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))

print("== 1. integrity")
name_lab = keys.obj.str[3].map({"L": 1, "N": 0}).values
print("label == L/N in object name, all 200k:", (name_lab == y).all())
print("folds.parquet obj/label/fold aligned:", (F.obj.values == keys.obj.values).all(), (F.label.values == y).all(),
      (F.fold.values == data_all.folds(keys)).all())
Xh = np.load(os.path.join(W, "data", "holdout", "X_holdout.npy"), mmap_mode="r")
r = np.random.RandomState(0).choice(30000, 500, replace=False)
print("CatRows holdout rows == X_holdout (500 random):", np.array_equal(X[170000 + r], Xh[r]))
for k in range(10):
    tr = F.fold.values != k
    print(f"  fold {k}: train rows from former holdout {np.sum(tr & (F.source == 'holdout').values)}, "
          f"val rows from former holdout {np.sum(~tr & (F.source == 'holdout').values)}") if k < 3 else None

print("== 2. per-object loss: is the gap a few objects?")
ze = np.load(os.path.join(P, "results", "oof_logits_ensemble.npy")); m = ~np.isnan(ze)
L = bce_v(ze, y.astype(float))
for nm, s in [("old170k", m & (F.source == "trainval").values), ("hold30k", m & (F.source == "holdout").values)]:
    l = np.sort(L[s])[::-1]
    print(f" {nm}: n {s.sum()}  mean {l.mean():.4f}  top10 losses {np.round(l[:10], 1)}  "
          f"mean without top 5 {l[5:].mean():.4f}  without top 10 {l[10:].mean():.4f}  median {np.median(l):.5f}")
    print(f"   errors at p=0.5: {np.sum(l > np.log(2))}  rate {np.mean(l > np.log(2)):.5f}")
print("== 3. bootstrap: 9,122 random rows of the old 170k vs the former-holdout rows")
so = np.where(m & (F.source == "trainval").values)[0]; sh = np.where(m & (F.source == "holdout").values)[0]
rs = np.random.RandomState(1)
bs = np.array([L[rs.choice(so, len(sh), replace=False)].mean() for _ in range(2000)])
print(f" old-row subsets of n={len(sh)}: mean {bs.mean():.4f}, 95% range {np.percentile(bs, 2.5):.4f}-{np.percentile(bs, 97.5):.4f}, "
      f"P(>= {L[sh].mean():.4f}) = {np.mean(bs >= L[sh].mean()):.3f}")
print(" per fold (old / holdout):", [(k, round(L[m & (F.fold.values == k) & (F.source == 'trainval').values].mean(), 4),
      round(L[m & (F.fold.values == k) & (F.source == 'holdout').values].mean(), 4)) for k in sorted(set(F.fold[m]))])
print("== 4. the OLD model on these exact objects")
kh = keys[keys.source == "holdout"].reset_index(drop=True)
yh = np.load(os.path.join(W, "results", "holdout_predictions", "holdout_y.npy")).ravel()
assert (yh == kh.label.values).all()
zo = np.load(os.path.join(W, "results", "holdout_predictions", "ENS_final_kfold_plain", "holdout_logits.npy")).ravel()
Lo = bce_v(zo, yh.astype(float)); hrow = sh - 170000
print(f" old 20-model ensemble: all 30k {Lo.mean():.4f}; on these 9,122 objects {Lo[hrow].mean():.4f}; on the other 20,878 "
      f"{np.delete(Lo, hrow).mean():.4f}")
for d in sorted(glob.glob(os.path.join(W, "results", "holdout_predictions", "kf_*plain_f*"))):
    f = glob.glob(os.path.join(d, "holdout_logits_tta8.npy"))
    if f:
        z1 = np.load(f[0]).ravel(); l1 = bce_v(z1, yh.astype(float))
        print(f"  old single model {os.path.basename(d)}: all 30k {l1.mean():.4f}, these 9,122 {l1[hrow].mean():.4f}")
