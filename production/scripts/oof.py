"""Assemble and score the production out-of-fold (OOF) predictions (CPU, light).

For every family and fold k it loads experiments/<family>_f<k>/val_logits_tta8.npy and val_rows.npy. It REFUSES to
continue unless val_rows equals exactly the rows of fold k in results/folds.parquet, i.e. the model never trained on
the rows it is scored on. The family-balanced rule (PLAN.md): logit = ½ ViT + ½ mean(CNN families).
Usage: python production/scripts/oof.py   (partial results allowed; it says which folds are missing)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import data_all

sys.path.insert(0, os.path.join(data_all.W, "src"))
from lagn.metrics import all_metrics  # noqa: E402

FAMILIES = {"pA_vit": "vit", "pB_cnn800": "cnn", "pC_cnn1600": "cnn"}
REF = {"old_oof_170k": 0.0198, "old_holdout_30k_20models": 0.0192}  # full-set numbers; NOT comparable to partial subsets


def old_reference_logits():
    """The previous final classifier (final_kfold_plain) on all 200k rows: its OOF logits on the old 170k (½ ViT + ½ CNN)
    and its 20-model holdout logits on the former 30k. Bars must be computed on the SAME rows as the production numbers."""
    k = os.path.join(data_all.W, "results")
    zo = 0.5 * np.load(os.path.join(k, "kfold", "oof_vit_plain.npy")).ravel() + \
        0.5 * np.load(os.path.join(k, "kfold", "oof_cnn400_plain.npy")).ravel()
    zh = np.load(os.path.join(k, "holdout_predictions", "ENS_final_kfold_plain", "holdout_logits.npy")).ravel()
    return np.concatenate([zo, zh])


def bce(z, t):
    z = z.astype(np.float64)
    return float(np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))))


def family_oof(fam, fold):
    z, got = np.full(len(fold), np.nan), []
    per_fold = {}
    for k in range(data_all.N_FOLDS):
        d = os.path.join(data_all.PROD, "experiments", f"{fam}_f{k}")
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        rows = np.load(os.path.join(d, "val_rows.npy"))
        if not np.array_equal(np.sort(rows), np.where(fold == k)[0]):
            sys.exit(f"REFUSED: {fam}_f{k} val_rows differ from fold {k} (possible train/val overlap)")
        z[rows] = np.load(os.path.join(d, "val_logits_tta8.npy"))
        per_fold[k] = json.load(open(os.path.join(d, "metrics.json")))["val_loss_tta8"]
        got.append(k)
    return z, got, per_fold


def main():
    F = pd.read_parquet(os.path.join(data_all.PROD, "results", "folds.parquet"))
    _, y, keys = data_all.load_all()
    assert (F.obj.values == keys.obj.values).all()
    fold, y = F.fold.values, y.astype(int)
    rep, fam_z = {"reference": REF, "families": {}}, {}
    for fam, kind in FAMILIES.items():
        z, got, pf = family_oof(fam, fold)
        if not got:
            continue
        m = ~np.isnan(z)
        rep["families"][fam] = {"folds_done": got, "fold_loss_mean": float(np.mean(list(pf.values()))),
                                "fold_loss_std": float(np.std(list(pf.values()))), "oof_loss": bce(z[m], y[m]),
                                "n_rows": int(m.sum())}
        fam_z[fam] = (kind, z)
    vit = [z for k_, z in fam_z.values() if k_ == "vit"]
    cnn = [z for k_, z in fam_z.values() if k_ == "cnn"]
    if vit and cnn:
        zc = np.nanmean(np.stack(cnn), 0) if len(cnn) > 1 else cnn[0]
        z = 0.5 * vit[0] + 0.5 * zc
        m = ~np.isnan(z)
        p = 1 / (1 + np.exp(-z))
        subsets = {"all": m, "old_170k_rows": m & (F.source == "trainval").values,
                   "former_holdout_30k_rows": m & (F.source == "holdout").values,
                   "encoder_saw_unlabelled_153k": m & F.seen_in_pretraining.values,
                   "never_seen_47k": m & ~F.seen_in_pretraining.values}
        rep["ensemble"] = {}
        zold = old_reference_logits()
        assert len(zold) == len(y)
        for nm, s in subsets.items():
            if s.sum() == 0:
                continue
            r = {"n": int(s.sum()), "logloss": bce(z[s], y[s]), "old_model_same_rows": bce(zold[s], y[s])}
            for fam, (_, zf) in fam_z.items():
                r[f"{fam}_alone"] = bce(zf[s], y[s])
            if nm in ("all", "never_seen_47k", "former_holdout_30k_rows"):
                mm = all_metrics(y[s], p[s], keys[s].reset_index(drop=True))
                r.update({k_: mm[k_] for k_ in ("auc", "tpr@1e-3", "tpr@1e-2", "per_pipeline", "per_pipeline_tpr@1e-3",
                                                "auc_flux_matched", "auc_flux_matched_per_pipeline")})
            rep["ensemble"][nm] = r
        np.save(os.path.join(data_all.PROD, "results", "oof_logits_ensemble.npy"), z.astype(np.float32))
    for fam, (_, z) in fam_z.items():
        np.save(os.path.join(data_all.PROD, "results", f"oof_logits_{fam}.npy"), z.astype(np.float32))
    json.dump(rep, open(os.path.join(data_all.PROD, "results", "oof_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
