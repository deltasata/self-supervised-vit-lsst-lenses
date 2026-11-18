"""Prediction store: TRAIN and VAL logits of every finished run (best.pt, dihedral TTA 8), so ROC curves
and any metric can be rebuilt from scores. Never reads the holdout (holdout scores are written only by
scripts/eval_holdout.py, for frozen models).

Layout (results/predictions/):
  train_y.npy, train_keys.parquet, val_y.npy, val_keys.parquet      labels + evaluation metadata, fixed row order
  <run>/train_logits_tta8.npy, <run>/val_logits_tta8.npy            best.pt, TTA 8 (the frozen predictor)
  <run>/val_logits_snapens_tta8.npy                                 snapshot ensemble, copied when the run made one
  k-fold members (config train.fold set): the model trained on 9 folds of all 170k train+val rows, so
  <run>/trainval_logits_tta8.npy (all 170k rows, X_trainval order), <run>/train_rows.npy (its 9 training folds),
  <run>/val_rows.npy (its own fold); train_/val_logits_tta8.npy are those rows of trainval_logits_tta8.npy
  --summary (CPU): train / val / holdout log loss + AUC per run (k-fold members: val = own fold; k-fold ensembles: OOF) -> results/holdout_predictions/summary_train_val_holdout.csv
  (a report kept in the holdout folder; nothing used for model selection reads it)

Usage: python scripts/predict_store.py [--runs R ...] [--budget_min 1380]   (resumable; skips runs already stored)
       python scripts/predict_store.py --summary
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import W, load_trainval  # noqa: E402

PRED = os.path.join(W, "results", "predictions")                  # train / val only
HPRED = os.path.join(W, "results", "holdout_predictions")         # holdout, written only by eval_holdout.py
KEYCOLS = ["obj", "simset", "pipeline", "label", "id", "ap6_i", "snr6_det"]


def all_runs():
    return sorted(r for r in os.listdir(os.path.join(W, "experiments"))
                  if os.path.exists(os.path.join(W, "experiments", r, "best.pt"))
                  and os.path.exists(os.path.join(W, "experiments", r, "config.json")))


def predict(a):
    import torch
    from lagn.data import tta_predict
    from lagn.models import build_model
    t0 = time.time()
    X, y, keys = load_trainval()
    tr = np.where((keys.split == "train").values)[0]
    va = np.where((keys.split == "val").values)[0]
    os.makedirs(PRED, exist_ok=True)
    for nm, idx in (("train", tr), ("val", va)):
        if not os.path.exists(os.path.join(PRED, f"{nm}_y.npy")):
            np.save(os.path.join(PRED, f"{nm}_y.npy"), y[idx].astype(np.int8))
            keys.iloc[idx][KEYCOLS].reset_index(drop=True).to_parquet(os.path.join(PRED, f"{nm}_keys.parquet"))
    Xg = {"train": torch.from_numpy(np.ascontiguousarray(X[tr])).cuda(),
          "val": torch.from_numpy(np.ascontiguousarray(X[va])).cuda()}
    allrows = np.concatenate([tr, va])                    # k-fold members: predict every train+val row once
    for r in (a.runs or all_runs()):
        d, od = os.path.join(W, "experiments", r), os.path.join(PRED, r)
        if os.path.exists(os.path.join(od, "train_logits_tta8.npy")):
            continue
        if (time.time() - t0) / 60 > a.budget_min:
            print("time budget reached; resubmit to continue", flush=True)
            return
        try:
            cfg = json.load(open(os.path.join(d, "config.json")))
            cfg["model"].pop("init_ckpt", None)
            cfg["model"]["pretrained"] = False
            m = build_model(cfg).cuda()
            m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cuda"))
            m.eval()
        except Exception as e:                                             # noqa: BLE001
            print(f"SKIP {r}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        os.makedirs(od, exist_ok=True)
        z = {}
        for nm in ("val", "train"):
            out = []
            with torch.no_grad():
                for i in range(0, len(Xg[nm]), 2048):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out.append(tta_predict(m, Xg[nm][i:i + 2048], 8).float().squeeze(1).cpu())
            z[nm] = torch.cat(out).numpy().astype(np.float32)
        fold = cfg["train"].get("fold")
        if fold is not None:                              # k-fold member: re-index to its own train / val rows
            zall = np.empty(len(y), np.float32)
            zall[allrows] = np.concatenate([z["train"], z["val"]])
            vrows = np.load(os.path.join(d, "val_rows.npy"))
            trows = np.setdiff1d(np.arange(len(y)), vrows)
            np.save(os.path.join(od, "trainval_logits_tta8.npy"), zall)
            np.save(os.path.join(od, "val_rows.npy"), vrows)
            np.save(os.path.join(od, "train_rows.npy"), trows)
            z = {"val": zall[vrows], "train": zall[trows]}
            yt, yv = y[trows], y[vrows]
        else:
            yt, yv = y[tr], y[va]
        ref = os.path.join(d, "val_logits_tta8.npy")
        dv = float(np.abs(z["val"] - np.load(ref).ravel()).max()) if os.path.exists(ref) else float("nan")
        np.save(os.path.join(od, "val_logits_tta8.npy"), z["val"])
        np.save(os.path.join(od, "train_logits_tta8.npy"), z["train"])      # written last = completion marker
        se = os.path.join(d, "val_logits_snapens_tta8.npy")
        if os.path.exists(se):
            shutil.copy(se, os.path.join(od, "val_logits_snapens_tta8.npy"))
        bce = lambda q, t: float(np.mean(np.logaddexp(0, q) - t * q))
        print(f"{r}: train {bce(z['train'], yt):.4f} val {bce(z['val'], yv):.4f} | val max|diff| vs stored {dv:.3g} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        del m
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


def kfold_ens_train(r, n):
    """In-sample train logits of a frozen k-fold ensemble: per recipe, each row's mean over the fold models that trained on
    it (all except the row's own fold), then the mean over recipes. None until every member's trainval logits exist."""
    rec = {"plain": ["vit_plain", "cnn400_plain"], "robust": ["vit_rob", "cnn400_rob"]}[r.split("_")[-1]]
    out = []
    for x in rec:
        s_, c_ = np.zeros(n), np.zeros(n)
        for f in range(10):
            d = os.path.join(PRED, f"kf_{x}_f{f}")
            if not os.path.exists(os.path.join(d, "trainval_logits_tta8.npy")):
                return None
            tr_ = np.load(os.path.join(d, "train_rows.npy"))
            s_[tr_] += np.load(os.path.join(d, "trainval_logits_tta8.npy"))[tr_]
            c_[tr_] += 1
        out.append(s_ / c_)
    return np.mean(out, 0)


def val_from_experiments(r):
    """(logits, labels, pipeline, val_set) for runs outside the store: k-fold members from experiments/<r>/ on their own
    fold rows; the frozen k-fold ensembles (ENS_final_kfold_*) as out-of-fold logits over all 170k train+val rows."""
    _, y, keys = load_trainval()
    if r.startswith("ENS_final_kfold_"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from kfold_ens import oof
        rec = {"plain": ["vit_plain", "cnn400_plain"], "robust": ["vit_rob", "cnn400_rob"]}[r.split("_")[-1]]
        z = np.mean([oof(x, len(y)) for x in rec], 0)
        m = np.isfinite(z)
        return z[m], y[m].astype(np.float64), keys.pipeline.values[m], f"OOF{int(m.sum())}"
    hj = os.path.join(W, "experiments", "holdout_evals", f"{r[4:]}.json")
    if r.startswith("ENS_") and os.path.exists(hj):              # other frozen ensembles: mean of members' val logits
        runs = json.load(open(hj))["runs"]
        va = np.where(keys.split.values == "val")[0]
        z = np.mean([np.load(os.path.join(W, "experiments", x, "val_logits_tta8.npy")).ravel() for x in runs], 0)
        return z.astype(np.float64), y[va].astype(np.float64), keys.pipeline.values[va], "val"
    d = os.path.join(W, "experiments", r)
    if os.path.exists(f"{d}/val_logits_tta8.npy") and os.path.exists(f"{d}/val_rows.npy"):
        rows = np.load(f"{d}/val_rows.npy")
        fold = json.load(open(f"{d}/config.json"))["train"].get("fold")
        return (np.load(f"{d}/val_logits_tta8.npy").ravel().astype(np.float64), y[rows].astype(np.float64),
                keys.pipeline.values[rows], "val" if fold is None else f"fold{fold}")
    return None


def summary():
    from sklearn.metrics import roc_auc_score
    rows = []
    base = {"train": PRED, "val": PRED, "holdout": HPRED}
    lab = {s: np.load(os.path.join(base[s], f"{s}_y.npy")) for s in base
           if os.path.exists(os.path.join(base[s], f"{s}_y.npy"))}
    pipe = {s: pd.read_parquet(os.path.join(base[s], f"{s}_keys.parquet")).pipeline.values for s in lab}
    _, y_all, keys_all = load_trainval()
    pipe_all = keys_all.pipeline.values
    runs = sorted({r for d in (PRED, HPRED) if os.path.isdir(d) for r in os.listdir(d) if os.path.isdir(os.path.join(d, r))})
    for r in runs:
        row = {"run": r}
        for s, d, f in (("train", PRED, "train_logits_tta8.npy"), ("val", PRED, "val_logits_tta8.npy"),
                        ("val_snapens", PRED, "val_logits_snapens_tta8.npy"),
                        ("holdout", HPRED, "holdout_logits_tta8.npy"), ("holdout", HPRED, "holdout_logits.npy")):
            p = os.path.join(d, r, f)
            ss = s.split("_")[0]
            rf = os.path.join(d, r, f"{ss}_rows.npy")
            if os.path.exists(p) and ss in ("train", "val") and os.path.exists(rf):     # k-fold member: own rows
                rows_ = np.load(rf)
                z, t = np.load(p).astype(np.float64), y_all[rows_].astype(np.float64)
                row[f"{ss}_set"] = f"{len(rows_)} own rows"
                for g, nm in (("D1", "SLSim"), ("D2", "SIMCT")):
                    mm = pipe_all[rows_] == g
                    row[f"{s}_auc_{nm}"] = round(float(roc_auc_score(t[mm], z[mm])), 6)
                row[f"{s}_loss"] = round(float(np.mean(np.logaddexp(0, z) - t * z)), 5)
                row[f"{s}_auc"] = round(float(roc_auc_score(t, z)), 6)
                continue
            if os.path.exists(p) and ss in lab:
                z, t = np.load(p).astype(np.float64), lab[ss]
                row[f"{s}_loss"] = round(float(np.mean(np.logaddexp(0, z) - t * z)), 5)
                row[f"{s}_auc"] = round(float(roc_auc_score(t, z)), 6)
                for g, nm in (("D1", "SLSim"), ("D2", "SIMCT")):
                    m = pipe[ss] == g
                    row[f"{s}_auc_{nm}"] = round(float(roc_auc_score(t[m], z[m])), 6)
        if r.startswith("ENS_final_kfold_") and "train_loss" not in row:
            tz = kfold_ens_train(r, len(y_all))         # in-sample: each row averaged over the 9 folds that trained on it
            if tz is not None:
                row["train_set"] = "in-sample 170k"
                t = y_all.astype(np.float64)
                row["train_loss"] = round(float(np.mean(np.logaddexp(0, tz) - t * tz)), 5)
                row["train_auc"] = round(float(roc_auc_score(t, tz)), 6)
                for g, nm in (("D1", "SLSim"), ("D2", "SIMCT")):
                    row[f"train_auc_{nm}"] = round(float(roc_auc_score(t[pipe_all == g], tz[pipe_all == g])), 6)
        if "val_loss" not in row:                       # not in the train/val store: val from the run dir
            vz = val_from_experiments(r)
            if vz is not None:
                (z, t, pp), row["val_set"] = vz[:3], vz[3]
                row["val_loss"] = round(float(np.mean(np.logaddexp(0, z) - t * z)), 5)
                row["val_auc"] = round(float(roc_auc_score(t, z)), 6)
                for g, nm in (("D1", "SLSim"), ("D2", "SIMCT")):
                    row[f"val_auc_{nm}"] = round(float(roc_auc_score(t[pp == g], z[pp == g])), 6)
        rows.append(row)
    df = pd.DataFrame(rows)
    order = ["run"] + [c for pre in ("train_set", "train_", "val_set", "val_loss", "val_auc", "val_snapens", "holdout_")
                       for c in df.columns if c.startswith(pre)]
    df = df[list(dict.fromkeys(order + list(df.columns)))]
    os.makedirs(HPRED, exist_ok=True)
    df.to_csv(os.path.join(HPRED, "summary_train_val_holdout.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--budget_min", type=float, default=1380)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else predict(a)
