"""Baseline 0 — the shortcut floor. CPU only.

Hand-crafted per-band photometric features (aperture fluxes, peaks, concentration, colours, sizes,
noise level, peak counts) from data/verify/obj_stats.parquet  ->  logistic regression, small MLP and
LightGBM. Fit on train, early-stop / select on val; report val metrics. Run with and without a pipeline flag.
Also a per-pipeline variant (separate model per pipeline).

The holdout is NOT touched: rows in the holdout are dropped via data/splits.parquet.
Usage: python scripts/baseline0.py
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.metrics import all_metrics  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(W, "experiments", "baseline0")
os.makedirs(OUT, exist_ok=True)
BANDS = "grizy"


def slog(x):
    return np.sign(x) * np.log10(1 + np.abs(x))


def build(df):
    f = {}
    for b in BANDS:
        for c in ("ap3", "ap6", "ap10", "peak", "peakc"):
            f[f"{c}_{b}"] = slog(df[f"{c}_{b}"])
        for c in ("snr6", ):
            f[f"{c}_{b}"] = slog(df[f"{c}_{b}"])
        f[f"conc_{b}"] = df[f"conc_{b}"].clip(-5, 5)
        f[f"rmom_{b}"] = df[f"rmom_{b}"]
        f[f"dcen_{b}"] = np.hypot(df[f"cy_{b}"] - 20, df[f"cx_{b}"] - 20)
        f[f"mad_{b}"] = df[f"mad_{b}"]
    for a, b in zip(BANDS[:-1], BANDS[1:]):  # colours (asinh-mag-like differences)
        f[f"col_{a}{b}"] = np.arcsinh(df[f"ap6_{a}"] / 0.5) - np.arcsinh(df[f"ap6_{b}"] / 0.5)
    f["snr6_det"] = slog(df["snr6_det"])
    f["npeaks5_det"] = df["npeaks5_det"]
    return pd.DataFrame(f).astype(np.float32)


def main():
    st = pd.read_parquet(os.path.join(W, "data", "verify", "obj_stats.parquet"))
    sp = pd.read_parquet(os.path.join(W, "data", "splits.parquet"))  # obj order == canonical order
    assert (sp.simset.values == st.simset.values).all() and (sp.id.values == st.id.values).all()
    F = build(st)
    F_nonoise = F[[c for c in F.columns if not c.startswith("mad_")]]
    y = st.label.values
    pipe = (st.pipeline == "D2").astype(np.float32).values
    tr, va = (sp.split == "train").values, (sp.split == "val").values
    res = {}

    def run(name, X, kind):
        Xtr, Xva = X[tr], X[va]
        if kind == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xva)[:, 1]
        elif kind == "mlp":
            m = make_pipeline(StandardScaler(), MLPClassifier((256, 256), early_stopping=True, max_iter=200,
                                                              random_state=0))
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xva)[:, 1]
        else:
            m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=127, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=32)
            m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
            p = m.predict_proba(Xva)[:, 1]
        met = all_metrics(y[va], p, st[va].reset_index(drop=True))
        res[name] = met
        np.save(os.path.join(OUT, f"valpred_{name}.npy"), p.astype(np.float32))
        print(name, json.dumps({k: met[k] for k in ("auc", "logloss", "tpr@1e-2", "tpr@1e-3")}),
              "| per-pipe AUC", met["per_pipeline"], flush=True)

    for kind in ("lr", "mlp", "gbm"):
        run(f"{kind}_nopipe", F.values, kind)
    run("gbm_pipe", np.c_[F.values, pipe], "gbm")
    run("gbm_nopipe_nonoise", F_nonoise.values, "gbm")
    # pipeline flag only (sanity: should be AUC 0.5 since classes balanced per pipeline)
    run("lr_pipeonly", pipe[:, None], "lr")
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
