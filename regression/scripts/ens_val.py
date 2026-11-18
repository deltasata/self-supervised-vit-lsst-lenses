"""Validation-only ensemble check of regression runs (CPU, light): equal-weight mixture of the runs' TTA8 Gaussians
(mean of means; variance = mean variance + spread of means) and the mean quad logit. Never reads the holdout.
Usage: python regression/scripts/ens_val.py RUN [RUN ...]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from targets import TARGETS  # noqa: E402

RW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


Z = TARGETS.index("z_src")


def load(run):
    """rows, mean, variance, quad logit, z_src mixture (means, sigmas, log-weights; a Gaussian = 1 component)."""
    z = np.load(os.path.join(RW, "experiments", run, "val_pred.npz"), allow_pickle=True)
    if "zmix_mean" in z:
        mix = (z["zmix_mean"], z["zmix_sigma"], z["zmix_logw"])
    else:
        mix = (z["mean"][:, Z:Z + 1], z["sigma"][:, Z:Z + 1], np.zeros((len(z["rows"]), 1)))
    return z["rows"], z["mean"], z["sigma"] ** 2, z["quad_logit"], mix


def score(mu, var, ql, T, std, mix=None):
    t = T[TARGETS].values
    nll = 0.5 * (np.log(var / std ** 2) + (t - mu) ** 2 / var + np.log(2 * np.pi))      # in standardised units
    if mix is not None:                                          # exact mixture NLL for z_src
        m, s, lw = mix
        logn = -0.5 * (np.log((s / std[Z]) ** 2) + ((t[:, Z:Z + 1] - m) / s) ** 2 + np.log(2 * np.pi))
        a = lw + logn
        mx = a.max(1, keepdims=True)
        nll[:, Z] = -(mx[:, 0] + np.log(np.exp(a - mx).sum(1)))
    qt = T.is_quad.values
    bce = np.mean(np.logaddexp(0, ql) - qt * ql)
    out = {"val_loss": float(nll.mean() + bce)}
    for p in ("D1", "D2"):
        s = T.pipeline.values == p
        out[p] = {c: round(float(np.median(np.abs(mu[s, j] - t[s, j]))), 4) for j, c in enumerate(TARGETS)}
        out[p]["src_off"] = round(float(np.median(np.hypot(mu[s, 1] - t[s, 1], mu[s, 2] - t[s, 2]))), 4)
        out[p]["quad_acc"] = round(float(np.mean((ql[s] > 0) == qt[s])), 4)
    return out


if __name__ == "__main__":
    runs = sys.argv[1:]
    T = pd.read_parquet(os.path.join(RW, "results", "targets_trainval.parquet")).set_index("row")
    import json
    std = np.array(json.load(open(os.path.join(RW, "experiments", runs[0], "scaler.json")))["std"])
    P = [load(r) for r in runs]
    assert all((p[0] == P[0][0]).all() for p in P)
    Tv = T.loc[P[0][0]].reset_index()
    for r, (_, mu, var, ql, mix) in zip(runs, P):
        print(r, score(mu, var, ql, Tv, std, mix))
    mus = np.stack([p[1] for p in P])
    mu = mus.mean(0)
    var = np.mean([p[2] for p in P], 0) + mus.var(0)
    ql = np.mean([p[3] for p in P], 0)
    mix = tuple(np.concatenate([p[4][j] for p in P], 1) for j in range(3))
    mix = (mix[0], mix[1], mix[2] - np.log(len(P)))             # equal-weight mixture of the members' mixtures
    print("ENSEMBLE", score(mu, var, ql, Tv, std, mix))
