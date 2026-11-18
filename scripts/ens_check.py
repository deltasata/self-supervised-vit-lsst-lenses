"""Quick val check of a run's snapshot-ensemble logits: raw loss, cross-fit TS loss, and the TS loss when added to
the current best 3-run ensemble. Usage: python scripts/ens_check.py <run> [<run> ...]"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

k = pd.read_parquet("data/keys_trainval.parquet")
y = np.load("data/y_trainval.npy")
va = (k.split == "val").values
yv = y[va].astype(float)


def bce(z, t):
    return float(np.mean(np.logaddexp(0, z) - t * z))


p = np.random.RandomState(20261008).permutation(len(yv))
h = [p[:len(p) // 2], p[len(p) // 2:]]


def ts(z):
    o = np.empty_like(z)
    for a, b in [(0, 1), (1, 0)]:
        t = minimize_scalar(lambda s: bce(z[h[a]] * s, yv[h[a]]), bounds=(0.3, 3), method="bounded").x
        o[h[b]] = z[h[b]] * t
    return bce(o, yv)


def L(r):
    return np.load(f"experiments/{r}/val_logits_snapens_tta8.npy").ravel()


base = ["mae_vitb_p3_42_in1kmae_e800_ft", "in1kmaeB_ft_s1", "mimv2b_llrd75"]
zb = np.mean([L(r) for r in base], 0)
print(f"base3 TS {ts(zb):.5f}")
for r in sys.argv[1:]:
    z = L(r)
    print(f"{r}: snap {bce(z, yv):.5f} TS {ts(z):.5f} | base3+this TS {ts((zb * 3 + z) / 4):.5f}")
