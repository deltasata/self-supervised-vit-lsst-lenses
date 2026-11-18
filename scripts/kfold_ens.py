"""OOF ensemble check across k-fold recipes (CPU, val-only): log loss of every equal-weight logit average of recipes,
on the rows where all chosen recipes have finished folds. Never reads the holdout.
Usage: python scripts/kfold_ens.py [recipe ...]   (default: vit_plain vit_rob cnn400_plain cnn400_rob)
"""
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import W, load_trainval  # noqa: E402


def oof(recipe, n):
    z = np.full(n, np.nan)
    for d in sorted(glob.glob(os.path.join(W, "experiments", f"kf_{recipe}_f*"))):
        if os.path.exists(f"{d}/metrics.json") and os.path.exists(f"{d}/val_logits_tta8.npy"):
            z[np.load(f"{d}/val_rows.npy")] = np.load(f"{d}/val_logits_tta8.npy").ravel()
    return z


if __name__ == "__main__":
    recipes = sys.argv[1:] or ["vit_plain", "vit_rob", "cnn400_plain", "cnn400_rob"]
    _, y, _ = load_trainval()
    R = {p: oof(p, len(y)) for p in recipes}
    m = np.all([~np.isnan(v) for v in R.values()], 0)
    print(f"rows with all recipes: {m.sum()}")
    loss = lambda z: float(np.mean(np.logaddexp(0, z[m]) - y[m] * z[m]))
    for r in range(1, len(R) + 1):
        for c in itertools.combinations(R, r):
            print(f"{' + '.join(c):55s} {loss(np.mean([R[k] for k in c], 0)):.5f}")
