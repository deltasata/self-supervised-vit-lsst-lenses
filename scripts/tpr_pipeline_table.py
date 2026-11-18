"""Per-pipeline TPR at FPR = 1e-3 (and 1e-2) on val, TTA8 logits, for the leaderboard models and the cross-fitted
ensembles (results/ensemble/cf_logits_*.npy from ensemble_val.py). Each pipeline has 4,250 val negatives, so
FPR 1e-3 allows only ~4 false positives: the bootstrap std (200 resamples of val rows within the pipeline) is printed.
Usage: python scripts/tpr_pipeline_table.py run_id [run_id ...]
"""
import os
import sys

import numpy as np
import pandas as pd

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(W, "src"))
from lagn.metrics import tpr_at_fpr  # noqa: E402

keys = pd.read_parquet(os.path.join(W, "data/keys_trainval.parquet"))
va = np.where((keys.split == "val").values)[0]
y = np.load(os.path.join(W, "data/y_trainval.npy"))[va].astype(int)
pipe = keys.pipeline.values[va]
rng = np.random.RandomState(0)
boots = {g: [rng.choice(np.where(pipe == g)[0], (pipe == g).sum()) for _ in range(200)] for g in ("D1", "D2")}


def row(name, z):
    out = [f"{name:28s}"]
    for g in ("D1", "D2"):
        m = pipe == g
        t3 = tpr_at_fpr(y[m], z[m], 1e-3)
        sd = np.std([tpr_at_fpr(y[b], z[b], 1e-3) for b in boots[g]])
        t2 = tpr_at_fpr(y[m], z[m], 1e-2)
        out.append(f"{t3:.3f}±{sd:.3f} {t2:.3f}")
    print("  ".join(out), flush=True)


print(f"{'model':28s}  SLSim@1e-3     @1e-2  SIMCT@1e-3     @1e-2")
for a in sys.argv[1:]:
    f = (os.path.join(W, f"results/ensemble/cf_logits_{a[4:]}.npy") if a.startswith("ens:")
         else os.path.join(W, f"experiments/{a}/val_logits_tta8.npy"))
    row(a, np.load(f).astype(np.float64))
