"""Projection of the production OOF ensemble to realistic test conditions.
170k original train+val rows only. For each pipeline: lens recall (TPR) at fixed contaminant pass rates, and the precision
(purity) of the selected sample when lenses are rare (lens fraction 1:100 and 1:1000), for the training 50/50
double/quad mix and for double-heavy mixes (75 % and 90 % doubles; lenses reweighted, contaminants unchanged).
precision = TPR*pi / (TPR*pi + FPR*(1-pi)).  Output: production/results/realistic_projection.md
"""
import os
import numpy as np
import pandas as pd

P = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
W = os.path.dirname(P)
F = pd.read_parquet(os.path.join(P, "results", "folds.parquet"))
z = np.load(os.path.join(P, "results", "oof_logits_ensemble.npy")).astype(float)
keys = pd.read_parquet(os.path.join(W, "data", "keys_trainval.parquet"))
keep = F.obj.isin(set(keys.obj)).values & np.isfinite(z)
F, z = F[keep].reset_index(drop=True), z[keep]
reg = pd.read_parquet(os.path.join(W, "regression", "results", "targets_trainval.parquet")).set_index("obj")
quad = np.full(len(F), -1)
lm = F.label.values == 1
quad[lm] = reg.loc[F.obj.values[lm], "is_quad"].values.astype(int)
out = ["# Realistic-condition projection of the production OOF ensemble (A+B, 170k train+val rows)\n",
       "TPR = lens recall; precision = purity of the selected sample at lens fraction pi (contaminants unchanged).\n"]
for pipe in ("D1", "D2"):
    neg = z[(F.simset == f"{pipe}_N").values]
    out.append(f"\n## {'SLSim' if pipe == 'D1' else 'SIMCT'}\n\n| doubles | FPR | TPR | precision @1:100 | precision @1:1000 |\n|---|---|---|---|---|")
    for fd in (0.5, 0.75, 0.9):
        pos_d = z[(F.simset == f"{pipe}_L").values & (quad == 0)]
        pos_q = z[(F.simset == f"{pipe}_L").values & (quad == 1)]
        for fpr in (1e-2, 1e-3, 1e-4):
            t = np.quantile(neg, 1 - fpr)
            fpr_eff = max(np.mean(neg > t), 1 / len(neg))
            tpr = fd * np.mean(pos_d > t) + (1 - fd) * np.mean(pos_q > t)
            pr = [tpr * pi / (tpr * pi + fpr_eff * (1 - pi)) for pi in (1e-2, 1e-3)]
            out.append(f"| {fd:.0%} | {fpr:g} | {tpr:.3f} | {pr[0]:.3f} | {pr[1]:.3f} |")
open(os.path.join(P, "results", "realistic_projection.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
