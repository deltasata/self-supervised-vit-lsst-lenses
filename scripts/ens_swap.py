"""Ensemble additions/swaps vs best5 (cross-fit TS loss, SIMCT TPR@1e-3). Val only.
Usage: python scripts/ens_swap.py <run>..."""
import sys
import numpy as np
import leaderboard_now as lb
B5 = ["mae_vitb_p3_42_in1kmae_e800_ft", "mae_vitb_p3_42_in1kmae_e400_ft", "mae_in1kB_e400_s1_ft",
      "mae_in1kB_e800_s1_ft", "mimv2b_llrd75"]
def show(name, rs):
    r = lb.row(name, np.mean([lb.logits(x) for x in rs], 0), "-")
    print(f"{name:45s} loss {r['loss']:.4f} TS {r['loss_TS']:.4f} AUC_SIMCT {r['AUC_SIMCT']:.5f} TPR3_SIMCT {r['TPR3_SIMCT']:.3f}")
show("best5", B5)
for n in sys.argv[1:]:
    show(f"best5+{n}", B5 + [n])
    for i in range(4):
        show(f"best5 -{B5[i][:22]} +{n[:18]}", B5[:i] + B5[i + 1:] + [n])
