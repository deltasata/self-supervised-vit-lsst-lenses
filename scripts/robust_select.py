"""Val-only choice of the robust ensemble members (best.pt TTA8 logits = the frozen predictor). Usage: python scripts/robust_select.py"""
import sys
import numpy as np
sys.argv = sys.argv[:1]
from leaderboard_now import row, yv  # noqa: E402
import pandas as pd

V = ["e800_robaug", "e400_robaug", "e400_robaug_strong", "e400_robaug_s1", "e800_robaug_s1", "e400_robaug_s2"]
C = ["mimv2b_robaug", "mimv2b_robaug_s1"]
L = lambda r: np.load(f"experiments/{r}/val_logits_tta8.npy").ravel()
sets = {"robust6": V[:5] + C[:1], "robust5_vit": V[:5], "robust6v": V, "robust7": V + C[:1], "robust8": V + C}
rows = [row(r, L(r), "-") for r in V + C] + [row(k, np.mean([L(r) for r in s], 0), len(s)) for k, s in sets.items()]
print(pd.DataFrame(rows).to_string(index=False))
