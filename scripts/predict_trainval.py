"""Teacher logits for knowledge distillation: TTA8 logits of finished runs on ALL trainval rows (train + val,
load_trainval order; never the holdout). Saves experiments/<run>/trainval_logits_tta8.npy per run and the plain
mean over runs to --out. Val rows of the mean must equal the stored val_logits_tta8 mean (checked).
Usage: python scripts/predict_trainval.py --out experiments/kd_teachers/ens7.npy RUN [RUN ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lagn.data import W, load_trainval  # noqa: E402
from lagn.models import build_model  # noqa: E402
from train import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("runs", nargs="+")
a = ap.parse_args()
X, y, keys = load_trainval()
va = np.where((keys.split == "val").values)[0]
Xg = torch.from_numpy(np.ascontiguousarray(X)).cuda()
yg = torch.from_numpy(y.astype(np.int64)).cuda()
allz = []
for r in a.runs:
    d = os.path.join(W, "experiments", r)
    f = os.path.join(d, "trainval_logits_tta8.npy")
    if os.path.exists(f):
        z = np.load(f)
    else:
        cfg = json.load(open(os.path.join(d, "config.json")))
        cfg["model"].pop("init_ckpt", None)
        cfg["model"]["pretrained"] = False
        m = build_model(cfg).cuda()
        m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location="cuda"))
        _, z = evaluate(m, Xg, yg, tta=8)
        np.save(f, z.astype(np.float32))
        del m
    ref = np.load(os.path.join(d, "val_logits_tta8.npy"))
    tr = (keys.split == "train").values
    print(r, "val max|diff| vs stored", float(np.abs(z[va] - ref).max()),
          "train loss", float(np.mean(np.logaddexp(0, z[tr]) - y[tr] * z[tr])),
          "val loss", float(np.mean(np.logaddexp(0, z[va]) - y[va] * z[va])), flush=True)
    allz.append(z)
zm = np.mean(allz, 0).astype(np.float32)
os.makedirs(os.path.dirname(os.path.join(W, a.out)), exist_ok=True)
np.save(os.path.join(W, a.out), zm)
tr = (keys.split == "train").values
print("MEAN train loss", float(np.mean(np.logaddexp(0, zm[tr]) - y[tr] * zm[tr])),
      "val loss", float(np.mean(np.logaddexp(0, zm[va]) - y[va] * zm[va])), flush=True)
