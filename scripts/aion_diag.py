"""AION-1 input-conversion check: how much of the stamp survives the frozen HSC image tokenizer for
each canvas mode of AIONNet ("zero" = original zero-padded 49 px; "noise" = sky-noise canvas; "magnify" = x1.97).

For 96 train stamps per simset: canvas x -> tokens -> decoded x'. Per band, in the canvas pixels that map to the
central disk (stamp radius <= 6 px), report
  chi2   = mean((x' - x)^2) / sigma^2   (sigma = robust noise of x in the stamp's outer annulus): ~1 means the residual
           is at the noise level, >> 1 means real signal is lost;
  keep   = 1 - sum((x' - x)^2) / sum((x - bkg)^2) on the central disk: fraction of the central signal energy kept.
Train stamps only (no val / holdout). Usage: python scripts/aion_diag.py
"""
import os
import sys

import numpy as np
import torch

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(W, "src"))
from lagn.data import load_trainval  # noqa: E402
from lagn.models import AIONNet, _aion_codec  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
X, y, keys = load_trainval()
rng = np.random.RandomState(0)
tr = keys.split.values == "train"
sel = {}
for ss in sorted(keys.simset.unique()):
    idx = np.where(tr & (keys.simset.values == ss))[0]
    sel[ss] = np.sort(rng.choice(idx, 96, replace=False))

net = AIONNet("aion-base").to(dev).eval()
codec = _aion_codec(torch.device(dev))
from aion.modalities import HSCImage  # noqa: E402

SCALE = {"zero": 49 / 41, "noise": 96 / 81, "magnify": 96 / 41}
yy, xx = np.mgrid[:96, :96]
rc = np.hypot(yy - 47.5, xx - 47.5)
print(f"{'mode':8s} {'simset':6s} " + " ".join(f"chi2_{b:1s} keep_{b:1s}" for b in "grizy") + "  uniq_tok")
for mode in ("zero", "noise", "magnify"):
    net.canvas = mode
    rs = rc / SCALE[mode]                                   # canvas radius in stamp pixels
    core, ring = rs <= 6, (rs >= 15) & (rs <= 20)
    for ss, idx in sel.items():
        raw = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev)
        with torch.no_grad():
            x = net.make_canvas(raw)
            tok = codec.encode(HSCImage(flux=x.clone(), bands=net.BANDS))
            xd = codec.decode(tok, bands=net.BANDS).flux
        x, xd = x.cpu().numpy(), xd.float().cpu().numpy()
        out = []
        for b in range(5):
            xb, db = x[:, b], xd[:, b]
            v = xb[:, ring]
            med = np.median(v, 1)
            sig = 1.4826 * np.median(np.abs(v - med[:, None]), 1)
            r2 = ((db - xb)[:, core] ** 2).mean(1)
            chi2 = np.median(r2 / sig ** 2)
            sigE = (((xb[:, core] - med[:, None]) ** 2).sum(1))
            keep = np.median(1 - ((db - xb)[:, core] ** 2).sum(1) / np.maximum(sigE, 1e-12))
            out.append(f"{chi2:6.2f} {keep:6.3f}")
        print(f"{mode:8s} {ss:6s} " + " ".join(out) + f"  {len(np.unique(tok.cpu().numpy()))}", flush=True)
