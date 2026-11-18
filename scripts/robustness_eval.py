"""Shortcut and robustness tests on VALIDATION only (never the holdout).

Each condition perturbs the val stamps at test time; the frozen candidate models (best.pt, dihedral TTA) score them.
  clean            unchanged
  outer_noise      pixels r >= 16 px replaced by fresh Gaussian noise (per-stamp, per-band outer median and robust sigma)
  outer_swap_same  pixels r >= 16 px taken from a random val stamp of the same pipeline and the SAME class (control)
  outer_swap_opp   pixels r >= 16 px taken from a random val stamp of the same pipeline and the OPPOSITE class
  inner_noise      pixels r < 16 px replaced by noise -> the model sees only the outer background
  noise_x1.25/1.5  extra white noise so the per-band noise sigma grows by that factor
  psf_+10%/+20%    Gaussian blur widening a ~4.2 px FWHM PSF by 10 / 20 %, then white noise added back to the original sigma
  bkg_+0.5sig      constant sky offset of +0.5 sigma per band
  flux_x0.8/1.25   all pixel values scaled (zero-point / calibration error)
Labels are used only to build the swap tests (diagnostic). Outputs: results/robustness/<run>/<cond>.npy (val logits,
row order = val rows of data/keys_trainval.parquet) and, with --report, results/robustness/report.md + summary.json.

Usage: python scripts/robustness_eval.py --runs R1 R2 ... [--tta 4] [--budget_min 13]   (resumable; skips done files)
       python scripts/robustness_eval.py --runs R1 R2 ... --report                       (CPU, tables)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import W, load_trainval  # noqa: E402

OUT = os.path.join(W, "results", "robustness")
CONDS = ["clean", "outer_noise", "outer_swap_same", "outer_swap_opp", "inner_noise", "noise_x1.25", "noise_x1.5",
         "psf_+10%", "psf_+20%", "bkg_+0.5sig", "flux_x0.8", "flux_x1.25"]
R_OUT = 16
FWHM0 = 4.2                     # typical empirical PSF FWHM in px (3.7-6.1 by band/pipeline)


def perturb(cond, X, yv, pipe, gen):
    """X: (n,5,41,41) float32 torch tensor on the GPU (raw values). Returns a perturbed copy."""
    import torch
    import torch.nn.functional as F
    n = X.shape[0]
    dev = X.device
    yy, xx = torch.meshgrid(torch.arange(41, device=dev), torch.arange(41, device=dev), indexing="ij")
    outer = ((yy - 20) ** 2 + (xx - 20) ** 2).float().sqrt() >= R_OUT
    o = X[:, :, outer]                                                   # (n,5,m)
    med = o.median(-1).values
    sig = 1.4826 * (o - med[..., None]).abs().median(-1).values          # per-stamp, per-band robust sigma
    noise = lambda: torch.randn(X.shape, generator=gen, device=dev)
    Z = X.clone()
    if cond == "clean":
        return Z
    if cond == "outer_noise":
        fill = med[..., None, None] + sig[..., None, None] * noise()
        return torch.where(outer, fill, Z)
    if cond == "inner_noise":
        fill = med[..., None, None] + sig[..., None, None] * noise()
        return torch.where(outer, Z, fill)
    if cond in ("outer_swap_same", "outer_swap_opp"):
        rng = np.random.RandomState(1)
        src = np.empty(n, int)
        for p in np.unique(pipe):
            for c in (0, 1):
                tgt = np.where((pipe == p) & (yv == c))[0]
                pool = np.where((pipe == p) & (yv == (c if cond == "outer_swap_same" else 1 - c)))[0]
                s = rng.choice(pool, len(tgt))
                if cond == "outer_swap_same":                            # never swap with itself
                    same = s == tgt
                    s[same] = rng.choice(pool, same.sum())
                src[tgt] = s
        S = X[torch.from_numpy(src).to(dev)]
        return torch.where(outer, S, Z)
    if cond.startswith("noise_x"):
        k = float(cond[len("noise_x"):])
        return Z + sig[..., None, None] * math.sqrt(k * k - 1) * noise()
    if cond.startswith("psf_"):
        f = 1 + float(cond[len("psf_+"):-1]) / 100
        s0 = FWHM0 / 2.3548
        se = s0 * math.sqrt(f * f - 1)
        r = int(math.ceil(4 * se))
        t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k1 = torch.exp(-0.5 * (t / se) ** 2)
        k1 = k1 / k1.sum()
        k2 = (k1[:, None] * k1[None, :])
        w = k2[None, None].expand(5, 1, -1, -1).contiguous()
        B = F.conv2d(F.pad(Z, (r, r, r, r), mode="reflect"), w, groups=5)
        keep = float((k2 ** 2).sum().sqrt())                             # white-noise sigma after the blur
        return B + sig[..., None, None] * math.sqrt(max(0.0, 1 - keep ** 2)) * noise()
    if cond == "bkg_+0.5sig":
        return Z + 0.5 * sig[..., None, None]
    if cond.startswith("flux_x"):
        return Z * float(cond[len("flux_x"):])
    raise ValueError(cond)


def run_models(a):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lagn.data import tta_predict
    from lagn.models import build_model
    t0 = time.time()
    X, y, keys = load_trainval()
    va = np.where((keys.split == "val").values)[0]
    yv, pipe = y[va].astype(int), keys.pipeline.values[va]
    Xv = torch.from_numpy(np.ascontiguousarray(X[va])).cuda()
    for r in a.runs:
        od = os.path.join(OUT, r)
        os.makedirs(od, exist_ok=True)
        todo = [c for c in CONDS if not os.path.exists(os.path.join(od, f"{c}.npy"))]
        if not todo:
            continue
        cfg = json.load(open(os.path.join(W, "experiments", r, "config.json")))
        cfg["model"].pop("init_ckpt", None)
        cfg["model"]["pretrained"] = False
        m = build_model(cfg).cuda()
        m.load_state_dict(torch.load(os.path.join(W, "experiments", r, "best.pt"), map_location="cuda"))
        m.eval()
        for c in todo:
            if (time.time() - t0) / 60 > a.budget_min:
                print("time budget reached; resubmit to continue", flush=True)
                return
            gen = torch.Generator(device="cuda")
            gen.manual_seed(1234)                                        # same perturbation for every model
            Xp = perturb(c, Xv, yv, pipe, gen)
            out = []
            with torch.no_grad():
                for i in range(0, len(Xp), 1024):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out.append(tta_predict(m, Xp[i:i + 1024], a.tta).float().squeeze(1).cpu())
            z = torch.cat(out).numpy().astype(np.float32)
            np.save(os.path.join(od, f"{c}.npy"), z)
            print(f"{r} {c}: loss {np.mean(np.logaddexp(0, z) - yv * z):.4f} ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
            del Xp
        del m
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


def report(a):
    from sklearn.metrics import roc_auc_score
    keys = pd.read_parquet(os.path.join(W, "data", "keys_trainval.parquet"))
    y = np.load(os.path.join(W, "data", "y_trainval.npy"))
    va = (keys.split == "val").values
    yv, pipe = y[va].astype(int), keys.pipeline.values[va]
    names = {"D1": "SLSim", "D2": "SIMCT"}
    Z = {r: {c: np.load(os.path.join(OUT, r, f"{c}.npy")) for c in CONDS} for r in a.runs}
    Z["ENSEMBLE"] = {c: np.mean([Z[r][c] for r in a.runs], 0) for c in CONDS}
    bce = lambda z, t: float(np.mean(np.logaddexp(0, z) - t * z))
    summ, L = {}, ["# Robustness and shortcut tests (val only, test-time perturbations)", "",
                   f"Models: {', '.join(a.runs)} (best.pt, dihedral TTA {a.tta}); ENSEMBLE = mean logit.",
                   "Thresholds for TPR/FPR are fixed per pipeline on the CLEAN scores at FPR 1e-3 (deployment view: "
                   "threshold set once, then conditions change). FPR is the measured false-positive rate at that "
                   "threshold (target 0.001).", ""]
    for r in list(a.runs) + ["ENSEMBLE"]:
        L += [f"## {r}", "", "| condition | loss | AUC | SLSim AUC | SIMCT AUC | SLSim TPR | SLSim FPR | SIMCT TPR | "
              "SIMCT FPR | mean Δlogit lens | mean Δlogit non-lens |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        thr = {p: np.quantile(Z[r]["clean"][(pipe == p) & (yv == 0)], 1 - 1e-3) for p in ("D1", "D2")}
        summ[r] = {}
        for c in CONDS:
            z = Z[r][c]
            row = {"loss": bce(z, yv), "auc": roc_auc_score(yv, z)}
            for p in ("D1", "D2"):
                m = pipe == p
                row[f"auc_{p}"] = roc_auc_score(yv[m], z[m])
                row[f"tpr_{p}"] = float(np.mean(z[m & (yv == 1)] > thr[p]))
                row[f"fpr_{p}"] = float(np.mean(z[m & (yv == 0)] > thr[p]))
            d = z - Z[r]["clean"]
            row["dlogit_lens"], row["dlogit_non"] = float(d[yv == 1].mean()), float(d[yv == 0].mean())
            summ[r][c] = row
            L.append(f"| {c} | {row['loss']:.4f} | {row['auc']:.5f} | {row['auc_D1']:.5f} | {row['auc_D2']:.5f} | "
                     f"{row['tpr_D1']:.3f} | {row['fpr_D1']:.4f} | {row['tpr_D2']:.3f} | {row['fpr_D2']:.4f} | "
                     f"{row['dlogit_lens']:+.2f} | {row['dlogit_non']:+.2f} |")
        L.append("")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(L) + "\n")
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--budget_min", type=float, default=13)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a) if a.report else run_models(a)
