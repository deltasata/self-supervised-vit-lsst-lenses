"""Val error analysis of an ensemble per-pipeline flux / SNR bins (vs Baseline 0), precision at assumed
lens fractions, worst false positives / negatives grids, and simple descriptors of the errors. Val only, never holdout.

Usage: python scripts/error_analysis.py --out results/error_analysis/best5 run1 run2 ...
Scores = mean of each run's snapshot-ens TTA8 val logits (else TTA8 logits).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SIG = np.array([0.0615, 0.0721, 0.134, 0.358, 0.597], np.float32)   # outer-annulus noise, g r i z y


def logits(r):
    for f in ("val_logits_snapens_tta8.npy", "val_logits_tta8.npy"):
        p = os.path.join(W, "experiments", r, f)
        if os.path.exists(p):
            return np.load(p).ravel()
    raise FileNotFoundError(r)


def tpr_thr(zneg, fpr):
    return np.quantile(zneg, 1 - fpr)


def rgb(x):
    """x (5,41,41) raw -> RGB from i,r,g in noise units with asinh stretch (same stretch for every stamp)."""
    c = np.stack([x[2] / SIG[2], x[1] / SIG[1], x[0] / SIG[0]], -1)
    c = np.arcsinh(c / 3.0) / np.arcsinh(300 / 3.0)
    return np.clip(c, 0, 1)


def grid(X, idx, titles, path, ncol=8, bands=False):
    n = len(idx)
    if bands:
        fig, ax = plt.subplots(n, 6, figsize=(9, 1.55 * n))
        for j, (i, t) in enumerate(zip(idx, titles)):
            x = np.asarray(X[i], np.float32)
            ax[j, 0].imshow(rgb(x), origin="lower")
            ax[j, 0].set_ylabel(t, fontsize=6)
            for b in range(5):
                v = x[b] / SIG[b]
                ax[j, b + 1].imshow(np.arcsinh(v / 3), origin="lower", cmap="gray",
                                    vmin=-1, vmax=max(1.0, np.arcsinh(np.percentile(v, 99.8) / 3)))
                if j == 0:
                    ax[j, b + 1].set_title("grizy"[b], fontsize=8)
            for a in ax[j]:
                a.set_xticks([]); a.set_yticks([])
    else:
        nr = int(np.ceil(n / ncol))
        fig, ax = plt.subplots(nr, ncol, figsize=(1.6 * ncol, 1.75 * nr))
        ax = np.atleast_2d(ax)
        for a in ax.ravel():
            a.axis("off")
        for j, (i, t) in enumerate(zip(idx, titles)):
            a = ax.ravel()[j]
            a.imshow(rgb(np.asarray(X[i], np.float32)), origin="lower")
            a.set_title(t, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k = pd.read_parquet(os.path.join(W, "data/keys_trainval.parquet"))
    y = np.load(os.path.join(W, "data/y_trainval.npy"))
    vidx = np.where((k.split == "val").values)[0]
    kv = k.iloc[vidx].reset_index(drop=True)
    yv = y[vidx].astype(int)
    z = np.mean([logits(r) for r in a.runs], 0)
    b0 = np.load(os.path.join(W, "experiments/baseline0/valpred_mlp_nopipe.npy"))   # canonical row order
    sp = pd.read_parquet(os.path.join(W, "data/splits.parquet"))
    cv = np.where((sp.split == "val").values)[0]                     # canonical rows of val, in B0's order
    b0 = b0[np.searchsorted(cv, kv.canon_row.values)]
    assert roc_auc_score(yv, b0) > 0.95, "Baseline 0 row mapping failed"
    rep = {"runs": a.runs}
    lines = [f"# Error analysis — ensemble of {len(a.runs)}: {', '.join(a.runs)}", ""]

    # ---- per pipeline, flux and SNR bins (operating point = pipeline-wide FPR 1e-3 threshold) ----
    for pipe, nm in (("D1", "SLSim"), ("D2", "SIMCT")):
        m = (kv.pipeline == pipe).values
        zp, yp, bp = z[m], yv[m], b0[m]
        thr3, thr2 = tpr_thr(zp[yp == 0], 1e-3), tpr_thr(zp[yp == 0], 1e-2)
        lines += [f"## {nm}: AUC {roc_auc_score(yp, zp):.5f} (Baseline 0 {roc_auc_score(yp, bp):.4f}); "
                  f"TPR@1e-3 {np.mean(zp[yp == 1] > thr3):.3f}, @1e-2 {np.mean(zp[yp == 1] > thr2):.3f}", ""]
        for col, lab in (("ap6_i", "i-band aperture flux (r ≤ 6 px)"), ("snr6_det", "detection SNR")):
            v = kv[col].values[m]
            q = np.quantile(v, [0, .1, .25, .5, .75, .9, 1])
            lines += [f"### {nm} by {lab} (decile/quartile bins over all {nm} val objects)", "",
                      "| bin | n lens | n non-lens | lens frac | AUC | B0 AUC | TPR@1e-3 thr | FPR at thr | FN | FP |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for lo, hi in zip(q[:-1], q[1:]):
                s = (v >= lo) & (v <= hi)
                nl, nn = int((yp[s] == 1).sum()), int((yp[s] == 0).sum())
                auc = roc_auc_score(yp[s], zp[s]) if nl and nn else float("nan")
                bauc = roc_auc_score(yp[s], bp[s]) if nl and nn else float("nan")
                tp = np.mean(zp[s][yp[s] == 1] > thr3) if nl else float("nan")
                fp = np.mean(zp[s][yp[s] == 0] > thr3) if nn else float("nan")
                fn_ = int((zp[s][yp[s] == 1] <= thr3).sum())
                fpn = int((zp[s][yp[s] == 0] > thr3).sum())
                lines.append(f"| {lo:.3g}–{hi:.3g} | {nl} | {nn} | {nl / max(nl + nn, 1):.2f} | {auc:.4f} | {bauc:.3f} | "
                             f"{tp:.3f} | {fp:.4f} | {fn_} | {fpn} |")
            lines.append("")
        rep[nm] = {"auc": roc_auc_score(yp, zp), "tpr3": float(np.mean(zp[yp == 1] > thr3))}

    # ---- precision at assumed lens fractions ----
    lines += ["## Precision at assumed lens fractions (FPR measured on val non-lenses of that pipeline)", "",
              "| set | target TPR | FPR | precision @1:100 | precision @1:1000 |", "|---|---|---|---|---|"]
    for pipe, nm in (("D1", "SLSim"), ("D2", "SIMCT"), (None, "all")):
        m = np.ones(len(z), bool) if pipe is None else (kv.pipeline == pipe).values
        zp, yp = z[m], yv[m]
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            thr = np.quantile(zp[yp == 1], 1 - t)
            f = np.mean(zp[yp == 0] >= thr)
            nneg = int((yp == 0).sum())
            fs = f"{f:.2e}" if f > 0 else f"<{1 / nneg:.1e}"
            fe = max(f, 0.5 / nneg)            # 0 observed -> use half a count as a rough bound
            p100 = t * 0.01 / (t * 0.01 + fe * 0.99)
            p1000 = t * 0.001 / (t * 0.001 + fe * 0.999)
            pre = "≥" if f == 0 else ""
            lines.append(f"| {nm} | {t:.2f} | {fs} | {pre}{p100:.3f} | {pre}{p1000:.3f} |")
    lines.append("")

    # ---- worst errors ----
    X = np.load(os.path.join(W, "data/X_trainval.npy"), mmap_mode="r")
    for pipe, nm in (("D2", "SIMCT"), ("D1", "SLSim")):
        m = (kv.pipeline == pipe).values
        neg = np.where(m & (yv == 0))[0]
        pos = np.where(m & (yv == 1))[0]
        thr3 = tpr_thr(z[neg], 1e-3)
        fp = neg[np.argsort(-z[neg])][:32]
        fn = pos[np.argsort(z[pos])][:32]
        for tag, sel in (("FP", fp), ("FN", fn)):
            t = [f"{kv.obj[i]} p={1 / (1 + np.exp(-z[i])):.3f}\ni={kv.ap6_i[i]:.0f} snr={kv.snr6_det[i]:.0f}" for i in sel]
            grid(X, vidx[sel], t, os.path.join(a.out, f"worst_{tag}_{nm}.png"))
            grid(X, vidx[sel[:10]], [s.split("\n")[0] for s in t[:10]], os.path.join(a.out, f"worst_{tag}_{nm}_bands.png"),
                 bands=True)
        # random correct examples for reference
        rng = np.random.RandomState(0)
        tp_ = rng.choice(pos[z[pos] > thr3], 16, replace=False)
        tn_ = rng.choice(neg[z[neg] <= thr3], 16, replace=False)
        grid(X, vidx[np.r_[tp_, tn_]], [f"{'L' if yv[i] else 'N'} p={1 / (1 + np.exp(-z[i])):.3f}" for i in np.r_[tp_, tn_]],
             os.path.join(a.out, f"random_correct_{nm}.png"))
        # descriptors of errors vs the class population
        nfp, nfn = int((z[neg] > thr3).sum()), int((z[pos] <= thr3).sum())
        lines += [f"## {nm} errors at the FPR 1e-3 operating point: {nfp} FP of {len(neg)}, {nfn} FN of {len(pos)}", "",
                  "| group | n | median ap6_i | median snr | median colour g−i (ap) |", "|---|---|---|---|---|"]
        xs = np.asarray(X[vidx[np.r_[neg, pos]]], np.float32)
        yy, xx = np.mgrid[:41, :41]
        ap = (np.hypot(yy - 20, xx - 20) <= 6)
        fl = {i: f for i, f in zip(np.r_[neg, pos], xs[:, :, ap].sum(-1))}
        def col(ids):
            f = np.array([fl[i] for i in ids])
            g, i_ = np.clip(f[:, 0], 1e-3, None), np.clip(f[:, 2], 1e-3, None)
            return np.median(-2.5 * np.log10(g / i_))
        for gname, ids in (("all non-lenses", neg), ("FP (non-lens above thr)", neg[z[neg] > thr3]),
                           ("all lenses", pos), ("FN (lens below thr)", pos[z[pos] <= thr3])):
            if len(ids):
                lines.append(f"| {gname} | {len(ids)} | {np.median(kv.ap6_i.values[ids]):.1f} | "
                             f"{np.median(kv.snr6_det.values[ids]):.1f} | {col(ids):.2f} |")
        lines.append("")
        rep[nm].update(n_fp=nfp, n_fn=nfn, worst_fp=[kv.obj[i] for i in fp[:10]], worst_fn=[kv.obj[i] for i in fn[:10]])

    open(os.path.join(a.out, "report.md"), "w").write("\n".join(lines))
    json.dump(rep, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
