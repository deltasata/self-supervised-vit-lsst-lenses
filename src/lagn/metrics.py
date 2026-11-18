"""Evaluation metrics shared by all models.

all_metrics(y, p, meta) -> dict with
  auc, logloss (p clipped), pr_auc, tpr@1e-2, tpr@1e-3,
  per_pipeline AUC, per flux-bin AUC (i-band aperture-flux deciles within each pipeline),
  flux-matched AUC (negatives reweighted to the positives' flux histogram within each pipeline),
  precision at TPR 0.5 / 0.8 for lens fractions 1:100 and 1:1000.
`meta` (optional) is a DataFrame aligned with y holding columns pipeline, simset, ap6_i (from obj_stats).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def tpr_at_fpr(y, p, fpr_target, w=None):
    fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
    i = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(i, 0)])


def logloss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def precision_at(y, p, tpr_target, prior):
    fpr, tpr, _ = roc_curve(y, p)
    i = min(np.searchsorted(tpr, tpr_target, side="left"), len(tpr) - 1)
    t, f = tpr[i], fpr[i]
    return float(prior * t / (prior * t + (1 - prior) * f + 1e-300))


def flux_matched_weights(y, flux, groups, nbins=20):
    """Weights for negatives so their flux distribution matches the positives within each group."""
    w = np.ones(len(y), float)
    for g in np.unique(groups):
        m = groups == g
        lf = np.log10(np.clip(flux[m], 1e-3, None))
        edges = np.quantile(lf[y[m] == 1], np.linspace(0, 1, nbins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(edges, lf, side="right") - 1, 0, nbins - 1)
        hp = np.bincount(b[y[m] == 1], minlength=nbins).astype(float)
        hn = np.bincount(b[y[m] == 0], minlength=nbins).astype(float)
        ratio = np.where(hn > 0, hp / np.maximum(hn, 1), 0.0)
        wm = np.where(y[m] == 1, 1.0, ratio[b])
        w[m] = wm
    return w


def all_metrics(y, p, meta=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "logloss": logloss(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "tpr@1e-2": tpr_at_fpr(y, p, 1e-2),
        "tpr@1e-3": tpr_at_fpr(y, p, 1e-3),
    }
    for prior, nm in ((1e-2, "1:100"), (1e-3, "1:1000")):
        for t in (0.5, 0.8):
            out[f"prec@tpr{t}_{nm}"] = precision_at(y, p, t, prior)
    if meta is not None:
        pipe = meta["pipeline"].values
        out["per_pipeline"] = {g: float(roc_auc_score(y[pipe == g], p[pipe == g])) for g in np.unique(pipe)
                               if len(np.unique(y[pipe == g])) == 2}
        out["per_pipeline_tpr@1e-2"] = {g: tpr_at_fpr(y[pipe == g], p[pipe == g], 1e-2) for g in np.unique(pipe)
                                        if len(np.unique(y[pipe == g])) == 2}
        out["per_pipeline_tpr@1e-3"] = {g: tpr_at_fpr(y[pipe == g], p[pipe == g], 1e-3) for g in np.unique(pipe)
                                        if len(np.unique(y[pipe == g])) == 2}
        if "ap6_i" in meta:
            flux = meta["ap6_i"].values.astype(float)
            fb = {}
            for g in np.unique(pipe):
                m = pipe == g
                q = np.quantile(flux[m], np.linspace(0, 1, 11))
                b = np.clip(np.searchsorted(q, flux[m], side="right") - 1, 0, 9)
                ym, pm = y[m], p[m]
                fb[g] = [float(roc_auc_score(ym[b == k], pm[b == k])) if len(np.unique(ym[b == k])) == 2 else None
                         for k in range(10)]
            out["auc_per_flux_decile"] = fb
            w = flux_matched_weights(y, flux, pipe)
            out["auc_flux_matched"] = float(roc_auc_score(y, p, sample_weight=w))
            out["auc_flux_matched_per_pipeline"] = {
                g: float(roc_auc_score(y[pipe == g], p[pipe == g], sample_weight=w[pipe == g])) for g in np.unique(pipe)
                if len(np.unique(y[pipe == g])) == 2}
    return out
