"""Production version 2 vs version 1: out-of-fold (OOF) scoring (production/PLAN.md addendum; CPU, light).

version 1 = 1/2 * mean(A fold models: ViT, 41->42 bilinear resize) + 1/2 * mean(B, C CNN families)   (as oof.py)
version 2 = 1/2 * mean(A2 fold models: ViT, pad42 = raw 41x41 + 1 zero row/column) + 1/2 * mean(B, C)
Every object is scored only by fold models that never trained on it (val_rows checked against folds.parquet by
oof.family_oof). Pre-registered default: submit v2 unless its all-200k OOF BCE is worse than v1 by > +2 SE
(object bootstrap, 2,000 resamples). Extra row, NOT used for any choice: ViT half pooled over A and A2.
Output: results/oof_v2_report.json + oof_v2_report.md, results/oof_logits_ensemble_v2.npy, oof_logits_pA2_vitpad42.npy
"""
import json
import os

import numpy as np
import pandas as pd

import data_all
import oof
from lagn.metrics import all_metrics

R = os.path.join(data_all.PROD, "results")


def rows_bce(z, y):
    z = z.astype(np.float64)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def boot_se(d, n_boot=2000, seed=20261116):
    rng = np.random.default_rng(seed)
    return float(np.std([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)], ddof=1))


def main():
    F = pd.read_parquet(os.path.join(R, "folds.parquet"))
    _, y, keys = data_all.load_all()
    assert (F.obj.values == keys.obj.values).all()
    fold, y = F.fold.values, y.astype(float)
    Z, pf = {}, {}
    for fam in ("pA_vit", "pA2_vitpad42", "pB_cnn800", "pC_cnn1600"):
        z, got, pf[fam] = oof.family_oof(fam, fold)
        assert got == list(range(10)) and np.isfinite(z).all(), f"{fam}: folds {got}"
        Z[fam] = z
    cnn = 0.5 * (Z["pB_cnn800"] + Z["pC_cnn1600"])
    V = {"v1 (ViT A interp42 + CNN B,C)": 0.5 * Z["pA_vit"] + 0.5 * cnn,
         "v2 (ViT A2 pad42 + CNN B,C)": 0.5 * Z["pA2_vitpad42"] + 0.5 * cnn,
         "extra: ViT A+A2 pooled + CNN B,C": 0.5 * (0.5 * Z["pA_vit"] + 0.5 * Z["pA2_vitpad42"]) + 0.5 * cnn,
         "ViT A interp42 alone": Z["pA_vit"], "ViT A2 pad42 alone": Z["pA2_vitpad42"],
         "old model (final_kfold_plain)": oof.old_reference_logits()}
    v1_saved = np.load(os.path.join(R, "oof_logits_ensemble.npy")).astype(np.float64)
    chk_v1 = float(np.max(np.abs(v1_saved - V["v1 (ViT A interp42 + CNN B,C)"])))
    assert chk_v1 < 1e-3, f"v1 recomputation differs from the saved final OOF by {chk_v1}"

    subsets = {"all 200k": np.ones(len(y), bool), "old 170k train+val": (F.source == "trainval").values,
               "former holdout 30k": (F.source == "holdout").values,
               "never seen by any encoder 47k": ~F.seen_in_pretraining.values}
    L = {k: rows_bce(z, y) for k, z in V.items()}
    tab = {s: {k: float(L[k][m].mean()) for k in V} for s, m in subsets.items()}
    for s, m in subsets.items():
        tab[s]["n"] = int(m.sum())

    k1, k2 = "v1 (ViT A interp42 + CNN B,C)", "v2 (ViT A2 pad42 + CNN B,C)"
    d = L[k2] - L[k1]
    se = boot_se(d)
    dec = {"delta_v2_minus_v1": float(d.mean()), "rel_%": 100 * float(d.mean()) / tab["all 200k"][k1], "se_boot": se,
           "delta_over_se": float(d.mean()) / se,
           "rule": "submit v2 unless delta > +2 SE (pre-registered in production/PLAN.md)",
           "default_choice": "v1" if d.mean() > 2 * se else "v2"}
    dv = L["ViT A2 pad42 alone"] - L["ViT A interp42 alone"]
    sev = boot_se(dv)
    vit_cmp = {"delta_A2_minus_A": float(dv.mean()), "rel_%": 100 * float(dv.mean()) / tab["all 200k"]["ViT A interp42 alone"],
               "se_boot": sev, "delta_over_se": float(dv.mean()) / sev}
    folds = []
    for k in range(10):
        m = fold == k
        folds.append({"fold": k, "A interp42": float(L["ViT A interp42 alone"][m].mean()),
                      "A2 pad42": float(L["ViT A2 pad42 alone"][m].mean()),
                      "v1": float(L[k1][m].mean()), "v2": float(L[k2][m].mean())})
    sec = {}
    for k in (k1, k2):
        p = 1 / (1 + np.exp(-V[k]))
        mm = all_metrics(y.astype(int), p, keys.reset_index(drop=True))
        mn = ~F.seen_in_pretraining.values
        mn_ = all_metrics(y[mn].astype(int), p[mn], keys[mn].reset_index(drop=True))
        sec[k] = {"auc_all": mm["auc"], "tpr@1e-3_all_D1": mm["per_pipeline_tpr@1e-3"]["D1"],
                  "tpr@1e-3_all_D2": mm["per_pipeline_tpr@1e-3"]["D2"],
                  "tpr@1e-3_never_seen_D1": mn_["per_pipeline_tpr@1e-3"]["D1"],
                  "tpr@1e-3_never_seen_D2": mn_["per_pipeline_tpr@1e-3"]["D2"],
                  "bce_D1": float(L[k][keys.pipeline.values == "D1"].mean()),
                  "bce_D2": float(L[k][keys.pipeline.values == "D2"].mean())}

    rep = {"bce_by_subset": tab, "decision_v2_vs_v1": dec, "vit_alone_A2_vs_A": vit_cmp, "per_fold": folds,
           "secondary": sec, "check_v1_equals_saved_final_oof_maxabs": chk_v1}
    json.dump(rep, open(os.path.join(R, "oof_v2_report.json"), "w"), indent=1)
    np.save(os.path.join(R, "oof_logits_ensemble_v2.npy"), V[k2].astype(np.float32))
    np.save(os.path.join(R, "oof_logits_pA2_vitpad42.npy"), Z["pA2_vitpad42"].astype(np.float32))

    cols = list(V)
    md = ["# Production v2 (ViT pad42) vs v1 (ViT interp42): out-of-fold BCE", "",
          "Each object is scored only by fold models that never trained on it (10 folds, 200k objects). "
          "Logits are 8x dihedral TTA, raw (no calibration). Lower BCE is better.", "",
          "| objects | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for s, r in tab.items():
        md.append(f"| {s} | {r['n']:,} | " + " | ".join(f"{r[c]:.5f}" for c in cols) + " |")
    md += ["", "| comparison (all 200k objects) | delta BCE | rel. | SE (object bootstrap) | delta/SE |", "|---|---|---|---|---|",
           f"| v2 − v1 (full classifiers) | {dec['delta_v2_minus_v1']:+.5f} | {dec['rel_%']:+.1f} % | {dec['se_boot']:.5f} | "
           f"{dec['delta_over_se']:+.2f} |",
           f"| ViT A2 pad42 − ViT A interp42 (ViT alone) | {vit_cmp['delta_A2_minus_A']:+.5f} | {vit_cmp['rel_%']:+.1f} % | "
           f"{vit_cmp['se_boot']:.5f} | {vit_cmp['delta_over_se']:+.2f} |",
           "", f"Pre-registered rule: {dec['rule']} → default choice: **{dec['default_choice']}**", "",
           "| fold (20,000 objects each) | ViT A interp42 | ViT A2 pad42 | v1 | v2 |", "|---|---|---|---|---|"]
    for r in folds:
        md.append(f"| {r['fold']} | {r['A interp42']:.5f} | {r['A2 pad42']:.5f} | {r['v1']:.5f} | {r['v2']:.5f} |")
    md += ["", "| classifier | AUC all | BCE SLSim | BCE SIMCT | TPR@FPR1e-3 SLSim (all) | TPR@FPR1e-3 SIMCT (all) | "
           "TPR@FPR1e-3 SLSim (never seen 47k) | TPR@FPR1e-3 SIMCT (never seen 47k) |", "|---|---|---|---|---|---|---|---|"]
    for k, r in sec.items():
        md.append(f"| {k} | {r['auc_all']:.5f} | {r['bce_D1']:.5f} | {r['bce_D2']:.5f} | {r['tpr@1e-3_all_D1']:.4f} | "
                  f"{r['tpr@1e-3_all_D2']:.4f} | {r['tpr@1e-3_never_seen_D1']:.4f} | {r['tpr@1e-3_never_seen_D2']:.4f} |")
    open(os.path.join(R, "oof_v2_report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
