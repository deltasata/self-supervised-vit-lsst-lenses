"""Input-geometry control: 41->42 bilinear resize ("interp42") vs zero padding to 45 ("pad45", 15x15
tokens) vs one-sided zero padding to 42 ("pad42", 14x14 tokens). ViT-B/3, ImageNet-MAE start, 400-epoch stamp MAE,
fine-tune recipe of production family A; FT seeds 0, 1, 2 per arm. Analysis fixed BEFORE any result was seen.

Primary metric: val BCE of best.pt with 8x dihedral TTA (val_logits_tta8.npy), raw (no temperature scaling), on the
same 17,000 val objects for every run (row identity asserted).
Secondary (reported, not used for the decision): BCE per pipeline, AUC, TPR at FPR 1e-3 per pipeline.
Statistic for arm A vs interp42: delta = mean_seeds BCE(A) - mean_seeds BCE(interp42).
Uncertainty: SE_seed = sqrt(sd_A^2/n_A + sd_I^2/n_I) (training noise, from the seed scatter) and SE_obj = bootstrap sd of
delta over val objects (2,000 resamples, same resample for all runs); SE = sqrt(SE_seed^2 + SE_obj^2).
Decision (pre-registered): delta < -2 SE -> A better; delta > +2 SE -> A worse; otherwise no detectable difference
(keep interp42, no production rebuild). If 1 <= |delta|/SE < 2 after 3 seeds, at most ONE extension to 5 seeds per arm
is allowed, then the same rule is applied once more and the result is final.
Check: interp42 seed 0 rerun vs the original mae_vitb_p3_42_in1kmae_e400_ft (same recipe, older code, separate MAE run)
must agree within 3 seed-sd; otherwise investigate before comparing arms.
MAE losses are NOT compared across arms (the resize smooths pixels, so its reconstruction target is easier).
Output: results/inputgeom/compare.md + compare.json
Also used for the multi-task test (--arms/--base/--out/--title/--ref_old none; same rule).
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from lagn.data import W, load_trainval  # noqa: E402

ARMS = {"interp42": ["ig_interp42_ft", "ig_interp42_ft_s1", "ig_interp42_ft_s2"],
        "pad45": ["ig_pad45_ft", "ig_pad45_ft_s1", "ig_pad45_ft_s2"],
        "pad42": ["ig_pad42_ft", "ig_pad42_ft_s1", "ig_pad42_ft_s2"]}
# Pre-registered extension: pad42 vs interp42 had |delta|/SE = 1.08 after 3 seeds -> seeds 3, 4 for
# these two arms only; the pad45 verdict is final at 3 seeds. Run with --extended for the final pad42 verdict.
ARMS_EXT = {"interp42": ARMS["interp42"] + ["ig_interp42_ft_s3", "ig_interp42_ft_s4"],
            "pad42": ARMS["pad42"] + ["ig_pad42_ft_s3", "ig_pad42_ft_s4"]}
REF_OLD = "mae_vitb_p3_42_in1kmae_e400_ft"


def bce_rows(z, y):
    z = z.astype(np.float64)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def tpr_at(z, y, fpr=1e-3):
    t = np.quantile(z[y == 0], 1 - fpr)
    return float(np.mean(z[y == 1] > t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", help="json {arm: [run_ids]} to override (dry runs)")
    ap.add_argument("--ref_old", default=REF_OLD)
    ap.add_argument("--out", default=os.path.join(W, "results", "inputgeom"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--base", default="interp42", help="reference arm")
    ap.add_argument("--title", default=None)
    ap.add_argument("--extended", action="store_true", help="final pad42 vs interp42 verdict with 5 seeds per arm")
    a = ap.parse_args()
    arms = json.loads(a.arms) if a.arms else (ARMS_EXT if a.extended else ARMS)
    if a.extended and not a.arms:
        a.out = os.path.join(a.out, "extended")
    _, y_all, keys = load_trainval()
    va = np.where(keys.split.values == "val")[0]
    y, ss = y_all[va].astype(np.float64), keys.pipeline.values[va]

    runs, L, Z = {}, {}, {}
    for r in sorted({r for v in arms.values() for r in v} | {a.ref_old}):
        d = os.path.join(W, "experiments", r)
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        if os.path.exists(os.path.join(d, "val_rows.npy")):
            assert np.array_equal(np.load(os.path.join(d, "val_rows.npy")), va), f"{r}: different val rows"
        z = np.load(os.path.join(d, "val_logits_tta8.npy")).astype(np.float64).ravel()
        assert len(z) == len(va), r
        Z[r], L[r] = z, bce_rows(z, y)
        m = {"bce": L[r].mean(), "auc": roc_auc_score(y, z)}
        for p in ("D1", "D2"):
            k = ss == p
            m[f"bce_{p}"], m[f"auc_{p}"], m[f"tpr1e-3_{p}"] = L[r][k].mean(), roc_auc_score(y[k], z[k]), tpr_at(z[k], y[k])
        runs[r] = {k: float(v) for k, v in m.items()}

    summ = {}
    for arm, rs in arms.items():
        rs = [r for r in rs if r in runs]
        if rs:
            b = np.array([runs[r]["bce"] for r in rs])
            summ[arm] = {"runs": rs, "n": len(rs), "bce_mean": float(b.mean()),
                         "bce_sd": float(b.std(ddof=1)) if len(rs) > 1 else float("nan"),
                         **{f"{k}_mean": float(np.mean([runs[r][k] for r in rs]))
                            for k in ("bce_D1", "bce_D2", "auc", "auc_D2", "tpr1e-3_D1", "tpr1e-3_D2")}}

    rng = np.random.default_rng(20261113)
    boots = [rng.integers(0, len(va), len(va)) for _ in range(a.boot)]
    comp = {}
    I = summ.get(a.base)
    for arm in [x for x in arms if x != a.base and x in summ]:
        A = summ[arm]
        if not I or A["n"] < 2 or I["n"] < 2:
            comp[arm] = {"status": "insufficient runs", "n": A["n"], "n_interp": I["n"] if I else 0}
            continue
        delta = A["bce_mean"] - I["bce_mean"]
        se_seed = float(np.sqrt(A["bce_sd"] ** 2 / A["n"] + I["bce_sd"] ** 2 / I["n"]))
        mA, mI = np.mean([L[r] for r in A["runs"]], 0), np.mean([L[r] for r in I["runs"]], 0)
        dd = mA - mI
        se_obj = float(np.std([dd[ix].mean() for ix in boots], ddof=1))
        se = float(np.hypot(se_seed, se_obj))
        zsc = delta / se
        verdict = (f"better than {a.base}" if zsc < -2 else f"worse than {a.base}" if zsc > 2
                   else "no detectable difference")
        paired = {r_a: runs[r_a]["bce"] - runs[r_i]["bce"] for r_a, r_i in zip(A["runs"], I["runs"])}
        comp[arm] = {"delta_bce": delta, "rel_delta_%": 100 * delta / I["bce_mean"], "se_seed": se_seed,
                     "se_obj": se_obj, "se": se, "delta_over_se": zsc, "verdict": verdict,
                     "extension_allowed": bool(1 <= abs(zsc) < 2), "paired_by_seed": paired,
                     "delta_bce_D2": A["bce_D2_mean"] - I["bce_D2_mean"]}

    check = None
    if I and a.ref_old in runs and I["runs"] and I["runs"][0] in runs:
        d0 = runs[I["runs"][0]]["bce"] - runs[a.ref_old]["bce"]
        check = {"rerun_minus_original": d0, "interp_seed_sd": I["bce_sd"],
                 "ok": bool(np.isnan(I["bce_sd"]) or abs(d0) <= 3 * I["bce_sd"])}

    os.makedirs(a.out, exist_ok=True)
    res = {"runs": runs, "arms": summ, "comparison": comp, "reproducibility_check": check, "arm_def": arms}
    json.dump(res, open(os.path.join(a.out, "compare.json"), "w"), indent=1)
    lines = [a.title or "# Input-geometry control: interp42 vs pad45 vs pad42 (pre-registered analysis)", "",
             "Val BCE (best.pt, TTA8, 17,000 val objects; same rows for every run).", "",
             "| run | BCE | BCE D1 | BCE D2 | AUC | AUC D2 | TPR@1e-3 D1 | TPR@1e-3 D2 |", "|---|---|---|---|---|---|---|---|"]
    for r, m in runs.items():
        lines.append(f"| {r} | {m['bce']:.5f} | {m['bce_D1']:.5f} | {m['bce_D2']:.5f} | {m['auc']:.6f} | {m['auc_D2']:.6f} | "
                     f"{m['tpr1e-3_D1']:.4f} | {m['tpr1e-3_D2']:.4f} |")
    lines += ["", "| arm | n seeds | BCE mean | BCE sd | BCE D2 mean | TPR@1e-3 D2 mean |", "|---|---|---|---|---|---|"]
    for arm, s in summ.items():
        lines.append(f"| {arm} | {s['n']} | {s['bce_mean']:.5f} | {s['bce_sd']:.5f} | {s['bce_D2_mean']:.5f} | "
                     f"{s['tpr1e-3_D2_mean']:.4f} |")
    lines += ["", f"| arm vs {a.base} | delta BCE | rel. | SE_seed | SE_obj | SE | delta/SE | verdict |", "|---|---|---|---|---|---|---|---|"]
    for arm, c in comp.items():
        if "delta_bce" in c:
            lines.append(f"| {arm} | {c['delta_bce']:+.5f} | {c['rel_delta_%']:+.1f} % | {c['se_seed']:.5f} | {c['se_obj']:.5f} | "
                         f"{c['se']:.5f} | {c['delta_over_se']:+.2f} | {c['verdict']} |")
        else:
            lines.append(f"| {arm} | {c['status']} | | | | | | |")
    lines += ["", f"Reproducibility check (interp42 seed-0 rerun vs {a.ref_old}): {json.dumps(check)}"]
    open(os.path.join(a.out, "compare.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
