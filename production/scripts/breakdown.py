"""Error breakdown of the production OOF ensemble by lens / contaminant type.

Rows: the 170k original train+val objects only (the 30k former-holdout rows are excluded so truth for those rows is not
read here). Truth tables are used for grouping only. Per pipeline the operating threshold t3 is set where 1 in 1,000
contaminants of that pipeline (same rows) pass (FPR 1e-3).
Lens groups: missed at p=0.5, missed at t3, BCE. Contaminant groups: FP at p=0.5, FP rate at t3, share among the hardest
1 % of contaminants vs share of all contaminants ("enrichment"), BCE.
Output: production/results/breakdown.md + breakdown.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.realpath(os.path.join(HERE, ".."))
W = os.path.realpath(os.path.join(P, ".."))
sys.path.insert(0, os.path.join(W, "scripts"))
import truth_io  # noqa: E402


def bce(z, y):
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def md(d):
    fm = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fm(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def main():
    F = pd.read_parquet(os.path.join(P, "results", "folds.parquet"))
    z = np.load(os.path.join(P, "results", "oof_logits_ensemble.npy")).astype(float)
    keys = pd.read_parquet(os.path.join(W, "data", "keys_trainval.parquet"))
    keep = F.obj.isin(set(keys.obj)).values & np.isfinite(z)
    F, z = F[keep].reset_index(drop=True), z[keep]
    y = F.label.values.astype(float)
    ids = F.obj.str[4:].astype(int).values
    ss = F.simset.values
    L = bce(z, y)
    reg = pd.read_parquet(os.path.join(W, "regression", "results", "targets_trainval.parquet")).set_index("obj")
    t1, t2n, t2l = truth_io.load("D1_N"), truth_io.load("D2_N"), truth_io.load("D2_L")

    grp = np.array([""] * len(F), dtype=object)
    sub = {}
    for pipe in ("D1", "D2"):
        lm = ss == f"{pipe}_L"
        r = reg.loc[F.obj.values[lm]]
        q = np.where(r.is_quad.values > 0.5, "quad", "double")
        te = r.theta_e.values
        br = np.hypot(r.src_x.values, r.src_y.values) / te
        sub[f"{pipe}_L"] = {"type": q,
                            "theta_E": pd.cut(te, [0, 0.5, 0.75, 1.0, 1.5, 10], labels=["<0.5\"", "0.5-0.75\"", "0.75-1\"", "1-1.5\"", ">1.5\""]).astype(str),
                            "beta/theta_E": pd.cut(br, [-1, 0.25, 0.5, 0.75, 1.0, 100], labels=["<0.25", "0.25-0.5", "0.5-0.75", "0.75-1", ">1"]).astype(str)}
    m = ss == "D1_N"
    blk = ids[m] // 10000
    names = {0: "0 red galaxy + bluer light", 1: "1 galaxy, fainter", 2: "2 no galaxy, blue", 3: "3 no galaxy, medium",
             4: "4 no galaxy, red/faint"}
    sub["D1_N"] = {"ID block (inferred type)": np.array([names[b] for b in blk])}
    m = ss == "D2_N"
    tt = t2n.iloc[ids[m]]
    ct = np.array([{1: "star+star", 2: "qso+qso", 3: "star+qso"}[int(c)] for c in tt.cont_type.values])
    bcg = np.where(tt.cont_has_bcg.values > 0, "+BCG", "no BCG")
    sep = np.hypot(tt.cont_src1_x.values - tt.cont_src2_x.values, tt.cont_src1_y.values - tt.cont_src2_y.values)
    sub["D2_N"] = {"pair type": ct, "BCG": bcg, "pair x BCG": np.char.add(np.char.add(ct.astype(str), " "), bcg.astype(str)),
                   "pair separation [px]": pd.cut(sep, [-1, 4, 8, 12, 16, 100], labels=["<4", "4-8", "8-12", "12-16", ">16"]).astype(str)}

    rows = []
    for pipe in ("D1", "D2"):
        neg = ss == f"{pipe}_N"
        t3 = np.quantile(z[neg], 1 - 1e-3)
        hard = neg & (z >= np.quantile(z[neg], 0.99))
        for s in (f"{pipe}_L", f"{pipe}_N"):
            sm = ss == s
            for dim, lab in sub[s].items():
                for g in sorted(set(lab)):
                    gm = np.zeros(len(F), bool)
                    gm[np.where(sm)[0][lab == g]] = True
                    r = {"simset": s, "split by": dim, "group": g, "n": int(gm.sum()), "BCE": L[gm].mean()}
                    if s.endswith("_L"):
                        r["miss@0.5 %"] = 100 * np.mean(z[gm] < 0)
                        r["miss@FPR1e-3 %"] = 100 * np.mean(z[gm] <= t3)
                    else:
                        r["FP@0.5 %"] = 100 * np.mean(z[gm] >= 0)
                        r["pass@t3 %"] = 100 * np.mean(z[gm] > t3)
                        r["enrich hardest1%"] = (gm & hard).sum() / hard.sum() / (gm.sum() / neg.sum())
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P, "results", "breakdown.csv"), index=False)
    with open(os.path.join(P, "results", "breakdown.md"), "w") as f:
        f.write("# Production OOF error breakdown (A+B ensemble, 170k original train+val rows)\n\n")
        for s, d in df.groupby("simset", sort=False):
            f.write(f"## {s}\n\n" + md(d.drop(columns="simset").dropna(axis=1, how="all")) + "\n\n")
    print(open(os.path.join(P, "results", "breakdown.md")).read())


if __name__ == "__main__":
    main()
