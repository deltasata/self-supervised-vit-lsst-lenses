"""Per-parameter error / bias table of a validation ensemble (CPU, light; never reads the holdout).
For each target and pipeline: mean |error|, typical % error = median(|pred-true|/|true|), bias = mean(pred-true) and
bias % = mean(pred-true)/mean(true). Source position: offset |beta| as its own parameter; position error |d beta| in % of theta_E.
Usage: python regression/scripts/err_table.py RUN [RUN ...]
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ens_val import RW, load, score  # noqa: E402
from targets import TARGETS  # noqa: E402

runs = sys.argv[1:]
T = pd.read_parquet(os.path.join(RW, "results", "targets_trainval.parquet")).set_index("row")
std = np.array(json.load(open(os.path.join(RW, "experiments", runs[0], "scaler.json")))["std"])
P = [load(r) for r in runs]
assert all((p[0] == P[0][0]).all() for p in P)
Tv = T.loc[P[0][0]].reset_index()
assert (Tv.split == "val").all()
mus = np.stack([p[1] for p in P]); mu = mus.mean(0); var = np.mean([p[2] for p in P], 0) + mus.var(0)
ql = np.mean([p[3] for p in P], 0)
mix = tuple(np.concatenate([p[4][j] for p in P], 1) for j in range(3)); mix = (mix[0], mix[1], mix[2] - np.log(len(P)))
print("ensemble of", len(runs), "runs; val NLL", round(score(mu, var, ql, Tv, std, mix)["val_loss"], 4), "; n val lenses", len(Tv))
t = Tv[TARGETS].values
rows = []
for pipe, nm in (("D1", "SLSim"), ("D2", "SIMCT")):
    s = Tv.pipeline.values == pipe
    P_ = {c: mu[s, j] for j, c in enumerate(TARGETS)}; Q = {c: t[s, j] for j, c in enumerate(TARGETS)}
    P_["src_offset"] = np.hypot(P_["src_x"], P_["src_y"]); Q["src_offset"] = np.hypot(Q["src_x"], Q["src_y"])
    for c in ["theta_e", "src_offset", "z_lens", "z_src"]:
        d = P_[c] - Q[c]
        rows.append([nm, c, len(d), Q[c].mean(), np.mean(np.abs(d)), np.median(np.abs(d)),
                     100 * np.median(np.abs(d) / np.abs(Q[c])), d.mean(), 100 * d.mean() / Q[c].mean(),
                     100 * np.sqrt(np.mean(d ** 2)) / Q[c].mean()])
    dpos = np.hypot(P_["src_x"] - Q["src_x"], P_["src_y"] - Q["src_y"])
    rows.append([nm, "src_position_err_%thetaE", s.sum(), np.nan, np.mean(dpos), np.median(dpos),
                 100 * np.median(dpos / Q["theta_e"]), np.nan, np.nan, np.nan])
    for c in ("src_x", "src_y"):
        d = P_[c] - Q[c]
        rows.append([nm, c + "_bias_%thetaE", s.sum(), np.nan, np.mean(np.abs(d)), np.median(np.abs(d)), np.nan, d.mean(),
                     100 * d.mean() / Q["theta_e"].mean(), np.nan])
    zq = Q["z_src"]; zo = np.mean(np.abs(P_["z_src"] - zq) / (1 + zq) > 0.15)
    qt = Tv.is_quad.values[s]; qp = ql[s] > 0
    print(f"{nm}: quad accuracy {100*np.mean(qp == qt):.2f} %, predicted quad fraction {100*qp.mean():.2f} % vs true {100*qt.mean():.2f} %;"
          f" z_src outliers |dz|/(1+z)>0.15: {100*zo:.1f} %")
    for c, lo_hi in (("theta_e", None), ("z_src", None)):
        q = np.quantile(Q[c], [0, 1/3, 2/3, 1])
        print(f"   {nm} {c} bias % by truth tercile:",
              [f"{q[i]:.2f}-{q[i+1]:.2f}: {100*np.mean(P_[c][(Q[c]>=q[i])&(Q[c]<=q[i+1])]-Q[c][(Q[c]>=q[i])&(Q[c]<=q[i+1])])/np.mean(Q[c][(Q[c]>=q[i])&(Q[c]<=q[i+1])]):+.1f}%" for i in range(3)])
R = pd.DataFrame(rows, columns=["pipeline", "param", "n", "mean_truth", "mean_abs_err", "median_abs_err", "typical_pct_err",
                                "bias", "bias_pct", "rmse_pct"])
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
