"""Auxiliary (multi-task) targets for the classifier, from the truth tables.

TARGETS ONLY, never model inputs. Train+val objects only (the holdout is not touched).
All targets are invariant under the 8 dihedral transforms, so the training augmentation needs no target transform.

Columns (NaN where not defined):
  is_quad        bin  lenses: 4 images (SLSim IDs >= 25,000; SIMCT from regression/results/targets_trainval.parquet)
  log_theta_e    reg  lenses: ln Einstein radius [arcsec]
  beta_ratio     reg  lenses: |source offset| / theta_E (how asymmetric / how faint the counter-image is)
  z_lens, z_src  reg  lenses
  has_galaxy     bin  contaminants: a central galaxy is present (SLSim: z_central tabulated, IDs < 20,000;
                      SIMCT: cont_has_bcg)
  pair_star_star, pair_qso_qso, pair_star_qso  bin  SIMCT contaminants: cont_type 1 / 2 / 3
  log_pair_sep   reg  SIMCT contaminants: ln separation of the two point sources [px]
Regression columns are standardised with mean/std over TRAIN-split rows; stats saved in the json.
Output: results/aux_targets/aux_targets_trainval.parquet (+ .json)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import truth_io  # noqa: E402
from lagn.data import W, load_trainval  # noqa: E402

KINDS = {"is_quad": "bin", "log_theta_e": "reg", "beta_ratio": "reg", "z_lens": "reg", "z_src": "reg",
         "has_galaxy": "bin", "pair_star_star": "bin", "pair_qso_qso": "bin", "pair_star_qso": "bin",
         "log_pair_sep": "reg"}


def main():
    _, y, keys = load_trainval()
    out = pd.DataFrame({"obj": keys.obj.values, "split": keys.split.values})
    for c in KINDS:
        out[c] = np.nan
    ids = keys["id"].astype(int).values
    ss = keys.simset.values

    reg = pd.read_parquet(os.path.join(W, "regression", "results", "targets_trainval.parquet")).set_index("obj")
    m = np.isin(ss, ["D1_L", "D2_L"])
    r = reg.loc[keys.obj.values[m]]
    out.loc[m, "is_quad"] = r.is_quad.values.astype(float)
    out.loc[m, "log_theta_e"] = np.log(r.theta_e.values)
    out.loc[m, "beta_ratio"] = np.hypot(r.src_x.values, r.src_y.values) / r.theta_e.values
    out.loc[m, "z_lens"] = r.z_lens.values
    out.loc[m, "z_src"] = r.z_src.values

    t1 = truth_io.load("D1_N")                       # row k = ID k
    m1 = ss == "D1_N"
    out.loc[m1, "has_galaxy"] = t1.z_central.notna().values[ids[m1]].astype(float)
    t2 = truth_io.load("D2_N")
    m2 = ss == "D2_N"
    tt = t2.iloc[ids[m2]]
    out.loc[m2, "has_galaxy"] = tt.cont_has_bcg.values.astype(float)
    for k, c in ((1, "pair_star_star"), (2, "pair_qso_qso"), (3, "pair_star_qso")):
        out.loc[m2, c] = (tt.cont_type.values == k).astype(float)
    sep = np.hypot(tt.cont_src1_x.values - tt.cont_src2_x.values, tt.cont_src1_y.values - tt.cont_src2_y.values)
    out.loc[m2, "log_pair_sep"] = np.log(np.maximum(sep, 0.5))

    stats = {}
    trm = out.split.values == "train"
    for c, kd in KINDS.items():
        if kd == "reg":
            v = out.loc[trm, c].dropna().values
            mu, sd = float(v.mean()), float(v.std())
            out[c] = (out[c] - mu) / sd
            stats[c] = {"mean": mu, "std": sd}
    d = os.path.join(W, "results", "aux_targets")
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "aux_targets_trainval.parquet"), index=False)
    cover = {c: {s: int(out.loc[ss == s, c].notna().sum()) for s in ("D1_L", "D1_N", "D2_L", "D2_N")} for c in KINDS}
    json.dump({"kinds": KINDS, "standardisation_train": stats, "coverage": cover}, open(
        os.path.join(d, "aux_targets_trainval.json"), "w"), indent=1)
    print(json.dumps(cover))
    print(out.describe().T[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
