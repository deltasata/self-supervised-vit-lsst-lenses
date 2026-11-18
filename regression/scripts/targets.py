"""Regression target table in one common frame for SLSim (D1_L) and SIMCT (D2_L) lenses (conventions in
regression/PLAN.md).

Frame: x -> +column, y -> +row, arcsec (0.2"/px), origin = stamp centre pixel (20, 20) = lens centre.
Targets (only quantities common to both pipelines with a settled convention;
mass ellipticity/orientation, shear, magnification and source magnitudes are excluded):
  theta_e        Einstein radius, arcsec (Rein, both pipelines)
  src_x, src_y   source position relative to the lens, arcsec (D1 srcx/srcy; D2 source_x/y are HSC pixels -> x0.168)
                 HSC pixels are 0.168"/px; confirmed: with 0.168 every SIMCT source lies within exactly
                 0.500 theta_E (hard cap of the simulation), with 0.2 the cap would be 0.595. v1 table kept as
                 results/targets_trainval_v1_d2pix0.2.parquet.
  z_lens, z_src  lens and source redshift (zlens, zsrc)
  is_quad        1 = quad, 0 = double (D1 num_images, D2 image_count; binary target)

  python regression/scripts/targets.py      -> regression/results/targets_trainval.parquet (+ printed summary)

Only train+val objects are written; holdout rows are built by the evaluation script through build() alone.
"""
import os
import sys

import numpy as np
import pandas as pd

RW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # regression/
W = os.path.dirname(RW)
sys.path.insert(0, os.path.join(W, "scripts"))
from truth_io import load  # noqa: E402

PIX = 0.2
PIX_HSC = 0.168                                                     # SIMCT truth positions are in HSC pixels
TARGETS = ["theta_e", "src_x", "src_y", "z_lens", "z_src"]          # Gaussian NLL
BINARY = ["is_quad"]                                                # BCE


def build(simset):
    t = load(simset)
    a = lambda c: np.asarray(t[c], float)
    if simset == "D1_L":
        sx, sy = a("srcx"), a("srcy")
        nimg = a("num_images")
    elif simset == "D2_L":
        sx, sy = a("source_x") * PIX_HSC, a("source_y") * PIX_HSC
        nimg = a("image_count")
    else:
        raise ValueError(f"no lens-model targets for {simset}")
    assert set(np.unique(nimg)) == {2.0, 4.0}
    return pd.DataFrame({"obj": t["Lens ID"].values, "simset": simset, "pipeline": simset[:2],
                         "theta_e": a("Rein"), "src_x": sx, "src_y": sy, "z_lens": a("zlens"), "z_src": a("zsrc"),
                         "is_quad": (nimg == 4).astype(np.int8)})


def main():
    keys = pd.read_parquet(os.path.join(W, "data", "keys_trainval.parquet"))
    T = pd.concat([build("D1_L"), build("D2_L")], ignore_index=True)
    k = keys[keys.label == 1][["obj", "split"]].copy()
    k["row"] = k.index.values                                           # row in X_trainval / keys_trainval
    T = k.merge(T, on="obj", how="left", validate="one_to_one")
    assert len(T) == (keys.label == 1).sum() and T[TARGETS + BINARY].notna().all().all(), "missing targets"
    T = T.sort_values("row").reset_index(drop=True)
    os.makedirs(os.path.join(RW, "results"), exist_ok=True)
    T.to_parquet(os.path.join(RW, "results", "targets_trainval.parquet"), index=False)
    print(f"lens rows {len(T)}  " + "  ".join(f"{s}:{n}" for s, n in T.groupby(["simset", "split"]).size().items()))
    cols = TARGETS + BINARY
    print(T.groupby("simset")[cols].describe(percentiles=[.01, .5, .99]).T.round(3).to_string())
    print("|src| median (arcsec):", T.assign(r=np.hypot(T.src_x, T.src_y)).groupby("simset").r.median().round(3).to_dict())


if __name__ == "__main__":
    main()
