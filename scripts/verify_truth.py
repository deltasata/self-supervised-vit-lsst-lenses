"""Sanity checks of the organizers' truth tables against our stamps (CPU, light; train+val stamps only, never holdout
stamps; holdout rows enter only through table-level counts, which use no pixels).

  python scripts/verify_truth.py            -> prints a report, writes results/data_checks/truth_checks.json

Checks: ID coverage; D1 RA/Dec vs FITS headers; image-position convention (pixel values at the tabulated image
positions under the 8 axis conventions); photometric zero point implied by tabulated magnitudes vs stamp flux;
per-simset composition (doubles/quads, contaminant types); D2 galaxy_id reuse across simsets and splits.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from truth_io import load  # noqa: E402
from lagn.data import W, load_trainval  # noqa: E402

RAW = os.environ.get("LAGN_RAW_DIR", "data/raw/all_data")
PIX, C = 0.2, 20                       # assumed arcsec/px and centre pixel
BANDS = "grizy"
rng = np.random.RandomState(0)
R = {}


def idcol(t):
    return t["Lens ID"] if "Lens ID" in t else t["Object ID"]


def main():
    X, y, keys = load_trainval()
    objs = pd.read_parquet(os.path.join(W, "data", "objects.parquet")) if os.path.exists(
        os.path.join(W, "data", "objects.parquet")) else None
    row = {o: i for i, o in enumerate(keys.obj.values)}             # train+val rows only
    T = {s: load(s) for s in ["D1_L", "D1_N", "D2_L", "D2_N"]}

    # A. IDs
    for s, t in T.items():
        ids = idcol(t)
        exp = {f"{s}{i:08d}" for i in range(50000)}
        R[f"A_ids_{s}"] = dict(n=len(t), unique=int(ids.nunique()), match_expected=bool(set(ids) == exp),
                               rows_in_order=bool((ids.values == np.array(sorted(exp))).all()),
                               in_trainval=int(ids.isin(row).sum()))
    # B. D1 coordinates vs FITS header (train+val objects)
    for s in ["D1_L", "D1_N"]:
        t = T[s].set_index(idcol(T[s]))
        sel = rng.choice([o for o in t.index if o in row], 300, replace=False)
        d = []
        for o in sel:
            h = fits.getheader(os.path.join(RAW, f"{o}_i.fits"))
            d.append([h["RA_OBJ"] - t.loc[o, "RA"], h["DEC_OBJ"] - t.loc[o, "Dec"]])
        d = np.abs(np.array(d))
        R[f"B_radec_{s}"] = dict(n=len(sel), max_abs_dRA=float(d[:, 0].max()), max_abs_dDec=float(d[:, 1].max()))

    # C. image positions (D1_L): mean i-band pixel (bkg-subtracted) at tabulated images, 8 conventions
    t = T["D1_L"].set_index("Lens ID")
    sel = [o for o in rng.permutation(t.index.values)[:4000] if o in row][:2000]
    idx = np.array([row[o] for o in sel])
    order = np.argsort(idx)
    img = np.asarray(X[idx[order]])[:, 2]                            # i band
    sel = [sel[k] for k in order]
    img = img - np.median(img[:, :5, :], axis=(1, 2))[:, None, None]
    conv = {}
    for sx in (1, -1):
        for sy in (1, -1):
            for swap in (False, True):
                v, vr = [], []
                for k, o in enumerate(sel):
                    for j in range(1, 5):
                        xa, ya = t.loc[o, f"x_img_{j}"], t.loc[o, f"y_img_{j}"]
                        if not np.isfinite(xa):
                            continue
                        a, b = (ya, xa) if swap else (xa, ya)
                        col, rw = int(round(C + sx * a / PIX)), int(round(C + sy * b / PIX))
                        if 0 <= col < 41 and 0 <= rw < 41:
                            v.append(img[k, rw, col] / (np.abs(img[k]).max() + 1e-9))
                            rc, rr = rng.randint(0, 41, 2)
                            vr.append(img[k, rr, rc] / (np.abs(img[k]).max() + 1e-9))
                conv[f"sx{sx:+d}_sy{sy:+d}_{'swap' if swap else 'xy'}"] = round(float(np.mean(v)), 4)
    R["C_imgpos_convention_meanpeakfrac"] = conv
    R["C_random_position_meanpeakfrac"] = round(float(np.mean(vr)), 4)
    rad = np.hypot(t[[f"x_img_{j}" for j in range(1, 5)]].values, t[[f"y_img_{j}" for j in range(1, 5)]].values)
    R["C_D1L_image_radius_arcsec"] = dict(median=float(np.nanmedian(rad)), p99=float(np.nanpercentile(rad, 99)),
                                          frac_outside_stamp=float(np.nanmean(rad > 20 * PIX)))

    # D. zero point: total tabulated flux vs stamp sum (bkg-subtracted), per band, train+val subsample
    def stamp_flux(rows):
        a = np.asarray(X[np.sort(rows)]).astype(np.float64)
        bk = np.median(np.concatenate([a[:, :, :4, :].reshape(len(a), 5, -1), a[:, :, -4:, :].reshape(len(a), 5, -1)], 2), 2)
        return np.sort(rows), (a - bk[:, :, None, None]).sum((2, 3))

    def zp(s, magfun, name):
        tt = T[s].set_index(idcol(T[s]))
        cand = [o for o in rng.permutation(tt.index.values)[:6000] if o in row]
        rows, F = stamp_flux(np.array([row[o] for o in cand]))
        inv = {r: o for o, r in zip(cand, [row[o] for o in cand])}
        out = {}
        for b, B in enumerate(BANDS):
            m = np.array([magfun(tt.loc[inv[r]], B) for r in rows])
            ok = np.isfinite(m) & (F[:, b] > 0)
            z = m[ok] + 2.5 * np.log10(F[ok, b])
            if not ok.any():
                continue
            out[B] = dict(n=int(ok.sum()), zp_median=round(float(np.median(z)), 3),
                          zp_iqr=round(float(np.subtract(*np.percentile(z, [75, 25]))), 3))
        R[f"D_zp_{name}"] = out

    def magsum(ms):
        ms = np.array([m for m in ms if np.isfinite(m)])
        return -2.5 * np.log10(np.sum(10 ** (-0.4 * ms))) if len(ms) else np.nan

    zp("D1_L", lambda r, B: magsum([r[f"mag_lens_{B}"]] + [r[f"mag_ps_img_{B}_{j}"] for j in range(1, 5)]),
       "D1_L_lens+images")
    zp("D1_N", lambda r, B: r[f"mag_lens_{B}"], "D1_N_galaxy_only")
    zp("D2_N", lambda r, B: magsum([r[f"cont_src1_{B}mag"], r[f"cont_src2_{B}mag"]]
                                   + ([r["cont_bcg_imag"]] if (B == "i" and r["cont_has_bcg"] == 1) else [])),
       "D2_N_contaminant_sources")
    zp("D2_L", lambda r, B: magsum([r[f"mag_lens_{B}"]]) if f"mag_lens_{B}" in r else np.nan, "D2_L_lens_galaxy_only")

    # E. composition
    R["E_D1L_num_images"] = T["D1_L"].num_images.value_counts().to_dict()
    R["E_D2L_image_count"] = T["D2_L"].image_count.value_counts().to_dict()
    R["E_D2N_image_count"] = T["D2_N"].image_count.value_counts().to_dict()
    R["E_D2L_field"] = T["D2_L"].field.value_counts().to_dict()
    R["E_D2N_run"] = T["D2_N"].run.value_counts().to_dict()
    d2n = T["D2_N"]
    R["E_D2N_type_bcg_src"] = d2n.groupby(["cont_type", "cont_has_bcg", "cont_src1_type", "cont_src2_type"]).size() \
        .rename("n").reset_index().to_dict("records")
    oid = d2n["Original ID"].str.extract(r"bcg(\d)_type(\d)").astype(int)
    R["E_D2N_origid_matches_cols"] = dict(bcg=bool((oid[0] == d2n.cont_has_bcg).all()), type=bool((oid[1] == d2n.cont_type).all()))
    R["E_D2N_origid_galid_match"] = float((d2n["Original ID"].str.extract(r"cont_(\d+)_")[0].astype("int64") == d2n.galaxy_id).mean())
    R["E_D1N_has_central_galaxy"] = int(T["D1_N"].z_central.notna().sum())
    ids = T["D1_N"]["Object ID"].str[4:].astype(int)
    R["E_D1N_central_galaxy_by_5k_block"] = T["D1_N"].z_central.notna().groupby(ids // 5000).mean().round(3).to_dict()
    ids = T["D1_L"]["Lens ID"].str[4:].astype(int)
    R["E_D1L_quad_frac_by_5k_block"] = (T["D1_L"].num_images == 4).groupby(ids // 5000).mean().round(3).to_dict()
    R["E_D1L_ps_is_src_centre"] = bool(np.allclose(T["D1_L"].psx, T["D1_L"].srcx) and np.allclose(T["D1_L"].psy, T["D1_L"].srcy))

    # G. D1_N without central galaxy: is the stamp near-blank? (train+val, i-band aperture flux from keys)
    k = keys[keys.simset == "D1_N"].copy()
    k["has_gal"] = k.obj.map(T["D1_N"].set_index("Object ID").z_central.notna())
    R["G_D1N_ap6i_median_by_has_gal"] = k.groupby("has_gal").ap6_i.median().round(2).to_dict()
    R["G_D1N_frac_snr_lt5_by_has_gal"] = k.groupby("has_gal").snr6_det.apply(lambda s: float((s < 5).mean())).round(4).to_dict()

    # F. D2 galaxy reuse
    gl, gn = set(T["D2_L"].galaxy_id), set(T["D2_N"].galaxy_id)
    R["F_D2_galaxy_id_shared_L_N"] = len(gl & gn)
    R["F_D2L_galaxy_id_eq_original_id"] = bool((T["D2_L"].galaxy_id == T["D2_L"]["Original ID"]).all())
    R["F_D1_vs_D2_radec_overlap"] = "D1 RA 55-70 Dec -43..-30 (DC2-like); D2 HSC fields"

    os.makedirs(os.path.join(W, "results", "data_checks"), exist_ok=True)
    json.dump(R, open(os.path.join(W, "results", "data_checks", "truth_checks.json"), "w"), indent=1, default=str)
    for kk, v in R.items():
        print(kk, ":", v)


if __name__ == "__main__":
    main()
