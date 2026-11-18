"""Data preparation: labels, locked holdout, validation split, packed arrays.

Grouping: the unit of splitting is the 8-digit ID *across all four simsets* (an ID goes entirely to
holdout, val or train in D1_L, D1_N, D2_L and D2_N), optionally merged with verified duplicate pairs.
Splitting whole ID-groups also makes the holdout exactly stratified: 7,500 IDs -> 7,500 per simset.

Outputs
  data/objects.parquet           canonical order: obj, simset, pipeline, label, id, group
  data/splits.parquet            canonical order + split in {train, val, holdout}
  data/X_trainval.npy etc.       shuffled (seed) train+val rows, float32 LE, (N,5,41,41), g r i z y, raw
  data/holdout/X_holdout.npy, y_holdout.npy, keys_holdout.parquet   (write-once)
  data/norm_stats.json           per-band global noise sigma from TRAIN rows
Usage: python scripts/prepare_data.py [--dup_pairs file.csv]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from astropy.io import fits

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(W, "data")
V = os.path.join(D, "verify")
HO = os.path.join(D, "holdout")
R = os.environ.get("LAGN_RAW_DIR", "data/raw/all_data")
SIMSETS = ["D1_L", "D1_N", "D2_L", "D2_N"]
SEED_HOLDOUT, SEED_VAL, SEED_SHUFFLE = 20261006, 20261007, 20261008


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup_pairs", default=None)
    a = ap.parse_args()

    st = pd.read_parquet(os.path.join(V, "obj_stats.parquet"), columns=["simset", "id", "ap6_i", "snr6_det"])
    N = len(st)
    obj = pd.DataFrame({"obj": [f"{s}{i:08d}" for s, i in zip(st.simset, st.id)], "simset": st.simset.values,
                        "pipeline": st.simset.str[:2].values, "label": (st.simset.str[3] == "L").astype(np.int8).values,
                        "id": st.id.values.astype(np.int32)})
    assert N == 200000 and (obj.simset.values == np.repeat(SIMSETS, 50000)).all()

    # ---- groups over IDs (0..49999). verify_dups.py found no exact/near duplicates, no ID pairing and no
    # background reuse (scripts/verify_dups.py), so no extra links; --dup_pairs can add verified pairs (csv: id_a,id_b).
    uf = UF(50000)
    nlinks = 0
    if a.dup_pairs:
        for ida, idb in pd.read_csv(a.dup_pairs)[["id_a", "id_b"]].values:
            uf.union(int(ida), int(idb))
            nlinks += 1
    root = np.array([uf.find(j) for j in range(50000)])
    ug, gsize = np.unique(root, return_counts=True)
    print(f"dup links {nlinks}; id-groups {len(ug)}; largest {gsize.max()}; groups>1: {(gsize > 1).sum()}")
    obj["group"] = root[obj.id.values]

    # ---- holdout: 7,500 IDs worth of groups (greedy fill in random group order)
    rng = np.random.RandomState(SEED_HOLDOUT)
    order = rng.permutation(len(ug))
    def take(order, target, gsz):
        sel, n = [], 0
        for g in order:
            if n + gsz[g] <= target:
                sel.append(g)
                n += gsz[g]
            if n == target:
                break
        return np.array(sel), n
    if (gsize == 1).all():
        # singleton groups: stratify by ID block of 5,000 (D1 IDs are ordered in blocks of different subtypes,
        # e.g. D1_L 0-24999 vs 25000-49999): 750 holdout + 425 val IDs per block
        ids = np.arange(50000)
        hold_ids = np.zeros(50000, bool)
        val_ids = np.zeros(50000, bool)
        rv = np.random.RandomState(SEED_VAL)
        for blk in range(10):
            b = ids[blk * 5000:(blk + 1) * 5000]
            h = rng.choice(b, 750, replace=False)
            hold_ids[h] = True
            v_ = rv.choice(np.setdiff1d(b, h), 425, replace=False)
            val_ids[v_] = True
    else:
        hsel, nh = take(order, 7500, gsize)
        hold_ids = np.isin(root, ug[hsel])
        rest = np.setdiff1d(np.arange(len(ug)), hsel)
        rng = np.random.RandomState(SEED_VAL)
        vsel, nv = take(rng.permutation(rest), 4250, gsize)
        val_ids = np.isin(root, ug[vsel])
    assert hold_ids.sum() == 7500 and val_ids.sum() == 4250
    assert not (hold_ids & val_ids).any()
    split_id = np.where(hold_ids, "holdout", np.where(val_ids, "val", "train"))
    obj["split"] = split_id[obj.id.values]
    print(obj.groupby(["simset", "split"]).size().unstack())
    obj[["obj", "simset", "pipeline", "label", "id", "group"]].to_parquet(os.path.join(D, "objects.parquet"))
    obj.to_parquet(os.path.join(D, "splits.parquet"))

    # ---- pack
    X = np.load(os.path.join(V, "raw_all.npy"), mmap_mode="r")
    tv = np.where(obj.split.values != "holdout")[0]
    tv = tv[np.random.RandomState(SEED_SHUFFLE).permutation(len(tv))]
    ho = np.where(obj.split.values == "holdout")[0]
    ho = ho[np.random.RandomState(SEED_SHUFFLE + 1).permutation(len(ho))]

    keys_tv = obj.iloc[tv].reset_index(drop=True).copy()
    keys_tv["canon_row"] = tv
    keys_tv["ap6_i"] = st.ap6_i.values[tv]          # evaluation metadata only (flux bins), never a model input
    keys_tv["snr6_det"] = st.snr6_det.values[tv]
    Xtv = np.ascontiguousarray(X[np.sort(tv)])       # sorted read, then reorder in memory
    Xtv = Xtv[np.argsort(np.argsort(tv))]
    assert np.array_equal(Xtv[0], X[tv[0]]) and np.array_equal(Xtv[-1], X[tv[-1]])
    np.save(os.path.join(D, "X_trainval.npy"), Xtv.astype("<f4"))
    np.save(os.path.join(D, "y_trainval.npy"), keys_tv.label.values.astype(np.int8))
    keys_tv.to_parquet(os.path.join(D, "keys_trainval.parquet"))

    # norm stats from TRAIN rows only
    tr = keys_tv.split.values == "train"
    yy, xx = np.mgrid[:41, :41]
    outer = np.hypot(yy - 20, xx - 20) >= 15
    sub = Xtv[np.where(tr)[0][:40000]][:, :, outer].astype(np.float64)
    med = np.median(sub, axis=2, keepdims=True)
    sig = 1.4826 * np.median(np.abs(sub - med), axis=2)
    ns = {"sigma": np.median(sig, axis=0).tolist(), "bands": "grizy",
          "note": "median over 40k train stamps of per-stamp robust sigma (outer annulus r>=15)"}
    json.dump(ns, open(os.path.join(D, "norm_stats.json"), "w"), indent=1)
    print("norm_stats", ns)

    os.makedirs(HO, exist_ok=True)
    keys_ho = obj.iloc[ho].reset_index(drop=True).copy()
    keys_ho["canon_row"] = ho
    keys_ho["ap6_i"] = st.ap6_i.values[ho]
    keys_ho["snr6_det"] = st.snr6_det.values[ho]
    Xho = np.ascontiguousarray(X[np.sort(ho)])
    Xho = Xho[np.argsort(np.argsort(ho))]
    np.save(os.path.join(HO, "X_holdout.npy"), Xho.astype("<f4"))
    np.save(os.path.join(HO, "y_holdout.npy"), keys_ho.label.values.astype(np.int8))
    keys_ho.to_parquet(os.path.join(HO, "keys_holdout.parquet"))

    # ---- validate packed arrays against 2,000 random FITS files (fresh reads)
    rng = np.random.RandomState(1)
    for name, Xp, kp in (("trainval", Xtv, keys_tv), ("holdout", Xho, keys_ho)):
        rows = rng.choice(len(kp), 1000, replace=False)
        bad = 0
        for r in rows:
            band = "grizy"[rng.randint(5)]
            b = "grizy".index(band)
            d = fits.getdata(f"{R}/{kp.obj[r]}_{band}.fits").astype(np.float32)
            bad += not np.array_equal(d, Xp[r, b])
        print(f"validate {name}: {len(rows)} random (row, band) vs FITS -> mismatches {bad}")
        assert bad == 0
    print("done")


if __name__ == "__main__":
    main()
