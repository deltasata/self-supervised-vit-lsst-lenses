"""Read all 1,000,000 FITS files once into a canonical array for data verification.

Output (data/verify/, verification only -- contains holdout objects, never a training input):
  raw_all.npy        (200000, 5, 41, 41) float32, bands g r i z y, canonical order:
                     simset D1_L, D1_N, D2_L, D2_N; id ascending within each
  raw_all_meta.parquet  one row per object: simset, id, per-band data sha1, header signature,
                     RA_OBJ/DEC_OBJ (if present), dtype/shape/NaN info, max |float32-float64| error
Usage: python scripts/verify_pack_raw.py [nproc]
"""
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy.io import fits

R = os.environ.get("LAGN_RAW_DIR", "data/raw/all_data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "verify")
SIMSETS = ["D1_L", "D1_N", "D2_L", "D2_N"]
BANDS = "grizy"
NID = 50000


def read_obj(key):
    s, i = key
    img = np.empty((5, 41, 41), np.float32)
    rec = {"simset": s, "id": i}
    sigs = []
    for b, band in enumerate(BANDS):
        fn = f"{R}/{s}{i:08d}_{band}.fits"
        with fits.open(fn, memmap=False) as hd:
            rec.setdefault("n_hdu", len(hd))
            h = hd[0].header
            d = hd[0].data
            sigs.append(";".join(f"{k}" for k in h.keys()))
            if b == 0:
                rec["ra"] = h.get("RA_OBJ", np.nan)
                rec["dec"] = h.get("DEC_OBJ", np.nan)
                rec["dtype"] = str(d.dtype)
                rec["shape"] = "x".join(map(str, d.shape))
                rec["bitpix"] = h.get("BITPIX")
            if d.shape != (41, 41):
                raise ValueError(f"{fn} shape {d.shape}")
            d64 = np.asarray(d, dtype=np.float64)
            rec[f"nan_{band}"] = int((~np.isfinite(d64)).sum())
            rec[f"sha_{band}"] = hashlib.sha1(np.ascontiguousarray(d64.astype("<f8")).tobytes()).hexdigest()[:16]
            img[b] = d64
            err = np.abs(img[b].astype(np.float64) - d64)
            rec[f"f32err_{band}"] = float(err.max())
            rec[f"fsize_{band}"] = os.path.getsize(fn)
    rec["hdr_sig"] = "|".join(sorted(set(sigs)))
    rec["hdr_same_all_bands"] = len(set(sigs)) == 1
    return img, rec


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    os.makedirs(OUT, exist_ok=True)
    keys = [(s, i) for s in SIMSETS for i in range(NID)]
    X = np.lib.format.open_memmap(os.path.join(OUT, "raw_all.npy.tmp"), mode="w+", dtype="<f4",
                                  shape=(len(keys), 5, 41, 41))
    recs = []
    t = time.time()
    with Pool(nproc) as p:
        for n, (img, rec) in enumerate(p.imap(read_obj, keys, chunksize=256)):
            X[n] = img
            recs.append(rec)
            if n % 20000 == 0:
                print(n, f"{time.time() - t:.0f}s", flush=True)
    X.flush()
    del X
    os.replace(os.path.join(OUT, "raw_all.npy.tmp"), os.path.join(OUT, "raw_all.npy"))
    pd.DataFrame(recs).to_parquet(os.path.join(OUT, "raw_all_meta.parquet"))
    print("done", f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
