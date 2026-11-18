"""Loader for the organizers' truth tables (data/truth/fits_tables/, from fits_tables.zip).
Metadata only: these tables are never model inputs; they are used for evaluation breakdowns, regression targets and
auxiliary training targets.
"""
import os

import numpy as np
import pandas as pd
from astropy.table import Table

TDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "truth", "fits_tables")
FILES = {"D1_L": "table_lensed_quasars_injected.fits", "D1_N": "table_non_lenses_injected.fits",
         "D2_L": "table_simct_lenses.fits", "D2_N": "table_simct_non_lenses.fits"}


def load(simset):
    t = Table.read(os.path.join(TDIR, FILES[simset]))
    cols = {}
    for c in t.colnames:
        a = np.asarray(t[c])
        if a.dtype.kind in "iuf":
            a = a.astype(a.dtype.newbyteorder("="))           # FITS is big-endian; pandas needs native order
        if a.ndim == 2:                                   # e.g. n_s_sers (2 values per row)
            for k in range(a.shape[1]):
                cols[f"{c}_{k}"] = a[:, k]
        else:
            cols[c] = np.char.strip(a.astype(str)) if a.dtype.kind in "SU" else a
    return pd.DataFrame(cols)
