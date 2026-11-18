"""All 200,000 labelled objects (train+val 170k, then the former holdout 30k) for the production classifier.

production/ trains on all 200k with 10-fold CV; there is no holdout.
Row order: data/X_trainval.npy rows 0..169,999, then data/holdout/X_holdout.npy rows as 170,000..199,999. Stamps stay
memory-mapped; only the rows a run asks for are read.
"""
import os
import sys

import numpy as np
import pandas as pd

PROD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(PROD)
sys.path.insert(0, os.path.join(W, "src"))
from lagn.data import kfold_assign  # noqa: E402

N_FOLDS = 10


class CatRows:
    """Row-indexable view of two (N, 5, 41, 41) arrays stacked along axis 0 (index arrays only)."""

    def __init__(self, a, b):
        self.a, self.b, self.n = a, b, len(a)
        self.shape = (len(a) + len(b),) + a.shape[1:]
        self.dtype = a.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.where(idx)[0]
        assert idx.ndim == 1 and idx.dtype.kind in "iu", "CatRows supports 1-D index arrays only"
        out = np.empty((len(idx),) + self.a.shape[1:], self.dtype)
        m = idx < self.n
        out[m] = self.a[idx[m]]
        out[~m] = self.b[idx[~m] - self.n]
        return out


def load_all(mmap=True):
    mode = "r" if mmap else None
    Xa = np.load(os.path.join(W, "data", "X_trainval.npy"), mmap_mode=mode)
    Xb = np.load(os.path.join(W, "data", "holdout", "X_holdout.npy"), mmap_mode=mode)
    y = np.concatenate([np.load(os.path.join(W, "data", "y_trainval.npy")),
                        np.load(os.path.join(W, "data", "holdout", "y_holdout.npy"))])
    ka = pd.read_parquet(os.path.join(W, "data", "keys_trainval.parquet")).assign(source="trainval")
    kb = pd.read_parquet(os.path.join(W, "data", "holdout", "keys_holdout.parquet")).assign(source="holdout")
    keys = pd.concat([ka, kb], ignore_index=True)
    assert len(Xa) + len(Xb) == len(y) == len(keys) == 200_000
    assert keys.obj.is_unique, "an object appears twice in the 200k"
    assert (keys.label.values == y).all(), "labels in keys and y disagree"
    return CatRows(Xa, Xb), y, keys


def folds(keys):
    """Fold id for every row: stratified by simset x 5,000-ID block, deterministic (lagn.data.kfold_assign)."""
    return kfold_assign(keys, N_FOLDS)
