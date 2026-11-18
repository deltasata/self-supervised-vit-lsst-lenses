"""Leakage / integrity checks for the production 10-fold split; writes production/results/folds.parquet.

Checks: 200k unique objects, 50k per simset, label = simset type; every object in exactly one validation fold; folds
balanced per simset; deterministic. Content duplicates (exact and near, 8 dihedral transforms) over all 200k objects were
already excluded (scripts/verify_dups.py), so object-level folds cannot share images.
"""
import json
import os

import numpy as np
import pandas as pd

import data_all

X, y, keys = data_all.load_all()
f = data_all.folds(keys)
assert (f == data_all.folds(keys.copy())).all(), "fold assignment not deterministic"
rep = {"n": int(len(keys)), "per_simset": keys.simset.value_counts().to_dict()}
assert keys.obj.is_unique and len(keys) == 200_000
assert all(v == 50_000 for v in rep["per_simset"].values())
assert ((keys.simset.str[-1] == "L").values.astype(int) == y).all(), "label does not match simset"
assert set(np.unique(f)) == set(range(data_all.N_FOLDS))
tab = pd.crosstab(f, keys.simset)
rep["fold_sizes"] = {int(k): int(v) for k, v in pd.Series(f).value_counts().sort_index().items()}
rep["per_fold_simset_min_max"] = [int(tab.values.min()), int(tab.values.max())]
assert tab.values.min() >= 4900 and tab.values.max() <= 5100, tab
for k in range(data_all.N_FOLDS):            # train and validation of model k are disjoint and cover everything
    va, tr = keys.obj[f == k], keys.obj[f != k]
    assert not set(va) & set(tr) and len(va) + len(tr) == 200_000
# encoder exposure: all production encoders were pretrained (unlabelled) on the 153k original train rows only
seen = (keys.source == "trainval") & (keys.split == "train")
rep["seen_in_pretraining"] = int(seen.sum())
pd.DataFrame({"obj": keys.obj, "simset": keys.simset, "label": y, "source": keys.source, "fold": f,
              "seen_in_pretraining": seen.values}).to_parquet(os.path.join(data_all.PROD, "results", "folds.parquet"))
rep["passed"] = True
json.dump(rep, open(os.path.join(data_all.PROD, "results", "fold_checks.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
