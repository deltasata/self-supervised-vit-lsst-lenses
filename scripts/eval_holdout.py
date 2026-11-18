"""The ONLY script that reads the locked holdout.

  python scripts/eval_holdout.py --check_isolation                      (no model; not an evaluation)
  python scripts/eval_holdout.py --runs RUN_A [RUN_B ...] --name EVAL_NAME [--tta 8]

* --check_isolation proves the holdout never entered training: sha256 of every data file vs data/MANIFEST.md, no shared
  object or ID group between holdout and train/val, no holdout stamp byte-identical (any band) to a train/val stamp,
  and no experiment config that mentions the holdout. Result: results/holdout_isolation.json. An evaluation refuses to
  run unless this check passed for the current data files.
* every run must be marked frozen in the frozen-model table EXPERIMENTS.md (table "Frozen models", column "frozen?" = yes)
* several runs -> ensemble = mean of logits (each best.pt with dihedral TTA)
* budget: every model whose holdout scores are computed for the first time counts as one evaluation, and an ensemble
  of several counts one more. Scores already in the store are reused, not recomputed or recounted.
  Beyond 25 it refuses unless --user_approved_over_budget is given.
* the same eval name can be run only once; every call is appended to logs/holdout_evals.log
Results:
  experiments/holdout_evals/<name>.json                                  metrics
  results/holdout_predictions/<run>/holdout_logits_tta8.npy              per-model holdout logits (holdout row order)
  results/holdout_predictions/ENS_<name>/holdout_logits.npy              ensemble mean logit
  results/holdout_predictions/holdout_y.npy, holdout_keys.parquet        labels and evaluation metadata (row order)

--score_all:
  after the final models are frozen, score EVERY run with a best.pt on the holdout for the record. These scores are
  archive only: no selection, tuning, thresholding, calibration or ensembling may ever read results/holdout_predictions/
  (only this script writes it; scripts/predict_store.py --summary reads it to report). Logged as SCORE_ALL, outside the
  25-evaluation budget. Requires at least one frozen model (selection finished first).
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import W, tta_predict  # noqa: E402
from lagn.metrics import all_metrics  # noqa: E402
from lagn.models import build_model  # noqa: E402

LOG = os.path.join(W, "logs", "holdout_evals.log")
HO = os.path.join(W, "data", "holdout")
OUT = os.path.join(W, "experiments", "holdout_evals")
PRED = os.path.join(W, "results", "holdout_predictions")     # kept apart from train/val scores
ISO = os.path.join(W, "results", "holdout_isolation.json")
BUDGET = 25
KEYCOLS = ["obj", "simset", "pipeline", "label", "id", "ap6_i", "snr6_det"]


def frozen_runs():
    txt = open(os.path.join(W, "EXPERIMENTS.md")).read().splitlines()
    hdr = next(l for l in txt if l.startswith("| ID |"))
    cols = [c.strip() for c in hdr.strip("|").split("|")]
    fi = cols.index("frozen?")
    out = set()
    for l in txt:
        if l.startswith("|") and not l.startswith("| ID") and not l.startswith("|---"):
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) > fi and c[fi].lower().startswith("yes"):
                out.add(c[0])
    return out


def n_evals():
    if not os.path.exists(LOG):
        return 0, set()
    names, n = set(), 0
    for l in open(LOG):
        m = re.search(r"\| EVAL \| name=(\S+)", l)
        if m:
            c = re.search(r"\| counted=(\d+)", l)
            n += int(c.group(1)) if c else 1
            names.add(m.group(1))
    return n, names


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_isolation():
    man = open(os.path.join(W, "data", "MANIFEST.md")).read()
    files = ["X_trainval.npy", "y_trainval.npy", "keys_trainval.parquet", "holdout/X_holdout.npy",
             "holdout/y_holdout.npy", "holdout/keys_holdout.parquet"]
    res = {"time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}", "sha256": {}}
    for f in files:
        h = sha256(os.path.join(W, "data", f))
        res["sha256"][f] = {"sha256": h, "matches_manifest": h in man}
    ktv = pd.read_parquet(os.path.join(W, "data", "keys_trainval.parquet"))
    kho = pd.read_parquet(os.path.join(HO, "keys_holdout.parquet"))
    res["n_trainval"], res["n_holdout"] = len(ktv), len(kho)
    res["holdout_split_values"] = sorted(kho.split.unique().tolist())
    res["trainval_split_values"] = sorted(ktv.split.unique().tolist())
    res["shared_objects"] = int(len(set(ktv.obj) & set(kho.obj)))
    res["shared_id_groups"] = int(len(set(ktv.group) & set(kho.group)))
    # byte-identical stamps (per band), holdout vs train/val
    Xtv = np.load(os.path.join(W, "data", "X_trainval.npy"), mmap_mode="r")
    Xho = np.load(os.path.join(HO, "X_holdout.npy"), mmap_mode="r")
    def band_hashes(X):
        s = set()
        for i in range(0, len(X), 5000):
            a = np.ascontiguousarray(X[i:i + 5000])
            for j in range(len(a)):
                for b in range(5):
                    s.add(hashlib.sha1(a[j, b].tobytes()).digest())
        return s
    htv, hho = band_hashes(Xtv), band_hashes(Xho)
    res["identical_band_images"] = int(len(htv & hho))
    # experiment configs that mention the holdout (none may)
    bad = [f for f in glob.glob(os.path.join(W, "experiments", "*", "config.json"))
           if "holdout" in open(f).read().lower()]
    res["configs_mentioning_holdout"] = [os.path.relpath(f, W) for f in bad]
    res["passed"] = bool(all(v["matches_manifest"] for v in res["sha256"].values())
                         and res["shared_objects"] == 0 and res["shared_id_groups"] == 0
                         and res["identical_band_images"] == 0 and not bad
                         and res["holdout_split_values"] == ["holdout"] and "holdout" not in res["trainval_split_values"])
    os.makedirs(os.path.dirname(ISO), exist_ok=True)
    json.dump(res, open(ISO, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(f"{res['time']} | CHECK | isolation passed={res['passed']} | job={os.environ.get('SLURM_JOB_ID')}\n")
    print(json.dumps(res, indent=1))
    return res["passed"]


def score_runs(runs, X, tta, t0=None, budget_min=1e9):
    """Holdout TTA logits for each run, written to PRED/<run>/holdout_logits_tta8.npy (skips runs already scored)."""
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = None
    for r in runs:
        f = os.path.join(PRED, r, "holdout_logits_tta8.npy")
        if os.path.exists(f):
            continue
        if t0 is not None and (time.time() - t0) / 60 > budget_min:
            print("time budget reached; resubmit to continue", flush=True)
            return False
        if Xt is None:
            Xt = torch.from_numpy(X).to(dev)
        try:
            cfg = json.load(open(os.path.join(W, "experiments", r, "config.json")))
            cfg["model"].pop("init_ckpt", None)
            cfg["model"]["pretrained"] = False
            m = build_model(cfg).to(dev)
            m.load_state_dict(torch.load(os.path.join(W, "experiments", r, "best.pt"), map_location=dev))
            m.eval()
        except Exception as e:                                             # noqa: BLE001
            print(f"SKIP {r}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 2048):
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    out.append(tta_predict(m, Xt[i:i + 2048], tta).float().squeeze(1).cpu())
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.save(f, torch.cat(out).numpy().astype(np.float32))
        print(f"scored {r}", flush=True)
        del m
    return True


def save_labels(y, keys):
    os.makedirs(PRED, exist_ok=True)
    if not os.path.exists(os.path.join(PRED, "holdout_y.npy")):
        np.save(os.path.join(PRED, "holdout_y.npy"), y.astype(np.int8))
        keys[KEYCOLS].to_parquet(os.path.join(PRED, "holdout_keys.parquet"))


def score_all(a):
    import time
    t0 = time.time()
    if not os.path.exists(ISO) or not json.load(open(ISO))["passed"]:
        sys.exit("REFUSED: run --check_isolation first (it must pass)")
    fz = frozen_runs()
    if not fz:
        sys.exit("REFUSED: freeze the final models first (selection must be finished before any holdout scores exist)")
    runs = sorted(r for r in os.listdir(os.path.join(W, "experiments"))
                  if os.path.exists(os.path.join(W, "experiments", r, "best.pt"))
                  and os.path.exists(os.path.join(W, "experiments", r, "config.json"))
                  and (not a.only or r in a.only))
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | SCORE_ALL | runs={len(runs)} | frozen={','.join(sorted(fz))} "
                f"| archive only | job={os.environ.get('SLURM_JOB_ID')}\n")
    X = np.load(os.path.join(HO, "X_holdout.npy"))
    save_labels(np.load(os.path.join(HO, "y_holdout.npy")), pd.read_parquet(os.path.join(HO, "keys_holdout.parquet")))
    done = score_runs(runs, X, a.tta, t0, a.budget_min)
    print("ALL DONE" if done else "PARTIAL", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--name")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--check_isolation", action="store_true")
    ap.add_argument("--score_all", action="store_true")
    ap.add_argument("--only", nargs="+", help="--score_all: restrict to these finished runs (archive top-up)")
    ap.add_argument("--budget_min", type=float, default=1380)
    ap.add_argument("--user_approved_over_budget", action="store_true")
    ap.add_argument("--rerun_after_failure", action="store_true",
                    help="re-run an eval NAME whose earlier call died before writing "
                         "experiments/holdout_evals/<name>.json (no result ever existed); same runs required; counted 0")
    a = ap.parse_args()
    if a.check_isolation:
        sys.exit(0 if check_isolation() else "ISOLATION CHECK FAILED")
    if a.score_all:
        return score_all(a)
    if not (a.runs and a.name):
        sys.exit("--runs and --name are required")
    if not os.path.exists(ISO) or not json.load(open(ISO))["passed"]:
        sys.exit("REFUSED: run --check_isolation first (it must pass)")
    fz = frozen_runs()
    notfz = [r for r in a.runs if r not in fz]
    if notfz:
        sys.exit(f"REFUSED: not frozen in EXPERIMENTS.md: {notfz}")
    n, names = n_evals()
    rerun = False
    if a.name in names:
        prev = [l for l in open(LOG) if f"| EVAL | name={a.name} |" in l]
        if not (a.rerun_after_failure and not os.path.exists(os.path.join(OUT, f"{a.name}.json"))
                and all(f"| runs={','.join(a.runs)} |" in l for l in prev)):
            sys.exit(f"REFUSED: eval name {a.name} already used (one evaluation per frozen model/ensemble)")
        rerun = True
    new = [r for r in a.runs if not os.path.exists(os.path.join(PRED, r, "holdout_logits_tta8.npy"))]
    counted = 0 if rerun else len(new) + (1 if len(a.runs) > 1 else 0)
    if n + counted > BUDGET and not a.user_approved_over_budget:
        sys.exit(f"REFUSED: {n} holdout evaluations already, this call adds {counted} (budget ~{BUDGET}); override with --user_approved_over_budget")
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | EVAL | name={a.name} | runs={','.join(a.runs)} | "
                f"new={','.join(new) or '-'} | tta={a.tta} | counted={counted} | total={n + counted} | "
                f"job={os.environ.get('SLURM_JOB_ID')}{' | RERUN_AFTER_FAILURE' if rerun else ''}\n")

    X = np.load(os.path.join(HO, "X_holdout.npy"))
    y = np.load(os.path.join(HO, "y_holdout.npy"))
    keys = pd.read_parquet(os.path.join(HO, "keys_holdout.parquet"))
    save_labels(y, keys)
    score_runs(a.runs, X, a.tta)
    logits, per_run = [], {}
    for r in a.runs:
        z = np.load(os.path.join(PRED, r, "holdout_logits_tta8.npy"))
        logits.append(z)
        per_run[r] = all_metrics(y, 1 / (1 + np.exp(-z.astype(np.float64))), keys)
    lg = np.mean(logits, 0).astype(np.float32)
    if len(a.runs) > 1:
        os.makedirs(os.path.join(PRED, f"ENS_{a.name}"), exist_ok=True)
        np.save(os.path.join(PRED, f"ENS_{a.name}", "holdout_logits.npy"), lg)
    met = all_metrics(y, 1 / (1 + np.exp(-lg.astype(np.float64))), keys)
    res = {"name": a.name, "runs": a.runs, "tta": a.tta, "eval_total_after": n + counted, "holdout": met,
           "holdout_per_run": per_run}
    json.dump(res, open(os.path.join(OUT, f"{a.name}.json"), "w"), indent=1)
    np.save(os.path.join(OUT, f"{a.name}_logits.npy"), lg)
    print(json.dumps({k: met[k] for k in ("auc", "logloss", "tpr@1e-2", "tpr@1e-3", "pr_auc", "per_pipeline",
                                          "per_pipeline_tpr@1e-3", "auc_flux_matched")}, indent=1))
    for r, m_ in per_run.items():
        print(r, {k: m_[k] for k in ("auc", "logloss", "per_pipeline")})


if __name__ == "__main__":
    main()
