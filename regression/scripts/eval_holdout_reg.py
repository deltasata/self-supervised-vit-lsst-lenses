"""The ONLY regression script that reads the locked holdout: holdout lens stamps and the truth-table
targets of holdout objects are used here and nowhere else.

  python regression/scripts/eval_holdout_reg.py --runs RUN [RUN ...] --name EVAL_NAME [--tta 8]
  python regression/scripts/eval_holdout_reg.py --runs RUN [RUN ...] --dry_run_val
      pipeline test on the VALIDATION lenses (never opens data/holdout): rebuilds the val targets through targets.build(),
      checks them against targets_trainval.parquet, and checks that each run's TTA loss reproduces its metrics.json.

Rules (the same as scripts/eval_holdout.py for classification):
* the classification isolation check (results/holdout_isolation.json) must have passed; in addition no regression config
  may mention the holdout and the regression target table may hold no holdout object
* every run must be marked frozen in the frozen-model table regression/EXPERIMENTS.md (table with "| ID |" header, column "frozen?" = yes)
* several runs -> ensemble = equal-weight Gaussian mixture in standardised units (mean of means; variance = mean variance
  + spread of the means) and the mean quad logit, as regression/scripts/ens_val.py does on val
* one call per eval name; each new model counts as one evaluation and an ensemble of several one more. The regression
  budget is REG_BUDGET, separate from the classification count; beyond it the script refuses unless
  --user_approved_over_budget. Every call is appended to logs/holdout_evals.log as "| REG_EVAL |" (a tag the
  classification counter does not match)
* holdout loss = the validation loss of train_reg.py: mean standardised Gaussian NLL of the 5 targets + w_quad * BCE(quad)
Results (regression/results/ only):
  holdout_evals/<name>.json                       metrics: loss, per pipeline / per double-quad, members, val reference
  holdout_predictions/<run>/holdout_pred_tta8.npz  per-model TTA prediction (standardised mean/var, quad logit)
  holdout_predictions/ENS_<name>/holdout_pred.npz  ensemble prediction in physical units (mean, sigma, quad_logit)
  holdout_predictions/holdout_targets.parquet      holdout lens targets and row order (evaluation metadata)
"""
import argparse
import datetime
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

RW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(RW)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_reg import RegNet, dihedral_mats, predict, report  # noqa: E402  (also puts src/ and scripts/ on the path)
from targets import BINARY, TARGETS, build  # noqa: E402

LOG = os.path.join(W, "logs", "holdout_evals.log")
HO = os.path.join(W, "data", "holdout")
ISO = os.path.join(W, "results", "holdout_isolation.json")
OUT = os.path.join(RW, "results", "holdout_evals")
PRED = os.path.join(RW, "results", "holdout_predictions")
REG_BUDGET = 10


def frozen_runs():
    txt = open(os.path.join(RW, "EXPERIMENTS.md")).read().splitlines()
    hdr = next(l for l in txt if l.startswith("| ID |"))
    cols = [c.strip() for c in hdr.strip("|").split("|")]
    fi = cols.index("frozen?")
    out = set()
    for l in txt:
        if l.startswith("|") and not l.startswith("| ID") and not l.startswith("|---"):
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) > fi and c[fi].lower().startswith("yes"):
                out.add(c[0].strip("`*"))
    return out


def n_evals():
    if not os.path.exists(LOG):
        return 0, set()
    names, n = set(), 0
    for l in open(LOG):
        m = re.search(r"\| REG_EVAL \| name=(\S+)", l)
        if m:
            n += int(re.search(r"\| counted=(\d+)", l).group(1))
            names.add(m.group(1))
    return n, names


def check_isolation():
    """Refuses unless the holdout never reached regression training."""
    if not os.path.exists(ISO) or not json.load(open(ISO))["passed"]:
        sys.exit("REFUSED: the classification isolation check (scripts/eval_holdout.py --check_isolation) has not passed")
    bad = [f for f in glob.glob(os.path.join(RW, "experiments", "*", "config.json")) + glob.glob(os.path.join(RW, "configs", "*.json"))
           if "holdout" in open(f).read().lower()]
    if bad:
        sys.exit(f"REFUSED: regression configs mention the holdout: {bad}")
    T = pd.read_parquet(os.path.join(RW, "results", "targets_trainval.parquet"))
    kho = pd.read_parquet(os.path.join(HO, "keys_holdout.parquet"))
    if set(T.obj) & set(kho.obj) or set(T.split) - {"train", "val"}:
        sys.exit("REFUSED: holdout objects in regression/results/targets_trainval.parquet")


def lens_targets(keys):
    """Targets of the lens rows of `keys` (any split), built from the truth tables; column `row` = row in keys/X."""
    k = keys[keys.label == 1][["obj"]].copy()
    k["row"] = np.flatnonzero(keys.label.values == 1)
    k["pipeline_key"] = keys.pipeline.values[k.row.values]
    T = k.merge(pd.concat([build("D1_L"), build("D2_L")], ignore_index=True), on="obj", how="left", validate="one_to_one")
    assert T[TARGETS + BINARY].notna().all().all() and (T.pipeline == T.pipeline_key).all(), "missing targets"
    return T.drop(columns="pipeline_key").sort_values("row").reset_index(drop=True)


def load_run(r, dev):
    d = os.path.join(RW, "experiments", r)
    cfg = json.load(open(os.path.join(d, "config.json")))
    cfg["model"].pop("init_ckpt", None)                      # weights come from best.pt
    cfg["model"]["pretrained"] = False
    m = RegNet(cfg).to(dev)
    m.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=dev))
    m.eval()
    sc = json.load(open(os.path.join(d, "scaler.json")))
    assert sc["targets"] == TARGETS
    return m, cfg, {"mean": np.array(sc["mean"]), "std": np.array(sc["std"])}


def run_preds(runs, X, rows, tta, dev, cache=None):
    """Per run: standardised TTA mean/var and quad logit for X[rows]. Cached per run under `cache` if given."""
    Md = dihedral_mats().to(dev)
    Xt, P, scs, cfgs = None, {}, [], []
    for r in runs:
        f = os.path.join(cache, r, f"holdout_pred_tta{tta}.npz") if cache else None
        m, cfg, sc = load_run(r, dev) if not (f and os.path.exists(f)) else (None, None, None)
        if m is None:
            cfg = json.load(open(os.path.join(RW, "experiments", r, "config.json")))
            s = json.load(open(os.path.join(RW, "experiments", r, "scaler.json")))
            sc = {"mean": np.array(s["mean"]), "std": np.array(s["std"])}
            z = np.load(f)
            assert (z["rows"] == rows).all()
            P[r] = (z["mu"], z["var"], z["quad_logit"])
        else:
            if Xt is None:
                Xt = torch.from_numpy(np.ascontiguousarray(X[rows])).to(dev)
            mu, var, ql = predict(m, Xt, Md, tta=tta)
            P[r] = (mu.cpu().numpy().astype(np.float64), var.cpu().numpy().astype(np.float64), ql.cpu().numpy().astype(np.float64))
            if f:
                os.makedirs(os.path.dirname(f), exist_ok=True)
                np.savez(f, rows=rows, mu=P[r][0], var=P[r][1], quad_logit=P[r][2], targets=np.array(TARGETS))
            print(f"predicted {r}", flush=True)
            del m
        scs.append(sc), cfgs.append(cfg)
    for sc in scs[1:]:                                        # same train split -> identical scalers
        assert np.allclose(sc["mean"], scs[0]["mean"]) and np.allclose(sc["std"], scs[0]["std"])
    wq = {c["train"]["w_quad"] for c in cfgs}
    assert len(wq) == 1
    return P, scs[0], wq.pop()


def mixture(P):
    mus = np.stack([p[0] for p in P])
    mu = mus.mean(0)
    return mu, np.mean([p[1] for p in P], 0) + mus.var(0), np.mean([p[2] for p in P], 0)


def loss(mu, var, ql, T, sc, wq):
    t = (T[TARGETS].values - sc["mean"]) / sc["std"]
    nll = 0.5 * (np.log(var) + (t - mu) ** 2 / var + math.log(2 * math.pi))
    q = T.is_quad.values
    bce = np.mean(np.logaddexp(0, ql) - q * ql)
    return {"loss": float(nll.mean() + wq * bce), "nll": dict(zip(TARGETS, nll.mean(0).round(5).tolist())),
            "quad_bce": float(bce)}


def by_type(mu, var, ql, T, sc):
    """Median absolute error per target and quad accuracy, per pipeline x double/quad."""
    m = mu * sc["std"] + sc["mean"]
    out = {}
    for p in ("D1", "D2"):
        for qn, qv in (("double", 0), ("quad", 1)):
            s = (T.pipeline.values == p) & (T.is_quad.values == qv)
            e = np.abs(m[s] - T[TARGETS].values[s])
            r = {c: float(np.median(e[:, j])) for j, c in enumerate(TARGETS)}
            r["src_offset"] = float(np.median(np.hypot(*(m[s][:, 1:3] - T[["src_x", "src_y"]].values[s]).T)))
            r["quad_acc"] = float(np.mean((ql[s] > 0) == qv))
            r["n"] = int(s.sum())
            out[f"{p}_{qn}"] = r
    return out


def val_reference(runs, sc, wq):
    """Validation loss of the same members / ensemble from the stored val_pred.npz (for the side-by-side record)."""
    T = pd.read_parquet(os.path.join(RW, "results", "targets_trainval.parquet")).set_index("row")
    P = []
    for r in runs:
        z = np.load(os.path.join(RW, "experiments", r, "val_pred.npz"), allow_pickle=True)
        P.append(((z["mean"] - sc["mean"]) / sc["std"], (z["sigma"] / sc["std"]) ** 2, z["quad_logit"]))
        rows = z["rows"]
    Tv = T.loc[rows].reset_index()
    return {"members": {r: json.load(open(os.path.join(RW, "experiments", r, "metrics.json")))["val_loss_tta8"] for r in runs},
            "ensemble": loss(*mixture(P), Tv, sc, wq)["loss"]}


def dry_run_val(a, dev):
    from lagn.data import load_trainval
    X, y, keys = load_trainval()
    T = lens_targets(keys)
    Tp = pd.read_parquet(os.path.join(RW, "results", "targets_trainval.parquet"))
    assert (T.row.values == Tp.row.values).all() and np.allclose(T[TARGETS].values, Tp[TARGETS].values) \
        and (T.is_quad.values == Tp.is_quad.values).all(), "targets.build() differs from targets_trainval.parquet"
    T = T[keys.split.values[T.row.values] == "val"].reset_index(drop=True)
    P, sc, wq = run_preds(a.runs, X, T.row.values, a.tta, dev)
    for r in a.runs:
        ref = json.load(open(os.path.join(RW, "experiments", r, "metrics.json")))[f"val_loss_tta{a.tta}"]
        got = loss(*P[r], T, sc, wq)["loss"]
        vp = np.load(os.path.join(RW, "experiments", r, "val_pred.npz"), allow_pickle=True)
        dm = np.abs(P[r][0] * sc["std"] + sc["mean"] - vp["mean"]).max(0)
        print(f"DRYRUN {r}: val loss {got:.5f} vs metrics.json {ref:.5f} (diff {got - ref:+.5f}); max |dmean| per target "
              f"{dict(zip(TARGETS, dm.round(5)))}", flush=True)
    ens = mixture([P[r] for r in a.runs])
    print("DRYRUN ensemble", json.dumps(loss(*ens, T, sc, wq)), "| val_reference", json.dumps(val_reference(a.runs, sc, wq)))
    print("DRYRUN by_type", json.dumps(by_type(*ens, T, sc)))
    print("DRYRUN report D2 theta_e", json.dumps(report(*ens, T, sc, T.pipeline.values)["D2"]["theta_e"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--dry_run_val", action="store_true")
    ap.add_argument("--user_approved_over_budget", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    if a.dry_run_val:
        return dry_run_val(a, dev)
    if not a.name:
        sys.exit("--name is required")
    check_isolation()
    notfz = [r for r in a.runs if r not in frozen_runs()]
    if notfz:
        sys.exit(f"REFUSED: not frozen in regression/EXPERIMENTS.md: {notfz}")
    n, names = n_evals()
    if a.name in names:
        sys.exit(f"REFUSED: eval name {a.name} already used (one evaluation per frozen model/ensemble)")
    new = [r for r in a.runs if not os.path.exists(os.path.join(PRED, r, f"holdout_pred_tta{a.tta}.npz"))]
    counted = len(new) + (1 if len(a.runs) > 1 else 0)
    if n + counted > REG_BUDGET and not a.user_approved_over_budget:
        sys.exit(f"REFUSED: {n} regression holdout evaluations already, this call adds {counted} (budget {REG_BUDGET}); override with --user_approved_over_budget")
    with open(LOG, "a") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | REG_EVAL | name={a.name} | runs={','.join(a.runs)} | "
                f"new={','.join(new) or '-'} | tta={a.tta} | counted={counted} | total={n + counted} | "
                f"job={os.environ.get('SLURM_JOB_ID')}\n")

    X = np.load(os.path.join(HO, "X_holdout.npy"), mmap_mode="r")
    keys = pd.read_parquet(os.path.join(HO, "keys_holdout.parquet"))
    T = lens_targets(keys)
    os.makedirs(PRED, exist_ok=True)
    tf = os.path.join(PRED, "holdout_targets.parquet")
    if not os.path.exists(tf):
        T.to_parquet(tf, index=False)
    P, sc, wq = run_preds(a.runs, X, T.row.values, a.tta, dev, cache=PRED)
    mu, var, ql = mixture([P[r] for r in a.runs])
    pipe = T.pipeline.values
    res = {"name": a.name, "runs": a.runs, "tta": a.tta, "n_lenses": len(T),
           "n_per_pipeline": T.pipeline.value_counts().to_dict(), "reg_eval_total_after": n + counted,
           "holdout": {**loss(mu, var, ql, T, sc, wq),
                       **{p: loss(mu[pipe == p], var[pipe == p], ql[pipe == p], T[pipe == p], sc, wq)["loss"]
                          for p in ("D1", "D2")}},
           "holdout_report": report(mu, var, ql, T, sc, pipe), "holdout_by_type": by_type(mu, var, ql, T, sc),
           "holdout_per_run": {r: loss(*P[r], T, sc, wq)["loss"] for r in a.runs},
           "val_reference": val_reference(a.runs, sc, wq)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{a.name}.json"), "w"), indent=1)
    if len(a.runs) > 1:
        os.makedirs(os.path.join(PRED, f"ENS_{a.name}"), exist_ok=True)
    d = os.path.join(PRED, f"ENS_{a.name}" if len(a.runs) > 1 else a.runs[0])
    np.savez(os.path.join(d, "holdout_pred.npz"), rows=T.row.values, obj=T.obj.values, mean=mu * sc["std"] + sc["mean"],
             sigma=np.sqrt(var) * sc["std"], quad_logit=ql, targets=np.array(TARGETS))
    r = res["holdout_report"]
    print(json.dumps({"holdout_loss": res["holdout"]["loss"], "D1": res["holdout"]["D1"], "D2": res["holdout"]["D2"],
                      "val_reference": res["val_reference"],
                      **{f"{p}_{c}_medae": round(r[p][c]["medae"], 4) for p in ("D1", "D2") for c in TARGETS},
                      **{f"{p}_src_offset": round(r[p]["src_pos"]["median_offset"], 4) for p in ("D1", "D2")},
                      **{f"{p}_quad_acc": round(r[p]["is_quad"]["acc"], 4) for p in ("D1", "D2")}}, indent=1))


if __name__ == "__main__":
    main()
