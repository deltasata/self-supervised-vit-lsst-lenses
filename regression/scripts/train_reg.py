"""Lens-parameter regression trainer (1 GPU), lenses only.

Targets (regression/scripts/targets.py): theta_e, src_x, src_y, z_lens, z_src -> heteroscedastic Gaussian NLL (mean and
log-variance per target); is_quad -> BCE. Outputs: 5 means + 5 log-variances + 1 quad logit, from the same pretrained
encoders as the classifier (src/lagn build_model + a new head).

Dihedral augmentation is equivariant: the source position is rotated/flipped together with the image (the other
targets are invariant). 8x dihedral TTA maps each view's prediction back to the original frame; the TTA mean is the mean
of means and the variance that of the 8-component mixture.

Usage: python regression/scripts/train_reg.py --cfg regression/configs/<x>.json [--set key=value ...] [--run_id ID]
  - selection / early stopping on the validation loss (mean NLL of the 5 standardised targets + w_quad * BCE, TTA8)
  - checkpoint each epoch, automatic resume, clean exit before the time limit
  - final: best weights, TTA8 on val -> metrics.json (physical units, per pipeline), val_pred.npz
Train/val = the classification split, lens rows only. Never reads the holdout.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

RW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(RW)
sys.path.insert(0, os.path.join(W, "src"))
sys.path.insert(0, os.path.join(W, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lagn.data import dihedral, load_trainval, norm_stats, random_correlated_noise, random_shift  # noqa: E402
from lagn.models import build_model  # noqa: E402
from train import deep_update, ema_multi_avg_fn, git_hash, set_key  # noqa: E402
from targets import BINARY, TARGETS  # noqa: E402

DEFAULT = {
    "model": {},
    "train": {"epochs": 100, "bs": 512, "lr": 1.5e-4, "wd": 0.05, "warmup_epochs": 3, "ema": 0.999, "patience": 20,
              "grad_clip": 2.0, "llrd": 0.75, "shift": 1, "noise_aug": 0.5, "w_quad": 1.0, "sel_tta": 8,
              "lv_max": 7.0, "channels_last": True, "pipelines": None,
              "zmix": 0},                                       # zmix K > 0: z_src as a K-component Gaussian mixture
    "seed": 0,
    "time_limit_min": 1420,
}
POS = [TARGETS.index("src_x"), TARGETS.index("src_y")]
NT = len(TARGETS)
Z = TARGETS.index("z_src")


def dihedral_mats():
    """M[k] (2x2) maps (x=col-20, y=row-20) of a point to its position after lagn.data.dihedral(., k)."""
    M = np.zeros((8, 2, 2))
    for k in range(8):
        for j, (dx, dy) in enumerate([(1, 0), (0, 1)]):
            img = torch.zeros(41, 41)
            img[20 + dy, 20 + dx] = 1
            r, c = np.unravel_index(int(dihedral(img, k).argmax()), (41, 41))
            M[k, :, j] = [c - 20, r - 20]
    return torch.tensor(M, dtype=torch.float32)


def llrd_groups(model, cfg):
    tc, arch = cfg["train"], cfg["model"]["arch"]
    bb = model.net.backbone
    if "convnext" in arch:
        n2 = len(bb.stages[2].blocks)
        s3 = 3 + (n2 - 1) // 3 + 1
        top = s3 + 1

        def lid(n):
            if n.startswith("stem"):
                return 0
            if n.startswith("stages."):
                p = n.split(".")
                st = int(p[1])
                if st < 2:
                    return st + 1
                if st == 3:
                    return s3
                return 3 + int(p[3]) // 3 if p[2] == "blocks" else 3
            return top
    else:                                                   # plain ViT
        top = len(bb.blocks) + 1

        def lid(n):
            if n.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed")):
                return 0
            if n.startswith("blocks."):
                return int(n.split(".")[1]) + 1
            return top
    by, head = {}, []
    for n, p in model.net.named_parameters():                # Net: .norm (no params), .backbone (timm)
        if n.startswith("backbone.") and not n.startswith(("backbone.head", "backbone.fc")):
            by.setdefault(lid(n[len("backbone."):]), []).append(p)
        else:
            head.append(p)
    return [{"params": ps, "lr": tc["lr"] * tc["llrd"] ** (top - i)} for i, ps in sorted(by.items())] + \
        [{"params": head, "lr": tc["lr"]}]


class RegNet(torch.nn.Module):
    """build_model(cfg) encoder with a (2*NT + 1)-output head: means, log-variances, quad logit."""

    def __init__(self, cfg):
        super().__init__()
        net = build_model(cfg)
        self.K = cfg["train"].get("zmix", 0)
        net.backbone.reset_classifier(2 * NT + len(BINARY) + 3 * self.K)
        self.net, self.lv_max = net, cfg["train"]["lv_max"]

    def forward(self, x, return_mix=False):
        """mu, log-variance (soft-clamped), quad logit (n, 1); with return_mix also the z_src mixture
        (means, log-variances, weight logits; each (n, K)) or None. With a mixture, the single-Gaussian z_src
        outputs are unused."""
        o = self.net(x).float()
        lv = self.lv_max * torch.tanh(o[:, NT:2 * NT] / self.lv_max)          # soft clamp of the log-variance
        nq = 2 * NT + len(BINARY)
        if not return_mix:
            return o[:, :NT], lv, o[:, 2 * NT:nq]
        mix = None
        if self.K:
            m = o[:, nq:]
            mix = (m[:, :self.K], self.lv_max * torch.tanh(m[:, self.K:2 * self.K] / self.lv_max), m[:, 2 * self.K:])
        return o[:, :NT], lv, o[:, 2 * NT:nq], mix


def mix_nll(t, m, lv, lw):
    """-log sum_k w_k N(t | m_k, exp(lv_k)); lw = normalised log-weights; t (n,), the rest (n, K)."""
    logn = -0.5 * (lv + (t[:, None] - m) ** 2 * torch.exp(-lv) + math.log(2 * math.pi))
    return -torch.logsumexp(lw + logn, 1)


def to_view(t, M):
    """Transform standardised targets t (n, NT) into the frame of per-sample dihedral matrices M (n, 2, 2)."""
    t = t.clone()
    t[:, POS] = torch.einsum("nij,nj->ni", M, t[:, POS])
    return t


@torch.no_grad()
def predict(model, X, Md, tta=8, bs=1024, return_mix=False):
    """TTA prediction in the original frame: mean (n, NT), variance (n, NT), quad logit (n,).
    Mixture models (train.zmix): the z_src mean/variance are the moments of the mixture over all components of all
    views; return_mix=True also returns that mixture (means, variances, normalised log-weights; (n, tta*K)) or None."""
    model.eval()
    MU, VAR, QL, MIX = [], [], [], []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        mus, vs, qs, cm, cv, cw = [], [], [], [], [], []
        for k in range(tta):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, lv, ql, mix = model(dihedral(xb, k), return_mix=True)
            if mix is not None:                                      # z_src is dihedral-invariant
                cm.append(mix[0].float()), cv.append(mix[1].float().exp())
                cw.append(torch.log_softmax(mix[2].float(), 1) - math.log(tta))
            mu, v = mu.float(), lv.float().exp()
            Mi = Md[k].T                                             # inverse of an orthogonal signed permutation
            mu[:, POS] = mu[:, POS] @ Mi.T
            v[:, POS] = v[:, POS] @ (Mi.T ** 2)
            mus.append(mu), vs.append(v), qs.append(ql.float().squeeze(1))
        mus, vs = torch.stack(mus), torch.stack(vs)
        m = mus.mean(0)
        v = vs.mean(0) + ((mus - m) ** 2).mean(0)
        if cm:
            cm, cv, cw = torch.cat(cm, 1), torch.cat(cv, 1), torch.cat(cw, 1)
            w = cw.exp()
            m[:, Z] = (w * cm).sum(1)
            v[:, Z] = (w * (cv + cm ** 2)).sum(1) - m[:, Z] ** 2
            MIX.append((cm, cv, cw))
        MU.append(m), VAR.append(v), QL.append(torch.stack(qs).mean(0))
    out = torch.cat(MU), torch.cat(VAR), torch.cat(QL)
    if return_mix:
        mix = tuple(torch.cat([b[j] for b in MIX]) for j in range(3)) if MIX else None
        return (*out, mix)
    return out


def losses(mu, var, ql, t, q, mix=None):
    nll = 0.5 * (torch.log(var) + (t - mu) ** 2 / var + math.log(2 * math.pi))
    if mix is not None:
        nll[:, Z] = mix_nll(t[:, Z], mix[0], torch.log(mix[1]), mix[2])
    bce = F.binary_cross_entropy_with_logits(ql, q)
    return nll.mean(0), bce


def report(mu, var, ql, T, sc, pipe):
    """Metrics in physical units, overall and per pipeline."""
    from sklearn.metrics import roc_auc_score
    m = mu * sc["std"] + sc["mean"]
    s = np.sqrt(var) * sc["std"]
    t = T[TARGETS].values
    R = {}
    for name, sel in [("all", np.ones(len(T), bool)), ("D1", pipe == "D1"), ("D2", pipe == "D2")]:
        if not sel.any():                                     # specialist run without this pipeline
            continue
        r = {}
        for j, c in enumerate(TARGETS):
            e, z = m[sel, j] - t[sel, j], (m[sel, j] - t[sel, j]) / s[sel, j]
            r[c] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                    "medae": float(np.median(np.abs(e))), "bias": float(np.mean(e)),
                    "r2": float(1 - np.mean(e ** 2) / np.var(t[sel, j])),
                    "cover68": float(np.mean(np.abs(z) < 1)), "cover95": float(np.mean(np.abs(z) < 1.96)),
                    "nll_std": float(np.mean(0.5 * (np.log(var[sel, j]) + (mu[sel, j] - (t[sel, j] - sc["mean"][j]) / sc["std"][j]) ** 2 / var[sel, j]
                                                    + np.log(2 * np.pi)))),
                    "median_sigma": float(np.median(s[sel, j]))}
        th = t[sel, TARGETS.index("theta_e")]
        r["theta_e"]["median_frac_err"] = float(np.median(np.abs(m[sel, 0] - th) / th))
        dp = np.hypot(m[sel, POS[0]] - t[sel, POS[0]], m[sel, POS[1]] - t[sel, POS[1]])
        r["src_pos"] = {"median_offset": float(np.median(dp)), "p90_offset": float(np.percentile(dp, 90))}
        qt, qp = T["is_quad"].values[sel], 1 / (1 + np.exp(-ql[sel]))
        r["is_quad"] = {"auc": float(roc_auc_score(qt, qp)), "acc": float(np.mean((qp > 0.5) == qt)),
                        "bce": float(np.mean(-(qt * np.log(qp + 1e-12) + (1 - qt) * np.log(1 - qp + 1e-12))))}
        R[name] = r
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run_id")
    a = ap.parse_args()
    cfg = deep_update(copy.deepcopy(DEFAULT), json.load(open(a.cfg)))
    for s in a.set:
        k, v = s.split("=", 1)
        set_key(cfg, k, v)
    run_id = a.run_id or os.path.splitext(os.path.basename(a.cfg))[0]
    out = os.path.join(RW, "experiments", run_id)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = "cuda"

    # ---------------- data: lens rows of train / val
    X, y, keys = load_trainval()
    T = pd.read_parquet(os.path.join(RW, "results", "targets_trainval.parquet"))
    assert (keys.obj.values[T.row.values] == T.obj.values).all() and (y[T.row.values] == 1).all()
    if tc["pipelines"]:                                       # specialist runs: train and validate on these pipelines only
        T = T[T.pipeline.isin(tc["pipelines"])]
    Ttr, Tva = T[T.split == "train"].reset_index(drop=True), T[T.split == "val"].reset_index(drop=True)
    mean = Ttr[TARGETS].mean().to_numpy(dtype=np.float64, copy=True)
    std = Ttr[TARGETS].std().to_numpy(dtype=np.float64, copy=True)
    mean[POS] = 0.0                                           # positions: symmetric, one shared scale for x and y
    std[POS] = np.sqrt(np.mean(Ttr[["src_x", "src_y"]].values ** 2))
    sc = {"mean": mean, "std": std}
    json.dump({"targets": TARGETS, "mean": mean.tolist(), "std": std.tolist()}, open(os.path.join(out, "scaler.json"), "w"))
    std_t = lambda d: torch.tensor((d[TARGETS].values - mean) / std, dtype=torch.float32, device=dev)
    Xtr = torch.from_numpy(np.ascontiguousarray(X[Ttr.row.values])).to(dev)
    Xva = torch.from_numpy(np.ascontiguousarray(X[Tva.row.values])).to(dev)
    ttr, tva = std_t(Ttr), std_t(Tva)
    qtr = torch.tensor(Ttr.is_quad.values, dtype=torch.float32, device=dev)
    qva = torch.tensor(Tva.is_quad.values, dtype=torch.float32, device=dev)
    np.save(os.path.join(out, "val_rows.npy"), Tva.row.values)
    print(f"train {len(Ttr)}  val {len(Tva)}  scaler {sc}  load {time.time() - t_start:.0f}s", flush=True)
    Md = dihedral_mats().to(dev)
    sigma_b = norm_stats()["sigma"]

    # ---------------- model
    model = RegNet(cfg).to(dev)
    if tc["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_multi_avg_fn(tc["ema"]), use_buffers=True) \
        if tc["ema"] else None
    groups = llrd_groups(model, cfg)
    print(f"params {nparam / 1e6:.1f}M  lr groups {[len(g['params']) for g in groups]}", flush=True)
    opt = torch.optim.AdamW(groups, lr=tc["lr"], weight_decay=tc["wd"])
    spe = len(Ttr) // tc["bs"]
    total, warm = tc["epochs"] * spe, tc["warmup_epochs"] * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))

    start_ep, hist, best, bad = 0, [], float("inf"), 0
    lastp = os.path.join(out, "last.pt")
    if os.path.exists(lastp):
        ck = torch.load(lastp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        start_ep, hist, best, bad = ck["epoch"] + 1, ck["hist"], ck["best"], ck["bad"]
        torch.set_rng_state(ck["rng_cpu"].cpu()), torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(f"resumed from epoch {start_ep}", flush=True)
    else:
        json.dump({**cfg, "run_id": run_id, "git": git_hash(), "n_train": len(Ttr), "n_val": len(Tva),
                   "params_M": nparam / 1e6, "slurm_job": os.environ.get("SLURM_JOB_ID")},
                  open(os.path.join(out, "config.json"), "w"), indent=1)

    def val_loss(m_):
        mu, var, ql, mix = predict(m_, Xva, Md, tta=tc["sel_tta"], return_mix=True)
        nll, bce = losses(mu, var, ql, tva, qva, mix)
        return float(nll.mean() + tc["w_quad"] * bce), nll.tolist(), float(bce)

    # ---------------- train
    for ep in range(start_ep, tc["epochs"]):
        if bad >= tc["patience"]:
            break
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Ttr), device=dev)
        tl, nb = 0.0, 0
        for b in range(spe):
            idx = perm[b * tc["bs"]:(b + 1) * tc["bs"]]
            xb, tb, qb = Xtr[idx], ttr[idx], qtr[idx]
            k = torch.randint(0, 8, (len(idx),), device=dev)
            xo = torch.empty_like(xb)
            for g in range(8):
                sel = k == g
                if sel.any():
                    xo[sel] = dihedral(xb[sel], g)
            tb = to_view(tb, Md[k])
            xb = random_shift(xo, tc["shift"])
            if tc["noise_aug"]:
                xb = random_correlated_noise(xb, sigma_b, p=tc["noise_aug"])
            if tc["channels_last"]:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu, lv, ql, mix = model(xb, return_mix=True)
            mu, lv = mu.float(), lv.float()
            nll = 0.5 * (lv + (tb - mu) ** 2 * torch.exp(-lv))
            if mix is not None:
                nll = torch.cat([nll[:, :Z], (mix_nll(tb[:, Z], mix[0].float(), mix[1].float(),
                                                      torch.log_softmax(mix[2].float(), 1))
                                              - 0.5 * math.log(2 * math.pi))[:, None], nll[:, Z + 1:]], 1)
            loss = nll.mean() + tc["w_quad"] * F.binary_cross_entropy_with_logits(ql.float().squeeze(1), qb)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            if b % 20 == 0:
                tl, nb = tl + loss.item(), nb + 1
        evm = ema.module if ema is not None else model
        vl, vnll, vbce = val_loss(evm)
        vl_raw = val_loss(model)[0] if ema is not None else vl
        use_raw = vl_raw < vl
        vsel = min(vl, vl_raw)
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "val_loss_raw": vl_raw, "val_loss_sel": vsel,
               "sel": "raw" if use_raw else "ema", "val_quad_bce": vbce,
               **{f"val_nll_{c}": v for c, v in zip(TARGETS, vnll)}, "lr": sched.get_last_lr()[-1], "sec": time.time() - t0}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if vsel < best - 1e-5:
            best, bad = vsel, 0
            torch.save((model if use_raw else evm).state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "epoch": ep, "hist": hist, "best": best,
                    "bad": bad, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, lastp + ".tmp")
        os.replace(lastp + ".tmp", lastp)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        el = (time.time() - t_start) / 60
        if ep + 1 < tc["epochs"] and bad < tc["patience"] and el + 2 * np.mean([h["sec"] for h in hist[-3:]]) / 60 + 5 > cfg["time_limit_min"]:
            print(f"time limit approaching ({el:.0f} min); exiting for resubmission", flush=True)
            open(os.path.join(out, "NEEDS_RESUME"), "w").write(str(ep))
            return
    if os.path.exists(os.path.join(out, "NEEDS_RESUME")):
        os.remove(os.path.join(out, "NEEDS_RESUME"))

    # ---------------- final: best weights, TTA1 and TTA8 on val
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
    res = {"run_id": run_id, "best_epoch": int(np.argmin([h["val_loss_sel"] for h in hist])), "epochs_run": len(hist)}
    for tta in (1, 8):
        mu, var, ql, mix = predict(model, Xva, Md, tta=tta, return_mix=True)
        nll, bce = losses(mu, var, ql, tva, qva, mix)
        res[f"val_loss_tta{tta}"] = float(nll.mean() + tc["w_quad"] * bce)
        res[f"val_nll_tta{tta}"] = dict(zip(TARGETS, nll.tolist()))      # exact (mixture) NLL per target
        mu, var, ql = mu.cpu().numpy(), var.cpu().numpy(), ql.cpu().numpy()
        res[f"val_tta{tta}"] = report(mu, var, ql, Tva, sc, Tva.pipeline.values)
    m = mu * std + mean
    np.savez(os.path.join(out, "val_pred.npz"), rows=Tva.row.values, obj=Tva.obj.values, mean=m,
             sigma=np.sqrt(var) * std, quad_logit=ql, targets=np.array(TARGETS),
             **({"zmix_mean": mix[0].cpu().numpy() * std[Z] + mean[Z], "zmix_sigma": mix[1].sqrt().cpu().numpy() * std[Z],
                 "zmix_logw": mix[2].cpu().numpy()} if mix is not None else {}))
    res["gpu_hours"] = (time.time() - t_start) / 3600
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    r = res["val_tta8"]
    print("FINAL", json.dumps({"val_loss_tta8": res["val_loss_tta8"],
                               **{f"{p}_{c}_rmse": round(r[p][c]["rmse"], 4) for p in ("D1", "D2") for c in TARGETS},
                               **{f"{p}_quad_auc": round(r[p]["is_quad"]["auc"], 4) for p in ("D1", "D2")}}), flush=True)
    os.remove(lastp)


if __name__ == "__main__":
    main()
