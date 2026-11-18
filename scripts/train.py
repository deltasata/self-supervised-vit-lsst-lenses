"""Generic supervised trainer (1 GPU). Whole train/val array lives on the GPU; augmentation on GPU.

Usage: python scripts/train.py --cfg configs/<x>.json [--set key.sub=value ...] [--run_id ID]
  - checkpoint every epoch to experiments/<run_id>/last.pt and resumes from it automatically
  - selection / early stopping on VALIDATION LOSS (BCE); best weights -> best.pt (EMA if enabled)
  - stops cleanly before the Slurm time limit (cfg.time_limit_min) so a resubmission resumes
  - final: best weights, 8x dihedral TTA on val -> metrics.json, val_pred.npy
Never reads the holdout.
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import (W, load_trainval, norm_stats, random_correlated_noise, random_psf_blur, random_sky_offset, random_dihedral, random_dimming,  # noqa: E402
                       kfold_assign, mixup_flux, random_rotation, random_shift, random_subpixel_shift, tta_predict)
from lagn.metrics import all_metrics  # noqa: E402
from lagn.models import build_model  # noqa: E402

DEFAULT = {
    "model": {"arch": "resnet18", "stem": "native", "pretrained": False, "norm": "asinh", "norm_a": 3.0},
    "train": {"epochs": 40, "bs": 512, "lr": 1e-3, "wd": 0.05, "warmup_epochs": 1, "opt": "adamw",
              "label_smoothing": 0.0, "shift": 1, "subpixel": 0.0, "noise_aug": 0.0, "ema": 0.999, "patience": 8, "grad_clip": 2.0,
              "train_pipelines": ["D1", "D2"], "max_train": None, "channels_last": True, "compile": False,
              "lr_backbone_mult": 1.0,
              # pipe_weight: training-loss weight of SIMCT (D2) samples relative to SLSim (D1), weights renormalised
              # to mean 1 over the training set. Metadata is used only as a loss weight, never as a model input.
              "pipe_weight": 1.0,
              # plateau policy: "none" = single cosine + patience;
              # "restart" = on plateau roll back to the best weights, re-warm to peak*U_log(range) and anneal again
              # over restart_epochs; each cycle's best weights = snapshot; optional SAM from cycle sam_from_cycle
              "plateau_policy": "none", "restart_patience": 5, "restart_epochs": 12,
              "restart_lr_range": [0.25, 2.0], "max_restarts": 4, "sam_rho": 0.0, "sam_from_cycle": 0,
              # sel_half="A": choose best/snapshot weights on a fixed half A of val only; half B stays unbiased
              "sel_half": None,
              # sel_tta=8: per-epoch val (selection, early stopping, restarts) uses the same dihedral TTA as the final
              # metric; long restart runs otherwise pick checkpoints that are optimistic for the single view only
              #. History columns val_loss / val_loss_raw are then TTA losses.
              "sel_tta": 1,
              # llrd>0: layer-wise lr decay (timm ViT / ConvNeXt / ResNet), lr *= llrd**(top-layer_id)
              "llrd": 0.0,
              # knowledge distillation: kd_path = .npy of teacher logits for ALL trainval rows (same order as
              # load_trainval); target = (1-kd_alpha)*y + kd_alpha*sigmoid(teacher/kd_T)
              "kd_path": None, "kd_alpha": 0.0, "kd_T": 1.0,
              # dimming augmentation (data.random_dimming): fraction dim_aug of samples made fainter, s in [dim_min, 1]
              "dim_aug": 0.0, "dim_min": 0.3,
              # mixup in raw flux (a superposition of two stamps) on a fraction `mixup` of samples, lam ~ Beta(a, a);
              # noise re-added so the mixed stamp keeps the per-band sky noise level; labels mixed with the same lam
              "mixup": 0.0, "mixup_alpha": 0.4,
              # arbitrary-angle rotation (data.random_rotation, Fourier three-shear: flux- and noise-preserving) on a
              # fraction rot_aug of samples, angle in [-rot_max, rot_max] deg on top of the dihedral group
              "rot_aug": 0.0, "rot_max": 45.0,
              # PSF-width and sky-level augmentation (data.random_psf_blur / random_sky_offset)
              "psf_aug": 0.0, "psf_fmax": 1.25, "sky_aug": 0.0, "sky_amax": 0.5,
              # k-fold over ALL train+val rows: fold=f trains on the other n_folds-1 folds and uses
              # fold f for early stopping / selection; folds from lagn.data.kfold_assign (stratified simset x ID block)
              "fold": None, "n_folds": 10},
    "seed": 0,
    "time_limit_min": 1420,
    "track": "cnn",
}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def set_key(d, key, val):
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        pass
    d[ks[-1]] = val


def git_hash():
    try:
        return subprocess.run(["git", "-C", W, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return "?"


@torch.no_grad()
def evaluate(model, X, y, bs=2048, tta=1):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = tta_predict(model, xb, tta) if tta > 1 else model(xb).float()
        out.append(lo.float().squeeze(1))
    logits = torch.cat(out)
    if not torch.isfinite(logits).all():          # diverged weights: report inf instead of crashing
        return float("inf"), torch.nan_to_num(logits, 0.0, 0.0, 0.0).cpu().numpy()
    loss = F.binary_cross_entropy_with_logits(logits, y.float()).item()
    return loss, logits.cpu().numpy()


def ema_multi_avg_fn(decay, warmup=True):
    """EMA with timm-style warm-up: effective decay min(decay, (1+n)/(10+n)) so the average does not lag the
    fast-moving early weights (the plain 0.999 EMA left BN statistics and weights mismatched, LOG 21:30)."""
    @torch.no_grad()
    def fn(ema_list, cur_list, num_averaged):
        n = float(num_averaged)
        d = min(decay, (1 + n) / (10 + n)) if warmup else decay
        fl = [(e, c) for e, c in zip(ema_list, cur_list) if e.is_floating_point()]
        if fl:
            torch._foreach_lerp_([e for e, _ in fl], [c for _, c in fl], 1 - d)
        for e, c in zip(ema_list, cur_list):      # integer buffers (BN num_batches_tracked): copy
            if not e.is_floating_point():
                e.copy_(c)
    return fn


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
    run_id = a.run_id or cfg.get("run_id") or os.path.splitext(os.path.basename(a.cfg))[0]
    out = os.path.join(W, "experiments", run_id)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    dev = "cuda"

    # ---------------- data (to GPU)
    X, y, keys = load_trainval()
    tr = (keys.split == "train").values & keys.pipeline.isin(tc["train_pipelines"]).values
    va = (keys.split == "val").values
    if tc["fold"] is not None:
        fa = kfold_assign(keys, tc["n_folds"])
        va = fa == tc["fold"]
        tr = ~va & keys.pipeline.isin(tc["train_pipelines"]).values
    tr_idx = np.where(tr)[0]
    if tc.get("max_train"):
        tr_idx = np.random.RandomState(cfg["seed"]).choice(tr_idx, tc["max_train"], replace=False)
        tr_idx.sort()
    va_idx = np.where(va)[0]
    Xtr = torch.from_numpy(np.ascontiguousarray(X[tr_idx])).to(dev)
    ytr = torch.from_numpy(y[tr_idx].astype(np.int64)).to(dev)
    wtr = None
    if tc["pipe_weight"] != 1.0:
        w_ = np.where(keys.pipeline.values[tr_idx] == "D2", tc["pipe_weight"], 1.0).astype(np.float32)
        wtr = torch.from_numpy(w_ / w_.mean()).to(dev)
    if tc["kd_path"]:
        t_lo = np.load(os.path.join(W, tc["kd_path"])).astype(np.float32)
        assert len(t_lo) == len(X), "kd_path must hold logits for all trainval rows"
        ktr = torch.sigmoid(torch.from_numpy(t_lo[tr_idx]) / tc["kd_T"]).to(dev)
    Xva = torch.from_numpy(np.ascontiguousarray(X[va_idx])).to(dev)
    yva = torch.from_numpy(y[va_idx].astype(np.int64)).to(dev)
    meta_va = keys.iloc[va_idx].reset_index(drop=True)
    perm_v = np.random.RandomState(20261008).permutation(len(va_idx))      # fixed val halves (seed-independent)
    vA, vB = np.sort(perm_v[:len(perm_v) // 2]), np.sort(perm_v[len(perm_v) // 2:])
    yva_np = y[va_idx].astype(np.float64)

    def bce_np(z, idx):
        z, t = z[idx].astype(np.float64), yva_np[idx]
        return float(np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))))
    np.save(os.path.join(out, "val_rows.npy"), va_idx)                    # trainval row index of each val logit
    print(f"train {len(tr_idx)}  val {len(va_idx)}  load {time.time() - t_start:.0f}s", flush=True)

    sigma_b = norm_stats()["sigma"]
    # ---------------- model
    model = build_model(cfg).to(dev)
    if tc["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    if hasattr(model, "set_train_rows"):        # multi-task heads (scripts/train_aux.py); no-op for other models
        model.set_train_rows(tr_idx, keys)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params {nparam / 1e6:.2f}M", flush=True)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=ema_multi_avg_fn(tc["ema"]), use_buffers=True) if tc["ema"] else None
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.") and "head" not in n
          and "fc." not in n and "classifier" not in n]
    other = [p for n, p in model.named_parameters() if not any(p is q for q in bb)]
    groups = [{"params": bb, "lr": tc["lr"] * tc["lr_backbone_mult"]}, {"params": other, "lr": tc["lr"]}]
    if tc["llrd"]:
        # layer ids (0 = input side). ViT: patch/pos/cls 0, blocks.i -> i+1, rest top.
        # ConvNeXt (official-style): stem 0, stage0 1, stage1 2, stage2 blocks grouped by 3 from 3, stage3 next, rest top.
        # ResNet: conv1/bn1 0, layer1..4 -> 1..4, rest top. lr *= llrd**(top - id).
        bbm = model.backbone
        arch_name = cfg["model"]["arch"]
        if "convnext" in arch_name and hasattr(bbm, "stem") and hasattr(bbm, "stages"):
            n2 = len(bbm.stages[2].blocks)
            s3 = 3 + (n2 - 1) // 3 + 1
            top = s3 + 1
            def lid(n):
                n = n[len("backbone."):]
                if n.startswith("stem"):
                    return 0
                if n.startswith("stages."):
                    st = int(n.split(".")[1])
                    if st < 2:
                        return st + 1
                    if st == 3:
                        return s3
                    parts = n.split(".")
                    return 3 + int(parts[3]) // 3 if parts[2] == "blocks" else 3
                return top
        elif "resnet" in arch_name and hasattr(bbm, "layer1"):
            top = 5
            def lid(n):
                n = n[len("backbone."):]
                if n.startswith(("conv1", "bn1")):
                    return 0
                if n.startswith("layer"):
                    return int(n[5])
                return top
        elif arch_name == "dual":
            # DualNet: each branch gets its own layer ids (ViT blocks / ConvNeXt stages as above); the group key is
            # minus the depth below that branch's top, so lr = lr * llrd**depth in both branches (top = 0)
            vb, cb = bbm["vit"].backbone, bbm["cnn"].backbone
            nvb = len(vb.blocks)
            n2 = len(cb.stages[2].blocks)
            s3 = 3 + (n2 - 1) // 3 + 1
            top = 0
            def lid(n):
                parts = n.split(".")                        # backbone.<vit|cnn>.backbone.<rest>
                br, r = parts[1], ".".join(parts[3:])
                if br == "vit":
                    if r.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed")):
                        i = 0
                    elif r.startswith("blocks."):
                        i = int(r.split(".")[1]) + 1
                    else:
                        i = nvb + 1
                    return i - (nvb + 1)
                if r.startswith("stem"):
                    i = 0
                elif r.startswith("stages."):
                    st = int(r.split(".")[1])
                    rp = r.split(".")
                    i = st + 1 if st < 2 else (s3 if st == 3 else (3 + int(rp[3]) // 3 if rp[2] == "blocks" else 3))
                else:
                    i = s3 + 1
                return i - (s3 + 1)
        elif arch_name == "aion":
            top = len(bbm["blocks"]) + 1                    # AIONNet: token embedding 0, encoder block i -> i+1
            def lid(n):
                n = n[len("backbone."):]
                if n.startswith("emb."):
                    return 0
                if n.startswith("blocks."):
                    return int(n.split(".")[1]) + 1
                return top
        elif hasattr(bbm, "blocks") and hasattr(bbm, "patch_embed") and not hasattr(bbm, "stages") \
                and isinstance(bbm.blocks, torch.nn.Sequential | torch.nn.ModuleList):
            top = len(bbm.blocks) + 1                       # plain ViT family (ViT, DeiT, BEiT, EVA, XCiT, Hiera...)
            def lid(n):
                n = n[len("backbone."):]
                if n.startswith(("cls_token", "pos_embed", "reg_token", "patch_embed", "rel_pos", "absolute_pos")):
                    return 0
                if n.startswith("blocks."):
                    return int(n.split(".")[1]) + 1
                return top
        else:
            # generic (Swin, MaxViT, CoAtNet, MetaFormer, PVT, ...): parameters in registration order (input ->
            # output in timm) split into 12 buckets of equal parameter count
            top = 12
            names = [n for n, p_ in model.named_parameters() if n.startswith("backbone.") and any(p_ is q for q in bb)]
            sizes = np.array([dict(model.named_parameters())[n].numel() for n in names], dtype=np.float64)
            frac = (np.cumsum(sizes) - sizes / 2) / sizes.sum()
            gid = {n: min(top - 1, int(f * top)) for n, f in zip(names, frac)}
            def lid(n):
                return gid.get(n, top)
        by = {}
        for n, p_ in model.named_parameters():
            if n.startswith("backbone.") and any(p_ is q for q in bb):
                by.setdefault(lid(n), []).append(p_)
        groups = [{"params": ps, "lr": tc["lr"] * tc["lr_backbone_mult"] * tc["llrd"] ** (top - i)}
                  for i, ps in sorted(by.items())] + [{"params": other, "lr": tc["lr"]}]
        print("llrd groups", [(i, len(ps)) for i, ps in sorted(by.items())], flush=True)
    if tc["opt"] == "adamw":
        opt = torch.optim.AdamW(groups, lr=tc["lr"], weight_decay=tc["wd"])
    else:
        opt = torch.optim.SGD(groups, lr=tc["lr"], momentum=0.9, weight_decay=tc["wd"], nesterov=True)
    steps_per_epoch = len(tr_idx) // tc["bs"]
    total = tc["epochs"] * steps_per_epoch
    warm = tc["warmup_epochs"] * steps_per_epoch

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    fwd = torch.compile(model) if tc["compile"] else model

    def make_cycle_sched(mult, n_epochs):
        """restart cycle: 1 warm-up epoch to mult*base lr, then cosine to 0 over n_epochs."""
        cw, ctot = steps_per_epoch, n_epochs * steps_per_epoch
        def lam(s_):
            if s_ < cw:
                return mult * (s_ + 1) / cw
            return mult * 0.5 * (1 + math.cos(math.pi * min(1.0, (s_ - cw) / max(1, ctot - cw))))
        return torch.optim.lr_scheduler.LambdaLR(opt, lam)

    policy = tc["plateau_policy"]
    cyc = {"n": 0, "mult": 1.0, "epochs": tc["epochs"], "start_ep": 0, "best": float("inf")}
    params = [p_ for p_ in model.parameters() if p_.requires_grad]

    # ---------------- resume
    start_ep, hist, best, bad = 0, [], float("inf"), 0
    lastp = os.path.join(out, "last.pt")
    if os.path.exists(lastp):
        ck = torch.load(lastp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        start_ep, hist, best, bad = ck["epoch"] + 1, ck["hist"], ck["best"], ck["bad"]
        cyc = ck.get("cyc", cyc)
        if cyc["n"] > 0:
            sched = make_cycle_sched(cyc["mult"], cyc["epochs"])
            sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["rng_cpu"].cpu())
        torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        print(f"resumed from epoch {start_ep}", flush=True)
    else:
        json.dump({**cfg, "run_id": run_id, "git": git_hash(), "n_train": int(len(tr_idx)),
                   "n_val": int(len(va_idx)), "params_M": nparam / 1e6,
                   "slurm_job": os.environ.get("SLURM_JOB_ID")}, open(os.path.join(out, "config.json"), "w"), indent=1)

    # ---------------- train
    if policy == "none":
        done = start_ep >= tc["epochs"] or bad >= tc["patience"]
    else:
        done = ck.get("done", False) if os.path.exists(lastp) else False
    ep = start_ep
    while not done:
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(tr_idx), device=dev)
        tl, nb, nskip = 0.0, 0, 0
        for b in range(steps_per_epoch):
            idx = perm[b * tc["bs"]:(b + 1) * tc["bs"]]
            xb = random_dihedral(Xtr[idx])
            xb = random_shift(xb, tc["shift"])
            if tc["subpixel"]:
                xb = random_subpixel_shift(xb, tc["subpixel"])
            if tc["rot_aug"]:
                xb = random_rotation(xb, p=tc["rot_aug"], max_deg=tc["rot_max"])
            if tc["dim_aug"]:
                xb = random_dimming(xb, sigma_b, p=tc["dim_aug"], smin=tc["dim_min"])
            if tc["psf_aug"]:
                xb = random_psf_blur(xb, sigma_b, p=tc["psf_aug"], fmax=tc["psf_fmax"])
            if tc["sky_aug"]:
                xb = random_sky_offset(xb, sigma_b, p=tc["sky_aug"], amax=tc["sky_amax"])
            if tc["noise_aug"]:
                xb = random_correlated_noise(xb, sigma_b, p=tc["noise_aug"])
            if tc["channels_last"]:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = ytr[idx].float()
            if tc["kd_path"]:
                yb = (1 - tc["kd_alpha"]) * yb + tc["kd_alpha"] * ktr[idx]
            if tc["mixup"]:
                xb, yb = mixup_flux(xb, yb, sigma_b, p=tc["mixup"], alpha=tc["mixup_alpha"])
            if tc["label_smoothing"]:
                yb = yb * (1 - tc["label_smoothing"]) + 0.5 * tc["label_smoothing"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = fwd(xb).float().squeeze(1)
            loss = F.binary_cross_entropy_with_logits(lo, yb, weight=None if wtr is None else wtr[idx])
            aux = getattr(model, "aux_logits", None)          # DualNet: branch heads (deep supervision)
            if aux is not None and model.aux_w:
                loss = loss + model.aux_w * sum(F.binary_cross_entropy_with_logits(a.float().squeeze(1), yb)
                                                for a in aux) / len(aux)
            if hasattr(model, "aux_loss"):           # multi-task heads (scripts/train_aux.py); no-op otherwise
                loss = loss + model.aux_loss(idx)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                nskip += 1
                if nskip > 50:
                    raise RuntimeError("more than 50 non-finite training losses this epoch")
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"] or 1e9)
            if not torch.isfinite(gn):                 # NaN/inf gradients with a finite loss: skip the step
                nskip += 1
                opt.zero_grad(set_to_none=True)
                if nskip > 50:
                    raise RuntimeError("more than 50 non-finite gradient steps this epoch")
                continue
            if tc["sam_rho"] > 0 and cyc["n"] >= tc["sam_from_cycle"]:
                # SAM: ascend to w + rho*g/|g|, recompute the gradient there, step from w
                with torch.no_grad():
                    gnorm = torch.norm(torch.stack([p_.grad.norm() for p_ in params if p_.grad is not None]))
                    eps = []
                    for p_ in params:
                        e = None if p_.grad is None else p_.grad * (tc["sam_rho"] / (gnorm + 1e-12))
                        if e is not None:
                            p_.add_(e)
                        eps.append(e)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo2 = fwd(xb).float().squeeze(1)
                F.binary_cross_entropy_with_logits(lo2, yb).backward()
                with torch.no_grad():
                    for p_, e in zip(params, eps):
                        if e is not None:
                            p_.sub_(e)
                gn2 = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"] or 1e9)
                if not torch.isfinite(gn2):
                    nskip += 1
                    opt.zero_grad(set_to_none=True)
                    continue
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tl += loss.item() if b % 20 == 0 else 0
            nb += 1 if b % 20 == 0 else 0
        evm = ema.module if ema is not None else model
        vl, vlog = evaluate(evm, Xva, yva, tta=tc["sel_tta"])
        from sklearn.metrics import roc_auc_score
        vauc = roc_auc_score(yva.cpu().numpy(), vlog) if np.isfinite(vl) else 0.5
        vl_raw, vlog_raw = evaluate(model, Xva, yva, tta=tc["sel_tta"]) if ema is not None else (vl, vlog)
        half = {}
        if tc["sel_half"] == "A" and np.isfinite(vl) and np.isfinite(vl_raw):
            vl_s, vl_raw_s = bce_np(vlog, vA), bce_np(vlog_raw, vA)
            half = {"val_loss_A": vl_s, "val_loss_B": bce_np(vlog, vB), "val_loss_raw_A": vl_raw_s}
        else:
            vl_s, vl_raw_s = vl, vl_raw
        # selection on the better of EMA and raw weights (both are legitimate candidates; val loss decides)
        use_raw = ema is not None and vl_raw_s < vl_s
        vsel = min(vl_s, vl_raw_s)
        rec = {**half, "epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vl, "val_auc": vauc, "val_loss_raw": vl_raw,
               "val_loss_sel": vsel, "sel": "raw" if use_raw else "ema", "skipped_steps": nskip, "cycle": cyc["n"],
               "lr": sched.get_last_lr()[-1], "sec": time.time() - t0}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if vsel < best - 1e-5:
            best, bad = vsel, 0
            torch.save((model if use_raw else evm).state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
        if policy == "restart" and vsel < cyc["best"] - 1e-5:      # per-cycle snapshot
            cyc["best"] = vsel
            torch.save((model if use_raw else evm).state_dict(), os.path.join(out, f"snap_{cyc['n']}.pt"))
        ep += 1
        if policy == "none":
            done = ep >= tc["epochs"] or bad >= tc["patience"]
        else:
            cyc_ep = ep - cyc["start_ep"]
            # cycle 0 may end on patience; restart cycles always run to the end of their cosine
            cycle_over = cyc_ep >= cyc["epochs"] or (cyc["n"] == 0 and bad >= tc["restart_patience"])
            done = False
            if cycle_over and cyc["n"] < tc["max_restarts"]:
                sd = torch.load(os.path.join(out, "best.pt"), map_location=dev)
                model.load_state_dict(sd)
                if ema is not None:
                    ema.module.load_state_dict(sd)
                    ema.n_averaged.zero_()
                lo_, hi_ = tc["restart_lr_range"]
                fac = float(np.exp(np.random.uniform(np.log(lo_), np.log(hi_))))
                cyc = {"n": cyc["n"] + 1, "mult": cyc["mult"] * fac, "epochs": tc["restart_epochs"],
                       "start_ep": ep, "best": float("inf")}
                sched = make_cycle_sched(cyc["mult"], cyc["epochs"])
                bad = 0
                msg = {"restart": cyc["n"], "at_epoch": ep, "factor": fac, "peak_lr_mult": cyc["mult"], "best_so_far": best}
                print("RESTART", json.dumps(msg), flush=True)
                with open(os.path.join(out, "restarts.jsonl"), "a") as f_:
                    f_.write(json.dumps(msg) + "\n")
            elif cycle_over:
                done = True
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None, "epoch": ep - 1, "hist": hist,
                    "best": best, "bad": bad, "cyc": cyc, "done": done, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state()}, lastp + ".tmp")
        os.replace(lastp + ".tmp", lastp)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        elapsed = (time.time() - t_start) / 60
        ep_min = np.mean([h["sec"] for h in hist[-3:]]) / 60
        if not done and elapsed + 2 * ep_min + 5 > cfg["time_limit_min"]:
            print(f"time limit approaching ({elapsed:.0f} min); exiting for resubmission", flush=True)
            open(os.path.join(out, "NEEDS_RESUME"), "w").write(str(ep))
            return
    if os.path.exists(os.path.join(out, "NEEDS_RESUME")):
        os.remove(os.path.join(out, "NEEDS_RESUME"))

    # ---------------- final evaluation of the best (val-loss) weights, 8x TTA
    evm = model  # best.pt holds EMA or raw weights (same keys); load into the raw model
    evm.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
    vl1, p1 = evaluate(evm, Xva, yva, tta=1)
    vl8, p8 = evaluate(evm, Xva, yva, tta=8)
    yv = yva.cpu().numpy()
    m1 = all_metrics(yv, 1 / (1 + np.exp(-p1)), meta_va)
    m8 = all_metrics(yv, 1 / (1 + np.exp(-p8)), meta_va)
    np.save(os.path.join(out, "val_logits_tta8.npy"), p8.astype(np.float32))
    np.save(os.path.join(out, "val_logits.npy"), p1.astype(np.float32))
    snaps = sorted(f_ for f_ in os.listdir(out) if f_.startswith("snap_") and f_.endswith(".pt"))
    snap_res = {}
    if len(snaps) > 1:
        lg = []
        for f_ in snaps:
            evm.load_state_dict(torch.load(os.path.join(out, f_), map_location=dev))
            l_, p_ = evaluate(evm, Xva, yva, tta=8)
            snap_res[f_] = l_
            lg.append(p_)
        pe = np.mean(lg, 0)
        le = float(F.binary_cross_entropy_with_logits(torch.tensor(pe), yva.cpu().float()))
        snap_res["ensemble_val_loss_tta8"] = le
        snap_res["ensemble_val_tta8"] = all_metrics(yv, 1 / (1 + np.exp(-pe)), meta_va)
        np.save(os.path.join(out, "val_logits_snapens_tta8.npy"), pe.astype(np.float32))
        print("SNAPSHOT-ENSEMBLE", json.dumps({"n": len(snaps), "val_loss_tta8": le,
                                               "auc": snap_res["ensemble_val_tta8"]["auc"]}), flush=True)
    halves = {"val_loss_tta8_A": bce_np(p8, vA), "val_loss_tta8_B": bce_np(p8, vB), "sel_half": tc["sel_half"],
              "val_tta8_B": all_metrics(yv[vB], 1 / (1 + np.exp(-p8[vB])), meta_va.iloc[vB].reset_index(drop=True))}
    res = {"run_id": run_id, "snapshots": snap_res, "halves": halves, "best_epoch": int(np.argmin([h.get("val_loss_sel", h["val_loss"]) for h in hist])),
           "epochs_run": len(hist), "val_loss": vl1, "val_loss_tta8": vl8, "val": m1, "val_tta8": m8,
           "gpu_hours": (time.time() - t_start) / 3600 + sum(0 for _ in hist)}
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print("FINAL", json.dumps({"val_loss": vl1, "val_loss_tta8": vl8, "auc": m1["auc"], "auc_tta8": m8["auc"],
                               "tpr@1e-2_tta8": m8["tpr@1e-2"], "per_pipe_tta8": m8["per_pipeline"],
                               "auc_fluxmatched_tta8": m8.get("auc_flux_matched")}), flush=True)
    os.remove(lastp)  # keep only best.pt (shared-cluster hygiene)


if __name__ == "__main__":
    main()
