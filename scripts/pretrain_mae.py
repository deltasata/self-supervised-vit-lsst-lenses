"""Track B: masked-autoencoder (MAE) pretraining on the stamps, then supervised fine-tuning.

Uses TRAIN + VAL stamps only (data/X_trainval.npy), never the holdout. Labels are not used in pretraining.

Encoder = the same timm model the fine-tuning config builds (Net.backbone), e.g.
  * ViT-S, patch 3, 42x42 input (41 -> 42 bilinear), from scratch            -> 196 tokens
  * ViT-B/16 MAE-ImageNet init, 224x224 input (upsampled), continued MAE      -> 196 tokens
Input = Normalizer(asinh) output (5 bands), dihedral augmentation. Random masking (ratio cfg.mask_ratio),
light decoder (timm Blocks), loss = MSE on masked patches with per-patch normalised targets (norm_pix_loss).

Usage (also via run_many.py --script scripts/pretrain_mae.py):
  python scripts/pretrain_mae.py --cfg configs/p2b/<x>.json [--run_id ID] [--set key=val ...]
cfg keys: finetune_cfg (train.py config), mae: {epochs, bs, lr, wd, warmup_epochs, mask_ratio, dec_dim,
dec_depth, dec_heads}. Output experiments/<run_id>/encoder.pt (timm state_dict for Net.backbone), history,
then runs train.py with model.init_ckpt=encoder.pt as run <run_id>_ft.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import W, load_trainval, random_dihedral  # noqa: E402
from lagn.models import build_model  # noqa: E402

DEFAULT_MAE = {"epochs": 300, "bs": 512, "lr": 1.5e-4, "wd": 0.05, "warmup_epochs": 10, "mask_ratio": 0.75,
               "dec_dim": 256, "dec_depth": 4, "dec_heads": 8, "norm_pix_loss": True, "max_images": None,
               # extensions: band_drop = prob. of hiding each band inside VISIBLE patches (cross-band
               # prediction); target = "pixel" (MAE) | "latent" (data2vec-2-style: predict the EMA teacher's
               # instance-normed mean of the top `latent_k` block outputs) ; pixel_weight adds the pixel loss to latent
               "band_drop": 0.0, "target": "pixel", "latent_k": 4, "pixel_weight": 0.0,
               "ema_start": 0.999, "ema_end": 0.9999,
               "split": "trainval"}   # "train" = pretrain on train stamps only (control: no val images in SSL)


def set_key(d, key, val):
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        pass
    d[ks[-1]] = val


class MAE(nn.Module):
    def __init__(self, net, mc):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.net = net                       # Net: .norm, .size, .backbone (timm VisionTransformer)
        enc = net.backbone
        self.enc = enc
        D = enc.embed_dim
        self.p = enc.patch_embed.patch_size[0]
        self.grid = enc.patch_embed.grid_size
        self.N = self.grid[0] * self.grid[1]
        self.C = net.norm.out_ch
        self.npref = enc.num_prefix_tokens
        self.mask_ratio = mc["mask_ratio"]
        self.norm_pix = mc["norm_pix_loss"]
        dd = mc["dec_dim"]
        self.dec_embed = nn.Linear(D, dd)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dd))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.N, dd))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.dec = nn.Sequential(*[Block(dd, mc["dec_heads"], 4.0, qkv_bias=True) for _ in range(mc["dec_depth"])])
        self.dec_norm = nn.LayerNorm(dd)
        self.dec_pred = nn.Linear(dd, self.p * self.p * self.C)
        self.band_drop = mc["band_drop"]
        self.target_mode, self.latent_k, self.pixel_weight = mc["target"], mc["latent_k"], mc["pixel_weight"]
        self.teacher = None
        if self.target_mode == "latent":
            self.dec_lat = nn.Linear(dd, D)
            self.teacher = copy.deepcopy(enc).requires_grad_(False)

    @torch.no_grad()
    def ema_update(self, tau):
        for pt, ps in zip(self.teacher.parameters(), self.enc.parameters()):
            pt.lerp_(ps.detach(), 1 - tau)

    @torch.no_grad()
    def teacher_targets(self, x):
        """Mean of the instance-normed patch-token outputs of the teacher's top-k blocks (full, unmasked input)."""
        te = self.teacher
        te.eval()
        t = te.patch_embed(x)
        if t.dim() == 4:
            t = t.flatten(1, 2)
        pos = te.pos_embed
        has_prefix_pos = pos.shape[1] == self.N + self.npref
        t = t + (pos[:, self.npref:] if has_prefix_pos else pos)
        B = t.shape[0]
        pref = []
        if te.cls_token is not None:
            pref.append((te.cls_token + (pos[:, :1] if has_prefix_pos else 0)).expand(B, -1, -1))
        if getattr(te, "reg_token", None) is not None:
            pref.append(te.reg_token.expand(B, -1, -1))
        t = torch.cat(pref + [t], 1)
        t = te.norm_pre(t) if hasattr(te, "norm_pre") else t
        outs = []
        nb = len(te.blocks)
        for i, blk in enumerate(te.blocks):
            t = blk(t)
            if i >= nb - self.latent_k:
                outs.append(F.layer_norm(t[:, self.npref:].float(), t.shape[-1:]))
        y = torch.stack(outs).mean(0)
        return F.layer_norm(y, y.shape[-1:])

    def prep(self, x):
        x = self.net.norm(x)
        if self.net.size:
            x = F.interpolate(x, size=(self.net.size, self.net.size), mode="bilinear", align_corners=False)
        if getattr(self.net, "pad", None):                    # same zero padding as Net.forward
            x = F.pad(x, self.net.pad)
        return x

    def patchify(self, x):
        B, C, H, Wd = x.shape
        p = self.p
        x = x.reshape(B, C, H // p, p, Wd // p, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, (H // p) * (Wd // p), p * p * C)

    def forward(self, raw):
        x = self.prep(raw)
        enc = self.enc
        B0, C0, H0, W0 = x.shape
        bdrop = None
        xin = x
        if self.band_drop > 0:                                        # hide whole bands inside patches
            bdrop = (torch.rand(B0, C0, H0 // self.p, W0 // self.p, device=x.device) < self.band_drop).float()
            xin = x * (1 - F.interpolate(bdrop, scale_factor=self.p, mode="nearest"))
        t = enc.patch_embed(xin)                                      # (B, N, D)
        if t.dim() == 4:                                              # NHWC output in some timm versions
            t = t.flatten(1, 2)
        pos = enc.pos_embed
        has_prefix_pos = pos.shape[1] == self.N + self.npref
        t = t + (pos[:, self.npref:] if has_prefix_pos else pos)
        B, N, D = t.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=t.device)
        ids_shuffle = noise.argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        ids_keep = ids_shuffle[:, :keep]
        t = t.gather(1, ids_keep[..., None].expand(-1, -1, D))
        pref = []
        if enc.cls_token is not None:
            c = enc.cls_token + (pos[:, :1] if has_prefix_pos else 0)
            pref.append(c.expand(B, -1, -1))
        if getattr(enc, "reg_token", None) is not None:
            pref.append(enc.reg_token.expand(B, -1, -1))
        t = torch.cat(pref + [t], 1)
        t = enc.norm_pre(t) if hasattr(enc, "norm_pre") else t
        t = enc.blocks(t)
        t = enc.norm(t)
        t = self.dec_embed(t[:, self.npref:])
        m = self.mask_token.expand(B, N - keep, -1)
        t = torch.cat([t, m], 1).gather(1, ids_restore[..., None].expand(-1, -1, t.shape[-1]))
        t = self.dec_norm(self.dec(t + self.dec_pos))
        mask = torch.ones(B, N, device=t.device)
        mask[:, :keep] = 0
        mask = mask.gather(1, ids_restore)
        total = 0.0
        if self.target_mode == "latent":
            tl = self.teacher_targets(x)
            ll = ((self.dec_lat(t).float() - tl) ** 2).mean(-1)
            total = (ll * mask).sum() / mask.sum()
        if self.target_mode == "pixel" or self.pixel_weight > 0:
            pred = self.dec_pred(t)
            target = self.patchify(x)
            if self.norm_pix:
                mu = target.mean(-1, keepdim=True)
                var = target.var(-1, keepdim=True)
                target = (target - mu) / (var + 1e-6) ** 0.5
            err = ((pred.float() - target.float()) ** 2).reshape(B, N, self.p * self.p, self.C).mean(2)  # (B,N,C)
            w = mask[..., None].expand(-1, -1, self.C)
            if bdrop is not None:                     # also score the hidden bands of visible patches
                w = torch.maximum(w, bdrop.flatten(2).transpose(1, 2))
            lp = (err * w).sum() / w.sum()
            total = total + (lp if self.target_mode == "pixel" else self.pixel_weight * lp)
        return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--run_id")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = json.load(open(a.cfg))
    for s in a.set:
        k, v = s.split("=", 1)
        set_key(cfg, k, v)
    run_id = a.run_id or os.path.splitext(os.path.basename(a.cfg))[0]
    out = os.path.join(W, "experiments", run_id)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    time_limit = float(cfg.get("time_limit_min", 1420))
    mc = {**DEFAULT_MAE, **cfg.get("mae", {})}
    ftc = json.load(open(os.path.join(W, cfg["finetune_cfg"])))
    torch.manual_seed(cfg.get("seed", 0))
    dev = "cuda"

    enc_path = os.path.join(out, "encoder.pt")
    if not os.path.exists(enc_path):
        X, y, keys = load_trainval()                                   # train (+ val) stamps, labels unused
        idx = np.arange(len(X)) if mc["split"] == "trainval" else np.where(keys.split.values == "train")[0]
        if mc["max_images"]:
            idx = idx[:mc["max_images"]]
        Xg = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev)
        net = build_model(ftc)
        model = MAE(net, mc).to(dev)
        params = [p for n, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=mc["lr"] * mc["bs"] / 256, betas=(0.9, 0.95), weight_decay=mc["wd"])
        spe = len(idx) // mc["bs"]
        total, warm = mc["epochs"] * spe, mc["warmup_epochs"] * spe
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / max(1, total - warm)))))
        ck = os.path.join(out, "mae_last.pt")
        ep0, hist = 0, []
        if os.path.exists(ck):
            c = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(c["model"])
            opt.load_state_dict(c["opt"])
            sched.load_state_dict(c["sched"])
            ep0, hist = c["epoch"] + 1, c["hist"]
            print(f"resumed MAE at epoch {ep0}", flush=True)
        else:
            json.dump({**cfg, "run_id": run_id, "mae_resolved": mc, "n_images": int(len(idx))},
                      open(os.path.join(out, "config.json"), "w"), indent=1)
        print(f"MAE: {len(idx)} images, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
              f"tokens {model.N}, patch {model.p}", flush=True)
        for ep in range(ep0, mc["epochs"]):
            model.train()
            t0 = time.time()
            perm = torch.randperm(len(idx), device=dev)
            tl = []
            for b in range(spe):
                xb = random_dihedral(Xg[perm[b * mc["bs"]:(b + 1) * mc["bs"]]])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 3.0)
                opt.step()
                sched.step()
                if model.teacher is not None:
                    frac = min(1.0, (ep * spe + b) / max(1, total))
                    model.ema_update(mc["ema_start"] + (mc["ema_end"] - mc["ema_start"]) * frac)
                if b % 20 == 0:
                    tl.append(loss.item())
            rec = {"epoch": ep, "mae_loss": float(np.mean(tl)), "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            if ep % 5 == 4 or ep == mc["epochs"] - 1:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "epoch": ep, "hist": hist}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                pd.DataFrame(hist).to_csv(os.path.join(out, "mae_history.csv"), index=False)
            el = (time.time() - t_start) / 60
            if el + 3 * np.mean([h["sec"] for h in hist[-3:]]) / 60 + 10 > time_limit and ep < mc["epochs"] - 1:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "epoch": ep, "hist": hist}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                print("time limit approaching; exit for resubmission", flush=True)
                return
        torch.save(model.enc.state_dict(), enc_path)
        pd.DataFrame(hist).to_csv(os.path.join(out, "mae_history.csv"), index=False)
        os.remove(ck)
        del Xg, model, opt
        torch.cuda.empty_cache()
        print("MAE done; encoder saved", flush=True)

    # ---------------- fine-tune with train.py
    left = time_limit - (time.time() - t_start) / 60 - 2
    if left < 20:
        print(f"only {left:.0f} min left; fine-tune later", flush=True)
        return
    cmd = [sys.executable, "-u", os.path.join(W, "scripts", "train.py"), "--cfg", cfg["finetune_cfg"],
           "--run_id", f"{run_id}_ft", "--set", f"model.init_ckpt={enc_path}", f"time_limit_min={left:.0f}"]
    cmd += cfg.get("finetune_set", [])
    print("fine-tune:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=W))


if __name__ == "__main__":
    main()
