"""Track B for CNNs: SimMIM-style masked-image-modelling pretraining on the stamps, then supervised fine-tuning.

The CNN counterpart of scripts/pretrain_mae.py (which needs a ViT). TRAIN + VAL stamps only, labels unused,
never the holdout.

Recipe (SimMIM / A2MIM-style dense masking, which works for CNNs without sparse convolutions):
  * input = Normalizer output (5 bands), dihedral + 1-px shift augmentation
  * random block mask (block `mask_px` px, ratio `mask_ratio`); masked pixels are replaced by a learnable
    per-band mask value
  * encoder = the fine-tuning model's feature extractor (timm forward_features for Net, .f for LensCNN),
    with the native stride-1 stem, giving a ~6x6 map at stride ~8
  * decoder = 1x1 conv -> pixel shuffle (x stride) -> crop to 41x41 -> one 3x3 conv (light, discarded after)
  * loss = L1 on masked pixels only
Output: experiments/<run_id>/encoder.pt (state_dict for model.backbone, or the whole LensCNN), history, then
train.py with model.init_ckpt=encoder.pt as run <run_id>_ft.

Usage: python scripts/pretrain_mim_cnn.py --cfg configs/p2b_cnn/<x>.json [--run_id ID] [--set key=val ...]
cfg keys: finetune_cfg, finetune_set (list of extra train.py --set items), mim: {epochs, bs, lr, wd,
warmup_epochs, mask_ratio, mask_px}.
"""
import argparse
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
from lagn.data import W, load_trainval, random_dihedral, random_shift  # noqa: E402
from lagn.models import LensCNN, build_model  # noqa: E402

DEFAULT_MIM = {"epochs": 300, "bs": 256, "lr": 1e-3, "wd": 0.05, "warmup_epochs": 10, "mask_ratio": 0.6,
               "mask_px": 4, "max_images": None, "split": "trainval"}


def set_key(d, key, val):
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    try:
        val = json.loads(val)
    except json.JSONDecodeError:
        pass
    d[ks[-1]] = val


class MIM(nn.Module):
    def __init__(self, net, mc, H=41):
        super().__init__()
        self.net = net
        self.is_lens = isinstance(net, LensCNN)
        self.C = net.norm.out_ch
        self.H = H
        self.ratio, self.mpx = mc["mask_ratio"], mc["mask_px"]
        self.mask_val = nn.Parameter(torch.zeros(1, self.C, 1, 1))
        with torch.no_grad():
            f = self.features(torch.zeros(2, self.C, H, H))
        D, h = f.shape[1], f.shape[-1]
        self.stride = math.ceil(H / h)
        self.dec = nn.Conv2d(D, self.C * self.stride ** 2, 1)
        self.refine = nn.Conv2d(self.C, self.C, 3, padding=1)

    def encoder(self):
        return self.net if self.is_lens else self.net.backbone

    def features(self, x):
        return self.net.f(x) if self.is_lens else self.net.backbone.forward_features(x)

    def forward(self, raw):
        x = self.net.norm(raw)
        B = x.shape[0]
        g = math.ceil(self.H / self.mpx)
        m = (torch.rand(B, 1, g, g, device=x.device) < self.ratio).float()
        m = F.interpolate(m, scale_factor=self.mpx, mode="nearest")[..., :self.H, :self.H]
        xin = x * (1 - m) + self.mask_val * m
        f = self.features(xin)
        rec = F.pixel_shuffle(self.dec(f), self.stride)[..., :self.H, :self.H]
        rec = rec + self.refine(rec)
        l1 = (rec.float() - x.float()).abs() * m
        return l1.sum() / (m.sum() * self.C + 1e-6)


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
    mc = {**DEFAULT_MIM, **cfg.get("mim", {})}
    ftc = json.load(open(os.path.join(W, cfg["finetune_cfg"])))
    torch.manual_seed(cfg.get("seed", 0))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    enc_path = os.path.join(out, "encoder.pt")
    if not os.path.exists(enc_path):
        X, _, keys = load_trainval()                                  # train (+ val) stamps, labels unused
        idx = np.arange(len(X)) if mc["split"] == "trainval" else np.where(keys.split.values == "train")[0]
        if mc["max_images"]:
            idx = idx[:mc["max_images"]]
        Xg = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev)
        net = build_model(ftc)                                       # honours pretrained=True (continued MIM)
        model = MIM(net, mc).to(dev).to(memory_format=torch.channels_last)
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr=mc["lr"], betas=(0.9, 0.95), weight_decay=mc["wd"])
        spe = len(idx) // mc["bs"]
        total, warm = mc["epochs"] * spe, mc["warmup_epochs"] * spe
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / max(1, total - warm)))))
        ck = os.path.join(out, "mim_last.pt")
        ep0, hist = 0, []
        if os.path.exists(ck):
            c = torch.load(ck, map_location=dev, weights_only=False)
            model.load_state_dict(c["model"])
            opt.load_state_dict(c["opt"])
            sched.load_state_dict(c["sched"])
            ep0, hist = c["epoch"] + 1, c["hist"]
            print(f"resumed MIM at epoch {ep0}", flush=True)
        else:
            json.dump({**cfg, "run_id": run_id, "mim_resolved": mc, "n_images": int(len(idx))},
                      open(os.path.join(out, "config.json"), "w"), indent=1)
        print(f"MIM: {len(idx)} images, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
              f"feature stride {model.stride}", flush=True)

        def save(ep):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "hist": hist}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            pd.DataFrame(hist).to_csv(os.path.join(out, "mim_history.csv"), index=False)

        nbad = 0
        for ep in range(ep0, mc["epochs"]):
            model.train()
            t0 = time.time()
            perm = torch.randperm(len(idx), device=dev)
            tl = []
            for b in range(spe):
                xb = random_shift(random_dihedral(Xg[perm[b * mc["bs"]:(b + 1) * mc["bs"]]]), 1)
                xb = xb.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    loss = model(xb)
                if not torch.isfinite(loss):
                    nbad += 1
                    assert nbad < 50, "too many non-finite losses"
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, 3.0)
                if not torch.isfinite(gn):
                    nbad += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                sched.step()
                if b % 20 == 0:
                    tl.append(loss.item())
            rec = {"epoch": ep, "mim_loss": float(np.mean(tl)), "lr": sched.get_last_lr()[0],
                   "sec": time.time() - t0, "skipped": nbad}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            if ep % 5 == 4 or ep == mc["epochs"] - 1:
                save(ep)
            el = (time.time() - t_start) / 60
            if el + 3 * np.mean([h["sec"] for h in hist[-3:]]) / 60 + 10 > time_limit and ep < mc["epochs"] - 1:
                save(ep)
                print("time limit approaching; exit for resubmission", flush=True)
                return
        torch.save(model.encoder().state_dict(), enc_path)
        pd.DataFrame(hist).to_csv(os.path.join(out, "mim_history.csv"), index=False)
        os.remove(ck)
        del Xg, model, opt
        torch.cuda.empty_cache()
        print("MIM done; encoder saved", flush=True)

    if cfg.get("skip_finetune"):                                     # encoder-only runs (e.g. for regression)
        print("skip_finetune: done", flush=True)
        return
    left = time_limit - (time.time() - t_start) / 60 - 2
    if left < 20:
        print(f"only {left:.0f} min left; fine-tune later", flush=True)
        return
    cmd = [sys.executable, "-u", os.path.join(W, "scripts", "train.py"), "--cfg", cfg["finetune_cfg"],
           "--run_id", f"{run_id}_ft", "--set", f"model.init_ckpt={enc_path}", "model.pretrained=false",
           f"time_limit_min={left:.0f}"]
    cmd += cfg.get("finetune_set", [])
    print("fine-tune:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=W))


if __name__ == "__main__":
    main()
