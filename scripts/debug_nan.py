"""Diagnose early-training NaNs (r50_native_scratch, cnxt_native_in1k, r18_up_in1k diverged in epoch 0).

Replays epoch 0 of a config with the train.py recipe (bf16 autocast, AdamW, warm-up, dihedral+shift aug)
and logs per step: loss, grad norm, max |pixel| of the batch, max |activation| at the backbone output.
At the first non-finite loss/grad: forward hooks find the first module with non-finite output, the batch's
extreme stamps are listed, and the same step is recomputed in fp32 (autocast off) for comparison.
Usage: python scripts/debug_nan.py configs/p1/r50_native_scratch.json [max_steps]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lagn.data import load_trainval, random_dihedral, random_shift  # noqa: E402
from lagn.models import build_model  # noqa: E402

cfg = json.load(open(sys.argv[1]))
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
tc = {"bs": 512, "lr": 1e-3, "wd": 0.05, **cfg.get("train", {})}
torch.manual_seed(0)
dev = "cuda"
X, y, keys = load_trainval()
tr = np.where(keys.split.values == "train")[0]
Xtr = torch.from_numpy(np.ascontiguousarray(X[tr])).to(dev)
ytr = torch.from_numpy(y[tr].astype(np.int64)).to(dev)
model = build_model(cfg).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
spe = len(tr) // tc["bs"]
warm = spe
bad_mod = []


def hook(name):
    def f(m, i, o):
        if isinstance(o, torch.Tensor) and not torch.isfinite(o).all() and not bad_mod:
            bad_mod.append((name, type(m).__name__, float(o.float().abs().nan_to_num(0, 0, 0).max())))
    return f


perm = torch.randperm(len(tr), device=dev)
for step in range(min(max_steps, spe)):
    for g in opt.param_groups:
        g["lr"] = tc["lr"] * (step + 1) / warm
    idx = perm[step * tc["bs"]:(step + 1) * tc["bs"]]
    xb = random_shift(random_dihedral(Xtr[idx]), 1).contiguous(memory_format=torch.channels_last)
    yb = ytr[idx].float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = model(xb).float().squeeze(1)
    loss = F.binary_cross_entropy_with_logits(lo, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    pmax = float(Xtr[idx].abs().amax())
    if step % 20 == 0 or not torch.isfinite(loss) or not torch.isfinite(gn):
        print(f"step {step} loss {loss.item():.4f} gn {gn.item():.3e} lr {opt.param_groups[0]['lr']:.2e} "
              f"max|logit| {lo.abs().max().item():.2f} max|pix| {pmax:.3e}", flush=True)
    if not torch.isfinite(loss) or not torch.isfinite(gn):
        print("=== first non-finite at step", step)
        mx = Xtr[idx].abs().flatten(1).amax(1)
        top = mx.argsort(descending=True)[:5]
        print("batch extreme stamps (max |pix|, obj):",
              [(float(mx[t]), keys.obj.values[tr[int(idx[t])]]) for t in top])
        hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb)
        print("bf16 first non-finite module:", bad_mod)
        bad_mod.clear()
        with torch.no_grad():
            lo32 = model(xb.float()).squeeze(1)
        print("fp32 logits finite:", bool(torch.isfinite(lo32).all()), "first non-finite module:", bad_mod)
        for h in hs:
            h.remove()
        # parameters already non-finite?
        badp = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
        print("non-finite params before step:", badp[:10])
        gbad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        print("non-finite grads:", gbad[:10], "of", len(gbad))
        break
    opt.step()
else:
    print("no non-finite value in", min(max_steps, spe), "steps")

# ---------------- evaluation-path check (crashes happened at the first validation pass)
print("=== eval-path check", flush=True)
va = np.where(keys.split.values == "val")[0]
Xva = torch.from_numpy(np.ascontiguousarray(X[va])).to(dev)
ema = torch.optim.swa_utils.AveragedModel(
    model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999), use_buffers=True)
for mname, m in (("raw", model), ("ema_copy", ema.module)):
    m.eval()
    for bs in (2048, 512):
        for amp in (True, False):
            bad = []
            with torch.no_grad():
                for i in range(0, len(Xva), bs):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        o = m(Xva[i:i + bs]).float().squeeze(1)
                    nf = (~torch.isfinite(o)).nonzero().flatten()
                    bad += [i + int(k) for k in nf]
            print(f"{mname} bs={bs} bf16={amp}: non-finite logits {len(bad)}",
                  [keys.obj.values[va[k]] for k in bad[:5]], flush=True)
    m.train()
