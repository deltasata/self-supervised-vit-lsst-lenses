"""CPU smoke test of zoo configs (offline weights, 5-band input, forward pass, params, LLRD grouping).
Usage: HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 python scripts/smoke_zoo.py configs/zoo/*.json"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
torch.set_num_threads(4)
from lagn.models import build_model  # noqa: E402

src = open("scripts/train.py").read()
blk = src[src.index('    if tc["llrd"]:\n'):src.index('        print("llrd groups"')]
code = "\n".join(line[4:] for line in blk.splitlines())
for p in sys.argv[1:]:
    tag = p.split("/")[-1][:-5]
    try:
        cfg = json.load(open(p))
        m = build_model(cfg).eval()
        with torch.no_grad():
            o = m(torch.rand(2, 5, 41, 41))
        tc = {"llrd": 0.75, "lr": 1.0, "lr_backbone_mult": 1.0}
        bb = [q for n, q in m.named_parameters() if n.startswith("backbone.") and "head" not in n and "fc." not in n
              and "classifier" not in n]
        other = [q for n, q in m.named_parameters() if not any(q is r for r in bb)]
        g = {"model": m, "tc": tc, "bb": bb, "other": other, "cfg": cfg, "np": np, "torch": torch}
        exec(code, g)
        print(f"OK {tag:22s} params {sum(q.numel() for q in m.parameters()) / 1e6:6.1f}M out {tuple(o.shape)} "
              f"llrd groups {len(g['by'])} top {g['top']} head/other {len(other)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {tag}: {repr(e)[:300]}", flush=True)
