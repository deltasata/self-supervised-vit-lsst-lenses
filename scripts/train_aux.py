"""Multi-task classifier: scripts/train.py unchanged except two no-op hooks,
plus auxiliary heads trained on truth-table targets (scripts/aux_targets.py). Targets only, never inputs.

The timm head gets 1 + K outputs: column 0 is the lens logit (all selection, early stopping and metrics use it
exactly as in train.py); columns 1..K are auxiliary outputs whose loss is added with weight train.aux_w:
  binary targets -> BCE, regression targets (standardised) -> smooth-L1; each column averaged over the rows of the
  batch where it is defined, then averaged over columns.
model(x) still returns only the lens logit (B, 1), so evaluate/TTA/predict code is unchanged.
After training: aux metrics of best.pt on val (single view) -> experiments/<run_id>/aux_val.json.

Usage: train_aux.py --cfg <cfg> --run_id <id> [--set train.aux_w=0.5 ...]
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train  # noqa: E402

AUX = os.path.join(train.W, "results", "aux_targets", "aux_targets_trainval")
META = json.load(open(AUX + ".json"))
COLS = list(META["kinds"])
IS_BIN = torch.tensor([META["kinds"][c] == "bin" for c in COLS])
_orig_build = train.build_model
AUX_W = {"w": 0.5}


def _mixin(base):
    class NetAux(base):
        def forward(self, x):
            out = super().forward(x)
            self.aux_out = out[:, 1:]
            return out[:, :1]

        def set_train_rows(self, tr_idx, keys):
            t = pd.read_parquet(AUX + ".parquet").set_index("obj").loc[keys.obj.values[tr_idx], COLS]
            self._aux_t = torch.tensor(t.values, dtype=torch.float32, device=next(self.parameters()).device)
            print(f"aux targets: {len(COLS)} columns, defined per column {np.isfinite(t.values).sum(0).tolist()}",
                  flush=True)

        def aux_loss(self, idx):
            t = self._aux_t[idx]
            o = self.aux_out.float()
            ok = torch.isfinite(t)
            tz = torch.where(ok, t, torch.zeros_like(t))
            isb = IS_BIN.to(t.device)[None].expand_as(t)
            lb = F.binary_cross_entropy_with_logits(o, tz.clamp(0, 1), reduction="none")
            lr = F.smooth_l1_loss(o, tz, reduction="none")
            el = torch.where(isb, lb, lr) * ok
            n = ok.sum(0)
            per_col = el.sum(0) / n.clamp(min=1)
            used = n > 0
            if not used.any():
                return o.sum() * 0
            return AUX_W["w"] * per_col[used].mean()
    return NetAux


def build_aux(cfg):
    model = _orig_build(cfg)
    model.backbone.reset_classifier(1 + len(COLS))
    model.__class__ = _mixin(type(model))
    AUX_W["w"] = float(cfg["train"].get("aux_w", 0.5))
    return model


@torch.no_grad()
def aux_val_report(run_id, cfg):
    out = os.path.join(train.W, "experiments", run_id)
    if os.path.exists(os.path.join(out, "NEEDS_RESUME")) or not os.path.exists(os.path.join(out, "metrics.json")):
        return
    X, y, keys = train.load_trainval()
    va = np.load(os.path.join(out, "val_rows.npy"))
    model = build_aux(cfg).cuda()
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location="cuda"))
    model.eval()
    outs = []
    for i in range(0, len(va), 2048):
        xb = torch.from_numpy(np.ascontiguousarray(X[va[i:i + 2048]])).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(xb)
        outs.append(model.aux_out.float().cpu().numpy())
    o = np.concatenate(outs)
    t = pd.read_parquet(AUX + ".parquet").set_index("obj").loc[keys.obj.values[va], COLS].values
    ss = keys.simset.values[va]
    rep = {}
    for j, c in enumerate(COLS):
        for s in ("D1_L", "D1_N", "D2_L", "D2_N"):
            m = (ss == s) & np.isfinite(t[:, j])
            if m.sum() == 0:
                continue
            if META["kinds"][c] == "bin":
                rep[f"{c}/{s}/acc"] = float(np.mean((o[m, j] > 0) == (t[m, j] > 0.5)))
            else:
                sd = META["standardisation_train"][c]["std"]
                rep[f"{c}/{s}/rmse_std"] = float(np.sqrt(np.mean((o[m, j] - t[m, j]) ** 2)))
                rep[f"{c}/{s}/corr"] = float(np.corrcoef(o[m, j], t[m, j])[0, 1])
                rep[f"{c}/{s}/rmse_phys"] = float(np.sqrt(np.mean((o[m, j] - t[m, j]) ** 2)) * sd)
    json.dump(rep, open(os.path.join(out, "aux_val.json"), "w"), indent=1)
    print("AUX-VAL", json.dumps(rep), flush=True)


if __name__ == "__main__":
    train.DEFAULT["train"]["aux_w"] = 0.5
    train.DEFAULT["track"] = "R-multitask"
    train.build_model = build_aux
    train.main()
    a = sys.argv
    cfg = train.deep_update(copy.deepcopy(train.DEFAULT), json.load(open(a[a.index("--cfg") + 1])))
    for s in (a[a.index("--set") + 1:] if "--set" in a else []):
        if s.startswith("--"):
            break
        k, v = s.split("=", 1)
        train.set_key(cfg, k, v)
    aux_val_report(a[a.index("--run_id") + 1], cfg)
