"""AstroDINO backbone (ViT-L/12, 144 px, RGB = (z, r, g)) -> init_ckpt for our native-resolution layout
timm vit_large_patch16_224(img_size=42, patch_size=3, init_values=1e-5, in_chans=5, norm "asinh"), used as the starting
point for continued MAE pretraining on the stamps:
patch embedding resampled 12x12 -> 3x3 (timm resample_patch_embed, pseudo-inverse resize) and mapped to bands
g r i z y = B, G, (G+R)/2, R, R (x 3/5 to keep the summed response); pos_embed resampled 12x12 -> 14x14 (bicubic)."""
import glob
import os
import re

import torch
from timm.layers import resample_abs_pos_embed, resample_patch_embed

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
f = glob.glob(os.path.join(W, "cache/huggingface/hub/models--polymathic-ai--astrodino/snapshots/*/astrodino.ckpt"))[0]
sd = torch.load(f, map_location="cpu", weights_only=True)["teacher"]
out = {}
for k, v in sd.items():
    if not k.startswith("backbone.") or k == "backbone.mask_token":
        continue
    k = k[len("backbone."):]
    m = re.match(r"blocks\.(\d+)\.(\d+)\.(.*)", k)
    if m:
        k = f"blocks.{int(m.group(2))}.{m.group(3)}"
    out[k] = v.float()
w = resample_patch_embed(out["patch_embed.proj.weight"], [3, 3])            # (1024, 3, 3, 3), channels R G B
R, G, B = w[:, 0:1], w[:, 1:2], w[:, 2:3]
out["patch_embed.proj.weight"] = torch.cat([B, G, (G + R) / 2, R, R], 1) * 3 / 5
out["pos_embed"] = resample_abs_pos_embed(out["pos_embed"], new_size=(14, 14), num_prefix_tokens=1)
dst = os.path.join(W, "cache/converted/astrodino_vitl3_42_5band.pt")
torch.save(out, dst)
print(dst, out["patch_embed.proj.weight"].shape, out["pos_embed"].shape)
