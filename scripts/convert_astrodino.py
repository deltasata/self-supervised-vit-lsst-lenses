"""Convert the AstroDINO teacher backbone (AstroCLIP's image encoder: DINOv2 ViT-L/12, LayerScale, 4 block chunks,
3-channel Legacy Survey RGB = (z, r, g) at 144 px) into an init_ckpt for
timm vit_large_patch16_224(patch_size=12, img_size=144, init_values=1e-5, in_chans=5) used with norm "lsrgb":
chunked block indices flattened, mask_token and DINO head dropped, patch embedding 3 -> 5 channels with the two extra
channels (i, y) zero-initialised, so at init the model sees exactly its pretraining input."""
import glob
import os
import re

import torch

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
idx = sorted({int(k.split(".")[1]) for k in out if k.startswith("blocks.")})
assert idx == list(range(24)), idx
w = out["patch_embed.proj.weight"]
out["patch_embed.proj.weight"] = torch.cat([w, torch.zeros(w.shape[0], 2, *w.shape[2:])], 1)
dst = os.path.join(W, "cache/converted/astrodino_vitl12_144_5band.pt")
torch.save(out, dst)
print(dst, len(out), out["patch_embed.proj.weight"].shape, out["pos_embed"].shape)
