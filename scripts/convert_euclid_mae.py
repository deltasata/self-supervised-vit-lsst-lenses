"""Convert Zoobot's Euclid MAE ViT-S/8 encoder (custom timm arch name, 3 channels, 224 px) into an init_ckpt for
timm vit_small_patch8_224(global_pool='avg', fc_norm=True) with 5 bands at img_size 112:
patch_embed inflated 3->5 channels (timm adapt_input_conv), pos_embed resampled 28x28 -> 14x14 (bicubic)."""
import glob
import os

import torch
from safetensors.torch import load_file
from timm.layers import resample_abs_pos_embed
from timm.models import adapt_input_conv

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
f = glob.glob(os.path.join(W, "cache/huggingface/hub/models--mwalmsley--euclid_encoder_mae_zoobot_vit_small_patch8_224/snapshots/*/model.safetensors"))[0]
sd = load_file(f)
sd["patch_embed.proj.weight"] = adapt_input_conv(5, sd["patch_embed.proj.weight"])
sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=(14, 14), num_prefix_tokens=1)
out = os.path.join(W, "cache/converted/euclid_mae_vits8_112_5band.pt")
torch.save(sd, out)
print(out, sd["patch_embed.proj.weight"].shape, sd["pos_embed"].shape)
