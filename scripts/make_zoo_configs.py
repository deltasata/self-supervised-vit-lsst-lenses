"""Model-zoo screening configs: many pretrained backbones, one fixed fine-tuning recipe, so the
comparison is about the backbone/pretraining, not the schedule.

Recipe: input upsampled (bilinear) so a 16-px patch covers ~3 stamp pixels (224 for p16/p14 and hierarchical models,
112 for p8 models); 5 bands via timm's in_chans inflation; LLRD 0.75 (ViT-family by block, others by 12 parameter-
order buckets, see train.py); AdamW lr 3e-4 at bs 256, warm-up 2, cosine 30 epochs, patience 8, EMA, noise aug 0.5,
drop-path 0.1. Up to 8 cosine restarts from the best weights, random
peak-lr factor in [0.25, 2], 12 epochs each; 24 h jobs.
Writes configs/zoo/<tag>.json.
"""
import json
import os

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ZB = "hf_hub:mwalmsley/"
MODELS = {
    # tag: (timm name, input size, extra model kwargs, family)
    # --- Zoobot (Galaxy Zoo / Euclid pretrained) ---
    "zb_cnxt_nano": (ZB + "zoobot-encoder-convnext_nano", 224, {}, "zoobot-cnn"),
    "zb_cnxt_tiny": (ZB + "zoobot-encoder-convnext_tiny", 224, {}, "zoobot-cnn"),
    "zb_cnxt_base": (ZB + "zoobot-encoder-convnext_base", 224, {}, "zoobot-cnn"),
    "zb_effb0": (ZB + "zoobot-encoder-efficientnet_b0", 224, {}, "zoobot-cnn"),
    "zb_r50": (ZB + "zoobot-encoder-resnet50", 224, {}, "zoobot-cnn"),
    "zb_maxvit_tiny": (ZB + "zoobot-encoder-maxvit_tiny_rw_224", 224, {}, "zoobot-hybrid"),
    "zb_maxvit_base": (ZB + "zoobot-encoder-maxvit_base_rw_224", 224, {}, "zoobot-hybrid"),
    "zb_maxvit_base_euclid": (ZB + "zoobot-encoder-euclid-maxvit-base", 224, {}, "zoobot-hybrid"),
    "zb_maxvit_base_evo": (ZB + "zoobot-encoder-evo-maxvit-base", 224, {}, "zoobot-hybrid"),
    # custom arch name on the hub: weights converted by scripts/convert_euclid_mae.py (5-band patch embed, 14x14 pos)
    "zb_euclid_mae_vits8": ("vit_small_patch8_224", 112,
                            {"img_size": 112, "global_pool": "avg", "fc_norm": True, "pretrained": False,
                             "init_ckpt": "cache/converted/euclid_mae_vits8_112_5band.pt"}, "zoobot-vit"),
    # --- plain ViTs, different pretraining objectives ---
    "dino_vits8": ("vit_small_patch8_224.dino", 112, {"img_size": 112}, "vit"),
    "dino_vitb16": ("vit_base_patch16_224.dino", 224, {}, "vit"),
    "dinov2_vits14": ("vit_small_patch14_dinov2.lvd142m", 224, {"img_size": 224}, "vit"),
    "deit3_s": ("deit3_small_patch16_224.fb_in22k_ft_in1k", 224, {}, "vit"),
    "deit3_b": ("deit3_base_patch16_224.fb_in22k_ft_in1k", 224, {}, "vit"),
    "augreg_vits16": ("vit_small_patch16_224.augreg_in21k_ft_in1k", 224, {}, "vit"),
    "sbb_vitm_reg4": ("vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k", 224, {"img_size": 224}, "vit"),
    "beitv2_b": ("beitv2_base_patch16_224.in1k_ft_in22k_in1k", 224, {}, "vit"),
    "eva02_t": ("eva02_tiny_patch14_224.mim_in22k", 224, {}, "vit"),
    "eva02_b": ("eva02_base_patch14_224.mim_in22k", 224, {}, "vit"),
    "siglip_b": ("vit_base_patch16_siglip_224.webli", 224, {}, "vit"),
    "clip_b": ("vit_base_patch16_clip_224.openai", 224, {}, "vit"),
    "xcit_s12_p8": ("xcit_small_12_p8_224.fb_dist_in1k", 112, {"img_size": 112}, "vit"),
    # --- hierarchical / hybrid transformers ---
    "swin_t": ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 224, {}, "hier"),
    "maxvit_t": ("maxvit_tiny_tf_224.in1k", 224, {}, "hier"),
    "coatnet_0": ("coatnet_0_rw_224.sw_in1k", 224, {}, "hier"),
    "caformer_s18": ("caformer_s18.sail_in22k", 224, {}, "hier"),
    "tinyvit_21m": ("tiny_vit_21m_224.dist_in22k_ft_in1k", 224, {}, "hier"),
    "pvtv2_b2": ("pvt_v2_b2.in1k", 224, {}, "hier"),
    "mvitv2_t": ("mvitv2_tiny.fb_in1k", 224, {}, "hier"),
    "hiera_b": ("hiera_base_224.mae_in1k_ft_in1k", 224, {}, "hier"),
    "davit_t": ("davit_tiny.msft_in1k", 224, {}, "hier"),
    "gcvit_t": ("gcvit_tiny.in1k", 224, {}, "hier"),
}


def main():
    os.makedirs(os.path.join(W, "configs/zoo"), exist_ok=True)
    for tag, (arch, size, kw, fam) in MODELS.items():
        cfg = {
            "model": {"arch": arch, "stem": "plain", "size": size, "pretrained": True, "norm": "asinh",
                      "norm_a": 3.0, "drop_path_rate": 0.1, **kw},
            "train": {"epochs": 30, "bs": 256, "lr": 3e-4, "wd": 0.05, "warmup_epochs": 2, "patience": 8,
                      "llrd": 0.75, "noise_aug": 0.5, "channels_last": False,
                   # cosine restarts from the best weights
                   "plateau_policy": "restart", "restart_patience": 8, "restart_epochs": 12, "max_restarts": 8,
                   "restart_lr_range": [0.25, 2.0], "sel_tta": 8},
            "track": f"ZOO-{fam}",
        }
        json.dump(cfg, open(os.path.join(W, f"configs/zoo/{tag}.json"), "w"), indent=1)
    print(len(MODELS), "configs")


if __name__ == "__main__":
    main()
