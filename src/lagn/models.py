"""Model zoo. Every model = Normalizer -> (optional resize) -> backbone -> 1 logit.

Input to Net.forward: raw pixel values (B, 5, 41, 41), bands g r i z y.

CNN stem adaptation for 41x41 stamps:
  stem="native"   : first conv stride 1 and no max-pool (ResNet), stem stride 1 (ConvNeXt, EffNetV2)
  stem="upsample" : keep the standard ImageNet stem and resize the input to `size` px (bilinear)
  stem="plain"    : no stem change (ViTs etc.; pass img_size / patch_size through **kw), optional resize
Pretrained (ImageNet) weights: timm, with the first conv inflated from 3 to C channels by timm's
`in_chans` mechanism (weights repeated and rescaled).
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Normalizer


def _adapt_stem_native(m, arch):
    if arch.startswith("resnet") or arch.startswith("resnext") or arch.startswith("seresnet"):
        if isinstance(m.conv1, nn.Sequential):      # deep/tiered stems (resnet10t, resnet14t, resnet26d...)
            m.conv1[0].stride = (1, 1)
        else:
            m.conv1.stride = (1, 1)
        m.maxpool = nn.Identity()
    elif arch.startswith("convnext"):
        c = m.stem[0]
        # 4x4/4 patchify -> 4x4 stride 1 (output 42x42 with pad 2, cropped by the next downsample)
        c.stride, c.padding = (1, 1), (2, 2)
    elif "efficientnet" in arch:
        m.conv_stem.stride = (1, 1)
    elif arch.startswith("regnet"):
        m.stem.conv.stride = (1, 1)
    else:
        raise ValueError(f"no native stem rule for {arch}")
    return m


class Net(nn.Module):
    def __init__(self, arch="resnet18", stem="native", size=None, pretrained=False, norm="asinh", norm_a=3.0,
                 drop_rate=0.0, drop_path_rate=0.0, resize_mode="bilinear", input_mask=None, no_down=None, pad=None,
                 **kw):
        super().__init__()
        self.norm = Normalizer(norm, a=norm_a)
        # pad=[left, right, top, bottom]: zero padding (= sky level after asinh) of the normalised stamp instead of
        # resizing, e.g. [2,2,2,2] -> 45 px = 15x3 for 3-px ViT patches with raw pixels. Default off.
        self.pad = tuple(int(v) for v in pad) if pad else None
        assert not (self.pad and size), "use either resize (size) or pad, not both"
        # diagnostic input masks (applied after normalisation, masked pixels = 0):
        #   "annulusR": keep only r >= R (background-only shortcut test); "centerR": keep only r < R
        if not input_mask:
            self.input_mask = None
        else:
            import numpy as np
            yy, xx = np.mgrid[:41, :41]
            rr = np.hypot(yy - 20, xx - 20)
            R = float("".join(ch for ch in input_mask if ch.isdigit() or ch == "."))
            keep = rr >= R if input_mask.startswith("annulus") else rr < R
            self.register_buffer("input_mask", torch.tensor(keep, dtype=torch.float32)[None, None])
        self.size, self.resize_mode = size, resize_mode
        extra = {"drop_path_rate": drop_path_rate} if drop_path_rate else {}
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=1, in_chans=self.norm.out_ch,
                                          drop_rate=drop_rate, **extra, **kw)
        if stem == "native":
            _adapt_stem_native(self.backbone, arch)
            assert size is None, "native stem uses the 41x41 input"
        elif stem == "upsample":
            assert size is not None
        elif stem == "plain":
            pass  # model used as created (e.g. ViT with img_size/patch_size kwargs); optional resize to `size`
        else:
            raise ValueError(stem)
        # no_down=[i, ...]: ConvNeXt stage i keeps its resolution (2x2 downsample conv -> stride 1), i.e. a
        # higher-resolution feature map for the 2.5-10 px image separations
        for i in (no_down or []):
            ds = self.backbone.stages[i].downsample
            conv = [m for m in ds.modules() if isinstance(m, nn.Conv2d)][0]
            conv.stride = (1, 1)

    def forward(self, x):
        x = self.norm(x)
        if self.input_mask is not None:
            x = x * self.input_mask
        if self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode=self.resize_mode,
                              align_corners=False if self.resize_mode != "nearest" else None)
        if self.pad:
            x = F.pad(x, self.pad)
        return self.backbone(x)


class LensCNN(nn.Module):
    """Compact lens-finder CNN (in the spirit of the Bologna/CMU lens-finding challenge CNNs, ~1-3M params):
    3x3 conv blocks at full 41x41 resolution, BN+GELU, two stride-2 downsamplings, global pooling."""

    def __init__(self, norm="asinh", norm_a=3.0, width=64, nblk=3, drop_rate=0.2, **_):
        super().__init__()
        self.norm = Normalizer(norm, a=norm_a)
        c = self.norm.out_ch

        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.GELU())
        w = width
        # stages at 41, 21, 11 px with nblk conv layers each, then a 6-px stage with 2 (nblk=3 -> 11 layers)
        layers, cin = [], c
        for st, cout in enumerate((w, 2 * w, 4 * w)):
            for k in range(nblk):
                layers.append(blk(cin, cout, 2 if (k == 0 and st > 0) else 1))
                cin = cout
        layers += [blk(cin, 8 * w, 2), blk(8 * w, 8 * w)]
        self.f = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(8 * w * 2, 1))

    def forward(self, x):
        h = self.f(self.norm(x))
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.head(h)


class BandTokenViT(nn.Module):
    """Per-band-token ViT (from scratch, native resolution).

    Each band is patchified separately with a shared 1-channel patch embedding (patch p on the 42x42 resized
    stamp), plus a learned band embedding and a learned 2-D position embedding, so attention can compare the same
    sky position across bands (colour consistency of lensed images) and different positions within a band
    (multiplicity / geometry). Tokens = 5 x (42/p)^2 (+ cls).
    """

    def __init__(self, norm="asinh", norm_a=3.0, patch=3, dim=256, depth=8, heads=8, mlp_ratio=4.0,
                 drop_path_rate=0.1, size=42, **_):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.norm = Normalizer(norm, a=norm_a)
        self.size, self.p = size, patch
        g = size // patch
        self.embed = nn.Conv2d(1, dim, patch, patch)
        self.band = nn.Parameter(torch.zeros(1, self.norm.out_ch, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1, g * g, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.band, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i])
                                      for i in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        B, C = x.shape[:2]
        t = self.embed(x.reshape(B * C, 1, self.size, self.size)).flatten(2).transpose(1, 2)  # (B*C, N, D)
        t = t.reshape(B, C, -1, t.shape[-1]) + self.band + self.pos
        t = torch.cat([self.cls.expand(B, -1, -1), t.flatten(1, 2)], 1)
        t = self.ln(self.blocks(t))
        return self.head(t[:, 0])


class PointSetNet(nn.Module):
    """Point-source set transformer. Detect up to `k` local maxima of a noise-normalised
    g+r+i+z detection map (the stamp centre is always token 0), cut a (p x p x 5) patch around each, encode each patch
    with a small CNN plus Fourier features of its position relative to the centre, and let a transformer compare the
    sources (colour match and geometry of multiple images). It never sees the stamp as a whole grid, so its errors
    should differ from the CNN/ViT members. Detection is a fixed, non-learned step on the normalised pixels."""

    def __init__(self, norm="asinh", norm_a=3.0, k=10, patch=11, dim=256, depth=6, heads=8, thr=0.3,
                 drop_path_rate=0.1, nfreq=8, **_):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.norm = Normalizer(norm, a=norm_a)
        c = self.norm.out_ch
        self.k, self.p, self.thr = k, patch, thr
        self.enc = nn.Sequential(
            nn.Conv2d(c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 192, 3, 1, 1), nn.BatchNorm2d(192), nn.GELU())
        self.freq = nn.Parameter(torch.randn(2, nfreq) * 0.5, requires_grad=False)
        self.proj = nn.Linear(192 * 2 + 4 * nfreq + 3, dim)
        self.is_centre = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.is_centre, std=0.02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[Block(dim, heads, 4.0, qkv_bias=True, drop_path=dpr[i]) for i in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    @torch.no_grad()
    def detect(self, xn):
        B, _, H, Wd = xn.shape
        det = F.avg_pool2d(xn[:, :4].mean(1, keepdim=True), 3, 1, 1)               # g r i z, 3x3 smoothed
        peak = (det == F.max_pool2d(det, 5, 1, 2)) & (det > self.thr)
        score = torch.where(peak, det, torch.full_like(det, -1e4)).flatten(1)
        cidx = (H // 2) * Wd + Wd // 2
        score[:, cidx] = -1e4                                                       # centre added separately
        top = score.topk(self.k - 1, 1)
        idx = torch.cat([torch.full((B, 1), cidx, device=xn.device, dtype=torch.long), top.indices], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=xn.device), top.values > -1e3], 1)
        return idx, valid, det.flatten(1).gather(1, idx)

    def forward(self, x):
        xn = self.norm(x)
        B, C, H, Wd = xn.shape
        idx, valid, dval = self.detect(xn)
        r = self.p // 2
        cols = F.unfold(F.pad(xn, (r, r, r, r)), self.p)                            # (B, C*p*p, H*W)
        pt = cols.gather(2, idx[:, None, :].expand(-1, cols.shape[1], -1))           # (B, C*p*p, k)
        pt = pt.transpose(1, 2).reshape(B * self.k, C, self.p, self.p)
        h = self.enc(pt)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1).reshape(B, self.k, -1)
        yy = (idx // Wd).float() - H // 2
        xx = (idx % Wd).float() - Wd // 2
        pos = torch.stack([xx, yy], -1) / 10.0                                       # ~image-separation units
        ang = pos @ self.freq
        fe = torch.cat([ang.sin(), ang.cos(), (2 * ang).sin(), (2 * ang).cos(), pos,
                        pos.norm(dim=-1, keepdim=True)], -1)
        t = self.proj(torch.cat([h, fe], -1))
        t = t + self.is_centre * (torch.arange(self.k, device=x.device) == 0).float()[None, :, None]
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), valid], 1)
        am = torch.zeros(B, 1, 1, mask.shape[1], device=x.device, dtype=t.dtype)
        am = am.masked_fill(~mask[:, None, None, :], float("-inf"))                 # padded sources are never keys
        for blk in self.blocks:
            t = blk(t, attn_mask=am)
        return self.head(self.ln(t[:, 0]))


# ---------------------------------------------------------------- p4m (D4) group-equivariant CNN
def _d4(x, g):
    """Apply D4 element g in 0..7 (f = g // 4 flips W first, then r = g % 4 rot90s) to the last two dims."""
    if g // 4:
        x = x.flip(-1)
    return torch.rot90(x, g % 4, (-2, -1))


def _d4_tables():
    a = torch.arange(25.).reshape(5, 5) ** 1.37                     # asymmetric test pattern
    imgs = [_d4(a, g) for g in range(8)]
    comp = [[next(c for c in range(8) if torch.equal(_d4(imgs[b], a_), imgs[c])) for b in range(8)] for a_ in range(8)]
    inv = [next(b for b in range(8) if comp[g][b] == 0) for g in range(8)]
    return comp, inv


_D4_COMP, _D4_INV = _d4_tables()


class GConv(nn.Module):
    """D4 group convolution. lifting=True: input C plain channels; else input (C x 8) group channels.
    Output (O x 8) channels, channel index = o * 8 + g. Equivariant when stride-2 is only used on odd sizes."""

    def __init__(self, cin, cout, k=3, stride=1, lifting=False):
        super().__init__()
        self.cin, self.cout, self.k, self.stride, self.lifting = cin, cout, k, stride, lifting
        shape = (cout, cin, k, k) if lifting else (cout, cin, 8, k, k)
        fan_in = cin * k * k * (1 if lifting else 8)
        self.weight = nn.Parameter(torch.randn(shape) * (2.0 / fan_in) ** 0.5)
        if not lifting:   # perm[g][g'] = index of g^-1 * g'
            self.register_buffer("perm", torch.tensor([[_D4_COMP[_D4_INV[g]][h] for h in range(8)] for g in range(8)]),
                                 persistent=False)

    def full_weight(self):
        ws = []
        for g in range(8):
            w = self.weight if self.lifting else self.weight[:, :, self.perm[g]]
            ws.append(_d4(w, g))
        w = torch.stack(ws, 1)                                        # (O, 8, Cin[, 8], k, k)
        return w.reshape(self.cout * 8, -1, self.k, self.k)

    def forward(self, x):
        return F.conv2d(x, self.full_weight(), stride=self.stride, padding=self.k // 2)


class GBN(nn.Module):
    """BatchNorm sharing statistics across the 8 group copies of a channel (keeps equivariance)."""

    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm3d(c)

    def forward(self, x):
        B, CG, H, W = x.shape
        return self.bn(x.view(B, CG // 8, 8, H, W)).view(B, CG, H, W)


class P4MCNN(nn.Module):
    """LensCNN layout with D4 group convolutions: exactly invariant to the 8 dihedral transforms of the stamp
    (rotation about the central pixel 20,20), so no symmetry has to be learned and no TTA is needed.
    Stages at 41/21/11 px (nblk layers each) + 2 layers at 6 px; stride 2 only on odd sizes."""

    def __init__(self, norm="asinh", norm_a=3.0, width=16, nblk=3, drop_rate=0.2, **_):
        super().__init__()
        self.norm = Normalizer(norm, a=norm_a)
        w = width
        layers, cin, lifting = [], self.norm.out_ch, True
        for st, cout in enumerate((w, 2 * w, 4 * w)):
            for k in range(nblk):
                layers += [GConv(cin, cout, 3, 2 if (k == 0 and st > 0) else 1, lifting), GBN(cout), nn.GELU()]
                cin, lifting = cout, False
        layers += [GConv(cin, 8 * w, 3, 2), GBN(8 * w), nn.GELU(), GConv(8 * w, 8 * w), GBN(8 * w), nn.GELU()]
        self.f = nn.Sequential(*layers)
        self.c = 8 * w
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(4 * self.c, 1))

    def forward(self, x):
        h = self.f(self.norm(x))
        B, CG, H, W = h.shape
        h = h.view(B, self.c, 8, H * W)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3)), h.mean(3).amax(2), h.amax(3).mean(2)], 1)  # group+space invariant
        return self.head(h)


# ---------------------------------------------------------------- AION-1 (Polymathic AI) fine-tuning
_AION_CODEC = {}


def _aion_codec(device):
    """Frozen HSC image tokenizer, kept OUT of the module tree (not in optimizer / EMA / checkpoints)."""
    key = str(device)
    if key not in _AION_CODEC:
        from aion.codecs import CodecManager
        from aion.codecs.preprocessing.image import RescaleToLegacySurvey
        from aion.modalities import HSCImage
        # polymathic-aion 0.0.2 bug: backward() calls a missing method; only used by decode (diagnostics)
        RescaleToLegacySurvey.backward = lambda self, img, survey: img * (
            self.convert_zeropoint(27.0) if survey == "HSC" else 1.0)
        codec = CodecManager(device="cpu")._load_codec(HSCImage).to(device).eval().requires_grad_(False)
        _AION_CODEC[key] = codec
    return _AION_CODEC[key]


def _aion_sdpa_patch():
    """Replace AION's explicit-softmax self-attention by SDPA (same maths, no B*H*N*N tensor kept).
    Only for mask=None and plain softmax; otherwise falls back to the original forward."""
    from aion.fourm import fm_utils
    for cls in (fm_utils.Attention, fm_utils.NormAttention):
        if getattr(cls, "_sdpa", False):
            continue
        orig = cls.forward

        def fwd(self, x, mask=None, _orig=orig):
            if mask is not None or self.allow_zero_attn:
                return _orig(self, x, mask)
            B, N, C = x.shape
            q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
            if hasattr(self, "q_norm"):
                q, k = self.q_norm(q), self.k_norm(k)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
                                               scale=self.scale)
            return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))
        cls.forward = fwd
        cls._sdpa = True


class AIONNet(nn.Module):
    """AION-1 encoder fine-tuned as a classifier.

    Input raw stamps (B,5,41,41) at 0.2"/px -> flux-conserving resample to HSC 0.168"/px (49 px) -> zero-pad to the
    codec's 96x96 -> frozen HSC image codec (MagViT + FSQ, 576 discrete tokens on a 24x24 grid) -> AION encoder
    (trainable) -> mean of the tokens covering the stamp (+ all-token mean) -> linear head.
    Bands g r i z y = HSC-G..Y. `flux_scale` multiplies the raw values (units assumed ~HSC zp27).
    """

    BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

    def __init__(self, variant="aion-base", flux_scale=1.0, drop_rate=0.0, canvas="zero", **_):
        super().__init__()
        from aion import AION
        _aion_sdpa_patch()
        m = AION.from_pretrained(f"polymathic-ai/{variant}")
        self.backbone = nn.ModuleDict({"emb": m.encoder_embeddings["tok_image_hsc"],
                                       "blocks": m.encoder, "norm": m.encoder_norm})
        self._fm = [m]                                            # helper methods only (not registered)
        self.flux_scale = flux_scale
        D = m.encoder_norm.weight.shape[0]
        # canvas: "zero" = 49-px stamp zero-padded to 96 (original); "noise" = stamp embedded in an
        # 81-px raw canvas of white sky noise at the stamp's own per-band sigma, then resampled to 96 px at 0.168"/px;
        # "magnify" = stamp resampled straight to 96 px (x1.97 magnification, flux-conserving), so each 4-px token
        # covers ~1.7 stamp px instead of ~3.4 and there is no artificial border.
        assert canvas in ("zero", "noise", "magnify")
        self.canvas = canvas
        g = torch.zeros(24, 24, dtype=torch.bool)
        if canvas == "magnify":
            g[6:18, 6:18] = True                                  # central 12x12 tokens = central ~20 stamp px
        else:
            g[5:19, 5:19] = True                                  # tokens covering the 49-px stamp (+margin)
        self.register_buffer("center", g.flatten(), persistent=False)
        self.head = nn.Sequential(nn.LayerNorm(2 * D), nn.Dropout(drop_rate), nn.Linear(2 * D, 1))

    @torch.no_grad()
    def tokenize(self, raw):
        from aion.modalities import HSCImage
        with torch.autocast("cuda", enabled=False):
            x = self.make_canvas(raw)
            codec = _aion_codec(raw.device)
            return codec.encode(HSCImage(flux=x, bands=self.BANDS)).long()   # (B, 576)

    @torch.no_grad()
    def make_canvas(self, raw):
        """Raw (B,5,41,41) stamps -> (B,5,96,96) HSC-like flux images at 0.168"/px (see `canvas`)."""
        with torch.autocast("cuda", enabled=False):
            x = raw.float() * self.flux_scale
            if self.canvas == "zero":
                x = F.interpolate(x, size=(49, 49), mode="bilinear", align_corners=False) * (41 / 49) ** 2
                x = F.pad(x, (23, 24, 23, 24)).contiguous()
            elif self.canvas == "magnify":
                x = (F.interpolate(x, size=(96, 96), mode="bilinear", align_corners=False) * (41 / 96) ** 2).contiguous()
            else:
                B, C, H, Wd = x.shape
                yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(Wd, device=x.device), indexing="ij")
                ring = ((yy - H // 2) ** 2 + (xx - Wd // 2) ** 2) >= 15 ** 2
                v = x[:, :, ring]                                           # (B, C, K) outer-annulus pixels
                med = v.median(-1, keepdim=True).values
                sig = 1.4826 * (v - med).abs().median(-1).values           # robust per-stamp, per-band sigma
                cv = med[..., None] + torch.randn(B, C, 81, 81, device=x.device) * sig[..., None, None]  # sky level + noise
                cv[:, :, 20:61, 20:61] = x
                x = (F.interpolate(cv, size=(96, 96), mode="bilinear", align_corners=False) * (81 / 96) ** 2).contiguous()
            return x

    def forward(self, raw):
        tok = self.tokenize(raw)
        e = self.backbone["emb"]({"tensor": tok, "input_mask": torch.zeros_like(tok, dtype=torch.bool)})
        x = e["x"] + e["emb"]
        for blk in self.backbone["blocks"]:
            x = blk(x, mask=None)
        x = self.backbone["norm"](x)
        h = torch.cat([x[:, self.center].mean(1), x.mean(1)], 1)
        return self.head(h)


class DualNet(nn.Module):
    """Two-branch hybrid: a native-resolution ViT (3-px patches, input 41->42) and a native-stem
    ConvNeXt read the same normalised stamp. Each branch is a full `Net` (its timm head = an auxiliary 1-logit head);
    the fused head is LayerNorm(concat(features)) -> Linear -> 1 logit.
    - init: each branch from its own checkpoint: a fine-tuned Net state_dict (keys 'backbone.*'/'norm.*') or a
      pretrained encoder (timm backbone state_dict, head skipped).
    - aux_w: weight of the two branch losses added to the fused loss (train.py reads `aux_logits`), so each branch
      stays a good classifier on its own (keeps the ensemble-like diversity).
    - branch_drop: during training, with this probability one branch's features (chosen at random, per sample) are
      zeroed in the fused head (x2 on the other), so the fusion cannot lean on one branch only.
    - out: "fuse" (fused head), "mean3" (mean of fused + 2 branch logits)."""

    def __init__(self, vit=None, cnn=None, vit_ckpt=None, cnn_ckpt=None, aux_w=0.5, branch_drop=0.2, out="fuse",
                 norm="asinh", norm_a=3.0, stem=None, pretrained=False):
        super().__init__()                                  # stem/pretrained: train.py defaults, set per branch
        vk = {"arch": "vit_base_patch16_224", "stem": "plain", "size": 42, "img_size": 42, "patch_size": 3,
              "drop_path_rate": 0.3, **(vit or {})}
        ck = {"arch": "convnextv2_base.fcmae", "stem": "native", "drop_path_rate": 0.3, **(cnn or {})}
        self.backbone = nn.ModuleDict({"vit": Net(norm=norm, norm_a=norm_a, **vk),
                                       "cnn": Net(norm=norm, norm_a=norm_a, **ck)})
        for name, path in (("vit", vit_ckpt), ("cnn", cnn_ckpt)):
            if path:
                sd = torch.load(path, map_location="cpu")
                br = self.backbone[name]
                if any(k.startswith("backbone.") for k in sd):
                    r = br.load_state_dict(sd, strict=False)          # fine-tuned Net (incl. its head)
                else:
                    r = br.backbone.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head")},
                                                    strict=False)
                print(f"dual {name} <- {path}: missing {r.missing_keys} unexpected {r.unexpected_keys}", flush=True)
        self.dims = [self.backbone[n].backbone.num_features for n in ("vit", "cnn")]
        self.fuse = nn.Sequential(nn.LayerNorm(sum(self.dims)), nn.Dropout(0.1), nn.Linear(sum(self.dims), 1))
        self.aux_w, self.branch_drop, self.out = aux_w, branch_drop, out
        self.aux_logits = None

    def _branch(self, name, x):
        net = self.backbone[name]
        h = net.norm(x)
        if net.size:
            h = F.interpolate(h, size=(net.size, net.size), mode=net.resize_mode, align_corners=False)
        bb = net.backbone
        f = bb.forward_head(bb.forward_features(h), pre_logits=True)
        return f, bb.get_classifier()(f)

    def forward(self, x):
        fv, zv = self._branch("vit", x)
        fc, zc = self._branch("cnn", x)
        if self.training and self.branch_drop > 0:
            B = x.shape[0]
            drop = torch.rand(B, device=x.device) < self.branch_drop
            which = torch.rand(B, device=x.device) < 0.5
            mv = torch.where(drop & which, 0.0, torch.where(drop, 2.0, 1.0))[:, None].to(fv.dtype)
            mc = torch.where(drop & ~which, 0.0, torch.where(drop, 2.0, 1.0))[:, None].to(fc.dtype)
            fv, fc = fv * mv, fc * mc
        z = self.fuse(torch.cat([fv, fc], 1))
        if self.out == "mean3":
            z = (z + zv + zc) / 3
        self.aux_logits = (zv, zc) if self.training else None
        return z


def build_model(cfg):
    m = dict(cfg["model"])
    arch = m.pop("arch")
    init_ckpt = m.pop("init_ckpt", None)
    if arch == "lenscnn":
        model = LensCNN(**m)
    elif arch == "bandvit":
        model = BandTokenViT(**m)
    elif arch == "p4mcnn":
        model = P4MCNN(**m)
    elif arch == "aion":
        model = AIONNet(**m)
    elif arch == "dual":
        model = DualNet(**m)
    elif arch == "pointset":
        model = PointSetNet(**m)
    else:
        model = Net(arch=arch, **m)
    if init_ckpt:
        # e.g. MAE-pretrained encoder saved as a timm state_dict for model.backbone (head excluded)
        sd = torch.load(init_ckpt, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("head")}
        target = getattr(model, "backbone", model)         # LensCNN: whole model (encoder = .f)
        missing, unexpected = target.load_state_dict(sd, strict=False)
        print(f"init_ckpt {init_ckpt}: missing {missing} unexpected {unexpected}", flush=True)
    return model
