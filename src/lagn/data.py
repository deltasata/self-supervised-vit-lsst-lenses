"""Data loading, normalisation and GPU augmentation.

Arrays (see data/MANIFEST.md): data/X_trainval.npy (N,5,41,41) float32 raw values, bands g r i z y;
y_trainval.npy int8; keys_trainval.parquet (obj, simset, pipeline, id, split in {train,val}).
Models see pixel values only.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

W = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(W, "data")


def load_trainval(mmap=True):
    X = np.load(os.path.join(DATA, "X_trainval.npy"), mmap_mode="r" if mmap else None)
    y = np.load(os.path.join(DATA, "y_trainval.npy"))
    keys = pd.read_parquet(os.path.join(DATA, "keys_trainval.parquet"))
    assert len(X) == len(y) == len(keys)
    return X, y, keys


def kfold_assign(keys, n_folds=10, seed=20261022):
    """Fold id (0..n_folds-1) for every train+val row (row order of load_trainval). Stratified by simset x 5,000-ID
    block (as the holdout split); the original train/val split is ignored. Deterministic."""
    blk = keys.simset.astype(str) + "_" + (keys["id"].astype(int) // 5000).astype(str)
    rng = np.random.RandomState(seed)
    fold = np.empty(len(keys), np.int64)
    for g in sorted(blk.unique()):
        idx = np.where(blk.values == g)[0]
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = (np.arange(len(idx)) + rng.randint(n_folds)) % n_folds
    return fold


def norm_stats():
    with open(os.path.join(DATA, "norm_stats.json")) as f:
        return json.load(f)


class Normalizer(torch.nn.Module):
    """Fixed per-band normalisation (global constants fitted on train; preserves colour).

    kinds:
      asinh   : asinh(x / (a * sigma_b)) / s     (linear within ~a*sigma, log beyond; handles 1e5 range)
      linear  : x / sigma_b, clipped to +-clip
      asinh_ch: asinh with two channels per band (soft a=1 and hard a=50) -> 10 channels
      lsrgb   : AstroDINO / AstroCLIP input: zp27 flux -> nanomaggies per 0.262" Legacy Survey pixel
                (surface brightness kept), then the Legacy Survey viewer stretch used in their pretraining
                (decals_to_rgb: channels (z, r, g) * (2.2, 3.4, 6.0) + 0.03, arcsinh(20 I)/sqrt(20)/I, clip [0, 1]);
                i and y get the same per-pixel stretch factor (scale 1) as extra channels 3, 4 -> 5 channels
    sigma_b = global median background noise per band (train set).
    """

    def __init__(self, kind="asinh", a=3.0, clip=1e3, sigma=None):
        super().__init__()
        sigma = sigma if sigma is not None else norm_stats()["sigma"]
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32).view(1, -1, 1, 1))
        self.kind, self.a, self.clip = kind, a, clip
        self.out_ch = 10 if kind == "asinh_ch" else 5

    def forward(self, x):
        z = x / self.sigma
        if self.kind == "asinh":
            return torch.asinh(z / self.a)
        if self.kind == "linear":
            return z.clamp(-self.clip, self.clip) / 10
        if self.kind == "asinh_ch":
            return torch.cat([torch.asinh(z / 1.0) / 3, torch.asinh(z / 50.0)], 1)
        if self.kind == "lsrgb":
            v = x.float() * ((0.262 / 0.2) ** 2 / 10 ** (4.5 / 2.5))
            g, r, i, zb, y = v.unbind(1)
            ch = [zb * 2.2 + 0.03, r * 3.4 + 0.03, g * 6.0 + 0.03]
            inten = sum(c.clamp(min=0) for c in ch) / 3
            k = torch.asinh(20 * inten) / 20 ** 0.5 / (inten + (inten == 0) * 1e-6)
            out = torch.stack(ch + [i + 0.03, y + 0.03], 1) * k[:, None]
            return out.clamp(0, 1).to(x.dtype)
        raise ValueError(self.kind)


def dihedral(x, k):
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def random_dihedral(x):
    """Independent random dihedral transform per sample (GPU)."""
    k = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for t in range(8):
        m = k == t
        if m.any():
            out[m] = dihedral(x[m], t)
    return out


def random_shift(x, max_shift=1):
    """Integer translation by up to +-max_shift px with reflection padding (keeps PSF/colours)."""
    if max_shift <= 0:
        return x
    n = x.shape[0]
    p = max_shift
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    dy = torch.randint(0, 2 * p + 1, (n,), device=x.device)
    dx = torch.randint(0, 2 * p + 1, (n,), device=x.device)
    H, Wd = x.shape[-2:]
    iy = (torch.arange(H, device=x.device)[None, :] + dy[:, None])  # (n,H)
    ix = (torch.arange(Wd, device=x.device)[None, :] + dx[:, None])
    xp = xp.gather(2, iy[:, None, :, None].expand(-1, x.shape[1], -1, xp.shape[-1]))
    xp = xp.gather(3, ix[:, None, None, :].expand(-1, x.shape[1], H, -1))
    return xp


def tta_predict(model, x, n=8):
    """Average logits over the first n dihedral transforms."""
    out = 0
    for t in range(n):
        out = out + model(dihedral(x, t)).float()
    return out / n


def random_subpixel_shift(x, max_shift=0.5):
    """Random sub-pixel translation per sample via Fourier phase shift (band-consistent: same shift for all
    bands; preserves the PSF up to the periodic boundary). Used to remove sub-pixel-centring shortcuts."""
    if max_shift <= 0:
        return x
    n, c, h, w = x.shape
    d = (torch.rand(n, 2, device=x.device) * 2 - 1) * max_shift
    fy = torch.fft.fftfreq(h, device=x.device)
    fx = torch.fft.rfftfreq(w, device=x.device)
    ph = torch.exp(-2j * torch.pi * (d[:, 0, None, None] * fy[None, :, None] + d[:, 1, None, None] * fx[None, None, :]))
    X = torch.fft.rfft2(x.float())
    return torch.fft.irfft2(X * ph[:, None], s=(h, w)).to(x.dtype)


def random_correlated_noise(x, sigma, p=0.5, amp=(0.3, 1.0), smooth=(0.0, 1.2)):
    """Add Gaussian noise with random amplitude (in units of the per-band global sigma) and random spatial
    correlation (Gaussian smoothing width in px), to a fraction p of the samples. Randomises the noise
    texture so the SIMCT noise-correlation shortcut is not usable."""
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < p
    if not sel.any():
        return x
    k = int(sel.sum())
    z = torch.randn(k, x.shape[1], x.shape[2], x.shape[3], device=x.device)
    s = torch.empty(k, device=x.device).uniform_(*smooth)
    # Gaussian smoothing in Fourier space (periodic; avoids MIOpen grouped conv, which gave NaN on ROCm)
    h, w = x.shape[2], x.shape[3]
    fy = torch.fft.fftfreq(h, device=x.device)
    fx = torch.fft.rfftfreq(w, device=x.device)
    f2 = fy[:, None] ** 2 + fx[None, :] ** 2
    filt = torch.exp(-2 * torch.pi ** 2 * s[:, None, None] ** 2 * f2[None])          # (k, h, w//2+1)
    zz = torch.fft.irfft2(torch.fft.rfft2(z) * filt[:, None], s=(h, w))
    zz = zz / zz.flatten(2).std(2)[..., None, None].clamp(min=1e-6)
    a = torch.empty(k, 1, 1, 1, device=x.device).uniform_(*amp)
    sig = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    out = x.clone()
    out[sel] = x[sel] + a * sig * zz
    return out


def random_dimming(x, sigma, p=0.3, smin=0.3):
    """Dimming augmentation: for a fraction p of the samples, scale the stamp by s ~ U_log(smin, 1)
    and add white Gaussian noise with per-band std sigma*sqrt(1-s^2), so the total noise level stays the same.
    The result is the same object seen fainter, at the same depth. It adds many new faint examples (the hard
    regime) and weakens the brightness shortcut. Labels are unchanged."""
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < p
    if not sel.any():
        return x
    k = int(sel.sum())
    s = torch.exp(torch.empty(k, 1, 1, 1, device=x.device).uniform_(float(torch.log(torch.tensor(smin))), 0.0))
    sig = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    z = torch.randn(k, *x.shape[1:], device=x.device, dtype=x.dtype)
    out = x.clone()
    out[sel] = s * x[sel] + sig * torch.sqrt(1 - s ** 2) * z
    return out


def random_psf_blur(x, sigma, p=0.5, fmax=1.25, fwhm0=4.2):
    """PSF-width augmentation: for a fraction p of the samples, widen a ~fwhm0 px Gaussian PSF by a
    factor f ~ U(1, fmax) (extra Gaussian blur, Fourier space, periodic), then add white noise so the per-band noise
    sigma is restored (blur lowers white noise by the kernel's L2 norm). The robustness test (results/robustness)
    showed the models are fragile to a 10-20 % wider PSF. Labels unchanged."""
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < p
    if not sel.any():
        return x
    k = int(sel.sum())
    f = torch.empty(k, device=x.device).uniform_(1.0, fmax)
    se = (fwhm0 / 2.3548) * torch.sqrt((f ** 2 - 1).clamp(min=0))                     # extra blur sigma, px
    h, w = x.shape[2], x.shape[3]
    fy = torch.fft.fftfreq(h, device=x.device)
    fx = torch.fft.rfftfreq(w, device=x.device)
    f2 = fy[:, None] ** 2 + fx[None, :] ** 2
    filt = torch.exp(-2 * torch.pi ** 2 * se[:, None, None] ** 2 * f2[None])          # (k, h, w//2+1)
    b = torch.fft.irfft2(torch.fft.rfft2(x[sel].float()) * filt[:, None], s=(h, w))
    # white-noise std after the blur = sqrt(mean |filt|^2) over the full spectrum (rfft: weight the inner columns x2)
    wgt = torch.full((fx.shape[0],), 2.0, device=x.device)
    wgt[0] = 1.0
    if w % 2 == 0:
        wgt[-1] = 1.0
    keep = torch.sqrt((filt ** 2 * wgt).sum((1, 2)) / (h * w))                       # (k,)
    sig = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).view(1, -1, 1, 1)
    z = torch.randn(k, *x.shape[1:], device=x.device)
    out = x.clone()
    out[sel] = (b + sig * torch.sqrt((1 - keep ** 2).clamp(min=0))[:, None, None, None] * z).to(x.dtype)
    return out


def random_sky_offset(x, sigma, p=0.5, amax=0.5):
    """Sky-level augmentation: for a fraction p of the samples, add a constant per-band offset
    U(-amax, amax) * sigma_b (imperfect sky subtraction). The robustness test showed a +0.5 sigma offset raises the
    SLSim false-positive rate from 0.1 % to 5 % (the models use the SLSim sky level). Labels unchanged."""
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < p
    if not sel.any():
        return x
    k = int(sel.sum())
    sig = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    a = (torch.rand(k, x.shape[1], 1, 1, device=x.device, dtype=x.dtype) * 2 - 1) * amax
    out = x.clone()
    out[sel] = x[sel] + a * sig
    return out


def mixup_flux(x, y, sigma, p=0.5, alpha=0.4):
    """Mixup on raw-flux stamps for a fraction p of the batch: x' = lam*x + (1-lam)*x[perm] + extra noise so that
    the sky noise stays sigma_b (mixing two noisy stamps lowers it by sqrt(lam^2+(1-lam)^2)); y' = lam*y + (1-lam)*y[perm]."""
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < p
    if not sel.any():
        return x, y
    perm = torch.randperm(n, device=x.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(x.device)
    lam = torch.where(sel, lam, torch.ones_like(lam))
    l4 = lam.view(-1, 1, 1, 1).to(x.dtype)
    sig = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    extra = torch.sqrt(torch.clamp(1 - l4 ** 2 - (1 - l4) ** 2, min=0.0))
    xm = l4 * x + (1 - l4) * x[perm] + sig * extra * torch.randn_like(x)
    ym = lam * y + (1 - lam) * y[perm]
    return xm, ym


def _shear_rows(x, a):
    """Shift row i of every (H, W) image by a * (i - c) px along W (Fourier phase shift, periodic). a: (n,)."""
    n, c, h, w = x.shape
    i = torch.arange(h, device=x.device, dtype=torch.float32) - (h - 1) / 2
    fx = torch.fft.rfftfreq(w, device=x.device)
    s = a[:, None] * i[None, :]                                          # (n, h) shift per row
    ph = torch.exp(-2j * torch.pi * s[:, :, None] * fx[None, None, :])   # (n, h, w//2+1)
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * ph[:, None], n=w, dim=-1)


def random_rotation(x, p=0.5, max_deg=45.0):
    """Rotate a fraction p of the stamps by a random angle in [-max_deg, max_deg] (same for all bands) with the
    three-shear method in Fourier space (Unser et al. 1995): every step is a pure phase shift, so flux and the
    white-noise power spectrum are preserved (no interpolation smoothing). Periodic boundaries: the stamp corners
    wrap, which only moves sky pixels for a centred object. With the 8 dihedral transforms this covers all angles."""
    n = x.shape[0]
    sel = torch.rand(n, device=x.device) < p
    if not sel.any():
        return x
    th = (torch.rand(int(sel.sum()), device=x.device) * 2 - 1) * max_deg * torch.pi / 180
    a, b = -torch.tan(th / 2), torch.sin(th)
    z = x[sel].float()
    z = _shear_rows(z, a)                                                # shear along x
    z = _shear_rows(z.transpose(-1, -2), b).transpose(-1, -2)            # shear along y
    z = _shear_rows(z, a)                                                # shear along x
    x = x.clone()
    x[sel] = z.to(x.dtype)
    return x
