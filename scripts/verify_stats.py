"""Per-object photometric statistics on the full canonical array (data verification).

Input : data/verify/raw_all.npy, raw_all_meta.parquet  (from verify_pack_raw.py)
Output: data/verify/obj_stats.parquet  per-object features (also used later by Baseline 0)
        results/data_checks/*.csv, *.png  summaries per simset and band
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(W, "data", "verify")
OUT = os.path.join(W, "results", "data_checks")
BANDS = "grizy"
SIMSETS = ["D1_L", "D1_N", "D2_L", "D2_N"]
os.makedirs(OUT, exist_ok=True)

X = np.load(os.path.join(V, "raw_all.npy"), mmap_mode="r")
meta = pd.read_parquet(os.path.join(V, "raw_all_meta.parquet"))
N = len(X)
yy, xx = np.mgrid[:41, :41]
rr = np.hypot(yy - 20, xx - 20)
outer = rr >= 15            # background annulus (noise estimate)
ap = {r: rr <= r for r in (3, 6, 10)}

feat = {"simset": meta.simset.values, "id": meta.id.values}
CH = 20000
cols = {}


def add(name, arr, sl):
    cols.setdefault(name, np.empty(N, np.float32))[sl] = arr


for s0 in range(0, N, CH):
    sl = slice(s0, min(N, s0 + CH))
    x = np.asarray(X[sl], dtype=np.float64)              # (n,5,41,41)
    n = len(x)
    bg = x[:, :, outer]                                  # (n,5,npix)
    med = np.median(bg, axis=2)
    mad = np.median(np.abs(bg - med[..., None]), axis=2)
    sig = 1.4826 * mad
    xs = x - med[..., None, None]                        # background-subtracted
    for b, band in enumerate(BANDS):
        add(f"bgmed_{band}", med[:, b], sl)
        add(f"mad_{band}", mad[:, b], sl)
        add(f"std_outer_{band}", bg[:, b].std(axis=1), sl)
        for r, m in ap.items():
            f = xs[:, b, m].sum(axis=1)
            add(f"ap{r}_{band}", f, sl)
            add(f"snr{r}_{band}", f / (sig[:, b] * np.sqrt(m.sum()) + 1e-12), sl)
        add(f"peak_{band}", xs[:, b].max(axis=(1, 2)), sl)
        add(f"peakc_{band}", xs[:, b, 18:23, 18:23].max(axis=(1, 2)), sl)
        add(f"min_{band}", x[:, b].min(axis=(1, 2)), sl)
        add(f"max_{band}", x[:, b].max(axis=(1, 2)), sl)
        add(f"nzero_{band}", (x[:, b] == 0).sum(axis=(1, 2)), sl)
        # flux-weighted centroid and second-moment size within r<=10, positive pixels only
        w = np.clip(xs[:, b], 0, None) * ap[10]
        ws = w.sum(axis=(1, 2)) + 1e-12
        cy = (w * yy).sum(axis=(1, 2)) / ws
        cx = (w * xx).sum(axis=(1, 2)) / ws
        r2 = (w * ((yy - cy[:, None, None]) ** 2 + (xx - cx[:, None, None]) ** 2)).sum(axis=(1, 2)) / ws
        add(f"cy_{band}", cy, sl)
        add(f"cx_{band}", cx, sl)
        add(f"rmom_{band}", np.sqrt(r2), sl)
        # half-light radius from cumulative curve of growth inside r<=10
        add(f"conc_{band}", xs[:, b, ap[3]].sum(axis=1) / (xs[:, b, ap[10]].sum(axis=1) + 1e-12), sl)
    # detection image = inverse-variance weighted g+r+i
    wts = 1 / (sig[:, :3] ** 2 + 1e-12)
    det = (xs[:, :3] * wts[..., None, None]).sum(1) / np.sqrt(wts.sum(1))[..., None, None]
    add("snr6_det", det[:, ap[6]].sum(axis=1) / np.sqrt(ap[6].sum()), sl)
    add("peak_det", det[:, 18:23, 18:23].max(axis=(1, 2)), sl)
    # number of 5-sigma peaks (local maxima) in the detection image within r<=12
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(det, size=(1, 3, 3))
    pk = (det == mx) & (det > 5) & (rr <= 12)
    add("npeaks5_det", pk.sum(axis=(1, 2)), sl)
    print(s0, flush=True)

df = pd.DataFrame(feat)
for k, v in cols.items():
    df[k] = v
df["label"] = (df.simset.str[3] == "L").astype(np.int8)
df["pipeline"] = df.simset.str[:2]
df.to_parquet(os.path.join(V, "obj_stats.parquet"))

# ---------------- summaries
q = lambda a: pd.Series(np.percentile(a, [1, 5, 25, 50, 75, 95, 99]), index=["p1", "p5", "p25", "p50", "p75", "p95", "p99"])
rows = []
for s in SIMSETS:
    d = df[df.simset == s]
    for band in BANDS:
        for c in ("mad", "std_outer", "bgmed", "ap6", "snr6", "peak", "peakc", "rmom", "conc", "max", "min", "nzero",
                  "cy", "cx"):
            r = q(d[f"{c}_{band}"]).to_dict()
            r.update(simset=s, band=band, stat=c, mean=float(d[f"{c}_{band}"].mean()),
                     max=float(d[f"{c}_{band}"].max()))
            rows.append(r)
    for c in ("snr6_det", "npeaks5_det"):
        r = q(d[c]).to_dict()
        r.update(simset=s, band="det", stat=c, mean=float(d[c].mean()), max=float(d[c].max()))
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(OUT, "stats_quantiles.csv"), index=False)

# near-empty: detection SNR in r<=6 below thresholds
emp = df.groupby("simset").apply(lambda d: pd.Series({f"frac_snr_det<{t}": (d.snr6_det < t).mean() for t in (3, 5, 10)}))
emp.to_csv(os.path.join(OUT, "near_empty.csv"))
print(emp)

# ---------------- plots
fig, axs = plt.subplots(3, 5, figsize=(22, 12))
for b, band in enumerate(BANDS):
    for s in SIMSETS:
        d = df[df.simset == s]
        axs[0, b].hist(np.log10(np.clip(d[f"ap6_{band}"], 1e-2, None)), bins=150, histtype="step", label=s, density=True)
        axs[1, b].hist(np.log10(np.clip(d[f"snr6_{band}"], 1e-2, None)), bins=150, histtype="step", label=s, density=True)
        axs[2, b].hist(d[f"mad_{band}"], bins=150, histtype="step", label=s, density=True)
    axs[0, b].set_title(f"{band}: log10 aperture flux r<=6 (clipped at 0.01)")
    axs[1, b].set_title(f"{band}: log10 SNR r<=6")
    axs[2, b].set_title(f"{band}: background MAD")
axs[0, 0].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "flux_snr_noise_per_band.png"), dpi=80)
print("done")
