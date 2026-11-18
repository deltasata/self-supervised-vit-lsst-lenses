"""Exact/near-duplicate search incl. 8 dihedral transforms, ID pairing across simsets, background reuse.

Runs on 1 GPU. Input data/verify/raw_all.npy (canonical order D1_L, D1_N, D2_L, D2_N x id 0..49999).
Two feature spaces per object (bands g, r, i concatenated; each band background-subtracted and
divided by its outer-annulus robust sigma):
  full  : all pixels                                            -> catches duplicated objects
  resid : minus the azimuthal median profile, pixels r >= 5 only -> robust to the central source;
          catches reused backgrounds / neighbours
Pixel values are asinh(z/3) of the noise-normalised z (compresses the 1e5+ dynamic range).
Each vector is mean-centred and L2-normalised; similarity = Pearson correlation.
For every object keep the top-K neighbours over all 8 dihedral transforms of the query (self excluded).
Outputs data/verify/nn_<feat>.npz and results/data_checks/dups_*.csv / .txt
"""
import os
import sys
import time

import numpy as np
import torch

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(W, "data", "verify")
OUT = os.path.join(W, "results", "data_checks")
K = 5
SIMSETS = ["D1_L", "D1_N", "D2_L", "D2_N"]
dev = os.environ.get("DEV", "cuda")
torch.backends.cuda.matmul.allow_tf32 = True

X = np.load(os.path.join(V, "raw_all.npy"), mmap_mode="r")
N = int(os.environ.get("NLIM", X.shape[0]))  # NLIM: quick test on a prefix
yy, xx = np.mgrid[:41, :41]
rr = np.hypot(yy - 20, xx - 20)
rbin = torch.tensor(np.round(rr).astype(np.int64), device=dev)  # 0..28
outer = torch.tensor(rr >= 15, device=dev)
mask5 = torch.tensor(rr >= 5, device=dev)


def features(x):
    """x: (n,3,41,41) float32 tensor on GPU -> dict of (n, d) normalised features (identity orientation)."""
    bg = x[:, :, outer]
    med = bg.median(dim=2).values
    sig = 1.4826 * (bg - med[..., None]).abs().median(dim=2).values + 1e-6
    z = torch.asinh((x - med[..., None, None]) / sig[..., None, None] / 3)  # compress 1e5+ dynamic range
    # azimuthal median profile per radius bin
    prof = torch.zeros(z.shape[0], z.shape[1], 29, device=dev)
    flat = z.flatten(2)
    rb = rbin.flatten()
    for r in range(29):
        sel = rb == r
        if sel.any():
            prof[:, :, r] = flat[:, :, sel].median(dim=2).values
    resid = z - prof[:, :, rbin]
    return {"full": z, "resid": resid * mask5}


def norm(v):
    v = v.flatten(1)
    v = v - v.mean(1, keepdim=True)
    return v / (v.norm(dim=1, keepdim=True) + 1e-6)


def dihedral(t, k):
    """k in 0..7: rotations by 90*k%4, flipped if k>=4. t: (..., H, W)."""
    if k >= 4:
        t = t.flip(-1)
    return torch.rot90(t, k % 4, dims=(-2, -1))


def main():
    t0 = time.time()
    feats = {"full": [], "resid": []}
    CH = 10000
    for s in range(0, N, CH):
        x = torch.from_numpy(np.ascontiguousarray(X[s:s + CH, :3])).to(dev)
        f = features(x)
        for k in feats:
            feats[k].append(f[k].float())
    feats = {k: torch.cat(v) for k, v in feats.items()}  # (N,3,41,41) fp32
    print("features", f"{time.time() - t0:.0f}s", flush=True)

    lines = []
    for name, F in feats.items():
        keys = norm(F.float()).to(torch.float16)                     # (N,d)
        best_v = torch.full((N, K), -2.0, device=dev)
        best_i = torch.zeros((N, K), dtype=torch.int64, device=dev)
        best_t = torch.zeros((N, K), dtype=torch.int8, device=dev)
        QC = 4096
        for q0 in range(0, N, QC):
            q1 = min(N, q0 + QC)
            cand_v, cand_i, cand_t = [], [], []
            for k in range(8):
                q = norm(dihedral(F[q0:q1].float(), k)).to(torch.float16)
                sim = q @ keys.T                                     # (QC, N)
                if True:  # exclude self under every transform (symmetric objects match their own rotations)
                    sim[torch.arange(q1 - q0, device=dev), torch.arange(q0, q1, device=dev)] = -2
                v, i = sim.float().topk(K, dim=1)
                cand_v.append(v)
                cand_i.append(i)
                cand_t.append(torch.full_like(i, k, dtype=torch.int8))
            v = torch.cat(cand_v, 1)
            i = torch.cat(cand_i, 1)
            tt = torch.cat(cand_t, 1)
            # a neighbour found under several transforms: keep the best (sort, then unique per row is
            # awkward on GPU; duplicates across transforms are rare except for symmetric images, fine)
            o = v.argsort(dim=1, descending=True)[:, :K]
            best_v[q0:q1] = v.gather(1, o)
            best_i[q0:q1] = i.gather(1, o)
            best_t[q0:q1] = tt.gather(1, o)
        print(name, "nn done", f"{time.time() - t0:.0f}s", flush=True)
        bv, bi, bt = best_v.cpu().numpy(), best_i.cpu().numpy(), best_t.cpu().numpy()
        np.savez(os.path.join(V, f"nn_{name}.npz"), v=bv, i=bi, t=bt)

        # null for this feature: random pairs
        rp = torch.randint(0, N, (2, 200000), device=dev)
        rp = rp[:, rp[0] != rp[1]]
        null = (keys[rp[0]].float() * keys[rp[1]].float()).sum(1).cpu().numpy()
        lines.append(f"== {name}: random-pair corr p50={np.median(null):.4f} p99.9={np.percentile(null, 99.9):.4f} "
                     f"max={null.max():.4f}")
        top = bv[:, 0]
        for s_i, s in enumerate(SIMSETS):
            sl = slice(s_i * 50000, (s_i + 1) * 50000)
            lines.append(f"{name} {s}: top1 corr percentiles p50={np.median(top[sl]):.3f} p90={np.percentile(top[sl], 90):.3f} "
                         f"p99={np.percentile(top[sl], 99):.3f} max={top[sl].max():.4f} | "
                         + " ".join(f">{th}:{(top[sl] > th).sum()}" for th in (0.3, 0.5, 0.8, 0.95, 0.999)))
            # where do the high-similarity neighbours live?
            hi = top[sl] > 0.5
            if hi.any():
                nb = bi[sl][hi, 0] // 50000
                lines.append(f"    neighbours of >0.5 pairs by simset: " +
                             ", ".join(f"{SIMSETS[j]}:{(nb == j).sum()}" for j in range(4)) +
                             f"; same id (other simset): {((bi[sl][hi, 0] % 50000 == np.arange(50000)[hi]) & (nb != s_i)).sum()}" +
                             f"; transforms: {np.bincount(bt[sl][hi, 0], minlength=8).tolist()}")
        # ID pairing: corr of same id across simsets (identity orientation and best over transforms)
        for a in range(4):
            for b in range(a + 1, 4):
                ia = torch.arange(a * 50000, (a + 1) * 50000, device=dev)
                ib = ia - a * 50000 + b * 50000
                c = (keys[ia].float() * keys[ib].float()).sum(1).cpu().numpy()
                lines.append(f"{name} same-id {SIMSETS[a]} vs {SIMSETS[b]}: p50={np.median(c):.4f} p99={np.percentile(c, 99):.4f} "
                             f"max={c.max():.4f} n>0.3={(c > 0.3).sum()}")
        # exact duplicates on fp16 features are not reliable; exact hashes are checked on CPU
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "dups_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
