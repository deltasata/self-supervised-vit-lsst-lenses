"""Image-count test of truth-table conventions: solve SIE+shear for the tabulated source and compare the number of
images with the tabulated num_images / image_count. Usage: python regression/scripts/check_counts.py D1_L|D2_L|D2_N N"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lensmodel as L  # noqa: E402
from truth_io import load  # noqa: E402

s, n = sys.argv[1], int(sys.argv[2])
t = load(s).sample(n, random_state=0)
a = lambda c: np.asarray(t[c], float)
cnt_true = a("num_images") if "num_images" in t else a("image_count")
if s == "D1_L":
    srcs = {"arcsec": (a("srcx"), a("srcy"))}
    pas, spas = ["90-phi", "-phi"], ["90-phi"]
else:
    srcs = {"arcsec": (a("source_x"), a("source_y")), "pixels/5": (a("source_x") * 0.2, a("source_y") * 0.2)}
    pas, spas = ["-phi", "90-phi"], ["-phi", "90-phi"]
for (sn, (bx, by)), ed, pa, spa, nm in itertools.product(srcs.items(), ["e=(1-q)/(1+q)", "1-q"], pas, spas,
                                                        ["lenstronomy"]):
    q = L.q_from_ell(a("ell_m"), ed)
    ph, sp = L.angle_rad(a("ell_m_PA"), pa), L.angle_rad(a("sh_PA"), spa)
    g1, g2 = a("sh") * np.cos(2 * sp), a("sh") * np.sin(2 * sp)
    c = np.array([L.count_images(bx[i], by[i], a("Rein")[i], q[i], ph[i], g1[i], g2[i], norm=nm, n=301)
                  for i in range(n)]) - 1   # triangle counting also catches the singular centre
    tab = {int(k): {int(v): int(((cnt_true == k) & (c == v)).sum()) for v in np.unique(c)} for k in np.unique(cnt_true)}
    print(f"{s} src={sn:8s} ell={ed:14s} pa={pa:7s} shpa={spa:7s} agree={np.mean(c == cnt_true):.3f}  true->found {tab}",
          flush=True)
