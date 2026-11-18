"""Minimal SIE + external shear lens model in numpy (no lenstronomy in the environment; no installs allowed).

Deflection of the singular isothermal ellipsoid in its own frame (major axis along x', axis ratio q < 1), Keeton (2001)
with core s = 0:
    psi = sqrt(q^2 x'^2 + y'^2)
    ax' = b / sqrt(1-q^2) * arctan (sqrt(1-q^2) x' / psi)
    ay' = b / sqrt(1-q^2) * arctanh(sqrt(1-q^2) y' / psi)
b is the normalisation; lenstronomy's convention is b = theta_E * sqrt((1+q^2)/(2q)) ("NORM" variants below).
External shear: ax = g1 x + g2 y, ay = g2 x - g1 y.

The convention variants (ellipticity definition, angle origin/sign, normalisation) are parameters so that
verify_conventions.py can test which one reproduces the organizers' tabulated image positions.
"""
import numpy as np


def q_from_ell(ell, ell_def):
    ell = np.asarray(ell, float)
    if ell_def == "e=(1-q)/(1+q)":
        return (1 - ell) / (1 + ell)
    if ell_def == "1-q":
        return 1 - ell
    if ell_def == "(1-q2)/(1+q2)":
        return np.sqrt((1 - ell) / (1 + ell))
    raise ValueError(ell_def)


def angle_rad(pa_deg, pa_def):
    pa = np.deg2rad(np.asarray(pa_deg, float))
    return {"phi": pa, "-phi": -pa, "90-phi": np.pi / 2 - pa, "phi+90": pa + np.pi / 2}[pa_def]


def norm_b(theta_e, q, norm):
    if norm == "lenstronomy":
        return theta_e * np.sqrt((1 + q ** 2) / (2 * q))
    if norm == "sqrt_q":
        return theta_e * np.sqrt(q)
    if norm == "plain":
        return theta_e * np.ones_like(q)
    raise ValueError(norm)


def alpha(x, y, theta_e, q, phi, g1, g2, norm="lenstronomy"):
    """Deflection at (x, y) (arrays broadcastable with the per-lens parameters)."""
    q = np.clip(q, 1e-4, 0.99999)
    b = norm_b(theta_e, q, norm)
    c, s = np.cos(phi), np.sin(phi)
    xp, yp = c * x + s * y, -s * x + c * y
    psi = np.sqrt(q ** 2 * xp ** 2 + yp ** 2) + 1e-12
    f = np.sqrt(1 - q ** 2)
    axp = b / f * np.arctan(f * xp / psi)
    ayp = b / f * np.arctanh(np.clip(f * yp / psi, -1 + 1e-12, 1 - 1e-12))
    ax, ay = c * axp - s * ayp, s * axp + c * ayp
    return ax + g1 * x + g2 * y, ay + g2 * x - g1 * y


def source_of(x, y, *a, **k):
    ax, ay = alpha(x, y, *a, **k)
    return x - ax, y - ay


def count_images(bx, by, theta_e, q, phi, g1, g2, norm="lenstronomy", n=401, half=None):
    """Number of images of one point source (scalars) by triangle ray-shooting on an n x n grid."""
    half = half or 3.0 * theta_e + 0.5
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    SX, SY = source_of(X, Y, theta_e, q, phi, g1, g2, norm=norm)
    SX, SY = SX - bx, SY - by
    cnt = 0
    for tri in (((0, 0), (0, 1), (1, 0)), ((1, 1), (0, 1), (1, 0))):
        P = [(SX[i:n - 1 + i, j:n - 1 + j], SY[i:n - 1 + i, j:n - 1 + j]) for i, j in tri]
        (x1, y1), (x2, y2), (x3, y3) = P
        d1 = (0 - x2) * (y1 - y2) - (x1 - x2) * (0 - y2)
        d2 = (0 - x3) * (y2 - y3) - (x2 - x3) * (0 - y3)
        d3 = (0 - x1) * (y3 - y1) - (x3 - x1) * (0 - y1)
        neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        cnt += int((~(neg & pos)).sum())
    return cnt
