"""V1 self-test of the estimators on synthetic signals with a known sign convention.

1. b(t) = a(t - tau), tau = -1.10 samples (B hears first / B sampled later): every estimator must return -1.10.
2. Diffuse-like pair (3-D sinc coherence, d = 1.25 cm) plus the -1.10 skew, 12 s, Welch 1024: phase slope and a
   sinc fit to |coherence| must return the truth (checks windowing / finite-average bias of the d and skew estimates).
3. coh_axisym(m=0) == sinc, coh_axisym(m=-0.5) == J0.
"""
import json
import os
import sys

import numpy as np
from scipy.special import j0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import C, FS, HERE, coh_axisym, delay_ls, gcc_phat, kd, xcorr_delay, xspec  # noqa: E402

rng = np.random.default_rng(7)
out = {}
N = 12 * FS
tau = -1.10


def frac_delay(x, t):
    n = 1 << int(np.ceil(np.log2(len(x) + 64)))
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x, n) * np.exp(-2j * np.pi * f * t), n)[:len(x)]


a = rng.standard_normal(N)
b = frac_delay(a, tau)
x = np.stack([a, b], 1)[1000:-1000]
f, Pab, Paa, Pbb, n, fr = xspec(x)
out["pure_delay"] = {"delay_ls_300_3000": delay_ls(f, Pab, Paa, Pbb, n, 300, 3000)[0],
                     "xcorr": xcorr_delay(x, 300, 3000), "gcc_phat": gcc_phat(x[:4000, 0], x[:4000, 1])}

# 2. synthetic diffuse field: shape white noise in the frequency domain with coherence gamma(f) = sinc * exp(j w tau)
d = 0.0125
M = 1 << 18
fM = np.fft.rfftfreq(M, 1 / FS)
gam = np.sinc(2 * fM * d / C) * np.exp(2j * np.pi * fM * tau / FS)  # target E{A conj B}/... = gamma
res2 = []
for rep in range(3):
    U = (rng.standard_normal(len(fM)) + 1j * rng.standard_normal(len(fM)))
    V = (rng.standard_normal(len(fM)) + 1j * rng.standard_normal(len(fM)))
    A = U
    # B = conj(gamma) U + sqrt(1-|gamma|^2) V  ->  E{A conj B} = gamma
    B = np.conj(gam) * U + np.sqrt(np.clip(1 - np.abs(gam) ** 2, 0, 1)) * V
    xa, xb = np.fft.irfft(A, M), np.fft.irfft(B, M)
    xx = np.stack([xa, xb], 1)[:N]
    f, Pab, Paa, Pbb, n, fr = xspec(xx)
    t300 = delay_ls(f, Pab, Paa, Pbb, n, 300, 3000)[0]
    g = np.abs(Pab) / np.sqrt(Paa * Pbb)
    m = (f >= 1000) & (f <= 6500)
    grid = np.arange(0.008, 0.02, 0.00005)
    errs = [np.sum((g[m] - np.abs(np.sinc(2 * f[m] * dd / C))) ** 2) for dd in grid]
    res2.append({"slope": round(t300, 4), "d_fit_cm": round(100 * grid[int(np.argmin(errs))], 3)})
out["synthetic_diffuse_d1.25_tau-1.10"] = res2

xx = kd(np.linspace(100, 7000, 50), 0.0125)
out["axisym_m0_vs_sinc_maxerr"] = float(np.max(np.abs(coh_axisym(xx, 0) - np.sinc(xx / np.pi))))
out["axisym_m-0.5_vs_J0_maxerr"] = float(np.max(np.abs(coh_axisym(xx, -0.5) - j0(xx))))
print(json.dumps(out, indent=1))
with open(os.path.join(HERE, "v1_selftest.json"), "w") as fh:
    json.dump(out, fh, indent=1)
