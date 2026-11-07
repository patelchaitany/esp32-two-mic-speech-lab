"""V1: the later drift-watch ambient takes (02:05, 02:15, 02:25; fan level falling from -41 to -59 dBFS) - skew and the
3-D (sinc) isotropic-equivalent spacing with my magnitude-only fit, plus band level and |coherence| at high frequency.
Shows how strongly the 'diffuse' d estimate depends on the noise field / floor in one boot and one board position.
Output: v1_drift.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import C, FS, HERE, HW, delay_ls, load, xspec  # noqa: E402


def r(x, k=3):
    return float(np.round(x, k))


out = {}
names = [n for n in ["rec2/drift_00", "rec2/drift_01", "rec2/drift_02", "rec2/drift_03"] if os.path.exists(os.path.join(HW, n + ".wav"))]
for name in names:
    x = load(name)
    f, ab, aa, bb, n, fr = xspec(x, 1024, 512, [(FS, int(11.9 * FS))])
    tau = delay_ls(f, ab, aa, bb, n, 300, 3000, intercept=False)[0]
    rot = np.exp(-2j * np.pi * f * tau / FS)
    centers = np.arange(1000, 6501, 125.0)
    g = np.array([np.abs(np.sum((ab * rot)[(f >= c - 62.5) & (f < c + 62.5)])) /
                  np.sqrt(np.sum(aa[(f >= c - 62.5) & (f < c + 62.5)]) * np.sum(bb[(f >= c - 62.5) & (f < c + 62.5)])) for c in centers])
    scale = 2 / (FS * np.sum((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)) ** 2))
    o = {"level_dbfs": [r(10 * np.log10(np.mean(x[FS:int(11.9 * FS), c] ** 2)), 1) for c in range(2)], "tau_300_3000": r(tau)}
    for lo, hi in [(4000, 5000), (5000, 6500)]:
        m = (f >= lo) & (f < hi)
        o[f"psd_dB_per_Hz_{lo}-{hi}"] = r(10 * np.log10(np.mean(aa[m]) * scale), 1)
        mm = (centers >= lo) & (centers < hi)
        o[f"abs_coh_{lo}-{hi}"] = r(np.mean(g[mm]))
    for lo, hi in [(1000, 6500), (1000, 5000), (2000, 6500)]:
        sel = (centers >= lo) & (centers <= hi)
        best = None
        for d in np.arange(0.008, 0.026, 0.0002):
            D = np.sinc(2 * centers[sel] * d / C)
            rho = np.clip(np.sum(D * g[sel]) / np.sum(D * D), 0.85, 1.0)
            e = np.sum((g[sel] - rho * D) ** 2)
            if best is None or e < best[0]:
                best = (e, d, rho)
        o[f"sinc_mag_fit_{lo}_{hi}"] = [r(100 * best[1], 2), r(best[2])]
    out[name] = o
    print(name, o)
with open(os.path.join(HERE, "v1_drift.json"), "w") as fh:
    json.dump(out, fh, indent=1)
