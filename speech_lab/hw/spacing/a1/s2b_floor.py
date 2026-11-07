"""Absolute level of the stationary incoherent floor (regression intercept across all ambient 1-s blocks) versus the
INMP441 datasheet self-noise, and how much it lowers the coherence of the loud-fan blocks used for the d fits.

PSD units: dB re (full-scale 1.0)^2 / Hz. INMP441: EIN 33 dBA SPL, sensitivity -26 dBFS @ 94 dB SPL -> -87 dBFS(A);
0 dBFS = full-scale sine = -3 dB re 1.0^2 -> about -90 dB re 1^2 over 20 Hz-20 kHz -> about -133 dB re 1^2/Hz if white.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FS, HERE, r3  # noqa: E402
from s2_fits import LOUD, SEGS, block_sums  # noqa: E402

W2 = np.sum((np.hanning(514)[1:-1]) ** 2)
f, blocks = block_sums(list(SEGS.keys()))
nfr = np.array([b[4] for b in blocks])[:, None]
Pab = np.array([b[1] for b in blocks]) / nfr
Paa = np.array([b[2] for b in blocks]) / nfr
Pbb = np.array([b[3] for b in blocks]) / nfr
loud = np.array([b[0] in LOUD for b in blocks])
rot = np.exp(2j * np.pi * f * 1.10 / FS)
psd = lambda p: 10 * np.log10(2 * p / (FS * W2))
rows = []
for fc in [375, 875, 1375, 2125, 3125, 4125, 5125, 5875, 6625, 7375]:
    m = (f >= fc - 125) & (f < fc + 125)
    nb = m.sum()
    y = np.real((Pab[:, m] * rot[m]).sum(1))
    x = np.sqrt(Paa[:, m].sum(1) * Pbb[:, m].sum(1))
    coef, *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
    floor = -coef[1] / coef[0] / nb                       # per-bin power of the incoherent floor
    loud_p = np.median(x[loud]) / nb
    quiet_p = np.min(x) / nb
    # coherence reduction factor for the median loud block: S/(S+N) with S = loud_p - floor
    red = (loud_p - floor) / loud_p if floor > 0 else 1.0
    rows.append([fc, r3(psd(max(floor, 1e-30))) if floor > 0 else None, r3(psd(loud_p)), r3(psd(quiet_p)),
                 r3(red), r3(coef[0])])
out = {"cols": ["f_center", "incoherent_floor_dB_re1/Hz", "median_loud_block_dB_re1/Hz", "quietest_block_dB_re1/Hz",
                "coherence_factor_loud=S/(S+N)", "regression_slope"],
       "rows": rows, "inmp441_datasheet_selfnoise_dB_re1/Hz_if_white": -133}
for r in rows:
    print(r)
with open(os.path.join(HERE, "s2b_floor.json"), "w") as fh:
    json.dump(out, fh, indent=1)
