"""V1: did the inter-channel delay of the SAME acoustic path change across the 01:51 esptool reboots?

Double difference of two left-speaker sweeps recorded with the board untouched: D(f) = [A2 conj(B2)] conj[A1 conj(B1)]
for the same time window of the two responses. Room, geometry and mic responses cancel; the phase of D is
2 pi f (tau_2 - tau_1)/fs if only the channel timing changed. Pairs: within boot 1 (sweep_L vs sweep_L2, sweep_L vs
s2_sweep_L: control, expect 0) and across the reboot (s2_sweep_L -> s3_sweep_L, sweep_L -> s3_sweep_L).
Output: v1_shift.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import FS, HERE, deconv, load, stretch  # noqa: E402
from v1_skew import stim_ref  # noqa: E402


def r(x, k=3):
    return float(np.round(x, k))


def response(name):
    h = deconv(load(name), stretch(stim_ref("stim/sweep_L.npz", "L"), 17e-6))
    p = int(np.argmax(np.abs(h[:3 * FS]).sum(1)))
    return h, p


def dd_slope(h1, p1, h2, p2, t0, t1, lo=300, hi=6500, n=8192, nper=None):
    """Slope (samples) of the double-difference phase in a window; Welch frames if nper is given."""
    def spec(h, p):
        i0, i1 = p + int(round(t0 * 16)), p + int(round(t1 * 16))
        if nper is None:
            seg = h[i0:i1] * np.hanning(i1 - i0 + 2)[1:-1, None]
            A, B = np.fft.rfft(seg[:, 0], n), np.fft.rfft(seg[:, 1], n)
            return (A * np.conj(B))[None, :]
        w = np.hanning(nper + 2)[1:-1]
        k = (i1 - i0 - nper) // (nper // 2) + 1
        idx = i0 + np.arange(nper)[None, :] + (nper // 2) * np.arange(k)[:, None]
        A, B = np.fft.rfft(h[idx, 0] * w, n, axis=1), np.fft.rfft(h[idx, 1] * w, n, axis=1)
        return A * np.conj(B)
    X1, X2 = spec(h1, p1), spec(h2, p2)
    f = np.fft.rfftfreq(n, 1 / FS)
    m = (f >= lo) & (f <= hi)
    if nper is None:
        Dm = (X2 * np.conj(X1))[0, m]
    else:  # per-frame double difference, phase-normalised cross-spectra averaged
        Dm = np.mean(X2[:, m] / np.abs(X2[:, m]), 0) * np.conj(np.mean(X1[:, m] / np.abs(X1[:, m]), 0))
    ff = f[m]
    grid = np.arange(-1, 1.0001, 0.002)
    s = np.real(np.exp(-2j * np.pi * np.outer(grid, ff) / FS) @ (Dm / np.maximum(np.abs(Dm), 1e-30) * np.abs(Dm) ** 0.5))
    t = grid[int(np.argmax(s))]
    ph = np.angle(Dm * np.exp(-2j * np.pi * ff * t / FS))
    w = np.abs(Dm) ** 0.5
    X = np.stack([2 * np.pi * ff / FS, np.ones_like(ff)], 1) * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X, ph * np.sqrt(w), rcond=None)
    # residual phase rms after the fit (deg), magnitude-weighted
    res = ph - (2 * np.pi * ff / FS * coef[0] + coef[1])
    return r(t + coef[0]), r(np.degrees(coef[1]), 1), r(np.degrees(np.sqrt(np.sum(w * res ** 2) / np.sum(w))), 1)


def main():
    H = {k: response(k) for k in ["rec/sweep_L", "rec/sweep_L2", "rec2/s2_sweep_L", "rec2/s3_sweep_L"]}
    pairs = [("rec/sweep_L", "rec/sweep_L2", "control: same boot, 30 s apart"),
             ("rec/sweep_L", "rec2/s2_sweep_L", "control: same boot, 30 min apart"),
             ("rec2/s2_sweep_L", "rec2/s3_sweep_L", "across the esptool reboots (6 min apart)"),
             ("rec/sweep_L", "rec2/s3_sweep_L", "across the reboots"),
             ("rec/sweep_L2", "rec2/s3_sweep_L", "across the reboots")]
    out = {}
    for a, b, note in pairs:
        h1, p1 = H[a]
        h2, p2 = H[b]
        o = {"note": note}
        for t0, t1 in [(-1, 2), (-1, 5), (-1, 10)]:
            o[f"direct_{t0}_{t1}ms[slope,icpt,res_rms]"] = dd_slope(h1, p1, h2, p2, t0, t1)
        for t0, t1 in [(10, 80), (30, 250), (80, 250)]:
            o[f"reverb_{t0}_{t1}ms_welch256[slope,icpt,res_rms]"] = dd_slope(h1, p1, h2, p2, t0, t1, lo=300, hi=3000, n=256, nper=256)
        out[f"{a} -> {b}"] = o
        print(a, "->", b, note)
        for k, v in o.items():
            if k != "note":
                print("   ", k, v)
    with open(os.path.join(HERE, "v1_shift.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
