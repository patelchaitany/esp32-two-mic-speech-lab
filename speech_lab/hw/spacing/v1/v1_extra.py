"""V1 extra checks.
1. Is the high-frequency coherence limited by a stationary incoherent floor (A1's regression claim)? If so, the louder
   post-reboot fan (FAN2) must show HIGHER |coherence| at 4-6.5 kHz than session 1 (FAN1) where its band level is higher.
2. Frequency independence and precision of the skew change across the reboots (double difference s2_sweep_L -> s3_sweep_L,
   direct sound -1..5 ms) in sub-bands; size of the change in SCK periods (1/64 sample).
Output: v1_extra.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import FS, HERE, deconv, load, stretch, xspec  # noqa: E402
from v1_skew import SESSION1, noise_segments, stim_ref  # noqa: E402


def r(x, k=3):
    return float(np.round(x, k))


def band_stats(names_refs):
    rows = {}
    for name, ref in names_refs:
        x = load(name)
        segs, _ = noise_segments(name, x, ref)
        f, ab, aa, bb, n, fr = xspec(x, 1024, 512, segs)
        scale = 2 / (FS * np.sum((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)) ** 2))
        o = {}
        for lo, hi in [(1000, 2000), (3000, 4000), (4000, 5000), (5000, 6500)]:
            m = (f >= lo) & (f < hi)
            o[f"{lo}-{hi}"] = [r(10 * np.log10(np.mean(aa[m]) * scale), 1), r(np.abs(np.sum(ab[m] * np.exp(-2j * np.pi * f[m] * (-1.07) / FS))) / np.sqrt(np.sum(aa[m]) * np.sum(bb[m])))]
        rows[name] = o
    return rows


def main():
    out = {}
    S1 = [("rec/" + k, v) for k, v in SESSION1.items()]
    POST = [(f"rec2/reboot_{i:02d}", None) for i in range(8)] + [("rec2/s3_ambient_after", None), ("rec2/drift_00", None)]
    out["hf_level_dB_per_Hz_and_band_coh"] = {"session1": band_stats(S1), "post_reboot": band_stats(POST)}
    for grp, rows in out["hf_level_dB_per_Hz_and_band_coh"].items():
        for k, v in rows.items():
            print(grp, f"{k:24s}", v)
    # 2. double difference in sub-bands
    def resp(name):
        h = deconv(load(name), stretch(stim_ref("stim/sweep_L.npz", "L"), 17e-6))
        p = int(np.argmax(np.abs(h[:3 * FS]).sum(1)))
        return h[p - 16:p + 80]
    h1, h2 = resp("rec2/s2_sweep_L"), resp("rec2/s3_sweep_L")
    n = 8192
    w = np.hanning(len(h1) + 2)[1:-1, None]
    X1 = np.fft.rfft(h1[:, 0] * w[:, 0], n) * np.conj(np.fft.rfft(h1[:, 1] * w[:, 0], n))
    X2 = np.fft.rfft(h2[:, 0] * w[:, 0], n) * np.conj(np.fft.rfft(h2[:, 1] * w[:, 0], n))
    f = np.fft.rfftfreq(n, 1 / FS)
    D = X2 * np.conj(X1)
    sub = {}
    for lo, hi in [(300, 1000), (1000, 2000), (2000, 3000), (3000, 4500), (4500, 6500), (300, 6500)]:
        m = (f >= lo) & (f <= hi)
        Dm, ff = D[m], f[m]
        wt = np.abs(Dm) ** 0.5
        grid = np.arange(-1, 1.0001, 0.001)
        sc = np.real(np.exp(-2j * np.pi * np.outer(grid, ff) / FS) @ (Dm / np.abs(Dm) * wt))
        t0 = grid[int(np.argmax(sc))]
        ph = np.angle(Dm * np.exp(-2j * np.pi * ff * t0 / FS))  # small residual, no unwrapping needed
        X = np.stack([2 * np.pi * ff / FS, np.ones(m.sum())], 1) * np.sqrt(wt)[:, None]
        coef, *_ = np.linalg.lstsq(X, ph * np.sqrt(wt), rcond=None)
        res = ph - X[:, 0] / np.sqrt(wt) * coef[0] - coef[1]
        sub[f"{lo}-{hi}"] = [r(t0 + coef[0], 4), r(np.degrees(coef[1]), 1), r(np.degrees(np.sqrt(np.sum(wt * res ** 2) / np.sum(wt))), 1)]
    out["double_difference_s2_to_s3_subbands[slope,icpt_deg,res_rms_deg]"] = sub
    out["change_in_SCK_periods"] = r(sub["300-6500"][0] * 64, 2)
    print("double difference sub-bands", sub, "in SCK periods", out["change_in_SCK_periods"])
    with open(os.path.join(HERE, "v1_extra.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
