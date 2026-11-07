"""Direct sound of each sweep: TDOA by several estimators and windows, structure of the first ms (upsampled),
and the direct level / DRR per speaker (placement cue)."""
import json
import os
import sys

import numpy as np
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FS, HERE, gcc_phat, irs, phase_slope, r3  # noqa: E402

P0 = 80
H = irs()
res = {}
for name, h in H.items():
    r = {}
    # GCC-PHAT on windows of different length, full band and 300-6500 Hz
    for t0, t1 in [(-1, 1.5), (-1, 3), (-2, 4), (-1, 6), (-2, 10)]:
        i0, i1 = P0 + int(t0 * 16), P0 + int(t1 * 16)
        seg = h[i0:i1] * np.hanning(i1 - i0 + 2)[1:-1, None]
        r[f"gcc_full_{t0}_{t1}ms"] = r3(gcc_phat(seg[:, 0], seg[:, 1]))
        r[f"gcc_300_6500_{t0}_{t1}ms"] = r3(gcc_phat(seg[:, 0], seg[:, 1], lo=300, hi=6500))
        r[f"gcc_1000_5000_{t0}_{t1}ms"] = r3(gcc_phat(seg[:, 0], seg[:, 1], lo=1000, hi=5000))
        A = np.fft.rfft(seg[:, 0], 4096)
        B = np.fft.rfft(seg[:, 1], 4096)
        f = np.fft.rfftfreq(4096, 1 / FS)
        P = A * np.conj(B)
        r[f"slope_magw_300_6500_{t0}_{t1}ms"] = r3(phase_slope(f, P / np.abs(P), w=np.abs(P), lo=300, hi=6500)[0])
    # time-domain: cross-correlation peak of x16-upsampled 2.5 ms windows (300-6500 Hz band-limited)
    seg = h[P0 - 32:P0 + 64]
    up = resample(seg, len(seg) * 16, axis=0)
    a, b = up[:, 0], up[:, 1]
    cc = np.correlate(b, a, "full")
    lag = (np.argmax(cc) - (len(a) - 1)) / 16
    r["xcorr_up16_-2_4ms"] = r3(lag)
    # peak positions (upsampled) of each channel, and envelope of the first 3 ms (dB re peak)
    pa, pb = np.argmax(np.abs(a)) / 16 - 32, np.argmax(np.abs(b)) / 16 - 32
    r["peak_idx_A_B_rel_P0"] = [r3(pa), r3(pb)]
    env = 20 * np.log10(np.abs(h[P0 - 16:P0 + 64, 0]) / np.abs(h[P0, 0]) + 1e-9)
    r["envelope_A_dB_from_-1ms_every_0.25ms"] = [r3(np.max(env[i:i + 4])) for i in range(0, len(env), 4)]
    # direct energy (first 2.5 ms) and level B re A, and energy after 2.5 ms (DRR-like)
    d = h[P0 - 16:P0 + 24]
    rv = h[P0 + 24:]
    r["direct_dB_A_B"] = [r3(10 * np.log10(np.sum(d[:, c] ** 2))) for c in range(2)]
    r["drr_2.5ms_dB_A_B"] = [r3(10 * np.log10(np.sum(d[:, c] ** 2) / np.sum(rv[:, c] ** 2))) for c in range(2)]
    res[name] = r
    print(name)
    for k, v in r.items():
        print("  ", k, v)
with open(os.path.join(HERE, "s1b_direct.json"), "w") as fh:
    json.dump(res, fh, indent=1)
