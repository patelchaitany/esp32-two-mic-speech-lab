"""Resolve the individual arrivals in the first ~3 ms of each speaker's response (x8 upsampled envelope) and the
TDOA of each arrival (short-window x8-upsampled cross-correlation and GCC-PHAT 500-6500 Hz).
Also: direct-path arrival time right minus left speaker in the two-speaker recordings (talkAB_LR, talkA_noise),
using the first arrival (not the strongest peak)."""
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks, hilbert, resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FS, HERE, gcc_phat, irs, r3  # noqa: E402
import s3_speech as S3  # noqa: E402

UP = 8


def arrivals(hk, p, pre_ms=4.0, post_ms=4.0, thr_db=-14):
    """hk (n,2) response, p = main-peak index. Returns list of arrivals (time re p in samples, level dB, tdoa)."""
    i0, i1 = p - int(pre_ms * 16), p + int(post_ms * 16)
    seg = hk[i0:i1]
    up = resample(seg, len(seg) * UP, axis=0)
    env = np.abs(hilbert(up, axis=0)).sum(1)
    env_db = 20 * np.log10(env / env.max())
    pk, _ = find_peaks(env_db, height=thr_db, distance=int(0.2 * 16 * UP))
    out = []
    for k in pk:
        t = (k / UP + i0) - p
        # short window +-0.35 ms around the arrival (Tukey) on the upsampled signals -> cross-correlation lag
        h = int(0.35 * 16 * UP)
        a0, a1 = max(0, k - h), min(len(up), k + h)
        w = np.hanning(a1 - a0 + 2)[1:-1]
        a, b = up[a0:a1, 0] * w, up[a0:a1, 1] * w
        cc = np.correlate(b, a, "full")
        lag = (np.argmax(cc) - (len(a) - 1)) / UP
        # GCC-PHAT on the original-rate signal, window +-0.5 ms
        c0 = int(round(t + p))
        sa = hk[c0 - 8:c0 + 9] * np.hanning(19)[1:-1, None]
        g = gcc_phat(sa[:, 0], sa[:, 1], lo=500, hi=6500)
        out.append({"t_re_peak_samples": r3(t), "level_dB": r3(env_db[k]), "tdoa_xcorr": r3(lag), "tdoa_gcc": r3(g)})
    return out


def first_arrival_time(hk, p, thr_db=-12):
    """Time (samples, fractional) where the x8 envelope of mic A first reaches thr_db re the max of the first 4 ms."""
    i0 = p - 64
    up = resample(hk[i0:p + 64, 0], 128 * UP)
    env = np.abs(hilbert(up))
    env_db = 20 * np.log10(env / env.max())
    k = int(np.argmax(env_db > thr_db))
    return i0 + k / UP


def main():
    res = {}
    H = irs()
    for name, h in H.items():
        res[name] = arrivals(h, 80)
    for name, keys in [("talkA_L", ["L"]), ("talkB_R", ["R"]), ("talkAB_LR", ["L", "R"]), ("talkA_noise", ["L", "R"]),
                       ("talkC_R_quiet", ["R"])]:
        h, lag = S3.estimate_irs(name, keys)
        for i, k in enumerate(keys):
            hk = h[:, :, i]
            p = int(np.argmax(np.abs(hk).sum(1)[:4000]))
            res[f"{name}:{k}"] = arrivals(hk, p)
            res[f"{name}:{k}:first_arrival_-12dB"] = r3(first_arrival_time(hk, p))
            res[f"{name}:{k}:first_arrival_-20dB"] = r3(first_arrival_time(hk, p, -20))
        if len(keys) == 2:
            for thr in ("-12dB", "-20dB"):
                d = res[f"{name}:R:first_arrival_{thr}"] - res[f"{name}:L:first_arrival_{thr}"]
                res[f"{name}:first_arrival_R_minus_L_{thr}"] = [r3(d), r3(d / FS * 343 * 100)]
    for k, v in res.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            print(k)
            for a in v:
                print("   ", a)
        else:
            print(k, v)
    with open(os.path.join(HERE, "s1c_arrivals.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
