"""Exp 4b: are the jointly fitted references of the acoustic overlap take (hw/rec/talkAB_LR.wav) trustworthy?

Compares, for A (left speaker) and B (right speaker), the responses fitted JOINTLY on talkAB_LR (bz4_overlap_tse.fit_joint,
400 and 1500 CG iterations) with the responses fitted on the single-talker takes talkA_L / talkB_R of the same session
(hw/realset.py fit_image), via the images they predict for the overlap take (same source signal, same alignment):
error of the joint image vs the single-take image (1 s of the source convolved with each response truncated to
0-5 ms, 0-50 ms or the full 4096 taps after the peak - 1 ms; dB re the joint image) after a best scalar gain and a
+-8 sample lag search. Room responses change between takes (HW_TESTS: direct sound
-28 dB, 5-50 ms -11 dB), so a ~-10 dB agreement of the early part is the expected best case.
Run: .venv/bin/python -I hw/realset/bizarre/bz4b_refcheck.py -> bz4b_refcheck.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

from bz_common import FS, HERE, HW, dump, load_wav  # noqa: E402
from bz4_overlap_tse import K, LEAD, fit_joint  # noqa: E402


def main():
    from analyze import bandpass, stretch, xcorr_lag
    import realset as rs
    y = load_wav(os.path.join(HW, "rec", "talkAB_LR.wav"))
    st = np.load(os.path.join(HW, "stim", "talkAB_LR.npz"))
    sA, sB = stretch(st["A"].astype(np.float64), 17e-6), stretch(st["B"].astype(np.float64), 17e-6)
    N = y.shape[1]
    n0 = xcorr_lag(bandpass(y[0], 150, 7000), bandpass(sA + sB, 150, 7000)) - 256
    pa, pb = np.zeros(N), np.zeros(N)
    pa[n0:n0 + len(sA)] = sA[:N - n0]
    pb[n0:n0 + len(sB)] = sB[:N - n0]
    res = {"n0": int(n0)}
    single = {}
    for s, take in (("A", "talkA_L"), ("B", "talkB_R")):
        ys = load_wav(os.path.join(HW, "rec", take + ".wav"))
        _, hs, _ = rs.fit_image(ys.T, np.load(os.path.join(HW, "stim", take + ".npz"))[s].astype(np.float64))
        single[s] = hs[:, 0]
    src = {"A": pa, "B": pb}
    for iters in (400, 1500):
        hA, hB, info = fit_joint(y[0], pa, pb, iters=iters)
        joint = {"A": hA, "B": hB}
        r = {"cg_info": info}
        for s in ("A", "B"):
            hj, hs = joint[s], single[s]
            pj = int(np.argmax(np.abs(hj)))
            ps = int(np.argmax(np.abs(hs)))
            r[f"peak_idx_{s}"] = [pj, ps]
            for name, L in (("0-5ms", 80), ("0-50ms", 800), ("full", K)):
                best = None
                for lag in range(-8, 9):
                    a = hj[max(0, pj - 16):pj + L]
                    q0 = ps - 16 + lag
                    if q0 < 0:
                        continue
                    b = hs[q0:q0 + len(a)]
                    n = min(len(a), len(b))
                    ex = src[s][n0 + LEAD + 8000:n0 + LEAD + 24000]   # 1 s of the source at full rate
                    ia = fftconvolve(ex, a[:n])
                    b = b[:n]
                    ib = fftconvolve(ex, b)
                    g = np.dot(ia, ib) / np.dot(ib, ib)
                    e = 10 * np.log10(np.sum((ia - g * ib) ** 2) / np.sum(ia ** 2))
                    if best is None or e < best[0]:
                        best = (float(e), lag, float(g))
                r[f"{s}_{name}"] = {"err_db": best[0], "lag": best[1], "gain": best[2]}
            # time-domain response similarity (normalized correlation) over the first 50 ms
            a = hj[max(0, pj - 16):pj + 800]
            b = hs[max(0, ps - 16):ps + 800]
            n = min(len(a), len(b))
            r[f"{s}_corr_50ms"] = float(np.dot(a[:n], b[:n]) / np.sqrt(np.dot(a[:n], a[:n]) * np.dot(b[:n], b[:n])))
        res[str(iters)] = r
        print(iters, r, flush=True)
    dump(res, os.path.join(HERE, "bz4b_refcheck.json"))


if __name__ == "__main__":
    main()
