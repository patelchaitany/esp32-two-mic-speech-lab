"""Exp 5b: can the skew be calibrated in a QUIET room? (complements bz5_calib_stability.py)

(a) Ambient-level emulation: the current-boot fan noise (take tails, reboot_07, s3_ambient, drift_00; loud,
    near-diffuse) is scaled to an in-band (150-7000 Hz) Mic-A level L and independent white noise per mic is added
    at the INMP441 self-noise (-87 dBFS, datasheet SNR 61 dB at -26 dBFS/94 dB SPL; the quiet drift files at
    -56..-58 dBFS still show rho 0.99, so the real floor is lower than about -77 dBFS). Skew std vs window length.
(b) Speech-only: the skew from the reverberant speech image of each take (no noise at all), using all frames
    and using only 'decay' frames (frames whose Mic-A level is >= 10 dB below the level 50 ms earlier, i.e.
    the reverberant tails after offsets). Truth: the current boot's -1.013 (bz5, 24 files).
Run: .venv/bin/python -I hw/realset/bizarre/bz5b_quiet_skew.py -> bz5b_quiet_skew.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from bz_common import FS, HERE, HW, coh_stats, dump, load_take, load_wav, skew_phase  # noqa: E402

TRUE = -1.013
BAND = butter(4, [150, 7000], "band", fs=FS, output="sos")
TAKES = ["A1", "A2", "A3", "B1", "B2", "B3", "AE1", "AE2", "BE1", "BE2", "M1", "M2", "M3", "M4", "M5", "M6",
         "MF1", "MF2"]


def loud_noise():
    segs = [load_wav(os.path.join(HW, "rec2", "reboot_07.wav"))[:, FS:11 * FS],
            load_wav(os.path.join(HW, "rec2", "s3_ambient_after.wav"))[:, FS:11 * FS],
            load_wav(os.path.join(HW, "rec2", "drift_00.wav"))[:, FS:11 * FS]]
    for t in TAKES:
        z = load_take(t)
        on = int(z["onset"])
        y = z["recording"].T.astype(np.float64)
        segs.append(y[:, on + int(4.9 * FS):y.shape[1] - FS // 4])
    return segs


def main():
    rng = np.random.default_rng(5)
    path = os.path.join(HERE, "bz5b_quiet_skew.json")
    only_speech = "speech" in sys.argv[1:]
    segs = [] if only_speech else loud_noise()
    res = {"ambient": {}, "speech": {}}
    if only_speech and os.path.exists(path):
        import json
        res["ambient"] = json.load(open(path))["ambient"]
    self_noise = 10 ** (-87 / 20)
    for L in (() if only_speech else (-45, -55, -65, -70, -75, -80)):
        res["ambient"][L] = {}
        for W in (0.5, 1.0, 2.0, 5.0):
            sk = []
            n = int(W * FS)
            for x in segs:
                g = 10 ** (L / 20) / np.sqrt(np.mean(sosfiltfilt(BAND, x[0]) ** 2))
                for s in range(0, x.shape[1] - n + 1, n):
                    w = g * x[:, s:s + n] + self_noise * rng.standard_normal((2, n))
                    f, aa, bb, ab = coh_stats(w, nperseg=1024 if W >= 1 else 512)
                    sk.append(skew_phase(f, aa, bb, ab))
            sk = np.array(sk)
            res["ambient"][L][W] = {"n": len(sk), "median": float(np.median(sk)), "std": float(sk.std(ddof=1)),
                                    "rmse_vs_true": float(np.sqrt(np.mean((sk - TRUE) ** 2))),
                                    "frac_within_0.05": float(np.mean(np.abs(sk - TRUE) <= 0.05))}
            print("ambient", L, W, res["ambient"][L][W], flush=True)
    # (b) speech-only skew from the fitted images (no noise)
    from methods._core import common as cm
    for t in TAKES:
        z = load_take(t)
        img = z["image"].T.astype(np.float64)
        X = cm.stft(img)
        e = 10 * np.log10(np.mean(np.abs(X[0, 20:128]) ** 2, 0) + 1e-20)    # 600-4000 Hz level per frame
        from scipy.ndimage import maximum_filter1d
        prev = maximum_filter1d(e, 13, origin=6)                            # max over the previous ~100 ms
        act = e > e.max() - 50
        dec = act & (e <= prev - 8)                                         # >= 8 dB below it: reverberant tails
        out = {}
        for name, sel in (("all", act), ("decay", dec)):
            Xs = X[:, :, sel]
            aa = np.mean(np.abs(Xs[0]) ** 2, 1)
            bb = np.mean(np.abs(Xs[1]) ** 2, 1)
            ab = np.mean(Xs[0] * np.conj(Xs[1]), 1)
            out[name] = {"skew": skew_phase(cm.FREQS, aa, bb, ab), "frames": int(sel.sum())}
        res["speech"][t] = out
        print("speech", t, out, flush=True)
    for name in ("all", "decay"):
        v = np.array([res["speech"][t][name]["skew"] for t in TAKES])
        res["speech"][f"summary_{name}"] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                                            "median_err_vs_true": float(np.median(v - TRUE))}
    dump(res, path)


if __name__ == "__main__":
    main()
