"""Endfire flip test: mic spacing and channel skew from one 3 s sweep per board orientation.

Position 1: the line through both mic holes points at the MacBook's left speaker, Mic A nearer the speaker.
Position 2: the same spot, board turned 180 degrees (Mic B nearer the speaker).
Then  tau1 = skew + d/c,  tau2 = skew - d/c  (tau = arrival at B minus arrival at A), so
      skew = (tau1 + tau2) / 2,  d = c * (tau1 - tau2) / 2.

Usage (from speech_lab):
  .venv/bin/python -I hw/geomcheck.py play <label> [L|R]      play sweep_L (or sweep_R), record 12 s, print the direct-path TDOA
  .venv/bin/python -I hw/geomcheck.py ambient <label>         record 12 s of silence (diffuse-field cross-check)
  .venv/bin/python -I hw/geomcheck.py analyze <wav>           TDOA of an existing sweep_L recording
  .venv/bin/python -I hw/geomcheck.py solve <pos1> <pos2> [<ambient>]
Recordings go to hw/rec2/<label>.wav (float32, ch0 = Mic A, ch1 = Mic B) with a .json result next to them.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import butter, csd, sosfiltfilt  # noqa: E402

from analyze import FS, deconvolve, gcc_phat_tdoa, stim, stretch  # noqa: E402
from esp32 import play_record  # noqa: E402

C = 343.0
DRIFT = 17e-6  # Mac output vs ESP32 clock, measured in session 1
OUT = os.path.join(HERE, "rec2")
_HP = butter(4, 80, "high", fs=FS, output="sos")


def load(path):
    x, fs = sf.read(path, dtype="float64")
    assert fs == FS
    return sosfiltfilt(_HP, x - x.mean(0), axis=0)


def direct_tdoa(path):
    """Direct-path TDOA (samples) and quality numbers for a sweep_L recording."""
    y = load(path)
    h = deconvolve(y, stretch(stim("sweep_L")["L"], DRIFT))
    p = int(np.argmax(np.abs(h).sum(1)))
    # short window around the direct sound (0.5 ms before, 1.5 ms after), tapered, to keep desk reflections out
    w = h[p - 8:p + 24] * np.hanning(32 + 16)[8:40, None]
    tau = gcc_phat_tdoa(w[:, 0], w[:, 1], maxlag=6.0)
    tau_long = gcc_phat_tdoa(h[p - 16:p + 48, 0], h[p - 16:p + 48, 1], maxlag=6.0)
    noise = np.mean(h[p + int(0.45 * FS):p + int(0.6 * FS)] ** 2, 0)
    snr = 10 * np.log10(np.max(h[p - 8:p + 24] ** 2, 0) / noise)
    lvl = 10 * np.log10(np.sum(w[:, 1] ** 2) / np.sum(w[:, 0] ** 2))
    return {"tdoa_samples": round(tau, 3), "tdoa_us": round(tau / FS * 1e6, 1), "tdoa_4ms_window_samples": round(tau_long, 3),
            "direct_snr_db": [round(float(s), 1) for s in snr], "level_B_re_A_db": round(float(lvl), 2),
            "onset_s": round(p / FS - 0.25, 3)}


def diffuse_fit(path, skew=None):
    """Fit rho * sinc(2 f d / c) * exp(j 2 pi f skew / fs) to the coherence A conj(B) of a silent recording."""
    x = load(path)[FS:11 * FS]
    f, ab = csd(x[:, 1], x[:, 0], FS, nperseg=1024)  # scipy: conj(B) A
    _, aa = csd(x[:, 0], x[:, 0], FS, nperseg=1024)
    _, bb = csd(x[:, 1], x[:, 1], FS, nperseg=1024)
    G = ab / np.sqrt(aa.real * bb.real)
    m = (f > 150) & (f < 7000)

    def res(p):
        rho, d = p[0], p[1]
        s = skew if skew is not None else p[2]
        e = G[m] - rho * np.sinc(2 * f[m] * d / C) * np.exp(2j * np.pi * f[m] * s / FS)
        return np.concatenate([e.real, e.imag])

    best = None
    for d0 in (0.01, 0.02, 0.04, 0.06):
        x0 = [0.95, d0] + ([] if skew is not None else [-1.0])
        lo = [0.2, 0.003] + ([] if skew is not None else [-4.0])
        hi = [1.0, 0.3] + ([] if skew is not None else [4.0])
        r = least_squares(res, x0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    out = {"d_cm": round(float(best.x[1]) * 100, 2), "rho": round(float(best.x[0]), 2),
           "rmse": round(float(np.sqrt(np.mean(best.fun ** 2))), 3)}
    out["skew_samples"] = round(float(skew if skew is not None else best.x[2]), 3)
    return out


def main():
    cmd = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if cmd in ("play", "ambient"):
        label = sys.argv[2]
        side = sys.argv[3] if len(sys.argv) > 3 else "L"
        stim_path = os.path.join(HERE, "stim", f"sweep_{side}.wav") if cmd == "play" else None
        fs, x, meta = play_record(stim_path, 12)
        wav = os.path.join(OUT, label + ".wav")
        sf.write(wav, x, fs, subtype="FLOAT")
        if cmd == "play":
            meta.update(direct_tdoa(wav), speaker=side)
        else:
            meta["diffuse_fit"] = diffuse_fit(wav)
        meta["label"] = label
        with open(os.path.join(OUT, label + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        keys = ("tdoa_samples", "tdoa_us", "tdoa_4ms_window_samples", "direct_snr_db", "level_B_re_A_db", "onset_s", "diffuse_fit")
        print(json.dumps({k: meta[k] for k in keys if k in meta}))
    elif cmd == "analyze":
        print(json.dumps(direct_tdoa(sys.argv[2])))
    elif cmd == "solve":
        t1 = json.load(open(os.path.join(OUT, sys.argv[2] + ".json")))["tdoa_samples"]
        t2 = json.load(open(os.path.join(OUT, sys.argv[3] + ".json")))["tdoa_samples"]
        skew = (t1 + t2) / 2
        d = C * (t1 - t2) / 2 / FS
        res = {"tau_pos1": t1, "tau_pos2": t2, "skew_samples": round(skew, 3), "skew_us": round(skew / FS * 1e6, 1),
               "spacing_cm": round(d * 100, 2),
               "note": "spacing is d*cos(angle off the axis): 1.5 % low at 10 degrees"}
        if len(sys.argv) > 4:
            amb = os.path.join(OUT, sys.argv[4] + ".wav")
            res["diffuse_fit_free"] = diffuse_fit(amb)
            res["diffuse_fit_skew_fixed"] = diffuse_fit(amb, skew)
        with open(os.path.join(OUT, "geometry.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
