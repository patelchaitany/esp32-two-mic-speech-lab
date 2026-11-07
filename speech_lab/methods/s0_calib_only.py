"""Sanity method: Mic A after the S0 front end (prefilter, STFT, blind calibration, ISTFT).

S0 corrects only X_B, so the output equals the high-passed Mic A; running the full S0 path checks
that it runs on every scene, stays time-aligned and costs little (DESIGN.md 2.0)."""
import numpy as np

from methods._core import calib, common as cm


def enhance(x, fs, info, return_stages=False):
    xh = calib.prefilter(np.asarray(x, dtype=np.float64), info.get("scenario", "bus"))
    X = cm.stft(xh)
    X = calib.apply(X, calib.calibrate(X, info))
    y = cm.istft(X[0], xh.shape[1])
    return {"out": y, "pre_agc": y} if return_stages else y
