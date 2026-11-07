"""M0 baseline (DESIGN M0): Mic A after S0's high-pass filter.

The filter is S0 step 1: a causal 2nd-order Butterworth high-pass at 120 Hz (bus) or 80 Hz (meeting), the
same filter as methods._core.calib.prefilter. It is written out here so the baseline does not depend on
core modules that are still changing. Its group delay is under 2 samples above 500 Hz (bus).
"""
import numpy as np
from scipy.signal import butter, sosfilt

HPF_HZ = {"bus": 120.0, "meeting": 80.0}
ALGO_LATENCY_MS = 0.0


def enhance(x, fs, info, return_stages=False):
    sos = butter(2, HPF_HZ[info.get("scenario", "bus")], btype="highpass", fs=fs, output="sos")
    y = sosfilt(sos, np.asarray(x[0], np.float64))
    return {"out": y, "pre_agc": y} if return_stages else y
