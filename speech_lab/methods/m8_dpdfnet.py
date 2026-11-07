"""M8 standalone, DPDFNet-4: Mic A -> S0 high-pass -> DPDFNet-4 (onnxruntime) -> attenuation limit (DESIGN M8).

Single channel, no spatial cue. The high-pass is S0 step 1 (calib.prefilter: 2nd-order Butterworth,
120 Hz bus / 80 Hz meeting). The model input is level-normalised per file (nn.TARGET_DB) and the
output scaled back; the attenuation limit (DPDFNet definition) mixes the input back in (20 dB bus,
12 dB meeting). The streaming export emits frame t - 4 at step t (40 ms); that delay is removed, so
the output is aligned with Mic A. Algorithmic latency 60 ms (20 ms window + 40 ms model delay).
Optional info keys (ablations): "m8_atten_db" (float, or None for no limit; default nn.ATTEN_DB),
"m8_level" ("file" default, "online", or None for the raw input level).
Stacks call methods._core.nn.post_filter() on their linear output and use the returned lab-grid mask.
"""
import numpy as np

from methods._core import calib, nn

MODEL = "dpdfnet"


def enhance(x, fs, info, return_stages=False):
    assert fs == nn.FS, "M8 expects 16 kHz input"
    scenario = info.get("scenario", "bus")
    atten_db = info.get("m8_atten_db", nn.ATTEN_DB[scenario])
    level = info.get("m8_level", "file")
    if x.shape[-1] == 0:
        y = np.zeros(0)
        return {"out": y, "pre_agc": y} if return_stages else y
    x_a = calib.prefilter(x[0], scenario)
    y = nn.denoise(x_a, MODEL, atten_db=atten_db, level=level)
    return {"out": y, "pre_agc": y} if return_stages else y
