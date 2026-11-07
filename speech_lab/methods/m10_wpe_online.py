"""M10 standalone, online: S0 high-pass on both mics -> causal RLS WPE -> Mic A (DESIGN M10 variant c).

Online defaults of methods._core.wpe: 10 taps, delay 4, forgetting factor α = 0.998 (memory 4 s; DESIGN
1.6 item 11). No per-speaker memory: keying the stored filters on S1's DP-means clusters scored worse
on the dev meetings (wpe.wpe_online(keys=...) still offers it, e.g. for embedding-keyed clusters).
The prefilter is S0 step 1 (120 Hz bus, 80 Hz meeting).
"""
from methods._core import calib, wpe


def enhance(x, fs, info, return_stages=False):
    y = wpe.dereverb(calib.prefilter(x, info.get("scenario", "meeting")), fs, online=True)[0]
    return {"out": y, "pre_agc": y} if return_stages else y
