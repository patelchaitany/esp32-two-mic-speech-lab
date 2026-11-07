"""M10 standalone, offline: S0 high-pass on both mics -> two-channel WPE -> Mic A (DESIGN M10).

Segment-wise WPE (DESIGN M10 variant b) with the tuned defaults of methods._core.wpe: 30 taps, delay 4,
one iteration, fixed 2 s segments, each filter estimated on [segment − 3 s, segment + 1 s] and applied
with 50 ms crossfades. S1 change points as segment bounds scored the same on the dev meetings, so S1 is
not run here. The prefilter is S0 step 1 (120 Hz bus, 80 Hz meeting); S0's per-band correction of Mic B
barely changes the Mic-A output of WPE, so it is skipped. Stacks call methods._core.wpe directly and
keep both channels. Meant for the meeting: bus partners sit inside the critical distance, so there is
little reverberation to remove and most of its small bus gain comes from cancelling engine tones.
"""
from methods._core import calib, wpe


def enhance(x, fs, info, return_stages=False):
    y = wpe.dereverb(calib.prefilter(x, info.get("scenario", "meeting")), fs)[0]
    return {"out": y, "pre_agc": y} if return_stages else y
