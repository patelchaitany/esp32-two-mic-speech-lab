"""x_meeting_v3_classic: best final meeting pipeline without neural models (chosen on dev, study/final/MEETING_V3.md): x_s0v2 self-calibration when no calib_json (else the calib_json) -> 2-ch WPE -> S1 -> steered superdirective M2 (WNG >= -3 dB) -> M6 (floor -12 dB, noise update in >= 300 ms S1-VAD pauses) -> M11 (one static gain per S1 speech segment, -1 dBTP limiter). Returns {out, pre_agc}."""
from methods import x_meeting_v3_stack as _st

CFG = dict(s0="v2")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
