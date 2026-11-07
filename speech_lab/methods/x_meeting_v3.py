"""x_meeting_v3: chosen final meeting pipeline (dev, natural speech; study/final/MEETING_V3.md): x_s0v2 self-calibration when no calib_json (else the calib_json) -> 2-ch WPE -> S1 -> steered superdirective M2 (WNG >= -3 dB) -> M8 GTCRN post-filter (12 dB attenuation limit, replaces M6) -> M11 (one static gain per S1 speech segment, -1 dBTP limiter). Returns {out, pre_agc}."""
from methods import x_meeting_v3_stack as _st

CFG = dict(s0="v2", ns="gtcrn", nn_atten_db=12.0)


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
