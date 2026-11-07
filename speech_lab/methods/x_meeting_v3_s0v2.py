"""x_meeting_v3 variant 's0v2': x_s0v2 self-calibration (when no calib_json) -> WPE -> S1 -> steered M2 -> M6 -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict(s0="v2")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
