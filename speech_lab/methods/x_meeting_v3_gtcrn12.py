"""x_meeting_v3 variant 'gtcrn12': S0 core -> WPE -> S1 -> steered M2 -> M8 GTCRN (12 dB limit, replaces M6) -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict(ns="gtcrn", nn_atten_db=12.0)


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
