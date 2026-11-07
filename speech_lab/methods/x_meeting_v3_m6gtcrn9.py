"""x_meeting_v3 variant 'm6gtcrn9': S0 core -> WPE -> S1 -> steered M2 -> M6 -> M8 GTCRN (9 dB limit, after M6) -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict(ns="m6+gtcrn", nn_atten_db=9.0)


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
