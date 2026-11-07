"""x_meeting_v3 variant 'core' (phase-1 classic): calib.py blind S0 -> WPE -> S1 -> steered M2 -> M6 -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict()


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
