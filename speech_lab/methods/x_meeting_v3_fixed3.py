"""x_meeting_v3 variant 'fixed3': S0 core -> WPE -> S1 -> 3 fixed beams + DESIGN best-beam selection -> M6 -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict(beam="fixed3")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
