"""x_meeting_v3 variant 'nobeam' (ablation): S0 core -> WPE -> S1 -> WPE Mic A (no beam) -> M6 -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict(beam="none")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
