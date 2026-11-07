"""x_meeting_v3 variant 'nons' (ablation): S0 core -> WPE -> S1 -> steered M2 -> (no NS) -> M11 segment keys."""
from methods import x_meeting_v3_stack as _st

CFG = dict(ns="none")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
