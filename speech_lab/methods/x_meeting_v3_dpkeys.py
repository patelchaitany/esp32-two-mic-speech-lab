"""x_meeting_v3 variant 'dpkeys' (ablation): classic chain with M11 keyed by S1 DP-means TDOA/CDR clusters (DESIGN gain memory by direction)."""
from methods import x_meeting_v3_stack as _st

CFG = dict(keys="cluster")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
