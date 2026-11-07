"""x_meeting_v3 variant 'embhyb': classic chain, M11 keys = WeSpeaker cluster where the region had its own embedding chunk, else the S1 segment."""
from methods import x_meeting_v3_stack as _st

CFG = dict(keys="embhyb")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
