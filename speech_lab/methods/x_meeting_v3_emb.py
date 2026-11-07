"""x_meeting_v3 variant 'emb' (candidate c): classic chain with M11 keyed by WeSpeaker embedding clusters (spk_onnx, THETA_S)."""
from methods import x_meeting_v3_stack as _st

CFG = dict(keys="emb")


def enhance(x, fs, info, return_stages=False):
    return _st.enhance_cfg(x, fs, info, CFG, return_stages)
