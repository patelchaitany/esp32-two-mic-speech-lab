"""M3 mask-driven variant, online (causal): as m3_mask_mvdr with the causal front end and recursions
(see m3_rtf_mvdr_online)."""
from methods._core import rtf
from methods.m3_mask_mvdr import MASK


def enhance(x, fs, info, return_stages=False):
    return rtf.enhance(x, fs, info, online=True, return_stages=return_stages, **MASK)
