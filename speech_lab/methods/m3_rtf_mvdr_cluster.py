"""M3 ablation, offline: m3_rtf_mvdr with the optional meeting mode on (DESIGN.md M3: meeting optional, steered
per cluster). In meetings an RTF-MVDR (WNG >= -3 dB) is steered to the active S1 talker cluster (0.3 s dwell,
50 ms crossfades) instead of passing Mic A through; the bus is unchanged. Kept to report the meeting mode on the
bench; the default wrappers leave meetings to Mic A (see methods/_core/rtf.py).
"""
from methods._core import rtf


def enhance(x, fs, info, return_stages=False):
    return rtf.enhance(x, fs, info, online=False, return_stages=return_stages, meeting="cluster")
