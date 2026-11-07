"""M3 standalone, online (causal): S0 (5 s refresh) -> causal S1 -> RTF-MVDR -> ISTFT (DESIGN.md M3, 2.1).

As m3_rtf_mvdr, but every block uses only past frames: the covariance recursion over labelled frames
(3 s memory) is read before each 0.5 s block, a partner's covariance restarts when its S1 TDOA moves by
more than 20 us, the state machine has no look-ahead, and blocks without two usable partner TDOAs
(e.g. the first seconds) use the fallback (Mic A, or the broadside MVDR when both partners are near
broadside). Meeting: Mic A. Algorithmic latency: the 512-point STFT (32 ms).
"""
from methods._core import rtf


def enhance(x, fs, info, return_stages=False):
    return rtf.enhance(x, fs, info, online=True, return_stages=return_stages)
