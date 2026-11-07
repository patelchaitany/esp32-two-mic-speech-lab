"""M3 standalone, offline: S0 -> S1 -> geometry-free RTF-MVDR (covariance whitening) -> ISTFT (DESIGN.md M3, 2.1).

Bus: partner covariances over S1-labelled frames (p_k > 0.85, direct, no double talk; two-sided 3 s window,
refreshed every 0.5 s), hybrid noise model Gamma_n + sigma_s I, RTF g = Phi_w v / (Phi_w v)_1 where it is
reliable (whitened SNR >= 5 dB, within 60 deg / 3 dB of S1's plane wave; else model steering at S1's TDOA),
symmetrized RTFs, beams w_A / w_AB (M2 phase rule) / w_B under the bus WNG floors, states A / AB / B with 50 ms
crossfades. When S1 cannot place two partners on opposite sides of broadside (one peak, TDOAs < 25 us apart or
beyond endfire, both on one side) the output is Mic A, or the broadside MVDR when both are near broadside.
Meeting: Mic A (M3's optional meeting mode is off by default; m3_rtf_mvdr_cluster turns it on).
Linear output referenced to the array centre (partner k lags Mic A by tau_k / 2, at most 1.4 samples); no
masks, no AGC. Stacks call methods._core.rtf.run(front_end, info) and keep the Beam (weights, DF, RTFs).
"""
from methods._core import rtf


def enhance(x, fs, info, return_stages=False):
    return rtf.enhance(x, fs, info, online=False, return_stages=return_stages)
