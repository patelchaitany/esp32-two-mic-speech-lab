"""x_bus_v3_classic: best night-B bus pipeline WITHOUT neural models (Tier 1, DESIGN 2.1), chosen on dev
(study/final/BUS_V3.md). Same config as x_bus_v3_c_m3.

x_s0v2 self-calibration (or info['calib_json']) -> S0/S1 -> M3 RTF-MVDR (its inseparable fallback: Mic A or
broadside MVDR) -> masks from the linear signals, combined once: M4 DOA-free CDR (mic pair) x M6 MMSE-SPP/LSA on
Y_lin with the noise update frozen where the pair's CDR/(1+CDR) > 0.6; m = max(-15 dB, product), instant rise,
fall <= 4.8 dB/hop -> split remix (T = m Y_lin; R = (1-m) Y_lin -> M6 -> slow 2:1 compression, mixed at -15 dB)
-> M11-bus AGC gated on p_A + p_B -> -1 dBTP limiter. Returns {'out', 'pre_agc'}; deterministic, stateless.
"""
from methods import x_bus_v3_stack as _s

CFG = _s.config(spatial="m3")
ALGO_LATENCY_MS = _s.latency_ms(CFG)


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
