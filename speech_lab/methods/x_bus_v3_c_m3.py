"""x_bus_v3_c_m3 (night-B final:bus variant): classical: x_s0v2 calib -> S0/S1 -> M3 RTF-MVDR -> M4 CDR x M6 (CDR-gated) mask, floor -15 dB -> split remix -15 dB -> M11-bus AGC + limiter.
Config bound to methods/x_bus_v3_stack.py (see its docstring and study/final/BUS_V3.md)."""
from methods import x_bus_v3_stack as _s

CFG = _s.config(**{'spatial': 'm3'})
ALGO_LATENCY_MS = _s.latency_ms(CFG)


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
