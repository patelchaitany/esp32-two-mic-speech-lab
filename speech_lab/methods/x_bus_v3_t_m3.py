"""x_bus_v3_t_m3 (night-B final:bus variant): (c) c_m3 with the M9 union-of-partners mask (info['enroll'] or self-enrollment), 20 dB limit.
Config bound to methods/x_bus_v3_stack.py (see its docstring and study/final/BUS_V3.md)."""
from methods import x_bus_v3_stack as _s

CFG = _s.config(**{'spatial': 'm3', 'tse': True})
ALGO_LATENCY_MS = _s.latency_ms(CFG)
OFFLINE_ONLY = True


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
