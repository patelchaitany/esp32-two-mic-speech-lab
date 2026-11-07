"""x_bus_v3_g12_m3 (night-B final:bus variant): c_m3 -> GTCRN on the target stream, 12 dB attenuation limit.
Config bound to methods/x_bus_v3_stack.py (see its docstring and study/final/BUS_V3.md)."""
from methods import x_bus_v3_stack as _s

CFG = _s.config(**{'spatial': 'm3', 'nn': 'gtcrn', 'nn_limit_db': 12.0})
ALGO_LATENCY_MS = _s.latency_ms(CFG)


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
