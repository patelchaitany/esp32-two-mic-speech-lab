"""x_bus_v3_c_m3_df (night-B final:bus variant): c_m3 with the M4 CDR corrected by the M3 beam's directivity factor.
Config bound to methods/x_bus_v3_stack.py (see its docstring and study/final/BUS_V3.md)."""
from methods import x_bus_v3_stack as _s

CFG = _s.config(**{'spatial': 'm3', 'cdr_df': True})
ALGO_LATENCY_MS = _s.latency_ms(CFG)


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
