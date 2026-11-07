"""x_bus_v3_m0d20 (night-B final:bus variant): (d) single-channel reference: Mic A -> DPDFNet-4 (20 dB limit) -> remix -> M11-bus AGC.
Config bound to methods/x_bus_v3_stack.py (see its docstring and study/final/BUS_V3.md)."""
from methods import x_bus_v3_stack as _s

CFG = _s.config(**{'spatial': 'mica', 'cdr': False, 'ns': False, 'nn': 'dpdfnet', 'nn_limit_db': 20.0})
ALGO_LATENCY_MS = _s.latency_ms(CFG)


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
