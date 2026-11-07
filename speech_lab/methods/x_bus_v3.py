"""x_bus_v3: FINAL bus pipeline of night-B (final:bus), chosen on dev with natural speech (study/final/BUS_V3.md).

Chosen config (same as x_bus_v3_m0d20): blind self-calibration (info['calib_json'] if given, else x_s0v2 from the
input) -> S0/S1 (S1 drives only the AGC gate) -> NO linear spatial stage (Mic A after S0's high-pass) -> M8 DPDFNet-4
lab-grid mask floored at -20 dB, combined once (m = max(-15 dB, mask), instant rise, fall <= 4.8 dB/hop) -> split
remix (residual -> M6 -> slow 2:1 compression, mixed at -15 dB) -> M11-bus AGC gated on p_A + p_B -> -1 dBTP limiter.
Why no spatial stage: on the 12-scene dev subset (SNR 0, LibriSpeech) every spatial/classical-mask variant in front
of the network lowered worst-partner dESTOI (M3 + CDR + M6 -> DPDFNet +0.016 vs Mic A -> DPDFNet +0.069).
Returns {'out', 'pre_agc'}; deterministic, stateless.
"""
from methods import x_bus_v3_stack as _s

CFG = _s.config(spatial="mica", cdr=False, ns=False, nn="dpdfnet", nn_limit_db=20.0)
ALGO_LATENCY_MS = _s.latency_ms(CFG)


def enhance(x, fs, info, return_stages=False):
    return _s.run(x, fs, info, CFG, return_stages)
