"""x_auto_m10_wpe: m10_wpe with x_s0v2 blind self-calibration (experimental, prefix x_).

If info has no 'calib_json', x_s0v2.estimate(x, fs, info) is computed from this very input (deterministic,
stateless: every call, including each Hagerman run, re-estimates from its own x) and passed on as
info['calib_json'] (S0 override: spacing, continuous skew tau0_s, HPF corners, per-band gain, model, rho,
gamma_n); otherwise the given calib_json is used unchanged. Everything else is m10_wpe.
Note: m10_wpe never reads S0 (it only applies calib.prefilter and two-channel WPE, which absorbs any fixed
inter-channel delay/gain), so this wrapper is a pass-through kept for a uniform method list; the estimate is
still computed so Hagerman runs and timing see the same algorithm.
"""
from methods import m10_wpe as _base
from methods import x_s0v2


def enhance(x, fs, info, return_stages=False, **kw):
    if info.get("calib_json") is None:
        info = dict(info, calib_json=x_s0v2.estimate(x, fs, info))
    return _base.enhance(x, fs, info, return_stages=return_stages, **kw)
