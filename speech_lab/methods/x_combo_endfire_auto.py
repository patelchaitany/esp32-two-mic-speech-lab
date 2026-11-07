"""x_combo_endfire_auto (experimental, real:run 2026-11-03): x_combo_endfire with blind x_s0v2 self-calibration when
info has no calib_json (as the x_auto_* wrappers; every call re-estimates from its own input), else identical to
x_combo_endfire (calib_json + gamma_n repair). Deployable form of the endfire rule: skew, spacing, gain and gamma_n
from the input's own noise, fixed MVDR toward +d_hat/c (device oriented with Mic A's end toward the talkers)."""
from methods import x_combo_endfire as _base
from methods import x_combo_s1beam as _sb


def enhance(x, fs, info, return_stages=False):
    return _base.enhance(x, fs, _sb.calibrated_info(x, fs, info), return_stages=return_stages)
