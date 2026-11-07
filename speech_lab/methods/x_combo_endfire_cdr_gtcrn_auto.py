"""x_combo_endfire_cdr_gtcrn_auto (experimental, real:run 2026-11-03): x_combo_endfire_cdr_gtcrn with blind x_s0v2
self-calibration when info has no calib_json (else identical to x_combo_endfire_cdr_gtcrn)."""
from methods import x_combo_endfire_cdr_gtcrn as _base
from methods import x_combo_s1beam as _sb


def enhance(x, fs, info, return_stages=False):
    return _base.enhance(x, fs, _sb.calibrated_info(x, fs, info), return_stages=return_stages)
