"""M7 standalone, offline: S0 + S1 front end -> cACGMM with directional classes at the TDOAs -d/c, 0, +d/c
and a diffuse noise class -> post-mask clip((1 - gamma_noise) / 0.5, 0.5, 1) on calibrated Mic A
(methods/_core/cacgmm.py PARAMS, tuned on the v2 dev bus set; DESIGN.md M7, 2.1).

Bus only. The tuned defaults deviate from DESIGN M7 (S1 partner anchors, per-partner Souden MVDRs,
max(0.3, gamma_A + gamma_B)); the reasons and dev numbers are in the cacgmm module docstring, and
cacgmm.DESIGN_PARAMS reproduces the DESIGN configuration. Meeting scenes pass the prefiltered Mic A
through: M7 keeps two partners (DESIGN 2.2). No level control, so 'pre_agc' equals 'out'.
"""
from methods._core import cacgmm, calib, common as cm, tracker


def enhance(x, fs, info, return_stages=False):
    if info.get("scenario", "bus") != "bus":
        xh = calib.prefilter(x, info["scenario"])
        y = xh[0]
    else:
        fe = tracker.front_end(x, fs, info)
        y = cm.istft(cacgmm.separate(fe.X, fe.calib, fe.track).Y, x.shape[1])
    return {"out": y, "pre_agc": y} if return_stages else y
