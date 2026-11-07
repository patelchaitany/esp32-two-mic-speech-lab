"""Shared helpers of the x_combo_* stacks (experimental, x_ prefix; real:run pass, 2026-11-03). Not a method.

gn_fix(x, fs, info): when info['calib_json'] is a dict without 'gamma_n', add x_s0v2.gamma_n_for(...) (the noise
coherence measured after the json's own skew/corner correction; see methods/x_gnfix_m4_cdr.py), else unchanged.
cdr_gain(fe): the tuned M4 DOA-free CDR gain with the DESIGN 2.1 fall limit (exactly what m4_cdr applies).
endfire_beam(fe, scenario, side): fixed WNG-constrained superdirective MVDR toward tau = side * d_hat / c (the
calibrated spacing), measured Gamma_n, the bus (or meeting) WNG floors; returns (Y (257, N), w (257, 2), df (257,)).
"""
import json

import numpy as np

from methods import x_s0v2
from methods._core import beams, common as cm, masks, ns


def gn_fix(x, fs, info):
    cfg = info.get("calib_json")
    if cfg is None:
        return info
    if not isinstance(cfg, dict):
        with open(cfg) as fh:
            cfg = json.load(fh)
    if "gamma_n" in cfg:
        return info
    g = x_s0v2.gamma_n_for(x, fs, cfg)
    cfg = dict(cfg, gamma_n={"f_hz": [float(v) for v in x_s0v2.F], "gamma": [float(v) for v in g]})
    return dict(info, calib_json=cfg)


def cdr_gain(fe):
    cal = fe.calib if fe.schedule is None else fe.schedule
    G = masks.cdr_gain(fe.X, cal, None)
    return ns.limit_gain(G, floor_db=None, fall_db=masks.FALL_DB)


def endfire_beam(fe, scenario="bus", side=+1.0):
    cal = fe.calib
    tau = side * float(cal.d_hat) / cm.C
    a = cm.steer(cm.FREQS, tau)                                   # (257, 2), centre-referenced
    g = beams.noise_coherence(cal, "measured")
    w, _ = beams.mvdr_weights(a, g, beams.wng_floor(scenario))
    Y = np.conj(w[:, 0])[:, None] * fe.X[0] + np.conj(w[:, 1])[:, None] * fe.X[1]
    df = 1.0 / beams.noise_power(w, g)
    return Y, w, df


def stages(y, pre=None, return_stages=False):
    y = np.asarray(y, np.float64)
    if not return_stages:
        return y
    return {"out": y, "pre_agc": y if pre is None else np.asarray(pre, np.float64)}


def enhance(x, fs, info, return_stages=False):
    """Not a stack: this module only holds helpers. As a method it returns the S0-prefiltered Mic A (= m0-like)."""
    from methods._core import calib
    return stages(calib.prefilter(np.asarray(x, np.float64)[0], info.get("scenario", "bus")), None, return_stages)
