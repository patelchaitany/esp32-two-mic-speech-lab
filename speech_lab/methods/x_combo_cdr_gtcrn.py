"""x_combo_cdr_gtcrn (experimental stack, real:run 2026-11-03): S0 (+ gamma_n repair) -> M4 CDR gain on Mic A ->
M8 GTCRN on the CDR output -> ONE combined lab-grid mask clip(G_cdr * mask_gtcrn_raw, a, 1), a = -LIMIT_DB.

Why: on the 1.19 cm board the spatial chains barely move the SNR (M1/M2 fall back to Mic A because S1 puts both
partners at one TDOA); M4's DOA-free CDR is the one spatial cue that still works (direct sound vs the diffuse fan /
bus noise), and M8 handles stationary noise. The combined limit (15 dB bus, 12 dB meeting) bounds speech damage.
gamma_n repair: as methods/x_gnfix_m4_cdr.py (only when a calib_json without gamma_n is given).
"""
import numpy as np

from methods import x_combo_common as xc
from methods._core import common as cm, nn, tracker

LIMIT_DB = {"bus": 15.0, "meeting": 12.0}


def enhance(x, fs, info, return_stages=False):
    scen = info.get("scenario", "bus")
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    info = xc.gn_fix(x, fs, info)
    fe = tracker.front_end(x, fs, info, partners=False, talkers=False)
    G = xc.cdr_gain(fe)
    y_cdr = cm.istft(G * fe.X[0], T)
    pf = nn.post_filter(y_cdr, "gtcrn", atten_db=None)
    M = np.clip(G * pf["mask_raw"], 10.0 ** (-LIMIT_DB.get(scen, 15.0) / 20.0), 1.0)
    return xc.stages(cm.istft(M * fe.X[0], T), None, return_stages)
