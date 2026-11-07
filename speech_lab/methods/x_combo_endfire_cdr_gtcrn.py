"""x_combo_endfire_cdr_gtcrn (experimental, real:run 2026-11-03; GEOMETRY PRIOR as x_combo_endfire): fixed endfire
MVDR toward +d_hat/c -> CDR post-filter (M4 gain from the mic pair) x GTCRN mask (computed on the CDR-filtered beam
output) -> one combined mask clip(., a, 1) on the beam output, limit 15 dB bus / 12 dB meeting.
The 'best spatial chain -> M8 post-filter with an attenuation limit' stack of the real:run task, with the endfire
beam standing in for the S1-steered beams (which fall back to Mic A on this board).
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
    Y, _, _ = xc.endfire_beam(fe, scen, +1.0)
    G = xc.cdr_gain(fe)
    y1 = cm.istft(G * Y, T)
    pf = nn.post_filter(y1, "gtcrn", atten_db=None)
    M = np.clip(G * pf["mask_raw"], 10.0 ** (-LIMIT_DB.get(scen, 15.0) / 20.0), 1.0)
    return xc.stages(cm.istft(M * Y, T), None, return_stages)
