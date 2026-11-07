"""x_combo_endfire_dpdf (experimental, real:run 2026-11-03; GEOMETRY PRIOR as x_combo_endfire): S0 (+ gamma_n repair)
-> fixed endfire MVDR toward +d_hat/c -> M8 DPDFNet-4 post-filter with the M8 attenuation limit (20 dB bus,
12 dB meeting) -> meeting only: M11 leveling (agc.level with the S1 track of the calibrated front end).
Returns {'out', 'pre_agc'} (pre_agc = DPDFNet output). Why: on the real meeting the endfire beam was the only method
with a positive worst-talker dESTOI and DPDFNet the best mean dESTOI; this stacks the two (and levels the talkers).
"""
import numpy as np

from methods import x_combo_common as xc
from methods._core import agc, common as cm, nn, tracker


def enhance(x, fs, info, return_stages=False):
    scen = info.get("scenario", "bus")
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    info = xc.gn_fix(x, fs, info)
    fe = tracker.front_end(x, fs, info, partners=(scen == "bus"), talkers=(scen == "meeting"))
    Y, _, _ = xc.endfire_beam(fe, scen, +1.0)
    yb = cm.istft(Y, T)
    y = nn.post_filter(yb, "dpdfnet", atten_db=nn.ATTEN_DB.get(scen, 12.0))["y"]
    if scen != "meeting":
        return xc.stages(y, None, return_stages)
    res = agc.level(y, fe.track, scen, Y=cm.stft(y))
    return xc.stages(res.out, res.pre, return_stages)
