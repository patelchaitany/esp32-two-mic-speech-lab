"""x_combo_s1beam_dpdf (experimental, real:run 2026-11-03): BLIND counterpart of x_combo_endfire_dpdf. x_combo_s1beam
(calib_json + gamma_n repair, or x_s0v2 self-calibration; one-sided MVDR at S1's partner TDOA when both partners are on
one side, M2's two-talker beam when on opposite sides, Mic A otherwise) -> M8 DPDFNet-4 with the M8 attenuation limit
(20 dB bus, 12 dB meeting). No AGC."""
import numpy as np

from methods import x_combo_common as xc
from methods import x_combo_s1beam as sb
from methods._core import common as cm, nn


def enhance(x, fs, info, return_stages=False):
    scen = info.get("scenario", "bus")
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    fe, Y, diag = sb.beam_output(x, fs, info)
    y = nn.post_filter(cm.istft(Y, T), "dpdfnet", atten_db=nn.ATTEN_DB.get(scen, 20.0))["y"]
    if not return_stages:
        return y
    return {"out": y, "pre_agc": y, "x_s1beam": diag}
