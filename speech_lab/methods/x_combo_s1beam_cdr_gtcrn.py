"""x_combo_s1beam_cdr_gtcrn (experimental, real:run 2026-11-03): x_combo_s1beam (blind one-sided / two-talker beam)
-> CDR gain (M4, from the calibrated mic pair) x GTCRN mask (on the CDR-filtered beam output) -> one combined mask
clip(., a, 1) on the beam output, attenuation limit 15 dB (bus) / 12 dB (meeting). The blind counterpart of
x_combo_endfire_cdr_gtcrn ('best spatial chain -> M8 post-filter with an attenuation limit')."""
import numpy as np

from methods import x_combo_common as xc
from methods import x_combo_s1beam as sb
from methods._core import common as cm, nn

LIMIT_DB = {"bus": 15.0, "meeting": 12.0}


def enhance(x, fs, info, return_stages=False):
    scen = info.get("scenario", "bus")
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    fe, Y, diag = sb.beam_output(x, fs, info)
    G = xc.cdr_gain(fe)
    pf = nn.post_filter(cm.istft(G * Y, T), "gtcrn", atten_db=None)
    M = np.clip(G * pf["mask_raw"], 10.0 ** (-LIMIT_DB.get(scen, 15.0) / 20.0), 1.0)
    y = cm.istft(M * Y, T)
    if not return_stages:
        return y
    return {"out": y, "pre_agc": y, "x_s1beam": diag}
