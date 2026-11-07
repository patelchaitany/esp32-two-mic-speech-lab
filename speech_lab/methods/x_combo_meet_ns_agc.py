"""x_combo_meet_ns_agc (experimental stack, real:run 2026-11-03): as x_combo_meet_chain without the beam and with the
classical M6 noise suppressor instead of the NN: S0 -> 2-ch WPE -> S1 -> Mic A (WPE output) -> M6 MMSE-SPP/LSA
(ns.suppress, meeting parameters) -> M11 leveling. Returns {'out': leveled, 'pre_agc': NS output}.
"""
import numpy as np

from methods import x_combo_common as xc
from methods import x_combo_meet_chain as mc
from methods._core import agc, common as cm, ns


def enhance(x, fs, info, return_stages=False):
    T = np.shape(x)[-1]
    scen, xh, cal, Xw, trk = mc.front(x, fs, info)
    y_ns = cm.istft(ns.suppress(Xw[0], scen, fs=fs)["S"], T)
    res = agc.level(y_ns, trk, scen, Y=cm.stft(y_ns))
    return xc.stages(res.out, res.pre, return_stages)
