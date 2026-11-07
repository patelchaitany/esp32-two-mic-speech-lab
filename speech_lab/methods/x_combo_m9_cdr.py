"""x_combo_m9_cdr (experimental, real:run 2026-11-03): M9 VoxLock x2 union (explicit enrollment from info['enroll'],
self-enrollment fallback as m9_voxlock2) times the M4 CDR gain (gamma_n-repaired S0) on Mic A, combined floor 20 dB.
'Spatial -> M9 union of both partners': the CDR gain (direct vs diffuse) and the TSE union mask are both lab-grid
gains on Mic A, so the stack multiplies them (order-free) and limits the product. Bus only; meeting = m9 (Mic A).
"""
import numpy as np

from methods import x_combo_common as xc
from methods._core import common as cm, tracker, tse

LIMIT_DB = 20.0


def enhance(x, fs, info, return_stages=False):
    scen = info.get("scenario", "bus")
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    r = tse.enhance(x, fs, info, mode="explicit", return_stages=False)
    if scen != "bus" or T == 0:
        return xc.stages(r, None, return_stages)
    info2 = xc.gn_fix(x, fs, info)
    fe = tracker.front_end(x, fs, info2, partners=False, talkers=False)
    G = xc.cdr_gain(fe)
    Y9 = cm.stft(np.asarray(r, np.float64))
    XA = fe.X[0]
    m9 = np.clip(np.abs(Y9) / np.maximum(np.abs(XA), 1e-10), 0.0, 1.0)
    M = np.clip(G * m9, 10.0 ** (-LIMIT_DB / 20.0), 1.0)
    return xc.stages(cm.istft(M * XA, T), None, return_stages)
