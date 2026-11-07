"""x_combo_endfire (experimental, real:run 2026-11-03; GEOMETRY PRIOR, not blind): S0 (+ gamma_n repair) -> FIXED
WNG-constrained superdirective MVDR toward tau = +d_hat/c (Mic A's endfire side), measured Gamma_n, bus WNG floors
(-15/-10/-6 dB; meeting -3 dB) -> ISTFT. No S1.

Why: on the real set both partners (and all meeting talkers) sit near endfire on Mic A's side (direct-path TDOA
+0.55 samples = +d/c), but S1 places them at ~0, so M1/M2 steer nowhere (M2's 'same side' fallback = Mic A). This
stack asks what a 1.19 cm pair can give when the device is pointed at the partners (deployment rule 'Mic A end
toward the people you talk to'). It uses that rule as a prior; it is a diagnostic of S1, not a blind method.
"""
import numpy as np

from methods import x_combo_common as xc
from methods._core import common as cm, tracker


def enhance(x, fs, info, return_stages=False):
    scen = info.get("scenario", "bus")
    x = np.asarray(x, np.float64)
    info = xc.gn_fix(x, fs, info)
    fe = tracker.front_end(x, fs, info, partners=False, talkers=False)
    Y, _, _ = xc.endfire_beam(fe, scen, +1.0)
    return xc.stages(cm.istft(Y, x.shape[-1]), None, return_stages)
