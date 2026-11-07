"""M2 bus beam (DESIGN.md M2, 2.1): S0 -> S1 partner model -> WNG-constrained two-talker beam -> ISTFT.

Per configuration (partner TDOAs tau_A >= tau_B from S1, S0's measured Gamma_n): the two-talker soft-LCMV w_AB
(h = [1, -1] where the dipole meets the floor, smooth phase continuation elsewhere) keeps both partners at unit
gain. states='ab' (default, tuned on v2 dev) uses it throughout; states='s1' is the DESIGN state machine (w_A /
w_AB / h_B* w_B from S1's posteriors: enter A after p_A > 0.9 and p_B < 0.2 for 300 ms, AB as soon as the other
posterior reaches 0.2, per-bin phase-continuity substitution, 50 ms crossfades). WNG floors -10 dB up to 2 kHz,
-6 dB above (a meeting scene gets -3 dB). Fallback (logged): S1 flags the partners inseparable, a partner TDOA lies
beyond 1.1 d_hat / c, or both lie on one side of broadside -> broadside MVDR (= average) when both are near
broadside, else Mic A alone.
online=True runs S0/S1 causally and redesigns when S1's TDOAs or S0's block change.
Shared code: methods/_core/beams.py (bus_beam). Output is centre-referenced (a partner at tau comes out tau/2
after Mic A, <= 1.4 samples at 6 cm).
"""
import logging

import numpy as np

from methods._core import beams, common as cm, tracker

log = logging.getLogger(__name__)


def enhance(x, fs, info, return_stages=False, online=False, states="ab"):
    fe = tracker.front_end(x, fs, info, online=online, partners=True, talkers=False)
    beam = beams.bus_beam(fe.X, fe.schedule if online else fe.calib, fe.track, scenario=info.get("scenario", "bus"),
                          states=states)
    if beam.diag["fallback_frac"] > 0:
        cfg = [c for c in beam.diag["configs"] if c["fallback"]]
        log.info("partners spatially inseparable or not localized %.0f%% of the time (%s -> %s); rotate the device "
                 "so that its axis points through both partners", 100 * beam.diag["fallback_frac"], cfg[0]["why"],
                 cfg[0]["fallback"])
    y = cm.istft(beam.apply(fe.X), np.shape(x)[-1])
    return {"out": y, "pre_agc": y} if return_stages else y
