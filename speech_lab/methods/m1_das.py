"""M1 delay-and-sum (DESIGN.md M1): S0 -> S1 -> delay-and-sum by per-bin phase rotation -> ISTFT.

mode 'steered' (M1b, default): Y = a(f, tau)^H X / 2 toward the active partner (bus: the partner with the higher
S1 posterior above 0.25, at its S1 TDOA, only with two partner peaks within 1.1 d_hat / c on opposite sides of
broadside) or the active talker (meeting: weighted median of S1's TDOA over 300 ms of speech, coherent-noise TDOA
excluded, 64-point grid), held >= 300 ms, 50 ms crossfades; Mic A alone before the first one and after 0.5 s
without evidence. mode 'broadside' (M1a): (X_A + X_B) / 2 after S0. online=True runs S0/S1 causally.
Shared code: methods/_core/beams.py (das_beam). Output is centre-referenced: a source at tau comes out tau/2
after Mic A (<= 1.4 samples at 6 cm).
"""
import numpy as np

from methods._core import beams, common as cm, tracker


def enhance(x, fs, info, return_stages=False, mode="steered", online=False):
    scenario = info.get("scenario", "bus")
    fe = tracker.front_end(x, fs, info, online=online, partners=scenario == "bus", talkers=False)
    beam = beams.das_beam(fe.X, fe.schedule if online else fe.calib, fe.track, scenario, mode=mode)
    y = cm.istft(beam.apply(fe.X), np.shape(x)[-1])
    return {"out": y, "pre_agc": y} if return_stages else y
