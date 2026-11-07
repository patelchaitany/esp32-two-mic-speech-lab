"""Spacing-study single-channel reference stack (x_, study/SPACING_STUDY.md): S0-prefiltered Mic A -> M8 GTCRN
(DESIGN attenuation limit) -> M11 leveling (methods._core.agc.level with the S1 track of the two mics, offline).
The signal path is Mic A only; the second mic only feeds S1's talker cues to the leveler, as in m11_level.
Pair with x_sp_meet_stack (beam first) to measure what the array adds at 12 / 40 mm.
"""
import numpy as np

from methods._core import agc, nn, tracker


def enhance(x, fs, info, return_stages=False):
    scenario = info.get("scenario", "meeting")
    fe = tracker.front_end(np.asarray(x, dtype=np.float64), fs, info)
    y_nn = nn.denoise(fe.x[0], "gtcrn", atten_db=nn.ATTEN_DB[scenario], level="file")
    res = agc.level(y_nn, fe.track, scenario)
    return {"out": res.out, "pre_agc": res.pre} if return_stages else res.out
