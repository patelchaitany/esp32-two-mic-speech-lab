"""Spacing-study meeting stack (x_, study/SPACING_STUDY.md): S0 + S1 -> M2 meeting beam (steered superdirective
MVDR, methods._core.beams.meeting_beam as in m2_meeting_beam) -> M8 GTCRN post-filter (12 dB limit) -> M11 leveling
(methods._core.agc.level with the same S1 track, offline). The DESIGN 2.2 order without WPE.
Compare with x_sp_mica_stack (the same GTCRN + M11 chain on Mic A alone) to see what the array adds at 12 / 40 mm.
Bus scenes: M2 bus-scenario beam weights are not used here; the bus gets the same chain with the bus AGC.
"""
import numpy as np

from methods._core import agc, beams, common as cm, nn, tracker


def enhance(x, fs, info, return_stages=False):
    scenario = info.get("scenario", "meeting")
    fe = tracker.front_end(np.asarray(x, dtype=np.float64), fs, info)
    beam = beams.meeting_beam(fe.X, fe.calib, fe.track, scenario=scenario, mode="steered")
    y_lin = cm.istft(beam.apply(fe.X), fe.x.shape[1])
    y_nn = nn.denoise(y_lin, "gtcrn", atten_db=nn.ATTEN_DB[scenario], level="file")
    res = agc.level(y_nn, fe.track, scenario)
    return {"out": res.out, "pre_agc": res.pre} if return_stages else res.out
