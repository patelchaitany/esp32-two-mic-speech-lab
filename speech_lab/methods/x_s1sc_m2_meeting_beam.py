"""Spacing-study variant (x_): methods.m2_meeting_beam (steered) with S1's microsecond constants scaled by d / 4 cm
(methods/x_s1scale.py) and the beam's coherent-noise exclusion (beams.NOISE_EXCL, 40 us) scaled the same way.
Identical to m2_meeting_beam at 4 cm."""
import numpy as np

from methods import x_s1scale
from methods._core import beams, common as cm, tracker


def enhance(x, fs, info, return_stages=False):
    s = x_s1scale.scale_of(info)
    with x_s1scale.scaled(s):
        fe = tracker.front_end(x, fs, info, partners=False, talkers=False)
        beam = beams.meeting_beam(fe.X, fe.calib, fe.track, scenario=info.get("scenario", "meeting"), mode="steered",
                                  noise_excl=beams.NOISE_EXCL * s)
    y = cm.istft(beam.apply(fe.X), np.shape(x)[-1])
    return {"out": y, "pre_agc": y} if return_stages else y
