"""M2 meeting beam (DESIGN.md M2, 2.2): S0 -> S1 -> superdirective MVDR toward the active talker -> ISTFT.

mode 'steered' (default): WNG >= -3 dB MVDR (S0's measured Gamma_n) steered to the active talker's TDOA (weighted
median of S1's tau over 300 ms of speech, weights vad * directness, frames within 40 us of a dominant coherent
noise source such as the projector fan excluded) on a 64-point grid over +-1.2 d_hat / c, held >= 400 ms between
talkers, 50 ms crossfades; Mic A alone before the first talker and after 0.5 s without talker evidence (a talker
S1 does not localize is never steered away from). mode 'fixed3': beams at tau = +d_hat/c, 0, -d_hat/c with
DESIGN's best-beam selection (highest smoothed 300-3000 Hz output power on speech frames, 500 ms hold, 50 ms
crossfade); in reverberant rooms it picks the broadside beam (= M1a) nearly always.
A bus scene gets the bus WNG floors. online=True runs S0/S1 causally (trailing TDOA window).
The DESIGN 2.2 stack runs WPE first: compose beams.meeting_beam(Xw, cal, tracker.track(Xw, ...)) there.
Shared code: methods/_core/beams.py (meeting_beam). Output is centre-referenced (<= 1.4 samples after Mic A).
"""
import numpy as np

from methods._core import beams, common as cm, tracker


def enhance(x, fs, info, return_stages=False, mode="steered", online=False):
    fe = tracker.front_end(x, fs, info, online=online, partners=False, talkers=False)
    beam = beams.meeting_beam(fe.X, fe.schedule if online else fe.calib, fe.track,
                              scenario=info.get("scenario", "meeting"), mode=mode)
    y = cm.istft(beam.apply(fe.X), np.shape(x)[-1])
    return {"out": y, "pre_agc": y} if return_stages else y
