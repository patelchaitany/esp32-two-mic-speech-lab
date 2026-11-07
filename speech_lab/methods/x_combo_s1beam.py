"""x_combo_s1beam (experimental, real:run 2026-11-03): BLIND one-sided beam. Calibration: info['calib_json'] completed
with a correctly rotated gamma_n (x_combo_common.gn_fix) when given, else the x_s0v2 blind estimate (as x_auto_*).
S1 bus partner model on that front end, then:
  * both partners on ONE side of broadside (S1 'one_peak', or tau_A, tau_B with equal signs, |tau| >= 0.3 d_hat/c):
    one WNG-constrained superdirective MVDR (bus floors) steered at tau* = mean(tau_A, tau_B) clipped to +-d_hat/c;
  * partners on opposite sides: M2's two-talker beam (beams.bus_beam, states 'ab');
  * no partner TDOA or both near broadside: Mic A (= the M2 fallback).
Why: m2_bus_beam falls back to Mic A whenever both partners are on one side (the case of every real take tonight),
and with calib_session.json S1 placed both partners at ~0 because calib._override re-blends a gamma_n rotated by the
blind skew (gamma_n 0.12 at 4 kHz instead of 0.83); with the repaired gamma_n S1 reads +0.61..+0.63 samples
(true direct path +0.55). diag in return_stages: 'x_s1beam' = {mode, tau_samples, tau_A, tau_B, why}.
"""
import numpy as np

from methods import x_combo_common as xc
from methods import x_s0v2
from methods._core import beams, common as cm, tracker

SAME_SIDE_FRAC = 0.3


def calibrated_info(x, fs, info):
    if info.get("calib_json") is None:
        return dict(info, calib_json=x_s0v2.estimate(x, fs, info))
    return xc.gn_fix(x, fs, info)


def beam_output(x, fs, info):
    """-> (fe, Y (257, N) beam output on the calibrated STFT, diag)."""
    scen = info.get("scenario", "bus")
    info = calibrated_info(x, fs, info)
    fe = tracker.front_end(x, fs, info, partners=True, talkers=False)
    cal, trk = fe.calib, fe.track
    dc = float(cal.d_hat) / cm.C
    ta, tb = float(trk.tau_A), float(trk.tau_B)
    diag = {"tau_A": ta * fs, "tau_B": tb * fs, "why": list(getattr(trk, "inseparable_why", ()))}
    if not (np.isfinite(ta) and np.isfinite(tb)):
        diag.update(mode="mic_a", tau_samples=None)
        return fe, fe.X[0].copy(), diag
    one_side = (ta == tb or np.sign(ta) == np.sign(tb)) and min(abs(ta), abs(tb)) >= SAME_SIDE_FRAC * dc
    if one_side:
        tau = float(np.clip(0.5 * (ta + tb), -dc, dc))
        a = cm.steer(cm.FREQS, tau)
        g = beams.noise_coherence(cal, "measured")
        w, _ = beams.mvdr_weights(a, g, beams.wng_floor(scen))
        Y = np.conj(w[:, 0])[:, None] * fe.X[0] + np.conj(w[:, 1])[:, None] * fe.X[1]
        diag.update(mode="one_side", tau_samples=tau * fs)
        return fe, Y, diag
    if np.sign(ta) != np.sign(tb) and min(abs(ta), abs(tb)) >= SAME_SIDE_FRAC * dc:
        beam = beams.bus_beam(fe.X, cal, trk, scenario=scen, states="ab")
        diag.update(mode="two_talker", tau_samples=None)
        return fe, beam.apply(fe.X), diag
    diag.update(mode="mic_a", tau_samples=None)
    return fe, fe.X[0].copy(), diag


def enhance(x, fs, info, return_stages=False):
    x = np.asarray(x, np.float64)
    fe, Y, diag = beam_output(x, fs, info)
    y = cm.istft(Y, x.shape[-1])
    if not return_stages:
        return y
    return {"out": y, "pre_agc": y, "x_s1beam": diag}
