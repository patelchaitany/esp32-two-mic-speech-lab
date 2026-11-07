#!/usr/bin/env python
"""B1 - two-mic physics numbers versus spacing, with the exact DESIGN.md v1.1 section-1 definitions.

Step 1 reproduces the published 2.5 / 4 / 6 cm values (validation rows).
Step 2 extends every quantity to d = 1.0, 1.2, 1.5, 2.0 cm (fs 16 kHz, c 343 m/s).

Definitions are copied from the scripts that produced the published numbers
(scratchpad arch/physics_check.py, revise/physics_check_v2.py, review/di_real.py, review/probe.py):
  * AI-DI = sum_i I_i * DI(f_i) [dB], ANSI S3.5 one-third-octave band-importance weights, 18 bands
    160 Hz .. 8 kHz (8 kHz centre evaluated at 7.9 kHz), DI evaluated at the band centre only.
  * 3-D diffuse: Gamma = sinc(2 f d / c) (numpy sinc); 2-D (horizontal) diffuse: Gamma = J0(2 pi f d / c).
  * WNG = |w^H a|^2 / (w^H w)  (0 dB = one omni mic, delay-and-sum = +3 dB).
  * Single-target beam: per-band MVDR against the diffuse model, w = (G + mu I)^-1 a / (a^H (G + mu I)^-1 a),
    mu = smallest value meeting the WNG floor (bisection on log10 mu in [-9, 6]).
  * Two-talker beam: soft LCMV, centre-referenced steering, A at 0 deg, B at 180 deg, h = [1, e^{j psi}],
    lambda_c = 1e3, psi on a 2.5 deg grid, strict WNG floor (violating candidates rejected), |r_A|/|r_B|
    within 1 dB, best DF kept, single omni (0 dB) where nothing qualifies; weights designed for the 3-D model.
  * "On the axis" rows: 2-D numbers use J0-designed weights. Azimuth-averaged, broadside and two-talker rows:
    2-D numbers use 3-D-designed weights evaluated against J0.
  * Azimuth average: mean of the AI-DI over 19 angles 0..90 deg (folded azimuth, uniform).
  * Dipole band: WNG = 1 - cos(2 pi f d / c) >= floor, contiguous up to min(c/2d, fs/2).
  * CDR usability: |Gamma_s - Gamma_n| >= 0.2 with Gamma_s = exp(+j 2 pi f tau), Gamma_n = 3-D sinc.
  * Meeting TDOA groups: 20 seats of lab scene meeting_s0 (8 m x 2 m table), mic axis horizontal at a uniformly
    random azimuth, tau = (|s-B| - |s-A|)/c, greedy 1-D grouping (new group when tau - group_start > resolution).

Run:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$LAB/.venv/bin/python" -I -B physics_b1.py "$LAB"
Writes physics.json and tables.md next to this script.  No audio, no serial port, analytic only (~seconds).
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0

T0 = time.process_time()
LAB = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

C = 343.0
FS = 16000.0
FC = np.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 7900.])
IW = np.array([.0083, .0095, .0150, .0289, .0440, .0578, .0653, .0711, .0818, .0844, .0882, .0898, .0868, .0844,
               .0771, .0527, .0364, .0185])
assert abs(IW.sum() - 1.0) < 1e-9
D_PUB = (0.025, 0.04, 0.06)
D_ALL = (0.010, 0.012, 0.015, 0.020, 0.025, 0.040, 0.060)
SKEW_US = 68.0                      # measured inter-channel skew magnitude (hardware session 1 fits, ~1.1 samples)
PSI = np.radians(np.arange(0, 360, 2.5))
BUS_FLOOR = np.where(FC < 500, -15.0, np.where(FC <= 2000, -10.0, -6.0))   # DESIGN M2 bus WNG schedule


# ----------------------------------------------------------------------------------------------------------- models
def coh(f, d, kind):
    x = 2.0 * np.asarray(f, float) * d / C
    return np.sinc(x) if kind == "3d" else j0(np.pi * x)


def steer_c(f, tau):
    """centre-referenced a(f, tau) = [exp(+j pi f tau), exp(-j pi f tau)] (DESIGN 2.0)"""
    p = np.exp(1j * np.pi * np.asarray(f, float) * tau)
    return p, np.conj(p)


def steer_m(f, tau):
    """Mic-A-referenced [1, exp(-j 2 pi f tau)] -- only for the convention check of M2"""
    f = np.asarray(f, float)
    return np.ones(np.broadcast(f, tau).shape, complex), np.exp(-2j * np.pi * f * tau)


def resp(w1, w2, a1, a2):
    return np.conj(w1) * a1 + np.conj(w2) * a2


def quad(w1, w2, g):
    """w^H G w for G = [[1, g], [g*, 1]] (g may be complex)"""
    return np.real(np.abs(w1) ** 2 + np.abs(w2) ** 2 + 2 * np.real(np.conj(w1) * g * w2))


def wng_db(w1, w2, a1, a2):
    return 10 * np.log10(np.abs(resp(w1, w2, a1, a2)) ** 2 / (np.abs(w1) ** 2 + np.abs(w2) ** 2) + 1e-300)


def aidi(di_bands):
    return np.sum(IW * di_bands, axis=-1)


def mvdr(f, tau, d, wmin_db, design="3d", n_iter=60):
    """Single-target MVDR with WNG floor; exact replica of physics_check.mvdr / di_real.mvdr (vectorised).
    f, tau, wmin_db broadcastable.  Returns (w1, w2, a1, a2)."""
    f, tau, wmin = np.broadcast_arrays(np.asarray(f, float), np.asarray(tau, float), np.asarray(wmin_db, float))
    g = coh(f, d, design)
    a1, a2 = steer_c(f, tau)

    def w_of(mu):
        det = (1 + mu) ** 2 - g ** 2
        x1 = ((1 + mu) * a1 - g * a2) / det
        x2 = ((1 + mu) * a2 - g * a1) / det
        den = np.conj(a1) * x1 + np.conj(a2) * x2
        return x1 / den, x2 / den

    w0 = w_of(1e-9)
    ok0 = wng_db(*w0, a1, a2) >= wmin
    lo = np.full(f.shape, -9.0)
    hi = np.full(f.shape, 6.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        good = wng_db(*w_of(10 ** mid), a1, a2) >= wmin
        hi = np.where(good, mid, hi)
        lo = np.where(good, lo, mid)
    wb = w_of(10 ** hi)
    return np.where(ok0, w0[0], wb[0]), np.where(ok0, w0[1], wb[1]), a1, a2


def di_db(w1, w2, a1, a2, f, d, kind, D=None):
    """10 log10 |w^H D a|^2 / (w^H D G D^H w); D = diag(1, e^{j phi}) models a channel-B error"""
    g = coh(f, d, kind)
    if D is not None:
        a2 = a2 * D
        g = g * np.conj(D)
    return 10 * np.log10(np.abs(resp(w1, w2, a1, a2)) ** 2 / quad(w1, w2, g))


def single_target(d, theta_deg, wmin_db, design, evals, f=FC):
    th = np.radians(np.asarray(theta_deg, float))[..., None]
    tau = d * np.cos(th) / C
    w1, w2, a1, a2 = mvdr(f, tau, d, wmin_db, design)
    return {k: di_db(w1, w2, a1, a2, f, d, k) for k in evals}, wng_db(w1, w2, a1, a2)


def dsb(d, theta_deg, kind, f=FC):
    tau = d * np.cos(np.radians(theta_deg)) / C
    a1, a2 = steer_c(f, tau)
    return di_db(a1 / 2, a2 / 2, a1, a2, f, d, kind)


def two_talker(d, wmin_db, evals=("3d", "2d"), steer=steer_c, psis=PSI, lam=1e3, n_iter=50, f=FC):
    """Exact replica of physics_check_v2.two_talker (vectorised over band x psi)."""
    F = f[:, None] * np.ones((1, len(psis)))
    P = np.ones((len(f), 1)) * np.asarray(psis)[None, :]
    wmin = np.broadcast_to(np.asarray(wmin_db, float)[..., None] if np.ndim(wmin_db) else wmin_db, F.shape)
    g = coh(F, d, "3d")                                  # design model is always 3-D
    aA1, aA2 = steer(F, d / C)
    aB1, aB2 = steer(F, -d / C)
    m11 = lam * (np.abs(aA1) ** 2 + np.abs(aB1) ** 2) + 1.0
    m22 = lam * (np.abs(aA2) ** 2 + np.abs(aB2) ** 2) + 1.0
    m12 = lam * (aA1 * np.conj(aA2) + aB1 * np.conj(aB2)) + g
    m21 = np.conj(m12)
    hc = np.exp(-1j * P)                                 # h* with h = [1, e^{j psi}]
    r1 = lam * (aA1 + hc * aB1)
    r2 = lam * (aA2 + hc * aB2)

    def w_of(mu):
        a, dd = m11 + mu, m22 + mu
        det = a * dd - m12 * m21
        return (dd * r1 - m12 * r2) / det, (-m21 * r1 + a * r2) / det

    def wng(w):
        return wng_db(w[0], w[1], aA1, aA2)

    w0 = w_of(1e-9)
    ok0 = wng(w0) >= wmin
    feas = wng(w_of(1e6)) >= wmin - 0.01
    lo = np.full(F.shape, -9.0)
    hi = np.full(F.shape, 6.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        good = wng(w_of(10 ** mid)) >= wmin
        hi = np.where(good, mid, hi)
        lo = np.where(good, lo, mid)
    wb = w_of(10 ** hi)
    okb = feas & (wng(wb) >= wmin - 0.01)
    w1 = np.where(ok0, w0[0], wb[0])
    w2 = np.where(ok0, w0[1], wb[1])
    valid = ok0 | okb
    rA, rB = resp(w1, w2, aA1, aA2), resp(w1, w2, aB1, aB2)
    bal = 20 * np.log10(np.minimum(np.abs(rA), np.abs(rB)) / np.maximum(np.abs(rA), np.abs(rB)) + 1e-30)
    cand = valid & (bal > -1.0)
    wng_all = wng((w1, w2))
    out = {}
    for kind in evals:
        di = 10 * np.log10(0.5 * (np.abs(rA) ** 2 + np.abs(rB) ** 2) / quad(w1, w2, coh(F, d, kind)))
        dm = np.where(cand, di, -np.inf)
        ib = dm.argmax(axis=1)
        best = dm[np.arange(len(f)), ib]
        feasible = np.isfinite(best)
        band = np.where(feasible, best, 0.0)
        psi_best = np.where(feasible, np.degrees(np.angle(rB / rA))[np.arange(len(f)), ib] % 360, np.nan)
        out[kind] = dict(aidi=float(aidi(band)), band_di=band, psi_deg=psi_best, feasible=feasible,
                         wng=np.where(feasible, wng_all[np.arange(len(f)), ib], np.nan))
    return out


def dipole_band(d, floor_db):
    """lower edge of the contiguous band ending at min(c/2d, fs/2) where 1 - cos(2 pi f d / c) >= floor(f)."""
    top = min(C / (2 * d), FS / 2)
    f = np.arange(20.0, top + 0.05, 0.1)
    fl = floor_db(f) if callable(floor_db) else np.full(f.shape, float(floor_db))
    ok = 10 * np.log10(1 - np.cos(2 * np.pi * f * d / C) + 1e-300) >= fl
    if not ok[-1]:
        return None
    bad = np.nonzero(~ok)[0]
    return float(f[bad[-1] + 1]) if len(bad) else float(f[0])


def dipole_edge_closed(d, floor_db):
    return float(np.arccos(1 - 10 ** (floor_db / 10)) * C / (2 * np.pi * d))


def cdr_edge(d, kind, where):
    """lowest f with |Gamma_s - Gamma_n| >= 0.2 (function is increasing up to well above the root)"""
    tau = 0.0 if where == "broadside" else d / C
    h = lambda f: abs(np.exp(2j * np.pi * f * tau) - coh(f, d, kind)) - 0.2
    return float(brentq(h, 1.0, 0.45 * C / d))


def coh_threshold(d, kind, gval):
    hi = C / (2 * d) if kind == "3d" else 0.38 * C / d
    return float(brentq(lambda f: coh(f, d, kind) - gval, 1.0, hi))


# --------------------------------------------------------------------------------------------- meeting TDOA groups
def load_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def tdoas(seats, A, B):
    return (np.linalg.norm(seats - B, axis=-1) - np.linalg.norm(seats - A, axis=-1)) / C


def greedy_groups(tau_sorted, res):
    k, start = 1, tau_sorted[0]
    for t in tau_sorted[1:]:
        if t - start > res:
            k += 1
            start = t
    return k


def groups_over_azimuth(seats, dev, d, res, azs):
    ks, gaps10 = [], []
    for az in azs:
        u = np.array([np.cos(az), np.sin(az), 0.0]) * d / 2
        t = np.sort(tdoas(seats, dev + u, dev - u))
        ks.append(greedy_groups(t, res))
        gaps10.append(int(np.sum(np.diff(t) < 10e-6)))
    return np.array(ks), np.array(gaps10)


# ============================================================================================================ main
R = {"meta": dict(c=C, fs=FS, band_centres_hz=FC.tolist(), band_importance=IW.tolist(), spacings_m=list(D_ALL),
                  skew_us=SKEW_US, script=os.path.abspath(__file__))}
val = []


def check(q, design, got, tol, fmt="{:.2f}", unit="dB", note=""):
    """tol: scalar or per-element (the stated rounding of each design cell, or 0.2 dB for AI-DI rows)"""
    g = np.atleast_1d(np.asarray(got, float))
    dsg = np.atleast_1d(np.asarray(design, float))
    ok = bool(np.all(np.abs(g - dsg) <= np.asarray(tol, float) + 1e-9))
    rep_s = " / ".join(fmt.format(v) for v in g) + (f" {unit}" if unit else "")
    val.append(dict(quantity=q, design_value=" / ".join(fmt.format(v) for v in dsg) + (f" {unit}" if unit else ""),
                    reproduced=rep_s + (f" ({note})" if note else ""), ok=ok,
                    tolerance=np.asarray(tol, float).tolist()))
    return ok


# ---- 1.1 geometry
geo = {}
for d in D_ALL:
    geo[d] = dict(alias_hz=C / (2 * d), c4d_hz=C / (4 * d), max_tdoa_us=d / C * 1e6, max_tdoa_samples=d / C * FS,
                  sinc_09_07_05_hz=[coh_threshold(d, "3d", g) for g in (0.9, 0.7, 0.5)],
                  j0_09_07_05_hz=[coh_threshold(d, "2d", g) for g in (0.9, 0.7, 0.5)])
check("1.1 c/2d aliasing (kHz) 2.5/4/6 cm", [6.86, 4.29, 2.86], [geo[d]["alias_hz"] / 1e3 for d in D_PUB], 0.005, unit="kHz")
check("1.1 max TDOA (us) 2.5/4/6 cm", [72.9, 116.6, 174.9], [geo[d]["max_tdoa_us"] for d in D_PUB], 0.05, "{:.1f}", "us")
check("1.1 max TDOA (samples) 2.5/4/6 cm", [1.17, 1.87, 2.80], [geo[d]["max_tdoa_samples"] for d in D_PUB], 0.005, unit="smp")
check("1.1 sinc coherence > 0.9 below (kHz) 2.5/4/6 cm", [1.72, 1.07, 0.72],
      [geo[d]["sinc_09_07_05_hz"][0] / 1e3 for d in D_PUB], 0.005, unit="kHz")
check("1.1 sinc coherence > 0.7 / 0.5 below at 4 cm (Hz; design 1.93 / 2.59 kHz, stated rounding +-5 Hz)", [1930, 2590],
      geo[0.04]["sinc_09_07_05_hz"][1:], 5.0, "{:.1f}", unit="Hz",
      note="1924.6 Hz is 1.92 kHz: last-digit slip in the DESIGN 1.1 table; the same formula reproduces every other "
           "cell, e.g. 3.08 / 1.54 / 1.28 kHz for > 0.7 at 2.5 / 5 / 6 cm")
check("1.1 sinc coherence > 0.7 below (kHz) at 2.5 / 5 / 6 cm", [3.08, 1.54, 1.28],
      [coh_threshold(d, "3d", 0.7) / 1e3 for d in (0.025, 0.05, 0.06)], 0.005, unit="kHz")
check("1.1 J0 coherence > 0.9 below (kHz) 2.5/4/6 cm", [1.40, 0.87, 0.58],
      [geo[d]["j0_09_07_05_hz"][0] / 1e3 for d in D_PUB], 0.005, unit="kHz")
R["geometry"] = {f"{d*100:.1f}cm": v for d, v in geo.items()}

# ---- 1.2 single-target beams (on the axis): 3-D design/eval, 2-D design/eval (J0 weights)
st = {}
for d in D_ALL:
    e = {}
    for wmin in (-3, -10, -20):
        r3, w3 = single_target(d, 0.0, wmin, "3d", ("3d",))
        r2, _ = single_target(d, 0.0, wmin, "2d", ("2d",))
        e[f"axis_wng{wmin}"] = dict(aidi_3d=float(aidi(r3["3d"])), aidi_2d=float(aidi(r2["2d"])),
                                    band_di_3d=r3["3d"].ravel().tolist(), band_wng_3d=w3.ravel().tolist())
    rb3, wb3 = single_target(d, 0.0, BUS_FLOOR, "3d", ("3d",))
    rb2, _ = single_target(d, 0.0, BUS_FLOOR, "2d", ("2d",))
    e["axis_bus_schedule"] = dict(aidi_3d=float(aidi(rb3["3d"])), aidi_2d=float(aidi(rb2["2d"])),
                                  band_di_3d=rb3["3d"].ravel().tolist())
    # broadside & azimuth average: 3-D-designed weights, evaluated 3-D and 2-D
    for wmin in (-3, -10):
        rz, _ = single_target(d, np.linspace(0, 90, 19), wmin, "3d", ("3d", "2d"))
        a3, a2 = aidi(rz["3d"]), aidi(rz["2d"])
        e[f"az_wng{wmin}"] = dict(per_angle_3d=a3.tolist(), per_angle_2d=a2.tolist(), avg_3d=float(a3.mean()),
                                  avg_2d=float(a2.mean()), broadside_3d=float(a3[-1]), broadside_2d=float(a2[-1]),
                                  endfire_3d=float(a3[0]), endfire_2d=float(a2[0]),
                                  broadside_band_di_3d=rz["3d"][-1].tolist(),
                                  rc_factor_avg_2d=float(10 ** (a2.mean() / 20)))
    e["dsb"] = dict(axis_3d=float(aidi(dsb(d, 0.0, "3d"))), broadside_3d=float(aidi(dsb(d, 90.0, "3d"))),
                    axis_2d=float(aidi(dsb(d, 0.0, "2d"))), broadside_2d=float(aidi(dsb(d, 90.0, "2d"))))
    st[d] = e
P = lambda key, sub: [st[d][key][sub] for d in D_PUB]
check("1.2 single target on axis, WNG>=-10, 3-D", [5.2, 5.0, 4.6], P("axis_wng-10", "aidi_3d"), 0.2)
check("1.2 single target on axis, WNG>=-10, 2-D (J0 weights)", [4.1, 3.9, 3.7], P("axis_wng-10", "aidi_2d"), 0.2)
check("1.2 single target on axis, WNG>=-3, 3-D", [4.1, 4.3, 4.3], P("axis_wng-3", "aidi_3d"), 0.2)
check("1.2 single target on axis, WNG>=-3, 2-D (J0 weights)", [3.4, 3.5, 3.4], P("axis_wng-3", "aidi_2d"), 0.2)
check("1.2 single target on axis, WNG>=-20, 3-D", [5.5, 5.1, 4.7], P("axis_wng-20", "aidi_3d"), 0.2)
check("1.2 single target azimuth-averaged, WNG>=-10, 3-D", [3.2, 3.3, 3.3], P("az_wng-10", "avg_3d"), 0.2)
check("1.2 single target azimuth-averaged, WNG>=-10, 2-D (3-D weights)", [2.5, 2.7, 2.7], P("az_wng-10", "avg_2d"), 0.2)
check("1.2 single target at broadside, WNG>=-10, 3-D", [0.5, 1.0, 1.5], P("az_wng-10", "broadside_3d"), 0.2)
check("1.2 single target at broadside, WNG>=-10, 2-D (3-D weights)", [0.8, 1.5, 2.0], P("az_wng-10", "broadside_2d"), 0.2)
check("1.2 delay-and-sum along the axis, 3-D", [1.3, 1.9, 2.3], [st[d]["dsb"]["axis_3d"] for d in D_PUB], 0.2)
check("1.2 delay-and-sum to broadside, 3-D", [0.5, 1.0, 1.5], [st[d]["dsb"]["broadside_3d"] for d in D_PUB], 0.2)
check("1.4 r_c factor of azimuth-averaged 2-D MVDR, min-max over 2.5-6 cm (x)", [1.34, 1.37],
      [min(P("az_wng-10", "rc_factor_avg_2d")), max(P("az_wng-10", "rc_factor_avg_2d"))], 0.005, unit="x")
check("M2 meeting: on the axis / broadside at 4 cm, WNG>=-3, 3-D", [4.3, 1.0],
      [st[0.04]["axis_wng-3"]["aidi_3d"], st[0.04]["az_wng-3"]["broadside_3d"]], 0.2)
# finer 4 cm numbers
d4 = 0.04
r, _ = single_target(d4, 0.0, -10, "3d", ("3d",), f=np.array([150, 250, 500, 1000, 2000, 3000, 4000, 6000.]))
check("1.2 per-band DI on axis 4 cm, WNG>=-10 (150/250/500/1k/2k/3k/4k/6k Hz)", [2.3, 3.9, 6.0, 5.9, 5.4, 4.5, 3.3, 2.9],
      r["3d"].ravel(), 0.1, "{:.1f}")
w1, w2, a1, a2 = mvdr(np.array([250, 500, 1000.]), d4 / C, d4, -99, "3d")
check("1.2 WNG of unconstrained endfire SD 4 cm (250/500/1000 Hz)", [-15.3, -9.3, -3.6], wng_db(w1, w2, a1, a2), 0.1, "{:.1f}")
fb = np.array([500, 1000, 2000, 3000, 4000.])
r, _ = single_target(d4, 90.0, -10, "3d", ("3d",), f=fb)
check("1.2 broadside MVDR DI 4 cm (0.5/1/2/3/4 kHz)", [0.05, 0.19, 0.76, 1.65, 2.7], r["3d"].ravel(), 0.05)
offax = []
for deg in (0, 30, 45, 60, 90):
    offax.append([float(aidi(single_target(d4, deg, wm, "3d", ("3d",))[0]["3d"]).ravel()[0]) for wm in (-10, -3)])
offax = np.array(offax)
check("1.2 off-axis steering 4 cm, WNG>=-10 (0/30/45/60/90 deg)", [5.0, 4.3, 3.6, 2.6, 1.0], offax[:, 0], 0.2, "{:.1f}")
check("1.2 off-axis steering 4 cm, WNG>=-3 (0/30/45/60/90 deg)", [4.3, 3.8, 3.1, 2.3, 1.0], offax[:, 1], 0.2, "{:.1f}")
R["single_target"] = {f"{d*100:.1f}cm": v for d, v in st.items()}

# ---- two-talker beam (A 0 deg, B 180 deg)
tt = {}
for d in D_ALL:
    e = {}
    for name, fl in (("wng-10", -10.0), ("wng-20", -20.0), ("bus_schedule", BUS_FLOOR)):
        o = two_talker(d, fl)
        e[name] = {k: dict(aidi=v["aidi"], band_di=v["band_di"].tolist(),
                           psi_deg=[None if np.isnan(p) else float(p) for p in v["psi_deg"]],
                           band_wng=[None if np.isnan(p) else float(p) for p in v["wng"]],
                           omni_fallback_bands_hz=[int(f) for f, ok in zip(FC, v["feasible"]) if not ok])
                   for k, v in o.items()}
    od = two_talker(d, -10.0, psis=np.array([np.pi]))
    e["fixed_dipole_wng-10"] = {k: v["aidi"] for k, v in od.items()}
    tt[d] = e
T = lambda name, kind: [tt[d][name][kind]["aidi"] for d in D_PUB]
check("1.2 two-talker beam, WNG>=-10 strict, 3-D", [3.4, 3.6, 3.5], T("wng-10", "3d"), 0.2)
check("1.2 two-talker beam, WNG>=-10 strict, 2-D (3-D weights)", [2.2, 2.3, 2.3], T("wng-10", "2d"), 0.2)
check("1.2 two-talker beam, WNG>=-20 strict, 3-D", [4.4, 4.1, 3.7], T("wng-20", "3d"), 0.2)
check("1.2 two-talker beam, WNG>=-20 strict, 2-D (3-D weights)", [2.8, 2.6, 2.4], T("wng-20", "2d"), 0.2)
psi_ok = []
for d in D_PUB:
    ps = np.array([np.nan if p is None else p for p in tt[d]["wng-10"]["3d"]["psi_deg"]])
    inband = (FC >= dipole_edge_closed(d, -10)) & (FC <= C / (2 * d))
    psi_ok.append(float(np.nanmax(np.abs(((ps[inband] - 180 + 180) % 360) - 180))))
check("M2/1.6-18: optimal psi_AB = 180 deg in every floor-feasible band up to c/2d (max |psi-180|, deg)", [0, 0, 0],
      psi_ok, 2.5, "{:.1f}", "deg")
mref = [two_talker(d, -10.0, evals=("3d",), steer=steer_m, psis=np.array([np.pi]))["3d"]["aidi"] for d in D_PUB]
check("M2 convention check: Mic-A-referenced h=[1,-1], WNG>=-10, 3-D", [0.9, -0.4, -1.6], mref, 0.2)
R["two_talker"] = {f"{d*100:.1f}cm": v for d, v in tt.items()}
R["two_talker_convention_check_micA_ref_3d"] = dict(zip(["2.5cm", "4.0cm", "6.0cm"], mref))

# ---- dipole band
dip = {}
for d in D_ALL:
    dip[d] = dict(edge_wng10_closed_hz=dipole_edge_closed(d, -10), edge_wng10_grid_hz=dipole_band(d, -10),
                  edge_wng15_hz=dipole_edge_closed(d, -15), edge_wng20_hz=dipole_edge_closed(d, -20),
                  edge_wng6_hz=dipole_edge_closed(d, -6), edge_wng3_hz=dipole_edge_closed(d, -3),
                  edge_bus_schedule_hz=dipole_band(d, lambda f: np.where(f < 500, -15., np.where(f <= 2000, -10., -6.))),
                  top_hz=min(C / (2 * d), FS / 2), top_08_c2d_hz=min(0.8 * C / (2 * d), FS / 2),
                  wng_db_at={str(f): float(10 * np.log10(1 - np.cos(2 * np.pi * f * d / C)))
                             for f in (125, 250, 500, 1000, 2000, 4000)})
check("M2 dipole band lower edge, WNG>=-10 (Hz)", [985, 616, 410], [dip[d]["edge_wng10_closed_hz"] for d in D_PUB], 0.5,
      "{:.0f}", "Hz")
check("Appendix B.1 dipole test band upper edge 0.8 c/2d (Hz)", [5488, 3430, 2287],
      [0.8 * C / (2 * d) for d in D_PUB], 0.5, "{:.0f}", "Hz")
R["dipole"] = {f"{d*100:.1f}cm": v for d, v in dip.items()}

# ---- CDR conditioning
cdr = {}
for d in D_ALL:
    cdr[d] = dict(broadside_3d_hz=cdr_edge(d, "3d", "broadside"), endfire_3d_hz=cdr_edge(d, "3d", "endfire"),
                  broadside_2d_hz=cdr_edge(d, "2d", "broadside"), endfire_2d_hz=cdr_edge(d, "2d", "endfire"),
                  at_1khz_broadside_3d=float(abs(1 - coh(1000, d, "3d"))),
                  at_1khz_endfire_3d=float(abs(np.exp(2j * np.pi * 1000 * d / C) - coh(1000, d, "3d"))))
check("M4 CDR usable, broadside partner, 3-D (kHz)", [2.47, 1.54, 1.03], [cdr[d]["broadside_3d_hz"] / 1e3 for d in D_PUB],
      0.005, unit="kHz")
check("M4 CDR usable, endfire partner, 3-D (kHz)", [0.44, 0.27, 0.18], [cdr[d]["endfire_3d_hz"] / 1e3 for d in D_PUB],
      0.005, unit="kHz")
check("M4 |Gs-Gn| at 1 kHz, broadside, 2.5/4 cm", [0.03, 0.09], [cdr[d]["at_1khz_broadside_3d"] for d in (0.025, 0.04)],
      0.005, unit="")
R["cdr"] = {f"{d*100:.1f}cm": v for d, v in cdr.items()}

# ---- near-field ILD (1.3) and mismatch / skew sensitivity
R["nearfield_ild_db"] = {f"{d*100:.1f}cm": {str(r): float(20 * np.log10((r + d) / r)) for r in (0.3, 0.5, 1.0)} for d in D_ALL}
check("1.3 near-field ILD 0.3/0.5/1.0 m, 4 cm", [1.1, 0.67, 0.34], list(R["nearfield_ild_db"]["4.0cm"].values()),
      [0.05, 0.005, 0.005])
check("1.3 near-field ILD 0.3/0.5/1.0 m, 6 cm", [1.6, 1.0, 0.5], list(R["nearfield_ild_db"]["6.0cm"].values()), 0.05)

rng = np.random.default_rng(12345)
mm = {}
TH = np.radians(np.linspace(0, 90, 19))[:, None]
for d in D_ALL:
    w1, w2, a1, a2 = mvdr(FC, d * np.cos(TH) / C, d, -10, "3d")
    e = {}
    for lbl, gdb, pdeg in (("pm1dB_pm3deg", 1.0, 3.0), ("pm0.25dB_pm1deg", 0.25, 1.0)):
        n = 200
        gg = 10 ** (rng.uniform(-gdb, gdb, (n, 2) + TH.shape[:1] + (len(FC),)) / 20)
        pp = np.radians(rng.uniform(-pdeg, pdeg, (n, 2) + TH.shape[:1] + (len(FC),)))
        D1, D2 = gg[:, 0] * np.exp(1j * pp[:, 0]), gg[:, 1] * np.exp(1j * pp[:, 1])
        g2 = coh(FC, d, "2d")
        num = np.abs(np.conj(w1) * D1 * a1 + np.conj(w2) * D2 * a2) ** 2
        den = (np.abs(w1 * np.conj(D1)) ** 2 + np.abs(w2 * np.conj(D2)) ** 2
               + 2 * np.real(np.conj(w1) * D1 * g2 * np.conj(D2) * w2))
        di = 10 * np.log10(np.mean(num / den, axis=0))
        e[lbl] = float(aidi(di).mean())
    e["ideal"] = float(aidi(di_db(w1, w2, a1, a2, FC, d, "2d")).mean())
    # residual (uncompensated) skew error delta: channel B rotated by exp(-j 2 pi f (+-delta)); worse sign kept
    wa = mvdr(FC, d / C, d, -10, "3d")
    for dus in (2.0, 5.0, 10.0, SKEW_US):
        az_v, ax_v = [], []
        for sgn in (1.0, -1.0):
            Dk = np.exp(-2j * np.pi * FC * sgn * dus * 1e-6)
            az_v.append(float(aidi(di_db(w1, w2, a1, a2, FC, d, "2d", D=Dk)).mean()))
            ax_v.append(float(aidi(di_db(*wa, FC, d, "3d", D=Dk))))
        e[f"skew_err_{dus:g}us_azavg_2d_worst"] = min(az_v)
        e[f"skew_err_{dus:g}us_axis_3d_worst"] = min(ax_v)
    e["axis_3d_ideal"] = st[d]["axis_wng-10"]["aidi_3d"]
    mm[d] = e
check("1.2 mismatch cost +-1 dB/+-3 deg, az-avg 2-D, WNG>=-10 (dB, <=0.1)", [0.1, 0.1, 0.1],
      [max(0.0, mm[d]["ideal"] - mm[d]["pm1dB_pm3deg"]) for d in D_PUB], 0.1, "{:.2f}")
val[-1]["design_value"] = "<= 0.1 dB at 2.5 / 4 / 6 cm"
val[-1]["ok"] = bool(all(mm[d]["ideal"] - mm[d]["pm1dB_pm3deg"] <= 0.1 + 0.02 for d in D_PUB))
R["mismatch_and_skew_azavg2d_wng-10"] = {f"{d*100:.1f}cm": v for d, v in mm.items()}

# mismatch sensitivity of the on-axis single-target beam (3-D) at the lower floors used on the bus
mmf = {}
for d in D_ALL:
    e = {}
    for fname, fl in (("wng-10", -10.0), ("wng-20", -20.0), ("bus_schedule", BUS_FLOOR)):
        w1, w2, a1, a2 = mvdr(FC, d / C, d, fl, "3d")
        g3 = coh(FC, d, "3d")
        e[fname] = {"ideal": float(aidi(di_db(w1, w2, a1, a2, FC, d, "3d")))}
        for lbl, gdb, pdeg in (("pm1dB_pm3deg", 1.0, 3.0), ("pm0.25dB_pm1deg", 0.25, 1.0)):
            n = 400
            gg = 10 ** (rng.uniform(-gdb, gdb, (n, 2, len(FC))) / 20)
            pp = np.radians(rng.uniform(-pdeg, pdeg, (n, 2, len(FC))))
            D1, D2 = gg[:, 0] * np.exp(1j * pp[:, 0]), gg[:, 1] * np.exp(1j * pp[:, 1])
            num = np.abs(np.conj(w1) * D1 * a1 + np.conj(w2) * D2 * a2) ** 2
            den = (np.abs(w1 * D1) ** 2 + np.abs(w2 * D2) ** 2 + 2 * np.real(np.conj(w1) * D1 * g3 * np.conj(D2) * w2))
            e[fname][lbl] = float(aidi(10 * np.log10(np.mean(num / den, axis=0))))
    mmf[d] = e
R["mismatch_axis3d_by_floor"] = {f"{d*100:.1f}cm": v for d, v in mmf.items()}

# ---- WNG needed by useful beams below 1 kHz (any spacing)
wn = {}
FL = np.array([125, 250, 500, 1000, 2000.])
for d in D_ALL:
    w1, w2, a1, a2 = mvdr(FL, d / C, d, -99, "3d")                 # unconstrained superdirective (hypercardioid)
    kd = 2 * np.pi * FL * d / C
    e = dict(freqs_hz=FL.tolist(), sd_endfire_unconstrained_wng_db=wng_db(w1, w2, a1, a2).tolist(),
             sd_endfire_unconstrained_di3d_db=di_db(w1, w2, a1, a2, FL, d, "3d").tolist(),
             cardioid_null_at_rear_wng_db=(10 * np.log10(1 - np.cos(2 * kd))).tolist(),
             dipole_wng_db=(10 * np.log10(1 - np.cos(kd))).tolist())
    for wmin in (-6, -10, -15, -20):
        r3, _ = single_target(d, 0.0, wmin, "3d", ("3d",), f=FL)
        e[f"sd_endfire_di3d_at_wng{wmin}"] = r3["3d"].ravel().tolist()
    wn[d] = e
R["low_freq_wng"] = {f"{d*100:.1f}cm": v for d, v in wn.items()}


def wng_for_di(d, f, target_db=3.0, design="3d"):
    """WNG (dB) of the on-axis MVDR whose DI (same model) equals target_db; nan if even mu->0 cannot reach it,
    and the delay-and-sum WNG if delay-and-sum already reaches it."""
    f = np.asarray(f, float)
    g = coh(f, d, design)
    a1, a2 = steer_c(f, d / C)

    def ev(mu):
        det = (1 + mu) ** 2 - g ** 2
        x1 = ((1 + mu) * a1 - g * a2) / det
        x2 = ((1 + mu) * a2 - g * a1) / det
        den = np.conj(a1) * x1 + np.conj(a2) * x2
        w1, w2 = x1 / den, x2 / den
        return di_db(w1, w2, a1, a2, f, d, design), wng_db(w1, w2, a1, a2)

    lo, hi = np.full(f.shape, -12.0), np.full(f.shape, 6.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        big = ev(10 ** mid)[0] >= target_db
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    di, wg = ev(10 ** lo)
    di_min, wg_dsb = ev(1e6)
    return np.where(ev(1e-12)[0] < target_db, np.nan, np.where(di_min >= target_db, wg_dsb, wg))


FU = np.array([125, 160, 250, 315, 500, 630, 800, 1000, 1250, 1600, 2000.])
R["wng_needed_for_3dB_DI_axis_3d"] = {"freqs_hz": FU.tolist()}
R["wng_needed_for_3dB_DI_axis_3d"].update({f"{d*100:.1f}cm": wng_for_di(d, FU).tolist() for d in D_ALL})

# frequency below which a broadside talker and diffuse noise are spatially indistinguishable (several readings)
bsi = {}
for d in D_ALL:
    dsb_thr = lambda gdb: (lambda gv: brentq(lambda f: coh(f, d, "3d") - gv, 1.0, C / (2 * d)))(2 / 10 ** (gdb / 10) - 1)
    bsi[d] = dict(coh09_3d_hz=geo[d]["sinc_09_07_05_hz"][0], coh09_2d_hz=geo[d]["j0_09_07_05_hz"][0],
                  cdr_rule_3d_hz=cdr[d]["broadside_3d_hz"], cdr_rule_2d_hz=cdr[d]["broadside_2d_hz"],
                  broadside_beam_di_reaches_0p5dB_3d_hz=dsb_thr(0.5), broadside_beam_di_reaches_1dB_3d_hz=dsb_thr(1.0))
R["broadside_vs_diffuse_indistinguishable"] = {f"{d*100:.1f}cm": v for d, v in bsi.items()}

# share of the speech band-importance above the lower edge of each spatial mechanism (band centres >= edge)
aiw = lambda fe: float(IW[FC >= fe].sum())
R["ai_weight_above_edge"] = {f"{d*100:.1f}cm": dict(
    dipole_wng10=aiw(dip[d]["edge_wng10_closed_hz"]), dipole_bus_schedule=aiw(dip[d]["edge_bus_schedule_hz"]),
    cdr_endfire=aiw(cdr[d]["endfire_3d_hz"]), cdr_broadside=aiw(cdr[d]["broadside_3d_hz"]),
    below_1khz=float(IW[FC < 1000].sum())) for d in D_ALL}

# ---- meeting TDOA groups
scene_dir = os.path.join(LAB, "scenes")
p40 = sorted(glob.glob(os.path.join(scene_dir, "meeting_*d40mm.npz")))
m0 = load_meta(p40[0])
seats0, dev0 = np.array(m0["seats"]), np.array(m0["device"])
# (a) exact replica of review/probe.py (same RNG stream, same order)
rng = np.random.default_rng(0)
rep = {}
for d in D_PUB:
    for res in (10e-6, 20e-6):
        ks = [greedy_groups(np.sort(tdoas(seats0, dev0 + u, dev0 - u)), res)
              for u in (np.array([np.cos(a), np.sin(a), 0.0]) * d / 2 for a in rng.uniform(0, 2 * np.pi, 500))]
        rep[(d, res)] = (float(np.median(ks)), int(min(ks)), int(max(ks)))
# probe draws az one at a time (rng.uniform(0, 2pi) per draw); a 500-vector draw gives the identical stream.
check("1.5 TDOA groups, 20 seats, 20 us, 4 cm: median (min-max)", [8, 6, 10],
      [rep[(0.04, 20e-6)][0], rep[(0.04, 20e-6)][1], rep[(0.04, 20e-6)][2]], 0.0, "{:.0f}", "")
check("1.5 TDOA groups, 20 seats, 10 us, 4 cm: median", [12], [rep[(0.04, 10e-6)][0]], 0.0, "{:.0f}", "")
check("1.5 TDOA groups, 20 seats, 20 us, 2.5 cm: median", [6], [rep[(0.025, 20e-6)][0]], 0.0, "{:.0f}", "")
gap_counts = []
for p in p40:
    m = load_meta(p)
    t = np.sort(tdoas(np.array(m["seats"]), np.array(m["mics"][0]), np.array(m["mics"][1])))
    gap_counts.append(int(np.sum(np.diff(t) < 10e-6)))
check("1.5 neighbouring seats < 10 us apart (of 19), 4 cm scenes, min-max", [8, 13], [min(gap_counts), max(gap_counts)],
      0.0, "{:.0f}", "")
# (b) extension: common azimuth grid (720 azimuths), scene s0 and all six seat layouts
AZ = (np.arange(720) + 0.5) / 720 * 2 * np.pi
layouts = {}
for s in (0, 1, 2, 10, 11, 12):
    cand = sorted(glob.glob(os.path.join(scene_dir, f"meeting_s{s}_d*mm.npz")))
    if cand:
        m = load_meta(cand[0])
        layouts[s] = (np.array(m["seats"]), np.array(m["device"]))
tg = {}
for d in D_ALL:
    e = {}
    for res in (10e-6, 20e-6):
        k0, g0 = groups_over_azimuth(seats0, dev0, d, res, AZ)
        allk = [np.median(groups_over_azimuth(se, dv, d, res, AZ)[0]) for se, dv in layouts.values()]
        e[f"{res*1e6:.0f}us"] = dict(s0_median=float(np.median(k0)), s0_min=int(k0.min()), s0_max=int(k0.max()),
                                     s0_p10=float(np.percentile(k0, 10)), s0_p90=float(np.percentile(k0, 90)),
                                     layouts_median_of_medians=float(np.median(allk)),
                                     layouts_medians=[float(x) for x in allk],
                                     s0_rng_replica_median=(rep[(d, res)][0] if (d, res) in rep else None))
        if res == 10e-6:
            e["neighbour_gaps_below_10us_of_19_median"] = float(np.median(g0))
    e["tdoa_span_us"] = 2 * d / C * 1e6
    tg[d] = e
R["tdoa_groups"] = {f"{d*100:.1f}cm": v for d, v in tg.items()}
R["tdoa_groups_note"] = (f"seats of {os.path.basename(p40[0])} (s0) for the headline; {len(layouts)} seat layouts "
                         f"(seeds {sorted(layouts)}) for robustness; 720 uniform azimuths; greedy grouping.")

# ---------------------------------------------------------------------------------------------------- the table
rows = []
for d in D_ALL:
    rows.append(dict(
        d_cm=round(d * 100, 1),
        alias_khz=round(geo[d]["alias_hz"] / 1e3, 2),
        max_tdoa_us=round(geo[d]["max_tdoa_us"], 1),
        max_tdoa_samples=round(geo[d]["max_tdoa_samples"], 3),
        coh_above_0p9_until_khz=round(geo[d]["sinc_09_07_05_hz"][0] / 1e3, 2),
        das_aidi_axis_db=round(st[d]["dsb"]["axis_3d"], 2),
        das_aidi_broadside_db=round(st[d]["dsb"]["broadside_3d"], 2),
        sd_aidi_axis_3d_db=round(st[d]["axis_wng-10"]["aidi_3d"], 2),
        sd_aidi_broadside_3d_db=round(st[d]["az_wng-10"]["broadside_3d"], 2),
        sd_aidi_azavg_2d_db=round(st[d]["az_wng-10"]["avg_2d"], 2),
        two_talker_aidi_aligned_2d_db=round(tt[d]["wng-10"]["2d"]["aidi"], 2),
        dipole_band_from_hz=round(dip[d]["edge_wng10_closed_hz"], 0),
        cdr_usable_endfire_from_hz=round(cdr[d]["endfire_3d_hz"], 0),
        cdr_usable_broadside_from_hz=round(cdr[d]["broadside_3d_hz"], 0),
        tdoa_groups_20_seats_20us=tg[d]["20us"]["s0_median"],
        tdoa_groups_20_seats_10us=tg[d]["10us"]["s0_median"],
        skew_vs_max_tdoa_ratio=round(SKEW_US / (d / C * 1e6), 2)))
R["table"] = rows
R["validation"] = val
R["cpu_seconds"] = time.process_time() - T0
with open(os.path.join(OUT, "physics.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ------------------------------------------------------------------------------------------------- tables.md
L = []
L.append("| quantity | design (2.5 / 4 / 6 cm) | reproduced | ok |\n|---|---|---|---|")
for v in val:
    L.append(f"| {v['quantity']} | {v['design_value']} | {v['reproduced']} | {'yes' if v['ok'] else '**NO**'} |")
L.append("")
hdr = ["d cm", "c/2d kHz", "max TDOA us (smp)", "sinc>0.9 until kHz", "DAS axis / bs 3-D", "SD axis 3-D", "SD bs 3-D",
       "SD az-avg 2-D", "2-talker 2-D", "dipole from Hz", "CDR endfire / bs from Hz", "TDOA groups 20 / 10 us",
       "skew / max TDOA"]
L.append("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for r in rows:
    L.append(f"| {r['d_cm']:.1f} | {r['alias_khz']:.2f} | {r['max_tdoa_us']:.1f} ({r['max_tdoa_samples']:.2f}) | "
             f"{r['coh_above_0p9_until_khz']:.2f} | {r['das_aidi_axis_db']:.2f} / {r['das_aidi_broadside_db']:.2f} | "
             f"{r['sd_aidi_axis_3d_db']:.2f} | {r['sd_aidi_broadside_3d_db']:.2f} | {r['sd_aidi_azavg_2d_db']:.2f} | "
             f"{r['two_talker_aidi_aligned_2d_db']:.2f} | {r['dipole_band_from_hz']:.0f} | "
             f"{r['cdr_usable_endfire_from_hz']:.0f} / {r['cdr_usable_broadside_from_hz']:.0f} | "
             f"{r['tdoa_groups_20_seats_20us']:.0f} / {r['tdoa_groups_20_seats_10us']:.0f} | {r['skew_vs_max_tdoa_ratio']:.2f} |")
with open(os.path.join(OUT, "tables.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nvalidation: {sum(v['ok'] for v in val)}/{len(val)} ok; cpu {time.process_time() - T0:.1f} s")
