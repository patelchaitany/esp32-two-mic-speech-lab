"""B2 (design impact): the extra numbers IMPACT.md needs that B1/V2 did not tabulate.

Analytic only (ideal free-field omnis, c = 343 m/s, fs = 16 kHz, ANSI S3.5 band importance), < 1 CPU-minute.
Reuses the verified solvers of hw/spacing/v2/v2_physics.py (V2 reproduced B1 and DESIGN v1.1 to 0.01 dB).
v2_extra.py is NOT imported (it runs at import time); its two helpers are re-implemented here.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I -B hw/spacing/b2/b2_numbers.py <speech_lab path>
Writes hw/spacing/b2/b2_numbers.json.
"""
import json
import os
import sys

import numpy as np

LAB = sys.argv[1]
sys.path.insert(0, os.path.join(LAB, "hw", "spacing", "v2"))
from v2_physics import (C, FC, FE, BIF, gamma, steer, G2, quad, sd_closed, di_db, wng_db, aidi,  # noqa: E402
                        cdr_edge, dipole_edge, tdoa_groups)

D_LIST = (0.0125, 0.014, 0.025, 0.04, 0.06)
BUS_DESIGN = np.where(FC < 500, -15.0, np.where(FC <= 2000, -10.0, -6.0))      # DESIGN M2 bus schedule
BUS_SMALL = np.where(FC < 500, -15.0, np.where(FC <= 3500, -10.0, -6.0))       # V2's retune for ~1.2 cm
UNI = lambda v: np.full(FC.shape, float(v))                                     # noqa: E731
OUT = {"spacings_m": list(D_LIST)}


def key(d):
    return f"{d * 100:g}"


# ---------------------------------------------------------------- single target with a per-band floor
def st(d, theta, floor, design="3d", evalm="3d"):
    tau = d * np.cos(np.radians(theta)) / C
    w, a = sd_closed(FE, d, tau, 10 ** (np.asarray(floor, float) / 10), design)
    return aidi(di_db(w, a, G2(gamma(FE, d, evalm))))


single = {}
for d in D_LIST:
    r = {}
    for nm, fl in (("-3", UNI(-3)), ("-6", UNI(-6)), ("-10", UNI(-10)), ("bus", BUS_DESIGN)):
        r[f"axis3d_{nm}"] = st(d, 0, fl)
        r[f"axis2d_{nm}"] = st(d, 0, fl, "2d", "2d")
        r[f"deg45_3d_{nm}"] = st(d, 45, fl)
        r[f"broad3d_{nm}"] = st(d, 90, fl)
        r[f"azavg2d_{nm}"] = float(np.mean([st(d, t, fl, "3d", "2d") for t in range(0, 91, 5)]))
    single[key(d)] = r
OUT["single_target"] = single


# ---------------------------------------------------------------- two-talker soft LCMV (DESIGN M2, strict floor)
def two_talker(d, floor_vec, sel_model, lam=1e3):
    """Same algorithm as V2 two_talker_pb: psi on a 2.5 deg grid, WNG bisection, +-1 dB balance, strict floor."""
    psis = np.radians(np.arange(0, 360, 2.5))
    aA, aB = steer(FE, d / C), steer(FE, -d / C)
    Gd, Gs = G2(gamma(FE, d, "3d")), G2(gamma(FE, d, sel_model))
    A = np.stack([aA, aB], -1)
    AAH = A @ np.conj(np.swapaxes(A, -1, -2))
    I2 = np.eye(2)
    out = np.zeros(len(FE))
    for i in range(len(FE)):
        W = 10 ** (floor_vec[i] / 10)
        h = np.stack([np.ones_like(psis), np.exp(1j * psis)], -1).astype(complex)
        rhs = lam * (A[i] @ h.conj().T).T

        def w_of(mu):
            return np.linalg.solve(Gd[i][None] + mu[:, None, None] * I2 + lam * AAH[i][None], rhs[..., None])[..., 0]

        def wng_a(w):
            return (np.abs(np.einsum("pi,i->p", w.conj(), aA[i])) ** 2
                    / np.real(np.einsum("pi,pi->p", w.conj(), w)))

        lo, hi = np.full(len(psis), -9.0), np.full(len(psis), 6.0)
        ok_lo = wng_a(w_of(10 ** lo)) >= W
        for _ in range(50):
            m = 0.5 * (lo + hi)
            ok = wng_a(w_of(10 ** m)) >= W
            hi, lo = np.where(ok, m, hi), np.where(ok, lo, m)
        w = w_of(np.where(ok_lo, 1e-9, 10 ** hi))
        rA = np.einsum("pi,i->p", w.conj(), aA[i])
        rB = np.einsum("pi,i->p", w.conj(), aB[i])
        ok = (wng_a(w) >= W * (1 - 1e-9)) & (np.abs(20 * np.log10(np.abs(rA) / np.abs(rB))) <= 1.0)
        df = 0.5 * (abs(rA) ** 2 + abs(rB) ** 2) / np.real(np.einsum("pi,ij,pj->p", w.conj(), Gs[i], w))
        out[i] = 10 * np.log10(df[np.argmax(np.where(ok, df, -np.inf))]) if ok.any() else 0.0
    return aidi(out)


tt = {}
for d in D_LIST:
    e = {}
    for nm, fl in (("bus_design", BUS_DESIGN), ("bus_small", BUS_SMALL), ("-10", UNI(-10)), ("-20", UNI(-20))):
        e[f"{nm}_3d"] = two_talker(d, fl, "3d")
        e[f"{nm}_2d"] = two_talker(d, fl, "2d")
    tt[key(d)] = e
OUT["two_talker"] = tt


# ---------------------------------------------------------------- band edges
OUT["edges_hz"] = {key(d): {"alias_c_2d": C / (2 * d), "s1_tdoa_top_0p8_alias": 0.8 * C / (2 * d),
                            "dipole_from_-10": float(dipole_edge(d, -10.0)),
                            "dipole_from_-6": float(dipole_edge(d, -6.0)),
                            "cdr_endfire_from": cdr_edge(d, d / C), "cdr_broadside_from": cdr_edge(d, 0.0),
                            "cdr_45deg_from": cdr_edge(d, d * np.cos(np.radians(45)) / C)}
                   for d in D_LIST}


# ---------------------------------------------------------------- M5 phase fence: ideal (noise-free) selectivity
def fence_db(d, theta_pass, band_top, gamma_f=4.0, floor=0.3):
    """AI-weighted mean fence gain (dB) on a passenger at theta_pass (deg from the axis, A side), partners at
    +-endfire (tau = +-d/c), band [1 kHz, band_top]; eta = max_k 1/(1 + g e_k^2), G = max(floor, eta)."""
    sel = (FC >= 1000) & (FC <= band_top)
    f = FE[sel]
    tp = d * np.cos(np.radians(theta_pass)) / C
    eta = 0.0
    for tk in (d / C, -d / C):
        e = np.angle(np.exp(1j * 2 * np.pi * f * (tp - tk)))
        eta = np.maximum(eta, 1.0 / (1.0 + gamma_f * e ** 2))
    g = 20 * np.log10(np.maximum(floor, eta))
    return float(np.sum(BIF[sel] * g) / np.sum(BIF[sel])), float(np.sum(BIF[sel]))


fence = {}
for d in D_LIST:
    top_std = min(4000.0, 0.8 * C / (2 * d))
    top_ext = min(6500.0, 0.8 * C / (2 * d))
    fence[key(d)] = {
        "band_std_hz": [1000.0, top_std], "band_ext_hz": [1000.0, top_ext],
        "broadside_std_db": fence_db(d, 90, top_std)[0], "broadside_ext_db": fence_db(d, 90, top_ext)[0],
        "deg60_std_db": fence_db(d, 60, top_std)[0], "deg45_std_db": fence_db(d, 45, top_std)[0],
        "ai_weight_std": fence_db(d, 90, top_std)[1], "ai_weight_ext": fence_db(d, 90, top_ext)[1],
        # gamma rescaled so the mask has the same angular width as gamma=4 at 4 cm, 2 kHz
        "broadside_ext_gscaled_db": fence_db(d, 90, top_ext, gamma_f=4.0 * (0.04 / d) ** 2)[0]}
OUT["fence_ideal"] = fence


# ---------------------------------------------------------------- residual-skew cost (on-axis SD, 3-D, -10 dB)
def skew_aidi(d, delta, floor=-10.0):
    w, a = sd_closed(FE, d, d / C, np.full(FE.shape, 10 ** (floor / 10)), "3d")
    Dl = np.exp(-2j * np.pi * FE * delta)
    am = a.copy()
    am[:, 1] = am[:, 1] * Dl
    g = gamma(FE, d, "3d")
    Gm = np.stack([np.stack([np.ones_like(g), g * np.conj(Dl)], -1), np.stack([g * Dl, np.ones_like(g)], -1)], -2)
    return aidi(10 * np.log10(np.abs(np.einsum("fi,fi->f", w.conj(), am)) ** 2 / quad(w, Gm)))


skew = {}
for d in D_LIST:
    ideal = skew_aidi(d, 0.0)
    e = {"ideal": ideal}
    # 2 us; 5.9 us = the 0.095-sample change seen across the esptool resets; 10 us; 33 us = worst blind S0
    # error on simulated scenes (0.53 samples); 68.75 us = the whole uncorrected boot-1 skew
    for us in (2.0, 5.9, 10.0, 33.0, 68.75):
        e[f"loss_{us:g}us_worst_sign"] = ideal - min(skew_aidi(d, us * 1e-6), skew_aidi(d, -us * 1e-6))
    skew[key(d)] = e
OUT["skew_cost_axis3d_-10"] = skew


# ---------------------------------------------------------------- self-noise: WNG needed for 3 dB on-axis DI (3-D)
def wng_for_di(d, f, target_db=3.0):
    lo, hi = -40.0, 3.0  # floor in dB; DI grows as the floor is lowered
    for _ in range(60):
        m = 0.5 * (lo + hi)
        w, a = sd_closed(np.array([f]), d, d / C, np.array([10 ** (m / 10)]), "3d")
        di = di_db(w, a, G2(gamma(np.array([f]), d, "3d")))[0]
        lo, hi = (m, hi) if di >= target_db else (lo, m)
    return lo


OUT["wng_for_3db_di_axis"] = {key(d): {f"{f:g}Hz": wng_for_di(d, f) for f in (250.0, 500.0, 1000.0)} for d in D_LIST}

# ---------------------------------------------------------------- bus partner separation (TDOA, same side / opposite)
sep = {}
for d in D_LIST:
    dc = d / C * 1e6
    sep[key(d)] = {"d_over_c_us": dc,
                   "aligned_pair_us": 2 * dc * np.cos(np.radians(15)),
                   "pair_45deg_opposite_us": 2 * dc * np.cos(np.radians(45)),
                   "side_by_side_20_50deg_us": dc * (np.cos(np.radians(20)) - np.cos(np.radians(50))),
                   "s1_25us_rule_in_d_over_c": 25.0 / dc}
OUT["partner_tdoa_separation"] = sep

# ---------------------------------------------------------------- meeting TDOA groups (lab meeting_s0 seats)
z = np.load(os.path.join(LAB, "scenes", "meeting_s0_d40mm.npz"), allow_pickle=False)
m = json.loads(str(z["meta"]))
seats, dev = np.array(m["seats"]), np.array(m["device"])
OUT["tdoa_groups_meeting_s0"] = {}
for d in D_LIST:
    g20 = tdoa_groups(seats, dev, d, 20e-6, n_az=360)
    g10 = tdoa_groups(seats, dev, d, 10e-6, n_az=360)
    OUT["tdoa_groups_meeting_s0"][key(d)] = {"median_20us": g20[0], "range_20us": [g20[1], g20[2]],
                                             "median_10us": g10[0]}

p = os.path.join(LAB, "hw", "spacing", "b2", "b2_numbers.json")
with open(p, "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
print(json.dumps(OUT, indent=1, default=lambda x: round(float(x), 3)))
