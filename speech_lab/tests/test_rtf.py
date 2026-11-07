"""Tests for methods/_core/rtf.py, M3 (DESIGN.md M3, 2.0, 2.1; Appendix B.1, B.4, B.7).

Run from LAB: .venv/bin/python -m tests.test_rtf [--quick]
--quick skips the end-to-end checks on synthetic scenes. Plain asserts; prints the measured numbers.
B.4 compares the M3 two-talker beam with a model-based, centre-referenced M2 beam implemented here
(scalar loops, np.linalg.solve), independent of rtf.py.
"""
import sys
import time
from dataclasses import replace

import numpy as np

from lab.common import C, FS
from lab.scenes import coherent_noise
from methods._core import calib, common as cm, rtf, tracker

F = cm.FREQS
BUS = {"scenario": "bus", "spacing_m": 0.04}
MEETING = {"scenario": "meeting", "spacing_m": 0.04}
FLOOR_BUS = rtf.wng_floor_db(rtf.PARAMS["wng_db"])
FLOOR_10 = np.full(cm.N_BINS, -10.0)
P10 = rtf.params("bus", wng_db=((np.inf, -10.0),))


def check(name, ok, msg):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


def wrap(x):
    return np.angle(np.exp(1j * x))


def diffuse_R(d, sigma=0.0):
    return rtf.normalize_cov(rtf.noise_model(np.sinc(2 * F * d / C), sigma))


def responses(W, gA, gB):
    """Responses w^H g of weights W to steering gA, gB scaled to ||g||^2 = 2."""
    rA = rtf._dot(W, gA) * np.sqrt(2 / rtf._n2(gA))
    rB = rtf._dot(W, gB) * np.sqrt(2 / rtf._n2(gB))
    return rA, rB


# ------------------------------------------------------------------ independent model-based M2 reference
def ref_lcmv(R, aA, aB, psi, fl, lam=1e3, steps=40):
    """Soft LCMV (physics_check_v2.lcmv semantics), floor strictly enforced for both partners; None if
    no mu in [1e-9, 1e6] meets it."""
    A = np.stack([aA, aB], 1)
    rhs = lam * A @ np.array([1.0, np.exp(-1j * psi)])
    M = lam * A @ A.conj().T + R

    def w_of(lmu):
        return np.linalg.solve(M + 10.0 ** lmu * np.eye(2), rhs)

    def wng(w):
        return min(abs(w.conj() @ aA) ** 2, abs(w.conj() @ aB) ** 2) / np.real(w.conj() @ w)

    if wng(w_of(-9.0)) >= fl:
        return w_of(-9.0)
    if wng(w_of(6.0)) < fl:
        return None
    lo, hi = -9.0, 6.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if wng(w_of(mid)) >= fl else (mid, hi)
    return w_of(hi)


def ref_figures(w, R, aA, aB):
    rA, rB = w.conj() @ aA, w.conj() @ aB
    nA, nB = abs(rA) ** 2, abs(rB) ** 2
    return dict(df=0.5 * (nA + nB) / np.real(w.conj() @ R @ w), bal=10 * np.log10(nA / nB),
                act=np.angle(rB / rA))


def ref_two_talker(R, tau_a, tau_b, d, floor_db):
    """Model-based centre-referenced two-talker beam with the M2 phase rule (DESIGN M2), bin by bin:
    dipole band (contiguous bins ending at c/2d where h = [1, -1] meets the floor, |r_A|/|r_B| within
    +-1 dB, phase within +-10 deg), continuation (+-30 deg window on the nominal and actual phase, then the
    full circle, then Mic A); partners not on opposite sides: full circle at 1 kHz, then continuation.
    Weights scaled to sqrt(|r_A| |r_B|) = 1. Returns (W (F, 2), mode (F,))."""
    aA, aB = cm.steer(F, tau_a), cm.steer(F, tau_b)
    fl = 10 ** (floor_db / 10)
    W, mode = np.zeros((len(F), 2), complex), np.zeros(len(F), int)
    grid = np.radians(np.arange(0.0, 360.0, 2.5))
    win = np.radians(np.arange(-30.0, 30.001, 2.5))

    def best(k, cands, prev, window):
        """Largest DF among feasible, balanced candidates (window: actual phase within 30 deg of prev);
        DF ties (1e-9) go to the actual phase nearest prev, then to the positive side."""
        found = []
        for ps in cands:
            w = ref_lcmv(R[k], aA[k], aB[k], ps, fl[k])
            if w is None:
                continue
            m = ref_figures(w, R[k], aA[k], aB[k])
            if abs(m["bal"]) > 1.0 or (window and abs(wrap(m["act"] - prev)) > np.radians(30) + 1e-9):
                continue
            found.append((m["df"], w, m["act"]))
        if not found:
            return None
        top = max(f[0] for f in found)
        near = [f for f in found if f[0] >= top * (1 - 1e-9)]
        return min(near, key=lambda f: abs(wrap(f[2] - prev)) - 1e-6 * wrap(f[2] - prev))

    def dipole(k):
        w = ref_lcmv(R[k], aA[k], aB[k], np.pi, fl[k])
        if w is None:
            return None
        m = ref_figures(w, R[k], aA[k], aB[k])
        return w if abs(m["bal"]) <= 1.0 and abs(wrap(m["act"] - np.pi)) <= np.radians(10) else None

    def mic_a(k):
        return np.array([np.conj(1 / aA[k][0]), 0]), np.angle(aB[k][0] / aA[k][0])

    k_top = int(np.argmin(np.abs(F - C / (2 * d))))
    if tau_a > 0 > tau_b and dipole(k_top) is not None:
        lo = k_top
        while lo > 0 and dipole(lo - 1) is not None:
            lo -= 1
        for k in range(lo, k_top + 1):
            W[k] = dipole(k)
        down, up, start = lo - 1, k_top + 1, np.pi
    else:
        ka = int(np.argmin(np.abs(F - 1000.0)))
        b = best(ka, grid, np.pi, False)
        (W[ka], start), mode[ka] = ((b[1], b[2]), 2) if b else (mic_a(ka), 3)
        down, up = ka - 1, ka + 1
    for ks in (range(down, -1, -1), range(up, len(F))):
        prev = start
        for k in ks:
            b, m = best(k, prev + win, prev, True), 1
            if b is None:
                b, m = best(k, grid, prev, False), 2
            (w, act), mode[k] = ((b[1], b[2]) if b else mic_a(k)), (m if b else 3)
            W[k] = w
            prev = prev + wrap(act - prev)
    s = np.sqrt(np.abs(rtf._dot(W, aA)) * np.abs(rtf._dot(W, aB)))
    return W / s[:, None], mode


# ------------------------------------------------------------------ covariance whitening, symmetrization
def test_cw_and_symmetrization():
    print("covariance whitening (DESIGN 1.6 item 3) and RTF symmetrization (DESIGN 2.0)")
    rng = np.random.default_rng(0)
    worst_cw, worst_v = 0.0, np.inf
    for d, sig in ((0.025, 0.0), (0.04, 0.01), (0.06, 0.3)):
        Rw = rtf.noise_model(np.clip(np.sinc(2 * F * d / C), -0.3, 0.98), sig)
        g = np.stack([np.ones(len(F)), 10 ** (0.8 / 20) * np.exp(-2j * np.pi * F * 0.7 * d / C + 0.1j)], -1)
        Phi = 2.5 * rtf._outer(g) + 0.4 * Rw
        G, _ = rtf.cw_rtf(Phi, Rw)
        worst_cw = max(worst_cw, np.max(np.abs(G - g[:, 1])))
        L = np.linalg.cholesky(Rw)
        Li = np.linalg.inv(L)
        v = np.linalg.eigh(Li @ Phi @ np.conj(np.swapaxes(Li, -1, -2)))[1][..., :, 1]
        worst_v = min(worst_v, np.median(np.abs(v[:, 1] / v[:, 0] - g[:, 1])))
    check("rank-1 target + Phi_w-shaped noise -> exact RTF", worst_cw < 1e-9, f"max |G err| {worst_cw:.1e}")
    check("v_max itself (not Phi_w v) is wrong", worst_v > 0.05, f"median |err| of v2/v1 >= {worst_v:.3f}")
    A = rng.standard_normal((len(F), 2, 2)) + 1j * rng.standard_normal((len(F), 2, 2))
    Rr = A @ np.conj(np.swapaxes(A, -1, -2)) + 0.1 * np.eye(2)
    g = rng.standard_normal((len(F), 2)) + 1j * rng.standard_normal((len(F), 2))
    G, _ = rtf.cw_rtf(5 * rtf._outer(g) + 2 * Rr, Rr)
    e = np.max(np.abs(G - g[:, 1] / g[:, 0]))
    check("exact for a random non-white Phi_w", e < 1e-8, f"max |G err| {e:.1e}")
    for d in (0.025, 0.04, 0.06):
        for tau in (-d / C, 0.0, 0.37 * d / C, d / C):
            half, gs = rtf.sym_rtf(rtf.model_rtf(F, tau), F, tau)
            e = np.max(np.abs(gs - cm.steer(F, tau)))
            assert e < 1e-12, (d, tau, e)
            assert np.max(np.abs(gs - cm.symmetrize_rtf(rtf.model_rtf(F, tau)))) < 1e-12
    check("symmetrized plane-wave RTF == steer(f, tau) (and == common.symmetrize_rtf)", True,
          "d 2.5/4/6 cm, tau -d/c..+d/c, max |err| < 1e-12")
    G = rtf.model_rtf(F, 0.9 * 0.06 / C).copy()
    G[150] *= np.exp(2.0j)                           # two noisy bins (+2 / -2 rad): unwrapping along frequency
    G[151] *= np.exp(-2.0j)                          # slips by 2 pi and flips the sign of g~ above them
    _, g_model = rtf.sym_rtf(G, F, 0.9 * 0.06 / C)
    g_freq = cm.symmetrize_rtf(G)
    others = np.r_[0:150, 152:len(F)]
    e_model = np.max(np.abs(g_model[others] - cm.steer(F[others], 0.9 * 0.06 / C)))
    e_freq = np.max(np.abs(g_freq[others] - cm.steer(F[others], 0.9 * 0.06 / C)))
    check("model-referenced unwrap confines a bad bin", e_model < 1e-12 and e_freq > 1.0,
          f"other bins: max err {e_model:.1e} (frequency unwrap: {e_freq:.2f})")


# ------------------------------------------------------------------ B.1 on the model, through rtf.py
def test_b1_beams():
    print("B.1 plane waves: MVDR unit response, two-talker beam r_A = +1 / r_B = -1, WNG floor, convention")
    for d, f_edge in ((0.025, 985.0), (0.04, 616.0), (0.06, 410.0)):
        R = diffuse_R(d)
        for tau in (-d / C, 0.0, d / C):
            a = cm.steer(F, tau)
            w, lmu = rtf.mvdr(R, a, FLOOR_BUS)
            r = rtf._dot(w, a)
            wng = 10 * np.log10(1 / rtf._n2(w))
            tight = np.max(np.abs(wng - FLOOR_BUS)[lmu > rtf.PARAMS["mu_lo"]]) if (lmu > -9).any() else 0.0
            assert np.max(np.abs(np.abs(r) - 1)) < 1e-6 and np.min(wng - FLOOR_BUS) > -1e-6 and tight < 0.01
        R = R[None]
        gA, gB = cm.steer(F, d / C)[None], cm.steer(F, -d / C)[None]
        k_top = np.array([int(np.argmin(np.abs(F - C / (2 * d))))])
        for floor_db, p, tag in ((FLOOR_10, P10, "-10 dB"), (FLOOR_BUS, rtf.PARAMS, "bus floors")):
            W, psi, mode = rtf.ab_beam(R, gA, gB, floor_db, np.array([True]), k_top, p)
            rA, rB = responses(W, gA, gB)
            rA, rB, mode = rA[0], rB[0], mode[0]
            wng = 10 * np.log10(np.minimum(np.abs(rA) ** 2, np.abs(rB) ** 2) / rtf._n2(W[0]))
            band = (F >= f_edge) & (F <= 0.8 * C / (2 * d))
            if tag == "-10 dB":
                dev_db = np.max(np.abs(20 * np.log10(np.abs(np.r_[rA[band], rB[band]]))))
                dev_ph = np.max(np.abs(np.degrees(np.r_[np.angle(rA[band]), wrap(np.angle(rB[band]) - np.pi)])))
                check(f"d={d*1e3:.0f} mm: r_A=+1, r_B=-1 over {f_edge:.0f}-{0.8*C/(2*d):.0f} Hz", dev_db <= 1.0
                      and dev_ph <= 10.0, f"max dev {dev_db:.4f} dB, {dev_ph:.4f} deg; dipole band "
                      f"{F[mode == 0].min():.0f}-{F[mode == 0].max():.0f} Hz")
                am_A = np.stack([np.ones(len(F)), np.exp(-2j * np.pi * F * d / C)], -1)
                am_B = np.stack([np.ones(len(F)), np.exp(+2j * np.pi * F * d / C)], -1)
                ratio = rtf._dot(W[0], am_B) / rtf._dot(W[0], am_A)
                e = np.max(np.abs(np.degrees(np.angle(ratio[band] / -np.exp(2j * np.pi * F[band] * d / C)))))
                check(f"d={d*1e3:.0f} mm: Mic-A-referenced r_B/r_A = -exp(+j2pi f d/c)", e <= 10.0, f"max {e:.2e} deg")
            steps = np.abs(np.diff(np.degrees(np.unwrap(np.angle(rB / rA)))))
            jump = (mode[1:] == 2) | (mode[:-1] == 2)                       # DESIGN full-circle fallback bins
            check(f"d={d*1e3:.0f} mm, {tag}: WNG >= floor in every bin; phase steps <= 30 deg outside "
                  f"full-circle bins", np.min(wng - floor_db) > -1e-6 and np.max(steps[~jump]) <= 30.0 + 1e-6,
                  f"min WNG-floor {np.min(wng - floor_db):+.1e} dB; max step {np.max(steps[~jump]):.1f} deg; "
                  f"full-circle bins {int(np.sum(mode == 2))} (steps there "
                  f"{np.round(steps[jump], 1).tolist()} deg), Mic-A bins {int(np.sum(mode == 3))}")


# ------------------------------------------------------------------ B.4
def stft_partner(G, n, snr_db, Rn, rng):
    """STFT-domain frames of one partner (RTF G, white source) plus noise of covariance Rn at snr_db."""
    S = (rng.standard_normal((len(F), n)) + 1j * rng.standard_normal((len(F), n))) / np.sqrt(2)
    L = np.linalg.cholesky(Rn)
    Z = (rng.standard_normal((len(F), 2, n)) + 1j * rng.standard_normal((len(F), 2, n))) / np.sqrt(2)
    N = L @ Z * 10 ** (-snr_db / 20)
    return np.stack([S + N[:, 0], G[:, None] * S + N[:, 1]])


def test_b4():
    print("B.4 M3 two-talker beam from symmetrized RTFs == model-based centre-referenced M2 beam (0.1 dB)")
    rng = np.random.default_rng(1)
    for d, ta, tb in ((0.04, 1.0, -1.0), (0.06, 0.5, -0.9), (0.025, 0.95, 0.3)):
        tau_a, tau_b = ta * d / C, tb * d / C
        R = diffuse_R(d, 0.01)
        t0 = time.perf_counter()
        W_ref, mode_ref = ref_two_talker(R, tau_a, tau_b, d, FLOOR_BUS)
        t_ref = time.perf_counter() - t0
        aA, aB = cm.steer(F, tau_a), cm.steer(F, tau_b)
        k_top = np.array([int(np.argmin(np.abs(F - C / (2 * d))))])
        opp = np.array([tau_a > 0 > tau_b])

        def figures(W):
            rA, rB = rtf._dot(W, aA), rtf._dot(W, aB)
            return np.stack([10 * np.log10(np.abs(rA) ** 2), 10 * np.log10(np.abs(rB) ** 2),
                             10 * np.log10(0.5 * (np.abs(rA) ** 2 + np.abs(rB) ** 2) / rtf._quad(W, R)),
                             10 * np.log10(np.minimum(np.abs(rA) ** 2, np.abs(rB) ** 2) / rtf._n2(W))]), \
                np.angle(rB / rA)

        f_ref, ph_ref = figures(W_ref)
        tag = f"d={d*1e3:.0f} mm, partners {ta:+.2f}/{tb:+.2f} d/c"
        # (i) analytic plane-wave RTFs, normalized to Mic A, then symmetrized by rtf.sym_rtf
        gA = rtf.sym_rtf(rtf.model_rtf(F, tau_a), F, tau_a)[1]
        gB = rtf.sym_rtf(rtf.model_rtf(F, tau_b), F, tau_b)[1]
        W, _, mode = rtf.ab_beam(R[None], gA[None], gB[None], FLOOR_BUS, opp, k_top)
        f_m3, ph_m3 = figures(W[0])
        e_db, e_ph = np.max(np.abs(f_m3 - f_ref)), np.max(np.abs(np.degrees(wrap(ph_m3 - ph_ref))))
        check(f"{tag}: analytic RTFs, every bin", e_db <= 0.1 and e_ph <= 1.0,
              f"max |diff| {e_db:.1e} dB (|r_A|, |r_B|, DF, WNG), {e_ph:.1e} deg; reference {t_ref:.1f} s, "
              f"modes ref {np.bincount(mode_ref, minlength=4).tolist()} m3 {np.bincount(mode[0], minlength=4).tolist()}")
        # (ii) RTFs estimated by covariance whitening from STFT-domain plane-wave data in diffuse + white noise.
        # Mirror-symmetric partners (tau_A = -tau_B) have exact DF ties between psi and -psi above c/2d that
        # estimation noise breaks either way, so there the phase is compared up to that mirror.
        Gs = []
        for tau in (tau_a, tau_b):
            Xk = stft_partner(rtf.model_rtf(F, tau), 8000, 30.0, R, rng)
            Phi = rtf._herm(*(r.mean(axis=-1) for r in cm.spectra(Xk)))[None]
            G, frac = rtf.checked_rtf(Phi, np.full((1, len(F)), 8000.0), R[None], np.array([tau]), rtf.PARAMS)
            Gs.append(G[0])
        gA = rtf.sym_rtf(Gs[0], F, tau_a)[1]
        gB = rtf.sym_rtf(Gs[1], F, tau_b)[1]
        W, _, mode = rtf.ab_beam(R[None], gA[None], gB[None], FLOOR_BUS, opp, k_top)
        f_m3, ph_m3 = figures(W[0])
        e_ph = np.abs(np.degrees(wrap(ph_m3 - ph_ref)))
        if np.isclose(tau_a, -tau_b):
            e_ph = np.minimum(e_ph, np.abs(np.degrees(wrap(ph_m3 + ph_ref))))
        # |r_A|, |r_B| and DF must match; WNG must meet the floor (below ~220 Hz the two steering vectors
        # almost coincide and ||w|| is barely constrained, so WNG itself is not compared there)
        e_db, e_ph = np.max(np.abs(f_m3[:3] - f_ref[:3])), np.max(e_ph)
        e_wng = np.max(np.abs(f_m3[3] - f_ref[3]))
        e_rtf = np.max(np.abs(np.r_[Gs[0] - rtf.model_rtf(F, tau_a), Gs[1] - rtf.model_rtf(F, tau_b)]))
        rA_e, rB_e = responses(W[0], gA, gB)
        wng_design = 10 * np.log10(np.minimum(np.abs(rA_e) ** 2, np.abs(rB_e) ** 2) / rtf._n2(W[0]))
        check(f"{tag}: estimated RTFs (30 dB SNR, 8000 frames), every bin",
              e_db <= 0.1 and e_ph <= 2.5 and np.min(wng_design - FLOOR_BUS) > -1e-6,
              f"max |diff| {e_db:.3f} dB (|r_A|, |r_B|, DF), {e_ph:.2f} deg; WNG >= floor (design RTFs), max WNG diff "
              f"{e_wng:.3f} dB at {F[np.argmax(np.abs(f_m3[3] - f_ref[3]))]:.0f} Hz; max |RTF err| {e_rtf:.4f}; "
              f"mode mismatches {int(np.sum(mode[0] != mode_ref))}")
        if tau_a > 0 > tau_b:
            # convention trap: h = [1, -1] on RTFs normalized to Mic A (unsymmetrized) is a different, poor beam
            rawA = np.stack([np.ones(len(F)), rtf.model_rtf(F, tau_a)], -1)
            rawB = np.stack([np.ones(len(F)), rtf.model_rtf(F, tau_b)], -1)
            band = (mode_ref == 0) & (F <= 0.8 * C / (2 * d))
            wr = rtf._lcmv_cands(R, rawA, rawB, 10 ** (FLOOR_BUS / 10), np.full((len(F), 1), np.pi), rtf.PARAMS)[0][:, 0]
            ok_raw = np.isfinite(rtf._quad(wr, R)) & (rtf._n2(wr) > 0)
            df_raw = figures(wr / np.maximum(np.sqrt(rtf._n2(wr)), 1e-300)[:, None])[0][2]
            loss = np.mean(f_ref[2][band & ok_raw] - df_raw[band & ok_raw])
            canonical = np.isclose(tau_a, -tau_b)         # DESIGN's case: A at 0 deg, B at 180 deg
            check(f"{tag}: Mic-A-referenced h=[1,-1] is a worse beam", loss > (1.0 if canonical else 0.0),
                  f"mean DF loss {loss:.1f} dB over the dipole band")


# ------------------------------------------------------------------ RTF estimation details
def test_rtf_estimation():
    print("RTF estimation: mismatch absorbed, plausibility check, freeze below 50 frames")
    rng = np.random.default_rng(2)
    d, tau = 0.04, 0.6 * 0.04 / C
    R = diffuse_R(d, 0.01)
    mis = 10 ** (1.0 / 20) * np.exp(1j * np.radians(5.0))       # unmodelled +1 dB / +5 deg on Mic B (e.g. scattering)
    Xk = stft_partner(mis * rtf.model_rtf(F, tau), 3000, 6.0, R, rng)
    Phi = rtf._herm(*(r.mean(axis=-1) for r in cm.spectra(Xk)))[None]
    G, frac = rtf.checked_rtf(Phi, np.full((1, len(F)), 3000.0), R[None], np.array([tau]), rtf.PARAMS)
    k = (F >= 300) & (F <= 3000)
    err = G[0, k] / (mis * rtf.model_rtf(F[k], tau))
    e_db, e_deg = np.median(np.abs(20 * np.log10(np.abs(err)))), np.median(np.abs(np.degrees(np.angle(err))))
    check("geometry-free: +1 dB / +5 deg mismatch recovered (6 dB SNR, 3000 frames)", e_db < 0.1 and e_deg < 1.0,
          f"median |err| {e_db:.3f} dB, {e_deg:.2f} deg; check band kept {frac:.2f}")
    Gbad = G.copy()
    Gbad[0, 40] *= np.exp(1j * np.radians(90))
    Gbad[0, 60] *= 10 ** (4 / 20)
    keep = rtf.plausible(Gbad, F, np.array([tau]), (300.0, 3000.0), 60.0, 3.0, 10.0)[0]
    check("plausibility check rejects +90 deg and +4 dB bins", not keep[40] and not keep[60] and keep[k].sum() == k.sum() - 2,
          f"rejected bins {np.nonzero(~keep & k)[0].tolist()}")
    G2, frac2 = rtf.checked_rtf(Phi, np.full((1, len(F)), 49.0), R[None], np.array([tau]), rtf.PARAMS)
    check("fewer than 50 labelled frames -> model steering", np.allclose(G2[0], rtf.model_rtf(F, tau)), f"kept {frac2}")


def test_recursion():
    print("labelled-frame recursion (forward, backward, resets, per-bin weights) vs a direct loop")
    rng = np.random.default_rng(3)
    Fn, N, alpha = 9, 300, 0.05
    X = rng.standard_normal((2, Fn, N)) + 1j * rng.standard_normal((2, Fn, N))
    for m in (rng.choice([0.0, 0.0, 0.4, 1.0], N), rng.uniform(0, 1, (Fn, N)) * (rng.uniform(size=N) > 0.3)):
        evals, resets = np.array([-1, 10, 57, 58, 150, 299]), [40, 151]
        for backward in (False, True):
            (a, b, c), n = rtf.recursive_cov(X, m, alpha, evals, backward=backward, resets=None if backward else resets,
                                             chunk=37)
            mm = np.broadcast_to(m, (Fn, N))
            for i, e in enumerate(evals):
                P, cnt = np.zeros((Fn, 2, 2), complex), np.zeros(Fn)
                frames = range(e + 1) if not backward else range(N - 1, max(e, 0) - 1 if e >= 0 else -1, -1)
                for l in frames:
                    if not backward and l in resets:
                        P[:], cnt[:] = 0, 0
                    x = X[:, :, l].T
                    P = (1 - alpha * mm[:, l])[:, None, None] * P + (alpha * mm[:, l])[:, None, None] * \
                        (x[:, :, None] * np.conj(x[:, None, :]))
                    cnt += mm[:, l]
                assert np.allclose(a[i], P[:, 0, 0].real) and np.allclose(b[i], P[:, 1, 1].real), (backward, e)
                assert np.allclose(c[i], P[:, 0, 1]) and np.allclose(n[i], cnt), (backward, e)
    check("recursive_cov == direct recursion", True, "frame and per-bin weights, chunked, with resets")


# ------------------------------------------------------------------ states, crossfade, phase continuity
def test_states():
    print("state machine (DESIGN M2) and 50 ms crossfades")
    p = rtf.PARAMS
    N = 420
    pA, pB = np.zeros(N), np.zeros(N)
    pA[50:150], pB[50:150] = 0.9, 0.1                  # A alone
    pA[200:260] = pB[200:260] = 0.5                    # double talk
    pB[300:400], pA[300:400] = 0.9, 0.05               # B alone
    pA[400:] = 0.4                                     # doubt
    sep = np.ones(N, bool)
    on = rtf.bus_states(pA, pB, sep, p, online=True)
    off = rtf.bus_states(pA, pB, sep, p, online=False)
    A, AB, B, FB = rtf.A_, rtf.AB_, rtf.B_, rtf.FB_
    check("online: AB before anyone talks, A only after 100 ms", (on[:62] == AB).all() and (on[62:150] == A).all(),
          f"first A frame {int(np.argmax(on == A))} (onset at 50, 13 frames = 104 ms)")
    check("online: hold A while nobody talks, AB on double talk, B after 100 ms, AB on doubt",
          (on[150:200] == A).all() and (on[200:260] == AB).all() and (on[312:400] == B).all() and (on[400:] == AB).all(),
          f"states 150-199 {set(on[150:200].tolist())}, 312-399 {set(on[312:400].tolist())}")
    check("offline look-ahead leaves A 100 ms before it ends", (off[62:137] == A).all() and (off[137:150] == AB).all(),
          f"last A frame {int(np.nonzero(off == A)[0].max())}")
    sep[100:120] = False
    check("inseparable frames use the fallback", (rtf.bus_states(pA, pB, sep, p, True)[100:120] == FB).all(), "FB")
    table = np.zeros((3, cm.N_BINS, 2), complex)
    for i in range(3):
        table[i, i, 0] = 1.0                           # entry i shows up as weight 1 in bin i: bins 0-2 = shares
    entry = np.r_[np.zeros(20, int), np.ones(10, int), np.full(20, 2)]
    bm = rtf.Beam(Y=None, table=table, entry=entry, n_x=6, state=entry, block=np.zeros(50, int), scenario="bus",
                  online=False)
    share = bm.weights(0, 50)[:, :3, 0].real
    check("crossfade: linear 6-frame (48 ms) ramps, shares sum to one",
          np.allclose(share.sum(axis=1), 1.0) and np.allclose(np.diff(share[19:26, 1]), 1 / 6)
          and np.allclose(share[:20, 0], 1.0) and np.allclose(share[35:, 2], 1.0),
          f"share of entry 1 over frames 19-26: {np.round(share[19:27, 1], 3).tolist()}")


def test_b7_phase_continuity():
    print("B.7 phase continuity across states A / AB / B (plane waves, bus floors)")
    worst_ph, worst_dip, n_rep = 0.0, 0.0, 0
    for d in (0.025, 0.04, 0.06):
        R = diffuse_R(d, 0.01)[None]
        kd = (F > 0) & (F <= 0.8 * C / (2 * d))
        for th_a, th_b in ((0, 180), (20, 150), (45, 135), (30, 100), (20, 50), (70, 110), (10, 170)):
            tau_a, tau_b = np.cos(np.radians([th_a, th_b])) * d / C
            gA, gB = cm.steer(F, tau_a)[None], cm.steer(F, tau_b)[None]
            k_top = np.array([int(np.argmin(np.abs(F - C / (2 * d))))])
            wA, wB = rtf.mvdr(R, gA, FLOOR_BUS)[0], rtf.mvdr(R, gB, FLOOR_BUS)[0]
            wAB = rtf.ab_beam(R, gA, gB, FLOOR_BUS, np.array([tau_a > 0 > tau_b]), k_top)[0]
            WA, WB, badA, badB = rtf.bus_state_weights(wA, wB, wAB, gA, gB)
            n_rep += int(badA[0, kd].sum() + badB[0, kd].sum())
            for g in (gA, gB):
                r = [rtf._dot(Wk, g)[0, kd] for Wk in (WA, wAB, WB)]
                for i, j in ((0, 1), (1, 2), (0, 2)):
                    r1, r2 = r[i], r[j]
                    ph = np.degrees(np.abs(np.angle(r2 / r1)))
                    strong = (np.abs(r1) >= 0.5) & (np.abs(r2) >= 0.5)
                    over = np.where(strong, ph - 30.0, ph - 90.0)
                    worst_ph = max(worst_ph, float(np.max(over)))
                    c = np.linspace(0, 1, 41)[:, None]
                    mid = np.min(np.abs((1 - c) * r1 + c * r2), axis=0)
                    dip = 20 * np.log10(np.minimum(np.abs(r1), np.abs(r2)) / np.maximum(mid, 1e-12))
                    worst_dip = max(worst_dip, float(np.max(dip)))
    check("phase change <= 30 deg (both >= -6 dB) / <= 90 deg elsewhere, every bin below 0.8 c/2d",
          worst_ph <= 1e-6, f"worst excess {worst_ph:+.2e} deg; bins switched to w_AB by the check: {n_rep}")
    check("no crossfade dip below the smaller endpoint by more than 1 dB", worst_dip <= 1.0, f"worst {worst_dip:.3f} dB")


def test_souden():
    print("Souden form (mask-driven variant)")
    rng = np.random.default_rng(4)
    d, tau = 0.04, -0.5 * 0.04 / C
    R = diffuse_R(d, 0.01)
    G = 10 ** (0.5 / 20) * rtf.model_rtf(F, tau)
    half, g = rtf.sym_rtf(G, F, tau)
    gh = np.stack([np.ones(len(F)), G], -1)
    h = rtf.souden(3.0 * rtf._outer(gh), R, g, half, FLOOR_BUS)
    w, _ = rtf.mvdr(R, g, FLOOR_BUS)
    check("rank-1 Phi_s: Souden == centre-referenced MVDR", np.max(np.abs(h - w)) < 1e-8, f"max |diff| {np.max(np.abs(h - w)):.1e}")
    h = rtf.souden(3.0 * rtf._outer(gh) + 0.5 * R, R, g, half, FLOOR_BUS)
    r = rtf._dot(h, g)
    wng = 10 * np.log10(2 / rtf._n2(g) * np.abs(r) ** 2 / rtf._n2(h))
    check("with noise in Phi_s: real response in (0, 1], WNG >= floor",
          np.max(np.abs(r.imag)) < 1e-9 and np.all((r.real > 0) & (r.real <= 1 + 1e-9)) and np.min(wng - FLOOR_BUS) > -1e-6,
          f"response {r.real.min():.3f}-{r.real.max():.3f}, min WNG-floor {np.min(wng - FLOOR_BUS):+.1e} dB")


# ------------------------------------------------------------------ end to end on synthetic scenes
def frac_delay(x, tau):
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * f * tau), nfft)[:n]


def partners(d, tau_a, tau_b, snr_db, dur, rng):
    """Alternating noise-burst partners (plane waves) in diffuse noise: components A, B, noise (2, T)."""
    T = int(dur * FS)
    comp = {"A": np.zeros((2, T)), "B": np.zeros((2, T)), "N": coherent_noise(T, d, rng) * 0.01}
    t, k = int(0.5 * FS), 0
    while t < T - 2 * FS:
        n = int(rng.uniform(0.8, 1.6) * FS)
        s = rng.standard_normal(n) * np.hanning(n) ** 0.25 * 0.01 * 10 ** (snr_db / 20)
        key, tau = (("A", tau_a), ("B", tau_b))[k]
        comp[key][0, t:t + n] += s
        comp[key][1, t:t + n] += frac_delay(s, tau)
        t += n + int(rng.uniform(0.3, 0.8) * FS)
        k = 1 - k
    return comp


def master_slave(bm, fe, comp, online, scenario="bus"):
    """Apply the method's own time-varying weights to each component (prefilter + S0 correction first)."""
    out = {}
    for k, c in comp.items():
        X = calib.apply(cm.stft(calib.prefilter(c, scenario)), fe.schedule if online else fe.calib)
        out[k] = (X, bm.apply(X))
    return out


def lag_samples(y, x, m=8):
    """Integer lag (samples) maximising |cross-correlation| of y against x (+ = y later); the magnitude,
    because the two-talker beam passes partner B with inverted polarity in the dipole band."""
    lags = np.arange(-m, m + 1)
    cc = [np.dot(y[max(0, l):len(y) + min(0, l)], x[max(0, -l):len(x) - max(0, l)]) for l in lags]
    return int(lags[int(np.argmax(np.abs(cc)))])


def test_end_to_end():
    print("end to end: S0 -> S1 -> M3 on synthetic partners (master/slave on the components)")
    rng = np.random.default_rng(5)
    for d, ta, tb in ((0.04, 0.8, -0.6), (0.06, 0.5, -0.9)):
        comp = partners(d, ta * d / C, tb * d / C, 6.0, 24.0, rng)
        x = sum(comp.values())
        for online, extra, tag in ((False, {}, "offline"), (True, {}, "online"),
                                   (False, dict(labels="mask", noise="measured", beam="souden"), "mask, offline")):
            t0 = time.perf_counter()
            fe = tracker.front_end(x, FS, BUS, online=online)
            bm = rtf.run(fe, BUS, online=online, **extra)
            rt = (time.perf_counter() - t0) / 24.0
            ms = master_slave(bm, fe, comp, online)
            Xn, Yn = ms["N"]
            res = []
            for k in ("A", "B"):
                Xk, Yk = ms[k]
                e = np.sum(np.abs(Xk[0]) ** 2, axis=0)
                act = e > 0.05 * e.max()
                snr = lambda S, Nn: 10 * np.log10(np.sum(np.abs(S[:, act]) ** 2) / np.sum(np.abs(Nn[:, act]) ** 2))
                d_snr = snr(Yk, Yn) - snr(Xk[0], Xn[0])
                ta_db = 10 * np.log10(np.sum(np.abs(Yk[:, act]) ** 2) / np.sum(np.abs(Xk[0][:, act]) ** 2))
                lag = lag_samples(cm.istft(Yk, x.shape[1]), cm.istft(Xk[0], x.shape[1]))
                res.append((d_snr, ta_db, lag))
            ta_fixed = [10 * np.log10(np.sum(np.abs(bm.steered(ms[k][0], s)) ** 2) / np.sum(np.abs(ms[k][0][0]) ** 2))
                        for k, s in (("A", rtf.A_), ("B", rtf.B_))]
            lim = 1.5 if extra else 1.0           # Souden form: noise in Phi_s acts like a Wiener gain (~ -1 dB)
            ok = all(r[0] > 1.0 and abs(r[1]) < lim and abs(r[2]) <= 2 for r in res) and rt < 0.5 \
                and np.all(np.isfinite(bm.Y)) and max(np.abs(ta_fixed)) < lim
            check(f"d={d*1e3:.0f} mm {ta:+.1f}/{tb:+.1f} d/c, {tag}", ok,
                  "dSNR A/B " + "/".join(f"{r[0]:+.2f}" for r in res) + " dB, level change "
                  + "/".join(f"{r[1]:+.2f}" for r in res) + " dB, lag " + "/".join(str(r[2]) for r in res)
                  + f" samples, steered(A/B) level " + "/".join(f"{v:+.2f}" for v in ta_fixed)
                  + f" dB, RTF {rt:.3f}, states {np.round([bm.diag['state_frac'][s] for s in rtf.STATE_NAMES], 2).tolist()}"
                  f" (A/AB/B/FB), RTF kept {bm.diag['rtf_kept_A']:.2f}/{bm.diag['rtf_kept_B']:.2f}")
    df = bm.df_frames()
    dip = (F >= 700.0) & (F <= 3000.0)
    check("DF table for M4 (active entry per frame)", df.shape == (cm.N_BINS, cm.n_frames(x.shape[1]))
          and np.all(np.isfinite(df)) and np.all(df > 0) and np.median(df[dip]) > 1.0,
          f"shape {df.shape}, median DF 0.7-3 kHz {10 * np.log10(np.median(df[dip])):+.2f} dB")
    y1 = rtf.enhance(x, FS, BUS)
    y2 = rtf.enhance(x.copy(), FS, BUS)
    check("deterministic, aligned length", np.array_equal(y1, y2) and y1.shape == (x.shape[1],), "two runs identical")
    flat = partners(0.04, 0.1 * 0.04 / C, -0.05 * 0.04 / C, 6.0, 20.0, rng)
    fe = tracker.front_end(sum(flat.values()), FS, BUS)
    bm = rtf.run(fe, BUS)
    check("inseparable (broadside) partners -> fallback beam only", bool(fe.track.inseparable) and
          bm.diag["state_frac"]["FB"] == 1.0, f"S1 {fe.track.inseparable_why}, states {bm.diag['state_frac']}")


def test_meeting():
    print("meeting mode (optional in DESIGN, off by default): RTF-MVDR steered to the active S1 cluster")
    rng = np.random.default_rng(7)
    d, seats = 0.04, (0.9, -0.7, 0.1)
    T = int(36 * FS)
    comp = {"N": coherent_noise(T, d, rng) * 0.002, "S": np.zeros((2, T))}
    t = int(0.5 * FS)
    for i in range(6):
        n = int(4.0 * FS)
        s = rng.standard_normal(n) * (0.15 + np.abs(np.sin(2 * np.pi * 4 * np.arange(n) / FS))) * 0.01
        comp["S"][0, t:t + n] += s
        comp["S"][1, t:t + n] += frac_delay(s, seats[i % 3] * d / C)
        t += n + int(1.2 * FS)
    x = sum(comp.values())
    for online in (False, True):
        fe = tracker.front_end(x, FS, MEETING, online=online)
        bm = rtf.run(fe, MEETING, online=online, meeting="cluster")
        ms = master_slave(bm, fe, comp, online, "meeting")
        (Xs, Ys), (Xn, Yn) = ms["S"], ms["N"]
        e = np.sum(np.abs(Xs[0]) ** 2, axis=0)
        act = e > 0.05 * e.max()
        lvl = 10 * np.log10(np.sum(np.abs(Ys[:, act]) ** 2) / np.sum(np.abs(Xs[0][:, act]) ** 2))
        d_snr = 10 * np.log10(np.sum(np.abs(Ys[:, act]) ** 2) / np.sum(np.abs(Yn[:, act]) ** 2)) \
            - 10 * np.log10(np.sum(np.abs(Xs[0][:, act]) ** 2) / np.sum(np.abs(Xn[0][:, act]) ** 2))
        check(f"{'online' if online else 'offline'}: talkers kept, noise not raised",
              abs(lvl) < 1.0 and d_snr > 0.0 and np.all(np.isfinite(bm.Y)),
              f"talker level change {lvl:+.2f} dB, dSNR {d_snr:+.2f} dB, clusters {bm.diag['n_clusters']}, "
              f"fallback frames {bm.diag['fallback_frac']:.2f}")


def test_fallbacks_and_modes():
    print("fallbacks and modes: 'auto' fallback, beyond-endfire TDOAs, states 'ab', fixed sigma_s, meeting modes")
    rng = np.random.default_rng(8)
    d = 0.04
    x = sum(partners(d, 0.8 * d / C, -0.6 * d / C, 6.0, 12.0, rng).values())
    fe = tracker.front_end(x, FS, BUS)
    N, trk = fe.X.shape[-1], fe.track

    def with_track(**kw):
        return replace(fe, track=replace(trk, **kw))

    a0 = cm.steer(F, 0.0)
    one = with_track(tau_B=trk.tau_A, tau_B_t=trk.tau_A_t.copy(), inseparable=True, inseparable_why=("one_peak",))
    bm = rtf.run(one, BUS, fallback="auto")
    check("'auto': one S1 peak -> Mic A (omni) in every frame", bm.diag["state_frac"]["FB"] == 1.0
          and np.array_equal(bm.Y, fe.X[0]), f"states {bm.diag['state_frac']}")
    flat = with_track(inseparable=True, inseparable_why=("both_broadside",))
    W = rtf.run(flat, BUS, fallback="auto").weights(0, N)
    r = rtf._dot(W, a0[None])
    check("'auto': both partners near broadside -> broadside MVDR (unit response, WNG >= floor)",
          np.allclose(r, 1.0) and np.min(10 * np.log10(1 / rtf._n2(W)) - FLOOR_BUS) > -1e-6,
          f"max |w^H a0 - 1| {np.max(np.abs(r - 1)):.1e}")
    far = 1.3 * fe.calib.d_hat / C
    bm = rtf.run(with_track(tau_A=far, tau_A_t=np.full(N, far)), BUS, fallback="auto")
    check("partner TDOA beyond 1.1 d_hat / c -> fallback", bm.diag["state_frac"]["FB"] == 1.0
          and np.array_equal(bm.Y, fe.X[0]), f"tau_A {far * 1e6:.0f} us, states {bm.diag['state_frac']}")
    t1, t2 = 0.9 * d / C, 0.4 * d / C                   # both partners on the Mic-A side of broadside
    same = with_track(tau_A=t1, tau_B=t2, tau_A_t=np.full(N, t1), tau_B_t=np.full(N, t2), inseparable=False,
                      inseparable_why=())
    fb = rtf.run(same, BUS, fallback="auto", same_side="fallback").diag["state_frac"]["FB"]
    srch = rtf.run(same, BUS, fallback="auto", same_side="search").diag["state_frac"]["FB"]
    check("same_side='fallback': same-side partners -> fallback ('search': two-talker beam)",
          fb == 1.0 and srch == 0.0, f"FB fraction {fb:.2f} / {srch:.2f}")
    bm = rtf.run(fe, BUS, states="ab", sigma_s=0.1)
    check("states='ab': the AB beam throughout; fixed sigma_s", bm.diag["state_frac"]["AB"] == 1.0
          and bm.diag["sigma_s_median"] == 0.1, f"states {bm.diag['state_frac']}")
    T = int(12 * FS)
    xm = coherent_noise(T, d, rng) * 0.002
    s = rng.standard_normal(T) * 0.01
    xm[0] += s
    xm[1] += frac_delay(s, 0.7 * d / C)
    fm = tracker.front_end(xm, FS, MEETING)
    om = rtf.run(fm, MEETING, meeting="omni")
    br = rtf.run(fm, MEETING, meeting="broadside")
    check("meeting='omni' is Mic A; meeting='broadside' never steers", np.array_equal(om.Y, fm.X[0])
          and br.diag["fallback_frac"] == 1.0 and br.diag["n_clusters"] == 0, f"clusters {br.diag['n_clusters']}")
    y = rtf.enhance(xm, FS, MEETING, return_stages=True)
    check("enhance(), meeting by default: S0's prefiltered Mic A", np.array_equal(y["out"], calib.prefilter(xm, "meeting")[0])
          and y["pre_agc"] is y["out"], "exact")
    ok = []
    for T in (0, 100, 511, 1000):
        xs = rng.standard_normal((2, T)) * 0.01
        for info, kw in ((BUS, {}), (BUS, dict(labels="mask", beam="souden")), (MEETING, dict(meeting="cluster"))):
            for online in (False, True):
                ys = rtf.enhance(xs, FS, info, online=online, **kw)
                ok.append(ys.shape == (T,) and bool(np.all(np.isfinite(ys))))
    check("short inputs (T = 0, 100, 511, 1000), offline and online", all(ok), f"{sum(ok)}/{len(ok)} runs")


def main():
    quick = "--quick" in sys.argv
    tests = [test_cw_and_symmetrization, test_b1_beams, test_b4, test_rtf_estimation, test_recursion, test_states,
             test_b7_phase_continuity, test_souden] + ([] if quick else [test_end_to_end, test_meeting,
                                                                        test_fallbacks_and_modes])
    t0 = time.perf_counter()
    for t in tests:
        t()
    print(f"all {len(tests)} test groups passed in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
