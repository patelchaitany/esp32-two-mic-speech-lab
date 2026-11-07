"""Tests for methods/_core: conventions, S0 calibration and S1 tracker (DESIGN.md Appendix B.1-B.4).

Run from LAB: .venv/bin/python -m tests.test_core [--quick]
--quick skips the behaviour checks on the v1 scenes (scenes/*.npz, read-only; tests may read their
meta, methods never do). Plain asserts; prints the measured numbers.
"""
import glob
import json
import sys
import time

import numpy as np
from scipy.signal import butter, freqz, sosfilt

from lab import dsp
from lab.common import C, FS
from lab.scenes import coherent_noise
from methods import s0_calib_only
from methods._core import calib, common as cm, spk, tracker

BUS = {"scenario": "bus", "spacing_m": 0.04}
MEETING = {"scenario": "meeting", "spacing_m": 0.04}


def frac_delay(x, tau):
    """Delay x by tau seconds (FFT phase shift, zero-padded to avoid wrap-around)."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * f * tau), nfft)[:n]


def plane_wave(tau, dur, rng):
    """2-channel white-noise plane wave with TDOA tau (Mic B delayed by tau)."""
    s = rng.standard_normal(int(dur * FS)) * 0.01
    return np.stack([s, frac_delay(s, tau)])


def cylindrical_noise(n, d, rng, k=32):
    """2-D (cylindrically isotropic) diffuse noise at spacing d: k independent white plane waves at equally
    spaced azimuths (random offset), centre-referenced; coherence J0(2 pi f d / c) up to 8 kHz at d <= 6 cm."""
    nfft = int(2 ** np.ceil(np.log2(n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    X = np.zeros((2, len(f)), complex)
    for phi in (np.arange(k) + rng.uniform()) * 2 * np.pi / k:
        S = np.fft.rfft(rng.standard_normal(nfft))
        X += S * cm.steer(f, d * np.cos(phi) / C).T
    return np.fft.irfft(X, nfft)[:, :n] / np.sqrt(k)


def check(name, ok, msg):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


# ------------------------------------------------------------------ conventions (B.1, B.4)
def test_conventions():
    print("conventions")
    rng = np.random.default_rng(0)
    f = cm.FREQS
    for tau in (-80e-6, 0.0, 50e-6):
        a = cm.steer(f, tau)
        g = cm.symmetrize_rtf(np.exp(-2j * np.pi * f * tau))          # RTF of B normalized to A
        check(f"symmetrized RTF == steer ({tau*1e6:+.0f} us)", np.allclose(a, g, atol=1e-9),
              "max dev %.1e" % np.max(np.abs(a - g)))
        check("plane coherence", np.allclose(cm.plane_coherence(f, tau), a[:, 0] * np.conj(a[:, 1])),
              "exp(+j2pi f tau)")
    x = plane_wave(100e-6, 1.0, rng)
    t1, t2 = cm.tdoa_gcc(x[0], x[1], max_tau=3e-4), dsp.gcc_phat(x[1], x[0], max_tau=3e-4)
    check("tdoa_gcc sign (B late -> +)", t1 > 0 and t1 == t2,
          f"{t1*1e6:+.2f} us (lab.dsp.gcc_phat(x_B, x_A) {t2*1e6:+.2f} us)")
    X = cm.stft(x)
    paa, pbb, pab = cm.spectra(X[:, :, 8:-12])
    ph = np.angle(pab.sum(1)[10:40])
    dev = np.max(np.abs(ph - 2 * np.pi * f[10:40] * 100e-6))
    check("Phi_AB phase = +2 pi f tau", dev < 0.02, "max dev %.3f rad" % dev)
    z = rng.standard_normal((2, 3 * cm.CHUNK * cm.HOP // 2 + 77))
    check("chunked stft == lab.dsp.stft", np.array_equal(cm.stft(z), dsp.stft(z)),
          "bitwise, %d frames" % cm.n_frames(z.shape[1]))
    y = np.zeros(4000)
    y[1000] = 1.0
    check("stft/istft aligned", np.argmax(np.abs(cm.istft(cm.stft(y), len(y)))) == 1000, "impulse at lag 0")
    check("n_frames", cm.stft(y).shape[-1] == cm.n_frames(len(y)), f"{cm.n_frames(len(y))}")
    check("lam table", abs(cm.lam(0.160) - cm.LAM_160MS) < 0.002 and abs(cm.lam(0.050) - cm.LAM_50MS) < 0.003,
          "exp(-8 ms / tau)")


# ------------------------------------------------------------------ B.1 plane wave, B.4 sign
def test_plane_wave():
    print("B.1 plane wave: S1 tau within +-2 us")
    rng = np.random.default_rng(1)
    worst = {}
    for d in (0.025, 0.04, 0.06):
        cal = calib.identity(d, "bus")
        for tau in (-d / C, 0.0, d / C):
            x = plane_wave(tau, 2.0, rng)
            X = cm.stft(calib.prefilter(x, "bus"))
            v = cm.valid_frames(X.shape[-1])
            for mode in ("fit", "design"):
                trk = tracker.track(X, cal, BUS, srp_mode=mode, partners=False)
                err = np.max(np.abs(trk.tau[v] - tau)) * 1e6
                worst[mode] = max(worst.get(mode, 0), err)
                check(f"d={d*1e3:.0f} mm tau={tau*1e6:+.1f} us [{mode}]", err <= 2.0, f"max |err| {err:.3f} us")
                t_gcc = cm.tdoa_gcc(x[0], x[1], max_tau=1.5 * d / C)
                if tau != 0:
                    check("B.4 gcc_phat(x_B, x_A) sign == S1 sign",
                          np.sign(t_gcc) == np.sign(np.median(trk.tau[v])), f"{t_gcc*1e6:+.1f} us")
            q = tracker.track(X, cal, BUS, partners=False).q[v]
            check("coherent fraction of a plane wave", np.median(q) > 0.95, f"median q {np.median(q):.3f}")
    print("  worst |tau error|:", {k: round(v, 3) for k, v in worst.items()}, "us")


def test_cdr():
    print("B.4 / B.2 CDR helpers")
    f = cm.FREQS
    for d in (0.025, 0.04, 0.06):
        tau = d / C
        Gs = cm.plane_coherence(f, tau)
        Gn = cm.diffuse_model(f, d)
        cond = (np.abs(Gs - Gn) >= 0.2) & (f >= 200) & (f <= 0.8 * C / (2 * d))
        cdr = 10 * np.log10(cm.cdr_doa(Gs, Gn, Gs)[cond])
        check(f"prop1, plane wave, true tau, d={d*1e3:.0f} mm", np.median(cdr) > 20,
              f"median {np.median(cdr):.1f} dB, min {np.min(cdr):.1f} dB over {cond.sum()} bins with |Gs - Gn| >= 0.2")
        # mixing model Gx = (CDR Gs + Gn) / (CDR + 1): prop1 (true tau) and prop3 (no DOA) return CDR exactly
        worst = 0.0
        for t in (-0.7 * tau, 0.3 * tau):
            Gs = cm.plane_coherence(f, t)
            for cdr_db in (-5.0, 0.0, 10.0):
                c = 10 ** (cdr_db / 10)
                Gx = (c * Gs + Gn) / (c + 1)
                k = (np.abs(Gs - Gn) >= 0.2) & (f >= 200) & (Gn <= 0.95) & (np.abs(Gx) <= 0.999)
                for est in (cm.cdr_doa(Gx, Gn, Gs), cm.cdr_nodoa(Gx, Gn)):
                    worst = max(worst, np.max(np.abs(10 * np.log10(np.maximum(est[k], 1e-12)) - cdr_db)))
        check(f"prop1 / prop3 on the mixing model, d={d*1e3:.0f} mm", worst < 0.1,
              f"max |error| {worst:.4f} dB (CDR -5 / 0 / 10 dB, two TDOAs)")
    rng = np.random.default_rng(2)
    d = 0.04
    x = coherent_noise(20 * FS, d, rng) * 0.01
    X = cm.stft(x)
    paa, pbb, pab = cm.spectra(X[:, :, 8:-12])
    G = cm.coherence(paa.sum(1), pbb.sum(1), pab.sum(1))
    k = f > 300
    cdr = 10 * np.log10(np.maximum(cm.cdr_nodoa(G[k], cm.diffuse_model(f[k], d)), 1e-4))
    check("long-term CDR of diffuse noise (prop3)", np.median(cdr) < -10,
          f"median {np.median(cdr):.1f} dB above 300 Hz")


# ------------------------------------------------------------------ B.2 diffuse field
def test_diffuse():
    print("B.2 diffuse field: Gamma_n, d_hat, model, rho")
    rng = np.random.default_rng(3)
    f = cm.FREQS
    k = f > 300
    for d in (0.025, 0.04, 0.06):
        x = coherent_noise(40 * FS, d, rng) * 0.01
        for snr_db in (None, 10.0):
            xx = x if snr_db is None else x + 0.01 * 10 ** (-snr_db / 20) * rng.standard_normal(x.shape)
            rho_true = 1.0 if snr_db is None else 1 / (1 + 10 ** (-snr_db / 10))
            cal = calib.calibrate(cm.stft(calib.prefilter(xx, "bus")), BUS)
            tag = f"d={d*1e3:.0f} mm" + ("" if snr_db is None else f", sensor noise {-snr_db:.0f} dB")
            gerr = np.max(np.abs(cal.gamma_n[k] - rho_true * np.sinc(2 * f[k] * d / C)))
            check(f"{tag}: Gamma_n vs rho*sinc above 300 Hz", gerr <= 0.05, f"max |err| {gerr:.3f}")
            check(f"{tag}: d_hat", abs(cal.d_hat - d) <= 0.002, f"{cal.d_hat*1e3:.2f} mm ({cal.diag['d_source']})")
            check(f"{tag}: model", cal.model == "sinc", cal.model)
            if snr_db is not None:
                check(f"{tag}: rho", abs(cal.rho - rho_true) <= 0.02, f"{cal.rho:.3f} vs {rho_true:.3f}")
            check(f"{tag}: no skew applied", cal.tau0 == 0.0,
                  f"tau0 {cal.tau0*1e6:.2f} us (fit {cal.diag['tau0_fit']*1e6:+.2f})")
        x = cylindrical_noise(30 * FS, d, rng) * 0.01
        cal = calib.calibrate(cm.stft(calib.prefilter(x, "bus")), BUS)
        j = cm.diffuse_model(f, d, "j0")
        kj = k & (j >= -0.3)                     # the measured part is clipped to [-0.3, 0.98] (DESIGN step 6)
        gerr = np.max(np.abs(cal.gamma_n[kj] - j[kj]))
        check(f"d={d*1e3:.0f} mm, 2-D field: J0 selected, d_hat, Gamma_n",
              cal.model == "j0" and abs(cal.d_hat - d) <= 0.002 and gerr <= 0.06 and cal.tau0 == 0.0,
              f"{cal.model}, {cal.d_hat*1e3:.2f} mm, Gamma_n vs J0 max |err| {gerr:.3f} where J0 >= -0.3, "
              f"tau0 {cal.tau0*1e6:.2f} us")


# ------------------------------------------------------------------ B.3 calibration
def test_calibration():
    print("B.3 calibration: +1.5 dB, 31.25 us, HPF corners 50 / 80 Hz")
    rng = np.random.default_rng(4)
    g = 10 ** (1.5 / 20)
    _, ha = freqz(*butter(1, 50, "high", fs=FS), worN=cm.FREQS, fs=FS)
    _, hb = freqz(*butter(1, 80, "high", fs=FS), worN=cm.FREQS, fs=FS)
    gain_true = np.array([10 * np.log10(np.sum(np.abs(ha[b]) ** 2) / np.sum(np.abs(g * hb[b]) ** 2))
                          for b in calib.BAND_BINS])
    _, ha_c = freqz(*butter(1, 50, "high", fs=FS), worN=calib.BANDS_HZ, fs=FS)
    _, hb_c = freqz(*butter(1, 80, "high", fs=FS), worN=calib.BANDS_HZ, fs=FS)
    phase_true = np.degrees(np.angle(ha_c / hb_c))
    for d in (0.025, 0.04, 0.06):
        x = coherent_noise(40 * FS, d, rng) * 0.01
        xx = np.stack([sosfilt(butter(1, 50, "high", fs=FS, output="sos"), x[0]),
                       g * frac_delay(sosfilt(butter(1, 80, "high", fs=FS, output="sos"), x[1]), 31.25e-6)])
        X0 = cm.stft(calib.prefilter(xx, "meeting"))
        cal = calib.calibrate(X0, MEETING)
        ge = np.max(np.abs(cal.gain_db - gain_true))
        pe = np.max(np.abs(cal.phase_deg - phase_true))
        te = abs(cal.diag["tau0_fit"] - 31.25e-6) * 1e6
        check(f"d={d*1e3:.0f} mm gain per band (125 Hz-6.3 kHz)", ge <= 0.2, f"max |err| {ge:.3f} dB")
        check(f"d={d*1e3:.0f} mm skew fit", te <= 3.0, f"tau0_fit {cal.diag['tau0_fit']*1e6:.2f} us (|err| {te:.2f})")
        check(f"d={d*1e3:.0f} mm skew applied", cal.diag["skew_applied"] and cal.tau0 == 0.5 / FS,
              f"tau0 {cal.tau0*1e6:.2f} us")
        check(f"d={d*1e3:.0f} mm corner phase per band", pe <= 1.0, f"max |err| {pe:.2f} deg")
        # end to end: residual phase of the diffuse cross-spectrum after apply()
        X = calib.apply(X0, cal)
        paa, pbb, pab = cm.spectra(X[:, :, 8:-12])
        S, PA, PB = pab.sum(1), paa.sum(1), pbb.sum(1)
        res = [abs(np.degrees(np.angle(S[b].sum()))) for b in calib.BAND_BINS
               if abs(S[b].sum()) > 0.3 * np.sqrt(PA[b].sum() * PB[b].sum())]
        check(f"d={d*1e3:.0f} mm residual band phase after apply", max(res) <= 1.5,
              f"max {max(res):.2f} deg over {len(res)} bands")
        lev = np.array([10 * np.log10(PA[b].sum() / PB[b].sum()) for b in calib.BAND_BINS])
        check(f"d={d*1e3:.0f} mm residual band level A/B after apply", np.max(np.abs(lev)) <= 0.3,
              f"max |A/B| {np.max(np.abs(lev)):.3f} dB over 18 bands")
        check(f"d={d*1e3:.0f} mm spacing with impairments", abs(cal.d_hat - d) <= 0.002, f"{cal.d_hat*1e3:.2f} mm")
    print("  short files (15 s, bus phase statistic), with and without the skew; offline and online")
    for d in (0.025, 0.04, 0.06):
        x = coherent_noise(15 * FS, d, rng) * 0.01
        for skew in (31.25e-6, 0.0):
            xx = np.stack([sosfilt(butter(1, 50, "high", fs=FS, output="sos"), x[0]),
                           g * frac_delay(sosfilt(butter(1, 80, "high", fs=FS, output="sos"), x[1]), skew)])
            X0 = cm.stft(calib.prefilter(xx, "bus"))
            cal = calib.calibrate(X0, BUS)
            on = calib.calibrate_online(X0, BUS)[-1][1]
            pe = np.max(np.abs(cal.phase_deg - phase_true))
            check(f"d={d*1e3:.0f} mm skew {skew*1e6:.2f} us, 15 s", cal.tau0 == skew and on.tau0 == skew and pe <= 1.5
                  and abs(cal.d_hat - d) <= 0.002,
                  f"tau0 {cal.tau0*1e6:.2f} (online {on.tau0*1e6:.2f}) us, fit {cal.diag['tau0_fit']*1e6:+.2f} us, "
                  f"nD {cal.diag['n_diffuse']}, corner phase max |err| {pe:.2f} deg, d_hat {cal.d_hat*1e3:.2f} mm")


def test_calib_json():
    print("calib.json overrides")
    rng = np.random.default_rng(5)
    x = coherent_noise(20 * FS, 0.04, rng) * 0.01
    X = cm.stft(calib.prefilter(x, "bus"))
    cfg = {"version": 1, "spacing_m": 0.0402, "tau0_s": 31.25e-6, "hpf_corner_hz": [50, 80],
           "gain_db": [0.5] * len(calib.BANDS_HZ), "model": "sinc", "rho": 0.95}
    cal = calib.calibrate(X, dict(BUS, calib_json=cfg))
    check("json spacing / skew", cal.d_hat == 0.0402 and cal.tau0 == 31.25e-6 and cal.source == "json",
          f"{cal.d_hat} {cal.tau0}")
    want = np.degrees(calib.corner_phase(calib.BANDS_HZ[0], 50, 80))
    check("json corner phase", abs(cal.phase_deg[0] - want) < 1e-9, f"{cal.phase_deg[0]:.2f} deg")
    drift = cal.diag["json_drift_db"]
    check("json gain + drift", np.allclose(cal.gain_db, 0.5 + drift) and abs(drift + 0.5) < 0.2,
          f"drift {cal.diag['json_drift_db']:+.2f} dB (blind gain is ~0 dB)")
    expect = 10 ** (0.5 / 20) * np.exp(1j * (calib.corner_phase(cm.FREQS, 50, 80) + 2 * np.pi * cm.FREQS * 31.25e-6))
    expect *= 10 ** (cal.diag["json_drift_db"] / 20)
    check("json correction vector", np.allclose(cal.corr, expect), "corr = g exp(j(corner + 2 pi f tau0))")


# ------------------------------------------------------------------ S1 bus partner model
def two_partners(d, tau_a, tau_b, snr_db, dur, rng):
    """Alternating noise-burst 'talkers' (plane waves) in diffuse noise; returns x and per-frame truth."""
    T = int(dur * FS)
    noise = coherent_noise(T, d, rng) * 0.01
    xa = np.zeros(T)
    xb = np.zeros(T)
    who = np.full(T, -1)
    t, k = int(0.5 * FS), 0
    env = lambda n: np.hanning(n) ** 0.25
    while t < T - 2 * FS:
        n = int(rng.uniform(0.8, 1.6) * FS)
        s = rng.standard_normal(n) * env(n) * 0.01 * 10 ** (snr_db / 20)
        tau = (tau_a, tau_b)[k]
        xa[t:t + n] += s
        xb[t:t + n] += frac_delay(s, tau)
        who[t:t + n] = k
        t += n + int(rng.uniform(0.3, 0.8) * FS)
        k = 1 - k
    x = noise + np.stack([xa, xb])
    cen = cm.frame_centers(cm.n_frames(T))
    w = np.where((cen >= 0) & (cen < T), who[np.clip(cen, 0, T - 1)], -1)
    return x, w


def test_partners():
    print("S1 bus partner model on synthetic partners")
    rng = np.random.default_rng(6)
    for d, ta, tb in ((0.04, 0.8, -0.6), (0.06, 0.5, -0.9), (0.025, 0.95, 0.3)):
        tau_a, tau_b = ta * d / C, tb * d / C
        x, who = two_partners(d, tau_a, tau_b, 6.0, 30.0, rng)
        for online in (False, True):
            fe = tracker.front_end(x, FS, BUS, online=online)
            trk = fe.track
            tag = f"d={d*1e3:.0f} mm, {ta:+.2f}/{tb:+.2f} d/c, {'online' if online else 'offline'}"
            e = max(abs(trk.tau_A - tau_a), abs(trk.tau_B - tau_b)) * 1e6
            check(f"{tag}: tau_A/tau_B", e <= 5.0, f"{trk.tau_A*1e6:+.1f}/{trk.tau_B*1e6:+.1f} us "
                  f"(true {tau_a*1e6:+.1f}/{tau_b*1e6:+.1f}), d_hat {trk.d_hat*1e3:.1f} mm")
            check(f"{tag}: not inseparable", not trk.inseparable, str(trk.inseparable_why))
            late = np.arange(len(who)) > int(8.0 / cm.HOP_S)       # online model needs its first seconds
            pa_a, pa_b = trk.p_A[(who == 0) & late].mean(), trk.p_A[(who == 1) & late].mean()
            pb_b = trk.p_B[(who == 1) & late].mean()
            check(f"{tag}: posteriors", pa_a > 0.6 and pb_b > 0.6 and pa_b < 0.2,
                  f"p_A|A {pa_a:.2f}, p_B|B {pb_b:.2f}, p_A|B {pa_b:.2f}, "
                  f"double talk {trk.double_talk[who >= 0].mean():.2f}")
            want = tracker.orientation_stratum(tracker.angle_deg(tau_a, d), tracker.angle_deg(tau_b, d))
            check(f"{tag}: stratum", trk.stratum == want, f"{trk.stratum} (true {want})")
    x, _ = two_partners(0.04, 0.2 * 0.04 / C, -0.15 * 0.04 / C, 6.0, 30.0, rng)
    trk = tracker.front_end(x, FS, BUS).track
    check("broadside pair flagged inseparable", trk.inseparable,
          f"{trk.inseparable_why}, tau {trk.tau_A*1e6:+.1f}/{trk.tau_B*1e6:+.1f} us")


def test_one_partner():
    print("S1 bus partner model with one histogram peak (only one partner localized)")
    rng = np.random.default_rng(10)
    d = 0.04
    tau = 0.8 * d / C
    x, who = two_partners(d, tau, tau, 6.0, 30.0, rng)       # every burst from the same TDOA
    talk = (who >= 0) & (np.arange(len(who)) > int(8.0 / cm.HOP_S))
    quiet = (who < 0) & cm.valid_frames(len(who))
    for online in (False, True):
        trk = tracker.front_end(x, FS, BUS, online=online).track
        tag = "online" if online else "offline"
        check(f"{tag}: one peak at the partner", len(trk.diag["peaks"]) == 1 and trk.tau_A == trk.tau_B
              and abs(trk.tau_A - tau) <= 5e-6 and trk.inseparable_why == ("one_peak",),
              f"peaks {[round(p[0] * 1e6, 1) for p in trk.diag['peaks']]} us (true {tau*1e6:+.1f}), "
              f"{trk.inseparable_why}")
        one = trk.tau_A_t == trk.tau_B_t
        gate = np.mean(trk.p_A[talk] + trk.p_B[talk] > 0.8)
        check(f"{tag}: single-peak posterior in p_A, p_B = 0, no double talk",
              trk.p_A[talk].mean() > 0.6 and not trk.p_B[one].any() and not trk.double_talk[one].any(),
              f"p_A|talk {trk.p_A[talk].mean():.2f}, p_A|quiet {trk.p_A[quiet].mean():.2f}, "
              f"max p_B {trk.p_B[one].max():.2f}")
        check(f"{tag}: AGC gate p_A + p_B > 0.8 fires while the partner talks", gate > 0.3,
              f"{gate:.2f} of talk frames (quiet frames {np.mean(trk.p_A[quiet] > 0.8):.2f})")


TAGS = ((300.0, 1200.0), (1200.0, 2400.0), (2400.0, 3400.0))     # spectral tag per seat


def tagged_burst(k, n, rng):
    """Speech-like burst (4 Hz syllabic modulation) with a +10 dB band that identifies seat k."""
    S = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    S *= 1 + 2.2 * ((f > TAGS[k][0]) & (f < TAGS[k][1]))
    syl = 0.15 + np.abs(np.sin(2 * np.pi * 4.0 * np.arange(n) / FS))
    ramp = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.03 * FS))   # 30 ms on/off
    return np.fft.irfft(S, n) * syl * ramp * 0.01


def fake_embedder(y):
    """Embedding stand-in: one-hot of the strongest tag band (tests the embedding route only)."""
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / FS)
    return np.eye(4)[int(np.argmax([P[(f > lo) & (f < hi)].mean() for lo, hi in TAGS]))]


def test_meeting_and_spk():
    print("S1 meeting clustering, change points and the embedding hook")
    rng = np.random.default_rng(7)
    d = 0.04
    seats = (0.9, -0.7, 0.1)
    T = int(48 * FS)
    x = coherent_noise(T, d, rng) * 0.002
    t, truth = int(0.5 * FS), []
    for i in range(9):
        n, k = int(4.0 * FS), i % 3
        s = tagged_burst(k, n, rng)
        x[0, t:t + n] += s
        x[1, t:t + n] += frac_delay(s, seats[k] * d / C)
        truth.append((t, t + n, k))
        t += n + int(1.2 * FS)
    cen = cm.frame_centers(cm.n_frames(T))

    def seat_clusters(lab):
        out = []
        for a, b, k in truth:
            m = (cen >= a + 0.3 * FS) & (cen < b - 0.3 * FS) & (lab >= 0)
            out.append((k, int(np.bincount(lab[m]).argmax()) if m.any() else -1))
        consistent = all(len({c for k2, c in out if k2 == k}) == 1 for k in range(3))
        return out, consistent and len({c for _, c in out}) == 3

    fe = tracker.front_end(x, FS, MEETING)
    out, ok = seat_clusters(fe.track.cluster)
    check("DP-means: 3 seats -> 3 consistent clusters", ok, f"(seat, cluster) {out}")
    hit = [fe.track.change[max(0, cm.frame_of_sample(a) - 10):cm.frame_of_sample(a) + 40].any() for a, _, _ in truth]
    check("change proposal at every turn", all(hit), f"{int(fe.track.change.sum())} proposals for {len(truth)} turns")
    # Online S0 refreshes its blind skew every 5 s. Here the pauses of the first 5 s hold one anechoic
    # talker, whose TDOA reads as a slope of ~+100 us, so 5-10 s get a false 31.25 us skew and DP-means
    # splits a seat (API.md, known limitations). This test checks the clustering: skew as from calib.json.
    on = tracker.front_end(x, FS, dict(MEETING, calib_json={"version": 1, "tau0_s": 0.0}), online=True).track
    out, ok = seat_clusters(on.cluster)
    check("online DP-means", ok, f"(seat, cluster) {out}")
    for online in (False, True):
        emb = tracker.track(fe.X, fe.calib, MEETING, online=online, x=fe.x[0], embedder=fake_embedder)
        out, ok = seat_clusters(emb.cluster)
        check(f"embedding-keyed clusters, fake embedder ({'online' if online else 'offline'})", ok,
              f"(seat, cluster) {out}")
    feats = spk.fbank(np.random.default_rng(0).standard_normal(16000) * 0.01)
    check("fbank shape", feats.shape == (98, 80) and np.isfinite(feats).all(), str(feats.shape))


def test_short_inputs():
    print("robustness: inputs shorter than one STFT frame (T < 512) and empty input")
    rng = np.random.default_rng(11)
    for T in (0, 1, 100, 511, 512, 1000):
        x = rng.standard_normal((2, T)) * 0.01
        for info in (BUS, MEETING):
            for online in (False, True):
                trk = tracker.front_end(x, FS, info, online=online).track
                arrays = [trk.tau, trk.q, trk.psi, trk.cdr_db, trk.vad, trk.level_db]
                arrays += [trk.p_A, trk.p_B] if trk.p_A is not None else []
                ok = all(len(a) == cm.n_frames(T) and np.all(np.isfinite(a)) for a in arrays)
                assert ok, f"T={T} {info['scenario']} online={online}: bad per-frame output"
            y = s0_calib_only.enhance(x, FS, info)
            assert y.shape == (T,) and np.all(np.isfinite(y)), f"T={T} s0_calib_only shape {y.shape}"
    check("front end and s0_calib_only on T = 0, 1, 100, 511, 512, 1000", True,
          "no exception, finite per-frame arrays, output length T")


def test_speed():
    print("speed: S0 + S1 RTF < 0.1")
    rng = np.random.default_rng(8)
    x, _ = two_partners(0.04, 0.8 * 0.04 / C, -0.6 * 0.04 / C, 6.0, 32.0, rng)
    for online in (False, True):
        t0 = time.perf_counter()
        tracker.front_end(x, FS, BUS, online=online)
        rtf = (time.perf_counter() - t0) / 32.0
        check(f"bus front end {'online' if online else 'offline'}", rtf < 0.1, f"RTF {rtf:.4f}")
    y = np.concatenate([x, x, x], axis=1)
    t0 = time.perf_counter()
    tracker.front_end(y, FS, MEETING)
    rtf = (time.perf_counter() - t0) / (y.shape[1] / FS)
    check("meeting front end (96 s)", rtf < 0.1, f"RTF {rtf:.4f}")


# ------------------------------------------------------------------ v1 scenes (behaviour, read-only)
def true_geometry(meta):
    mA, mB = np.array(meta["mics"][0]), np.array(meta["mics"][1])
    c, u = (mA + mB) / 2, (mA - mB) / np.linalg.norm(mA - mB)
    taus, thetas = [], []
    for p in meta.get("talkers", []):
        p = np.array(p)
        taus.append((np.linalg.norm(p - mB) - np.linalg.norm(p - mA)) / C)
        thetas.append(np.degrees(np.arccos(np.clip(u @ (p - c) / np.linalg.norm(p - c), -1, 1))))
    return taus, thetas


def test_v1_scenes():
    print("v1 scenes (behaviour; scenes/*.npz, meta read by the test only)")
    rows, d_err, tau_err, tau_err_design, strata, gain_err, false_skew = [], [], [], [], [], [], []
    for path in sorted(glob.glob("scenes/*.npz")):
        z = np.load(path)
        x, meta = z["mix"].astype(np.float64), json.loads(str(z["meta"]))
        info = {"scenario": meta["scenario"], "spacing_m": 0.04}
        t0 = time.perf_counter()
        fe = tracker.front_end(x, FS, info)
        rtf = (time.perf_counter() - t0) / (x.shape[1] / FS)
        cal, trk = fe.calib, fe.track
        name = path.split("/")[-1][:-4]
        d_err.append((meta["scenario"], (cal.d_hat - meta["spacing_m"]) * 1e3))
        g_true = meta["mic_gain_db"][0] - meta["mic_gain_db"][1]
        gain_err.append(np.median(cal.gain_db) - g_true)
        if cal.tau0 != 0.0:
            false_skew.append(name)
        line = (f"  {name:22s} d {meta['spacing_m']*1e3:.0f}->{cal.d_hat*1e3:5.1f} mm ({cal.diag['d_source'][:5]}) "
                f"gain err {gain_err[-1]:+.2f} dB tau0_fit {cal.diag['tau0_fit']*1e6:+5.1f} us RTF {rtf:.3f}")
        if meta["scenario"] == "bus":
            taus, thetas = true_geometry(meta)
            t_hi, t_lo = max(taus), min(taus)
            e = [abs(trk.tau_A - t_hi) * 1e6, abs(trk.tau_B - t_lo) * 1e6]
            tau_err += e
            dtrk = tracker.track(fe.X, cal, info, srp_mode="design")
            tau_err_design += [abs(dtrk.tau_A - t_hi) * 1e6, abs(dtrk.tau_B - t_lo) * 1e6]
            want = tracker.orientation_stratum(*thetas)
            strata.append((want, trk.stratum))
            line += (f" | tau {t_hi*1e6:+6.1f}/{t_lo*1e6:+6.1f} -> {trk.tau_A*1e6:+6.1f}/{trk.tau_B*1e6:+6.1f} us"
                     f" (design {dtrk.tau_A*1e6:+6.1f}/{dtrk.tau_B*1e6:+6.1f}) theta {thetas[0]:.0f}/{thetas[1]:.0f}"
                     f" -> {trk.theta_A_deg:.0f}/{trk.theta_B_deg:.0f} stratum {want}->{trk.stratum}"
                     f" insep {','.join(trk.inseparable_why) or '-'}")
        else:
            line += f" | clusters {len(trk.clusters)} vad {trk.vad.mean():.2f}"
        rows.append(line)
        for k in ("tau", "q", "psi", "cdr_db", "vad", "level_db"):
            assert np.all(np.isfinite(getattr(trk, k))), f"{name}: non-finite {k}"
        assert rtf < 0.1, f"{name}: RTF {rtf:.3f}"
    print("\n".join(rows))
    bus_d = np.abs([e for s, e in d_err if s == "bus"])
    meet_d = np.abs([e for s, e in d_err if s == "meeting"])
    te, td = np.array(tau_err), np.array(tau_err_design)
    agree = np.mean([a == b for a, b in strata])
    print(f"  bus |d_hat err|: median {np.median(bus_d):.1f} mm, <=3 mm {np.mean(bus_d <= 3):.2f}; "
          f"meeting |d_hat err| per scene {np.round(meet_d, 1).tolist()} mm (fits fall back to the prior)")
    print(f"  bus partner |tau err| (fit): median {np.median(te):.1f} us, <=30 us {np.mean(te <= 30):.2f}; "
          f"design SRP: median {np.median(td):.1f} us, <=30 us {np.mean(td <= 30):.2f}")
    print(f"  stratum agreement {agree:.2f}; median |gain err| {np.median(np.abs(gain_err)):.2f} dB")
    check("bus spacing (regression guard)", np.median(bus_d) <= 3.0, f"median {np.median(bus_d):.1f} mm")
    check("gain vs true mic mismatch", np.max(np.abs(gain_err)) <= 0.3, f"max {np.max(np.abs(gain_err)):.2f} dB")
    check("fit beats design SRP on partner TDOAs", np.median(te) < np.median(td),
          f"{np.median(te):.1f} vs {np.median(td):.1f} us")
    check("partner TDOAs (regression guard)", np.median(te) <= 30.0, f"median {np.median(te):.1f} us")
    check("false blind skews on v1 (no skew; regression guard)", len(false_skew) <= 3,
          f"{len(false_skew)} of {len(rows)}: {false_skew}")


def main():
    quick = "--quick" in sys.argv
    tests = [test_conventions, test_plane_wave, test_cdr, test_diffuse, test_calibration, test_calib_json,
             test_partners, test_one_partner, test_meeting_and_spk, test_short_inputs, test_speed]
    tests += [] if quick else [test_v1_scenes]
    t0 = time.perf_counter()
    for t in tests:
        t()
    print(f"all {len(tests)} test groups passed in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
