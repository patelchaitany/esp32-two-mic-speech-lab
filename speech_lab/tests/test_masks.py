"""Tests for methods/_core/masks.py (M4 CDR post-filter, M5 phase fence) and the m4_cdr / m5_fence
wrappers (DESIGN.md M4, M5, 2.1, Appendix B.2 and B.4).

Run from LAB: .venv/bin/python -m tests.test_masks
Synthetic signals only (no scene files). Plain asserts; prints the measured numbers.
"""
import time

import numpy as np

from lab.common import C, FS
from lab.scenes import coherent_noise
from methods import m4_cdr, m5_fence
from methods._core import calib, common as cm, masks, ns, tracker

BUS = {"scenario": "bus", "spacing_m": 0.04}
MEETING = {"scenario": "meeting", "spacing_m": 0.04}
F = cm.FREQS
DS = (0.025, 0.04, 0.06)
DESIGN_M4 = dict(mu=1.3, g_min=0.15, cond=0.2, fill_pct=50.0)   # DESIGN M4 start values (used by the B.2 gain check)


def check(name, ok, msg):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-12))


def frac_delay(x, tau):
    """Delay x by tau seconds (FFT phase shift, zero-padded against wrap-around)."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * f * tau), nfft)[:n]


def plane_wave(tau, dur, rng, level=0.01):
    s = rng.standard_normal(int(dur * FS)) * level
    return np.stack([s, frac_delay(s, tau)])


def coherence(X, lam):
    """Recursively smoothed complex coherence of a 2-channel STFT (F, N)."""
    _, _, sm = next(cm.smoothed_spectra(X, slice(None), (lam,), chunk=X.shape[-1]))
    return cm.coherence(*sm[lam])


def lag(y, ref, max_lag=100):
    """Lag (samples) of y relative to ref at the cross-correlation peak."""
    n = 1 << int(np.ceil(np.log2(2 * len(y))))
    xc = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(ref, n)), n)
    lags = np.r_[np.arange(max_lag + 1), np.arange(-max_lag, 0)]
    return int(lags[np.argmax(np.abs(np.r_[xc[:max_lag + 1], xc[-max_lag:]]))])


def burst(n, rng, level):
    """Voiced-speech-like burst of RMS `level`: harmonics (1/k) of a 100-220 Hz f0 with vibrato up
    to 7 kHz, 4 Hz syllabic modulation and 30 ms ramps (sparse in time and frequency, like speech)."""
    t = np.arange(n) / FS
    f0 = rng.uniform(100, 220) * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))
    ph = 2 * np.pi * np.cumsum(f0) / FS
    s = sum(np.sin(k * ph + rng.uniform(0, 2 * np.pi)) / k for k in range(1, int(7000 / f0.max()) + 1))
    syl = 0.15 + np.abs(np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, np.pi)))
    ramp = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.03 * FS))
    s = s * syl * ramp
    return s * level / np.sqrt(np.mean(s ** 2))


def speech_shape(f):
    """Amplitude spectrum of the test noise: flat to 500 Hz, then -6 dB per octave (babble-like tilt)."""
    return 1.0 / np.sqrt(1.0 + (f / 500.0) ** 2)


def partner_scene(d, taus, snr_db, dur, rng):
    """Alternating partner bursts (anechoic plane waves at TDOAs `taus`) in speech-shaped diffuse noise
    (RMS 0.01; snr_db = partner RMS relative to it). Returns the mix and its components: (2, T)
    target (both partners) and (2, T) noise, and a per-sample bool of partner activity."""
    T = int(dur * FS)
    noise = coherent_noise(T, d, rng, shape=speech_shape) * 0.01
    target = np.zeros((2, T))
    act = np.zeros(T, bool)
    t, k = int(0.5 * FS), 0
    while t < T - int(1.5 * FS):
        n = int(rng.uniform(0.8, 1.6) * FS)
        s = burst(n, rng, 0.01 * 10 ** (snr_db / 20))
        target[0, t:t + n] += s
        target[1, t:t + n] += frac_delay(s, taus[k])
        act[t:t + n] = True
        t += n + int(rng.uniform(0.3, 0.8) * FS)
        k = 1 - k
    return target + noise, target, noise, act


# ------------------------------------------------------------------ gain law and conditioning
def test_gain_law():
    print("M4 gain law G = max(G_min, 1 - sqrt(mu / (CDR * DF + 1)))")
    for mu, want in ((0.3, -6.9), (0.5, -10.7)):
        g = masks.gain_from_cdr(np.array(0.0), mu=mu, g_min=0.0)
        check(f"G(CDR = 0), mu {mu}", abs(20 * np.log10(g) - want) < 0.05, f"{20*np.log10(g):.2f} dB (DESIGN {want})")
    cdr = np.logspace(-3, 3, 50)
    g = masks.gain_from_cdr(cdr)
    check("range and monotone (mu 1.3, G_min 0.15)", g.min() >= 0.15 and g.max() <= 1 and np.all(np.diff(g) >= 0),
          f"{g.min():.3f} .. {g.max():.3f}")
    check("no information (CDR = inf) passes", masks.gain_from_cdr(np.array(np.inf)) == 1.0, "G = 1")
    check("beam correction CDR_out = CDR * DF", np.allclose(masks.gain_from_cdr(cdr, df=2.5),
                                                             masks.gain_from_cdr(2.5 * cdr)), "DF 2.5")
    p = masks.CDR_PARAMS
    g = masks.gain_from_cdr(np.array(0.0), p["mu"], p["g_min"])
    check("tuned default: deepest gain", np.isclose(g, max(p["g_min"], 1 - np.sqrt(p["mu"]))),
          f"{20*np.log10(g):.1f} dB (mu {p['mu']}, G_min {p['g_min']})")


def test_conditioning_edges():
    print("M4 conditioning |Gs - Gn| >= 0.2 (model Gn = sinc): lowest conditioned bin vs DESIGN M4")
    design = {"broadside": (2470, 1540, 1030), "endfire": (440, 270, 180)}
    for i, d in enumerate(DS):
        gn = np.minimum(cm.diffuse_model(F, d), 0.98)
        for name, tau in (("broadside", 0.0), ("endfire", d / C)):
            ok = np.abs(cm.plane_coherence(F, tau) - gn) >= 0.2
            first = F[np.argmax(ok & (F > 0))]
            check(f"d={d*1e3:.0f} mm {name}", abs(first - design[name][i]) <= 40 and ok[F >= first].all(),
                  f"conditioned from {first:.0f} Hz up (DESIGN {design[name][i]} Hz)")


def test_fill():
    print("M4 fill rule: ill-conditioned bins <- frame median of conditioned bins in 1-4 kHz")
    rng = np.random.default_rng(10)
    cdr = rng.uniform(0, 5, (len(F), 6))
    ok = np.ones_like(cdr, bool)
    ok[F < 1500] = False
    ok[:, 4] = (F > 4100)                     # frame 4: conditioned bins only above the band
    ok[:, 5] = False                          # frame 5: nothing conditioned
    band = (F >= 1000) & (F <= 4000)
    out = masks.fill_unconditioned(cdr, ok, band)
    want = np.median(cdr[(F >= 1500) & (F <= 4000), 0])
    check("conditioned bins kept", np.array_equal(out[ok], cdr[ok]), "unchanged")
    check("fill value = band median", np.allclose(out[F < 1500, 0], want), f"{want:.3f}")
    check("no conditioned bin in band -> median of all conditioned", np.allclose(out[~ok[:, 4], 4],
          np.median(cdr[F > 4100, 4])), "ok")
    check("nothing conditioned -> +inf (G = 1)", np.isinf(out[:, 5]).all(), "ok")
    out = masks.fill_unconditioned(cdr, ok, band, pct=80)
    check("fill_pct 80", np.allclose(out[F < 1500, 0], np.percentile(cdr[(F >= 1500) & (F <= 4000), 0], 80)), "ok")
    m = rng.uniform(size=(40, 300)) < 0.3
    m[:, :3] = False
    a = rng.standard_normal((40, 300))
    for q in (50, 25, 90):
        want = [np.percentile(a[m[:, j], j], q) if m[:, j].any() else np.nan for j in range(300)]
        check(f"masked column percentile q={q}", np.allclose(masks._col_pct(a, m, q), want, equal_nan=True),
              f"{(~m.any(0)).sum()} empty columns")


# ------------------------------------------------------------------ B.4 plane wave
def test_b4_plane_wave():
    print("B.4 plane wave: CDR prop1 with the true tau > +20 dB (median over conditioned bins)")
    rng = np.random.default_rng(11)
    for d in DS:
        gn = calib.identity(d, "bus").gamma_n
        for frac in (1.0, 0.5, 0.0, -0.7):
            tau = frac * d / C
            X = cm.stft(plane_wave(tau, 2.0, rng))
            v = cm.valid_frames(X.shape[-1])
            Gx = coherence(X, masks.CDR_PARAMS["lam"])[:, v]
            cond = np.abs(cm.plane_coherence(F, tau) - gn) >= 0.2
            res = {}
            for tag, taus, est in (("prop1", [[tau]], "prop1"), ("prop1 A+B", [[tau], [-tau - 0.4 * d / C]], "prop1"),
                                   ("prop2", [[tau]], "prop2"), ("prop3 (no DOA)", None, "prop1")):
                t = None if taus is None else np.repeat(np.array(taus), Gx.shape[1], axis=1)
                cdr = masks.cdr_estimate(Gx, gn, t, estimator=est)
                res[tag] = np.median(db(cdr[cond]))
            ok = min(res["prop1"], res["prop1 A+B"]) > 20
            check(f"d={d*1e3:.0f} mm tau={frac:+.1f} d/c", ok,
                  ", ".join(f"{k} {v:.1f} dB" for k, v in res.items()) + f" ({cond.sum()} bins)")


def test_known_cdr():
    print("estimator bias: stationary plane wave + diffuse noise at a known CDR (conditioned bins, 1-6 kHz)")
    rng = np.random.default_rng(17)
    for d in DS:
        gn = calib.identity(d, "bus").gamma_n
        for frac in (0.8, 0.0):
            tau = frac * d / C
            s = plane_wave(tau, 10.0, rng)
            n = coherent_noise(s.shape[1], d, rng) * 0.01
            cond = (np.abs(cm.plane_coherence(F, tau) - gn) >= 0.2) & (F >= 1000) & (F <= 6000)
            for true_db in (-5.0, 0.0, 10.0):
                X = cm.stft(s * 10 ** (true_db / 20) + n)
                v = cm.valid_frames(X.shape[-1])
                paa, pbb, pab = cm.spectra(X[:, :, v])
                Gl = cm.coherence(paa.sum(1), pbb.sum(1), pab.sum(1))[:, None]
                Gx = coherence(X, masks.CDR_PARAMS["lam"])[:, v]
                res = {}
                for tag, taus, est in (("prop1", [tau], "prop1"), ("prop2", [tau], "prop2"), ("prop3", None, "prop1")):
                    res[tag] = (np.median(db(masks.cdr_estimate(Gl, gn, taus, estimator=est)[cond, 0])),
                                np.median(db(masks.cdr_estimate(Gx, gn, taus, estimator=est)[cond])))
                ok = all(abs(lt - true_db) < 0.5 for lt, _ in res.values()) and abs(res["prop1"][1] - true_db) < 1.0
                check(f"d={d*1e3:.0f} mm tau={frac:.1f} d/c CDR {true_db:+.0f} dB: long-term all < 0.5 dB, "
                      f"short-time prop1 < 1 dB off", ok,
                      " | ".join(f"{k} {lt:+.1f} / {st:+.1f}" for k, (lt, st) in res.items()) + " dB (long / lam 0.9)")


def test_doa_selectivity():
    print("M4/M5 against an off-cone coherent source: broadside plane wave, partners at +-0.9 d/c")
    rng = np.random.default_rng(12)
    d = 0.04
    hyp = np.array([[0.9 * d / C], [-0.9 * d / C]])
    X = cm.stft(plane_wave(0.0, 2.0, rng))
    v = cm.valid_frames(X.shape[-1])
    Gx = coherence(X, masks.CDR_PARAMS["lam"])[:, v]
    gn = calib.identity(d, "bus").gamma_n
    band = (F >= 1500) & (F <= 3400)
    c1 = np.median(db(masks.cdr_estimate(Gx, gn, np.repeat(hyp, Gx.shape[1], 1))[band]))
    c3 = np.median(db(masks.cdr_estimate(Gx, gn, None)[band]))
    check("prop1 with partner hypotheses rejects it in 1.5-3.4 kHz", c1 < -3,
          f"median CDR {c1:.1f} dB (DOA-free prop3: {c3:.1f} dB, a coherent source passes)")
    pab = coherence(X, masks.FENCE_PARAMS["lam"])[:, v]
    i = np.argmin(np.abs(F - 2000))
    for frac in (1.0, 0.9):                   # 1.0: the DESIGN example (e = 1.47 rad, eta = 0.10)
        h = np.array([[frac * d / C], [-frac * d / C]])
        eta = np.median(masks.fence_mask(pab, np.repeat(h, pab.shape[1], 1), d)[i])
        e = 2 * np.pi * F[i] * frac * d / C
        check(f"fence eta at 2 kHz, partners at +-{frac} d/c", abs(eta - 1 / (1 + 4 * e ** 2)) < 0.005,
              f"eta {eta:.3f} (e {e:.2f} rad -> {1 / (1 + 4 * e ** 2):.3f})")


# ------------------------------------------------------------------ B.2 diffuse field
def test_b2_diffuse():
    print("B.2 diffuse field (lab.scenes.coherent_noise, 20 s); short-time gains at the DESIGN M4 start values "
          "(mu 1.3, G_min 0.15, cond 0.2), plus the tuned defaults")
    rng = np.random.default_rng(13)
    k300 = F > 300
    for d in DS:
        x = coherent_noise(20 * FS, d, rng) * 0.01
        X0 = cm.stft(calib.prefilter(x, "bus"))
        cal = calib.calibrate(X0, BUS)
        X = calib.apply(X0, cal)
        v = cm.valid_frames(X.shape[-1])
        paa, pbb, pab = cm.spectra(X[:, :, v])
        Gl = cm.coherence(paa.sum(1), pbb.sum(1), pab.sum(1))[:, None]
        hyp = np.array([[d / C], [-0.7 * d / C]])
        for cname, cc in (("model Gn", calib.identity(d, "bus")), ("S0 Gn", cal)):
            for tag, taus, est in (("prop3", None, "prop1"), ("prop1 A+B", hyp, "prop1"), ("prop2 A+B", hyp, "prop2")):
                lt = np.median(db(masks.cdr_estimate(Gl, cc.gamma_n, taus, estimator=est)[k300, 0]))
                kw = dict(DESIGN_M4, lam=0.68, estimator=est)
                G = masks.cdr_gain(X, cc, taus, median_bins=1, **kw)[:, v][k300]
                pg = db(np.mean(G ** 2))
                pg10 = db(np.mean(masks.cdr_gain(X, cc, taus, median_bins=1, **dict(kw, g_min=0.1))[:, v][k300] ** 2))
                full = db(np.mean(ns.limit_gain(masks.cdr_gain(X, cc, taus, **kw), None,
                                                masks.FALL_DB)[:, v][k300] ** 2))
                st = masks.cdr_map(X, cc, taus, **{k: kw[k] for k in ("lam", "cond", "fill_pct", "estimator")})[:, v][k300]
                tag2 = f"d={d*1e3:.0f} mm, {cname}, {tag}"
                check(f"{tag2}: long-term CDR median above 300 Hz < -10 dB", lt < -10, f"{lt:.1f} dB")
                check(f"{tag2}: short-time (lam 0.68) power-averaged gain <= -8 dB", pg <= -8,
                      f"{pg:.1f} dB at G_min 0.15 ({pg10:.1f} dB at G_min 0.1; {full:.1f} dB with the 3-bin median "
                      f"and the fall limit) [info: short-time median CDR {np.median(db(st)):+.1f} dB, "
                      f"{np.mean(st > 1)*100:.0f}% of bins > 0 dB]")
            tuned = ns.limit_gain(masks.cdr_gain(X, cc, None), None, masks.FALL_DB)[:, v]
            pt = db(np.mean(tuned[k300] ** 2))
            plo = db(np.mean(tuned[(F > 300) & (F < 1000)] ** 2))
            check(f"d={d*1e3:.0f} mm, {cname}: tuned default (DOA-free, CDR_PARAMS) power-averaged gain above "
                  f"300 Hz <= -5 dB", pt <= -5, f"{pt:.1f} dB (300-1000 Hz {plo:.1f} dB)")


# ------------------------------------------------------------------ M5 fence
def test_fence():
    print("M5 fence: band, outside rule, on-hypothesis pass, floor")
    for d, hi in ((0.025, 4000.0), (0.04, 0.8 * C / 0.08), (0.06, 0.8 * C / 0.12)):
        b = masks.fence_band(d)
        check(f"band at d={d*1e3:.0f} mm", F[b].min() >= 1000 and F[b].max() <= hi and F[b].max() > hi - 31.25,
              f"{F[b].min():.0f}-{F[b].max():.0f} Hz (edge {hi:.0f} Hz)")
    d, L = 0.04, 5
    tau = np.array([[0.8 * d / C] * L, [np.nan, -0.6 * d / C, np.nan, np.nan, -0.6 * d / C]])
    tau[0, 3] = np.nan
    pab = 0.3 * cm.plane_coherence(F, 0.8 * d / C)[:, None] * np.ones(L)
    pab[:, 4] = 0.3 * cm.plane_coherence(F, 0.0)                    # frame 4: broadside source
    eta = masks.fence_mask(pab, tau, d)
    inb = masks.fence_band(d)
    check("source on a hypothesis: eta = 1 in the band", np.allclose(eta[inb, :3], 1), "frames 0-2")
    check("frame without a finite hypothesis: 1", np.allclose(eta[:, 3], 1), "frame 3")
    e = np.abs(np.angle(np.exp(2j * np.pi * F[inb] * 0.8 * d / C)))
    want = np.maximum(1 / (1 + 4 * e ** 2), 1 / (1 + 4 * (2 * np.pi * F[inb] * 0.6 * d / C) ** 2))
    check("off-cone source: max_k 1/(1 + 4 e_k^2)", np.allclose(eta[inb, 4], want), f"band median {np.median(want):.3f}")
    check("bus: outside the band = frame's band median", np.allclose(eta[~inb, 4], np.median(want)), "ok")
    eta1 = masks.fence_mask(pab, tau, d, outside="one")
    check("meeting: outside the band = 1", np.allclose(eta1[~inb], 1), "ok")
    wrap = masks.fence_mask(np.exp(1j * np.full((len(F), 1), 3.0)), np.array([[-3.0 / (2 * np.pi * 2000)]]), d)
    i = np.argmin(np.abs(F - 2000))
    e_w = np.angle(np.exp(1j * (3.0 + 3.0 * F[i] / 2000)))
    check("phase error wraps to (-pi, pi]", np.isclose(wrap[i, 0], 1 / (1 + 4 * e_w ** 2)), f"e = {e_w:.3f} rad")


# ------------------------------------------------------------------ S1 hypotheses
def fake_track(ta, tb, n=10, d=0.04):
    trk = tracker.Track(tau=np.zeros(n), q=np.zeros(n), psi=np.zeros(n), cdr_db=np.zeros(n), level_db=np.zeros(n),
                        vad=np.zeros(n), noise_db=np.zeros(n), d_hat=d, online=False, srp_mode="fit")
    trk.tau_A_t, trk.tau_B_t = np.full(n, ta), np.full(n, tb)
    return trk


def test_hypotheses():
    print("S1 hypotheses: partner_taus (DESIGN 2.1 fallback) and talker_taus")
    d = 0.04
    cal = calib.identity(d, "bus")
    for name, ta, tb, sep in (("separable", 80e-6, -60e-6, True), ("one peak", 80e-6, 80e-6, False),
                              ("close", 80e-6, 60e-6, False), ("both broadside", 30e-6, -30e-6, False),
                              ("no partner model", np.nan, np.nan, False)):
        H = masks.partner_taus(fake_track(ta, tb), cal)
        check(name, np.isfinite(H).all() == sep and (sep or np.isnan(H).all()), f"finite {np.isfinite(H).all()}")
    H = masks.partner_taus(fake_track(80e-6, 80e-6), cal, fallback=False)
    check("fallback=False keeps a single peak", np.allclose(H, 80e-6), "ok")
    sched = [(0, calib.identity(0.06, "bus")), (5, calib.identity(0.025, "bus"))]
    H = masks.partner_taus(fake_track(45e-6, -40e-6), sched)          # 0.3 d/c = 52 us at 6 cm, 22 us at 2.5 cm
    check("per-block d_hat (online schedule)", np.isnan(H[:, :5]).all() and np.isfinite(H[:, 5:]).all(),
          "both broadside at 6 cm, separable at 2.5 cm")
    trk = fake_track(np.nan, np.nan, n=8)
    trk.cluster = np.array([-1, 0, 0, -1, 1, -1, -1, 0])
    trk.clusters = [{"id": 0, "tau": 50e-6}, {"id": 1, "tau": -20e-6}]
    want = np.array([np.nan, 50e-6, 50e-6, 50e-6, -20e-6, -20e-6, -20e-6, 50e-6])
    check("talker_taus holds the last cluster", np.allclose(masks.talker_taus(trk)[0], want, equal_nan=True),
          str(np.round(masks.talker_taus(trk)[0] * 1e6, 1)))


# ------------------------------------------------------------------ streaming
def test_streaming():
    print("streaming: chunks and online calibration blocks equal a one-shot computation")
    rng = np.random.default_rng(14)
    d = 0.04
    x, _, _, _ = partner_scene(d, (0.8 * d / C, -0.6 * d / C), 0.0, 36.0, rng)        # > CHUNK frames
    X = cm.stft(x)
    N = X.shape[-1]
    sched = [(0, calib.identity(0.04, "bus")), (1500, calib.identity(0.03, "bus")), (4200, calib.identity(0.05, "bus"))]
    taus = np.vstack([np.full(N, 0.8 * d / C), np.where(np.arange(N) % 700 < 350, -0.6 * d / C, np.nan)])
    for kw in ({}, DESIGN_M4):                # tuned defaults (no conditioning) and the DESIGN fill rule
        p = dict(masks.CDR_PARAMS, **kw)
        Gx = coherence(X, p["lam"])
        ref = np.empty((len(F), N))
        for (s, c), e in zip(sched, [1500, 4200, N]):
            cdr = masks.cdr_estimate(Gx[:, s:e], c.gamma_n, taus[:, s:e], cond=p["cond"], gn_max=p["gn_max"],
                                     fill_band=p["fill_band"], fill_pct=p["fill_pct"], estimator=p["estimator"])
            ref[:, s:e] = masks.gain_from_cdr(cdr, p["mu"], p["g_min"])
        G = masks.cdr_gain(X, sched, taus, median_bins=1, **kw)
        check(f"cdr_gain, {N} frames (chunk {cm.CHUNK}), 3 blocks, cond {p['cond']}", np.allclose(G, ref, atol=1e-10),
              f"max |diff| {np.max(np.abs(G - ref)):.1e}")
    Ge = np.concatenate([G[:1], G, G[-1:]])
    med = np.median(np.stack([Ge[:-2], Ge[1:-1], Ge[2:]]), axis=0)
    check("3-bin frequency median (edges replicated)", np.allclose(masks.cdr_gain(X, sched, taus, **DESIGN_M4), med),
          "ok")
    q = masks.FENCE_PARAMS
    pab = coherence(X, q["lam"])
    ref = np.empty((len(F), N))
    for (s, c), e in zip(sched, [1500, 4200, N]):
        ref[:, s:e] = np.maximum(q["floor"], masks.fence_mask(pab[:, s:e], taus[:, s:e], c.d_hat, gamma=q["gamma"],
                                                              band=q["band"], alias_frac=q["alias_frac"],
                                                              outside=q["outside"]))
    G = masks.fence_gain(X, sched, taus, median_bins=1)
    check("fence_gain, 3 blocks", np.allclose(G, ref, atol=1e-10), f"max |diff| {np.max(np.abs(G - ref)):.1e}")


# ------------------------------------------------------------------ wrappers
BANDS = ((0, 8000), (0, 500), (500, 1000), (1000, 4000), (4000, 8000))


def shadow(G, comp):
    """Gains (dB) of the time-varying mask G on a component's Mic A STFT in BANDS, and the output."""
    S = cm.stft(calib.prefilter(comp, "bus")[0])
    P, Q = np.abs(S) ** 2, np.abs(G * S) ** 2
    g = np.array([db(Q[(F >= lo) & (F < hi)].sum() / P[(F >= lo) & (F < hi)].sum()) for lo, hi in BANDS])
    return g, cm.istft(G * S, comp.shape[1])


def test_wrappers():
    print("m4_cdr / m5_fence, synthetic bus scene: voiced partners (anechoic, +0.8 / -0.6 d/c) in speech-shaped "
          "diffuse noise at 0 dB; tuned defaults (m4_cdr DOA-free)")
    rng = np.random.default_rng(15)
    d = 0.04
    x, tgt, noi, act = partner_scene(d, (0.8 * d / C, -0.6 * d / C), 0.0, 32.0, rng)
    for mod, name in ((m4_cdr, "m4_cdr"), (m5_fence, "m5_fence")):
        for online in (False, True):
            info = dict(BUS, online=online)
            t0 = time.perf_counter()
            y = mod.enhance(x, FS, info)
            rtf = (time.perf_counter() - t0) / 32.0
            tag = f"{name} {'online' if online else 'offline'}"
            check(f"{tag}: shape, finite, deterministic", y.shape == (x.shape[1],) and np.isfinite(y).all()
                  and np.array_equal(y, mod.enhance(x, FS, info)), f"RTF {rtf:.3f}")
            fe = tracker.front_end(x, FS, info, online=online)
            cal = fe.calib if fe.schedule is None else fe.schedule
            taus = masks.partner_taus(fe.track, cal)
            G = (masks.cdr_gain(fe.X, cal, taus if m4_cdr.BUS_HYPOTHESES else None) if mod is m4_cdr
                 else masks.fence_gain(fe.X, cal, taus))
            G = ns.limit_gain(G, None, masks.FALL_DB)
            check(f"{tag}: equals istft(limit(G) * X_A)", np.allclose(y, cm.istft(G * fe.X[0], x.shape[1])), "ok")
            gt, yt = shadow(G, tgt)
            gn_, _ = shadow(G, noi)
            ds = gt - gn_
            # M4 acts on the whole spectrum; the fence (outside rule 'one', at most -6 dB) only in 1-4 kHz
            k, band, thr = (0, "broadband", 0.5) if mod is m4_cdr else (3, "1-4 kHz", 0.2)
            check(f"{tag}: noise cut more than partners, {band} dSNR > {thr} dB and no band below -0.1 dB",
                  ds[k] > thr and ds.min() >= -0.1,
                  f"partners {gt[0]:+.1f} dB, noise {gn_[0]:+.1f} dB; dSNR all/<0.5/0.5-1/1-4/>4 kHz "
                  + "/".join(f"{v:+.1f}" for v in ds) + f" dB (tau_A/B {fe.track.tau_A*1e6:+.0f}/"
                  f"{fe.track.tau_B*1e6:+.0f} us)")
            lg = lag(yt[act], calib.prefilter(tgt, "bus")[0, act])   # the S0 high-pass is not the masks'
            check(f"{tag}: aligned (target component vs S0-prefiltered clean Mic A)", abs(lg) <= 2, f"lag {lg} samples")
            check(f"{tag}: RTF < 0.5", rtf < 0.5, f"{rtf:.3f}")
    ys = m4_cdr.enhance(x, FS, BUS, return_stages=True)
    check("return_stages", set(ys) == {"out", "pre_agc"} and np.array_equal(ys["out"], m4_cdr.enhance(x, FS, BUS)), "ok")


def test_inseparable_and_meeting():
    print("fallbacks: inseparable partners; meeting ablation")
    rng = np.random.default_rng(16)
    d = 0.04
    x, _, _, _ = partner_scene(d, (0.2 * d / C, -0.15 * d / C), 6.0, 24.0, rng)
    fe = tracker.front_end(x, FS, BUS)
    taus = masks.partner_taus(fe.track, fe.calib)
    check("broadside pair: S1 inseparable -> no hypotheses", fe.track.inseparable and np.isnan(taus).all(),
          f"{fe.track.inseparable_why}")
    check("M5 off (gain 1)", np.all(masks.fence_gain(fe.X, fe.calib, taus) == 1.0), "ok")
    check("M4 uses the DOA-free estimator", np.array_equal(masks.cdr_gain(fe.X, fe.calib, taus),
                                                           masks.cdr_gain(fe.X, fe.calib, None)), "ok")
    T = 30 * FS
    x = coherent_noise(T, d, rng) * 0.002
    seats = (0.9, -0.7, 0.1)
    t, i = int(0.5 * FS), 0
    while t < T - 4 * FS:
        n = int(3.0 * FS)
        s = burst(n, rng, 0.01)
        x[0, t:t + n] += s
        x[1, t:t + n] += frac_delay(s, seats[i % 3] * d / C)
        t, i = t + n + int(1.0 * FS), i + 1
    trk = tracker.front_end(x, FS, MEETING).track
    h = masks.talker_taus(trk)
    check("meeting: talker hypotheses exist", np.isfinite(h).mean() > 0.5,
          f"{len(trk.clusters)} clusters, finite in {np.isfinite(h).mean()*100:.0f}% of frames")
    for mod in (m4_cdr, m5_fence):
        y = mod.enhance(x, FS, MEETING)
        check(f"{mod.__name__} meeting", y.shape == (T,) and np.isfinite(y).all(),
              f"output/input {db(np.mean(y ** 2) / np.mean(x[0] ** 2)):+.1f} dB")


if __name__ == "__main__":
    t0 = time.perf_counter()
    for fn in (test_gain_law, test_conditioning_edges, test_fill, test_b4_plane_wave, test_known_cdr,
               test_doa_selectivity, test_b2_diffuse, test_fence, test_hypotheses, test_streaming, test_wrappers,
               test_inseparable_and_meeting):
        fn()
    print(f"all masks tests passed ({time.perf_counter() - t0:.1f} s)")
