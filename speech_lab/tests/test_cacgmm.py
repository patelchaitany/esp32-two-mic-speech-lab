"""Tests for M7: methods/_core/cacgmm.py and methods/m7_cacgmm.py (DESIGN.md M7; Appendix B.1, B.6, B.8).

Run from LAB: .venv/bin/python -m tests.test_cacgmm [--quick]
Synthetic signals only: TTS corpus speech (two voices taking turns, some overlap) as plane waves or in a
pyroomacoustics shoebox cabin, plus sinc-coherent speech-shaped diffuse noise at 0 dB. Oracle masks come
from the separated components. The DESIGN M7 configuration (cacgmm.DESIGN_PARAMS) keeps the original
quality checks; the tuned defaults are checked for G0-style safety (partner loss, SNR, alignment).
--quick skips the reverberant rooms and the long-file test. Plain asserts; prints the measured numbers.
"""
import sys
import time
from types import SimpleNamespace

import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter, fftconvolve, sosfiltfilt

from lab.common import C, FS, active_rms
from lab.scenes import coherent_noise, load_utt
from methods import m7_cacgmm
from methods._core import cacgmm, calib, common as cm, tracker

BUS = {"scenario": "bus", "spacing_m": 0.04}
F = cm.FREQS
LAG_SOS = butter(4, (1000.0, 4000.0), btype="band", fs=FS, output="sos")   # v2 harness delay band


def check(name, ok, msg):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


# ------------------------------------------------------------------ synthetic scenes
def frac_delay(x, tau):
    """Delay x by tau seconds (FFT phase shift, zero-padded)."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * f * tau), nfft)[:n]


def conversation(dur, rng, voices=("samantha", "daniel")):
    """Two voices taking turns (20 % of turns overlap by 0.1-0.6 s), each utterance at active RMS 1."""
    T = int(dur * FS)
    s = [np.zeros(T), np.zeros(T)]
    t, k, i = int(0.5 * FS), 0, [0, 0]
    while True:
        u = load_utt(f"{voices[k]}__bus{i[k] % 20:02d}.wav")
        i[k] += 1
        if t + len(u) > T - int(0.3 * FS):
            return s
        s[k][t:t + len(u)] += u / active_rms(u)
        t += len(u) + int((rng.uniform(-0.6, -0.1) if rng.random() < 0.2 else rng.uniform(0.2, 0.9)) * FS)
        k = 1 - k


def room_images(d, theta, rt60, sources):
    """Two-mic images of sources at 0.5 m and angles theta (deg from the axis) in an 11.5 x 2.5 x 2.3 m
    cabin (image sources, order 12). Returns (images, true TDOAs)."""
    dims, c = [11.5, 2.5, 2.3], np.array([5.0, 1.25, 1.2])
    e_abs, _ = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=12, use_rand_ism=False)
    mics = np.stack([c + [d / 2, 0, 0], c - [d / 2, 0, 0]])
    pos = [c + 0.5 * np.array([np.cos(np.radians(t)), 0.999 * np.sin(np.radians(t)), 0.05]) for t in theta]
    for p in pos:
        room.add_source(list(p))
    room.add_microphone_array(mics.T)
    room.compute_rir()
    imgs = [np.stack([fftconvolve(s, room.rir[m][k])[:len(s)] for m in range(2)]) for k, s in enumerate(sources)]
    return imgs, [(np.linalg.norm(p - mics[1]) - np.linalg.norm(p - mics[0])) / C for p in pos]


def mixture(d=0.04, kind="plane", theta=(20.0, 160.0), snr_db=0.0, dur=20.0, seed=0, rt60=0.4):
    """Partners A, B + diffuse speech-shaped noise (+ sensor noise) at snr_db; 2-ch components."""
    rng = np.random.default_rng(seed)
    sA, sB = conversation(dur, rng)
    if kind == "plane":
        taus = [d * np.cos(np.radians(t)) / C for t in theta]
        imgs = [np.stack([s, frac_delay(s, t)]) for s, t in zip((sA, sB), taus)]
    else:
        imgs, taus = room_images(d, theta, rt60, (sA, sB))
    T = len(sA)
    noise = coherent_noise(T, d, rng, shape=lambda f: 1 / (1 + (f / 800) ** 1.5) + 0.02)
    noise += 1e-2 * rng.standard_normal(noise.shape)
    act = (np.abs(sA) + np.abs(sB)) > 0
    ps = np.mean((imgs[0][0] + imgs[1][0])[act] ** 2)
    noise *= np.sqrt(ps / np.mean(noise[0][act] ** 2) / 10 ** (snr_db / 10))
    g = 0.05 / np.sqrt(ps)
    comps = {"A": imgs[0] * g, "B": imgs[1] * g, "N": noise * g}
    return comps["A"] + comps["B"] + comps["N"], comps, taus


def lag_samples(y, ref, max_lag=20, band=False):
    """Lag (samples, > 0: y late) of the cross-correlation peak of y against ref within +-max_lag;
    band=True band-passes both to 1-4 kHz first, as the v2 harness does (lab/metrics2.find_delay)."""
    if band:
        y, ref = sosfiltfilt(LAG_SOS, y), sosfiltfilt(LAG_SOS, ref)
    n = len(ref)
    cc = np.fft.irfft(np.fft.rfft(y, 2 * n) * np.conj(np.fft.rfft(ref, 2 * n)))
    lags = np.r_[0:max_lag + 1, -max_lag:0]
    return int(lags[np.argmax(np.r_[cc[:max_lag + 1], cc[-max_lag:]])])


def perm_correct(gamma, S, bins):
    """Fraction of the given bins (with enough partner frames) whose gamma_A - gamma_B correlates
    positively with the oracle A - B share; correlations below 0.05 in magnitude are not counted."""
    pa, pb, pn = (np.abs(S[k][0]) ** 2 for k in "ABN")
    tot = pa + pb + pn + 1e-30
    ok = []
    for k in np.nonzero(bins)[0]:
        m = (pa[k] + pb[k]) > pn[k]
        x, y = (gamma[0, k] - gamma[1, k])[m], ((pa[k] - pb[k]) / tot[k])[m]
        if m.sum() >= 50 and x.std() > 1e-9:
            c = np.corrcoef(x, y)[0, 1]
            if abs(c) > 0.05:
                ok.append(c > 0)
    return float(np.mean(ok)), len(ok)


def scores(sep, comps, T, cal):
    """Mask accuracy vs oracle, A/B accuracy, SNR changes (post-mask alone at Mic A; whole operator on
    the separated components) and partner gains and lags. Each component goes through the same S0
    chain as the mixture (prefilter, STFT, calib.apply with cal); lags are taken against the raw Mic-A
    component in 1-4 kHz (v2 harness; full band, the S0 high-pass's low-frequency phase lead moves the
    peak by a few samples, see lab/metrics2.find_delay)."""
    S = {k: calib.apply(cm.stft(calib.prefilter(v, "bus")), cal) for k, v in comps.items()}
    pa, pb, pn = (np.abs(S[k][0]) ** 2 for k in "ABN")
    ps = pa + pb
    hi = (F >= 1000)[:, None]
    gp = sep.directional
    pdom, ndom = (ps > 4 * pn) & hi, (pn > 4 * ps) & hi
    r = {"bal_acc": 0.5 * ((gp[pdom] > 0.5).mean() + (gp[ndom] < 0.5).mean()),
         "sep_gap": gp[pdom].mean() - gp[ndom].mean()}
    if sep.n_partner == 2:
        adom, bdom = (pa > 4 * (pb + pn)) & hi, (pb > 4 * (pa + pn)) & hi
        r["ab_acc"] = 0.5 * ((sep.gamma[0][adom] > sep.gamma[1][adom]).mean()
                             + (sep.gamma[1][bdom] > sep.gamma[0][bdom]).mean())
    snr = lambda s, n: 10 * np.log10(np.sum(s) / np.sum(n))
    irm = ps / (ps + pn + 1e-30)
    r["dsnr_mask"] = snr(sep.mask ** 2 * ps, sep.mask ** 2 * pn) - snr(ps, pn)
    r["dsnr_irm"] = snr(irm ** 2 * ps, irm ** 2 * pn) - snr(ps, pn)
    y = {k: cm.istft(cacgmm.apply(sep, S[k]), T) for k in S}
    yl = {k: cm.istft(cacgmm.apply(sep, S[k], post_mask=False), T) for k in S}
    x_in = {k: cm.istft(S[k][0], T) for k in S}
    s_in = snr((x_in["A"] + x_in["B"]) ** 2, x_in["N"] ** 2)
    r["dsnr_out"] = snr((y["A"] + y["B"]) ** 2, y["N"] ** 2) - s_in
    r["dsnr_lin"] = snr((yl["A"] + yl["B"]) ** 2, yl["N"] ** 2) - s_in
    for k in "AB":
        r[f"gain_{k}"] = snr(y[k] ** 2, x_in[k] ** 2)
        r[f"lag_{k}"] = lag_samples(y[k], comps[k][0], band=True)
        r[f"lag_{k}_full"] = lag_samples(y[k], comps[k][0])
    return r, S


# ------------------------------------------------------------------ building blocks
def test_helpers():
    print("helpers: features, eigenvalue floor, principal eigenvector, anchors, segments")
    rng = np.random.default_rng(0)
    X = cm.stft(rng.standard_normal((2, 4000)))
    ft = cacgmm.features(X)
    v = ft.valid
    check("features |z| = 1", np.allclose(np.abs(ft.c12[v]) ** 2, ft.p1[v] * (1 - ft.p1[v])) and
          (ft.p1[v] >= 0).all() and (ft.p1[v] <= 1).all(), "|z_A z_B*|^2 = p1 (1 - p1)")
    check("zero-pad frames invalid", not v[:, 0].any() and v[:, 10].all(), f"{(~v).sum()} invalid points")
    X[:, 5] = 0.0
    B0 = cacgmm.anchor_init(F, (1e-4, -1e-4), cm.diffuse_model(F, 0.04))
    for act in (None, rng.uniform(0.2, 1.0, (3, X.shape[-1]))):
        r = cacgmm.em(cacgmm.features(X), B0, np.full((3, len(F)), 1 / 3), 5, activity=act)
        check(f"bin without observations ({'static' if act is None else 'activity'} weights)",
              np.allclose(r.alpha[:, 5], 1 / 3) and np.allclose(r.gamma.sum(axis=0), 1.0),
              "keeps alpha = 1/3, posteriors sum to 1")
    A = rng.standard_normal((500, 2, 2)) + 1j * rng.standard_normal((500, 2, 2))
    B = A @ np.conj(np.swapaxes(A, 1, 2)) + np.diag([1e-6, 0.0])
    b = cacgmm._eig_floor(*cacgmm._parts(B), 1e-3)
    Bf = cacgmm._matrix(*b)
    lam = np.linalg.eigvalsh(Bf)
    check("eigenvalue floor", (lam[:, 0] >= 1e-3 * lam[:, 1] * (1 - 1e-9)).all() and
          np.allclose(np.trace(Bf, axis1=1, axis2=2).real, 2.0),
          f"min ratio {np.min(lam[:, 0] / lam[:, 1]):.2e}, trace 2")
    v_ref = np.linalg.eigh(B)[1][:, :, 1]
    for name, M in (("principal eigenvector", B), ("eigenvectors kept by the floor", Bf)):
        dev = np.max(np.abs(1 - np.abs(np.sum(np.conj(cacgmm._principal(M)) * v_ref, axis=-1))))
        check(name, dev < 1e-9, f"max |1 - |<v, v_eigh>|| {dev:.1e}")
    gn = cm.diffuse_model(F, 0.04)
    B0 = cacgmm.anchor_init(F, (100e-6, -80e-6), gn)
    tr = np.trace(B0, axis1=-2, axis2=-1).real
    al = [np.abs(np.sum(np.conj(cacgmm._principal(B0[k])) * cm.steer(F, t) / np.sqrt(2), -1)) ** 2
          for k, t in enumerate((100e-6, -80e-6))]
    check("anchor init", np.allclose(tr, 2) and min(a.min() for a in al) > 1 - 1e-9 and
          np.allclose(B0[2, :, 0, 1].real, gn * 2 / 2.1),
          "trace 2, principal eigenvector = steering, noise = Gamma_n")
    res = cacgmm.resolved_bins(F, 100e-6, -80e-6)
    check("resolved bins", np.array_equal(res, np.abs(np.cos(np.pi * F * 180e-6)) <= 0.8),
          f"{res.sum()} of {len(F)} bins (blind near 0 and {1 / 180e-6:.0f} Hz)")
    for N, L, o in ((100, 1000, 50), (2600, 1000, 100), (7777, 3000, 250)):
        segs = cacgmm._segments(N, L, o)
        w = np.zeros(N)
        for i, (s, e) in enumerate(segs):
            w[s:e] += cacgmm._ramp(N, segs, i, o)
        check(f"segments N={N}", np.allclose(w, 1) and max(e - s for s, e in segs) <= L and segs[-1][1] == N,
              f"{len(segs)} segments, crossfade weights sum to 1")


def test_em_known_mixture():
    print("EM on samples drawn from a known 3-class cACG mixture")
    rng = np.random.default_rng(1)
    f, N = np.linspace(500, 7500, 40), 3000
    taus, gn = (100e-6, -90e-6), 0.95 * cm.diffuse_model(np.linspace(500, 7500, 40), 0.04)
    alpha = np.array([0.25, 0.2, 0.55])
    Bt = cacgmm.anchor_init(f, taus, gn, load_partner=0.06)
    lab = rng.choice(3, size=(len(f), N), p=alpha)
    X = np.zeros((2, len(f), N), complex)
    for k in range(3):
        w = (rng.standard_normal((len(f), N, 2)) + 1j * rng.standard_normal((len(f), N, 2))) / np.sqrt(2)
        y = np.einsum("fij,fnj->fni", np.linalg.cholesky(Bt[k]), w) * rng.uniform(0.1, 10, (len(f), N, 1))
        X[0][lab == k], X[1][lab == k] = y[..., 0][lab == k], y[..., 1][lab == k]
    ft = cacgmm.features(X)
    bayes = cacgmm.em(ft, Bt.copy(), np.repeat(alpha[:, None], len(f), 1), 0, eig_floor=1e-9).gamma
    res = cacgmm.em(ft, cacgmm.anchor_init(f, taus, gn), np.full((3, len(f)), 1 / 3), 30)
    ll = np.diff(res.loglik)
    check("log-likelihood non-decreasing", ll.min() > -1e-9, f"{res.loglik[0]:.4f} -> {res.loglik[-1]:.4f}, "
          f"min step {ll.min():.1e}")
    a_em, a_b = np.mean(res.gamma.argmax(0) == lab), np.mean(bayes.argmax(0) == lab)
    check("accuracy vs Bayes-optimal", a_em > a_b - 0.02, f"EM {a_em:.3f}, true parameters {a_b:.3f}")
    dg = np.abs(res.gamma - bayes).mean()
    check("posteriors vs true-parameter posteriors", dg < 0.05, f"mean |diff| {dg:.3f}")
    check("class weights", np.abs(res.alpha.mean(1) - alpha).max() < 0.05, f"{np.round(res.alpha.mean(1), 3)}")
    ev = np.abs(np.sum(np.conj(cacgmm._principal(res.B[:2])) * cacgmm._principal(Bt[:2]), -1)) ** 2
    check("partner directions", ev.min() > 0.99, f"min |<v_em, v_true>|^2 {ev.min():.4f}")


def test_mvdr():
    print("Souden MVDR from posterior-weighted covariances (B.1-style distortionless check)")
    rng = np.random.default_rng(2)
    n, tau = 2000, 90e-6
    s = (rng.standard_normal((len(F), n)) + 1j * rng.standard_normal((len(F), n))) / np.sqrt(2)
    Xs = np.stack([s, s * np.exp(-2j * np.pi * F * tau)[:, None]])     # exact plane wave, B later
    gn = cm.diffuse_model(F, 0.04)
    for weak_b in (False, True):
        w = (rng.standard_normal((2, len(F), n)) + 1j * rng.standard_normal((2, len(F), n))) / np.sqrt(2)
        Xn = np.stack([w[0], gn[:, None] * w[0] + np.sqrt(1 - gn[:, None] ** 2) * w[1]])
        if weak_b:
            Xn[1] *= 0.5          # quieter noise at Mic B -> the SNR rule picks Mic B
        X = np.concatenate([Xs, Xn], axis=2)
        gamma = np.zeros((2, len(F), 2 * n))
        gamma[0, :, :n], gamma[1, :, n:] = 1.0, 1.0
        for ref in ("A", "snr"):
            H, refs = cacgmm.mvdr(X, gamma, [0], [tau], F, ref=ref)
            r = np.sum(np.conj(H[0]) * np.stack([np.ones(len(F)), np.exp(-2j * np.pi * F * tau)], -1), -1)
            k = F > 50
            check(f"response at Mic A = 1 (ref {ref}, ref mic {'AB'[refs[0]]})", np.max(np.abs(r[k] - 1)) < 1e-6,
                  f"max |r - 1| {np.max(np.abs(r[k] - 1)):.1e}")
        check("SNR rule picks the quieter-noise mic", refs[0] == int(weak_b), f"ref {'AB'[refs[0]]}")
    PhiN = cacgmm.covariance(X, gamma[1])
    PhiN += 1e-3 * 0.5 * np.trace(PhiN, axis1=1, axis2=2).real[:, None, None] * np.eye(2)
    PhiS = cacgmm.covariance(X, gamma[0])
    M = np.linalg.solve(PhiN, PhiS)
    h = M[:, :, 0] / np.trace(M, axis1=1, axis2=2)[:, None]
    H, _ = cacgmm.mvdr(X, gamma, [0], [tau], F, ref="A", load_n=1e-3)
    check("Souden formula", np.allclose(H[0], h), "h = Phi_N^-1 Phi_k e_ref / tr(Phi_N^-1 Phi_k)")
    a = np.stack([np.ones(len(F)), np.exp(-2j * np.pi * F * tau)], -1)
    worst = 0.0
    for ln in (1e-3, 0.1):
        Hs = cacgmm.steer_mvdr(X, gamma, [0], [tau], F, load_n=ln)
        worst = max(worst, np.max(np.abs(np.sum(np.conj(Hs[0]) * a, -1) - 1)))
    check("steering MVDR distortionless at Mic A", worst < 1e-9, f"max |h^H g - 1| {worst:.1e} (load 1e-3, 0.1)")


# ------------------------------------------------------------------ behaviour on mixtures
CASES = (("plane", 0.04, (20.0, 160.0)), ("room", 0.025, (15.0, 165.0)), ("room", 0.04, (20.0, 160.0)),
         ("room", 0.06, (30.0, 150.0)))
LIMITS = {"plane": dict(bal_acc=0.6, sep_gap=0.1, ab_acc=0.85, dsnr_out=5.0, dsnr_mask=3.0),
          "room": dict(bal_acc=0.54, sep_gap=0.05, ab_acc=0.7, dsnr_out=1.5, dsnr_mask=1.0)}
DESIGN = cacgmm.DESIGN_PARAMS          # S1 anchors, per-partner Souden MVDRs, max(0.3, gamma_A + gamma_B)
TUNED_MIN_DSNR = {"plane": 1.0, "room": 0.3}   # tuned post-mask with the true anchors (no beam): dSNR floor


def test_mixtures(quick=False):
    print("two partners + diffuse noise (0 dB): masks vs oracle, permutation, output SNR, alignment")
    print("  DESIGN configuration: quality thresholds apply with the true anchors; with S1's anchors, whose")
    print("  errors belong to S1, the checks are the mode, plus alignment and no SNR loss when S1 is within")
    print("  30 us of the truth. Tuned defaults (grid classes; true anchors): no partner loss beyond 3 dB")
    print("  (G0 TA), no SNR loss, alignment.")
    for i, (kind, d, theta) in enumerate(CASES[:1] if quick else CASES):
        x, comps, taus = mixture(d, kind, theta, seed=10 + i)
        T = x.shape[1]
        fe = tracker.front_end(x, FS, BUS)
        tag = f"{kind} d={d * 1e3:.0f} mm theta={theta[0]:.0f}/{theta[1]:.0f}"
        t0 = fe.calib.tau0                 # S0's blind skew; S1 and M7 work in the calibrated domain
        err = max(abs(fe.track.tau_A - taus[0] + t0), abs(fe.track.tau_B - taus[1] + t0)) * 1e6
        print(f"  {tag}: true tau {taus[0] * 1e6:+.1f}/{taus[1] * 1e6:+.1f} us, S0 skew {t0 * 1e6:.2f} us, "
              f"S1 {fe.track.tau_A * 1e6:+.1f}/{fe.track.tau_B * 1e6:+.1f} us (max error {err:.1f} us) "
              f"{fe.track.inseparable_why}")
        # geometric TDOAs in the calibrated STFT: S0 advanced X_B by its (blind) skew tau0
        true_ab = tuple(t - t0 for t in taus)
        for tuned in ("grid", "oracle"):
            sep = (cacgmm.separate(fe.X, fe.calib, fe.track) if tuned == "grid"
                   else cacgmm.separate_anchored(fe.X, true_ab, fe.calib.gamma_n))
            r, _ = scores(sep, comps, T, fe.calib)
            lo = 0.0 if tuned == "grid" else TUNED_MIN_DSNR[kind]
            ok = min(r["gain_A"], r["gain_B"]) >= -3.0 and r["dsnr_out"] >= lo and \
                max(abs(r["lag_A"]), abs(r["lag_B"])) <= 2 and sep.mask.min() >= cacgmm.PARAMS["g_min"]
            check(f"{tag} [tuned, {tuned}]", ok, f"partner gains {r['gain_A']:+.1f}/{r['gain_B']:+.1f} dB, dSNR "
                  f"{r['dsnr_out']:+.2f} dB (>= {lo}), lags {r['lag_A']}/{r['lag_B']}, mode {sep.diag.get('mode')}")
        for anchors in ("oracle", "S1"):
            if anchors == "oracle":
                ta, tb = true_ab
                sep = cacgmm.separate_anchored(fe.X, (ta, tb), fe.calib.gamma_n, **DESIGN)
            else:
                sep = cacgmm.separate(fe.X, fe.calib, fe.track, **DESIGN)
                if err <= 30.0:
                    check(f"{tag} S1 mode", sep.diag["mode"] == "3class", sep.diag["mode"])
                ta, tb = fe.track.tau_A, fe.track.tau_B
            r, S = scores(sep, comps, T, fe.calib)
            lim = LIMITS[kind]
            res = cacgmm.resolved_bins(F, ta, tb) & (F >= 200)
            p_res, n_res = perm_correct(sep.gamma, S, res)
            p_un, n_un = perm_correct(sep.gamma, S, ~res & (F >= 200))
            print("    " + anchors + ": " + ", ".join(f"{k} {v:+.3f}" if isinstance(v, float) else f"{k} {v}"
                                                      for k, v in r.items())
                  + f", perm resolved {p_res:.2f} ({n_res} bins), unresolved {p_un:.2f} ({n_un})")
            if anchors == "S1" and err > 30.0:
                print(f"    S1 anchors off by {err:.0f} us: not checked (known limitation: a partner without an "
                      f"anchor is nulled and floored, gains {r['gain_A']:+.1f}/{r['gain_B']:+.1f} dB)")
                continue
            check(f"{tag} [DESIGN, {anchors}] B.6 alignment", abs(r["lag_A"]) <= 2 and abs(r["lag_B"]) <= 2,
                  f"partner lags {r['lag_A']}/{r['lag_B']} samples in 1-4 kHz (full band {r['lag_A_full']}/"
                  f"{r['lag_B_full']}), gains {r['gain_A']:+.1f}/{r['gain_B']:+.1f} dB")
            if anchors == "S1":
                check(f"{tag} [DESIGN, S1] no SNR loss", r["dsnr_out"] >= 0.0, f"operator {r['dsnr_out']:+.2f} dB")
                continue
            ok = r["bal_acc"] >= lim["bal_acc"] and r["sep_gap"] >= lim["sep_gap"]
            check(f"{tag} [DESIGN, oracle] partner vs noise", ok,
                  f"balanced accuracy {r['bal_acc']:.3f}, posterior gap {r['sep_gap']:.3f} (>= 1 kHz)")
            check(f"{tag} [DESIGN, oracle] A/B", r["ab_acc"] >= lim["ab_acc"], f"accuracy {r['ab_acc']:.3f}")
            check(f"{tag} [DESIGN, oracle] permutation", p_res >= 0.9 and p_un >= 0.75,
                  f"resolved {p_res:.2f}, unresolved {p_un:.2f}")
            ok = r["dsnr_out"] >= lim["dsnr_out"] and r["dsnr_mask"] >= lim["dsnr_mask"]
            check(f"{tag} [DESIGN, oracle] SNR gain", ok,
                  f"operator {r['dsnr_out']:+.2f} dB (beams alone {r['dsnr_lin']:+.2f}), post-mask alone "
                  f"{r['dsnr_mask']:+.2f} dB (oracle IRM {r['dsnr_irm']:+.2f})")


def fake_track(N, ta, tb, why, rng):
    """Stand-in for an S1 Track with the fields separate() reads."""
    return SimpleNamespace(tau_A=ta, tau_B=tb, inseparable_why=why, p_A=rng.uniform(0, 1, N),
                           p_B=rng.uniform(0, 1, N))


def test_modes():
    print("anchors 's1' (S1 flags -> 3class / merged / passthrough), 'grid', 's1+grid'; beam / mask options;"
          " GSS-style activity option")
    rng = np.random.default_rng(3)
    x, comps, taus = mixture(0.04, "plane", (20.0, 160.0), dur=12.0, seed=4)
    fe = tracker.front_end(x, FS, BUS)
    N = fe.X.shape[-1]
    cases = (((), "3class", 3), (("close_tdoas",), "merged", 2), (("one_peak",), "passthrough", 2),
             (("both_broadside",), "passthrough", 2), (("close_tdoas", "both_broadside"), "passthrough", 2))
    for why, mode, K in cases:
        ta, tb = (taus[0], taus[0] - 15e-6) if "close_tdoas" in why else taus
        sep = cacgmm.separate(fe.X, fe.calib, fake_track(N, ta, tb, why, rng), **DESIGN)
        ok = sep.diag["mode"] == mode and sep.gamma.shape[0] == K and np.isfinite(sep.Y).all()
        if mode == "passthrough":
            ok &= np.array_equal(sep.Y, fe.X[0]) and (sep.mask == 1).all()
        if mode == "merged":
            ok &= abs(sep.taus[0] - (ta + tb) / 2) < 1e-12
        check(f"anchors 's1', flags {why or '()'}", ok, f"mode {sep.diag['mode']}, classes {sep.classes}")
    sep = cacgmm.separate(fe.X, fe.calib, fake_track(N, np.nan, np.nan, ("one_peak",), rng), anchors="s1")
    check("anchors 's1', no peak (NaN anchors)", sep.diag["mode"] == "passthrough", sep.diag["mode"])
    grid = np.array(cacgmm.grid_taus(fe.calib.d_hat, 3))
    check("grid_taus", np.allclose(grid, np.array([-1, 0, 1]) * fe.calib.d_hat / C) and
          cacgmm.grid_taus(fe.calib.d_hat, 3, avoid=(grid[2] - 20e-6,)) == tuple(grid[:2]),
          f"{np.round(grid * 1e6, 1)} us; a guard within 30 us of a partner anchor is dropped")
    for why, n_p in (((), 2), (("one_peak",), 0)):
        trk = fake_track(N, taus[0], taus[1], why, rng)
        g = cacgmm.separate(fe.X, fe.calib, trk)
        sg = cacgmm.separate(fe.X, fe.calib, trk, anchors="s1+grid")
        ok = g.diag["mode"] == "grid" and g.classes == ("G1", "G2", "G3", "noise") and g.n_partner == 0
        ok &= sg.n_partner == n_p and all(abs(t - a) > cacgmm.GUARD_SEP for t in sg.taus[n_p:] for a in sg.taus[:n_p])
        ok &= np.allclose(g.directional, 1 - g.noise) and not g.partner.any() and np.isfinite(sg.Y).all()
        check(f"anchors 'grid' / 's1+grid', flags {why or '()'}", ok, f"grid {g.classes}, s1+grid {sg.classes}")
    # default beam 'none': output = post-mask x calibrated Mic A, mask = clip(directional / scale, floor, 1)
    m = np.clip(g.directional / cacgmm.PARAMS["mask_scale"], cacgmm.PARAMS["g_min"], 1.0)
    check("beam 'none' and scaled post-mask", np.allclose(g.mask, m, atol=1e-6) and np.allclose(g.Y, g.mask * fe.X[0])
          and g.Y_k is None, f"mask range {g.mask.min():.2f}-{g.mask.max():.2f}")
    d = cacgmm.separate(fe.X, fe.calib, fe.track, **DESIGN)
    check("DESIGN_PARAMS", d.Y_k is not None and np.allclose(d.mask, np.maximum(0.3, d.partner), atol=1e-6),
          "per-partner MVDRs and max(0.3, gamma_A + gamma_B)")
    st = cacgmm.separate(fe.X, fe.calib, fe.track, anchors="s1", beam="steer")
    try:
        cacgmm.separate(fe.X, fe.calib, fe.track, beam="nope")
        bad = False
    except ValueError:
        bad = True
    check("beam 'steer'", st.Y_k is not None and st.Y_k.shape == (2,) + fe.X.shape[1:] and np.isfinite(st.Y).all()
          and bad, "per-partner steering MVDRs; an unknown beam raises ValueError")
    plain = cacgmm.separate(fe.X, fe.calib, fe.track, **DESIGN)
    gss = cacgmm.separate(fe.X, fe.calib, fe.track, activity="s1", **DESIGN)
    r0, _ = scores(plain, comps, x.shape[1], fe.calib)
    r1, _ = scores(gss, comps, x.shape[1], fe.calib)
    check("activity='s1' keeps the partner classes", gss.partner.mean() > 0.1 and np.isfinite(gss.Y).all(),
          f"mean partner posterior {gss.partner.mean():.3f} (static alpha {plain.partner.mean():.3f}); "
          f"A/B accuracy {r1['ab_acc']:.3f} vs {r0['ab_acc']:.3f}, operator dSNR {r1['dsnr_out']:+.2f} vs "
          f"{r0['dsnr_out']:+.2f} dB")
    try:
        cacgmm.params(nope=1)
        raise AssertionError("unknown parameter accepted")
    except KeyError:
        check("unknown parameter rejected", True, "KeyError")


def test_wrapper_and_cost(quick=False):
    print("wrapper m7_cacgmm: contract, determinism, edge cases, B.8 cost")
    x, _, _ = mixture(0.04, "plane", (25.0, 150.0), dur=32.0, seed=5)
    x32 = x.astype(np.float32)
    t0 = time.perf_counter()
    y = m7_cacgmm.enhance(x32, FS, dict(BUS))
    rtf = (time.perf_counter() - t0) / 32.0
    check("output shape / finite", y.shape == (x.shape[1],) and np.isfinite(y).all(), str(y.shape))
    check("deterministic", np.array_equal(y, m7_cacgmm.enhance(x32, FS, dict(BUS))), "two runs bit-identical")
    st = m7_cacgmm.enhance(x32, FS, dict(BUS), return_stages=True)
    check("return_stages", set(st) == {"out", "pre_agc"} and np.array_equal(st["out"], y), "out == pre_agc")
    check("B.8 RTF < 0.5 (32 s bus scene)", rtf < 0.5, f"RTF {rtf:.3f}")
    fe = tracker.front_end(x32, FS, BUS)
    t0 = time.perf_counter()
    cacgmm.separate(fe.X, fe.calib, fe.track)
    print(f"    M7 core alone: RTF {(time.perf_counter() - t0) / 32.0:.3f}")
    meet = dict(BUS, scenario="meeting")
    ym = m7_cacgmm.enhance(x32, FS, meet)
    check("meeting -> prefiltered Mic A", np.array_equal(ym, calib.prefilter(x32, "meeting")[0]), "passthrough")
    for name, z in (("silence", np.zeros((2, 3 * FS), np.float32)), ("0.3 s input", x32[:, :int(0.3 * FS)]),
                    ("empty input", x32[:, :0]), ("1 sample", x32[:, :1]), ("1 s input", x32[:, :FS])):
        out = m7_cacgmm.enhance(z, FS, dict(BUS))
        check(name, out.shape == (z.shape[1],) and np.isfinite(out).all(),
              f"max |y| {np.abs(out).max() if out.size else 0.0:.2e}")
    if quick:
        return
    x2, comps2, _ = mixture(0.04, "plane", (20.0, 160.0), dur=50.0, seed=6)
    fe2 = tracker.front_end(x2, FS, BUS)
    one = cacgmm.separate(fe2.X, fe2.calib, fe2.track)
    seg = cacgmm.separate(fe2.X, fe2.calib, fe2.track, max_seg_s=20.0)
    r1, _ = scores(one, comps2, x2.shape[1], fe2.calib)
    r2, _ = scores(seg, comps2, x2.shape[1], fe2.calib)
    ok = len(seg.diag["segments"]) == 3 and abs(r1["dsnr_out"] - r2["dsnr_out"]) < 1.5 and abs(r2["lag_A"]) <= 2
    check("segmented long file (3 x <= 20 s)", ok,
          f"operator dSNR {r2['dsnr_out']:+.2f} dB vs {r1['dsnr_out']:+.2f} dB in one segment")


def main():
    quick = "--quick" in sys.argv
    t0 = time.perf_counter()
    test_helpers()
    test_em_known_mixture()
    test_mvdr()
    test_mixtures(quick)
    test_modes()
    test_wrapper_and_cost(quick)
    print(f"all test groups passed in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
