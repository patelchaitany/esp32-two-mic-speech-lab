"""Unit tests for M10 WPE: methods/_core/wpe.py, methods/m10_wpe.py and methods/m10_wpe_online.py.

Run from the lab root:  .venv/bin/python -m tests.test_wpe   (about 1 min)
Dereverberation is measured as DESIGN T8 does: the known RIR is passed through the frozen filters
as a master/slave component ("probe"), and its early-to-late ratio (ELR50: direct path + 50 ms
versus the rest) and DRR (direct path ±1 ms versus the rest) are compared with the RIR's own.
Scenes: v1 hybrid RIRs (lab.scenes.hybrid_rirs, seeded) and corpus TTS speech, no noise unless
stated. Covers DESIGN Appendix B.6 (alignment) and B.8 (determinism, RTF). Each test returns the
measured numbers it asserted on. The tuning evidence on the v2 bench is in the tune:wpe report.
"""
import json
import os
import sys
import time

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from lab.dsp import stft
from lab.scenes import CORPUS, coherent_noise, hybrid_rirs, load_utt, place_array
from methods import m10_wpe, m10_wpe_online
from methods._core import calib, wpe

FS = 16000
INFO = {"scenario": "meeting", "spacing_m": 0.04}
INFO_BUS = {"scenario": "bus", "spacing_m": 0.04}
VARIANTS = {"global": dict(bounds=[]), "segment": {}, "online": dict(online=True)}


# ------------------------------------------------------------------ signals and measures
def speech(seconds, voice="samantha"):
    """Corpus TTS utterances of one voice with 0.3 s gaps, peak 0.1, exactly `seconds` long."""
    man = json.load(open(os.path.join(CORPUS, "manifest.json")))
    utts = [np.r_[load_utt(m["file"]), np.zeros(int(0.3 * FS))]
            for m in man if m["file"].startswith(f"{voice}__mtg")]
    s = np.concatenate(utts * (1 + int(seconds * FS) // sum(map(len, utts))))[: int(seconds * FS)]
    return s / np.max(np.abs(s)) * 0.1


def room(sources, rt60=0.6, spacing=0.04, seed=1, dims=(9.0, 7.0, 3.0)):
    """2-mic RIRs (2, L) and direct-path sample index for sources given as (distance m, azimuth rad)."""
    pra.random.seed(numpy=seed, libroom=seed)
    dev = [dims[0] / 2 - 0.3, dims[1] / 2 - 0.2, 0.8]
    pos = [[dev[0] + r * np.cos(a), dev[1] + r * np.sin(a), 1.2] for r, a in sources]
    return hybrid_rirs(list(dims), rt60, pos, place_array(dev, spacing, 0.3), np.random.default_rng(seed))


def conv(s, h):
    return np.stack([fftconvolve(s, h[m])[: len(s)] for m in range(2)])


def probe(h, at, T):
    """The RIR h placed at sample `at` of a silent (2, T) signal."""
    p = np.zeros((2, T))
    n = min(h.shape[1], T - at)
    p[:, at:at + n] = h[:, :n]
    return p


def ratios(r, at, d0, L):
    """(ELR50, DRR, direct energy) in dB of response r whose direct path is at sample at + d0."""
    d, split, end = at + d0, at + d0 + int(0.05 * FS), at + L + 2048
    e = lambda a, b: np.sum(r[max(a, 0):b] ** 2)
    direct = e(d - 16, d + 16)
    elr = e(at - 512, split) / e(split, end)
    drr = direct / (e(at - 512, d - 16) + e(d + 16, end))
    return 10 * np.log10(elr), 10 * np.log10(drr), 10 * np.log10(direct)


def db_ratio(a, b):
    return 10 * np.log10(np.sum(a ** 2) / np.sum(b ** 2))


def peak_lag(a, b, max_lag=64):
    """Lag (samples) of a relative to b at the cross-correlation peak."""
    lags = np.arange(-max_lag, max_lag + 1)
    xc = [np.dot(a[max(k, 0):len(a) + min(k, 0)], b[max(-k, 0):len(b) + min(-k, 0)]) for k in lags]
    return int(lags[np.argmax(np.abs(xc))])


def regressors(y, taps, delay):
    """Naive stacked regressors (D·K, N) of one bin y (D, N), order (k, d), oldest frame first."""
    D, N = y.shape
    yb = np.zeros((D * taps, N), complex)
    for t in range(N):
        for k in range(taps):
            tau = t - delay - taps + 1 + k
            if tau >= 0:
                yb[k * D:(k + 1) * D, t] = y[:, tau]
    return yb


# ------------------------------------------------------------------ tests
def test_offline_matches_equations():
    """Vectorised offline WPE equals a per-bin loop written straight from the M10 equations."""
    rng = np.random.default_rng(0)
    out = {}
    for taps, delay, its, ctx in ((10, 3, 3, 1), (3, 2, 2, 0), (4, 1, 1, 2)):
        D, F, N = 2, 5, 90
        Y = (rng.standard_normal((D, F, N)) + 1j * rng.standard_normal((D, F, N))) * np.exp(rng.standard_normal(N))
        kw = dict(taps=taps, delay=delay, iterations=its, psd_context=ctx)
        X = wpe.wpe_offline(Y, bounds=[], **kw)
        p = wpe.params(**kw)
        ref = np.empty_like(Y)
        for f in range(F):
            y = Y[:, f]
            yb = regressors(y, taps, delay)
            x = y
            for it in range(its):
                pw = np.mean(np.abs(x) ** 2, axis=0)
                lam = np.array([pw[max(0, t - ctx):t + ctx + 1].mean() for t in range(N)])
                if it == 0:
                    fl = p["floor"] * lam.mean() + 1e-30
                w = 1 / np.maximum(lam, fl)
                R = (yb * w) @ yb.conj().T
                P = (yb * w) @ y.conj().T
                R += (p["load"] * np.trace(R).real / (D * taps) + 1e-30) * np.eye(D * taps)
                G = np.linalg.solve(R, P)
                x = y - G.conj().T @ yb
            ref[:, f] = x
        err = np.abs(X - ref).max() / np.abs(ref).max()
        assert err < 1e-9, f"taps {taps} delay {delay}: relative error {err:.1e}"
        out[f"taps {taps} delay {delay} iterations {its} context {ctx}: max rel. error"] = f"{err:.1e}"
    return out


def test_segments_match_equations():
    """Segment-wise offline WPE equals per-window filters written straight from the M10 equations:
    first-iteration λ from the observation over the whole signal (±context), the floor relative to the
    bin's mean λ over the whole signal, R and P summed over each window [start − ctx_s[0], end + ctx_s[1]),
    later iterations re-weighted by the window's own output, complementary linear crossfades."""
    rng = np.random.default_rng(6)
    D, F, N, taps, delay, h = 2, 4, 400, 4, 2, 3
    Y = (rng.standard_normal((D, F, N)) + 1j * rng.standard_normal((D, F, N))) * np.exp(rng.standard_normal(N))
    edges, cl, cr = [0, 125, 250, 375, 400], 75, 25  # seg_s 1 s, ctx_s (0.6 s, 0.2 s) at 125 frames/s
    ramp = lambda t, b: np.clip((t - b + h + 0.5) / (2 * h), 0.0, 1.0)

    def psd(x):
        pw = np.mean(np.abs(x) ** 2, axis=0)
        return np.array([pw[max(0, t - 1):t + 2].mean() for t in range(x.shape[1])])

    out = {}
    for its in (1, 2):
        kw = dict(taps=taps, delay=delay, iterations=its, psd_context=1, seg_s=1.0, ctx_s=(0.6, 0.2), xfade_s=0.05)
        X = wpe.wpe_offline(Y, **kw)
        p = wpe.params(**kw)
        ref = np.zeros_like(Y)
        for f in range(F):
            y = Y[:, f]
            yb = regressors(y, taps, delay)
            lam0 = psd(y)
            fl = p["floor"] * lam0.mean() + 1e-30
            for s, e in zip(edges[:-1], edges[1:]):
                t0, t1 = max(0, s - cl), min(N, e + cr)
                lam = lam0[t0:t1]
                for it in range(its):
                    w = 1 / np.maximum(lam, fl)
                    R = (yb[:, t0:t1] * w) @ yb[:, t0:t1].conj().T
                    P = (yb[:, t0:t1] * w) @ y[:, t0:t1].conj().T
                    R += (p["load"] * np.trace(R).real / (D * taps) + 1e-30) * np.eye(D * taps)
                    G = np.linalg.solve(R, P)
                    lam = psd(y[:, t0:t1] - G.conj().T @ yb[:, t0:t1])
                t = np.arange(max(0, s - h), min(N, e + h))
                wt = (ramp(t, s) if s > 0 else 1.0) * (1.0 - ramp(t, e) if e < N else 1.0)
                ref[:, f, t] += wt * (y[:, t] - G.conj().T @ yb[:, t])
        err = np.abs(X - ref).max() / np.abs(ref).max()
        assert err < 1e-9, f"iterations {its}: relative error {err:.1e}"
        out[f"4 segments, iterations {its}: max rel. error"] = f"{err:.1e}"
    return out


def test_online_matches_weighted_ls():
    """The RLS recursion equals the exponentially weighted LS solution at every frame
    (R_t = αR_{t−1} + ỹỹᴴ/λ_t, P_t = αP_{t−1} + ỹyᴴ/λ_t, R_{−1} = rls_prior·I, frames without
    regressor energy skipped), with λ_t recomputed from the a-priori outputs (3-frame causal mean)."""
    rng = np.random.default_rng(1)
    D, F, N, taps, delay, alpha, prior = 2, 3, 150, 3, 2, 0.9, 10.0
    Y = (rng.standard_normal((D, F, N)) + 1j * rng.standard_normal((D, F, N))) * np.exp(rng.standard_normal(N))
    X = wpe.wpe_online(Y, taps=taps, delay=delay, alpha=alpha, rls_prior=prior, floor=0.0)
    worst = 0.0
    for f in range(F):
        y, x = Y[:, f], X[:, f]
        yb = regressors(y, taps, delay)
        px = np.mean(np.abs(x) ** 2, axis=0)
        R, P = prior * np.eye(D * taps, dtype=complex), np.zeros((D * taps, D), complex)
        for t in range(N):
            G = np.linalg.solve(R, P)
            pred = y[:, t] - G.conj().T @ yb[:, t]
            worst = max(worst, np.abs(pred - x[:, t]).max() / np.abs(y[:, t]).max())
            if np.vdot(yb[:, t], yb[:, t]).real > 0:
                lam = max(px[max(0, t - 2):t + 1].mean(), 1e-30)
                R = alpha * R + np.outer(yb[:, t], yb[:, t].conj()) / lam
                P = alpha * P + np.outer(yb[:, t], y[:, t].conj()) / lam
    assert worst < 1e-8, f"RLS deviates from the weighted LS solution: {worst:.1e}"
    return {"max rel. deviation of the a-priori output": f"{worst:.1e}"}


def test_dereverberation():
    """Single reverberant source, no noise: every variant raises ELR50 of the RIR probe by >= 4 / 8 dB
    and its DRR by >= 2 dB, and keeps the direct path; the master/slave split is exact (output = sum
    of filtered components). Delay 4 keeps reflections within about 32 ms (part of the 50 ms early
    image the bench scores against), so the DRR gain is smaller than the ELR50 gain."""
    out = {}
    for rt60, dist, min_gain in ((0.6, 2.5, 4.0), (0.4, 1.0, 8.0)):
        (h,), (d0,) = room([(dist, 0.6)], rt60)
        s = speech(12)
        h_early = h.copy()
        h_early[:, d0 + int(0.05 * FS):] = 0
        y, y_early = conv(s, h), conv(s, h_early)
        T, L = y.shape[1], h.shape[1]
        elr_in, drr_in, dir_in = ratios(h[0], 0, d0, L)
        out[f"RT60 {rt60} s, {dist} m: RIR ELR50 / DRR"] = f"{elr_in:.2f} / {drr_in:.2f} dB"
        for name, kw in VARIANTS.items():
            at = 6 * FS + (FS // 10 if name == "segment" else 0)  # inside one 2 s segment
            x, (pr, ce, cl) = wpe.dereverb(y, comps=[probe(h, at, T), y_early, y - y_early], **kw)
            elr, drr, direct = ratios(pr[0], at, d0, L)
            out[f"RT60 {rt60} s, {dist} m, {name}: dELR50 / dDRR / direct-path change"] = \
                f"{elr - elr_in:+.2f} / {drr - drr_in:+.2f} / {direct - dir_in:+.2f} dB"
            assert elr - elr_in >= min_gain, f"{name}, RT60 {rt60}: dELR50 {elr - elr_in:.2f} dB < {min_gain}"
            assert drr - drr_in >= 2.0, f"{name}, RT60 {rt60}: dDRR {drr - drr_in:.2f} dB"
            assert abs(direct - dir_in) <= 1.0, f"{name}: direct path changed by {direct - dir_in:.2f} dB"
            assert np.allclose(ce + cl, x, atol=1e-12 * np.abs(x).max()), f"{name}: master/slave mismatch"
    return out


def test_noise_and_silence():
    """Digital silence stays exactly zero; on noise-only input (white and pink-ish diffuse) the
    output power stays within [−1, +0.5] dB overall and +1 dB in every 0.5 s block (no blow-up),
    also after 5 s of digital silence (online wind-up check)."""
    rng = np.random.default_rng(3)
    out = {}
    z = np.zeros((2, 3 * FS))
    for name, kw in VARIANTS.items():
        assert np.abs(wpe.dereverb(z, **kw)).max() == 0.0, f"{name}: silence in, non-zero out"
    for mod in (m10_wpe, m10_wpe_online):
        assert np.abs(mod.enhance(z.astype(np.float32), FS, INFO)).max() == 0.0
    noises = {"white": rng.standard_normal((2, 10 * FS)) * 1e-3,
              "pinkish diffuse": coherent_noise(10 * FS, 0.04, rng, lambda f: 1 / np.maximum(f, 30) ** 0.6) * 1e-3}
    noises["silence then noise"] = np.concatenate([np.zeros((2, 5 * FS)), noises["white"][:, :5 * FS]], axis=1)
    for nname, x in noises.items():
        for name, kw in VARIANTS.items():
            o = wpe.dereverb(x, **kw)
            assert np.isfinite(o).all(), f"{nname}, {name}: non-finite output"
            act = np.flatnonzero(np.abs(x[0]) > 0)[0]
            blocks = [db_ratio(o[:, i:i + FS // 2], x[:, i:i + FS // 2]) for i in range(act, x.shape[1], FS // 2)]
            total = db_ratio(o[:, act:], x[:, act:])
            out[f"{nname}, {name}: overall / worst 0.5 s block"] = f"{total:+.2f} / {max(blocks):+.2f} dB"
            assert -1.0 <= total <= 0.5, f"{nname}, {name}: output/input power {total:.2f} dB"
            assert max(blocks) <= 1.0, f"{nname}, {name}: a 0.5 s block rose by {max(blocks):.2f} dB"
    return out


def test_scale_invariance():
    """WPE is scale invariant (all floors and the loading are relative): WPE(c·x) = c·WPE(x)."""
    (h,), _ = room([(2.5, 0.6)])
    x = conv(speech(4), h)
    out = {}
    for name, kw in VARIANTS.items():
        ref = wpe.dereverb(x, **kw)
        for c in (1e-3, 1e3):
            err = np.abs(wpe.dereverb(c * x, **kw) / c - ref).max() / np.abs(ref).max()
            assert err < 1e-6, f"{name}, scale {c}: relative error {err:.1e}"
            out[f"{name}, scale {c:g}: max rel. error"] = f"{err:.1e}"
    return out


def test_alignment_and_wrappers():
    """B.6: the probe's direct path stays at its input sample (|lag| <= 2) for every variant; the
    wrappers return channel A of WPE on the S0-prefiltered input (80 Hz meeting, 120 Hz bus), aligned
    with the prefiltered Mic A (lag 0)."""
    (h,), (d0,) = room([(2.5, 0.6)])
    y = conv(speech(8), h)
    T = y.shape[1]
    at = 5 * FS + FS // 10
    out = {}
    for name, kw in VARIANTS.items():
        _, (pr,) = wpe.dereverb(y, comps=[probe(h, at, T)], **kw)
        lag = int(np.argmax(np.abs(pr[0, at:at + 2 * d0]))) - int(np.argmax(np.abs(h[0, :2 * d0])))
        out[f"{name}: probe direct-path lag"] = f"{lag} samples"
        assert abs(lag) <= 2, f"{name}: direct path moved by {lag} samples"
    x = y.astype(np.float32)
    for info in (INFO, INFO_BUS):
        xh = calib.prefilter(x, info["scenario"])
        for mod, online in ((m10_wpe, False), (m10_wpe_online, True)):
            o = mod.enhance(x.copy(), FS, info)
            st = mod.enhance(x.copy(), FS, info, return_stages=True)
            assert o.shape == (T,) and np.isfinite(o).all()
            assert np.array_equal(st["out"], o) and np.array_equal(st["pre_agc"], o)
            assert np.array_equal(o, wpe.dereverb(xh, FS, online=online)[0])
            lag = peak_lag(o, xh[0])
            out[f"{mod.__name__}, {info['scenario']}: output vs prefiltered Mic A lag"] = f"{lag} samples"
            assert abs(lag) <= 2, f"{mod.__name__}, {info['scenario']}: lag {lag}"
    return out


def test_speaker_memory():
    """Variant (c): talker A, then B, then A again. Restoring A's stored (Q, G) at A's return
    beats continuing from B's filter; set_speaker stores and restores the state exactly."""
    (hA, hB), (dA, _) = room([(2.5, 0.6), (2.0, 2.6)])
    sA, sB = speech(12), speech(8, "daniel")
    x = np.concatenate([conv(sA[:8 * FS], hA), conv(sB, hB), conv(sA[8 * FS:], hA)], axis=1)
    T = x.shape[1]
    centres = np.arange(stft(np.zeros(T)).shape[-1]) * 128 - 256  # frame l is centred on 128·l − 256
    keys = np.where((centres >= 8 * FS) & (centres < 16 * FS), 1, 0)
    at = 16 * FS + FS // 2
    elr_in = ratios(hA[0], 0, dA, hA.shape[1])[0]
    res = {}
    for name, kw in (("no memory", {}), ("speaker memory", dict(keys=keys))):
        _, (pr,) = wpe.dereverb(x, online=True, comps=[probe(hA, at, T)], **kw)
        res[name] = ratios(pr[0], at, dA, hA.shape[1])[0] - elr_in
    assert res["speaker memory"] >= res["no memory"] + 2.0, res
    w = wpe.OnlineWPE(4, 2)
    rng = np.random.default_rng(2)
    frames = rng.standard_normal((40, 4, 2)) + 1j * rng.standard_normal((40, 4, 2))
    w.set_speaker(0)
    for y in frames[:20]:
        w.step(y)
    saved = w.Q.copy(), w.G.copy()
    w.set_speaker(1)
    for y in frames[20:]:
        w.step(y)
    assert not np.array_equal(w.G, saved[1])
    w.set_speaker(0)
    assert np.array_equal(w.Q, saved[0]) and np.array_equal(w.G, saved[1]), "state not restored"
    return {f"A returns (+0.5 s), {k}: dELR50": f"{v:+.2f} dB" for k, v in res.items()}


def test_odd_inputs():
    """Short and float32 inputs, degenerate segment bounds and parameter checks."""
    rng = np.random.default_rng(4)
    for T in (1, 100, 1000, 5000):
        x = (rng.standard_normal((2, T)) * 0.01).astype(np.float32)
        for mod in (m10_wpe, m10_wpe_online):
            o = mod.enhance(x, FS, INFO)
            assert o.shape == (T,) and np.isfinite(o).all(), f"{mod.__name__}, T={T}"
    Y = stft(rng.standard_normal((2, 3 * FS)))
    N = Y.shape[-1]
    ref = wpe.wpe_offline(Y, bounds=[])
    for b in ([0, N, -5, N + 9], [1, 2, N - 1], [100, 100, 101]):
        X = wpe.wpe_offline(Y, bounds=b)
        assert X.shape == Y.shape and np.isfinite(X).all()
    assert np.allclose(wpe.wpe_offline(Y, bounds=[0, N, -5, N + 9]), ref), "out-of-range bounds must be ignored"
    for bad in (dict(tap=3), dict(delay=0)):
        try:
            wpe.params(**bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"params({bad}) did not raise")
    return {"inputs of 1, 100, 1000, 5000 samples": "ok"}


def test_determinism_and_rtf():
    """B.8: identical output on a repeated call; RTF < 0.5 on 30 s (one process)."""
    (h,), _ = room([(2.5, 0.6)])
    rng = np.random.default_rng(5)
    s = conv(speech(30), h)
    x = (s + coherent_noise(s.shape[1], 0.04, rng) * np.sqrt(np.mean(s ** 2)) * 0.1).astype(np.float32)
    out = {}
    for mod in (m10_wpe, m10_wpe_online):
        t0 = time.perf_counter()
        y1 = mod.enhance(x.copy(), FS, INFO)
        rtf = (time.perf_counter() - t0) / (x.shape[1] / FS)
        y2 = mod.enhance(x.copy(), FS, INFO)
        assert np.array_equal(y1, y2), f"{mod.__name__}: non-deterministic output"
        assert rtf < 0.5, f"{mod.__name__}: RTF {rtf:.3f}"
        out[f"{mod.__name__} RTF"] = f"{rtf:.3f}"
    return out


TESTS = [test_offline_matches_equations, test_segments_match_equations, test_online_matches_weighted_ls,
         test_dereverberation, test_noise_and_silence, test_scale_invariance, test_alignment_and_wrappers,
         test_speaker_memory, test_odd_inputs, test_determinism_and_rtf]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        t0 = time.perf_counter()
        try:
            res = test() or {}
            status = "PASS"
        except AssertionError as e:
            failed, res, status = failed + 1, {"error": str(e)}, "FAIL"
        print(f"{status} {test.__name__} ({time.perf_counter() - t0:.1f} s)")
        for k, v in res.items():
            print(f"     {k}: {v}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
