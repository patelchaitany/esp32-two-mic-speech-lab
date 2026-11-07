"""Unit tests for M6 noise suppression: methods/_core/ns.py and methods/m6_ns.py.

Run from the lab root:  .venv/bin/python -m tests.test_ns
Seeded synthetic signals only: a speech-like generator (gliding harmonic source, moving formants,
unvoiced syllables, syllabic envelope, bursts with 5 ms onset ramps) mixed with coloured, tonal and
level-varying noise whose true PSD is known. Covers DESIGN Appendix B.6 (alignment) and B.8
(determinism, RTF). The tuned defaults (ns.PARAMS) are tested against theory and against the DESIGN
parameter set (ns.DESIGN_PARAMS), whose numbers are reported for comparison. Each test returns the
measured numbers it asserted on.
"""
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import exp1

from lab.dsp import stft, istft
from methods import m6_ns
from methods._core import calib, common, ns

FS = 16000
F_BINS = ns.bin_freqs(257)
FLOOR_TOL = 1e-12


# ------------------------------------------------------------------ signals
def speech_like(n, rng, bursts):
    """Speech-like signal, unit RMS over its active samples, zero outside bursts [(t0, t1) s]."""
    t = np.arange(n) / FS
    knots = rng.uniform(np.log(100), np.log(200), int(n / FS / 0.1) + 2)
    f0 = np.exp(np.interp(t, 0.1 * np.arange(len(knots)), knots))
    ph = 2 * np.pi * np.cumsum(f0) / FS
    src = sum(np.clip((7000 - h * f0) / 500, 0, 1) * np.sin(h * ph) for h in range(1, 71))
    tl = frame_times(stft(np.zeros(n)).shape[-1])
    fm = np.zeros((3, len(tl)))
    voiced = np.zeros(len(tl))
    env = np.zeros(n)
    for t0, t1 in bursts:
        ts = t0
        while ts < t1 - 0.05:
            d = min(rng.uniform(0.12, 0.3), t1 - ts)
            i0, i1 = int(ts * FS), int((ts + d) * FS)
            env[i0:i1] = np.sin(np.pi * np.arange(i1 - i0) / (i1 - i0)) ** 0.4 * rng.uniform(0.5, 1.0)
            sel = (tl >= ts) & (tl < ts + d)
            fm[:, sel] = np.array([rng.uniform(300, 850), rng.uniform(900, 2400), rng.uniform(2400, 3400)])[:, None]
            voiced[sel] = rng.random() < 0.8
            ts += d + rng.uniform(0.0, 0.05)
        i0, i1, r = int(t0 * FS), int(t1 * FS), int(0.005 * FS)
        env[i0:i0 + r] *= np.linspace(0, 1, r)
        env[i1 - r:i1] *= np.linspace(1, 0, r)
    f = F_BINS[:, None]
    shape = (np.exp(-0.5 * ((f - fm[0]) / 90) ** 2) + 0.6 * np.exp(-0.5 * ((f - fm[1]) / 130) ** 2)
             + 0.3 * np.exp(-0.5 * ((f - fm[2]) / 200) ** 2) + 0.02)
    hiss = 0.3 * np.exp(-0.5 * ((f - 4500) / 1200) ** 2)
    x = istft(stft(src) * shape * voiced + stft(rng.standard_normal(n)) * hiss * (1 - voiced), n) * env
    return x / np.sqrt(np.mean(x[env > 0] ** 2))


def coloured(n, rng, tilt):
    """Stationary Gaussian noise with amplitude spectrum 1/max(f, 100)^tilt, unit RMS."""
    m = 1 << int(np.ceil(np.log2(n)))
    f = np.fft.rfftfreq(m, 1 / FS)
    x = np.fft.irfft((rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.maximum(f, 100) ** tilt, m)[:n]
    return x / x.std()


def engine_brown(n, rng):
    """Tonal engine (40 +- 3 Hz fundamental, 24 harmonics) plus brown noise, unit RMS."""
    ph = 2 * np.pi * np.cumsum(40 + 3 * np.sin(2 * np.pi * 0.05 * np.arange(n) / FS)) / FS
    e = sum(np.sin(k * ph + rng.uniform(0, 6)) / k for k in range(1, 25))
    x = 0.7 * e / e.std() + coloured(n, rng, 1.0)
    return x / x.std()


NOISES = {"pink": lambda n, rng: coloured(n, rng, 0.5), "white": lambda n, rng: coloured(n, rng, 0.0),
          "engine+brown": engine_brown}


def frame_times(N):
    return common.frame_centers(N) / FS


def bursts_every(period, n_bursts, rng, dur=(1.0, 2.0), start=1.5):
    return [(start + period * i, start + period * i + rng.uniform(*dur)) for i in range(n_bursts)]


def burst_frames(tl, bursts, pre=0.016, post=0.016):
    m = np.zeros(len(tl), bool)
    for t0, t1 in bursts:
        m |= (tl >= t0 - pre) & (tl <= t1 + post)
    return m


def true_psd(v):
    """Expected periodogram per bin of a stationary noise signal v (mean over frames inside v)."""
    return np.mean(np.abs(stft(v)[:, 8:-8]) ** 2, axis=1)


def err_db(noise, tru, frames, band=(100.0, 7000.0)):
    """Log error 10 log10(estimate / truth) over bins in band and the given frames."""
    sel = (F_BINS >= band[0]) & (F_BINS <= band[1])
    tru = np.broadcast_to(tru[:, None] if tru.ndim == 1 else tru, noise.shape)
    return 10 * np.log10(noise[sel][:, frames] / tru[sel][:, frames])


def lag_of(y, ref, max_lag=100):
    """Lag (samples) maximising the cross-correlation; positive when y lags ref."""
    L = 1 << int(np.ceil(np.log2(2 * len(ref))))
    xc = np.fft.irfft(np.fft.rfft(y, L) * np.conj(np.fft.rfft(ref, L)), L)
    return int(np.argmax(np.r_[xc[-max_lag:], xc[:max_lag + 1]])) - max_lag


def speech_gain_db(G, S, frames):
    return 10 * np.log10(np.sum(np.abs(G[:, frames] * S[:, frames]) ** 2) / np.sum(np.abs(S[:, frames]) ** 2))


def fixed_point_db(xi_h1_db):
    """Fixed point of the mean MMSE-SPP recursion on stationary Gaussian noise, in dB re the true PSD:
    s solves s (1 - E[P(e/s)]) = E[(1 - P(e/s)) e] with e ~ Exp(1) (stagnation guard inactive)."""
    xi = 10 ** (xi_h1_db / 10)
    c0, c1 = 1 + xi, xi / (1 + xi)
    P = lambda g: 1 / (1 + c0 * np.exp(-c1 * g))

    def f(s):
        ep = quad(lambda e: P(e / s) * np.exp(-e), 0, 60)[0]
        e1 = quad(lambda e: (1 - P(e / s)) * e * np.exp(-e), 0, 60)[0]
        return s * (1 - ep) - e1

    return 10 * np.log10(brentq(f, 0.2, 2.0))


# ------------------------------------------------------------------ tests
def test_formulas():
    pb, pm = ns.params("bus"), ns.params("meeting")
    assert (pb["floor_db"], pb["gate"], pm["floor_db"], pm["gate"]) == (-15.0, "hold", -12.0, "none")
    assert (pb["xi_h1_db"], pb["alpha"], pb["alpha_dd"]) == (5.0, 0.97, 0.7), "tuned defaults"
    assert ns.params("bus", **ns.DESIGN_PARAMS)["xi_h1_db"] == 15.0
    try:
        ns.params("bus", no_such_param=1)
        raise AssertionError("unknown parameter accepted")
    except TypeError:
        pass
    # SPP: gamma = 0 gives 1 / (2 + xi_H1); P = 0.5 at gamma = ln(1 + xi)(1 + xi) / xi.
    out = {}
    for xdb in (15.0, pb["xi_h1_db"]):
        xi = 10 ** (xdb / 10)
        P = np.array([[0.0], [np.log1p(xi) * (1 + xi) / xi], [1.0]])
        noise, spp = ns.track_noise(P, np.ones(3), xi_h1_db=xdb)
        assert abs(spp[0, 0] - 1 / (2 + xi)) < 1e-12 and abs(spp[1, 0] - 0.5) < 1e-12
        a, p2 = pb["alpha"], spp[2, 0]
        assert abs(noise[2, 0] - (a + (1 - a) * ((1 - p2) * 1.0 + p2))) < 1e-12, "noise recursion"
        out[f"spp(gamma=0), xi_H1 {xdb:g} dB"] = f"{spp[0, 0]:.4f}"
    # LSA gain, first frame (decision-directed start Gr^2 gamma = 1): exact values and properties.
    gam = np.array([0.05, 0.5, 1.0, 4.0, 30.0, 1000.0])
    G = ns.lsa_gain(gam[:, None], np.ones((len(gam), 1)), floor_db=None, fall_db=None)[:, 0]
    a = pb["alpha_dd"]
    xdd = a + (1 - a) * np.maximum(gam - 1, 0)
    raw = xdd / (1 + xdd) * np.exp(0.5 * exp1(xdd * gam / (1 + xdd)))
    assert np.allclose(G, np.minimum(raw, 1.0), rtol=1e-12)
    assert np.all(raw >= xdd / (1 + xdd)), "LSA gain must not be below the Wiener gain"
    assert abs(G[-1] - xdd[-1] / (1 + xdd[-1])) < 1e-6, "LSA -> Wiener at high SNR"
    assert raw[0] > 1 and G[0] == 1.0, "LSA > 1 at low gamma is clipped"
    # Hold runs are cut to their first n_max frames.
    h = np.array([[0, 1, 1, 1, 0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]], bool)
    want = np.array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]], bool)
    assert np.array_equal(ns.cap_runs(h, 2), want), "cap_runs"
    return out


def test_gain_limits():
    rng = np.random.default_rng(10)
    F, N = 257, 400
    P = rng.exponential(1.0, (F, N)) * np.repeat(rng.choice([1.0, 3.0, 30.0, 300.0], N // 20), 20)[None, :]
    P[:, 190:200] = rng.exponential(1.0, (F, 10))      # noise-level frames ...
    P[:, 200] = 1e4                                     # ... then a strong onset
    out = {}
    for floor in (-15.0, -12.0):
        G = ns.lsa_gain(P, np.ones((F, N)), floor_db=floor)
        steps = 20 * np.log10(G[:, 1:] / G[:, :-1])
        assert G.min() >= 10 ** (floor / 20) - FLOOR_TOL and G.max() <= 1.0
        assert steps.min() >= -4.8 - 1e-9, f"fall limit violated: {steps.min():.2f} dB"
        assert G[:, 200].min() > 0.95, "rise must be instant by default"
        out[f"min G ({floor:g})"] = f"{20 * np.log10(G.min()):.2f} dB"
    G3 = ns.lsa_gain(P, np.ones((F, N)), rise_db=3.0)
    assert (20 * np.log10(G3[:, 1:] / G3[:, :-1])).max() <= 3.0 + 1e-9, "rise limit"
    out["max fall"] = f"{steps.min():.2f} dB/hop"
    # Raw gain for stacks (no floor, no limits) and limit_gain applied once afterwards.
    raw = ns.lsa_gain(P, np.ones((F, N)), floor_db=None, fall_db=None)
    assert raw.max() <= 1.0 and raw.min() < 10 ** (-15 / 20), "raw LSA gain must not be floored"
    assert np.array_equal(ns.limit_gain(raw, -15.0, 4.8), ns.lsa_gain(P, np.ones((F, N))))
    assert np.array_equal(ns.limit_gain(raw, None, None), raw)
    out["raw LSA min"] = f"{20 * np.log10(raw.min()):.1f} dB"
    return out


def test_stationary_tracking():
    """Noise-only bias near the theoretical fixed point of the recursion; no speech leakage into the
    estimate with the tuned set (the DESIGN set over-estimates in strongly coloured noise)."""
    rng = np.random.default_rng(1)
    n = 30 * FS
    bursts = bursts_every(3.0, 9, rng)
    s = speech_like(n, rng, bursts)
    tl = frame_times(stft(np.zeros(n)).shape[-1])
    sp = burst_frames(tl, bursts)
    late = tl > 0.3
    fp = {k: fixed_point_db(v) for k, v in (("tuned", ns.PARAMS["xi_h1_db"]), ("design", 15.0))}
    out = {"theory (fixed point, Gaussian noise)": f"tuned {fp['tuned']:+.2f} dB, DESIGN {fp['design']:+.2f} dB"}
    for name, gen in NOISES.items():
        v = gen(n, rng)
        for snr in (0, 5):
            vv = v * 10 ** (-snr / 20)
            Y = stft(s + vv)
            tru = true_psd(vv)
            row = []
            for tag, kw in (("tuned", {}), ("design", ns.DESIGN_PARAMS)):
                r = ns.suppress(Y, "bus", **kw)
                e_n, e_s = err_db(r["noise"], tru, late & ~sp), err_db(r["noise"], tru, late & sp)
                if tag == "tuned":
                    assert abs(np.median(e_n) - fp[tag]) <= 0.8, f"{name} {snr}: bias {np.median(e_n):.2f} dB"
                    assert np.mean(np.abs(e_n)) < 2.5, f"{name} {snr}: noise-only mean |err| {np.mean(np.abs(e_n)):.2f}"
                    assert np.median(e_s) <= np.median(e_n) + 1.0, f"{name} {snr}: speech leaks into the estimate"
                row.append(f"{np.median(e_n):+.2f}/{np.mean(np.abs(e_n)):.2f}/{np.median(e_s):+.2f}")
            out[f"{name}{snr:+d}"] = f"tuned {row[0]}; DESIGN {row[1]}"
    out["(noise-only median / mean|err| / speech-frame median, dB)"] = ""
    # Online initialisation (100 ms look-ahead) converges to the same estimate within 1 s.
    r_on = ns.suppress(stft(s + vv), "bus", init_s=0.1)
    e_on = err_db(r_on["noise"], tru, (tl > 1.0) & ~sp)
    assert abs(np.median(e_on) - fp["tuned"]) <= 0.8, f"online init: noise-only bias {np.median(e_on):.2f} dB"
    out["online init (init_s=0.1), noise-only median after 1 s"] = f"{np.median(e_on):+.2f} dB"
    return out


def _settle_time(e, tl, t_step, bias, tol=3.0, hold_s=0.5):
    """Seconds after t_step until the per-frame median error stays within tol of the bias for hold_s."""
    i0 = int(np.searchsorted(tl, t_step))
    ok = np.abs(e[i0:] - bias) <= tol
    w = int(round(hold_s * FS / 128))
    for i in range(len(ok) - w):
        if ok[i:i + w].all():
            return tl[i0 + i] - t_step
    return np.inf


def test_nonstationary_tracking():
    """Tracking speed: the tuned tracker trades speed for no speech leakage (+10 dB rise in about 3 s
    instead of 0.4 s); a fall is followed within about 1 s. Limits are measured values plus margin."""
    rng = np.random.default_rng(4)
    n = 30 * FS
    bursts = bursts_every(2.5, 11, rng, dur=(0.8, 1.4))
    s = speech_like(n, rng, bursts)
    v = coloured(n, rng, 0.5) * 10 ** (-5 / 20)                     # +5 dB SNR
    t = np.arange(n) / FS
    tl = frame_times(stft(np.zeros(n)).shape[-1])
    tru0 = true_psd(v)[:, None]
    sel = (F_BINS >= 100) & (F_BINS <= 7000)
    out = {}
    step = lambda tt: np.where((tt >= 10) & (tt < 20), 10 ** (10 / 20), 1.0)
    Y = stft(s + v * step(t))
    tru = tru0 * step(tl)[None, :] ** 2
    cases = (("bus", {}, "tuned", 4.0, 1.5), ("meeting", {}, "tuned", 4.0, 1.5),
             ("meeting", {"gate": "pause"}, "tuned", 8.0, 1.5), ("bus", ns.DESIGN_PARAMS, "DESIGN", 1.0, 0.5),
             ("meeting", dict(ns.DESIGN_PARAMS, gate="pause"), "DESIGN", 2.5, 1.0))
    for scen, kw, tag, up_max, down_max in cases:
        bias = fixed_point_db(ns.params(scen, **kw)["xi_h1_db"])
        e = np.median(10 * np.log10(ns.suppress(Y, scen, **kw)["noise"][sel] / tru[sel]), axis=0)
        up, down = _settle_time(e, tl, 10.0, bias), _settle_time(e, tl, 20.0, bias)
        name = f"{scen} {tag}" + (" pause gate" if kw.get("gate") == "pause" else "")
        assert up <= up_max and down <= down_max, f"{name}: +10 dB step {up:.2f} s, -10 dB step {down:.2f} s"
        out[f"{name} step +10/-10 dB"] = f"{up:.2f}/{down:.2f} s"
    am = lambda tt: 10 ** (6 / 20 * np.sin(2 * np.pi * 0.2 * tt))  # +-6 dB at 0.2 Hz
    for tag, kw, lim in (("tuned", {}, 4.5), ("DESIGN", ns.DESIGN_PARAMS, 2.0)):
        r = ns.suppress(stft(s + v * am(t)), "bus", **kw)
        e = np.median(10 * np.log10(r["noise"][sel] / (tru0 * am(tl)[None, :] ** 2)[sel]), axis=0)[tl > 2]
        assert np.mean(np.abs(e)) < lim, f"{tag} AM tracking mean |err| {np.mean(np.abs(e)):.2f} dB"
        out[f"bus {tag} AM +-6 dB @0.2 Hz mean|err| / mean err"] = f"{np.mean(np.abs(e)):.2f} / {np.mean(e):+.2f} dB"
    return out


def test_onsets():
    """The tracker must not learn onsets as noise, nor suppress them more than the same LSA gain
    driven by the true noise PSD (the remaining onset loss is the decision-directed lag at low
    local SNR; instant gain rise is checked in test_gain_limits)."""
    rng = np.random.default_rng(2)
    n = 30 * FS
    bursts = bursts_every(3.0, 9, rng)
    s = speech_like(n, rng, bursts)
    S = stft(s)
    tl = frame_times(S.shape[-1])
    band = (F_BINS >= 500) & (F_BINS <= 4000)
    on = burst_frames(tl, [(t0, t0 + 0.048) for t0, _ in bursts], pre=0.0, post=0.0)
    steady = burst_frames(tl, [(t0 + 0.2, t1 - 0.1) for t0, t1 in bursts], pre=0.0, post=0.0)
    out = {}
    for name in ("pink", "engine+brown"):
        v = NOISES[name](n, rng)
        for snr in (5, 10):
            vv = v * 10 ** (-snr / 20)
            Y = stft(s + vv)
            r = ns.suppress(Y, "bus")
            G_true = ns.lsa_gain(np.abs(Y) ** 2, np.repeat(true_psd(vv)[:, None], Y.shape[1], axis=1))
            rise, diff = [], []
            for t0, _ in bursts:
                i0, i1 = np.searchsorted(tl, t0 - 0.01), np.searchsorted(tl, t0 + 0.05)
                rise.append(np.median(10 * np.log10(r["noise"][band, i1] / r["noise"][band, i0])))
                one = (tl >= t0) & (tl < t0 + 0.048)
                diff.append(speech_gain_db(r["G"], S, one) - speech_gain_db(G_true, S, one))
            assert max(rise) <= 1.0, f"{name} {snr}: noise estimate rose {max(rise):.2f} dB at an onset"
            assert min(diff) >= -1.0 and np.mean(diff) >= -0.5, \
                f"{name} {snr}: onset gain vs true-noise gain worst {min(diff):.2f}, mean {np.mean(diff):.2f} dB"
            d = ns.suppress(Y, "bus", **ns.DESIGN_PARAMS)["G"]
            out[f"{name}{snr:+d}"] = (
                f"noise rise {max(rise):+.2f} dB; onset re true-noise worst {min(diff):+.2f} dB;"
                f" onset/steady speech gain {speech_gain_db(r['G'], S, on):+.1f}/{speech_gain_db(r['G'], S, steady):+.1f}"
                f" dB (true noise {speech_gain_db(G_true, S, on):+.1f}/{speech_gain_db(G_true, S, steady):+.1f};"
                f" DESIGN set {speech_gain_db(d, S, on):+.1f}/{speech_gain_db(d, S, steady):+.1f})")
    return out


def test_hold_gate():
    """Hold-gate mechanics with an oracle cue, shown with the DESIGN tracker (which leaks speech into
    the estimate without the gate) and checked for the tuned defaults."""
    rng = np.random.default_rng(3)
    n = 30 * FS
    bursts = bursts_every(3.0, 9, rng)
    s = speech_like(n, rng, bursts)
    vv = engine_brown(n, rng) * 10 ** (-10 / 20)
    Y = stft(s + vv)
    tl = frame_times(Y.shape[-1])
    sp = burst_frames(tl, bursts)
    cue = sp.astype(float)
    hi, lo = F_BINS >= 500, F_BINS < 500
    held = np.nonzero(sp & (np.arange(len(tl)) >= 1))[0]
    tru = true_psd(vv)
    late = tl > 0.3
    out = {}
    for tag, kw in (("DESIGN", ns.DESIGN_PARAMS), ("tuned", {})):
        r0 = ns.suppress(Y, "bus", **kw)
        r1 = ns.suppress(Y, "bus", target_prob=cue, **kw)
        assert np.array_equal(r1["noise"][hi][:, held], r1["noise"][hi][:, held - 1]), "held bins must be frozen"
        moving = np.mean(np.any(r1["noise"][lo][:, held] != r1["noise"][lo][:, held - 1], axis=0))
        assert moving > 0.9, f"bins below 500 Hz must keep updating ({moving:.2f})"
        e_sp1 = np.median(err_db(r1["noise"], tru, late & sp, (500, 7000)))
        e_no1 = np.median(err_db(r1["noise"], tru, late & ~sp, (500, 7000)))
        e_sp0 = np.median(err_db(r0["noise"], tru, late & sp, (2000, 7000)))
        e_sp1h = np.median(err_db(r1["noise"], tru, late & sp, (2000, 7000)))
        assert e_sp1 <= e_no1 + 0.5, f"{tag}: gated speech-frame error {e_sp1:.2f} vs noise-frame {e_no1:.2f} dB"
        if tag == "DESIGN":
            assert e_sp0 >= e_sp1h + 1.5, f"plain DESIGN tracker should over-estimate in speech ({e_sp0:.2f} dB)"
        per_bin = ns.suppress(Y, "bus", target_prob=np.repeat(cue[None, :], len(F_BINS), axis=0), **kw)
        assert np.array_equal(per_bin["G"], r1["G"]), "per-bin cue must match the broadcast frame cue"
        assert np.array_equal(ns.suppress(Y, "bus", target_prob=cue, gate="none", **kw)["G"], r0["G"])
        out[f"{tag}: speech-frame err >= 2 kHz plain/gated"] = f"{e_sp0:+.2f}/{e_sp1h:+.2f} dB"
        out[f"{tag}: noise-frame err gated; low bins updating"] = f"{e_no1:+.2f} dB; {moving:.2f}"
    return out


def test_stuck_cue():
    """A cue stuck at 'target present' must not freeze the estimate: holds longer than hold_max_s are
    released. Noise rises 10 dB at 10 s; without the cap the estimate stays 10 dB low."""
    rng = np.random.default_rng(8)
    n = 30 * FS
    s = speech_like(n, rng, bursts_every(3.0, 9, rng))
    v = coloured(n, rng, 0.5) * 10 ** (-10 / 20)
    t = np.arange(n) / FS
    Y = stft(s + v * np.where(t >= 10, 10 ** (10 / 20), 1.0))
    tl = frame_times(Y.shape[1])
    tru = true_psd(v)[:, None] * np.where(tl >= 10, 10.0, 1.0)[None, :]
    sel = (F_BINS >= 500) & (F_BINS <= 7000)
    late = tl >= 16
    out = {}
    for scen, gate in (("bus", "hold"), ("meeting", "pause")):
        for cap in (None, ns.PARAMS["hold_max_s"]):
            r = ns.suppress(Y, scen, target_prob=np.ones(Y.shape[1]), gate=gate, hold_max_s=cap)
            e = np.median(10 * np.log10(r["noise"][sel][:, late] / tru[sel][:, late]))
            if cap is None:
                assert e < -8.0, f"{scen}: stuck cue without a cap should freeze the estimate ({e:.2f} dB)"
            else:
                assert abs(e - fixed_point_db(ns.PARAMS["xi_h1_db"])) <= 1.0, f"{scen}: capped hold error {e:.2f} dB"
            out[f"{scen} ({gate}) stuck cue, hold_max_s={cap}: error 6-20 s after the rise"] = f"{e:+.2f} dB"
    return out


def test_pause_gate():
    """DESIGN meeting rule (gate='pause', optional): internal VAD calibrated independently of the
    tracker's xi_H1, update resumes 300 ms into a pause, no pumping between speech and pauses."""
    rng = np.random.default_rng(5)
    n = 30 * FS
    bursts = bursts_every(3.0, 9, rng, dur=(1.0, 1.8))
    s = speech_like(n, rng, bursts)
    vv = coloured(n, rng, 0.5) * 10 ** (-10 / 20)
    Y = stft(s + vv)
    tl = frame_times(Y.shape[-1])
    sp = burst_frames(tl, bursts, pre=0.0, post=0.0)
    in_talk = burst_frames(tl, bursts, pre=0.0, post=0.25)
    gap = np.zeros(len(tl), bool)
    for (_, t1), (t0n, _) in zip(bursts, bursts[1:] + [(tl[-1], None)]):
        gap |= (tl >= t1 + 0.5) & (tl <= t0n - 0.05)
    tru = true_psd(vv)
    out = {}
    for tag, kw in (("tuned", {}), ("DESIGN", ns.DESIGN_PARAMS)):
        r = ns.suppress(Y, "meeting", gate="pause", **kw)
        held = r["hold"][0]
        assert np.array_equal(r["hold"].all(axis=0), r["hold"].any(axis=0)), "pause gate holds whole frames"
        h_talk, u_gap = held[in_talk].mean(), (~held[gap]).mean()
        # Misses are weak burst onsets/endings (the SPP VAD needs up to ~150 ms on them).
        assert h_talk >= 0.93 and u_gap >= 0.9, f"{tag}: internal VAD gate: held in talk {h_talk:.2f}, open in gaps {u_gap:.2f}"
        e_sp = np.median(err_db(r["noise"], tru, sp & (tl > 0.3)))
        e_no = np.median(err_db(r["noise"], tru, ~sp & (tl > 0.3)))
        assert e_sp <= e_no + 0.5, f"{tag}: noise floor pumping: speech {e_sp:.2f} vs pause {e_no:.2f} dB"
        out[f"{tag}: held in talk / open in gaps"] = f"{h_talk:.3f} / {u_gap:.3f}"
        out[f"{tag}: err speech/pause"] = f"{e_sp:+.2f}/{e_no:+.2f} dB"
    r = ns.suppress(Y, "meeting")
    assert not r["hold"][:, 4:].any(), "meeting default is the plain tracker"
    e_sp, e_no = (np.median(err_db(r["noise"], tru, m & (tl > 0.3))) for m in (sp, ~sp))
    assert e_sp <= e_no + 0.5, f"plain tracker pumping: speech {e_sp:.2f} vs pause {e_no:.2f} dB"
    out["meeting default (no gate): err speech/pause"] = f"{e_sp:+.2f}/{e_no:+.2f} dB"
    re = ns.suppress(Y, "meeting", gate="pause", target_prob=sp.astype(float))
    expect = ~ns.pause_open(sp, 38)
    expect[:4] = True
    assert np.array_equal(re["hold"][0], expect), "external VAD: update must resume 300 ms into a pause"
    return out


def test_wrapper_alignment():
    """B.6: output aligned with Mic A (|lag| <= 2 samples); transparent at high SNR."""
    rng = np.random.default_rng(6)
    n = 12 * FS
    bursts = bursts_every(3.0, 3, rng)
    s = speech_like(n, rng, bursts) * 0.01
    act = burst_frames(np.arange(n) / FS, bursts, pre=0.0, post=0.0)
    w = coloured(n, rng, 0.0) * 0.01
    out = {}
    for scen in ("bus", "meeting"):
        for snr in (30, 0):
            xa = s + w * 10 ** (-snr / 20)
            x = np.stack([xa, np.roll(xa, 1)]).astype(np.float32)
            y = m6_ns.enhance(x, FS, {"scenario": scen, "spacing_m": 0.04})
            assert y.shape == (n,) and np.all(np.isfinite(y))
            ref = calib.prefilter(x[0].astype(np.float64), scen)
            lag_ref, lag_s = lag_of(y, ref), lag_of(y, s)
            assert lag_ref == 0 and abs(lag_s) <= 2, f"{scen} {snr} dB: lag {lag_ref} / {lag_s} samples"
            if snr == 30:
                g = 10 * np.log10(np.sum(y[act] ** 2) / np.sum(ref[act] ** 2))
                assert g >= -1.0, f"{scen}: speech loss {g:.2f} dB at +30 dB SNR"
                out[f"{scen} speech gain @+30 dB"] = f"{g:+.2f} dB"
            out[f"{scen} lag @{snr:+d} dB"] = f"{lag_ref}/{lag_s}"
        st = m6_ns.enhance(x, FS, {"scenario": scen, "spacing_m": 0.04}, return_stages=True)
        assert np.array_equal(st["out"], y) and np.array_equal(st["pre_agc"], y)
    return out


def test_determinism_and_rtf():
    """B.8: same input -> same output; real-time factor < 0.5."""
    rng = np.random.default_rng(7)
    n = 30 * FS
    s = speech_like(n, rng, bursts_every(3.0, 9, rng)) * 0.01
    x = np.stack([s + 0.01 * engine_brown(n, rng), s + 0.01 * engine_brown(n, rng)]).astype(np.float32)
    out = {}
    for scen in ("bus", "meeting"):
        t0 = time.perf_counter()
        y1 = m6_ns.enhance(x, FS, {"scenario": scen, "spacing_m": 0.04})
        rtf = (time.perf_counter() - t0) / (n / FS)
        y2 = m6_ns.enhance(x.copy(), FS, {"scenario": scen, "spacing_m": 0.04})
        assert np.array_equal(y1, y2), f"{scen}: non-deterministic output"
        assert rtf < 0.5, f"{scen}: RTF {rtf:.3f}"
        out[f"{scen} RTF"] = f"{rtf:.4f}"
    return out


TESTS = [test_formulas, test_gain_limits, test_stationary_tracking, test_nonstationary_tracking,
         test_onsets, test_hold_gate, test_stuck_cue, test_pause_gate, test_wrapper_alignment,
         test_determinism_and_rtf]

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
            print(f"     {k}: {v}" if v != "" else f"     {k}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
