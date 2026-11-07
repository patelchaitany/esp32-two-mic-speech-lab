"""Unit tests for M11 leveling: methods/_core/agc.py and methods/m11_level.py.

Run from the lab root:  .venv/bin/python -m tests.test_agc
Seeded synthetic signals only: harmonic 'voices' with syllabic envelopes at set band levels, white or
level-stepped noise, and 2-mic plane-wave scenes in diffuse noise for the wrapper. The AGC tests use
oracle talker keys (what S1 clusters provide) and the energy VAD of common.speech_presence, so they test
M11 itself; the wrapper tests run the real S0/S1 front end. Levels are measured in the band of the
parameter set under test: the tuned meeting set (150-7000 Hz) unless a test runs the DESIGN-literal set
(DESIGN, 300-4000 Hz) or the bus set. Covers the task's M11 checks (12 dB spread converges to +-1.5 dB,
T5 settling with and without memory, no pumping on noise, limiter <= -1 dBTP), the bus gate, the noise
caps, priors, off-attenuation, segment keys with static offline levels, noise subtraction, automix, and
Appendix B.6 (alignment) and B.8 (determinism, RTF). Each test returns the numbers it asserted on.
"""
import math
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample

from lab.scenes import coherent_noise
from methods import m11_level
from methods._core import agc, common as cm

FS = 16000
T_DB = agc.PARAMS["target_db"]
CEIL = agc.PARAMS["ceiling_dbtp"]
BAND = agc.params("meeting")["band_hz"]            # tuned meeting level band
BUS_BAND = agc.params("bus")["band_hz"]
# DESIGN-literal meeting values (DESIGN M11 / the untuned implementation), for the mechanism tests
DESIGN = dict(band_hz=(300.0, 4000.0), p_update=0.95, min_run_s=0.12, noise_sub_db=0.0, noise_max_db=-50.0,
              boost_max_db=math.inf, off_db=-3.0, level_tau_s=1.0, rate_db_s=6.0, prime_s=0.0)


# ------------------------------------------------------------------ signals and measures
def voice(n, rng, f0):
    """Harmonic voice: f0 with a 0.7 Hz vibrato, partials up to 3.8 kHz falling as h^-0.7, syllables of
    120-300 ms with 30-120 ms gaps; unit RMS over its voiced samples."""
    t = np.arange(n) / FS
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))) / FS
    src = sum(np.sin(h * ph + rng.uniform(0, 6)) / h ** 0.7 for h in range(1, int(3800 / f0)))
    env, i = np.zeros(n), 0
    while i < n:
        e = min(n, i + int(rng.uniform(0.12, 0.3) * FS))
        env[i:e] = np.sin(np.pi * np.arange(e - i) / (e - i)) ** 0.5
        i = e + int(rng.uniform(0.03, 0.12) * FS)
    s = src * env
    return s / np.sqrt(np.mean(s[env > 0] ** 2))


def active_db(x, band=BAND):
    """Band level (dBFS) of x over its active frames (band power above 1 % of the maximum)."""
    P = agc.band_power(x, band=band)
    return float(agc.dbfs(P[P > 0.01 * P.max()].mean()))


def at_level(x, band_db, band=BAND):
    return x * 10 ** ((band_db - active_db(x, band)) / 20)


def white(n, rng, band_db, band=BAND):
    """White noise at a band level of band_db dBFS."""
    w = rng.standard_normal(n)
    return w * 10 ** ((band_db - float(agc.dbfs(np.median(agc.band_power(w, band=band))))) / 20)


def frames_in(N, a, b):
    """(N,) bool: frames whose centre lies in samples [a, b)."""
    c = cm.frame_centers(N)
    return (c >= a) & (c < b)


def turns_scene(levels, order, rng, turn_s=4.0, gap_s=0.5, noise_db=-95.0, lead_s=1.0, voices=None, band=BAND):
    """Talker order[i] speaks turn i (turn_s, gaps gap_s) at band level levels[k] dBFS, in white noise.
    Returns y, turns [(a, b, k)] in samples, oracle keys (N,) (k on frames centred in a turn, else -1)
    and the clean speech s."""
    T = int((lead_s + len(order) * (turn_s + gap_s) + 1.0) * FS)
    s = np.zeros(T)
    turns = []
    for i, k in enumerate(order):
        a = int((lead_s + i * (turn_s + gap_s)) * FS)
        b = a + int(turn_s * FS)
        v = voice(b - a, rng, 100 + 35 * k) if voices is None else voices[k][:b - a]
        s[a:b] = at_level(v, levels[k], band)
        turns.append((a, b, k))
    N = cm.n_frames(T)
    keys = np.full(N, -1)
    for a, b, k in turns:
        keys[frames_in(N, a, b)] = k
    return SimpleNamespace(y=s + white(T, rng, noise_db, band), turns=turns, keys=keys, s=s)


def turn_level(y, upd, a, b, skip_s=1.0, band=BAND):
    """Band level (dBFS) of y over the level-update frames of a turn, after its first skip_s."""
    P = agc.band_power(y, band=band)
    m = frames_in(len(P), a + int(skip_s * FS), b) & upd
    return float(agc.dbfs(P[m].mean()))


def settle_time(out, inp, a, b, win=0.4, hop=0.05, tol=3.0, final=1.0):
    """T5 (DESIGN 4.4, as lab.metrics2): seconds from the turn onset until the 400 ms output/input
    gain stays within +-tol dB of the turn's final gain (median over windows in its last second);
    windows more than 20 dB below the loudest input window are ignored."""
    w, h = int(win * FS), int(hop * FS)
    st = np.arange(0, b - a - w + 1, h)
    co, ci = np.r_[0, np.cumsum(out[a:b] ** 2)], np.r_[0, np.cumsum(inp[a:b] ** 2)]
    po, pi = co[st + w] - co[st], ci[st + w] - ci[st]
    act = pi > 0.01 * pi.max()
    g = 10 * np.log10((po + 1e-30) / (pi + 1e-30))
    fin = np.median(g[act & (st >= b - a - final * FS)])
    bad = np.flatnonzero(act & (np.abs(g - fin) > tol))
    if not len(bad):
        return 0.0
    later = np.flatnonzero(act & (np.arange(len(st)) > bad[-1]))
    return float(st[later[0]] / FS) if len(later) else (b - a) / FS


def tp16_db(y):
    """Reference true peak (dBTP): 16x FFT interpolation of the zero-padded signal."""
    y = np.r_[np.zeros(256), y, np.zeros(256)]
    return float(20 * np.log10(np.abs(resample(y, 16 * len(y))).max()))


def lag_of(y, ref, max_lag=50):
    """Lag (samples) of the cross-correlation peak of y against ref (positive: y late)."""
    n = len(y) + len(ref)
    xc = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(ref, n)), n)
    lags = np.r_[np.arange(max_lag + 1), np.arange(-max_lag, 0)]
    return int(lags[np.argmax(np.r_[xc[:max_lag + 1], xc[-max_lag:]])])


def plane_pair(s, tau):
    """2-mic plane wave: X_B = X_A exp(-j 2 pi f tau) (FFT fractional delay)."""
    n = 1 << int(np.ceil(np.log2(len(s) + 64)))
    S = np.fft.rfft(s, n)
    return np.stack([s, np.fft.irfft(S * np.exp(-2j * np.pi * np.fft.rfftfreq(n, 1 / FS) * tau), n)[:len(s)]])


LEVELS_12 = [-62.0, -58.0, -54.0, -50.0]


# ------------------------------------------------------------------ AGC with oracle keys
def test_convergence_12db():
    """4 alternating talkers spread over 12 dB, 3 turns each: from 1 s into every turn, the output
    level on the gated frames is within +-1.5 dB of the target (tuned meeting set offline and online,
    and the DESIGN-literal set)."""
    sc = turns_scene(LEVELS_12, [0, 1, 2, 3] * 3, np.random.default_rng(1))
    sc_d = turns_scene(LEVELS_12, [0, 1, 2, 3] * 3, np.random.default_rng(1), band=DESIGN["band_hz"])
    out = {}
    for name, s, kw, band in (("tuned", sc, {}, BAND), ("DESIGN", sc_d, DESIGN, DESIGN["band_hz"])):
        for online in (False, True):
            res = agc.level(s.y, keys=s.keys, online=online, **kw)
            err = np.array([turn_level(res.out, res.update, a, b, band=band) - T_DB for a, b, _ in s.turns])
            lin = np.array([turn_level(s.y, res.update, a, b, band=band) for a, b, _ in s.turns])
            assert np.all(np.abs(err) <= 1.5), f"{name} online={online}: errors {np.round(err, 2)}"
            out[f"{name} {'online' if online else 'offline'}"] = (
                f"input spread {np.ptp(lin):.1f} dB -> output {np.ptp(err):.2f} dB; max |out - T| {np.abs(err).max():.2f} dB")
    return out


def test_settling_memory():
    """T5 (DESIGN 4.4) on returning talkers 12 dB apart, alternating: with speaker memory (oracle keys)
    every returning turn settles within 0.3 s offline (0.5 s online, which waits for confirmed speech);
    with one pooled key and the online dynamics (no memory) settling takes over 0.5 s, at least 3x as
    long. (A pooled key offline gets one static gain: it 'settles' at once but levels nothing.)"""
    sc = turns_scene(LEVELS_12, [0, 3] * 6, np.random.default_rng(2))
    out, med = {}, {}
    for name, keys, online, lim in (("memory offline", sc.keys, False, 0.3), ("memory online", sc.keys, True, 0.5),
                                    ("no memory online", "pooled", True, None)):
        res = agc.level(sc.y, keys=keys, online=online)
        t5 = np.array([settle_time(res.out, sc.y, a, b) for a, b, _ in sc.turns[2:]])
        med[name] = float(np.median(t5))
        if lim is not None:
            assert t5.max() <= lim, f"{name}: T5 {np.round(t5, 2)}"
        out[name] = f"T5 median {np.median(t5):.2f} s, max {t5.max():.2f} s (10 returning turns)"
    assert med["no memory online"] >= max(0.5, 3 * med["memory online"]), med
    return out


def test_online_label_lag():
    """Online S1 labels lag a talker change by 250 ms (the new talker's first 250 ms keep the previous
    label, with a change proposal at the onset). The confirmation hold keeps those frames out of the
    previous talker's level: every stored level stays within 1 dB of the talker's own."""
    sc = turns_scene(LEVELS_12, [0, 1, 2, 3] * 3, np.random.default_rng(3))
    N, lag = len(sc.keys), round(0.25 / cm.HOP_S)
    keys, change = sc.keys.copy(), np.zeros(N, bool)
    for i, (a, b, k) in enumerate(sc.turns):
        l0 = np.flatnonzero(frames_in(N, a, b))[0]
        change[l0] = True
        if i:
            keys[l0:l0 + lag] = sc.turns[i - 1][2]
    trk = SimpleNamespace(vad=cm.speech_presence(agc.band_power(sc.y))[0], change=change, tau=None)
    res = agc.level(sc.y, trk, keys=keys, online=True)
    lv = {k: v["level_db"] for k, v in res.diag["keys"].items()}
    ref = {k: turn_level(sc.y, res.update & (keys == k), *sc.turns[k][:2], skip_s=0.0) for k in range(4)}
    err = {k: lv[k] - ref[k] for k in range(4)}
    assert all(abs(e) <= 1.0 for e in err.values()), err
    blocked = sum(int((res.update & (keys == sc.turns[i - 1][2]) & frames_in(N, a, a + int(0.25 * FS))).sum())
                  for i, (a, b, k) in enumerate(sc.turns) if i)
    assert blocked == 0, f"{blocked} mislabelled onset frames updated a level"
    return {"stored level error per talker (dB)": {k: round(e, 2) for k, e in err.items()}}


def test_no_pumping_on_noise():
    """Noise must not pump the gain. (a) 6 s noise gaps between turns with a +8.6 dB noise step for 2 s,
    one pooled key: the DESIGN gate (p >= 0.95 for 120 ms) never updates on it; the tuned gate (p >= 0.5
    for 50 ms) does, and nothing but the absolute caps then bounds a pooled AGC (reported). (b) Noise-only
    files with a slow +-3 dB modulation or a +6 dB step: no update with either gate. (c) A noise burst 8 dB
    above the floor (VAD 0.5-0.95) as its own key (as an S1 segment) or labelled as a known talker: the
    DESIGN gate moves nothing unless switch='label' on a known key; under the tuned gate the burst counts
    as speech, but boost_max_db keeps its gain within 6 dB of the room-average gain. The off-attenuation
    makes one monotone transition on stationary noise."""
    rng = np.random.default_rng(4)
    out = {}
    # (a)
    sc = turns_scene([-55.0, -55.0], [0, 1, 0], rng, gap_s=6.0, noise_db=-80.0, band=DESIGN["band_hz"])
    y = sc.y.copy()
    for _, b, _ in sc.turns[:-1]:
        y[b + 2 * FS:b + 4 * FS] += white(2 * FS, rng, -72.0, DESIGN["band_hz"])
    for name, kw in (("DESIGN gate", DESIGN), ("tuned gate", {})):
        res = agc.level(y, keys="pooled", online=True, **kw)
        steps, n_upd = [], 0
        for (_, b, _), (a2, _, _) in zip(sc.turns[:-1], sc.turns[1:]):
            gap = frames_in(len(res.agc_db), b + int(0.1 * FS), a2)
            steps.append(np.ptp(res.agc_db[gap]))
            n_upd += int(res.update[gap].sum())
        if name == "DESIGN gate":
            assert n_upd == 0 and max(steps) <= 0.05, (n_upd, steps)
        out[f"(a) {name}: AGC range within gaps"] = f"{max(steps):.2f} dB, {n_upd} update frames in the gaps"
    # (b) the energy VAD calls a rise of about 10-12 dB above its 3 s minimum speech (p 0.5-0.95), so the
    # noise variations stay below that: +-3 dB slow modulation, and a +6 dB step for 3 s
    n = 40 * FS
    mod = white(n, rng, -70.0) * 10 ** (3 * np.sin(2 * np.pi * np.arange(n) / FS / 7) / 20)
    step = white(n, rng, -70.0)
    step[12 * FS:15 * FS] *= 10 ** (6 / 20)
    for name, y in (("+-3 dB modulation", mod), ("+6 dB step", step)):
        for gate, kw in (("DESIGN", DESIGN), ("tuned", {})):
            res = agc.level(y, **kw)
            assert res.update.sum() == 0 and np.ptp(res.agc_db) == 0.0, (name, gate, res.update.sum())
        out[f"(b) noise only, {name}"] = "0 updates, AGC range 0.000 dB (both gates)"
    off = agc.level(white(20 * FS, rng, -70.0), off_db=-3.0).off_db
    assert (np.diff(off) <= 0).all() and off.min() == -3.0, "off-attenuation toggles on noise"
    # (c) talkers 0 (-60) and 1 (-48 dBFS) are known; after talker 0's last turn a 1.5 s noise burst 8 dB
    # above the floor is labelled as a new key (7) or as talker 1
    for gate, kw, band in (("DESIGN", DESIGN, DESIGN["band_hz"]), ("tuned", {}, BAND)):
        sc = turns_scene([-60.0, -48.0], [0, 1, 0], np.random.default_rng(5), gap_s=4.0, noise_db=-90.0, band=band)
        a_b = sc.turns[-1][1] + FS
        y = sc.y.copy()
        y[a_b:a_b + int(1.5 * FS)] += white(int(1.5 * FS), np.random.default_rng(55), -82.0, band)
        burst = frames_in(len(sc.keys), a_b, a_b + int(1.5 * FS))
        for label, kb, switch in (("a new key", 7, "speech"), ("a new key", 7, "label"),
                                  ("known key 1", 1, "speech"), ("known key 1", 1, "label")):
            keys = sc.keys.copy()
            keys[burst] = kb
            res = agc.level(y, keys=keys, switch=switch, **kw)
            assert (res.p[burst] >= 0.5).any()
            g0 = res.agc_db[np.flatnonzero(burst)[0] - 1]
            mv = float(np.abs(res.agc_db[burst] - g0).max())
            if gate == "DESIGN":
                assert not res.update[burst].any()
                assert (mv > 0.05) == (kb == 1 and switch == "label"), f"{label} ({switch}): moved {mv:.2f} dB"
            else:
                over = float(res.agc_db[burst].max() - res.leveler.room)
                assert over <= agc.params("meeting")["boost_max_db"] + 1e-9, (label, switch, over)
                mv = f"{mv:.2f} dB, peak {over:+.2f} dB re the room gain"
            out[f"(c) {gate} gate, burst as {label}, switch='{switch}'"] = (
                f"max gain change {mv:.2f} dB" if isinstance(mv, float) else f"max gain change {mv}")
    return out


# ------------------------------------------------------------------ bus gate, noise caps, priors, pauses
def test_bus_gate():
    """Bus: levels update only when p_A + p_B > 0.8, never on babble, even louder babble that the VAD
    marks as speech; the partner level converges to the target. With the meeting VAD gate the babble
    would pull the gain down."""
    rng = np.random.default_rng(6)
    sc = turns_scene([-45.0, -45.0], [0, 1] * 3, rng, turn_s=3.0, gap_s=2.0, noise_db=-95.0, band=BUS_BAND)
    y, N = sc.y.copy(), len(sc.keys)
    pa = np.zeros(N)
    pb = np.zeros(N)
    gaps = []
    for (a, b, k), nxt in zip(sc.turns, sc.turns[1:] + [(len(y) - FS, None, None)]):
        (pa if k == 0 else pb)[frames_in(N, a, b)] = 0.95
        m = slice(b + int(0.4 * FS), nxt[0] - int(0.4 * FS))
        y[m] += at_level(voice(m.stop - m.start, rng, 230.0), -38.0, BUS_BAND)       # bystander, 7 dB louder
        gaps.append(frames_in(N, m.start, m.stop))
    babble = np.any(gaps, axis=0)
    trk = SimpleNamespace(p_A=pa, p_B=pb, vad=cm.speech_presence(agc.band_power(y))[0], tau=None)
    assert trk.vad[babble].mean() > 0.5
    res = agc.level(y, trk, "bus", noise_db=-95.0)       # noise cap out of the way: the gate alone
    assert not (res.update & babble).any(), "bus level updated on babble"
    assert max(np.ptp(res.agc_db[m]) for m in gaps) == 0.0, "bus gain moved on babble"
    err = [turn_level(res.out, res.update, a, b, band=BUS_BAND) - T_DB for a, b, _ in sc.turns[1:]]
    assert max(map(abs, err)) <= 1.5, err
    vad_res = agc.level(y, trk, "bus", p=trk.vad, p_update=0.95, noise_db=-95.0)
    g_vad = vad_res.leveler.target(vad_res.leveler.mem[0], 99.0)
    g_bus = res.leveler.target(res.leveler.mem[0], 99.0)
    assert g_vad < g_bus - 2.0, (g_vad, g_bus)
    return {"babble frames updated": int((res.update & babble).sum()),
            "partner level error (dB)": [round(e, 2) for e in err],
            "final gain: p_A+p_B gate vs VAD gate": f"{g_bus:.1f} vs {g_vad:.1f} dB"}


def test_noise_cap():
    """Absolute cap (DESIGN, the bus default): gain <= max(-50 dBFS - noise level, 0): capped boost in
    noise; never forced below 0 dB by the cap; attenuation of loud speech still allowed (to -10 dB)."""
    out = {}
    for sp_db, nz_db, want in ((-60.0, -75.0, 25.0), (-28.0, -45.0, 0.0), (-16.0, -40.0, -10.0), (-60.0, -95.0, 34.0)):
        sc = turns_scene([sp_db], [0, 0], np.random.default_rng(7), turn_s=5.0, noise_db=nz_db, band=DESIGN["band_hz"])
        res = agc.level(sc.y, keys=sc.keys, **DESIGN)
        g = res.agc_db[frames_in(len(sc.keys), sc.turns[1][0] + 2 * FS, sc.turns[1][1])]
        lv = res.diag["keys"][0]["level_db"]
        expect = min(max(min(T_DB - lv, max(-50.0 - float(np.median(res.noise_db)), 0.0)), -10.0), 45.0)
        assert abs(g.mean() - expect) <= 0.3 and abs(expect - want) <= 2.0, (sp_db, nz_db, g.mean(), expect)
        out[f"speech {sp_db:.0f} / noise {nz_db:.0f} dBFS"] = f"gain {g.mean():+.2f} dB (rule {expect:+.2f})"
    sc = turns_scene([-60.0], [0], np.random.default_rng(7), turn_s=5.0, noise_db=-95.0, band=DESIGN["band_hz"])
    g = agc.level(sc.y, keys=sc.keys, noise_db=-70.0, **DESIGN).agc_db
    assert abs(g.max() - 20.0) <= 0.01, g.max()
    out["noise_db override -70 dBFS"] = f"max gain {g.max():.2f} dB"
    return out


def test_boost_limit():
    """Relative cap (tuned meeting default): no key gets more than boost_max_db above the room-average
    gain (offline: first-pass median over keys; online: running median of the known keys). Five talkers
    near -50 dBFS and one at -66 dBFS: the quiet one is lifted by 6 dB above the room gain, not 16; with
    the cap off it is fully leveled."""
    sc = turns_scene([-50.0, -51.0, -49.0, -50.5, -49.5, -66.0], [0, 1, 2, 3, 4, 5] * 2, np.random.default_rng(15),
                     noise_db=-100.0)
    out = {}
    for online in (False, True):
        for bmax in (agc.params("meeting")["boost_max_db"], math.inf):
            res = agc.level(sc.y, keys=sc.keys, online=online, boost_max_db=bmax)
            a, b, _ = sc.turns[-1]
            g = float(np.median(res.agc_db[frames_in(len(sc.keys), a + FS, b)]))
            ref = res.leveler.room if res.leveler.room is not None else res.leveler.ref
            if math.isfinite(bmax):
                assert abs(g - (ref + bmax)) <= 0.3, (online, g, ref)
            else:                                # gated level about 1 dB below the set active level
                assert abs(g - (T_DB + 66.0)) <= 1.5, (online, g)
            out[f"{'online' if online else 'offline'}, boost_max_db={bmax}"] = (
                f"quiet talker gain {g:.1f} dB, room gain {ref:.1f} dB")
    return out


def test_noise_subtraction():
    """noise_sub_db: a talker at 5 dB band SNR in stationary noise. On the gated frames the noisy level
    reads about 1 dB high; subtracting the tracked noise power brings the stored level within 0.5 dB of
    the clean speech level on the same frames. (The gate's own selection of louder frames is a separate
    bias, reported against the clean active level.)"""
    sc = turns_scene([-70.0], [0, 0, 0], np.random.default_rng(16), turn_s=5.0, gap_s=2.0, noise_db=-75.0)
    Ps = agc.band_power(sc.s, band=BAND)
    out = {}
    for sub in (0.0, 10.0):
        res = agc.level(sc.y, keys=sc.keys, noise_sub_db=sub)
        lv = res.diag["keys"][0]["level_db"]
        clean = float(agc.dbfs(Ps[res.update].mean()))
        out[f"noise_sub_db={sub}"] = (f"stored level {lv - clean:+.2f} dB re the clean level on the gated frames "
                                      f"({lv - active_db(sc.s):+.2f} dB re the clean active level)")
        if sub:
            assert abs(lv - clean) <= 0.5, (sub, lv - clean)
        else:
            assert lv - clean >= 0.8, (sub, lv - clean)
    return out


def test_segment_keys():
    """Offline segment keys: segment_keys() reproduces S1's speech segments (runs of one cluster label
    split at change proposals), and with the static first-pass levels every segment is leveled on its
    own: one talker alternating between two loudnesses 8 dB apart comes out within +-1 dB, while a single
    (cluster) key leaves most of the 8 dB."""
    lab = np.array([-1, 0, 0, 0, -1, -1, 2, 2, 2, 2, 0, 0, -1, 1, 1])
    ch = np.zeros(len(lab), bool)
    ch[8] = True
    keys = agc.segment_keys(SimpleNamespace(cluster=lab, change=ch))
    assert keys.tolist() == [-1, 0, 0, 0, -1, -1, 1, 1, 2, 2, 3, 3, -1, 4, 4], keys.tolist()
    sc = turns_scene([-60.0, -52.0], [0, 1] * 3, np.random.default_rng(17), gap_s=0.6)
    seg = np.where(sc.keys >= 0, 0, -1)                         # one talker (one cluster) ...
    edges = np.flatnonzero(np.diff(np.r_[-1, seg]) == 1)
    trk = SimpleNamespace(cluster=seg, change=np.isin(np.arange(len(seg)), edges), tau=None,
                          vad=cm.speech_presence(agc.band_power(sc.y))[0])
    out = {}
    for mode in ("segment", "cluster"):
        res = agc.level(sc.y, trk, keys=mode)
        lv = np.array([turn_level(res.out, res.update, a, b, skip_s=0.3) for a, b, _ in sc.turns])
        out[mode] = f"turn levels re T {np.round(lv - T_DB, 1).tolist()} dB"
        if mode == "segment":
            assert np.abs(lv - T_DB).max() <= 1.0, lv - T_DB
        else:
            assert np.ptp(lv) >= 5.0, lv
    return out


def test_tau_prior():
    """A new key starts from the median stored gain of known keys in its 10 us |tau| bin when the mic
    axis lies along the table; otherwise from the room average (offline: first pass; online: median of
    the known keys); online with nothing known, from its first confirmed level. (prime_s = 0: with the
    offline first-pass levels every key is known from the start.)"""
    levels = [-62.0, -56.0, -50.0, -60.0]
    sc = turns_scene(levels, [0, 1, 2, 3], np.random.default_rng(8))
    N = len(sc.keys)
    tau = np.full(N, 0.0)
    for (a, b, k), t in zip(sc.turns, (15e-6, 45e-6, -85e-6, -47e-6)):
        tau[frames_in(N, a, b)] = t
    trk = SimpleNamespace(vad=cm.speech_presence(agc.band_power(sc.y))[0], tau=tau, directness=np.ones(N))
    out = {}
    for axis, online, src in ((True, True, "tau_bin"), (True, False, "tau_bin"), (False, True, "room_known"),
                              (False, False, "room")):
        res = agc.level(sc.y, trk, keys=sc.keys, online=online, axis_along_table=axis, prime_s=0.0)
        d, lv = res.diag["keys"], res.leveler
        assert d[3]["prior_src"] == src, (axis, online, d[3]["prior_src"])
        if src == "tau_bin":
            want = d[1]["gain_db"]
        elif src == "room_known":
            want = float(np.median([d[k]["gain_db"] for k in range(3)]))
        else:
            want = lv.room
        assert abs(d[3]["prior_db"] - want) <= 0.3, (axis, online, d[3]["prior_db"], want)
        assert abs(d[1]["tau_us"] - 45.0) < 0.5
        out[f"axis={axis} online={online}"] = f"{src}: prior {d[3]['prior_db']:.2f} dB (expected {want:.2f})"
    res = agc.level(sc.y, trk, keys=sc.keys, online=True)
    assert res.diag["keys"][0]["prior_src"] == "first_level"
    assert agc.level(sc.y, trk, keys=sc.keys).diag["keys"][3]["prior_src"] == "first_pass"
    return out


def test_off_attenuation():
    """off_db = -3 (DESIGN; the tuned meeting default is 0): -3 dB after 300 ms without speech; offline,
    fully restored at every speech frame (20 ms look-ahead); online, restored within 20 ms of the
    onset. Its share of the noise modulation is 3 dB."""
    sc = turns_scene([-55.0, -55.0], [0, 1, 0], np.random.default_rng(9), gap_s=2.0, noise_db=-85.0)
    out = {}
    assert not agc.level(sc.y, keys=sc.keys).off_db.any()
    for online in (False, True):
        res = agc.level(sc.y, keys=sc.keys, online=online, off_db=-3.0)
        sp = res.p >= agc.PARAMS["p_speech"]
        on = np.flatnonzero(sp[1:] & ~sp[:-1]) + 1
        if online:
            late = [float(np.max(-res.off_db[l + 3:l + 4])) for l in on]
            assert max(late) <= 1e-9, f"not restored 24 ms after onsets: {late}"
        else:
            assert np.abs(res.off_db[sp]).max() <= 1e-9, "off-attenuation active on speech frames"
        last = np.flatnonzero(sp)
        gap = frames_in(len(sp), sc.turns[0][1], sc.turns[1][0])
        g = np.flatnonzero(gap & (res.off_db < -1e-9))
        first_off = (g[0] - last[last < g[0]].max()) * cm.HOP_S
        assert first_off >= 0.3 - cm.HOP_S, first_off
        idle = np.mean(res.off_db[gap & (res.off_db <= -2.99)])
        out[f"{'online' if online else 'offline'}"] = f"idle {idle:.2f} dB from {first_off:.2f} s after speech"
    assert abs(idle + 3.0) <= 0.01
    return out


# ------------------------------------------------------------------ limiter and automix
def test_limiter_true_peak():
    """The output never exceeds -1 dBTP on the limiter's 4x meter (DESIGN: 4x oversampled detection):
    inter-sample peaks of an fs/4 sine, a 7.5 kHz sine, clicks, loud noise and loud voices, and the
    level() output pushed into the limiter. Against an ideal 16x FFT reference, speech-like outputs
    also stay under -1 dBTP; for near-Nyquist synthetic content the 4x grid under-reads (reported).
    Below the ceiling the limiter is exactly transparent; attenuation starts at most 5 ms before a
    click and releases with a 50 ms time constant."""
    rng = np.random.default_rng(10)
    n = 2 * FS
    t = np.arange(n) / FS
    clicks = white(n, rng, -30.0)
    clicks[rng.integers(100, n - 100, 20)] = 3.0
    sigs = {"fs/4 sine at 45 deg (sample peak 0.71 A)": (1.5 * np.cos(2 * np.pi * 4000 * t + np.pi / 4), False),
            "7.5 kHz sine": (1.2 * np.sin(2 * np.pi * 7500 * t), False),
            "clicks on noise": (clicks, False),
            "white noise rms 0.7": (0.7 * rng.standard_normal(n), False),
            "voice peaking at +10 dBFS": (voice(n, rng, 150.0) * 10 ** (10 / 20) / 5.0, True)}
    pairs = {name: (y, *agc.limiter(y), speech_like) for name, (y, speech_like) in sigs.items()}
    sc = turns_scene([-40.0, -30.0], [0, 1], rng, noise_db=-90.0)
    res = agc.level(sc.y, keys=sc.keys, target_db=-8.0)
    assert res.diag["limited_frac"] > 0
    y_agc = res.pre * agc.frame_gain(res.agc_db + res.off_db, len(res.pre))
    z, g = agc.limiter(y_agc)
    assert np.array_equal(z, res.out)
    pairs["level() output, target -8 dBFS"] = (y_agc, z, g, True)
    out = {}
    for name, (y, z, g, speech_like) in pairs.items():
        tp4, tp16 = agc.true_peak_db(z), tp16_db(z)
        assert tp4 <= CEIL + 1e-6, f"{name}: {tp4:.4f} dBTP"
        assert np.all(g <= 1.0) and np.all(g > 0)
        if speech_like:
            assert tp16 <= CEIL + 1e-3, f"{name}: 16x reference {tp16:.4f} dBTP"
        out[name] = f"in {agc.true_peak_db(y):+.2f} -> out {tp4:+.3f} dBTP (16x reference {tp16:+.3f})"
    out["level() output, target -8 dBFS"] += f", limited {100 * res.diag['limited_frac']:.1f} % of samples"
    quiet = 0.05 * rng.standard_normal(n)
    z, g = agc.limiter(quiet)
    assert np.array_equal(z, quiet) and np.all(g == 1.0)
    y = np.zeros(n)
    y[FS] = 2.0
    _, g = agc.limiter(y)
    pre = np.flatnonzero(g < 1.0)[0]
    assert FS - pre <= round(0.005 * FS) + 1, FS - pre      # the peak at FS - 0.25 needs gain from FS - 1
    a_pk = -20 * np.log10(g[FS])
    a_rel = -20 * np.log10(g[FS + 1 + 40 + 800])           # hold ends at FS + 1; box centre 40 samples back
    assert abs(a_rel / a_pk - np.exp(-1)) < 0.02, a_rel / a_pk
    out["click"] = (f"attenuation starts {1000 * (FS - pre) / FS:.2f} ms early; 50 ms after the hold "
                    f"{a_rel / a_pk:.3f} of the peak attenuation (e^-1 = 0.368)")
    return out


def test_dugan_automix():
    """One node passes through; two nodes: the talking node gets ~all the gain, sum g^2 = 1, and the
    background of two uncorrelated nodes stays at one node's level."""
    rng = np.random.default_rng(11)
    n = 6 * FS
    z, g = agc.dugan_automix(np.ones((1, n)) * 0.1)
    assert np.array_equal(z, np.ones(n) * 0.1) and np.all(g == 1)
    nz = np.stack([white(n, rng, -60.0), white(n, rng, -60.0)])
    sp = np.zeros(n)
    sp[2 * FS:4 * FS] = at_level(voice(2 * FS, rng, 140.0), -40.0)
    x = nz + np.stack([sp, 0.1 * sp])
    mix, g = agc.dugan_automix(x)
    assert np.abs((g ** 2).sum(0) - 1).max() < 1e-9
    talk = slice(int(2.3 * FS), int(3.9 * FS))
    g_on = np.median(g[:, talk], axis=1)
    assert g_on[0] > 0.9 and g_on[1] < 0.45, g_on
    quiet = slice(5 * FS, 6 * FS)
    rel = 10 * np.log10(np.mean(mix[quiet] ** 2) / np.mean(nz[0, quiet] ** 2))
    assert abs(rel) <= 1.0, rel
    up = np.flatnonzero(g[0, 2 * FS:] > 0.9)[0] / FS
    return {"talking node gains (node 1, node 2)": np.round(g_on, 3).tolist(),
            "background vs one node": f"{rel:+.2f} dB", "gain > 0.9 after onset": f"{1000 * up:.0f} ms"}


# ------------------------------------------------------------------ wrapper (real S0/S1 front end)
def meeting_scene(rng, d=0.04, rounds=3, turn_s=3.0, noise_db=-85.0):
    """3 plane-wave talkers (tau -90 / 0 / +90 us) at band levels -62 / -56 / -50 dBFS taking turns,
    in diffuse (sinc-coherent) white noise. Returns x (2, T), clean Mic-A speech, turns."""
    taus, levels = (-90e-6, 0.0, 90e-6), (-62.0, -56.0, -50.0)
    order = [0, 1, 2] * rounds
    T = int((1.0 + len(order) * (turn_s + 0.5) + 1.0) * FS)
    x, s = np.zeros((2, T)), np.zeros(T)
    turns = []
    for i, k in enumerate(order):
        a = int((1.0 + i * (turn_s + 0.5)) * FS)
        b = a + int(turn_s * FS)
        v = at_level(voice(b - a, rng, 105 + 40 * k), levels[k])
        x[:, a:b] += plane_pair(v, taus[k])
        s[a:b] += v
        turns.append((a, b, k))
    nz = coherent_noise(T, d, rng)
    x += nz * 10 ** ((noise_db - float(agc.dbfs(np.median(agc.band_power(nz[0]))))) / 20)
    return x, s, turns


def test_wrapper_meeting():
    """m11_level on a synthetic meeting (offline and info['online']): aligned with Mic A (B.6),
    'pre_agc' is Mic A after the high-pass, and the 12 dB talker spread shrinks to <= 3 dB on returning
    turns."""
    x, s, turns = meeting_scene(np.random.default_rng(12))
    out = {}
    for online in (False, True):
        info = {"scenario": "meeting", "spacing_m": 0.04, "online": online}
        st = m11_level.enhance(x.astype(np.float32), FS, dict(info), return_stages=True)
        y, pre = st["out"], st["pre_agc"]
        assert y.shape == pre.shape == (x.shape[1],) and np.all(np.isfinite(y))
        assert np.array_equal(m11_level.enhance(x.astype(np.float32), FS, dict(info)), y)
        lag_pre, lag_s = lag_of(y, pre), lag_of(y, s)
        assert lag_pre == 0 and abs(lag_s) <= 2, (lag_pre, lag_s)
        lin = [active_db(pre[a + FS:b]) for a, b, _ in turns[3:]]
        lout = [active_db(y[a + FS:b]) for a, b, _ in turns[3:]]
        assert np.ptp(lout) <= 3.0, np.round(lout, 1)
        out["online" if online else "offline"] = (f"lag vs pre / clean {lag_pre} / {lag_s} samples; spread on "
                                                  f"returning turns {np.ptp(lin):.1f} -> {np.ptp(lout):.1f} dB; "
                                                  f"levels {np.round(lout, 1).tolist()} dBFS; "
                                                  f"{agc.true_peak_db(y):+.1f} dBTP")
    return out


def test_wrapper_bus():
    """m11_level on a synthetic bus pair in loud diffuse babble-like noise: aligned, finite, and the
    noise cap holds the gain in [-10, 0] dB (no boost of the noise)."""
    rng = np.random.default_rng(13)
    n = 20 * FS
    x = np.zeros((2, n))
    s = np.zeros(n)
    for i in range(6):
        a, b = int((1 + 3 * i) * FS), int((3.5 + 3 * i) * FS)
        v = at_level(voice(b - a, rng, 120 + 80 * (i % 2)), -40.0)
        x[:, a:b] += plane_pair(v, (100e-6, -100e-6)[i % 2])
        s[a:b] += v
    nz = coherent_noise(n, 0.04, rng) * 10 ** (3 * np.sin(2 * np.pi * np.arange(n) / FS / 0.4) / 20)
    x += nz * 10 ** ((-45.0 - float(agc.dbfs(np.median(agc.band_power(nz[0]))))) / 20)
    st = m11_level.enhance(x.astype(np.float32), FS, {"scenario": "bus", "spacing_m": 0.04}, return_stages=True)
    y, pre = st["out"], st["pre_agc"]
    assert np.all(np.isfinite(y)) and lag_of(y, pre) == 0 and abs(lag_of(y, s)) <= 2
    m = np.abs(pre) > 1e-4
    g = 20 * np.log10(np.abs(y[m]) / np.abs(pre[m]))
    assert g.max() <= 1e-6 and g.min() >= -10.0 - 1e-6, (g.min(), g.max())
    return {"sample gain range": f"{g.min():+.2f} .. {g.max():+.2f} dB", "lag vs clean": lag_of(y, s)}


def test_determinism_and_rtf():
    """B.8: same input -> same output; RTF < 0.5 for the wrapper (60 s, both scenarios) and for the AGC
    alone on a 10 min signal."""
    rng = np.random.default_rng(14)
    x, _, _ = meeting_scene(rng, rounds=6)
    x = np.concatenate([x, x], axis=1)[:, :60 * FS].astype(np.float32)
    out = {}
    for scen in ("meeting", "bus"):
        info = {"scenario": scen, "spacing_m": 0.04}
        t0 = time.perf_counter()
        y1 = m11_level.enhance(x, FS, dict(info))
        rtf = (time.perf_counter() - t0) / (x.shape[1] / FS)
        assert np.array_equal(y1, m11_level.enhance(x.copy(), FS, dict(info))), f"{scen}: not deterministic"
        assert rtf < 0.5, rtf
        out[f"m11_level {scen} RTF (60 s)"] = f"{rtf:.4f}"
    y = np.tile(x[0].astype(np.float64), 10)
    t0 = time.perf_counter()
    agc.level(y)
    rtf = (time.perf_counter() - t0) / (len(y) / FS)
    assert rtf < 0.5, rtf
    out["agc.level alone RTF (10 min, energy VAD, pooled key)"] = f"{rtf:.4f}"
    return out


TESTS = [test_convergence_12db, test_settling_memory, test_online_label_lag, test_no_pumping_on_noise,
         test_bus_gate, test_noise_cap, test_boost_limit, test_noise_subtraction, test_segment_keys,
         test_tau_prior, test_off_attenuation, test_limiter_true_peak, test_dugan_automix, test_wrapper_meeting,
         test_wrapper_bus, test_determinism_and_rtf]

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
