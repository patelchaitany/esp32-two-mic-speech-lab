"""Tests for M9 target-speaker extraction (methods/_core/tse.py, methods/m9_voxlock2*.py).

Run from LAB: .venv/bin/python -m tests.test_tse [--quick]
--quick skips the slower model checks (release test vector, wrapper determinism, self-enrollment wrapper,
online-preset runs).
Plain asserts; prints the measured numbers. Data used (read only, approved): the VoxLock release test wavs
(models/voxlock/.../test_wavs, loaded with soundfile) and LibriSpeech test-clean utterances.
Run time depends on the machine load (VoxLock runs at RTF ~0.45 per pass on an idle M1 Pro core, 1-1.3 under
the load of the other agents). Measured: full run 294 s at load average 6-24, 610 s at 110-165; --quick
239 s at 150-165.
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from lab.common import FS
from lab.scenes import coherent_noise
from methods._core import calib, common as cm, tracker, tse

BUS = {"scenario": "bus", "spacing_m": 0.04}
CLEAN = os.path.join(tse.LAB, "data", "librispeech_test_clean", "LibriSpeech", "test-clean")
# Two official target-DEV speakers (corpus_libri/splits.json): F 1284, M 7127, with a scene chapter that is not
# their enrollment chapter. Explicit enrollment = their official reserved sample (corpus_libri/enrollment.json),
# a different recording from the scene, as the enrollment interface requires.
SPK = {"A": ("1284", "1180"), "B": ("7127", "75947")}
ENROLL_JSON = os.path.join(tse.LAB, "corpus_libri", "enrollment.json")


def check(name, ok, msg):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


def snr_db(est, ref):
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((est - ref) ** 2), 1e-30)))


def frac_delay(x, tau):
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * f * tau), nfft)[:n]


def utts(spk, chapter, n, min_s=3.0):
    """First n utterances (sorted) of a chapter that are at least min_s long, unit active level."""
    d = os.path.join(CLEAN, spk, chapter)
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".flac"):
            x, fs = sf.read(os.path.join(d, fn), dtype="float64")
            assert fs == FS
            if len(x) >= min_s * FS:
                out.append(x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 0.03)
            if len(out) == n:
                break
    return out


def conversation(turn_s=5.0):
    """Dry two-talker conversation A, B, A, B (each turn cut to turn_s), plus solo masks and enrollment
    clips from the other chapter. Returns (src {'A','B'} (T,), act {'A','B'} bool (T,), enroll {'A','B'})."""
    n = int(turn_s * FS)
    parts = {k: utts(s, c, 2, turn_s) for k, (s, c) in SPK.items()}
    T = 4 * n
    src = {k: np.zeros(T) for k in "AB"}
    act = {k: np.zeros(T, bool) for k in "AB"}
    for i, k in enumerate("ABAB"):
        seg = parts[k][i // 2][:n]
        src[k][i * n: i * n + len(seg)] = seg
        act[k][i * n: (i + 1) * n] = True
    import json
    reserved = json.load(open(ENROLL_JSON))["speakers"]
    enroll = {}
    for k, (spk, _) in SPK.items():
        smp = reserved[spk]["samples"][0]
        x = np.concatenate([sf.read(os.path.join(tse.LAB, p), dtype="float64")[0] for p in smp["paths"]])
        enroll[k] = x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 0.03
    return src, act, enroll


def room_scene(src, ang=(30.0, 150.0), r=0.6, rt60=0.3, noise_db=-15.0, seed=0):
    """2-ch scene: deterministic pyroomacoustics shoebox (no randomised ISM; 6 x 4 x 2.6 m, ISM order <= 12), 4 cm pair
    along x at (3, 2, 1) m, partner A / B at r m and azimuth ang (deg from the Mic-A axis), mouths 0.2 m above
    the mics, plus diffuse sinc-coherent noise at noise_db re 0.03 rms. S1 needs a room (in anechoic synthetic
    scenes its S0/S1 statistics found one or no partner). Returns (x (2, T), true TDOAs {'A', 'B'} in s)."""
    import pyroomacoustics as pra
    from scipy.signal import fftconvolve
    dims, c, u = [6.0, 4.0, 2.6], np.array([3.0, 2.0, 1.0]), np.array([1.0, 0.0, 0.0])
    mics = np.array([c + 0.02 * u, c - 0.02 * u])
    e_abs, order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=min(order, 12))
    pos = {}
    for k, a in zip("AB", ang):
        pos[k] = c + r * np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0]) + np.array([0, 0, 0.2])
        room.add_source(list(pos[k]))
    room.add_microphone_array(mics.T)
    room.compute_rir()
    T = len(src["A"])
    x = np.zeros((2, T))
    for i, k in enumerate("AB"):
        for m in range(2):
            x[m] += fftconvolve(src[k], room.rir[m][i])[:T]
    x += coherent_noise(T, 0.04, np.random.default_rng(seed)) * 0.03 * 10 ** (noise_db / 20)
    tau = {k: (np.linalg.norm(pos[k] - mics[1]) - np.linalg.norm(pos[k] - mics[0])) / 343.0 for k in "AB"}
    return x, tau


# ------------------------------------------------------------------ tests
def test_model_meta():
    t = time.perf_counter()
    m = tse.get_model(1)
    load = time.perf_counter() - t
    mag, ctx = m.enroll(np.zeros(tse.SAMPLE, np.float32) + 1e-3)
    check("meta", m.gain == np.float32(-0.5) and m.state_shape == (6, 32, 256)
          and m.frames_per_run == tse.REF_FRAMES_PER_RUN == 100 and m.version == "1.1.1",
          f"gain {m.gain}, state {m.state_shape}, frames/run {m.frames_per_run}, version {m.version}, "
          f"load {load:.2f} s")
    check("enroll shapes", mag.shape == (1, 257, 376) and ctx.shape == (1, 512, 298), f"{mag.shape} {ctx.shape}")
    check("threads", m.threads == 1, "intra_op_num_threads = 1 (RTF is measured single-process)")


def test_fit_sample():
    rng = np.random.default_rng(1)
    x3 = rng.standard_normal(tse.SAMPLE).astype(np.float32) * 0.01
    s, st = tse.fit_sample(x3)
    check("exactly 3 s", st == 0 and np.array_equal(s, x3), "returned unchanged")
    x5 = rng.standard_normal(5 * FS) * 0.01
    x5[24000:72000] *= 10                           # loudest 3 s start at 1.5 s
    s, st = tse.fit_sample(x5)
    check("loudest 3 s", st == 24000 and len(s) == tse.SAMPLE, f"start {st}")
    x40 = rng.standard_normal(40 * FS) * 0.01
    x40[10 * FS: 13 * FS] *= 5
    x40[32 * FS: 35 * FS] *= 20                     # louder, but beyond the first 30 s
    s, st = tse.fit_sample(x40)
    check("within 30 s", st == 10 * FS, f"start {st / FS:.2f} s (louder part at 32 s ignored)")
    try:
        tse.fit_sample(np.ones(tse.SAMPLE - 1))
        ok = False
    except ValueError:
        ok = True
    check("< 3 s rejected", ok, "ValueError")
    bad = x40.copy()
    bad[5 * FS] = np.nan
    late = x40.copy()
    late[35 * FS] = np.inf                          # beyond the 30 s fit_sample reads: harmless
    try:
        tse.fit_sample(bad)
        ok = False
    except ValueError:
        ok = True
    s, st = tse.fit_sample(late)
    check("NaN/inf rejected where read", ok and st == 10 * FS and np.all(np.isfinite(s)),
          "NaN at 5 s -> ValueError; inf at 35 s ignored")
    model = tse.get_model(1)
    msgs = {}
    for name, clip in (("zeros", np.zeros(5 * FS)), ("1e-30 noise", rng.standard_normal(5 * FS) * 1e-30),
                       ("NaN", np.full(5 * FS, np.nan))):
        try:
            tse.enroll(clip, model)
            msgs[name] = "accepted"
        except ValueError as e:
            msgs[name] = str(e)
    c = tse.enroll(rng.standard_normal(5 * FS) * 1e-6, model)        # quiet but not silent: accepted
    check("unusable clips rejected", all(v != "accepted" for v in msgs.values()) and c.sample.shape == (tse.SAMPLE,),
          f"{msgs}; 1e-6 noise accepted")


def test_union_plumbing():
    """Without the model: an estimate equal to the input gives m = 1 everywhere and y == x."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(3 * FS) * 0.01
    X = cm.stft(x)
    cond = SimpleNamespace(sample=np.zeros(tse.SAMPLE, np.float32))
    r = tse.union(x, {"A": cond, "B": cond}, model=SimpleNamespace(), X=X,
                  est={"A": x, "B": np.zeros_like(x)}, use_guard=False)
    live = np.abs(X) > 0                            # the all-zero padded edge frames have mask 0 (0 / eps)
    check("identity union", np.max(np.abs(r.y - x)) < 1e-9 and np.allclose(r.mask[live], 1.0, atol=1e-6),
          f"max |y - x| {np.max(np.abs(r.y - x)):.2e}, min mask {r.mask[live].min():.9f}")
    check("strength floor", np.allclose(r.w, tse.STRENGTH_MIN, atol=1e-5) and r.f.max() < 1e-5,
          f"f <= {r.f.max():.1e} -> w = {r.w.mean():.6f}")
    # random masks: union >= each; m_eff between m_tse and 1; w within [0.2, 1]
    mA, mB = rng.uniform(0, 1, X.shape), rng.uniform(0, 1, X.shape)
    m_tse = np.maximum(mA, mB)
    w, f = tse.adaptive_strength(m_tse, X, np.ones(X.shape[-1]))
    m_eff = (1 - w) + w * m_tse
    check("strength range", w.min() >= 0.2 - 1e-12 and w.max() <= 1 + 1e-12 and np.all(m_eff >= m_tse - 1e-12)
          and np.all(m_eff <= 1 + 1e-12), f"w {w.min():.3f}..{w.max():.3f}, mean f {f.mean():.3f}")
    m = tse.lab_mask(cm.stft(0.5 * x), X)
    check("lab mask", m.shape == X.shape and np.allclose(m[:, 5:-9], 0.5, atol=1e-6), "half-level estimate -> 0.5")


def test_guard():
    """Guard: the pass that keeps < -6 dB in its S1 partner's segments is restored there (m = 1)."""
    N = 600
    X = np.ones((cm.N_BINS, N), complex)
    pA, pB = np.zeros(N), np.zeros(N)
    pA[100:250] = 1.0
    pB[350:500] = 1.0
    mA = np.full((cm.N_BINS, N), 0.05)              # pass A suppresses everything (-26 dB) ...
    mA[:, 100:250] = 0.3                            # ... and keeps -10.5 dB in its partner's segment
    mB = np.full((cm.N_BINS, N), 0.05)
    mB[:, 350:500] = 0.9                            # pass B keeps its partner (-0.9 dB)
    trk = SimpleNamespace(p_A=pA, p_B=pB, inseparable=False)
    out, d = tse.guard({"A": mA, "B": mB}, X, trk)
    check("guard assoc", d["assoc"] == {"A": "A", "B": "B"}, str(d["assoc"]))
    check("guard restore", np.all(out["A"][:, 100:250] == 1.0) and np.all(out["A"][:, :100] == 0.05)
          and np.array_equal(out["B"], mB), f"segments {d['segments']}")


def test_reference_vector(quick):
    """Port parity: the release voice sample is the loudest 3 s of VoxLock's fp32 output on the release test
    call (automatic sample). Our int8 port (no priming, as the reference) must contain it at >= 25 dB SNR,
    unit gain, right polarity, and the match must be sample-sharp."""
    if quick:
        print("  [skip] reference vector (--quick)")
        return
    d = os.path.join(tse.VOXLOCK_DIR, "test_wavs")
    call, _ = sf.read(os.path.join(d, "en-combined.wav"), dtype="float32")
    ref, _ = sf.read(os.path.join(d, "en-voice-sample-f.wav"), dtype="float64")
    smp, _ = tse.loud_sample(call)                  # automatic mode: a test vector only, never used on the bus
    y = tse.voxlock_isolate(call, tse.enroll(smp), prime=False).astype(np.float64)
    c = np.correlate(y, ref, "valid")
    i = int(np.argmax(np.abs(c)))
    seg = y[i: i + len(ref)]
    g = 20 * np.log10(abs(ref @ seg) / (seg @ seg))
    s0, s1 = snr_db(seg, ref), max(snr_db(y[i + 1: i + 1 + len(ref)], ref), snr_db(y[i - 1: i - 1 + len(ref)], ref))
    check("release vector", s0 >= 25 and c[i] > 0 and abs(g) < 0.5 and s1 < s0 - 10,
          f"SNR {s0:.1f} dB at offset {i / FS:.2f} s (+-1 sample: {s1:.1f} dB), gain {g:+.2f} dB")


def test_partition_and_latency():
    """Run partition: same partition -> bit-identical; 10-frame runs vs 100 -> int8 quantization only
    (sherpa-onnx reports 46-47 dB); streaming output is 384 samples (24 ms) behind the input."""
    call, _ = sf.read(os.path.join(tse.VOXLOCK_DIR, "test_wavs", "en-combined.wav"), dtype="float32")
    x = call[: 6 * FS]
    cond = tse.enroll(tse.loud_sample(call)[0])
    y1 = tse.voxlock_isolate(x, cond, prime=False)
    y2 = tse.voxlock_isolate(x, cond, prime=False)
    y3 = tse.voxlock_isolate(x, cond, frames_per_run=10, prime=False)
    check("determinism", np.array_equal(y1, y2), "two runs bit-identical")
    s = snr_db(y3, y1)
    check("partition", s >= 30, f"10-frame runs vs 100-frame runs: {s:.1f} dB")
    st = tse.VoxLockStream(tse.get_model(), cond)
    n = 256 + 128 * 40
    out = st.push(x[:n], frames_per_run=5)
    check("stream latency", len(out) == n - 384, f"{n} samples in -> {len(out)} out ({n - len(out)} = 24 ms held)")


def test_extraction_and_alignment():
    """Dry mixed-gender conversation (A, B, A, B; A talks first). Default (unprimed) passes keep their own
    partner, the union keeps both, outputs are time-aligned (lag 0). A pass primed with its voice sample
    ('sample') must still keep its partner. (VoxLock's first-voice latching is speaker-dependent: seen for
    other pairs, not for this one; see tse.PRIME.)"""
    src, act, enr = conversation()
    x = src["A"] + src["B"]
    X = cm.stft(x)
    T = len(x)
    conds = {k: tse.enroll(enr[k]) for k in "AB"}
    r = tse.union(x, conds, X=X, use_guard=False, strength="full")

    def share(mk, k, o):
        e = {j: float(np.sum(cm.istft(mk * cm.stft(src[j]), T) ** 2)) for j in "AB"}
        return e[k] / (e[k] + e[o])

    lags = {}
    for k in "AB":
        sel = act[k]
        c = np.correlate(r.est[k][sel], src[k][sel][100:-100], "valid")
        lags[k] = int(np.argmax(c)) - 100
    sh = {k: share(r.masks[k], k, o) for k, o in (("A", "B"), ("B", "A"))}
    check("passes", min(sh.values()) >= 0.8 and abs(lags["A"]) <= 2 and abs(lags["B"]) <= 2,
          f"own share A {sh['A']:.2f}, B {sh['B']:.2f} (unprimed, B talks second); lags {lags}")
    ta = {k: 10 * np.log10(np.sum(cm.istft(r.mask * cm.stft(src[k]), T)[act[k]] ** 2) / np.sum(src[k][act[k]] ** 2))
          for k in "AB"}
    check("union keeps both", min(ta.values()) > -3.0, f"TA_A {ta['A']:+.1f} dB, TA_B {ta['B']:+.1f} dB")
    c = np.correlate(r.y, x[200:-200], "valid")
    check("union aligned", int(np.argmax(c)) - 200 == 0, f"lag {int(np.argmax(c)) - 200}")
    yb = tse.voxlock_isolate(x, conds["B"], prime="sample")
    shb = share(tse.lab_mask(yb, X), "B", "A")
    check("primed pass B", shb >= 0.8, f"own share B {shb:.2f} with a 'sample' primer (unprimed {sh['B']:.2f})")


def _truth_track(act, taus, N):
    """S1-like Track from ground truth (p_k = 1 in k's solo frames, tau = tau_k there, CDR +5 dB, SNR 20 dB)."""
    c = np.clip(cm.frame_centers(N), 0, len(act["A"]) - 1)
    solo = {"A": act["A"][c] & ~act["B"][c], "B": act["B"][c] & ~act["A"][c]}
    tau = np.where(solo["A"], taus["A"], np.where(solo["B"], taus["B"], 0.0))
    on = solo["A"] | solo["B"]
    return SimpleNamespace(p_A=solo["A"].astype(float), p_B=solo["B"].astype(float), tau=tau,
                           tau_A=taus["A"], tau_B=taus["B"],
                           tau_A_t=np.full(N, taus["A"]), tau_B_t=np.full(N, taus["B"]),
                           cdr_db=np.where(on, 5.0, -5.0), double_talk=np.zeros(N, bool),
                           level_db=np.where(on, -40.0, -60.0), noise_db=np.full(N, -60.0),
                           inseparable=False, inseparable_why=())


def test_self_enrollment():
    """Self-enrollment on a 2-ch room conversation (A at 30 deg, B at 150 deg, 0.5 m).
    (1) Selection logic with a ground-truth track (independent of S1's own behaviour): the chosen segments lie in
    the right partner's turns, the voice samples are 3 s, the speaker-model gates pass (clean speech).
    (2) The real S1 front end on two geometries: 30/150 deg (where the S0 version of 00:30 applies tau0 = +31 us
    and S1 sees one peak) and 60/170 deg (S1 resolves both partners with that version). Where S1 resolves both
    partners, the segments must lie in the right turns; where not, self_enroll must fail cleanly with an S1
    reason (S1 is owned elsewhere and still changing)."""
    from methods._core import calib
    src, act, _ = conversation(turn_s=8.0)
    x, taus = room_scene(src, r=0.5, noise_db=-20.0)
    xh = calib.prefilter(x, "bus")
    X = cm.stft(xh)
    fe = SimpleNamespace(track=_truth_track(act, taus, X.shape[-1]), X=X, x=xh)
    se = tse.self_enroll(fe, min_total_s=1.5)
    for k in "AB":
        segs = se.segments.get(k, [])
        tot = sum(b - a for a, b, _ in segs)
        inside = sum(int(act[k][a:b].sum()) for a, b, _ in segs)
        check(f"segments {k} (truth track)", tot >= 1.5 * FS and inside / tot >= 0.9,
              f"{len(segs)} runs, {tot / FS:.2f} s, {100 * inside / max(tot, 1):.0f} % inside {k}'s turns")
    check("self ok (truth track)", se.ok and all(len(c.sample) == tse.SAMPLE for c in se.conds.values()),
          f"ok={se.ok} reasons={se.reasons} sims={se.sims}")
    resolved = 0
    for ang in ((30.0, 150.0), (60.0, 170.0)):
        if ang != (30.0, 150.0):
            x, taus = room_scene(src, ang=ang, r=0.5, noise_db=-20.0)
        fe = tracker.front_end(x, FS, BUS)
        trk = fe.track
        se = tse.self_enroll(fe, min_total_s=1.5)
        tag = f"real S1 {ang[0]:.0f}/{ang[1]:.0f} deg"
        if trk.inseparable:
            check(f"{tag} (inseparable)", not se.ok and any("S1" in r for r in se.reasons),
                  f"S1 tau_A {trk.tau_A * 1e6:.1f} us, tau_B {trk.tau_B * 1e6:.1f} us {trk.inseparable_why} -> clean "
                  f"failure: {se.reasons}")
            continue
        resolved += 1
        for k in "AB":                              # S1 'A' = larger TDOA = the source nearer the Mic-A axis = A
            segs = se.segments.get(k, [])
            tot = sum(b - a for a, b, _ in segs)
            inside = sum(int(act[k][a:b].sum()) for a, b, _ in segs)
            check(f"{tag} segments {k}", tot == 0 or inside / tot >= 0.9,
                  f"tau_{k} {(trk.tau_A if k == 'A' else trk.tau_B) * 1e6:.1f} us (true {taus[k] * 1e6:.1f}); "
                  f"{len(segs)} runs, {tot / FS:.2f} s, {100 * inside / max(tot, 1):.0f} % inside {k}'s turns; "
                  f"ok={se.ok} {se.reasons}")
    if not resolved:
        print("  [note] real S1 resolved neither geometry: the separable path was not exercised (S1 owner)")


def test_wrappers(quick):
    """Method contract: shape, finite, alignment, determinism, meeting passthrough, enrollment fallback."""
    import methods.m9_voxlock2 as m9
    import methods.m9_voxlock2_self as m9s
    src, act, enr = conversation(turn_s=4.0)
    x, _ = room_scene(src, seed=3)
    T = x.shape[1]
    info = dict(BUS, enroll=enr)
    t = time.perf_counter()
    st = m9.enhance(x, FS, info, return_stages=True)
    sec = time.perf_counter() - t
    y = st["out"]
    check("shape/finite", y.shape == (T,) and np.all(np.isfinite(y)) and st["diag"]["used"] == "tse"
          and st["diag"]["latency_ms"] == m9.ALGO_LATENCY_MS,
          f"{y.shape}, used {st['diag']['used']}, {sec:.1f} s for {T / FS:.0f} s (RTF {sec / (T / FS):.2f}), "
          f"latency {st['diag']['latency_ms']:.0f} ms")
    xa = calib.prefilter(x, "bus")[0]
    c = np.correlate(y, xa[200:-200], "valid")
    check("aligned", int(np.argmax(c)) - 200 == 0, f"lag {int(np.argmax(c)) - 200} vs prefiltered Mic A")
    ym = m9.enhance(x, FS, {"scenario": "meeting", "spacing_m": 0.04})
    check("meeting passthrough", np.allclose(ym, calib.prefilter(x, "meeting")[0]), "TSE disabled in the meeting")
    yf = m9.enhance(x, FS, BUS, return_stages=True)
    check("fallback", "self-enrollment instead" in " ".join(yf["diag"]["reasons"]),
          f"no info['enroll'] -> {yf['diag']['used']}: {yf['diag']['reasons'][:2]}")
    if quick:
        print("  [skip] wrapper determinism and self-enrollment wrapper (--quick)")
        return
    y2 = m9.enhance(x, FS, info)
    check("determinism", np.array_equal(y, y2), "two calls bit-identical")
    ys = m9s.enhance(x, FS, dict(BUS, tse={"self_min_total_s": 1.5}), return_stages=True)
    d = ys["diag"]["self_enrollment"]
    check("self wrapper", ys["out"].shape == (T,) and "segments" in d and np.all(np.isfinite(ys["out"])),
          f"used {ys['diag']['used']}, ok {d['ok']}, segments {[len(v) for v in d['segments'].values()]}, "
          f"reasons {d['reasons']}")


def test_contract():
    """Method contract on edge inputs: the bench flags (OFFLINE_ONLY, ALGO_LATENCY_MS, read as lab/bench2 reads
    them), T = 0, very short inputs, silence, unusable enrollment clips (all-zero, 1e-30, NaN) -> fallback with a
    reason and never NaN, fallback 'none', enrollment as a (2, n) array or a list, and a non-finite pass output
    (an all-zero enroll spectrum, which makes the chunk graph emit NaN) -> fallback."""
    import methods.m9_voxlock2 as m9
    import methods.m9_voxlock2_self as m9s
    for mod in (m9, m9s):
        check(f"flags {mod.__name__.split('.')[-1]}", getattr(mod, "OFFLINE_ONLY", None) is True
              and getattr(mod, "ALGO_LATENCY_MS", None) == tse.latency_ms() == 856.0,
              f"OFFLINE_ONLY {getattr(mod, 'OFFLINE_ONLY', None)}, ALGO_LATENCY_MS {getattr(mod, 'ALGO_LATENCY_MS', None)}")
    check("online latency", m9.ALGO_LATENCY_ONLINE_MS == tse.latency_ms(tse.ONLINE_FRAMES_PER_RUN) == 96.0,
          f"{m9.ALGO_LATENCY_ONLINE_MS} ms (24 look-ahead + 40 run + 32 lab STFT)")
    try:
        from lab import bench2
    except Exception as e:  # noqa: BLE001  (lab/bench2 is owned elsewhere and may be mid-edit)
        print(f"  [skip] bench2 view: {type(e).__name__}: {e}")
    else:
        got = {}
        for name in ("m9_voxlock2", "m9_voxlock2_self"):
            mod, _, stages = bench2.load_method(name)              # raises PolicyError on a contract breach
            got[name] = (bool(getattr(mod, "OFFLINE_ONLY", False)), getattr(mod, "ALGO_LATENCY_MS", None), stages)
        check("bench2 view", all(v[0] and v[1] == 856.0 and v[2] for v in got.values()),
              f"(offline_only, algo_latency_ms, stages) {got}; policy check passed")
    src, act, enr = conversation(turn_s=3.0)
    n = min(len(enr["A"]), len(enr["B"]))
    enr = {k: v[:n] for k, v in enr.items()}                     # same length: dict, (2, n) array, list agree
    rng = np.random.default_rng(4)
    s = (src["A"] + src["B"])[: 3 * FS]
    x = np.stack([s, np.roll(s, 1)]) + rng.standard_normal((2, len(s))) * 1e-4
    xa = calib.prefilter(x, "bus")[0]
    info = dict(BUS, enroll=enr)
    shapes = {}
    for name, fn in (("explicit bus", lambda: m9.enhance(np.zeros((2, 0)), FS, info, return_stages=True)["out"]),
                     ("explicit meeting", lambda: m9.enhance(np.zeros((2, 0)), FS, {"scenario": "meeting"})),
                     ("self bus", lambda: m9s.enhance(np.zeros((2, 0)), FS, BUS))):
        shapes[name] = fn().shape
    check("T = 0", all(v == (0,) for v in shapes.values()), str(shapes))
    short = {}
    for T in (1, 100, 257, 512, 4000):
        y = m9.enhance(rng.standard_normal((2, T)) * 0.01, FS, info)
        short[T] = bool(y.shape == (T,) and np.all(np.isfinite(y)))
    check("short inputs", all(short.values()), f"T -> shape ok and finite: {short}")
    st = m9.enhance(np.zeros((2, 3 * FS)), FS, info, return_stages=True)
    check("silence", st["out"].shape == (3 * FS,) and np.all(st["out"] == 0) and st["diag"]["used"] == "tse",
          f"used {st['diag']['used']}, peak {np.max(np.abs(st['out'])):.1e}")
    nan = enr["B"].copy()
    nan[1000] = np.nan
    res = {}
    for name, clips, opt in (("all-zero A", {"A": np.zeros(n), "B": enr["B"]}, None),
                             ("1e-30 B", {"A": enr["A"], "B": np.full(n, 1e-30)}, None),
                             ("NaN in B", {"A": enr["A"], "B": nan}, None),
                             ("all-zero A, fallback none", {"A": np.zeros(n), "B": enr["B"]}, {"fallback": "none"})):
        st = m9.enhance(x, FS, dict(BUS, enroll=clips, tse=opt), return_stages=True)
        r = st["diag"]["reasons"]
        ok = (np.all(np.isfinite(st["out"])) and np.array_equal(st["out"], xa) and st["diag"]["used"] == "passthrough"
              and r and r[0].startswith("explicit enrollment unusable")
              and (("self_enrollment" in st["diag"]) == (opt is None)))
        res[name] = (bool(ok), r[0] if r else None)
    check("unusable clips", all(v[0] for v in res.values()), "; ".join(f"{k}: {v[1]}" for k, v in res.items()))
    y_dict = m9.enhance(x, FS, info)
    y_arr = m9.enhance(x, FS, dict(BUS, enroll=np.stack([enr["A"], enr["B"]])))
    y_list = m9.enhance(x, FS, dict(BUS, enroll=[enr["A"], enr["B"]]))
    check("enroll formats", np.all(np.isfinite(y_dict)) and np.array_equal(y_dict, y_arr)
          and np.array_equal(y_dict, y_list) and not np.array_equal(y_dict, xa),
          "dict, (2, n) array and list give the same TSE output")
    orig = tse.enroll

    def zero_spectrum(audio, model=None, source="explicit", segments=None, diag=None):
        c = orig(audio, model, source, segments, diag)
        c.enroll_mag = np.zeros_like(c.enroll_mag)
        return c

    tse.enroll = zero_spectrum
    try:
        st = m9.enhance(x, FS, dict(info, tse={"fallback": "none"}), return_stages=True)
    finally:
        tse.enroll = orig
    r = st["diag"]["reasons"]
    check("non-finite pass", np.array_equal(st["out"], xa) and st["diag"]["used"] == "passthrough"
          and any("not finite" in v for v in r), f"{r}")


def test_online(quick):
    """Online preset (DESIGN 2.3 online variant, info['tse'] = {'online': True}): options, no S1 front end,
    causal output (the output for a prefix of the input equals the output for the whole input, bit for bit,
    except in the last 0.25 s before the cut), the strength VAD (common.speech_presence) is causal, and the
    output is close to the offline no-guard output (int8 per-run quantization only)."""
    o = tse.options({"online": True})
    o2 = tse.options({"online": True, "frames_per_run": 10})
    check("preset", o["frames_per_run"] == 5 and o["guard"] is False and o["fallback"] == "none"
          and o2["frames_per_run"] == 10, f"online -> frames_per_run {o['frames_per_run']}, guard {o['guard']}, "
                                          f"fallback {o['fallback']}; own keys override")
    rng = np.random.default_rng(7)
    xs = rng.standard_normal(20 * FS) * np.repeat(rng.uniform(1e-3, 0.1, 80), FS // 4)
    p_full, _ = cm.speech_presence(tse.band_energy(cm.stft(xs)).mean(axis=0))
    same = []
    for n in (100, 600, 1500, 2400):                # prefixes of n hops; compare frames that end before the cut
        pp, _ = cm.speech_presence(tse.band_energy(cm.stft(xs[: n * cm.HOP])).mean(axis=0))
        same.append(bool(np.array_equal(pp[: n - 4], p_full[: n - 4])))
    check("VAD causal", all(same), f"speech_presence on prefixes == on the whole signal: {same}")
    if quick:
        print("  [skip] online runs (--quick)")
        return
    import methods.m9_voxlock2 as m9
    src, act, enr = conversation(turn_s=1.5)
    x, _ = room_scene(src, seed=6)
    T = x.shape[1]
    info = dict(BUS, enroll=enr, tse={"online": True})
    t = time.perf_counter()
    st = m9.enhance(x, FS, info, return_stages=True)
    sec = time.perf_counter() - t
    y = st["out"]
    check("online run", st["diag"]["used"] == "tse" and st["front_end"] is None and y.shape == (T,)
          and np.all(np.isfinite(y)) and st["diag"]["latency_ms"] == 96.0,
          f"used {st['diag']['used']}, no S1, latency {st['diag']['latency_ms']:.0f} ms, {sec:.1f} s for "
          f"{T / FS:.0f} s")
    n = 4 * FS
    yp = m9.enhance(x[:, :n], FS, info)
    m = n - FS // 4
    d = float(np.max(np.abs(yp[:m] - y[:m])))
    check("online causal", d == 0.0, f"output for the first {n / FS:.0f} s == output for all {T / FS:.0f} s up to "
                                     f"0.25 s before the cut (max diff {d:.1e})")
    yo = m9.enhance(x, FS, dict(BUS, enroll=enr, tse={"guard": False}))
    s = snr_db(y, yo)
    check("online vs offline", s >= 20.0, f"SNR {s:.1f} dB against the offline no-guard output (100-frame runs)")


def main():
    quick = "--quick" in sys.argv
    t0 = time.perf_counter()
    tests = [("model meta", test_model_meta), ("fit_sample", test_fit_sample), ("union plumbing", test_union_plumbing),
             ("guard", test_guard), ("reference vector", lambda: test_reference_vector(quick)),
             ("partition/latency", test_partition_and_latency), ("extraction/alignment", test_extraction_and_alignment),
             ("self-enrollment", test_self_enrollment), ("wrappers", lambda: test_wrappers(quick)),
             ("contract", test_contract), ("online", lambda: test_online(quick))]
    for name, fn in tests:
        t = time.perf_counter()
        print(f"{name}:")
        fn()
        print(f"  ({time.perf_counter() - t:.1f} s)")
    print(f"all {len(tests)} tse tests passed in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
