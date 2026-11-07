"""Unit tests for M8 neural post-filters: methods/_core/nn.py, methods/m8_gtcrn.py, methods/m8_dpdfnet.py.

Run from the lab root:  .venv/bin/python -m tests.test_nn      (about 1 minute)
Uses the approved on-disk data: one LibriSpeech test-clean utterance of a DEV-split speaker
(237-134493-0002, use = scene in corpus_libri/utterances.tsv) and a
DEMAND TBUS ch01 segment, plus seeded synthetic signals. Covers the graph I/O and metadata, the
STFT definition (centre reflect padding, windows, perfect reconstruction), model delay and output
alignment (DESIGN Appendix B.6), the attenuation limit, the lab-grid mask, level invariance,
silence / short / full-scale / clipped inputs, determinism and RTF (B.8). Each test returns the
measured numbers it asserted on.
"""
import os
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from lab.common import active_rms, rms
from lab.metrics import align, estoi, si_sdr
from methods import m8_dpdfnet, m8_gtcrn
from methods._core import calib, common, nn

FS = 16000
LAB = nn.LAB
SPEECH = os.path.join(LAB, "data/librispeech_test_clean/LibriSpeech/test-clean/237/134493/237-134493-0002.flac")
NOISE = os.path.join(LAB, "data/demand_tbus/TBUS/ch01.wav")
WRAPPERS = {"gtcrn": m8_gtcrn, "dpdfnet": m8_dpdfnet}
BUS = {"scenario": "bus", "spacing_m": 0.04}


def speech_and_noise(level_db=-30.0, snr_db=0.0):
    """LibriSpeech utterance (active level level_db dBFS) and TBUS noise at in-band SNR snr_db
    (lab convention: 150-7000 Hz band-pass, speech power over active frames)."""
    s = sf.read(SPEECH, dtype="float64")[0]
    n = sf.read(NOISE, dtype="float64", start=10 * FS, frames=len(s))[0]
    s = s / active_rms(s) * 10 ** (level_db / 20)
    bp = butter(4, [150, 7000], btype="band", fs=FS, output="sos")
    g = active_rms(sosfilt(bp, s)) / rms(sosfilt(bp, n)) * 10 ** (-snr_db / 20)
    return s, n * g


def stereo(x):
    return np.stack([x, x]).astype(np.float32)


# ------------------------------------------------------------------ graph, metadata, sessions
def test_graph_and_sessions():
    out = {}
    for name in nn.MODELS:
        sess, spec = nn.load(name)
        assert nn.load(name)[0] is sess, "session not cached"
        so = sess.get_session_options()
        assert so.intra_op_num_threads == 1 and so.inter_op_num_threads == 1
        ins = {i.name: tuple(i.shape) for i in sess.get_inputs()}
        outs = [o.name for o in sess.get_outputs()]
        if name == "gtcrn":
            assert ins == {"mix": (1, 257, 1, 2), "conv_cache": (2, 1, 16, 16, 33),
                           "tra_cache": (2, 3, 1, 1, 16), "inter_cache": (2, 1, 33, 16)}, ins
            assert outs == ["enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"], outs
            assert (spec.n_fft, spec.hop, spec.delay) == (512, 256, 0)
            assert all(not s.any() for s in spec.state_init)
        else:
            assert ins == {"spec": (1, 1, 161, 2), "state_in": (52592,)}, ins
            assert outs == ["spec_e", "state_out"], outs
            assert (spec.n_fft, spec.hop, spec.delay) == (320, 160, 4)
            st = spec.state_init[0]
            assert st[0] == -60.0 and abs(st[31] + 90.0) < 1e-4, "erb_norm_init not in state[0:32]"
            assert abs(st[32] - 1e-3) < 1e-9 and abs(st[127] - 1e-4) < 1e-9, "spec_norm_init not in state[32:128]"
            assert not st[128:].any()
        out[name] = f"n_fft {spec.n_fft}, hop {spec.hop}, delay {spec.delay} frames, latency {nn.latency_ms(name):.0f} ms"
    assert nn.latency_ms("gtcrn") == 32.0 and nn.latency_ms("dpdfnet") == 60.0
    return out


# ------------------------------------------------------------------ STFT definition
def _reference_stft(x, n_fft, hop, w):
    """Independent loop version of torch.stft / librosa.stft(center=True, pad_mode='reflect')."""
    p = n_fft // 2
    xp = np.concatenate([x[1:p + 1][::-1], x, x[-p - 1:-1][::-1]])
    n = 1 + (len(xp) - n_fft) // hop
    return np.array([np.fft.rfft(xp[i * hop: i * hop + n_fft] * w) for i in range(n)])


def test_stft_definition():
    rng = np.random.default_rng(0)
    out = {}
    # windows: GTCRN = periodic sqrt-Hann (equal to lab.dsp's); DPDFNet = Vorbis, power complementary
    assert np.allclose(nn.periodic_sqrt_hann(512), np.sqrt(np.hanning(513)[:-1]), atol=1e-15)   # = lab.dsp window
    v = nn.vorbis_window(320)
    assert np.allclose(v[:160] ** 2 + v[160:] ** 2, 1.0, atol=1e-12) and np.allclose(v, v[::-1])
    for name in nn.MODELS:
        _, spec = nn.load(name)
        for T in (spec.n_fft * 3 + 17, 16000 + 5):
            x = rng.standard_normal(T)
            X = nn.analysis(x, spec)
            assert X.shape == (1 + T // spec.hop, spec.n_fft // 2 + 1)
            err = np.abs(X - _reference_stft(x, spec.n_fft, spec.hop, spec.window)).max()
            assert err < 1e-10, f"{name}: STFT differs from the reference definition ({err})"
            pr = np.abs(nn.synthesis(X, spec, T) - x).max()
            assert pr < 1e-12, f"{name}: no perfect reconstruction ({pr})"
        out[name] = f"STFT err {err:.1e}, reconstruction err {pr:.1e}"
    return out


# ------------------------------------------------------------------ delay and alignment
def test_delay_and_alignment():
    """Raw model output lags its input by exactly delay * hop; denoise() and the wrappers are aligned
    with Mic A after the S0 high-pass (B.6: |lag| <= 2 samples)."""
    s, n = speech_and_noise(snr_db=5.0)
    x = s + n
    out = {}
    for name in nn.MODELS:
        _, spec = nn.load(name)
        X = nn.analysis(np.pad(x, (0, 2048)), spec)
        y_raw = nn.synthesis(nn.run_frames(X, name), spec, len(x))
        lag_raw = align(y_raw, s, max_lag=1600)[1]
        assert lag_raw == spec.delay * spec.hop, f"{name}: raw model lag {lag_raw}"
        lag = align(nn.denoise(x, name), s, max_lag=1600)[1]
        assert lag == 0, f"{name}: denoise lag {lag}"
        s_h = calib.prefilter(s, "bus")
        y = WRAPPERS[name].enhance(stereo(x), FS, BUS)
        lag_w = align(y, s_h, max_lag=1600)[1]
        assert abs(lag_w) <= 2, f"{name}: wrapper lag {lag_w} vs high-passed Mic A speech"
        out[name] = f"raw model lag {lag_raw} samples; denoise lag {lag}; wrapper lag {lag_w}"
    return out


# ------------------------------------------------------------------ enhancement on real speech + noise
def test_enhancement_gain():
    """LibriSpeech + DEMAND TBUS at 0 dB in-band SNR, bus wrapper (A = 20 dB)."""
    s, n = speech_and_noise(snr_db=0.0)
    x = s + n
    s_h, x_h = calib.prefilter(s, "bus"), calib.prefilter(x, "bus")
    sd0, e0 = si_sdr(x_h, s_h), estoi(x_h, s_h)
    out = {}
    for name, (min_sd, min_e) in (("gtcrn", (3.0, 0.03)), ("dpdfnet", (6.0, 0.08))):
        y = WRAPPERS[name].enhance(stereo(x), FS, BUS)
        dsd, de = si_sdr(y, s_h) - sd0, estoi(y, s_h) - e0
        assert dsd > min_sd and de > min_e, f"{name}: dSI-SDR {dsd:.2f} dB, dESTOI {de:+.3f}"
        out[name] = f"dSI-SDR {dsd:+.2f} dB, dESTOI {de:+.3f} (input SI-SDR {sd0:.2f} dB, ESTOI {e0:.3f})"
    return out


# ------------------------------------------------------------------ attenuation limit and lab-grid mask
def test_attenuation_limit_and_mask():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 161)) + 1j * rng.standard_normal((50, 161))
    S = rng.standard_normal((50, 161)) + 1j * rng.standard_normal((50, 161))
    a = 10 ** (-20 / 20)
    assert np.allclose(nn.attenuation_limit(X, S, 20.0), a * X + (1 - a) * S)
    for off in (None, 0.0, np.inf):
        assert nn.attenuation_limit(X, S, off) is S
    assert np.allclose(nn.attenuation_limit(X, 0 * S, 12.0), X * 10 ** (-12 / 20))
    assert nn.floor_of(20.0) == a and nn.floor_of(None) == 0.0
    s, n = speech_and_noise(snr_db=0.0)
    x = s + n
    out = {}
    for name in nn.MODELS:
        # noise only: attenuation bounded by the limit (about A dB at most)
        yn = nn.denoise(n, name, atten_db=12.0)
        att = 20 * np.log10(rms(yn) / rms(n))
        assert -12.5 < att < 0, f"{name}: noise attenuation {att:.1f} dB with a 12 dB limit"
        pf = nn.post_filter(x, name, 20.0)
        X_lab = common.stft(x)
        assert pf["mask"].shape == X_lab.shape
        assert pf["mask"].min() >= a - 1e-12 and pf["mask"].max() <= 1.0
        fid = si_sdr(common.istft(pf["mask"] * X_lab, len(x)), pf["y"])
        assert fid > 20.0, f"{name}: mask reproduces the model output at only {fid:.1f} dB"
        pf2 = nn.post_filter_stft(X_lab, len(x), name, 20.0)
        assert np.abs(pf2["y"] - pf["y"]).max() < 1e-9 and np.abs(pf2["mask"] - pf["mask"]).max() < 1e-6
        assert np.array_equal(pf["y"], nn.denoise(x, name, atten_db=20.0))
        assert np.array_equal(pf["y_raw"], nn.denoise(x, name))
        assert pf["mask_raw"].min() >= 0.0 and pf["mask_raw"].max() <= 1.0
        # the mix form keeps more than the DESIGN floor form wherever the model attenuates
        assert np.mean(pf["mask"] >= np.maximum(pf["mask_raw"], a) - 0.05) > 0.9
        out[name] = f"noise-only attenuation {att:.1f} dB (A = 12); mask fidelity {fid:.1f} dB"
    return out


# ------------------------------------------------------------------ level invariance
def test_level_invariance():
    """level='file' makes M8 scale invariant; level=None (raw, as sherpa-onnx) is not."""
    s, n = speech_and_noise(snr_db=0.0)
    x = s + n
    out = {}
    for name in nn.MODELS:
        y = nn.denoise(x, name, atten_db=20.0)
        for c in (10 ** (-40 / 20), 10 ** (15 / 20)):
            yc = nn.denoise(c * x, name, atten_db=20.0) / c
            rel = np.abs(yc - y).max() / np.abs(y).max()
            assert rel < 1e-4, f"{name}: not scale invariant at {20 * np.log10(c):+.0f} dB ({rel:.1e})"
        y_raw = nn.denoise(x * 10 ** (-40 / 20), name, atten_db=20.0, level=None) * 10 ** (40 / 20)
        diff = np.abs(y_raw - y).max() / np.abs(y).max()
        assert diff > 1e-2, f"{name}: raw-level output unexpectedly identical"
        y_on = nn.denoise(x, name, atten_db=20.0, level="online")
        assert np.all(np.isfinite(y_on)) and si_sdr(y_on, s) > si_sdr(x, s) + 2.0
        out[name] = f"file mode rel. diff {rel:.1e} at +15 dB; raw mode differs by {diff:.2f}"
    return out


# ------------------------------------------------------------------ edge cases
def test_edge_cases():
    rng = np.random.default_rng(2)
    out = {}
    for name, mod in WRAPPERS.items():
        y = mod.enhance(np.zeros((2, 3 * FS), np.float32), FS, BUS)
        assert y.shape == (3 * FS,) and not np.any(y), f"{name}: silence in, non-zero out"
        for T in (0, 1, 7, 100, 1000):
            yt = mod.enhance(np.full((2, T), 0.01, np.float32), FS, BUS)
            assert yt.shape == (T,) and np.all(np.isfinite(yt)), f"{name}: length {T}"
        xu = rng.uniform(-1, 1, 5 * FS)                      # full-scale noise
        yu = mod.enhance(stereo(xu), FS, BUS)
        att = 20 * np.log10(rms(yu) / rms(calib.prefilter(xu, "bus")))
        assert np.all(np.isfinite(yu)) and -22.0 < att < -10.0 and np.abs(yu).max() < 1.0, \
            f"{name}: full-scale noise {att:.1f} dB, peak {np.abs(yu).max():.2f}"
        s, n = speech_and_noise(snr_db=5.0)                 # 12 dB over full scale, hard clipped
        k = 4.0 / np.abs(s + n).max()
        xc = np.clip((s + n) * k, -1.0, 1.0)
        yc = mod.enhance(stereo(xc), FS, BUS)
        s_h = calib.prefilter(s * k, "bus")
        e_in, e_out = estoi(calib.prefilter(xc, "bus"), s_h), estoi(yc, s_h)
        assert np.all(np.isfinite(yc)) and np.abs(yc).max() < 1.5 and e_out > e_in, \
            f"{name}: clipped input ESTOI {e_in:.3f} -> {e_out:.3f}, peak {np.abs(yc).max():.2f}"
        out[name] = (f"full-scale noise {att:.1f} dB; clipped input ({100 * np.mean(np.abs(xc) >= 1):.0f}% "
                     f"samples) ESTOI {e_in:.3f} -> {e_out:.3f}, peak {np.abs(yc).max():.2f}")
    return out


# ------------------------------------------------------------------ wrapper contract, determinism, RTF
def test_wrapper_contract():
    s, n = speech_and_noise(snr_db=0.0)
    x2 = stereo(s + n)
    out = {}
    for name, mod in WRAPPERS.items():
        y = mod.enhance(x2, FS, BUS)
        st = mod.enhance(x2, FS, BUS, return_stages=True)
        assert np.array_equal(st["out"], y) and np.array_equal(st["pre_agc"], y)
        y_no = mod.enhance(x2, FS, dict(BUS, m8_atten_db=None))
        y_m = mod.enhance(x2, FS, {"scenario": "meeting", "spacing_m": 0.04})
        ref = nn.denoise(calib.prefilter(x2[0], "meeting"), name, atten_db=12.0)
        assert np.array_equal(y_m, ref), f"{name}: meeting path is not HPF 80 Hz + A = 12 dB"
        r_bus, r_no, r_mtg = (20 * np.log10(rms(v) / rms(x2[0])) for v in (y, y_no, y_m))
        assert r_no < r_bus < r_mtg, f"{name}: output levels no-limit {r_no:.1f} / A20 {r_bus:.1f} / A12 {r_mtg:.1f} dB"
        out[name] = f"output/input level: no limit {r_no:.1f} dB, A20 {r_bus:.1f} dB, meeting A12 {r_mtg:.1f} dB"
    return out


def test_determinism_and_rtf():
    """B.8: identical output for identical input; RTF < 0.5 (single process, one thread).
    CPU-time RTF is always asserted; wall-time RTF only when the machine is not overloaded
    (1-min load average below the core count), since other jobs inflate wall time."""
    s, n = speech_and_noise(snr_db=0.0)
    x = np.tile(s + n, 3)[: 30 * FS]
    load = os.getloadavg()[0]
    busy = load >= (os.cpu_count() or 1)
    out = {}
    for name, mod in WRAPPERS.items():
        mod.enhance(stereo(x[:FS]), FS, BUS)                 # load the session outside the timing
        t0, c0 = time.perf_counter(), time.process_time()
        y1 = mod.enhance(stereo(x), FS, BUS)
        rtf, rtf_cpu = (time.perf_counter() - t0) / 30.0, (time.process_time() - c0) / 30.0
        y2 = mod.enhance(stereo(x).copy(), FS, BUS)
        assert np.array_equal(y1, y2), f"{name}: non-deterministic output"
        assert rtf_cpu < 0.5, f"{name}: CPU-time RTF {rtf_cpu:.3f}"
        assert busy or rtf < 0.5, f"{name}: RTF {rtf:.3f} (load average {load:.1f})"
        out[name] = (f"RTF {rtf:.3f} wall, {rtf_cpu:.3f} CPU (30 s input; load average {load:.1f}"
                     + (", wall time not asserted)" if busy else ")"))
    return out


TESTS = [test_graph_and_sessions, test_stft_definition, test_delay_and_alignment, test_enhancement_gain,
         test_attenuation_limit_and_mask, test_level_invariance, test_edge_cases, test_wrapper_contract,
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
