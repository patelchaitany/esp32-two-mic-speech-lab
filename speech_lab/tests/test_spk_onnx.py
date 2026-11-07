"""Tests for methods/_core/spk_onnx.py (WeSpeaker ResNet34 speaker embeddings + numpy Kaldi fbank).

Run from LAB: .venv/bin/python -m tests.test_spk_onnx   (plain asserts, about 10-20 s)
Checks: mel banks and fbank against a literal per-frame re-implementation of kaldi-native-fbank
(both the WeSpeaker and the sherpa-onnx front ends), framing counts, a tone's mel peak, embedding
shape / norm / determinism / gain invariance, batching, short and resampled inputs, speaker
discrimination on LibriSpeech test-clean (skipped if the corpus is absent), the S1 hook contract
(spk.CosineClusterer and tracker.front_end(..., embedder=...)), and RTF of a 1.5 s embedding.
"""
import glob
import math
import os
import time

import numpy as np

from methods._core import spk_onnx as so

LAB = so.LAB
LIBRI = os.path.join(LAB, "data", "librispeech_test_clean", "LibriSpeech", "test-clean")


# ------------------------------------------------------------------ literal Kaldi reference
def _kaldi_mel_literal(n_mels, n_fft, fs, low_freq, high_freq):
    """knf MelBanks::InitKaldiMelBanks, written as the C++ loops (no vectorisation)."""
    nyq = 0.5 * fs
    hi = high_freq if high_freq > 0 else nyq + high_freq
    mel = lambda f: 1127.0 * math.log(1.0 + f / 700.0)  # noqa: E731
    m_lo, m_hi = mel(low_freq), mel(hi)
    delta = (m_hi - m_lo) / (n_mels + 1)
    W = np.zeros((n_mels, n_fft // 2 + 1))
    for b in range(n_mels):
        left, center, right = m_lo + b * delta, m_lo + (b + 1) * delta, m_lo + (b + 2) * delta
        for i in range(n_fft // 2):
            m = mel(fs / n_fft * i)
            if left < m < right:
                W[b, i] = (m - left) / (center - left) if m <= center else (right - m) / (right - center)
    return W


def _kaldi_fbank_literal(y, window, high_freq, snip_edges, cmn, fs=16000):
    """Per-frame knf ExtractWindow + ProcessWindow + FbankComputer::Compute (use_energy=false)."""
    wave = np.asarray(y, dtype=np.float64) * 32768.0
    T, flen, shift, n_fft = len(wave), 400, 160, 512
    n = (0 if T < flen else 1 + (T - flen) // shift) if snip_edges else (T + shift // 2) // shift
    a = 2 * math.pi / (flen - 1)
    if window == "hamming":
        win = np.array([0.54 - 0.46 * math.cos(a * i) for i in range(flen)])
    else:
        win = np.array([(0.5 - 0.5 * math.cos(a * i)) ** 0.85 for i in range(flen)])
    W = _kaldi_mel_literal(80, n_fft, fs, 20.0, high_freq)
    out = []
    for f in range(n):
        start = f * shift if snip_edges else f * shift + shift // 2 - flen // 2
        frame = np.zeros(flen)
        for s in range(flen):
            k = start + s
            while k < 0 or k >= T:
                k = -k - 1 if k < 0 else 2 * T - 1 - k
            frame[s] = wave[k]
        frame -= frame.mean()
        for i in range(flen - 1, 0, -1):
            frame[i] -= 0.97 * frame[i - 1]
        frame[0] -= 0.97 * frame[0]
        frame *= win
        padded = np.zeros(n_fft)
        padded[:flen] = frame
        X = np.fft.rfft(padded)
        power = X.real ** 2 + X.imag ** 2
        out.append([math.log(max(float(W[b] @ power), so.FLT_EPS)) for b in range(80)])
    out = np.array(out)
    if cmn:
        out -= out.mean(axis=0)
    return out


def _speech_like(seed=0, dur=2.0, fs=16000):
    """Deterministic noise-excited harmonic signal with an envelope (no corpus needed)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / fs
    y = sum(np.sin(k * ph) / k for k in range(1, 30))
    y = y * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2) + 0.05 * rng.standard_normal(len(t))
    return 0.1 * y / np.max(np.abs(y))


def _libri_speakers(n_spk=4, n_utt=3, dur=3.0):
    """{speaker: [dur-second segments starting 1 s into utterances >= dur + 2 s]} from fixed test-clean
    speakers (237 F, 1284 F, 260 M, 908 M: test half of the build:spk split), or None if absent."""
    spks = ["237", "1284", "260", "908"][:n_spk]
    if not all(os.path.isdir(os.path.join(LIBRI, s)) for s in spks):
        return None
    import soundfile as sf
    out = {}
    for s in spks:
        files = sorted(glob.glob(os.path.join(LIBRI, s, "*", "*.flac")))
        segs = []
        for f in files:
            y, fs = sf.read(f, dtype="float64")
            if len(y) >= (dur + 2) * fs:
                segs.append(y[fs:fs + int(dur * fs)])
            if len(segs) == n_utt:
                break
        out[s] = segs
    return out


# ------------------------------------------------------------------ tests
def test_mel_banks():
    for hf in (0.0, -400.0):
        W = so.mel_banks(80, 512, 16000, 20.0, hf)
        ref = _kaldi_mel_literal(80, 512, 16000, 20.0, hf)
        assert W.shape == (80, 257)
        assert np.max(np.abs(W - ref)) < 1e-12, np.max(np.abs(W - ref))
        assert np.all(W[:, -1] == 0), "Kaldi ignores the Nyquist bin"
        assert np.all(W.max(axis=1) > 0) and np.all(W <= 1.0)
        f = np.arange(257) * 16000 / 512
        assert np.all(W[:, f <= 20.0] == 0), "lowest filter starts at 20 Hz"
        top = 8000.0 if hf == 0 else 7600.0
        assert np.all(W[:, f >= top] == 0)
        peaks = np.argmax(W, axis=1)
        assert np.all(np.diff(peaks) >= 0)


def test_frame_counts():
    for T in (399, 400, 401, 559, 560, 24000, 24064, 48000):
        assert so.num_frames(T, True) == (0 if T < 400 else 1 + (T - 400) // 160)
        assert so.num_frames(T, False) == (T + 80) // 160
        y = np.random.default_rng(T).standard_normal(T) * 0.01
        assert so.fbank(y, snip_edges=True).shape == (so.num_frames(T, True), 80)
        assert so.fbank(y, snip_edges=False).shape == (so.num_frames(T, False), 80)


def test_fbank_matches_literal_kaldi():
    y = _speech_like(1, 0.83)          # odd length: exercises the edge frames
    for name in ("wespeaker", "sherpa", "sherpa_cmn"):
        kw = so.VARIANTS[name]
        got = so.fbank(y, **kw)
        ref = _kaldi_fbank_literal(y, kw["window"], kw["high_freq"], kw["snip_edges"], kw["cmn"])
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        err = np.max(np.abs(got - ref))
        assert err < 2e-4, (name, err)   # float32 output rounding only


def test_fbank_tone_and_scaling():
    fs = 16000
    t = np.arange(fs) / fs
    y = 0.5 * np.sin(2 * np.pi * 1000 * t)
    F = so.fbank(y, cmn=False)
    W = so.mel_banks()
    centers = (np.arange(257) * fs / 512)[np.argmax(W, axis=1)]
    k = int(np.argmax(F.mean(axis=0)))
    assert abs(centers[k] - 1000) <= 2 * fs / 512, (k, centers[k])
    # without CMN the log-mel shifts by 2 ln(gain); with CMN it is gain invariant
    F2 = so.fbank(0.25 * y, cmn=False)
    assert np.allclose(F2[:, k], F[:, k] + 2 * np.log(0.25), atol=1e-3)
    assert np.allclose(so.fbank(0.25 * y), so.fbank(y), atol=1e-3)


def test_embedding_basic():
    y = _speech_like(2, 1.5)
    e1, e2 = so.embed(y), so.embed(y)
    assert e1.shape == (so.EMB_DIM,) and e1.dtype == np.float64
    assert abs(np.linalg.norm(e1) - 1) < 1e-9
    assert np.array_equal(e1, e2), "deterministic"
    assert so.cosine(so.embed(0.05 * y), e1) > 0.9999, "gain invariant (CMN)"
    E = so.get_embedder()
    assert so.get_embedder() is E, "cached embedder"
    assert E.sess.get_inputs()[0].shape[2] == 80


def test_segments_and_batching():
    y = _speech_like(3, 3.0)
    segs = [(0, 24000), (16000, 40000), (8000, 32000), (4000, 30000)]
    B = so.embed_segments(y, segs)
    S = np.array([so.embed(y[a:b]) for a, b in segs])
    assert B.shape == (4, so.EMB_DIM)
    assert np.max(np.abs(B - S)) < 1e-5, np.max(np.abs(B - S))
    Bs = so.embed_segments(y, [(a / 16000, b / 16000) for a, b in segs], unit="s")
    assert np.max(np.abs(Bs - B)) < 1e-9
    assert np.allclose(np.linalg.norm(B, axis=1), 1.0)


def test_short_resampled_and_errors():
    y = _speech_like(4, 1.5)
    e = so.embed(y[:1600])               # 0.1 s: tiled to MIN_S, finite and unit norm
    assert np.all(np.isfinite(e)) and abs(np.linalg.norm(e) - 1) < 1e-9
    assert np.all(np.isfinite(so.embed(np.zeros(24000)))), "digital silence must not give NaN"
    y8 = y[::2]                          # crude 8 kHz version (aliasing irrelevant here)
    e8 = so.embed(y8, fs=8000)
    assert e8.shape == (so.EMB_DIM,) and np.all(np.isfinite(e8))
    assert np.array_equal(so.embed(y[None, :]), so.embed(y)), "(1, T) is accepted as mono"
    for bad in (lambda: so.embed(np.zeros(0)), lambda: so.WeSpeakerEmbedder(variant="nope"),
                lambda: so.embed_segments(y, [(100, 100)]), lambda: so.embed(np.stack([y, y])),
                lambda: so.embed_segments(y, [(0, 100)], unit="ms")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_cosine_and_centroid():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(8), rng.standard_normal(8)
    tol = 1e-9                           # _unit() divides by (norm + 1e-12)
    assert abs(so.cosine(a, a) - 1) < tol and abs(so.cosine(a, -a) + 1) < tol
    M = rng.standard_normal((5, 8))
    assert so.cosine(M, a).shape == (5,) and so.cosine(M, M).shape == (5, 5)
    assert abs(so.cosine(M, b)[2] - so.cosine(M[2], b)) < tol
    c = so.centroid(np.stack([a, a]), weights=[1, 3])
    assert abs(so.cosine(c, a) - 1) < tol and abs(np.linalg.norm(c) - 1) < tol


def test_speaker_discrimination():
    data = _libri_speakers()
    if data is None:
        print("  (LibriSpeech test-clean absent: skipped)")
        return
    E = {s: so.get_embedder().embed_batch(v) for s, v in data.items()}
    same = [so.cosine(e[i], e[j]) for e in E.values() for i in range(3) for j in range(i + 1, 3)]
    keys = list(E)
    diff = [so.cosine(E[a][i], E[b][j]) for x, a in enumerate(keys) for b in keys[x + 1:]
            for i in range(3) for j in range(3)]
    print(f"  same-speaker cos {np.mean(same):.3f} (min {np.min(same):.3f}), "
          f"different {np.mean(diff):.3f} (max {np.max(diff):.3f})")
    assert np.min(same) > np.max(diff), "3 s clean segments of 4 speakers must separate"
    assert np.mean(same) - np.mean(diff) > 0.4
    # the S1 clusterer at THETA_S groups the 12 segments into the 4 speakers
    from methods._core import spk
    cl = spk.CosineClusterer(thr=so.THETA_S)
    lab = {s: [cl.assign(e) for e in E[s]] for s in keys}
    assert all(len(set(v)) == 1 for v in lab.values()), lab
    assert len({v[0] for v in lab.values()}) == len(keys), lab


def test_s1_hook_end_to_end():
    """tracker.front_end(..., embedder=WeSpeakerEmbedder(), cos_thr=THETA_S) on a synthetic 2-mic meeting:
    6 s turns A B A C B separated by 0.6 s pauses, each talker at its own integer-sample TDOA.
    (The hook embeds complete 1.5 s chunks of confident speech per VAD region; a region without one keeps
    the previous label, so turns must be long enough for the per-turn majority to be the talker's own.)"""
    data = _libri_speakers(n_spk=3, n_utt=2, dur=6.0)
    if data is None:
        print("  (LibriSpeech test-clean absent: skipped)")
        return
    from methods._core import tracker
    spks = list(data)
    plan = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]        # (talker, utterance)
    lag = {0: 0, 1: 2, 2: -2}
    fs, gap = 16000, np.zeros(int(0.6 * 16000))
    rng = np.random.default_rng(0)
    a_parts, b_parts, spans, t = [gap], [gap], [], len(gap)
    for k, u in plan:
        y = data[spks[k]][u] / np.std(data[spks[k]][u]) * 0.05
        a_parts += [y, gap]
        b_parts += [np.roll(y, lag[k]), gap]
        spans.append((k, t, t + len(y)))
        t += len(y) + len(gap)
    x = np.stack([np.concatenate(a_parts), np.concatenate(b_parts)])
    x = x + 1e-4 * rng.standard_normal(x.shape)
    fe = tracker.front_end(x, fs, {"scenario": "meeting", "spacing_m": 0.04}, embedder=so.WeSpeakerEmbedder(),
                           cos_thr=so.THETA_S)
    lab = fe.track.cluster
    centers = np.arange(len(lab)) * 128 - 256
    maj = []
    for k, a, b in spans:
        v = lab[(centers >= a) & (centers < b)]
        v = v[v >= 0]
        assert len(v), "speech segment without a cluster"
        maj.append(int(np.bincount(v).argmax()))
    print(f"  turn labels {maj} for talkers {[k for k, _ in plan]}")
    assert maj[0] == maj[2] and maj[1] == maj[4], maj
    assert len({maj[0], maj[1], maj[3]}) == 3, maj


def test_rtf():
    y = _speech_like(5, 1.5)
    E = so.get_embedder()
    E.embed(y)
    t = []
    for _ in range(10):
        t0 = time.perf_counter()
        E.embed(y)
        t.append(time.perf_counter() - t0)
    rtf = float(np.median(t)) / 1.5
    print(f"  1.5 s embedding: median {1e3 * np.median(t):.1f} ms, RTF {rtf:.3f} (1 thread)")
    assert rtf < 0.5


if __name__ == "__main__":
    t_all = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            fn()
            print(f"ok {name} ({time.time() - t0:.1f} s)")
    print(f"all spk_onnx tests passed ({time.time() - t_all:.1f} s)")
