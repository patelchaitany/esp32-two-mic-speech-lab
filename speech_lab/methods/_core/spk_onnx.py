"""Speaker embeddings for S1 talker identity (DESIGN.md 2.0): WeSpeaker ResNet34 ONNX + numpy Kaldi fbank.

Model: models/wespeaker/wespeaker_en_voxceleb_resnet34.onnx (sherpa-onnx release of WeSpeaker's
VoxCeleb ResNet34-TSTP-emb256; SHA-256 5ef208a9...fe94; 26.5 MB). ONNX I/O: input `feats` (B, T, 80)
float32 log-mel fbank, output `embs` (B, 256). metadata_props: framework=wespeaker, sample_rate=16000,
output_dim=256, normalize_samples=0 (the model expects int16-scaled samples, i.e. float x 32768) and
no feature_normalize_type. The graph has no feature normalisation of its own (it starts with
Transpose -> Unsqueeze -> Conv) and ends with `embs - mean_vec`, where mean_vec is all zeros.

Front end (default VARIANT 'wespeaker' = the model's training front end): samples x 32768, Kaldi fbank
with 25 ms Hamming frames, 10 ms shift, snip_edges=True, DC removal, pre-emphasis 0.97 (Kaldi edge rule
d[0] -= 0.97 d[0]), no dither, 512-point FFT power spectrum, 80 triangular mel bins on the Kaldi mel scale
1127 ln(1 + f/700) from 20 Hz to Nyquist, natural log floored at FLT_EPSILON, then per-utterance mean
normalisation over time (CMN, no variance normalisation).
Sources (Apache-2.0; re-implemented in numpy, no code copied verbatim):
  - wenet-e2e/wespeaker wespeaker/dataset/processor.py compute_fbank / apply_cmvn (training: x (1 << 15),
    window_type='hamming', use_energy=False, CMN mean only), wespeaker/cli/speaker.py compute_features
    (inference: hamming + CMN), examples/voxceleb/v2/conf/resnet.yaml (80 bins, 25/10 ms);
  - torchaudio.compliance.kaldi.fbank defaults used there (dither 0 at inference, high_freq 0 = Nyquist,
    low_freq 20, snip_edges True, preemphasis 0.97, remove_dc_offset True, round_to_power_of_two True);
  - k2-fsa/sherpa-onnx csrc/features.{h,cc}, speaker-embedding-extractor-general-impl.h and
    csrc/speaker-embedding-extractor-model.cc, and csukuangfj/kaldi-native-fbank csrc/feature-window.cc,
    mel-computations.cc, feature-fbank.cc (framing, Povey window, mel banks, log floor).
sherpa-onnx itself feeds this model a different front end (VARIANT 'sherpa'): Povey window,
snip_edges=False (centred frames, symmetric reflection at the edges), mel bins 20-7600 Hz and NO mean
normalisation (feature_normalize_type is absent from this model's metadata). Measured below, that costs
4-5x in EER; a sherpa-onnx run is therefore not a valid numeric reference for this module.

API (all signals 1-D float at full scale 1.0; 16 kHz, other rates are resampled with resample_poly):
  embed(signal, fs=16000) -> (256,) unit-norm float64 embedding (module-level default embedder)
  embed_segments(signal, segments, fs=16000, unit='samples') -> (n, 256) unit-norm rows
  cosine(a, b) -> cosine similarity: scalar for two vectors, (n,) or (n, m) for matrices
  centroid(E, weights=None) -> unit-norm (weighted) mean embedding
  WeSpeakerEmbedder(path=MODEL_PATH, threads=1, variant=VARIANT): callable y -> unit-norm embedding,
      i.e. the S1 hook of methods/_core/spk.py:
      `tracker.front_end(x, fs, info, embedder=WeSpeakerEmbedder(), cos_thr=THETA_S)` (front_end passes
      x = prefiltered Mic A to the hook); also .embed(y, fs), .embed_batch(signals, fs),
      .embed_segments(y, segments, fs, unit), .features(y, fs)
  get_embedder(path, threads, variant) -> cached WeSpeakerEmbedder (one ONNX session per process)
  fbank(y, fs, window, high_freq, snip_edges, cmn, ...) -> (frames, 80) float32; mel_banks(...)
  THETA_S = 0.375: recommended S1 clustering threshold (cos_thr of spk.cluster_talkers / tracker.track) for
      far-field meetings; DESIGN 2.0's starting value 0.6 over-splits with this model (see THETA_S comment).

Inputs shorter than MIN_S are repeated (tiled) up to MIN_S so the ResNet's time-pooling std has at least
two frames; embeddings of less than about 1 s of speech are unreliable. onnxruntime runs with
intra_op_num_threads=1 and is deterministic. Cost (M1 Pro, 1 thread, machine load average 12-15 on 10
cores): 93 ms per 1.5 s embedding (fbank 0.7 ms, ONNX 91 ms) = RTF 0.062 (159 ms = RTF 0.106 seen during a
load spike to 31); 3 s: RTF 0.056-0.062; batching 16 segments gains ~5 %. Session load 0.05 s.
Evidence and reproduction scripts: tmp/build:spk/ (README.md there).

Validation (LibriSpeech test-clean, 40 speakers split by gender-sorted id: even -> dev (threshold, variant
and theta choice), odd -> test (report); 16 utterances per speaker, one 3 s window each with >= 80 % active
10 ms frames, 1.5 s = its first half; all pairs: 2400 target / 48640 non-target trials per cell; far field
= pyroomacoustics shoebox, measured RT60 0.60-0.91 s, 1.1-5.0 m, DRR -11...-0.6 dB, a random room per
segment; DEMAND ch01-16 at 5-15 dB SNR, dev from the first, test from the second half of each recording).
  Front end, dev EER % 1.5 s / 3 s: clean | reverb | reverb + OMEETING
    wespeaker (default)          2.33 / 0.54 |  7.37 / 2.42  | 11.62 / 5.04
    sherpa (sherpa-onnx, no CMN) 10.75 / 5.50 | 30.88 / 25.00 | 37.97 / 32.21
    sherpa_cmn                   2.34 / 0.67 |  7.50 / 2.54  | 11.30 / 5.09
    wespeaker_nocmn             10.33 / 4.91 | 30.71 / 24.46 | 37.29 / 32.25
    povey window + CMN           2.37 / 0.58 |  7.71 / 2.46  | 11.80 / 5.08
    hamming, 7600 Hz top + CMN   2.67 / 0.62 |  7.42 / 2.41  | 11.30 / 5.17
    spk.fbank (spk.py hook)      2.71 / 0.58 |  7.37 / 2.42  | 11.42 / 5.21
  Mean normalisation is what matters; window type and 7600 vs 8000 Hz are within noise (+-0.3 %).
  Test EER % 1.5 s / 3 s [cosine threshold at the dev EER point]: clean 1.83 / 0.71 [0.30 / 0.39];
  reverb 5.67 / 2.05 [0.23 / 0.34]; OMEETING 3.33 / 1.08 [0.27 / 0.36]; PCAFETER 3.25 / 1.25 [0.27 / 0.38];
  reverb + OMEETING 9.38 / 3.71 [0.20 / 0.31]; reverb + PCAFETER 10.32 / 4.45 [0.21 / 0.32].
  The meeting prefilter (80 Hz HPF) changes embeddings by cos 0.997-0.999 and EER by <= 0.1 %.
  Subtracting a dev mean embedding lowers test EER by 0.0-0.5 % only (not used).
"""
import functools
import os
from math import gcd

import numpy as np

FS = 16000
EMB_DIM = 256
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(LAB, "models", "wespeaker", "wespeaker_en_voxceleb_resnet34.onnx")
FLT_EPS = float(np.finfo(np.float32).eps)   # Kaldi / knf / torchaudio log floor
SAMPLE_SCALE = 32768.0                       # normalize_samples=0: model expects int16-range samples
MIN_S = 0.25                                 # shorter inputs are tiled up to this length

# Front-end variants (keyword arguments of fbank()). 'wespeaker' is the model's training/inference front end;
# 'sherpa' is what sherpa-onnx feeds this model; the other two isolate single differences.
VARIANTS = {
    "wespeaker": dict(window="hamming", high_freq=0.0, snip_edges=True, cmn=True),
    "sherpa": dict(window="povey", high_freq=-400.0, snip_edges=False, cmn=False),
    "sherpa_cmn": dict(window="povey", high_freq=-400.0, snip_edges=False, cmn=True),
    "wespeaker_nocmn": dict(window="hamming", high_freq=0.0, snip_edges=True, cmn=False),
}
VARIANT = "wespeaker"

# Recommended S1 clustering threshold theta_s = spk.cluster_talkers(cos_thr=...) / tracker.track(cos_thr=...)
# (cosine of a 1.5 s key to the cluster centroid, spk.CosineClusterer, at most 30 clusters).
# Chosen on dev sequences of a simulated 20-talker meeting (8 per split: one 8-12 x 6-9 m room, measured
# RT60 0.63-0.87 s, talkers at fixed seats 1-5.8 m from the device (median 2.4 m, 17 % beyond 4 m, 3 %
# beyond 5 m), 4 turns each in random order = 80 turns of 2.5-8 s, DEMAND OMEETING or PCAFETER at a fixed
# level, median turn SNR 10 dB, 5th-95th percentile 4-16 dB); F = harmonic mean of
# purity and coverage (inverse purity). Keys as in spk._emb_labels: every complete 1.5 s chunk of active
# speech is assigned on its own (167 chunks per sequence). Dev optimum 0.35 (OMEETING) / 0.375 (PCAFETER);
# 0.375 is within 0.017 dev F of both; clean close-talk at 0.375: dev F 0.962, test 0.982 (optimum 0.425).
# Test [purity / coverage / clusters for 20 talkers], theta 0.375 vs DESIGN's 0.6:
#   far OMEETING 0.907 / 0.849 / 30 vs 0.784 / 0.779 / 30; far PCAFETER 0.912 / 0.845 / 30 vs
#   0.780 / 0.768 / 30; clean 0.981 / 0.983 / 20.9 vs 0.958 / 0.900 / 30.
# Other keys (same simulation): first 1.5 s of each turn only -> optimum 0.375 (test OMEETING
# 0.927 / 0.806 / 30); mean of a turn's chunks -> 0.425-0.475 (0.964 / 0.850 / 29.8 at 0.475).
# Far-field runs reach the 30-cluster cap at theta >= 0.375 (29-30 clusters at 0.35): some talkers split
# into 2 clusters, and chunks after the cap are forced into the nearest cluster; at 0.6 that over-splitting
# is what costs purity. Splits grow with distance (test, OMEETING, theta 0.375): per-talker share of chunks
# in its main cluster 0.92 at 1-2 m, 0.83 at 2-3 m, 0.74-0.78 at 3-5 m; talkers spread over >= 2 clusters
# 11 % at 1-2 m, 33-35 % at 3-5 m.
THETA_S = 0.375


# ------------------------------------------------------------------ Kaldi-compatible fbank
@functools.lru_cache(maxsize=16)
def feature_window(kind, n):
    """Kaldi window of length n (symmetric, a = 2 pi / (n - 1)): 'hamming', 'povey' (hann^0.85),
    'hanning', 'rectangular'."""
    i = np.arange(n, dtype=np.float64)
    a = 2.0 * np.pi / (n - 1)
    if kind == "hamming":
        w = 0.54 - 0.46 * np.cos(a * i)
    elif kind == "povey":
        w = (0.5 - 0.5 * np.cos(a * i)) ** 0.85
    elif kind == "hanning":
        w = 0.5 - 0.5 * np.cos(a * i)
    elif kind == "rectangular":
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {kind!r}")
    w.flags.writeable = False
    return w


def mel_scale(f):
    """Kaldi mel scale 1127 ln(1 + f / 700)."""
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


@functools.lru_cache(maxsize=16)
def mel_banks(n_mels=80, n_fft=512, fs=FS, low_freq=20.0, high_freq=0.0):
    """(n_mels, n_fft // 2 + 1) Kaldi triangular mel filters (knf InitKaldiMelBanks, no VTLN).
    high_freq <= 0 is an offset from Nyquist (0 = Nyquist, -400 = 7600 Hz at 16 kHz). Bin k (frequency
    k fs / n_fft) has weight min(up, down) where the bin's mel lies strictly inside (left, right); Kaldi only
    uses bins 0 .. n_fft/2 - 1, so the Nyquist column is zero."""
    nyq = 0.5 * fs
    hi = high_freq if high_freq > 0 else nyq + high_freq
    if not (0.0 <= low_freq < nyq and 0.0 < hi <= nyq and hi > low_freq):
        raise ValueError(f"bad mel range {low_freq}..{hi} Hz")
    m_lo, m_hi = mel_scale(low_freq), mel_scale(hi)
    delta = (m_hi - m_lo) / (n_mels + 1)
    b = np.arange(n_mels, dtype=np.float64)[:, None]
    left, center, right = m_lo + b * delta, m_lo + (b + 1) * delta, m_lo + (b + 2) * delta
    m = mel_scale(fs / n_fft * np.arange(n_fft // 2))[None, :]
    w = np.maximum(0.0, np.minimum((m - left) / (center - left), (right - m) / (right - center)))
    w = np.concatenate([w, np.zeros((n_mels, 1))], axis=1)
    w.flags.writeable = False
    return w


def _mono(y):
    """float64 1-D view of a signal; (1, T) / (T, 1) are accepted, a 2-channel array is an error
    (pass one microphone, e.g. x[0], not the (2, T) array)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        if y.ndim == 2 and 1 in y.shape:
            return y.reshape(-1)
        raise ValueError(f"expected a 1-D signal, got shape {y.shape}")
    return y


def num_frames(n_samples, snip_edges=True, frame_len=400, shift=160):
    """Kaldi NumFrames (flush=True)."""
    if snip_edges:
        return 0 if n_samples < frame_len else 1 + (n_samples - frame_len) // shift
    return (n_samples + shift // 2) // shift


def _frames(y, frame_len, shift, snip_edges):
    """(n, frame_len) frames as Kaldi ExtractWindow: snip_edges=True -> frame f starts at f*shift;
    False -> frame f is centred on f*shift + shift/2, with symmetric reflection past both ends."""
    n = num_frames(len(y), snip_edges, frame_len, shift)
    if n == 0:
        return np.zeros((0, frame_len))
    start = np.arange(n) * shift
    if not snip_edges:
        start = start + shift // 2 - frame_len // 2
    idx = start[:, None] + np.arange(frame_len)[None, :]
    if idx.min() < 0 or idx.max() >= len(y):     # reflection (-1 -> 0, -2 -> 1, T -> T-1, T+1 -> T-2)
        T = len(y)
        idx = np.mod(idx, 2 * T)
        idx = np.where(idx >= T, 2 * T - 1 - idx, idx)
    return y[idx]


def fbank(y, fs=FS, window="hamming", high_freq=0.0, snip_edges=True, cmn=True, low_freq=20.0,
          n_mels=80, preemph=0.97, remove_dc=True, scale=SAMPLE_SCALE, frame_ms=25.0, shift_ms=10.0):
    """Kaldi-compatible log-mel filterbank, (frames, n_mels) float32 (no dither, no energy term).
    Per frame: DC removal, pre-emphasis (d[i] -= p d[i-1], d[0] -= p d[0]), window, zero-pad to the next
    power of two, |rfft|^2, mel banks, log(max(E, FLT_EPSILON)); then optional mean normalisation over
    frames (cmn). y is float at full scale 1.0 and is multiplied by `scale` first (32768: int16 range)."""
    y = _mono(y) * scale
    flen, shift = int(fs * 0.001 * frame_ms), int(fs * 0.001 * shift_ms)
    n_fft = 1 << int(np.ceil(np.log2(flen)))
    fr = _frames(y, flen, shift, snip_edges)
    if remove_dc:
        fr = fr - fr.mean(axis=1, keepdims=True)
    if preemph:
        fr = np.concatenate([fr[:, :1] * (1.0 - preemph), fr[:, 1:] - preemph * fr[:, :-1]], axis=1)
    spec = np.fft.rfft(fr * feature_window(window, flen), n_fft, axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    feats = np.log(np.maximum(power @ mel_banks(n_mels, n_fft, fs, low_freq, high_freq).T, FLT_EPS))
    if cmn and len(feats):
        feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


# ------------------------------------------------------------------ embedder
@functools.lru_cache(maxsize=4)
def _session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _unit(e):
    e = np.asarray(e, dtype=np.float64)
    return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12)


def _resample(y, fs):
    if fs == FS:
        return y
    from scipy.signal import resample_poly
    g = gcd(int(fs), FS)
    return resample_poly(y, FS // g, int(fs) // g)


class WeSpeakerEmbedder:
    """Callable speaker-embedding hook: WeSpeakerEmbedder()(y) -> (256,) unit-norm float64 embedding.
    y: 1-D float signal at 16 kHz, full scale 1.0 (any gain; CMN makes the features gain invariant).
    variant: key of VARIANTS (front end); threads: onnxruntime intra-op threads (1 for the bench)."""

    def __init__(self, path=MODEL_PATH, threads=1, variant=VARIANT, max_batch=16):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.path, self.threads, self.variant, self.max_batch = path, threads, variant, max_batch
        self.feat_kw = dict(VARIANTS[variant])
        self.sess = _session(os.path.abspath(path), threads)
        self.inp = self.sess.get_inputs()[0].name
        self.dim = int(self.sess.get_outputs()[0].shape[1])

    def features(self, y, fs=FS):
        """(frames, 80) float32 model input for one signal (resampled to 16 kHz, tiled up to MIN_S)."""
        y = _resample(_mono(y), fs)
        n_min = int(round(MIN_S * FS))
        if len(y) == 0:
            raise ValueError("empty signal")
        if len(y) < n_min:
            y = np.tile(y, -(-n_min // len(y)))[:n_min]
        return fbank(y, FS, **self.feat_kw)

    def _run(self, feats):
        """Raw (n, 256) model outputs for a list of feature matrices (equal lengths are batched)."""
        out = np.zeros((len(feats), self.dim))
        groups = {}
        for i, f in enumerate(feats):
            groups.setdefault(f.shape[0], []).append(i)
        for idx in groups.values():
            for k in range(0, len(idx), self.max_batch):
                sel = idx[k:k + self.max_batch]
                batch = np.stack([feats[i] for i in sel]).astype(np.float32)
                out[sel] = self.sess.run(None, {self.inp: batch})[0]
        return out

    def embed(self, y, fs=FS):
        """(256,) unit-norm embedding of one signal."""
        return _unit(self._run([self.features(y, fs)])[0])

    __call__ = embed

    def embed_batch(self, signals, fs=FS):
        """(n, 256) unit-norm embeddings of a list of signals (equal lengths share one ONNX call)."""
        if not len(signals):
            return np.zeros((0, self.dim))
        return _unit(self._run([self.features(s, fs) for s in signals]))

    def embed_segments(self, y, segments, fs=FS, unit="samples"):
        """(n, 256) unit-norm embeddings of y[a:b] for each (a, b) in segments.
        unit='samples' (integer sample indices at rate fs) or 's' (seconds)."""
        if unit not in ("samples", "s"):
            raise ValueError("unit must be 'samples' or 's'")
        y = _mono(y)
        k = fs if unit == "s" else 1
        sig = []
        for a, b in segments:
            a, b = int(round(a * k)), int(round(b * k))
            a, b = max(0, a), min(len(y), b)
            if b <= a:
                raise ValueError(f"empty segment ({a}, {b})")
            sig.append(y[a:b])
        return self.embed_batch(sig, fs)


@functools.lru_cache(maxsize=4)
def get_embedder(path=MODEL_PATH, threads=1, variant=VARIANT):
    """Cached WeSpeakerEmbedder (one ONNX session per process and configuration)."""
    return WeSpeakerEmbedder(path, threads, variant)


def embed(signal, fs=FS):
    """(256,) unit-norm WeSpeaker embedding of a 1-D signal (default embedder)."""
    return get_embedder().embed(signal, fs)


def embed_segments(signal, segments, fs=FS, unit="samples"):
    """(n, 256) unit-norm embeddings of signal[a:b] for (a, b) in segments (default embedder)."""
    return get_embedder().embed_segments(signal, segments, fs, unit)


# ------------------------------------------------------------------ scoring helpers
def cosine(a, b):
    """Cosine similarity. Two vectors -> float; (n, d) and (d,) -> (n,); (n, d) and (m, d) -> (n, m).
    Inputs need not be normalised."""
    a, b = _unit(a), _unit(b)
    if a.ndim == 1 and b.ndim == 1:
        return float(a @ b)
    if b.ndim == 1:
        return a @ b
    if a.ndim == 1:
        return b @ a
    return a @ b.T


def centroid(E, weights=None):
    """Unit-norm (weighted) mean of unit-norm embeddings E (n, d): the S1 cluster key."""
    E = _unit(np.atleast_2d(E))
    w = np.ones(len(E)) if weights is None else np.asarray(weights, dtype=np.float64)
    return _unit((w[:, None] * E).sum(axis=0))
