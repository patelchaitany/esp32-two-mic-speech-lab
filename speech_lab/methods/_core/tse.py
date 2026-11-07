"""M9 core: enrollment-based target-speaker extraction (TSE) with VoxLock, run once per bus partner
(DESIGN.md 3 M9; 2.1 G_tse). Wrappers: methods/m9_voxlock2.py (explicit enrollment) and
methods/m9_voxlock2_self.py (self-enrollment). Tests: .venv/bin/python -m tests.test_tse

Public API
----------
    model = get_model(threads=1)                    # cached VoxLock sessions (onnxruntime, CPU)
    cond  = enroll(audio, source='explicit')        # (n,) 16 kHz, >= 3 s -> Conditioning (3-s voice sample)
    ex    = extract(signal, cond, X=None)           # -> Extraction(y (T,), mask (257, N) on the lab grid)
    res   = union(signal, {'A': cA, 'B': cB}, ...)  # both partners -> UnionResult(y, mask, ...)
    se    = self_enroll(fe)                         # fe = tracker.front_end(x, fs, info) -> SelfEnrollment
    y     = enhance(x, fs, info, mode='explicit' | 'self', return_stages=False)   # the method body
    ms    = latency_ms(frames_per_run=None)         # algorithmic latency of the VoxLock + mask chain

Model (data, loaded with onnxruntime only)
------------------------------------------
models/voxlock/sherpa-onnx-voxlock-int8/: enroll.onnx (= voxlock_enroll.onnx, sha256 5d06048c...) and
chunk_int8.onnx (= voxlock_chunk_int8.onnx, sha256 df61a9d0...), VoxLock 1.1.1 weights, Apache-2.0,
(c) Arbaz Siddiqui, from github.com/arbazsiddiqui/voxlock/releases/tag/sherpa-onnx-models.
- enroll: `enroll` (1, 48000) -> `enroll_mag` (1, 257, 376), `context` (1, 512, 298). The voice sample must
  be EXACTLY 3 s: other lengths give other Fe/Fc and the chunk model then fails in a fixed attention reshape
  (verified here: 47872, 48128, 32000 and 64000 samples all fail).
- chunk: `wav` (1, (K-1)*128 + 512), `enroll_mag`, `context`, `h`, `c` (6, 32, 256) -> `frames` (1, K, 512)
  (Hann-windowed time frames: STFT, masking and inverse DFT happen inside the graph), `h_out`, `c_out`.
- meta data: sample_rate 16000, n_fft = window 512 (hann), hop 128, frames_per_chunk 100, output_gain -0.5
  (a LINEAR factor: the network returns the target at about -2x the input because it was trained with a
  scale-invariant loss), state_shape 6,32,256, voice_sample_seconds 3.

Inference loop (ported from voxlock 1.1.1, Apache-2.0: see "Third-party code" at the end of this docstring)
---------------------------------------------------------------------------------------------------------
Source: voxlock 1.1.1 `voxlock/core.py` (`loud_sample`, `Stream.push/_run/flush`, `VoxLock.isolate`),
github.com/arbazsiddiqui/voxlock commit 08ed37c (tag sherpa-onnx-models), Apache-2.0; the voice-sample
rule is voxlock 1.3.0 `fit_sample` = sherpa-onnx PR 4085 (`speech-denoiser-voxlock-stream.h`): a sample
longer than 3 s gives its loudest 3 s (10 ms energy frames) within its first 30 s; shorter is rejected.
(voxlock >= 1.2 ships different 'v2' ONNX graphs and refuses these 1.1 files, so 1.1.1 is the reference.)
1. reflect-pad 256 samples at the start (torch.stft centre padding);
2. run the chunk model on runs of <= 100 frames ((100-1)*128 + 512 samples; runs overlap by 384 samples),
   carrying h and c;
3. overlap-add the frames at hop 128 and divide by the summed squared Hann window;
4. drop the first 256 output samples; at the end reflect-pad 256 samples and flush; multiply by -0.5.
File-mode output is time-aligned with the input. Algorithmic latency: 384 samples (24 ms) of look-ahead plus
the chunk. The int8 graph quantizes activations dynamically PER RUN, so the output depends on the run
boundaries (sherpa-onnx PR 4085 reports 46-47 dB SNR between 100-frame runs and 40 ms chunks); this module
always uses the reference partition (100-frame runs from the file start), which makes it deterministic.
Parity with sherpa-onnx / the voxlock package could not be run here (no scoring_venv); tests.test_tse checks
the port against the release's test vector instead (en-voice-sample-f.wav = loudest 3 s of the fp32 output on
en-combined.wav): our int8 output contains it at 31.0 dB SNR, gain -0.006 dB, same polarity, sample-exact
(+-1 sample: 11 dB).

Union and strength (DESIGN M9)
------------------------------
m_k = clip(|STFT(s_k)| / (|X| + eps), 0, 1) on the lab grid (lab.dsp 512/128 sqrt-Hann), X = STFT of the TSE
input; m_tse = max(m_A, m_B) (combine='max', default) or clip(|S_A + S_B| / |X|) (combine='sum');
f(l) = clip(1 - sum m_tse^2 |X|^2 / sum |X|^2 over 300-4000 Hz, 0, 1) * VAD(l);
w(l) = a w(l-1) + (1 - a)(0.8 f(l) + 0.2) with a = 0.8 per 10 ms (0.8365 per 8 ms hop);
m_eff = (1 - w) + w m_tse; y = istft(m_eff X). Optional priming of each pass: see PRIME.
Guard (needs the S1 track): partner k's pass is associated with the S1 partner whose posterior best predicts
its kept energy; in each S1 segment of that partner (p > 0.9, gaps <= 0.3 s bridged, >= 0.3 s) where the pass
keeps less than -6 dB of the input band energy, m_k is set to 1.

Latency and online preset
-------------------------
latency_ms(frames_per_run) = 24 ms VoxLock look-ahead (384 samples) + one run of frames_per_run hops (8 ms
each) + 32 ms lab STFT (mask stage): 856 ms for the default 100-frame partition, 96 ms for 5-frame (40 ms)
runs. The default call is offline on top of that: the S1 partner model, the guard's S1 segments and
self-enrollment need the whole file (the wrappers set OFFLINE_ONLY = True). info['tse'] = {'online': True}
is the causal configuration (ONLINE preset: 5-frame runs, no guard, no self-enrollment fallback, no S1);
the VAD of the strength rule (common.speech_presence: recursive smoothing + running minimum) is causal.

Robustness (method contract)
----------------------------
enroll() rejects a clip with NaN/inf in the part it reads (fit_sample: the first 30 s) and a voice sample
below ENROLL_MIN_RMS (-140 dBFS, under one 24-bit LSB: an all-zero enroll spectrum makes the chunk graph
emit NaN); enhance() also checks every pass output. An unusable enrollment therefore falls back
(self-enrollment if allowed, else the prefiltered Mic A) with the reason in diag, never NaN. T = 0 gives an
empty output. info['enroll'] may be a dict {'A', 'B'} or a sequence of two clips ((2, n) array, list, tuple).

Third-party code
----------------
loud_sample(), VoxLockStream and voxlock_isolate() follow voxlock `voxlock/core.py` (loud_sample, Stream,
VoxLock.isolate) closely, with the same structure, variable names and comments, so they are a derivative of
VoxLock, Copyright 2026 Arbaz Siddiqui (NOTICE of github.com/arbazsiddiqui/voxlock), licensed under the
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; upstream core.py carries no header
of its own). Changes: the enroll model runs once in enroll() (Conditioning) instead of in Stream.__init__;
the voice-sample rule is voxlock 1.3.0 fit_sample; frames per run come from the model meta data; optional
priming; the output is zero-padded to the input length for inputs of <= 16 ms; no downloads, no file I/O,
no LiveStream.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import calib
from . import common as cm

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOXLOCK_DIR = os.path.join(LAB, "models", "voxlock", "sherpa-onnx-voxlock-int8")
WESPEAKER_PATH = os.path.join(LAB, "models", "wespeaker", "wespeaker_en_voxceleb_resnet34.onnx")

SR, HOP, NFFT, PAD = 16000, 128, 512, 256
SAMPLE = 3 * SR                    # the voice sample the graphs accept: exactly 3 s
SAMPLE_WITHIN_S = 30.0             # a longer sample: loudest 3 s within its first 30 s (voxlock 1.3 / sherpa)
WIN2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32) ** 2
MASK_EPS = 1e-9                    # |X| floor in the effective mask (digital full scale 1.0)
# A voice sample below this RMS is digital silence (-140 dBFS, under one 24-bit LSB = -138.5 dBFS; the INMP441
# noise floor is about -87 dBFS). Measured (tmp/fix:m9/probe_nan.py): an all-zero sample, or noise at 1e-30
# (enroll_mag underflows to 0 in float32), makes the chunk graph emit NaN; noise at 1e-20 already gives
# finite output, so this floor only rejects clips that carry no usable signal at all.
ENROLL_MIN_RMS = 1e-7

# Latency (DESIGN 4.2: "about 24 ms plus the chunk length" for VoxLock, 32 ms for the 512-point lab STFT)
REF_FRAMES_PER_RUN = 100           # the model meta data's frames_per_chunk (reference partition; tests check it)
ONLINE_FRAMES_PER_RUN = 5          # 40 ms runs (DESIGN M9: 40-80 ms chunks), the online preset
LOOKAHEAD_MS = 1000.0 * (NFFT - HOP) / SR       # 24 ms: VoxLock output is final 384 samples after the input
LAB_STFT_MS = 1000.0 * cm.N_FFT / cm.FS         # 32 ms: lab STFT (512/128) of the mask stage

# Union / strength / guard (DESIGN M9)
STRENGTH_BAND = (300.0, 4000.0)
STRENGTH_A = float(0.8 ** (cm.HOP_S / 0.010))   # 0.8 per 10 ms frame -> 0.8365 per 8 ms hop
STRENGTH_GAIN, STRENGTH_MIN = 0.8, 0.2
GUARD_P, GUARD_DROP_DB, GUARD_GAP_S, GUARD_MIN_S = 0.9, -6.0, 0.3, 0.3

# Priming (option, OFF by default = the reference VoxLock behaviour): a pass may first run over a primer whose
# output is dropped (primer(): 'sample' = the 3-s voice sample alone, 'mix' = the sample laid over the scene's
# first 3 s). Why it exists: VoxLock latches onto the first voice it hears (its training calls start with the
# caller, train/train.py); on clean dry mixed-gender conversations the partner who talks second kept only
# 0.44-0.89 of its pass output, 0.61-1.00 with a 'sample' primer (6 pairs, provisional split). The latching is
# speaker-dependent: dev pair 1284/7127 (tests.test_tse) keeps 0.99 unprimed. On two same-gender dev pairs
# priming changed nothing: both passes kept only 0.37-0.69 of their own partner even in clean dry speech.
# Why it is off: in noisy bus scenes it costs partner fidelity. Dev evaluation (64 scenes): 'sample' primed
# union ESTOI gain +0.012/+0.006 (1-ch/2-ch) vs +0.043/+0.022 unprimed, partner TA -4.6 vs -1.5 dB. Dev tuning
# seeds (12 scenes): 'mix' vs none: ESTOI -0.003 [-0.005, -0.001], TA -0.4 dB, interferers -0.4 dB, own-partner
# share of each pass +0.07 [+0.03, +0.12].
PRIME = False

# Self-enrollment gates (DESIGN M9 "Enrollment, bench"), see self_enroll(). The DESIGN's per-frame gates
# (p_k > 0.9 AND |tau - tau_k| < 10 us AND CDR > 3 dB, runs >= 0.5 s) gave <= 0.74 s per partner on 24 dev tuning
# scenes (30 s, seeds 9001-9012; S1 found two partners in 21): S1 meets all three on only a few % of solo frames.
# Here p_k > 0.9 and no double talk are per-frame gates, and directness / TDOA are run-level gates (median CDR
# > SELF_RUN_CDR_DB, >= 25 % of frames within 10 us of tau_k): 0-3.4 s per partner per 30 s, never >= 3 s for
# both partners (0 of 21). The picked frames lie in the right partner's solo turns (100 % in tests.test_tse), but at
# 0/+5 dB with competing talkers the partner is only ~1/3 of the material's energy (dev evaluation: 0.34).
SELF_P = 0.9
SELF_TAU_TOL = 10e-6
SELF_RUN_TAU_FRAC = 0.25
SELF_RUN_CDR_DB = 0.0
SELF_MIN_RUN_S = 0.3
SELF_GAP_FRAMES = 6                # bridge per-frame flicker inside a run (<= 48 ms)
SELF_MAX_TOTAL_S = 10.0
SELF_MIN_TOTAL_S = 3.0             # DESIGN: >= 3 s. Lower values (>= 1.5 s) tile the material up to 3 s
# Cosine gates with the S1 speaker model (WeSpeaker ResNet34 + spk.fbank). DESIGN says 0.7 for the halves, but
# on clean dev LibriSpeech this model gives same-speaker 1.5-s halves median 0.61, p10 0.49-0.51, min 0.43-0.46,
# different-speaker 1.5-s pairs max 0.31-0.34 and 3-s pairs max 0.35-0.42 (p95 0.28). A Kaldi-exact fbank
# (Nyquist upper mel edge) changed nothing (tmp/build:m9/embcal2.py).
SELF_HALF_SIM = 0.4                # cosine(first 1.5 s, second 1.5 s) of the voice sample must reach this
SELF_SAME_SIM = 0.5                # A and B samples at least this similar -> one voice, not two partners


# ------------------------------------------------------------------ model
def _meta(sess):
    return dict(sess.get_modelmeta().custom_metadata_map)


class VoxLock:
    """The two VoxLock sessions plus their checked meta data.

    threads: onnxruntime intra-op threads per session (default 1: RTF is measured single-process)."""

    def __init__(self, model_dir=VOXLOCK_DIR, threads=1, enroll_file="enroll.onnx", chunk_file="chunk_int8.onnx"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if int(threads) != 1:
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        prov = ["CPUExecutionProvider"]
        self.threads = int(threads)
        self.enroll_sess = ort.InferenceSession(os.path.join(model_dir, enroll_file), so, providers=prov)
        self.chunk_sess = ort.InferenceSession(os.path.join(model_dir, chunk_file), so, providers=prov)
        me, mc = _meta(self.enroll_sess), _meta(self.chunk_sess)
        for m, role in ((me, "enroll"), (mc, "chunk")):
            if m.get("model_type") != "voxlock" or m.get("model_role") != role:
                raise ValueError(f"not a VoxLock {role} model: {m.get('model_type')}/{m.get('model_role')}")
        for k, want in (("sample_rate", SR), ("n_fft", NFFT), ("hop_length", HOP), ("window_length", NFFT),
                        ("voice_sample_seconds", SAMPLE // SR)):
            if int(me[k]) != want or int(mc[k]) != want:
                raise ValueError(f"VoxLock meta {k}: enroll {me[k]}, chunk {mc[k]}, expected {want}")
        if me.get("window_type") != "hann" or mc.get("window_type") != "hann" or me["version"] != mc["version"]:
            raise ValueError("VoxLock meta: window_type / version mismatch")
        self.meta = mc
        self.gain = np.float32(float(mc["output_gain"]))                       # -0.5 (linear)
        self.state_shape = tuple(int(v) for v in mc["state_shape"].split(","))  # (6, 32, 256)
        self.frames_per_run = int(mc["frames_per_chunk"])                       # 100
        self.version = mc.get("model_version", "?")

    def enroll(self, sample):
        """(48000,) float32 voice sample -> (enroll_mag, context)."""
        mag, ctx = self.enroll_sess.run(None, {"enroll": np.asarray(sample, np.float32)[None]})
        return mag, ctx


_MODELS = {}


def get_model(threads=1):
    """Process-wide cached VoxLock (first call loads 79 MB of ONNX: 1.3-1.6 s measured on the M1 Pro).
    Only the sessions are cached; no signal, state or calibration is kept between calls."""
    if threads not in _MODELS:
        _MODELS[threads] = VoxLock(threads=threads)
    return _MODELS[threads]


def latency_ms(frames_per_run=None):
    """Algorithmic latency (ms) of the VoxLock + mask chain: 24 ms look-ahead + one run of frames_per_run hops
    (8 ms each) + 32 ms lab STFT. None = the 100-frame reference partition (856 ms); 5 = online preset (96 ms).
    The offline parts of the default call (S1 guard, self-enrollment) need the whole file on top of this."""
    k = REF_FRAMES_PER_RUN if frames_per_run is None else int(frames_per_run)
    return LOOKAHEAD_MS + 1000.0 * k * HOP / SR + LAB_STFT_MS


# ------------------------------------------------------------------ enrollment
@dataclass
class Conditioning:
    """Speaker conditioning for one partner: the VoxLock enroll outputs of a 3-s voice sample."""
    enroll_mag: np.ndarray            # (1, 257, 376) float32
    context: np.ndarray               # (1, 512, 298) float32
    sample: np.ndarray                # (48000,) float32: the voice sample actually used
    start: int = 0                    # first sample of `sample` within the audio given to enroll()
    source: str = "explicit"          # 'explicit' (recorded separately) | 'self' (picked from the scene)
    segments: list = field(default_factory=list)   # self: [(start, end, snr_db)] scene samples used
    diag: dict = field(default_factory=dict)


def loud_sample(x, sec=3.0, within=15.0):
    """voxlock.core.loud_sample (Apache-2.0, Copyright 2026 Arbaz Siddiqui; see "Third-party code"): the
    loudest `sec` seconds (10 ms energy frames) in the first `within` s.
    Returns (segment, start sample). Only used to pick a voice sample from a longer enrollment clip."""
    seg = np.asarray(x, np.float32)[: int(within * SR)]
    n = len(seg) // 160
    w = int(sec * SR / 160)
    if n <= w:
        return seg, 0
    e = (seg[: n * 160].reshape(n, 160) ** 2).mean(1)
    st = int(np.convolve(e, np.ones(w), "valid").argmax()) * 160
    return seg[st: st + int(sec * SR)], st


def fit_sample(audio, within_s=SAMPLE_WITHIN_S):
    """The voice sample as the graphs take it: exactly 3 s. Longer audio gives its loudest 3 s within its
    first `within_s` seconds (voxlock 1.3 fit_sample / sherpa-onnx). Raises ValueError below 3 s and when the
    part it reads (the first max(3 s, within_s)) contains NaN or inf."""
    s = np.asarray(audio, np.float32).ravel()
    if len(s) < SAMPLE:
        raise ValueError(f"voice sample must be >= 3 s, got {len(s) / SR:.2f} s")
    if not np.all(np.isfinite(s[: max(SAMPLE, int(within_s * SR))])):
        raise ValueError("enrollment clip contains NaN or inf")
    if len(s) == SAMPLE:
        return s.copy(), 0
    seg, st = loud_sample(s, within=within_s)
    return np.ascontiguousarray(seg[:SAMPLE]), st


def conditioning(sample, model=None, start=0, source="explicit", segments=None, diag=None):
    """Conditioning of an exactly-3-s voice sample, validated: the sample must not be digital silence
    (RMS >= ENROLL_MIN_RMS) and the enroll outputs must be finite with a non-zero spectrum (an all-zero
    enroll_mag makes the chunk graph emit NaN). Raises ValueError with the reason otherwise."""
    model = model or get_model()
    sample = np.ascontiguousarray(sample, np.float32)
    if sample.shape != (SAMPLE,):
        raise ValueError(f"voice sample must be exactly {SAMPLE} samples, got {sample.shape}")
    if not np.all(np.isfinite(sample)):
        raise ValueError("voice sample contains NaN or inf")
    rms = float(np.sqrt(np.mean(np.square(sample, dtype=np.float64))))
    if not rms >= ENROLL_MIN_RMS:
        lvl = "all zero" if rms == 0 else f"RMS {20 * np.log10(rms):.0f} dBFS"
        raise ValueError(f"voice sample is digital silence ({lvl} < {20 * np.log10(ENROLL_MIN_RMS):.0f} dBFS)")
    mag, ctx = model.enroll(sample)
    if not (np.all(np.isfinite(mag)) and np.all(np.isfinite(ctx)) and float(np.max(mag)) > 0.0):
        raise ValueError("VoxLock enroll output unusable (non-finite or all-zero spectrum)")
    return Conditioning(mag, ctx, sample, int(start), source, list(segments or []), dict(diag or {}))


def enroll(audio, model=None, source="explicit", segments=None, diag=None):
    """Conditioning from >= 3 s of one partner speaking alone (16 kHz mono float, any level).
    A 2-D clip ((2, n) or (n, 2), e.g. recorded on the device) uses its Mic A channel.
    Raises ValueError for clips that are too short, contain NaN/inf where read, or are digital silence."""
    model = model or get_model()
    a = np.asarray(audio, np.float32)
    if a.ndim == 2:
        a = a[0] if a.shape[0] <= a.shape[1] else a[:, 0]
    sample, st = fit_sample(a)
    return conditioning(sample, model, st, source, segments, diag)


# ------------------------------------------------------------------ extraction (voxlock 1.1.1 Stream port)
# VoxLockStream and voxlock_isolate() are derived from voxlock core.py (Stream, VoxLock.isolate), Copyright 2026
# Arbaz Siddiqui, Apache License 2.0; changes listed under "Third-party code" in the module docstring.
class VoxLockStream:
    """Streaming VoxLock for one conditioning (port of voxlock 1.1.1 core.Stream).
    push(chunk) returns the samples that are final (24 ms behind the input plus the run buffer);
    flush() returns the rest. push(x) + flush() on a whole file gives output aligned with x."""

    def __init__(self, model, cond):
        self.m = model
        self.mag, self.ctx = cond.enroll_mag, cond.context
        self.h = np.zeros(model.state_shape, np.float32)
        self.c = np.zeros_like(self.h)
        self.head = np.zeros(0, np.float32)          # first samples, until the start padding can be built
        self.buf = None                              # padded input not yet consumed by a frame
        self.ola = np.zeros(NFFT, np.float32)
        self.env = np.zeros(NFFT, np.float32)
        self.dropped = 0                             # leading padded samples still to drop from the output
        self.n_in = 0
        self.runs = 0

    def _run(self, frames_per_run):
        out = []
        while self.buf is not None and len(self.buf) >= NFFT:
            k = min(frames_per_run, 1 + (len(self.buf) - NFFT) // HOP)
            seg = self.buf[: (k - 1) * HOP + NFFT]
            fr, self.h, self.c = self.m.chunk_sess.run(None, {"wav": seg[None], "enroll_mag": self.mag,
                                                              "context": self.ctx, "h": self.h, "c": self.c})
            self.runs += 1
            ola = np.concatenate([self.ola, np.zeros(k * HOP, np.float32)])
            env = np.concatenate([self.env, np.zeros(k * HOP, np.float32)])
            for j in range(k):
                ola[j * HOP: j * HOP + NFFT] += fr[0, j]
                env[j * HOP: j * HOP + NFFT] += WIN2
            done = k * HOP                           # samples no later frame will touch
            y = ola[:done] / np.where(env[:done] > 1e-11, env[:done], 1.0)
            self.ola, self.env = ola[done:], env[done:]
            self.buf = self.buf[done:]
            cut = min(self.dropped, len(y))
            self.dropped -= cut
            out.append(y[cut:] * self.m.gain)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def push(self, chunk, frames_per_run=None):
        frames_per_run = frames_per_run or self.m.frames_per_run
        chunk = np.asarray(chunk, np.float32)
        self.n_in += len(chunk)
        if self.buf is None:
            self.head = np.concatenate([self.head, chunk])
            if len(self.head) <= PAD:
                return np.zeros(0, np.float32)
            self.buf = np.concatenate([self.head[1: PAD + 1][::-1], self.head])   # reflect, as torch centre
            self.dropped = PAD
        else:
            self.buf = np.concatenate([self.buf, chunk])
        return self._run(frames_per_run)

    def flush(self):
        if self.buf is None:
            return np.zeros(0, np.float32)
        x = np.concatenate([self.buf, self.buf[-PAD - 1: -1][::-1]])            # end padding (reflect)
        nfr = 1 + (len(x) - NFFT) // HOP
        self.buf = np.pad(x, (0, max(0, (nfr - 1) * HOP + NFFT - len(x))))
        y = self._run(self.m.frames_per_run)
        y = np.concatenate([y, self.ola / np.where(self.env > 1e-11, self.env, 1.0) * self.m.gain])
        self.buf = None
        return y


def _active_rms(x, frame=320, rel_db=-30.0):
    """RMS over 20 ms frames within rel_db of the loudest frame (lab.common.active_rms rule)."""
    k = len(x) // frame
    if k == 0:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-30))
    fr = np.sqrt(np.mean(np.square(x[: k * frame].reshape(k, frame), dtype=np.float64), axis=1) + 1e-30)
    return float(np.sqrt(np.mean(fr[fr > fr.max() * 10 ** (rel_db / 20)] ** 2)))


def primer(signal, cond, prime):
    """Audio run through VoxLock before the scene (its output is dropped). prime: False (nothing),
    True or 'sample' (the 3-s voice sample alone), 'mix' (the voice sample at the active level of the scene's
    first 3 s, added to those 3 s: the enrolled voice talking over the scene's own background)."""
    s = np.asarray(cond.sample, np.float32)
    if not prime:
        return np.zeros(0, np.float32)
    if prime is True or prime == "sample":
        return s
    if prime == "mix":
        head = np.zeros(len(s), np.float32)
        h = np.asarray(signal, np.float32).ravel()[: len(s)]
        head[: len(h)] = h
        g = _active_rms(head) / max(_active_rms(s), 1e-12) if np.any(head) else 1.0
        return (s * np.float32(g) + head).astype(np.float32)
    raise ValueError(f"prime must be False, True, 'sample' or 'mix', got {prime!r}")


def voxlock_isolate(signal, cond, model=None, frames_per_run=None, prime=None):
    """VoxLock on a whole signal (voxlock.VoxLock.isolate with a given sample): (T,) float32, aligned.
    prime (default PRIME): run over [primer, signal] and drop the primer's output, so the LSTM state has
    locked onto the enrolled voice before the scene starts (see PRIME and primer())."""
    model = model or get_model()
    prime = PRIME if prime is None else prime
    x = np.asarray(signal, np.float32).ravel()
    pre = primer(x, cond, prime)
    st = VoxLockStream(model, cond)
    y = np.concatenate([st.push(np.concatenate([pre, x]), frames_per_run), st.flush()])[len(pre): len(pre) + len(x)]
    if len(y) < len(x):                              # only for inputs <= 16 ms (no frame at all)
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def lab_mask(s_hat, X, eps=MASK_EPS):
    """Effective mask of a target estimate on the lab grid: clip(|STFT(s_hat)| / (|X| + eps), 0, 1)."""
    S = s_hat if np.iscomplexobj(s_hat) else cm.stft(np.asarray(s_hat, np.float64))
    return np.clip(np.abs(S) / (np.abs(X) + eps), 0.0, 1.0)


@dataclass
class Extraction:
    y: np.ndarray                     # (T,) float64 target estimate, time-aligned with the input
    mask: np.ndarray                  # (257, N) effective mask on the lab grid
    seconds: float                    # wall time of the VoxLock pass


def extract(signal, cond, X=None, model=None, frames_per_run=None, prime=None):
    """One VoxLock pass for one partner: target estimate and its effective lab-grid mask.
    X: lab STFT of `signal` (computed if None)."""
    t0 = time.perf_counter()
    y = voxlock_isolate(signal, cond, model, frames_per_run, prime).astype(np.float64)
    sec = time.perf_counter() - t0
    X = cm.stft(np.asarray(signal, np.float64)) if X is None else X
    return Extraction(y, lab_mask(y, X), sec)


# ------------------------------------------------------------------ union, strength, guard
def band_energy(X, band=STRENGTH_BAND):
    sl = cm.bins_between(*band)
    return np.abs(X[sl]) ** 2


def adaptive_strength(m_tse, X, vad, a=STRENGTH_A, gain=STRENGTH_GAIN, w_min=STRENGTH_MIN):
    """VoiceFilter-Lite style strength (DESIGN M9): f = clip(1 - kept band energy fraction, 0, 1) * VAD,
    w(l) = a w(l-1) + (1 - a)(gain f + w_min), w(-1) = w_min. Returns (w, f), each (N,)."""
    sl = cm.bins_between(*STRENGTH_BAND)
    P = np.abs(X[sl]) ** 2
    tot = P.sum(axis=0)
    kept = np.where(tot > cm.EPS, (m_tse[sl] ** 2 * P).sum(axis=0) / np.maximum(tot, cm.EPS), 1.0)  # empty: keep
    f = np.clip(1.0 - kept, 0.0, 1.0) * np.asarray(vad, float)
    w, _ = cm.smooth(gain * f + w_min, a, init=np.float64(w_min))
    return w, f


def _runs(mask, gap=0, min_len=1):
    """[(a, b)) runs of True in a bool array, gaps of <= gap False frames bridged, length >= min_len."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    br = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[br + 1]]
    ends = np.r_[idx[br], idx[-1]] + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_len]


def guard(masks, X, trk):
    """DESIGN M9 guard. masks {'A': m_A, 'B': m_B} (modified copies are returned) and diagnostics.
    Each enrolled pass k is matched to the S1 partner j (TDOA-defined) that maximises the summed
    posterior-weighted kept band energy (2x2 assignment). In S1 segments of j (p_j > 0.9, gaps <= 0.3 s
    bridged, >= 0.3 s long) where pass k keeps less than -6 dB of the input band energy, m_k = 1."""
    out = {k: m.copy() for k, m in masks.items()}
    diag = {"applied": False}
    if trk is None or trk.p_A is None or trk.inseparable or set(masks) != {"A", "B"}:
        diag["skipped"] = "no separable S1 partner model"
        return out, diag
    P = band_energy(X)
    tot = np.maximum(P.sum(axis=0), cm.EPS)
    sl = cm.bins_between(*STRENGTH_BAND)
    kept = {k: (m[sl] ** 2 * P).sum(axis=0) / tot for k, m in masks.items()}
    post = {"A": np.asarray(trk.p_A, float), "B": np.asarray(trk.p_B, float)}
    score = {(k, j): float((post[j] * kept[k]).sum() / max(post[j].sum(), 1e-9)) for k in kept for j in post}
    same = score[("A", "A")] + score[("B", "B")]
    swap = score[("A", "B")] + score[("B", "A")]
    assoc = {"A": "A", "B": "B"} if same >= swap else {"A": "B", "B": "A"}
    gap = int(round(GUARD_GAP_S / cm.HOP_S))
    min_len = int(round(GUARD_MIN_S / cm.HOP_S))
    diag.update(assoc=assoc, score={f"{k}->{j}": round(v, 4) for (k, j), v in score.items()}, segments={})
    for k, j in assoc.items():
        segs = []
        for a, b in _runs(post[j] > GUARD_P, gap, min_len):
            num = (masks[k][sl, a:b] ** 2 * P[:, a:b]).sum()
            drop = 10 * np.log10(max(num, cm.EPS) / max(P[:, a:b].sum(), cm.EPS))
            hit = bool(drop < GUARD_DROP_DB)
            if hit:
                out[k][:, a:b] = 1.0
                diag["applied"] = True
            segs.append((a, b, round(float(drop), 2), hit))
        diag["segments"][k] = segs
    return out, diag


@dataclass
class UnionResult:
    y: np.ndarray                     # (T,) output, time-aligned with the input
    mask: np.ndarray                  # (257, N) m_eff applied to X
    m_tse: np.ndarray                 # (257, N) union mask before the strength mix-back
    masks: dict                       # per partner effective masks (after the guard)
    est: dict                         # per partner raw VoxLock outputs (T,)
    w: np.ndarray                     # (N,) strength
    f: np.ndarray                     # (N,) removed-energy fraction x VAD
    guard: dict
    seconds: dict                     # wall times: per pass, union


def union(signal, conds, model=None, X=None, combine="max", strength="adaptive", vad=None, trk=None,
          use_guard=True, parallel=False, frames_per_run=None, est=None, prime=None):
    """Run one VoxLock pass per conditioning and combine them (DESIGN M9).

    signal: (T,) TSE input (prefiltered Mic A or a linear beam output y_lin); conds: {'A': Conditioning, ...}.
    combine: 'max' (m_tse = max_k m_k, default) | 'sum' (m_tse = clip(|S_A + S_B| / |X|)).
    strength: 'adaptive' (DESIGN), 'full' (w = 1, pure union mask) or a float w in [0, 1].
    vad: (N,) speech presence for the strength (default: common.speech_presence on 300-4000 Hz of X).
    trk: S1 Track for the guard (skipped when None or inseparable). parallel: run the passes in two threads
    (onnxruntime releases the GIL; wall time ~ one pass, CPU time unchanged).
    est: {k: (T,)} VoxLock outputs from an earlier call on the same signal (reused, no passes run).
    prime: primer for each pass (False, 'sample', 'mix'; default PRIME = False), see primer()."""
    model = model or get_model()
    x = np.asarray(signal, np.float64).ravel()
    X = cm.stft(x) if X is None else X
    t0 = time.perf_counter()
    keys = list(conds) if est is None else list(est)
    if est is not None:
        est, sec = {k: np.asarray(v, np.float64) for k, v in est.items()}, {"passes": 0.0}
    elif parallel and len(keys) > 1:
        with ThreadPoolExecutor(max_workers=len(keys)) as ex:
            futs = {k: ex.submit(voxlock_isolate, x, conds[k], model, frames_per_run, prime) for k in keys}
            est = {k: futs[k].result().astype(np.float64) for k in keys}
        sec = {"passes": time.perf_counter() - t0}
    else:
        est, sec = {}, {}
        for k in keys:
            t1 = time.perf_counter()
            est[k] = voxlock_isolate(x, conds[k], model, frames_per_run, prime).astype(np.float64)
            sec[f"pass_{k}"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    S = {k: cm.stft(v) for k, v in est.items()}
    masks = {k: lab_mask(S[k], X) for k in keys}
    gdiag = {"applied": False, "skipped": "disabled"}
    if use_guard:
        masks, gdiag = guard(masks, X, trk)
    if combine == "max":
        m_tse = np.max(np.stack([masks[k] for k in keys]), axis=0)
    elif combine == "sum":
        m_tse = lab_mask(sum(S.values()), X)
        if gdiag.get("applied"):                     # guarded frames pass through in the sum, too
            m_tse = np.maximum(m_tse, np.max(np.stack([np.where(masks[k] >= 1.0, 1.0, 0.0) for k in keys]), 0))
    else:
        raise ValueError(f"combine must be 'max' or 'sum', got {combine!r}")
    if strength == "adaptive":
        if vad is None:
            vad, _ = cm.speech_presence(band_energy(X).mean(axis=0))
        w, f = adaptive_strength(m_tse, X, vad)
    elif strength == "full":
        w, f = np.ones(X.shape[-1]), np.zeros(X.shape[-1])
    else:
        w, f = np.full(X.shape[-1], float(strength)), np.zeros(X.shape[-1])
    m_eff = (1.0 - w) + w * m_tse
    y = cm.istft(m_eff * X, len(x))
    sec["union"] = time.perf_counter() - t1
    return UnionResult(y, m_eff, m_tse, masks, est, w, f, gdiag, sec)


# ------------------------------------------------------------------ self-enrollment
_EMBEDDER = {}


def get_embedder():
    """The S1 speaker model (WeSpeaker ResNet34, 1 thread): spk_onnx.WeSpeakerEmbedder (validated Kaldi front
    end, API.md) when importable, else the generic spk.OnnxEmbedder hook; None if the model file is absent.
    The cosine gates were calibrated with both front ends (same distributions; tmp/build:m9/embcal2.py)."""
    if "e" not in _EMBEDDER:
        emb = None
        if os.path.exists(WESPEAKER_PATH):
            try:
                from . import spk_onnx
                emb = spk_onnx.WeSpeakerEmbedder(WESPEAKER_PATH, threads=1)
            except (ImportError, AttributeError, TypeError):
                from . import spk
                emb = spk.OnnxEmbedder(WESPEAKER_PATH, threads=1)
        _EMBEDDER["e"] = emb
    return _EMBEDDER["e"]


def steered_ds(X, tau):
    """Delay-and-sum beam toward TDOA tau with center-referenced steering: Y = a^H X / 2, (257, N)."""
    a = cm.steer(cm.FREQS, tau)
    return 0.5 * (np.conj(a[:, 0])[:, None] * X[0] + np.conj(a[:, 1])[:, None] * X[1])


def frames_to_samples(a, b, T):
    """Sample span of STFT frames [a, b) (frame centres l*128 - 256, +-64 samples), clipped to [0, T)."""
    return max(0, a * cm.HOP - cm.N_FFT // 2 - cm.HOP // 2), min(T, b * cm.HOP - cm.N_FFT // 2 - cm.HOP // 2)


def confident_frames(trk, k, p_thr=SELF_P):
    """(N,) bool per-frame gate: S1 posterior p_k > p_thr, no double talk, frame inside the signal."""
    p = trk.p_A if k == "A" else trk.p_B
    tk = trk.tau_A_t if k == "A" else trk.tau_B_t
    return (p > p_thr) & ~trk.double_talk & cm.valid_frames(len(p)) & np.isfinite(tk)


def confident_runs(trk, k, p_thr=SELF_P, tau_tol=SELF_TAU_TOL, run_tau_frac=SELF_RUN_TAU_FRAC,
                   run_cdr_db=SELF_RUN_CDR_DB, min_run_s=SELF_MIN_RUN_S, gap_frames=SELF_GAP_FRAMES):
    """Runs [(a, b, snr_db)] of confident_frames() (gaps <= gap_frames bridged, >= min_run_s) that pass the
    run-level near-field gates: median frame CDR > run_cdr_db and >= run_tau_frac of the frames within
    tau_tol of tau_k. Sorted by SNR (S1 level minus its noise floor), best first."""
    tk = trk.tau_A_t if k == "A" else trk.tau_B_t
    snr = trk.level_db - trk.noise_db
    out = []
    for a, b in _runs(confident_frames(trk, k, p_thr), gap_frames, int(np.ceil(min_run_s / cm.HOP_S))):
        if np.median(trk.cdr_db[a:b]) <= run_cdr_db:
            continue
        if np.mean(np.abs(trk.tau[a:b] - tk[a:b]) < tau_tol) < run_tau_frac:
            continue
        out.append((a, b, float(np.mean(snr[a:b]))))
    return sorted(out, key=lambda r: -r[2])


@dataclass
class SelfEnrollment:
    ok: bool
    conds: dict                       # {'A': Conditioning, 'B': Conditioning} for the partners that passed
    segments: dict                    # {'A': [(start, end, snr_db)], ...}: scene samples chosen, best SNR first
    sims: dict                        # cosine similarities: 'A_halves', 'B_halves', 'AB'
    reasons: list                     # why it failed (empty when ok)
    diag: dict = field(default_factory=dict)


def self_enroll(fe, model=None, embedder="default", min_total_s=SELF_MIN_TOTAL_S, max_total_s=SELF_MAX_TOTAL_S,
                half_sim=SELF_HALF_SIM, same_sim=SELF_SAME_SIM, **run_kw):
    """Pick each partner's confident solo segments with S1 and enroll VoxLock on them (DESIGN M9, bench).

    fe: tracker.front_end(x, fs, info) of the bus scene (offline). For k in A, B: confident_runs() (see the
    SELF_* constants; run_kw overrides them), highest SNR first, up to max_total_s; audio = delay-and-sum beam
    steered to tau_k. Voice sample = the best runs concatenated (best first) and cut at 3 s; at least
    min_total_s of runs are required (DESIGN: 3 s; with less than 3 s the material is tiled up to 3 s).
    The S1 speaker model must give cosine(first 1.5 s, second 1.5 s) >= half_sim (skipped when the material is
    tiled from < 3 s: the halves would repeat each other), and the A and B samples must be different voices
    (cosine < same_sim). Otherwise ok=False and the caller skips TSE for the file.
    segments[k] lists every chosen run as scene samples (start, end, snr_db) for diagnostics."""
    model = model or get_model()
    trk, X, T = fe.track, fe.X, fe.x.shape[-1]
    emb = get_embedder() if embedder == "default" else embedder
    res = SelfEnrollment(False, {}, {}, {}, [], {"inseparable_why": list(trk.inseparable_why or ()),
                                                  "min_total_s": min_total_s})
    if trk.p_A is None or trk.inseparable:
        res.reasons.append("S1 partners inseparable: " + ",".join(trk.inseparable_why or ("no partner model",)))
        return res
    samples, embs = {}, {}
    for k in ("A", "B"):
        tau_k = trk.tau_A if k == "A" else trk.tau_B
        runs = confident_runs(trk, k, **run_kw)
        chosen, total = [], 0
        for a, b, s in runs:
            if total >= max_total_s * SR:
                break
            s0, s1 = frames_to_samples(a, b, T)
            if s1 > s0:
                chosen.append((s0, s1, round(s, 2)))
                total += s1 - s0
        res.segments[k] = chosen
        res.diag[f"{k}_confident_frames"] = int(confident_frames(trk, k).sum())
        res.diag[f"{k}_runs"] = len(runs)
        res.diag[f"{k}_total_s"] = round(total / SR, 2)
        if total < min_total_s * SR or total == 0:
            res.reasons.append(f"{k}: {total / SR:.2f} s of confident solo speech < {min_total_s:g} s")
            continue
        y = cm.istft(steered_ds(X, tau_k), T)
        mat = np.concatenate([y[s0:s1] for s0, s1, _ in chosen])
        tiled = len(mat) < SAMPLE
        samples[k] = np.resize(mat, SAMPLE).astype(np.float32)          # np.resize tiles cyclically
        res.diag[f"{k}_tiled"] = bool(tiled)
        if emb is not None:
            embs[k] = emb(samples[k])
            if not tiled:
                e1, e2 = emb(samples[k][: SAMPLE // 2]), emb(samples[k][SAMPLE // 2:])
                res.sims[f"{k}_halves"] = round(float(e1 @ e2), 4)
                if res.sims[f"{k}_halves"] < half_sim:
                    res.reasons.append(f"{k}: halves cosine {res.sims[f'{k}_halves']:.2f} < {half_sim}")
    if len(embs) == 2:
        res.sims["AB"] = round(float(embs["A"] @ embs["B"]), 4)
        if res.sims["AB"] >= same_sim:
            res.reasons.append(f"A/B samples are one voice (cosine {res.sims['AB']:.2f} >= {same_sim})")
    if emb is None:
        res.diag["embedder"] = "missing: cosine checks skipped"
    for k, smp in samples.items():
        try:
            res.conds[k] = conditioning(smp, model, 0, "self", res.segments[k],
                                        {"tau": float(trk.tau_A if k == "A" else trk.tau_B)})
        except ValueError as e:
            res.reasons.append(f"{k}: {e}")
    res.ok = not res.reasons and len(res.conds) == 2
    return res


# ------------------------------------------------------------------ method body
# info['tse'] keys. fallback: what the explicit wrapper does when info['enroll'] is unusable ('self':
# self-enrollment, 'none': prefiltered Mic A). online: apply the ONLINE preset (causal configuration).
DEFAULTS = {"threads": 1, "parallel": False, "combine": "max", "strength": "adaptive", "guard": True,
            "frames_per_run": None, "prime": PRIME, "self_min_total_s": SELF_MIN_TOTAL_S,
            "fallback": "self", "online": False}
# Online preset (DESIGN 2.3 online variant): 40 ms VoxLock runs, no guard (it needs whole S1 segments), no
# self-enrollment (it searches the whole file), so no S1 front end at all; latency_ms(5) = 96 ms. Keys given
# in info['tse'] next to 'online' override the preset (e.g. frames_per_run 10 = 80 ms runs).
ONLINE = {"frames_per_run": ONLINE_FRAMES_PER_RUN, "guard": False, "fallback": "none"}


def options(user=None):
    """Effective options: DEFAULTS, then the ONLINE preset if user['online'], then the user's keys
    (info['tse']). Unknown keys are kept (and listed in diag by enhance)."""
    user = dict(user or {})
    opt = dict(DEFAULTS)
    if user.get("online"):
        opt.update(ONLINE)
    opt.update(user)
    if opt["fallback"] not in ("self", "none"):
        raise ValueError(f"info['tse']['fallback'] must be 'self' or 'none', got {opt['fallback']!r}")
    return opt


def enroll_clips(enr):
    """info['enroll'] -> {'A': clip, 'B': clip}. Accepts a dict with both keys or a sequence of two clips
    ((2, n) array, list or tuple; partner A first). Raises ValueError saying what is missing or wrong."""
    if enr is None:
        raise ValueError("no info['enroll']")
    if isinstance(enr, dict):
        miss = [k for k in ("A", "B") if enr.get(k) is None]
        if miss:
            raise ValueError("info['enroll'] has no clip for partner " + " and ".join(miss))
        return {"A": enr["A"], "B": enr["B"]}
    if isinstance(enr, (list, tuple)) and len(enr) == 2:          # clips may differ in length
        return {"A": enr[0], "B": enr[1]}
    if isinstance(enr, np.ndarray) and enr.ndim >= 1 and len(enr) == 2 and (enr.ndim >= 2 or enr.dtype == object):
        return {"A": enr[0], "B": enr[1]}
    shape = f" of shape {enr.shape}" if isinstance(enr, np.ndarray) else ""
    raise ValueError(f"info['enroll'] must be {{'A': clip, 'B': clip}} or two clips, got {type(enr).__name__}{shape}")


def _clip_s(c):
    """Duration (s) of the channel enroll() uses from a clip (1-D, or 2-D with channels on the short axis)."""
    return (max(np.shape(c)) if np.ndim(c) == 2 else np.size(c)) / SR


def enhance(x, fs, info, mode="explicit", return_stages=False):
    """M9 method body (see methods/m9_voxlock2.py and m9_voxlock2_self.py for the info keys).

    Bus: S0 prefilter (+ S1 via tracker.front_end, offline, when the guard or self-enrollment needs it) ->
    enrollment (mode 'explicit': info['enroll'], then self-enrollment unless options()['fallback'] is 'none';
    mode 'self': self-enrollment only; the online preset never self-enrolls) -> VoxLock on the prefiltered Mic A
    for A and B -> union mask, guard, adaptive strength -> y = istft(m_eff X_A). Every pass output is checked:
    an enrollment whose passes are not finite counts as unusable. If no enrollment is usable, the prefiltered
    Mic A is returned (TSE skipped, reasons in diag). Meeting: TSE is never used (it would remove the
    non-enrolled talkers, DESIGN 2.2): prefiltered Mic A. T = 0: empty output."""
    from . import tracker
    if fs != cm.FS:
        raise ValueError("M9 expects fs = 16000")
    if mode not in ("explicit", "self"):
        raise ValueError(f"mode must be 'explicit' or 'self', got {mode!r}")
    info = info or {}
    x = np.asarray(x, np.float64)
    T = x.shape[-1]
    opt = options(info.get("tse"))
    diag = {"mode": mode, "options": dict(opt), "used": "passthrough", "reasons": []}
    unknown = sorted(set(opt) - set(DEFAULTS))
    if unknown:
        diag["unknown_options"] = unknown
    t0 = time.perf_counter()
    scenario = info.get("scenario", "bus")
    if T == 0:
        diag["reasons"].append("empty input (T = 0)")
        return _out(np.zeros(0), diag, return_stages, t0, T)
    if scenario != "bus":
        y = calib.prefilter(x, scenario)[0]
        diag["reasons"].append(f"scenario {scenario!r}: TSE disabled (bus only)")
        return _out(y, diag, return_stages, t0, T)
    fe = None

    def front_end():                                      # S0 + S1, offline, computed once when needed
        nonlocal fe
        if fe is None:
            t1 = time.perf_counter()
            fe = tracker.front_end(x, fs, info)
            diag["seconds_front_end"] = round(time.perf_counter() - t1, 3)
        return fe

    # prefiltered Mic A (S0 HPF 120 Hz): front_end().x[0] is calib.prefilter(x, 'bus')[0]
    xin = front_end().x[0] if opt["guard"] else calib.prefilter(x, "bus")[0]
    model = get_model(int(opt["threads"]))
    sources = ["explicit"] if mode == "explicit" else []
    if mode == "self" or opt["fallback"] == "self":
        if opt["online"]:
            diag["reasons"].append("online preset: no self-enrollment (it searches the whole file)")
        else:
            sources.append("self")
    X = res = conds = None
    for i, src in enumerate(sources):
        then = "; self-enrollment instead" if "self" in sources[i + 1:] else "; TSE skipped"
        if src == "explicit":
            try:
                clips = enroll_clips(info.get("enroll"))
                conds = {k: enroll(clips[k], model, "explicit") for k in ("A", "B")}
            except (KeyError, ValueError, TypeError) as e:
                diag["reasons"].append(f"explicit enrollment unusable ({e}){then}")
                continue
            diag["enrollment"] = {k: {"start_s": c.start / SR, "clip_s": _clip_s(clips[k])} for k, c in conds.items()}
        else:
            se = self_enroll(front_end(), model, min_total_s=float(opt["self_min_total_s"]))
            diag["self_enrollment"] = {"ok": se.ok, "segments": se.segments, "sims": se.sims,
                                       "reasons": se.reasons, "diag": se.diag}
            if not se.ok:
                diag["reasons"] += se.reasons
                continue
            conds = se.conds
        X = cm.stft(xin) if X is None else X
        res = union(xin, conds, model, X=X, combine=opt["combine"], strength=opt["strength"],
                    trk=fe.track if opt["guard"] else None, use_guard=bool(opt["guard"]),
                    parallel=bool(opt["parallel"]), frames_per_run=opt["frames_per_run"], prime=opt["prime"])
        bad = [k for k, v in res.est.items() if not np.all(np.isfinite(v))]
        if bad or not np.all(np.isfinite(res.y)):
            diag["reasons"].append(f"{src} enrollment unusable (VoxLock output not finite: pass "
                                   f"{', '.join(bad) or 'union'}){then}")
            res = conds = None
            continue
        break
    if res is None:
        return _out(xin, diag, return_stages, t0, T)
    diag.update(used="tse", guard=res.guard, seconds=res.seconds, latency_ms=latency_ms(opt["frames_per_run"]),
                enroll_source={k: c.source for k, c in conds.items()})
    stages = {"tse": res, "conds": conds, "front_end": fe}
    return _out(res.y, diag, return_stages, t0, T, stages)


def _out(y, diag, return_stages, t0, T, stages=None):
    diag["seconds_total"] = round(time.perf_counter() - t0, 3)
    diag["rtf"] = round(diag["seconds_total"] / (T / SR), 4) if T else None
    y = np.asarray(y, np.float64)
    if not return_stages:
        return y
    return dict({"out": y, "pre_agc": y, "diag": diag}, **(stages or {}))
