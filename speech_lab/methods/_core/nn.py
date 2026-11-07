"""M8: neural single-channel post-filters GTCRN and DPDFNet-4 with onnxruntime (DESIGN.md M8).

Models (LAB/models, from the k2-fsa/sherpa-onnx speech-enhancement releases; loaded as data only):
  "gtcrn"    models/gtcrn/gtcrn_simple.onnx  GTCRN (Rong et al., ICASSP 2024; code MIT), streaming export.
  "dpdfnet"  models/dpdfnet/dpdfnet4.onnx    DPDFNet-4 (Rika et al. 2025, Ceva; Apache-2.0), streaming export.

Processing chains, re-implemented in numpy from the reference code (credited below) and checked
against each graph's inputs/outputs and metadata:
  GTCRN    STFT n_fft 512, hop 256, periodic sqrt-Hann 512, center=True with reflect padding, no
           normalisation (torch.stft defaults used in training). One frame per session call:
           mix (1, 257, 1, 2) = [Re, Im], caches conv (2,1,16,16,33), tra (2,3,1,1,16),
           inter (2,1,33,16), all zero at the start. Output enh (1, 257, 1, 2) is the input frame
           times a complex ratio mask. ISTFT: windowed overlap-add divided by the summed squared
           window, centre padding removed. Causal; output frame t belongs to input frame t.
  DPDFNet  STFT n_fft 320 (20 ms), hop 160 (10 ms), Vorbis window, center=True reflect, no
           normalisation. One frame per call: spec (1, 1, 161, 2) and one flat state (52592,) that
           starts as zeros with metadata erb_norm_init in [0:32] and spec_norm_init in [32:128].
           Output frame t belongs to input frame t - 4 (the export's 40 ms look-ahead becomes delay);
           the reference shifts the waveform back by 2 * window = 4 hops, here the frames are shifted.

Level normalisation (lab addition, not in the reference code): both models are level dependent
(the DNS-Challenge synthesizer, microsoft/DNS-Challenge noisyspeech_synthesizer.cfg, writes noisy
mixtures at random levels of -35...-15 dBFS; our input levels are arbitrary). On LibriSpeech dev
speakers + DEMAND (bus HPF, -5/+5 dB, no limit), raw input at -66 dBFS speech level gave dSI-SDR
-6.8 dB (DPDFNet) and +1.1 dB (GTCRN) against +8.7 / +4.8 dB normalised; at -12 dBFS GTCRN raw gave
+3.1 dB (DPDFNet raw +9.0 dB: it prefers hot input). The model input is therefore scaled to RMS
TARGET_DB and the output scaled back, which makes M8 scale invariant. TARGET_DB = -25 dBFS (GTCRN)
/ -20 dBFS (DPDFNet), from a paired sweep of -40...-10 dBFS (bus path, A = 20 dB, 36 dev mixtures):
GTCRN is flat over -30...-25 dBFS (dSI-SDR 4.65 / 4.69 dB), DPDFNet improves up to -20...-15 dBFS
(9.03 dB, dESTOI +0.185, against 8.72 dB / +0.177 at -30 dBFS).
  level="file"    one gain per file from the RMS of x (offline, default; gain clamped to GAIN_DB);
  level="online"  per model frame, from a causal power tracker (rise 0.5 s, fall 4 s time constants,
                  started from the power of the first 0.5 s), applied to the model input frame and
                  removed from the aligned output frame; within 0.2 dB SI-SDR, 0.006 ESTOI and
                  0.05 PESQ of "file" on the same dev mixtures;
  level=None      raw input, as in sherpa-onnx.

Attenuation limit (DeepFilterNet / DPDFNet definition): Y = a * X + (1 - a) * S with a = 10^(-A/20),
X the noisy STFT and S the delay-aligned model output on the model's own STFT grid (the reference
mixes S[t] with X[t - 4] before its waveform shift; same thing). The same input/output mix is the
generic version for GTCRN. DESIGN M8: A = 20 dB bus, 12 dB meeting (ATTEN_DB).

Effective gain on the lab grid (lab.dsp 512/128) for stacks that combine masks once:
mask = clip(|STFT(y)| / (|STFT(x)| + eps), a, 1), y the limited output, x the model input. Applied
as mask * STFT(x) it reproduces y to about 29-34 dB (GTCRN) / 22-26 dB (DPDFNet) SI-SDR; against the
clean speech it scores like y for GTCRN and 0.6 dB SI-SDR / 0.012 ESTOI below y for DPDFNet. The
DESIGN form max(mask_raw, a) is also returned; it scored lower ESTOI on the dev mixtures (DPDFNet
+0.160 vs +0.173 at A = 20, +0.107 vs +0.128 at A = 12; GTCRN +0.061 vs +0.067, +0.034 vs +0.050).
Power-averaging the gain over 3 bins (3x1, 3x3) lost up to 0.9 dB SI-SDR / 0.017 ESTOI (DPDFNet);
over 3 frames (1x3), or using the real projection Re(Y X*)/|X|^2, it was within 0.15 dB / 0.005.

API (x: 1-D float, 16 kHz, digital full scale 1.0):
  denoise(x, name, atten_db=None, level="file") -> y (len(x),), aligned with x
  post_filter(x, name, atten_db=None, level="file", X_lab=None) -> {'y', 'mask', 'y_raw', 'mask_raw'}
  post_filter_stft(X_lab, length, name, atten_db=None, level="file") -> same, for a common.stft input
  model_output(x, name, level) -> (X, S, spec) on the model grid; analysis() / synthesis() / run_frames()
  load(name) -> (session, ModelSpec); latency_ms(name): 32 ms GTCRN, 60 ms DPDFNet (window + delay)
Cost (M1 Pro, one onnxruntime thread, 30 s input): RTF 0.043 GTCRN, 0.19 DPDFNet; with 3 workers
busy 0.046 / 0.21 (max 0.27 per file); on a machine overloaded by other jobs DPDFNet reached 0.58.

Sources (re-implemented, not copied):
  sherpa-onnx (Apache-2.0) sherpa-onnx/csrc/offline-speech-denoiser-gtcrn-impl.h, -gtcrn-model.cc,
    offline-speech-denoiser-dpdfnet-impl.h, -dpdfnet-model.cc, math.cc (MakeVorbisWindow);
  kaldi-native-fbank (Apache-2.0) csrc/stft.cc, istft.cc, feature-window.cc (periodic "hann");
  ceva-ip/DPDFNet (Apache-2.0) onnx_model/infer_dpdfnet_onnx.py (state init, 4-frame offset,
    attenuation limit); Xiaobin-Rong/gtcrn (MIT) stream/gtcrn_stream.py (cache shapes, STFT).
"""
import os
from dataclasses import dataclass

import numpy as np

from . import common as cm

FS = cm.FS
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATHS = {
    "gtcrn": os.path.join(LAB, "models", "gtcrn", "gtcrn_simple.onnx"),
    "dpdfnet": os.path.join(LAB, "models", "dpdfnet", "dpdfnet4.onnx"),
}
MODELS = tuple(MODEL_PATHS)
ATTEN_DB = {"bus": 20.0, "meeting": 12.0}     # DESIGN M8 attenuation limits
DELAY_FRAMES = {"gtcrn": 0, "dpdfnet": 4}      # output frame t <-> input frame t - delay
MASK_EPS = 1e-10                                # |X| regulariser of the lab-grid gain (full scale 1.0)
TARGET_DB = {"gtcrn": -25.0, "dpdfnet": -20.0}  # RMS of the model input after level normalisation
GAIN_DB = (-40.0, 50.0)                         # clamp of the normalising gain (dB)
ONLINE_TAU = (0.5, 4.0)                         # online level tracker: rise / fall time constants (s)
ONLINE_INIT_S = 0.5                             # online tracker start value: power of the first 0.5 s


@dataclass(frozen=True)
class ModelSpec:
    """Model STFT and I/O layout, read from the graph and its metadata."""
    name: str
    n_fft: int
    hop: int
    window: np.ndarray
    delay: int            # frames
    spec_input: str       # name of the spectrum input
    spec_shape: tuple     # one frame: (1, 257, 1, 2) for GTCRN, (1, 1, 161, 2) for DPDFNet
    state_inputs: tuple   # names of the state inputs, in output order after the spectrum
    state_init: tuple     # initial value of each state


_CACHE = {}


def periodic_sqrt_hann(n):
    """sqrt of the periodic Hann window (torch.hann_window(n).pow(0.5), knf GetWindow('hann'))."""
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))


def vorbis_window(n):
    """Vorbis power-complementary window as in sherpa-onnx MakeVorbisWindow and DPDFNet."""
    s = np.sin(0.5 * np.pi * (np.arange(n) + 0.5) / (n / 2.0))
    return np.sin(0.5 * np.pi * s * s)


def _meta_floats(meta, key):
    return np.array([float(v) for v in meta[key].split(",")], dtype=np.float32)


def load(name):
    """Cached (onnxruntime session, ModelSpec) for 'gtcrn' or 'dpdfnet'.
    One session per process, intra_op_num_threads = inter_op_num_threads = 1 (RTF is measured
    single-process), sequential execution, CPU provider."""
    if name in _CACHE:
        return _CACHE[name]
    if name not in MODEL_PATHS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    sess = ort.InferenceSession(MODEL_PATHS[name], so, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    if meta.get("model_type") != name or int(meta.get("sample_rate", 0)) != FS:
        raise ValueError(f"{MODEL_PATHS[name]}: unexpected metadata {meta.get('model_type')}, "
                         f"{meta.get('sample_rate')} Hz")
    n_fft, hop, win_len = int(meta["n_fft"]), int(meta["hop_length"]), int(meta["window_length"])
    if win_len != n_fft or n_fft != 2 * hop:
        raise ValueError(f"{name}: expected window = n_fft = 2 hop, got {n_fft}/{hop}/{win_len}")
    ins = sess.get_inputs()
    if name == "gtcrn":
        if meta["window_type"] != "hann_sqrt":
            raise ValueError(f"gtcrn: unexpected window {meta['window_type']}")
        window = periodic_sqrt_hann(n_fft)
        states = tuple(np.zeros([int(v) for v in meta[k].split(",")], np.float32)
                       for k in ("conv_cache_shape", "tra_cache_shape", "inter_cache_shape"))
        spec_shape = (1, n_fft // 2 + 1, 1, 2)
    else:
        if meta["window_type"] != "vorbis" or meta.get("center", "1") != "1" or \
                meta.get("normalized", "0") != "0" or meta.get("pad_mode", "reflect") != "reflect":
            raise ValueError(f"dpdfnet: unexpected STFT metadata {meta}")
        window = vorbis_window(n_fft)
        state = np.zeros(int(meta["state_size"]), np.float32)
        ne, ns_ = int(meta["erb_norm_state_size"]), int(meta["spec_norm_state_size"])
        state[:ne] = _meta_floats(meta, "erb_norm_init")
        state[ne:ne + ns_] = _meta_floats(meta, "spec_norm_init")
        states = (state,)
        spec_shape = (1, 1, n_fft // 2 + 1, 2)
    if tuple(ins[0].shape) != spec_shape or len(ins) != 1 + len(states):
        raise ValueError(f"{name}: unexpected graph inputs {[(i.name, i.shape) for i in ins]}")
    spec = ModelSpec(name=name, n_fft=n_fft, hop=hop, window=window, delay=DELAY_FRAMES[name],
                     spec_input=ins[0].name, spec_shape=spec_shape,
                     state_inputs=tuple(i.name for i in ins[1:]), state_init=states)
    _CACHE[name] = (sess, spec)
    return _CACHE[name]


# ------------------------------------------------------------------ model-grid STFT
def analysis(x, spec):
    """Model STFT of a 1-D signal: (N, n_fft/2+1) complex128, N = 1 + len(x) // hop.
    center=True: n_fft/2 samples of reflect padding on both sides (constant padding if the signal
    is too short to reflect), frame i covers padded samples [i*hop, i*hop + n_fft)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    p = spec.n_fft // 2
    xp = np.pad(x, p, mode="reflect" if len(x) > p else "constant")
    n = 1 + (len(xp) - spec.n_fft) // spec.hop
    idx = np.arange(spec.n_fft)[None, :] + spec.hop * np.arange(n)[:, None]
    return np.fft.rfft(xp[idx] * spec.window, axis=-1)


def synthesis(S, spec, length):
    """Inverse of analysis(): windowed overlap-add / summed squared window, centre padding removed,
    cut or zero-padded to `length` samples (n_fft = 2 hop, so overlap-add is two half-frame sums)."""
    S = np.asarray(S)
    h, n = spec.hop, S.shape[0]
    fr = np.fft.irfft(S, n=spec.n_fft, axis=-1) * spec.window
    y = np.zeros((n + 1) * h)
    y[: n * h] += fr[:, :h].reshape(-1)
    y[h:] += fr[:, h:].reshape(-1)
    w2 = spec.window ** 2
    den = np.zeros((n + 1) * h)
    den[: n * h] += np.tile(w2[:h], n)
    den[h:] += np.tile(w2[h:], n)
    y = np.where(den > 1e-12, y / np.where(den > 1e-12, den, 1.0), y)
    y = y[spec.n_fft // 2:]
    return y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))


# ------------------------------------------------------------------ model
def run_frames(X, name):
    """Run the streaming model over model-STFT frames X (N, F) complex; returns the model output
    S (N, F) complex128 *as emitted*: S[t] belongs to input frame t - delay (not yet aligned)."""
    sess, spec = load(name)
    n, F = X.shape
    buf = np.zeros(spec.spec_shape, np.float32)
    view = buf.reshape(F, 2)          # both layouts are (F, 2) in memory
    states = list(spec.state_init)
    feed = {}
    out = np.empty((n, F, 2), np.float32)
    Xr = np.ascontiguousarray(np.stack([X.real, X.imag], axis=-1), dtype=np.float32)
    names = [spec.spec_input] + list(spec.state_inputs)
    for t in range(n):
        view[:] = Xr[t]
        feed[names[0]] = buf
        for k, s in zip(names[1:], states):
            feed[k] = s
        res = sess.run(None, feed)
        out[t] = res[0].reshape(F, 2)
        states = res[1:]
    return out[..., 0].astype(np.float64) + 1j * out[..., 1].astype(np.float64)


def attenuation_limit(X, S, atten_db):
    """DeepFilterNet/DPDFNet attenuation limit Y = a X + (1 - a) S, a = 10^(-atten_db/20).
    X, S time-aligned complex STFTs (any grid; the mix is linear, so it also holds for waveforms).
    atten_db None, <= 0 or inf: no limit (Y = S)."""
    if atten_db is None or not np.isfinite(atten_db) or atten_db <= 0:
        return S
    a = 10.0 ** (-float(atten_db) / 20.0)
    return a * X + (1.0 - a) * S


def latency_ms(name):
    """Algorithmic latency of the streaming chain: window length + model delay (ms)."""
    _, spec = load(name)
    return 1000.0 * (spec.n_fft + spec.delay * spec.hop) / FS


def _clamp_db(g_db):
    return np.clip(g_db, GAIN_DB[0], GAIN_DB[1])


def level_gains(x, spec, n_frames, level="file", target_db=None):
    """Per-model-frame input gains (n_frames,) that bring x to RMS target_db (see module doc;
    default TARGET_DB of the model)."""
    if target_db is None:
        target_db = TARGET_DB[spec.name]
    if level is None:
        return np.ones(n_frames)
    x = np.asarray(x, dtype=np.float64)
    if level == "file":
        p = float(np.mean(x ** 2)) if len(x) else 0.0
        g_db = target_db - 10 * np.log10(max(p, 1e-30))
        return np.full(n_frames, 10 ** (_clamp_db(g_db) / 20))
    if level != "online":
        raise ValueError(f"level must be 'file', 'online' or None, got {level!r}")
    # causal tracker on hop-sized blocks; frame t (centred on sample t*hop) uses blocks up to t
    h = spec.hop
    nb = n_frames
    xb = np.pad(x, (0, max(0, nb * h - len(x))))[: nb * h].reshape(nb, h)
    pb = np.mean(xb ** 2, axis=1)
    lam_up = np.exp(-h / FS / ONLINE_TAU[0])
    lam_dn = np.exp(-h / FS / ONLINE_TAU[1])
    P = np.empty(nb)
    acc = max(float(np.mean(x[: int(ONLINE_INIT_S * FS)] ** 2)) if len(x) else 0.0, 1e-30)
    for t in range(nb):
        lam = lam_up if pb[t] > acc else lam_dn
        acc = lam * acc + (1 - lam) * pb[t]
        P[t] = acc
    g_db = target_db - 10 * np.log10(np.maximum(P, 1e-30))
    return 10 ** (_clamp_db(g_db) / 20)


def model_output(x, name, level="file", target_db=None):
    """Run model `name` on a 1-D 16 kHz signal. Returns (X, S, spec): the model-grid STFT X of the
    end-padded input and the model output S on the same frames, delay-aligned (S[t] belongs to
    X[t]) and with the level-normalising gain removed. The input is zero-padded at the end by
    n_fft + delay hops so that the model delay flushes the last samples (the reference
    implementations drop them)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    _, spec = load(name)
    xp = np.pad(x, (0, spec.n_fft + spec.delay * spec.hop))
    X = analysis(xp, spec)
    g = level_gains(x, spec, X.shape[0], level, target_db)[:, None]
    S = run_frames(X * g, name)
    if spec.delay:
        S = np.concatenate([S[spec.delay:], np.zeros((spec.delay, S.shape[1]), S.dtype)])
    return X, S / g, spec


def denoise(x, name, atten_db=None, level="file", target_db=None):
    """Enhance a 1-D 16 kHz signal with model `name`: output (len(x),) float64, time-aligned with x
    (model delay compensated). atten_db: attenuation limit in dB (None = off).
    level / target_db: input level normalisation ("file", "online" or None; target_db None =
    TARGET_DB[name])."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return np.zeros(0)
    X, S, spec = model_output(x, name, level, target_db)
    return synthesis(attenuation_limit(X, S, atten_db), spec, len(x))


def lab_gain(x, y, floor=0.0, eps=MASK_EPS, X=None):
    """Effective per-bin gain of y relative to x on the lab grid (common.stft 512/128):
    clip(|Y| / (|X| + eps), floor, 1), shape (257, N). X: common.stft(x) if already computed."""
    X = cm.stft(x) if X is None else X
    return np.clip(np.abs(cm.stft(y)) / (np.abs(X) + eps), floor, 1.0)


def floor_of(atten_db):
    """Gain floor a = 10^(-atten_db/20) implied by an attenuation limit (0 when there is none)."""
    if atten_db is None or not np.isfinite(atten_db) or atten_db <= 0:
        return 0.0
    return 10.0 ** (-float(atten_db) / 20.0)


def post_filter(x, name, atten_db=None, level="file", X_lab=None):
    """M8 for stacks: x 1-D (e.g. the linear beam output y_lin), one model run. Returns
      'y'        output with the attenuation limit (DPDFNet mix), aligned with x;
      'mask'     its lab-grid gain clip(|STFT y| / |STFT x|, a, 1), a = 10^(-atten_db/20);
      'y_raw'    model output without the limit;
      'mask_raw' its lab-grid gain clipped to [0, 1] (DESIGN M8 form: use max(mask_raw, a)).
    X_lab: common.stft(x) if the caller already has it."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    X_lab = cm.stft(x) if X_lab is None else X_lab
    if len(x) == 0:
        z = np.zeros(0)
        return {"y": z, "mask": np.ones(X_lab.shape), "y_raw": z, "mask_raw": np.ones(X_lab.shape)}
    X, S, spec = model_output(x, name, level)
    y_raw = synthesis(S, spec, len(x))
    y = synthesis(attenuation_limit(X, S, atten_db), spec, len(x))
    return {"y": y, "mask": lab_gain(x, y, floor_of(atten_db), X=X_lab),
            "y_raw": y_raw, "mask_raw": lab_gain(x, y_raw, 0.0, X=X_lab)}


def post_filter_stft(X, length, name, atten_db=None, level="file"):
    """Like post_filter() for a lab-grid STFT X (257, N) of a signal with `length` samples
    (common.stft layout). Apply the returned mask as mask * X, combined once with the other masks."""
    return post_filter(cm.istft(X, length), name, atten_db, level, X_lab=X)
