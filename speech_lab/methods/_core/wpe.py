"""M10: two-channel weighted prediction error (WPE) dereverberation (DESIGN.md M10, 2.2, 1.6 item 11).

Written from the equations of Nakatani et al. 2010 [59] and Yoshioka & Nakatani 2012 [60], with
NARA-WPE's wpe_v8 / OnlineWPE [61] as the reference design (nothing vendored). Per frequency bin,
with D channels, K taps and a prediction delay of Δ frames:
    ỹ_t = [y_{t-Δ-K+1}; ...; y_{t-Δ}]     the D·K stacked past frames
    x_t = y_t − Gᴴ ỹ_t                     D dereverberated channels (G is D·K x D)
Offline, maximum likelihood by iteration (the first pass takes λ from the observation):
    λ_t = max(mean_d |x_t,d|² averaged over ±psd_context frames, floor · mean_t of the observed λ_t)
    R = Σ_t ỹ_t ỹ_tᴴ / λ_t,   P = Σ_t ỹ_t y_tᴴ / λ_t,   G = (R + δI)⁻¹ P,   δ = load · tr(R) / (D·K)
The first pass's statistics do not depend on the window, so overlapping segment windows share them.
Online, exponentially weighted RLS with forgetting factor α (Q = R⁻¹ starts at I/rls_prior, G at 0):
    x_t = y_t − G_{t−1}ᴴ ỹ_t                (a-priori output, the online result)
    λ_t = max(mean_d |x_d|² averaged over frames t−2·psd_context..t, floor · running mean input power)
    k = Q ỹ_t / (α λ_t + ỹ_tᴴ Q ỹ_t),   Q ← (Q − k (Q ỹ_t)ᴴ) / α,   G ← G + k x_tᴴ
The online λ comes from the a-priori output, as in RLS sparse linear prediction (Jukić et al., IEEE
SPL 2017), over the causal counterpart of the offline ±1-frame window; NARA-WPE averages the input
over 14 frames and starts from R = I. Frames without regressor energy (digital silence) skip the
update so that Q cannot wind up.

STFT arrays are laid out like lab.dsp.stft output: (D, F, N), 257 bins, hop 128 (125 frames/s).
Every entry point takes optional `comps`: other STFTs of the same shape (e.g. the early and late
parts of a source image) that receive exactly the filters estimated on Y (master/slave, DESIGN 4.2).

Defaults were tuned on the v2 dev meetings (tune:wpe): offline 30 taps, delay 4, one iteration, 2 s
segments (DESIGN's sweep was taps 10-20, delay 2-3, 3 iterations; longer filters kept helping in
0.5-0.9 s rooms, delay 4 cancels less of the speech itself, extra iterations did not help). Online
uses delay 4 too but keeps 10 taps: the RLS cost grows with taps², and 15 taps gained only +0.003
mean ΔESTOI for 1.7x the cost.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lab.dsp import istft, stft

FS = 16000
N_FFT = 512
HOP = 128
_TINY = 1e-30
_MEM = 1 << 26  # bytes of stacked regressors per bin chunk (bounds memory on long files)
_CHUNK = 4096  # frames per transposed block in the online loop (about 33 s)

PARAMS = dict(
    taps=30,            # K, prediction filter length (frames; 240 ms of past)
    delay=4,            # Δ, prediction delay (frames; 32 ms, no window overlap with frame t)
    iterations=1,       # offline ML iterations
    psd_context=1,      # λ averaged over ±1 frame offline, over frames t−2..t online
    floor=1e-5,         # λ floor relative to the bin's mean input power (−50 dB); numerical safety
    load=1e-6,          # offline diagonal loading relative to the mean eigenvalue of R
    seg_s=2.0,          # offline: segment length when no change points are given
    ctx_s=(3.0, 1.0),   # offline: estimation context before / after each segment
    xfade_s=0.05,       # offline: linear crossfade between segment filters
    alpha=0.998,        # online forgetting factor: memory 1/(1−α) = 500 frames = 4 s
    rls_prior=10.0,     # online R starts at rls_prior·I: about 10 frames of pull towards G = 0
)
ONLINE = dict(taps=10)  # online defaults that differ from PARAMS


def params(online=False, **overrides):
    """PARAMS (with ONLINE for the online RLS) and keyword overrides; unknown names raise."""
    bad = set(overrides) - set(PARAMS)
    if bad:
        raise TypeError(f"unknown WPE parameter(s): {sorted(bad)}")
    p = {**PARAMS, **(ONLINE if online else {}), **overrides}
    if p["taps"] < 1 or p["delay"] < 1:
        raise ValueError("taps and delay must be >= 1")
    return p


# ---------------------------------------------------------------- offline
def _as_stft(Y):
    """Y as an array, checked to be a multichannel STFT (D, F, N)."""
    Y = np.asarray(Y)
    if Y.ndim != 3:
        raise ValueError("expected an STFT of shape (D, F, N)")
    return Y


def _chunks(F, D, n, p):
    """Bin slices whose stacked regressors for n frames (complex128) stay within _MEM bytes."""
    step = max(1, _MEM // (16 * max(n, 1) * D * p["taps"]))
    return [slice(f, min(f + step, F)) for f in range(0, F, step)]


def _frames(Y, bins, t0, t1, p):
    """Regressors ỹ_t (f, n, D·K) and current frames y_t (f, n, D) for frames t in [t0, t1) of
    Y (D, F, N); frames before 0 are zeros."""
    K, o = p["taps"], p["taps"] + p["delay"] - 1
    a = max(0, t0 - o)
    W = Y[:, bins, a:t1].transpose(1, 0, 2).astype(complex)  # frames t0 − o .. t1 − 1
    if a > t0 - o:
        W = np.pad(W, ((0, 0), (0, 0), (a - (t0 - o), 0)))
    v = sliding_window_view(W[:, :, :t1 - t0 + K - 1], K, axis=-1)  # (f, D, n, K)
    Yt = v.transpose(0, 2, 3, 1).reshape(v.shape[0], t1 - t0, -1)  # stacking order (k, d)
    return Yt, W[:, :, o:].transpose(0, 2, 1)


def _psd(X, context):
    """Channel-mean power of X (f, n, D), averaged over ±context frames (fewer at the edges)."""
    lam = np.mean(X.real ** 2 + X.imag ** 2, axis=-1)
    if context:
        s, c = lam.copy(), np.ones(lam.shape[-1])
        for k in range(1, context + 1):
            s[:, k:] += lam[:, :-k]
            s[:, :-k] += lam[:, k:]
            c[k:] += 1
            c[:-k] += 1
        lam = s / c
    return lam


def _stats(Yt, Yc, w):
    """Weighted correlations R = Σ_t w_t ỹ_t ỹ_tᴴ (f, D·K, D·K) and P = Σ_t w_t ỹ_t y_tᴴ (f, D·K, D)."""
    A = Yt.transpose(0, 2, 1) * w[:, None, :]
    return A @ Yt.conj(), A @ Yc.conj()


def _solve(R, P, p):
    """G = (R + δI)⁻¹ P with δ = load · tr(R) / (D·K)."""
    dk = R.shape[-1]
    R = R + (p["load"] * np.trace(R, axis1=1, axis2=2).real / dk + _TINY)[:, None, None] * np.eye(dk)
    return np.linalg.solve(R, P)


def _estimate(Y, wins, p):
    """One WPE filter G (F, D·K, D) per estimation window [t0, t1) of frames, by iterated weighted least
    squares. The first iteration weights every frame by the observation's PSD, so its statistics are
    summed once per stretch between window edges and shared by the overlapping windows; later
    iterations re-weight each window by its own output. The floor is relative to the bin's mean
    observation PSD over the whole signal."""
    D, F, N = Y.shape
    dk = D * p["taps"]
    Gs = [np.zeros((F, dk, D), complex) for _ in wins]
    cuts = sorted({0, N} | {t for w in wins for t in w})
    pos = {t: i for i, t in enumerate(cuts)}
    for bins in _chunks(F, D, N, p):
        Yt, Yc = _frames(Y, bins, 0, N, p)
        lam = _psd(Yc, p["psd_context"])
        floor = p["floor"] * lam.mean(axis=-1, keepdims=True) + _TINY
        w0 = 1.0 / np.maximum(lam, floor)
        sums = [_stats(Yt[:, a:b], Yc[:, a:b], w0[:, a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
        for G, (t0, t1) in zip(Gs, wins):
            parts = sums[pos[t0]:pos[t1]]
            if not parts:  # empty window: G stays 0
                continue
            G[bins] = _solve(sum(r for r, _ in parts), sum(q for _, q in parts), p)
            for _ in range(1, p["iterations"]):
                Ytw, Ycw = Yt[:, t0:t1], Yc[:, t0:t1]
                lam = _psd(Ycw - Ytw @ G[bins].conj(), p["psd_context"])
                G[bins] = _solve(*_stats(Ytw, Ycw, 1.0 / np.maximum(lam, floor)), p)
    return Gs


def _apply(Y, G, t0, t1, p):
    """x_t = y_t − Gᴴ ỹ_t for frames t in [t0, t1): (D, F, n)."""
    D, F = Y.shape[:2]
    X = np.empty((F, t1 - t0, D), complex)
    for bins in _chunks(F, D, t1 - t0, p):
        Yt, Yc = _frames(Y, bins, t0, t1, p)
        X[bins] = Yc - Yt @ G[bins].conj()
    return X.transpose(2, 0, 1)


def _edges(bounds, N, h):
    """Segment edges [0, ..., N]; a boundary closer than 2h frames to the previous edge or to N is dropped."""
    e = [0]
    for b in sorted(int(round(b)) for b in bounds):
        if b - e[-1] >= 2 * h and N - b >= 2 * h:
            e.append(b)
    return e + [N]


def _ramp(t, b, h):
    """Weight of the segment that starts at frame b: rises linearly over the 2h frames centred on b."""
    return np.clip((t - b + h + 0.5) / (2 * h), 0.0, 1.0)


def wpe_offline(Y, bounds=None, comps=None, fs=FS, **kw):
    """Offline WPE of an STFT Y (D, F, N). Returns X (D, F, N), or (X, [filtered comps]) if comps.

    bounds: frames where a new segment starts, e.g. S1 change points (frame l is centred on sample
    128·l − 256; methods/_core/common.frame_of_sample converts). None splits every seg_s seconds;
    [] estimates one global filter for the whole signal (DESIGN M10 variant a).
    Each segment's filter is estimated on [start − ctx_s[0], end + ctx_s[1]] and applied to the
    segment, with xfade_s linear crossfades between neighbours (DESIGN M10 variant b).
    """
    p = params(**kw)
    srcs = [_as_stft(Y)] + [_as_stft(c) for c in (comps or [])]
    D, F, N = srcs[0].shape
    rate = fs / HOP
    h = max(1, int(round(p["xfade_s"] * rate / 2)))
    if bounds is None:
        step = max(1, int(round(p["seg_s"] * rate)))
        bounds = range(step, N, step)
    edges = _edges(bounds, N, h)
    cl, cr = (int(round(c * rate)) for c in p["ctx_s"])
    segs = list(zip(edges[:-1], edges[1:]))
    Gs = _estimate(srcs[0], [(max(0, s - cl), min(N, e + cr)) for s, e in segs], p)
    outs = [np.zeros((D, F, N), complex) for _ in srcs]
    for (s, e), G in zip(segs, Gs):
        o0, o1 = max(0, s - h), min(N, e + h)
        t = np.arange(o0, o1)
        w = (_ramp(t, s, h) if s > 0 else 1.0) * (1.0 - _ramp(t, e, h) if e < N else 1.0)
        for out, src in zip(outs, srcs):
            out[:, :, o0:o1] += w * _apply(src, G, o0, o1, p)
    return outs[0] if comps is None else (outs[0], outs[1:])


# ---------------------------------------------------------------- online
class OnlineWPE:
    """Causal frame-by-frame WPE (RLS) with per-speaker filter memory (DESIGN M10 variant c)."""

    def __init__(self, n_bins=N_FFT // 2 + 1, channels=2, **kw):
        p = params(online=True, **kw)
        dk = channels * p["taps"]
        self.taps, self.alpha, self.floor = p["taps"], p["alpha"], p["floor"]
        self.Q = np.tile(np.eye(dk, dtype=complex) / p["rls_prior"], (n_bins, 1, 1))  # R⁻¹
        self._outer = np.empty_like(self.Q)
        self.ctx = 2 * p["psd_context"]
        self.pw = np.zeros((n_bins, self.ctx))  # output power of the previous ctx frames, newest last
        self.G = np.zeros((n_bins, dk, channels), complex)
        self.hist = np.zeros((n_bins, p["taps"] + p["delay"] - 1, channels), complex)  # oldest first
        self.comp_hist = []
        self.pnum, self.pden = np.zeros(n_bins), 0.0
        self.speaker, self.memory = None, {}
        self.frames = 0

    def set_speaker(self, key):
        """On a confirmed talker change: store the current (Q, G), restore `key`'s if known."""
        if key == self.speaker:
            return
        if self.speaker is not None:
            self.memory[self.speaker] = (self.Q.copy(), self.G.copy())
        if key in self.memory:
            self.Q, self.G = (a.copy() for a in self.memory[key])
        self.speaker = key

    def step(self, y, comps=()):
        """Dereverberate one frame y (F, D). With comps (frames like y) returns (x, [filtered comps])."""
        F = y.shape[0]
        yt = self.hist[:, :self.taps].reshape(F, 1, -1)
        Gc = self.G.conj()
        x = y - (yt @ Gc)[:, 0]
        while len(self.comp_hist) < len(comps):
            self.comp_hist.append(np.zeros_like(self.hist))
        xc = []
        for c, hc in zip(comps, self.comp_hist):
            xc.append(c - (hc[:, :self.taps].reshape(F, 1, -1) @ Gc)[:, 0])
            hc[:, :-1] = hc[:, 1:]
            hc[:, -1] = c
        self.pnum = self.alpha * self.pnum + np.mean(np.abs(y) ** 2, axis=-1)
        self.pden = self.alpha * self.pden + 1.0
        px = np.mean(x.real ** 2 + x.imag ** 2, axis=-1)
        m = min(self.frames, self.ctx)
        lam = (self.pw[:, self.ctx - m:].sum(axis=-1) + px) / (m + 1)
        lam = np.maximum(lam, self.floor * self.pnum / self.pden + _TINY)
        if self.ctx:
            self.pw[:, :-1] = self.pw[:, 1:]
            self.pw[:, -1] = px
        yt = yt[:, 0]
        Qy = (self.Q @ yt[:, :, None])[:, :, 0]
        act = np.einsum("fi,fi->f", yt.conj(), yt).real > _TINY
        k = Qy / (self.alpha * lam + np.einsum("fi,fi->f", yt.conj(), Qy).real)[:, None]
        k[~act] = 0.0
        np.matmul(k[:, :, None], Qy.conj()[:, None, :], out=self._outer)
        self.Q -= self._outer
        if act.all():  # multiply: dividing a complex array is about 4x slower
            self.Q *= 1.0 / self.alpha
        else:  # no forgetting without excitation (digital silence), so Q cannot wind up
            self.Q[act] *= 1.0 / self.alpha
        self.G += k[:, :, None] @ x.conj()[:, None, :]
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = y
        self.frames += 1
        if self.frames % 64 == 0:  # keep Q Hermitian against rounding drift
            self.Q = 0.5 * (self.Q + self.Q.conj().transpose(0, 2, 1))
        return (x, xc) if len(comps) else x


def wpe_online(Y, keys=None, comps=None, **kw):
    """Causal online WPE over a whole STFT Y (D, F, N); returns like wpe_offline.

    keys: optional (N,) speaker label per frame (e.g. an S1 cluster id; −1 = unknown). A change to a
    label >= 0 stores the current filter state and restores that speaker's state (variant c).
    """
    srcs = [_as_stft(Y)] + [_as_stft(c) for c in (comps or [])]
    D, F, N = srcs[0].shape
    w = OnlineWPE(F, D, **kw)
    outs = [np.empty((D, F, N), complex) for _ in srcs]
    for c0 in range(0, N, _CHUNK):
        blk = [np.ascontiguousarray(s[:, :, c0:c0 + _CHUNK].transpose(2, 1, 0), dtype=complex) for s in srcs]
        res = [np.empty_like(b) for b in blk]  # (n, F, D) blocks, one frame per row
        for j in range(blk[0].shape[0]):
            if keys is not None and keys[c0 + j] >= 0:
                w.set_speaker(int(keys[c0 + j]))
            if len(blk) > 1:
                res[0][j], xc = w.step(blk[0][j], [b[j] for b in blk[1:]])
                for r, v in zip(res[1:], xc):
                    r[j] = v
            else:
                res[0][j] = w.step(blk[0][j])
        for out, r in zip(outs, res):
            out[:, :, c0:c0 + _CHUNK] = r.transpose(2, 1, 0)
    return outs[0] if comps is None else (outs[0], outs[1:])


# ---------------------------------------------------------------- time domain
def dereverb(x, fs=FS, online=False, comps=None, **kw):
    """Time-domain entry: x (D, T) -> dereverberated (D, T), or (that, [filtered comps]) if comps.

    kw: `bounds` (offline) or `keys` (online), in lab.dsp.stft frames, and PARAMS overrides (the
    online RLS starts from PARAMS updated with ONLINE).
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[-1]
    C = None if comps is None else [stft(np.asarray(c, dtype=float)) for c in comps]
    r = wpe_online(stft(x), comps=C, **kw) if online else wpe_offline(stft(x), comps=C, fs=fs, **kw)
    if comps is None:
        return istft(r, T)
    return istft(r[0], T), [istft(c, T) for c in r[1]]
