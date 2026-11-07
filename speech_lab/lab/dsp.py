"""Shared DSP helpers (optional for methods)."""
import numpy as np
from .common import FS, C


def _win(n):
    return np.sqrt(np.hanning(n + 1)[:-1])  # periodic sqrt-Hann (analysis = synthesis)


def stft(x, n_fft=512, hop=128):
    """x: (..., T) -> (..., F, N) complex. Perfect reconstruction with istft (WOLA)."""
    x = np.asarray(x, dtype=np.float64)
    w = _win(n_fft)
    pad = n_fft
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad + n_fft)])
    n = 1 + (xp.shape[-1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = xp[..., idx] * w  # (..., N, n_fft)
    return np.swapaxes(np.fft.rfft(frames, axis=-1), -1, -2)


def istft(X, length, n_fft=512, hop=128):
    w = _win(n_fft)
    frames = np.fft.irfft(np.swapaxes(X, -1, -2), n=n_fft, axis=-1) * w
    n = frames.shape[-2]
    T = (n - 1) * hop + n_fft
    y = np.zeros(frames.shape[:-2] + (T,))
    wsum = np.zeros(T)
    for i in range(n):
        y[..., i * hop: i * hop + n_fft] += frames[..., i, :]
        wsum[i * hop: i * hop + n_fft] += w ** 2
    y = y / np.maximum(wsum, 1e-8)
    return y[..., n_fft: n_fft + length]


def freqs(n_fft=512, fs=FS):
    return np.fft.rfftfreq(n_fft, 1 / fs)


def diffuse_coherence(f, spacing):
    return np.sinc(2 * f * spacing / C)


def gcc_phat(x0, x1, fs=FS, max_tau=None, interp=16):
    """TDOA of x0 relative to x1 (positive: x0 lags). Returns seconds."""
    n = len(x0) + len(x1)
    X = np.fft.rfft(x0, n) * np.conj(np.fft.rfft(x1, n))
    X /= np.abs(X) + 1e-12
    cc = np.fft.irfft(X, n * interp)
    m = int(interp * fs * max_tau) if max_tau else n * interp // 2
    cc = np.concatenate((cc[-m:], cc[: m + 1]))
    return (np.argmax(np.abs(cc)) - m) / float(interp * fs)


def recursive_avg(P, alpha, axis=-1):
    """First-order recursive smoothing along axis (frames)."""
    P = np.moveaxis(np.asarray(P), axis, -1)
    out = np.empty_like(P)
    acc = P[..., 0]
    for i in range(P.shape[-1]):
        acc = alpha * acc + (1 - alpha) * P[..., i]
        out[..., i] = acc
    return np.moveaxis(out, -1, axis)
