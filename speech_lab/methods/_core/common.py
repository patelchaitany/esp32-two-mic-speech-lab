"""Shared conventions and helpers for every method (DESIGN.md 2.0). API: methods/_core/API.md.

Conventions (every consumer relies on them):
- x[0] is Mic A, x[1] is Mic B. tau > 0 means Mic B hears the source later; a plane wave gives
  X_B = X_A * exp(-j 2 pi f tau), tau = d cos(theta) / c with theta the angle from the axis
  (theta = 0: beyond Mic A).
- Center-referenced steering a(f, tau) = [exp(+j pi f tau), exp(-j pi f tau)] (steer()).
- Phi_AB = E{X_A conj(X_B)}; a plane wave has coherence Gamma_s = exp(+j 2 pi f tau).
- STFT = lab.dsp (periodic sqrt-Hann 512, hop 128, 257 bins of 31.25 Hz). Frame l covers input
  samples [l*HOP - N_FFT, l*HOP); its centre is sample l*HOP - N_FFT/2.
- Sign trap: lab.dsp.gcc_phat(x0, x1) is positive when x0 lags, so tau = gcc_phat(x_B, x_A) (tdoa_gcc()).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, lfilter, sosfilt
from scipy.special import j0

from lab import dsp
from lab.common import C, FS

N_FFT = 512
HOP = 128
N_BINS = N_FFT // 2 + 1
HOP_S = HOP / FS
FREQS = dsp.freqs(N_FFT, FS)
CHUNK = 4096                      # frames per chunk in streaming loops (about 33 s)
EPS = 1e-20

# Recursive smoothing lambda = exp(-8 ms / tau_avg), rounded as in the DESIGN 2.0 table.
LAM_20MS = 0.67
LAM_50MS = 0.85
LAM_76MS = 0.90
LAM_160MS = 0.95
LAM_2S = 0.996


def lam(tau_avg_s, hop_s=HOP_S):
    """Exact smoothing constant exp(-hop / tau_avg) for an averaging time constant tau_avg_s."""
    return float(np.exp(-hop_s / tau_avg_s))


# ------------------------------------------------------------------ STFT and framing
def stft(x):
    """(..., T) -> (..., 257, N) complex128, identical to lab.dsp.stft(x, 512, 128) but framed in
    chunks of CHUNK frames, so a 10 min file needs about a third of the memory."""
    x = np.asarray(x, dtype=np.float64)
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(N_FFT, 2 * N_FFT)])
    n = 1 + (xp.shape[-1] - N_FFT) // HOP
    out = np.empty(x.shape[:-1] + (N_BINS, n), dtype=np.complex128)
    w = dsp._win(N_FFT)
    for l0 in range(0, n, CHUNK):
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(l0, min(n, l0 + CHUNK))[:, None]
        out[..., l0:l0 + len(idx)] = np.swapaxes(np.fft.rfft(xp[..., idx] * w, axis=-1), -1, -2)
    return out


def istft(X, length):
    """Inverse of stft(); the output is time-aligned with the input of stft()."""
    return dsp.istft(X, length, N_FFT, HOP)


def n_frames(n_samples):
    """Number of STFT frames stft() returns for n_samples samples."""
    return 1 + (n_samples + 2 * N_FFT) // HOP


def frame_centers(n):
    """Centre sample index of frames 0..n-1 (negative for the first frames, which overlap the pad)."""
    return np.arange(n) * HOP - N_FFT // 2


def frame_of_sample(s):
    """Index of the frame whose centre is nearest to sample index s."""
    return np.rint((np.asarray(s) + N_FFT // 2) / HOP).astype(int)


def valid_frames(n):
    """(n,) bool: frames that lie entirely inside the signal (not overlapping stft()'s zero pad)."""
    l = np.arange(n)
    return (l >= N_FFT // HOP) & (l <= n - 1 - 2 * N_FFT // HOP)


def bins_between(f_lo, f_hi):
    """slice of bin indices with f_lo <= f <= f_hi."""
    k = np.nonzero((FREQS >= f_lo) & (FREQS <= f_hi))[0]
    return slice(int(k[0]), int(k[-1]) + 1)


# ------------------------------------------------------------------ geometry and coherence models
def steer(f, tau):
    """Center-referenced steering vectors a(f, tau); shape broadcast(f, tau) + (2,)."""
    ph = np.pi * np.asarray(f) * np.asarray(tau)
    return np.stack([np.exp(1j * ph), np.exp(-1j * ph)], axis=-1)


def plane_coherence(f, tau):
    """Gamma_s = exp(+j 2 pi f tau): coherence (and Phi_AB phase) of a plane wave with TDOA tau."""
    return np.exp(2j * np.pi * np.asarray(f) * np.asarray(tau))


def diffuse_model(f, d, model="sinc"):
    """Diffuse-noise coherence at spacing d: 'sinc' = sinc(2fd/c) (3-D), 'j0' = J0(2 pi f d/c) (2-D)."""
    x = 2 * np.asarray(f, dtype=float) * np.asarray(d, dtype=float) / C
    if model == "sinc":
        return np.sinc(x)
    if model == "j0":
        return j0(np.pi * x)
    raise ValueError(f"unknown coherence model {model!r}")


def symmetrize_rtf(G):
    """RTF normalized to Mic A, g = [1, G] with G over frequency (..., F) -> center-referenced
    g~ = g / sqrt(g1 g2), shape (..., F, 2). The phase of G is unwrapped along frequency before
    halving, so a plane wave (G = exp(-j 2 pi f tau)) gives exactly steer(f, tau)."""
    G = np.asarray(G)
    half = np.sqrt(np.maximum(np.abs(G), 1e-12)) * np.exp(0.5j * np.unwrap(np.angle(G), axis=-1))
    return np.stack([1 / half, half], axis=-1)


def tdoa_gcc(x_a, x_b, fs=FS, max_tau=None, interp=16):
    """GCC-PHAT TDOA in our convention (tau > 0: Mic B later), i.e. lab.dsp.gcc_phat(x_b, x_a)."""
    return dsp.gcc_phat(x_b, x_a, fs=fs, max_tau=max_tau, interp=interp)


def hpf(x, fc, fs=FS, order=2):
    """Causal Butterworth high-pass applied identically to every row (S0 step 1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] == 0:                     # scipy's sosfilt rejects empty input
        return x.copy()
    return sosfilt(butter(order, fc, btype="high", fs=fs, output="sos"), x, axis=-1)


# ------------------------------------------------------------------ spectra, smoothing, coherence
def smooth(P, lam, init=None):
    """First-order recursive average along the last (frame) axis:
    y[l] = lam*y[l-1] + (1-lam)*P[l], with y[-1] = init (default P[..., 0], as lab.dsp.recursive_avg).
    Returns (y, y[..., -1]); pass the second value as init to continue on the next chunk."""
    P = np.asarray(P)
    init = P[..., 0] if init is None else np.asarray(init)
    y, _ = lfilter([1.0 - lam], [1.0, -lam], P, axis=-1, zi=lam * init[..., None])
    return y, y[..., -1]


def spectra(X):
    """Instantaneous (|X_A|^2, |X_B|^2, X_A conj(X_B)) of a 2-channel STFT X (2, F, N)."""
    return (X[0].real ** 2 + X[0].imag ** 2, X[1].real ** 2 + X[1].imag ** 2, X[0] * np.conj(X[1]))


def coherence(paa, pbb, pab):
    """Complex coherence Phi_AB / sqrt(Phi_AA Phi_BB)."""
    return pab / np.sqrt(paa * pbb + EPS)


def msc(paa, pbb, pab):
    """Magnitude-squared coherence |Phi_AB|^2 / (Phi_AA Phi_BB)."""
    return (pab.real ** 2 + pab.imag ** 2) / (paa * pbb + EPS)


def msc_weight(m):
    """S1 bin weight W = clip(MSC / (1 - MSC + 0.05), 0, 20)."""
    return np.clip(m / (1.0 - m + 0.05), 0.0, 20.0)


def smoothed_spectra(X, sl, lams, chunk=CHUNK):
    """Stream recursively smoothed spectra of X[:, sl, :] for each lambda in lams.
    Yields (l0, l1, {lam: (paa, pbb, pab)}) per chunk of frames; filter state carries over."""
    state = {lm: None for lm in lams}
    for l0 in range(0, X.shape[-1], chunk):
        l1 = min(X.shape[-1], l0 + chunk)
        raw = spectra(X[:, sl, l0:l1])
        out = {}
        for lm in lams:
            st = state[lm] or [None, None, None]
            res = [smooth(r, lm, s) for r, s in zip(raw, st)]
            out[lm] = tuple(r[0] for r in res)
            state[lm] = [r[1] for r in res]
        yield l0, l1, out


def _clip_mag(G, m):
    a = np.abs(G)
    return np.where(a > m, G * (m / np.maximum(a, EPS)), G)


def cdr_doa(Gx, Gn, Gs):
    """DOA-dependent CDR (Schwarz & Kellermann prop1, DESIGN M4), Gs = plane_coherence(f, tau_k).
    |Gx| is clipped to 0.999 and Gn to <= 0.95 first. Returns CDR >= 0 (linear power ratio)."""
    Gx = _clip_mag(Gx, 0.999)
    Gn = np.minimum(Gn, 0.95)
    num = np.real(np.conj(Gs) * (Gn - Gx))
    den = np.real(np.conj(Gs) * Gx) - 1.0
    return np.maximum(0.0, num / np.minimum(den, -1e-6))


def cdr_nodoa(Gx, Gn):
    """DOA-independent CDR (Schwarz & Kellermann prop3, DESIGN M4); same clipping as cdr_doa()."""
    Gx = _clip_mag(Gx, 0.999)
    Gn = np.minimum(Gn, 0.95)
    re, a2 = Gx.real, Gx.real ** 2 + Gx.imag ** 2
    root = np.sqrt(np.maximum(Gn ** 2 * re ** 2 - Gn ** 2 * a2 + Gn ** 2 - 2 * Gn * re + a2, 0.0))
    return np.maximum(0.0, (Gn * re - a2 - root) / np.minimum(a2 - 1.0, -1e-6))


# ------------------------------------------------------------------ steered response (S1)
def srp(G, W, f, grid):
    """Weighted steered response P[l, i] = sum_k W Re{G e^(-j 2 pi f_k grid_i)} / sum_k W.
    G, W: (K, L) coherence and weights for bins f (K,); grid (n,) in seconds. Returns (L, n)."""
    ph = 2 * np.pi * np.outer(f, grid)
    A = np.concatenate([(W * G.real).T, (W * G.imag).T], axis=1)
    B = np.concatenate([np.cos(ph), np.sin(ph)], axis=0)
    return (A @ B) / np.maximum(W.sum(axis=0), EPS)[:, None]


def coherent_fit(G, gn, f, grid):
    """Per-frame two-component fit G ~ q exp(+j 2 pi f tau) + (1 - q) gn (one plane wave in diffuse
    noise), least squares over the bins for every tau on the grid; q >= 0 in closed form.
    Returns J (L, n): explained power (Re sum D E*)^2 / sum |E|^2 (0 where q < 0), and Q (L, n): q,
    with D = G - gn and E = exp(+j 2 pi f tau) - gn. Unlike srp(), a diffuse field (G = gn) adds
    nothing at any tau, so it cannot pull the estimate toward broadside.
    G: (K, L) coherence for bins f (K,), gn: (K,) real noise coherence, grid (n,) seconds."""
    D = G - gn[:, None]
    th = 2 * np.pi * np.outer(f, grid)
    c = np.cos(th)
    num = D.real.T @ c + D.imag.T @ np.sin(th) - (gn[:, None] * D.real).sum(axis=0)[:, None]
    den = np.maximum((1 - 2 * gn[:, None] * c + gn[:, None] ** 2).sum(axis=0), EPS)[None, :]
    return np.where(num > 0, num ** 2 / den, 0.0), num / den


def srp_peak(P, grid):
    """Row-wise argmax of P on a uniform grid with parabolic refinement -> (tau, peak value)."""
    rows = np.arange(P.shape[0])
    i = np.argmax(P, axis=1)
    ic = np.clip(i, 1, P.shape[1] - 2)
    y0, y1, y2 = P[rows, ic - 1], P[rows, ic], P[rows, ic + 1]
    den = y0 - 2 * y1 + y2
    inner = (i == ic) & (den < 0)
    delta = np.clip(np.where(inner, 0.5 * (y0 - y2) / np.where(inner, den, -1.0), 0.0), -0.5, 0.5)
    tau = grid[i] + delta * (grid[1] - grid[0])
    return tau, np.where(inner, y1 - 0.25 * (y0 - y2) * delta, P[rows, i])


# ------------------------------------------------------------------ speech presence (VAD)
def speech_presence(e, valid=None, floor_s=3.0, thr_db=6.0, slope_db=1.5, bias_db=2.0):
    """Causal energy VAD. e: per-frame band energy (N,), linear power.
    Noise floor = minimum of the 20 ms-smoothed energy over the last floor_s seconds, raised by bias_db;
    p = logistic((10 log10(e_s / floor) - thr_db) / slope_db). Frames outside `valid` (default:
    valid_frames(N), i.e. frames overlapping the STFT pad) take the nearest valid frame's energy.
    Returns (p, floor) as (N,) arrays."""
    e = np.maximum(np.asarray(e, dtype=float), EPS)
    v = np.nonzero(valid_frames(len(e)) if valid is None else valid)[0]
    if len(v):
        e = e[np.clip(np.arange(len(e)), v[0], v[-1])]
    e_s, _ = smooth(e, LAM_20MS)
    w = max(1, int(round(floor_s / HOP_S)))
    padded = np.concatenate([np.full(w - 1, e_s[: w].max()), e_s])
    floor = sliding_window_view(padded, w).min(axis=1) * 10 ** (bias_db / 10)
    snr_db = 10 * np.log10(e_s / np.maximum(floor, EPS))
    return 1.0 / (1.0 + np.exp(-(snr_db - thr_db) / slope_db)), floor
