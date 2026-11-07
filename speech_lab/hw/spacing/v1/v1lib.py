"""Verifier V1: independent helpers (nothing imported from hw/ or hw/spacing/a1).

Conventions: ch0 = Mic A, ch1 = Mic B. tau = arrival at B minus arrival at A (samples, 16 kHz).
P_AB = E{A conj(B)}; a plane wave reaching B tau samples after A gives arg P_AB = +2 pi f tau / fs.
Run with: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I -B <script>
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.special import gamma as G
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
HW = os.path.abspath(os.path.join(HERE, "..", ".."))
FS = 16000
C = 343.0
_HP = butter(4, 80, "high", fs=FS, output="sos")


def rpath(name):
    """Recording path: 'rec/<n>', 'rec2/<n>' or 'realset/rec/<n>'."""
    return os.path.join(HW, name + ".wav")


def load(name, hp=True):
    x, fs = sf.read(rpath(name), dtype="float64")
    assert fs == FS and x.ndim == 2 and x.shape[1] == 2
    x = x - x.mean(0)
    return sosfiltfilt(_HP, x, axis=0) if hp else x


def nfft(n):
    return 1 << int(np.ceil(np.log2(n)))


def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic Hann


# ------------------------------------------------------------------ cross-spectra (Welch, my own frames)
def xspec(x, nper=1024, hop=512, segs=None):
    """Welch sums over frames. segs: list of (start, stop) sample ranges (frames never cross a range).
    Returns f, Pab, Paa, Pbb (means over frames), nfr, and per-frame arrays for jackknife (A conj B, |A|^2, |B|^2)."""
    segs = segs or [(0, len(x))]
    w = hann(nper)
    fa, fb = [], []
    for a, b in segs:
        n = (b - a - nper) // hop + 1
        if n <= 0:
            continue
        idx = a + np.arange(nper)[None, :] + hop * np.arange(n)[:, None]
        fa.append(np.fft.rfft(x[idx, 0] * w, axis=1))
        fb.append(np.fft.rfft(x[idx, 1] * w, axis=1))
    A = np.concatenate(fa)
    B = np.concatenate(fb)
    f = np.fft.rfftfreq(nper, 1 / FS)
    ab, aa, bb = A * np.conj(B), np.abs(A) ** 2, np.abs(B) ** 2
    return f, ab.mean(0), aa.mean(0), bb.mean(0), len(A), (ab, aa, bb)


def delay_ls(f, Pab, Paa, Pbb, nd, lo, hi, intercept=True):
    """Delay (samples) from the cross-spectrum phase over [lo, hi] Hz: coarse grid search, then weighted LS on the
    residual phase with inverse-variance weights 2 nd |g|^2 / (1 - |g|^2). Returns (tau, intercept_deg)."""
    m = (f >= lo) & (f <= hi)
    ff, g = f[m], (Pab / np.sqrt(Paa * Pbb))[m]
    grid = np.arange(-3, 3.0001, 0.005)
    s = np.real(np.exp(-2j * np.pi * np.outer(grid, ff) / FS) @ g)
    t0 = grid[int(np.argmax(s))]
    ph = np.angle(g * np.exp(-2j * np.pi * ff * t0 / FS))
    a2 = np.clip(np.abs(g) ** 2, 1e-6, 0.9999)
    w = 2 * nd * a2 / (1 - a2)
    X = np.stack([2 * np.pi * ff / FS] + ([np.ones_like(ff)] if intercept else []), 1)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], ph * sw, rcond=None)
    return float(t0 + coef[0]), (float(np.degrees(coef[1])) if intercept else 0.0)


def delay_jack(fr, f, lo, hi, nblk=10, intercept=True):
    """Delay estimate plus jackknife standard error over nblk contiguous frame blocks."""
    ab, aa, bb = fr
    n = len(ab)
    full = delay_ls(f, ab.mean(0), aa.mean(0), bb.mean(0), n, lo, hi, intercept)
    edges = np.linspace(0, n, nblk + 1).astype(int)
    js = []
    for i in range(nblk):
        keep = np.ones(n, bool)
        keep[edges[i]:edges[i + 1]] = False
        js.append(delay_ls(f, ab[keep].mean(0), aa[keep].mean(0), bb[keep].mean(0), keep.sum(), lo, hi, intercept)[0])
    js = np.array(js)
    se = float(np.sqrt((nblk - 1) / nblk * np.sum((js - js.mean()) ** 2)))
    return full[0], se, full[1]


def xcorr_delay(x, lo, hi, up=64, maxlag=4.0):
    """Time-domain cross-correlation delay of ch1 re ch0 after an ideal band-pass [lo, hi] (FFT), x up-sampled."""
    n = nfft(len(x)) * 2
    Af, Bf = np.fft.rfft(x[:, 0], n), np.fft.rfft(x[:, 1], n)
    f = np.fft.rfftfreq(n, 1 / FS)
    Gf = Bf * np.conj(Af)
    Gf[(f < lo) | (f > hi)] = 0
    # zero-padded spectrum -> up-sampled cross-correlation near zero lag
    L = int(maxlag * up)
    cc = np.fft.irfft(Gf, n * up)
    cc = np.concatenate([cc[-L:], cc[:L + 1]])
    i = int(np.argmax(cc))
    a, b, c = cc[i - 1], cc[i], cc[i + 1]
    return float((i - L + 0.5 * (a - c) / (a - 2 * b + c)) / up)


def gcc_phat(a, b, lo=300, hi=6500, up=64, maxlag=4.0):
    n = nfft(len(a)) * 4
    Gf = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    Gf /= np.abs(Gf) + 1e-20
    f = np.fft.rfftfreq(n, 1 / FS)
    Gf[(f < lo) | (f > hi)] = 0
    L = int(maxlag * up)
    cc = np.fft.irfft(Gf, n * up)
    cc = np.concatenate([cc[-L:], cc[:L + 1]])
    return float((int(np.argmax(cc)) - L) / up)


def phase_delay_window(seg, lo=300, hi=6000, n=4096):
    """Delay of ch1 re ch0 for one windowed segment: magnitude-weighted LS fit of the cross-spectrum phase (no intercept
    after a coarse search)."""
    A, B = np.fft.rfft(seg[:, 0], n), np.fft.rfft(seg[:, 1], n)
    f = np.fft.rfftfreq(n, 1 / FS)
    m = (f >= lo) & (f <= hi)
    P = (A * np.conj(B))[m]
    ff = f[m]
    grid = np.arange(-3, 3.0001, 0.005)
    s = np.real(np.exp(-2j * np.pi * np.outer(grid, ff) / FS) @ P)
    t0 = grid[int(np.argmax(s))]
    ph = np.angle(P * np.exp(-2j * np.pi * ff * t0 / FS))
    w = np.abs(P)
    X = np.stack([2 * np.pi * ff / FS, np.ones_like(ff)], 1) * np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X, ph * np.sqrt(w), rcond=None)
    return float(t0 + coef[0]), float(np.degrees(coef[1]))


# ------------------------------------------------------------------ sweeps
def stretch(x, alpha):
    """x as played on a clock (1 + alpha) faster/slower relative to the recorder (16x poly up-sampling + linear)."""
    if alpha == 0:
        return x.copy()
    up = resample_poly(x, 16, 1)
    t = np.arange(int(len(x) * (1 + alpha))) / (1 + alpha) * 16
    return np.interp(t, np.arange(len(up)), up)


def deconv(y, ref, eps=1e-4):
    n = nfft(len(y) + len(ref))
    X = np.fft.rfft(ref, n)
    Y = np.fft.rfft(y, n, axis=0)
    H = Y * np.conj(X)[:, None] / (np.abs(X)[:, None] ** 2 + eps * np.max(np.abs(X)) ** 2)
    return np.fft.irfft(H, n, axis=0)  # full circular length (negative times wrap to the end)


def frames_in(h, i0, i1, nper=256, hop=128):
    """Welch frames inside [i0, i1) of a 2-channel response; returns f, sums (ab, aa, bb), nfr."""
    w = hann(nper)
    n = (i1 - i0 - nper) // hop + 1
    idx = i0 + np.arange(nper)[None, :] + hop * np.arange(n)[:, None]
    A = np.fft.rfft(h[idx, 0] * w, axis=1)
    B = np.fft.rfft(h[idx, 1] * w, axis=1)
    return np.fft.rfftfreq(nper, 1 / FS), (A * np.conj(B)).sum(0), (np.abs(A) ** 2).sum(0), (np.abs(B) ** 2).sum(0), n


# ------------------------------------------------------------------ spatial-coherence models
def coh_axisym(x, m):
    """Coherence for an axisymmetric field with energy density p(u) ~ (1-u^2)^m, u = cos(angle to the mic axis);
    x = k d. m = 0: 3-D isotropic (sinc); m = -0.5: J0; m > 0: broadside-heavy. Real (symmetric field)."""
    x = np.maximum(np.asarray(x, float), 1e-9)
    nu = m + 0.5
    return G(m + 1.5) * (2.0 / x) ** nu * jv(nu, x)


def kd(f, d):
    return 2 * np.pi * np.asarray(f) * d / C
