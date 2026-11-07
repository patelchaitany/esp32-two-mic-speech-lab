"""Shared helpers for the hw/realset/bizarre experiments (numpy/scipy only; reads hw/ data, never plays or records).

Conventions as in methods/_core/API.md: x[0] = Mic A, x[1] = Mic B, fs 16 kHz, tau > 0 = Mic B hears later.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, csd, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REALSET = os.path.dirname(HERE)
HW = os.path.dirname(REALSET)
LAB = os.path.dirname(HW)
for p in (LAB, HW, os.path.join(REALSET, "build")):
    if p not in sys.path:
        sys.path.insert(0, p)

FS, C = 16000, 343.0
D_M = 0.0119
TAU0 = -1.015                      # samples, current boot (calib_session.json)
CALIB = json.load(open(os.path.join(REALSET, "calib_session.json")))
HP80 = butter(4, 80, "high", fs=FS, output="sos")
BANDS = [(150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]


def load_wav(path, hp=True):
    x, fs = sf.read(path, dtype="float64")
    assert fs == FS, path
    x = x - x.mean(0)
    if hp:
        x = sosfiltfilt(HP80, x, axis=0)
    return x.T                      # (2, T)


def load_take(t):
    z = np.load(os.path.join(REALSET, "img", t + ".npz"))
    return {k: z[k] for k in z.files}


def coh_stats(x, nperseg=1024):
    """x (2, T) -> f, Saa, Sbb, Sab = E[A conj(B)] (phase = +2 pi f tau when B lags)."""
    f, ab = csd(x[0], x[1], FS, nperseg=nperseg)   # scipy csd(x, y) = E[conj(X) Y] -> conj(A) B
    ab = np.conj(ab)
    _, aa = csd(x[0], x[0], FS, nperseg=nperseg)
    _, bb = csd(x[1], x[1], FS, nperseg=nperseg)
    return f, aa.real, bb.real, ab


def skew_phase(f, aa, bb, ab, lo=150.0, hi=7000.0, grid=None):
    """Skew (samples) = slope of the noise cross-spectrum phase, weighted by MSC/(1-MSC) (inverse phase variance).
    In a diffuse field sinc(2fd/c) > 0 below c/2d (14.4 kHz at 1.19 cm), so the phase is only the skew."""
    G = ab / np.sqrt(aa * bb)
    m = (f > lo) & (f < hi)
    msc = np.clip(np.abs(G[m]) ** 2, 0, 0.999)
    w = msc / (1 - msc)
    g = G[m] / (np.abs(G[m]) + 1e-12)
    if grid is None:
        grid = np.arange(-2.5, 1.0, 0.002)
    sc = np.real(np.exp(-2j * np.pi * np.outer(grid, f[m]) / FS) @ (w * g))
    i = int(np.argmax(sc))
    if 0 < i < len(grid) - 1:
        a, b, c = sc[i - 1], sc[i], sc[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        return float(grid[i] + off * (grid[1] - grid[0]))
    return float(grid[i])


def spacing_mag(f, aa, bb, ab, lo=150.0, hi=7000.0, skew=None):
    """Spacing (cm) and rho from |coherence| (after removing the skew phase) vs rho*sinc(2fd/c), grid search."""
    G = ab / np.sqrt(aa * bb)
    m = (f > lo) & (f < hi)
    if skew is not None:
        G = G * np.exp(-2j * np.pi * f * skew / FS)
    y = np.real(G[m])
    best = None
    for d in np.arange(0.004, 0.04, 0.0001):
        s = np.sinc(2 * f[m] * d / C)
        rho = float(np.clip(np.dot(y, s) / np.dot(s, s), 0.2, 1.0))
        e = float(np.mean((y - rho * s) ** 2))
        if best is None or e < best[0]:
            best = (e, d, rho)
    return best[1] * 100, best[2], float(np.sqrt(best[0]))


def band_gain_db(f, aa, bb, bands=BANDS):
    """B re A level (dB) per band."""
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(float(10 * np.log10(np.sum(bb[m]) / np.sum(aa[m]))))
    return out


def frac_delay(x, tau):
    """Delay x (..., T) by tau samples (fractional, FFT, zero-padded)."""
    T = x.shape[-1]
    n = 1 << int(np.ceil(np.log2(T + 512)))
    f = np.fft.rfftfreq(n)
    X = np.fft.rfft(x, n, axis=-1)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), n, axis=-1)[..., :T]


def correct_board(x, tau0=TAU0):
    """Advance Mic B by the skew (tau0 < 0: B leads -> delay B by |tau0|) and apply the broadband B gain (+1.3 dB
    is NOT applied here: callers that need it use methods._core.calib with CALIB)."""
    y = np.array(x, dtype=np.float64)
    y[1] = frac_delay(y[1], -tau0)
    return y


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def db(p):
    return float(10 * np.log10(max(float(p), 1e-30)))
