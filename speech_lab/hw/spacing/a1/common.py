"""Shared helpers for estimator A1 (mic spacing d and inter-channel skew tau0).

Conventions (same as DESIGN.md / hw/analyze.py):
  ch0 = Mic A, ch1 = Mic B.  tau = arrival at B minus arrival at A, in samples at 16 kHz.
  Cross-spectrum P_AB = E{A conj(B)}; a plane wave with B lagging A by tau gives arg P_AB = +2 pi f tau / fs.
  Skew tau0 < 0: B's sample n is taken |tau0| samples later in real time than A's sample n.
Models: gamma(f) = rho * D(f; d) * exp(+j 2 pi f tau0 / fs),
  D = numpy sinc(2 f d / c) (3-D diffuse) or J0(2 pi f d / c) (2-D diffuse), c = 343 m/s.

Loader functions are re-implemented here (same filters as hw/analyze.py) so that nothing in hw/ is imported
(importing would write bytecode into hw/__pycache__).
Run every script with:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I -B <script>
"""
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from scipy.special import j0

HERE = os.path.dirname(os.path.abspath(__file__))
HW = os.path.dirname(os.path.dirname(HERE))
FS = 16000
C = 343.0
_HP = butter(4, 80, "high", fs=FS, output="sos")

RECS = ["ambient", "sweep_L", "sweep_R", "sweep_L2", "talkA_L", "talkB_R", "talkAB_LR", "noise_LR",
        "talkA_noise", "talkC_R_quiet"]
STIM_OF = {"sweep_L2": "sweep_L"}


def load(name, hp=True):
    x, fs = sf.read(os.path.join(HW, "rec", name + ".wav"), dtype="float64")
    assert fs == FS
    x = x - x.mean(0)
    return sosfiltfilt(_HP, x, axis=0) if hp else x


def stim(name):
    d = np.load(os.path.join(HW, "stim", STIM_OF.get(name, name) + ".npz"))
    return {k: d[k].astype(np.float64) for k in d.files}


def irs():
    d = np.load(os.path.join(HW, "irs.npz"))
    return {k: d[k] for k in ["sweep_L", "sweep_R", "sweep_L2"]}


def nfft(n):
    return 1 << int(np.ceil(np.log2(n)))


def bandpass(x, lo, hi, order=4):
    return sosfiltfilt(butter(order, [lo, hi], "bandpass", fs=FS, output="sos"), x, axis=0)


def xcorr_lag(y, r):
    n = nfft(len(y) + len(r))
    cc = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(r, n)), n)
    return int(np.argmax(cc[:len(y)]))


def stretch(x, alpha):
    from scipy.signal import resample_poly
    up = resample_poly(x, 8, 1)
    t = np.arange(int(len(x) * (1 + alpha))) / (1 + alpha) * 8
    return np.interp(t, np.arange(len(up)), up)


def deconvolve(y, x, eps=1e-4):
    n = nfft(len(y) + len(x))
    X = np.fft.rfft(x, n)
    Y = np.fft.rfft(y, n, axis=0)
    H = Y * np.conj(X)[:, None] / (np.abs(X)[:, None] ** 2 + eps * np.max(np.abs(X)) ** 2)
    return np.fft.irfft(H, n, axis=0)[:len(y)]


# ---------------------------------------------------------------- cross-spectra
def frames_spec(x, nper=512, hop=256, win="hann"):
    """Per-frame spectra of a 2-channel signal. Returns f (Hz), A (nfr, nf), B (nfr, nf)."""
    w = np.hanning(nper + 2)[1:-1] if win == "hann" else np.ones(nper)
    n = (len(x) - nper) // hop + 1
    idx = np.arange(nper)[None, :] + hop * np.arange(n)[:, None]
    A = np.fft.rfft(x[idx, 0] * w, axis=1)
    B = np.fft.rfft(x[idx, 1] * w, axis=1)
    f = np.fft.rfftfreq(nper, 1 / FS)
    return f, A, B


def coh_from_frames(A, B, sel=None):
    if sel is not None:
        A, B = A[sel], B[sel]
    Pab = np.mean(A * np.conj(B), 0)
    Paa = np.mean(np.abs(A) ** 2, 0)
    Pbb = np.mean(np.abs(B) ** 2, 0)
    return Pab / np.sqrt(Paa * Pbb), Pab, Paa, Pbb


def coherence(x, nper=512, hop=256):
    f, A, B = frames_spec(x, nper, hop)
    g, Pab, Paa, Pbb = coh_from_frames(A, B)
    return f, g, Pab, Paa, Pbb


# ---------------------------------------------------------------- models and fits
def dmodel(f, d, kind="sinc"):
    """Diffuse coherence for spacing d (m) at frequencies f (Hz)."""
    if kind == "sinc":
        return np.sinc(2 * np.outer(np.atleast_1d(d), f) / C)
    return j0(2 * np.pi * np.outer(np.atleast_1d(d), f) / C)


D_GRID = np.arange(0.002, 0.0601, 0.0001)        # 0.2 .. 6 cm, 0.1 mm
T_GRID = np.arange(-2.5, 1.5001, 0.005)            # samples


def fit_diffuse(f, g, kind="sinc", w=None, d_grid=D_GRID, t_grid=T_GRID, rho_max=1.0, refine=True):
    """Complex LS fit of g(f) ~ rho * D(f;d) * exp(j 2 pi f tau0/fs). Returns dict(d, tau0, rho, rmse, ph_rms_deg)."""
    w = np.ones_like(f) if w is None else w
    Dm = dmodel(f, d_grid, kind)                       # (nd, nf)
    DW = Dm * w
    den = np.sum(DW * Dm, 1)                           # (nd,)
    tot = np.sum(w * np.abs(g) ** 2)
    best = (np.inf, None, None, None)
    for t in t_grid:
        z = g * np.exp(-2j * np.pi * f * t / FS)
        num = DW @ z.real                              # (nd,)
        rho = np.clip(num / den, 0, rho_max)
        cost = tot - 2 * rho * num + rho ** 2 * den
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (cost[i], d_grid[i], t, rho[i])
    _, d, t, rho = best
    if refine:
        from scipy.optimize import minimize

        def cost(p):
            dd, tt, rr = p
            m = rr * dmodel(f, dd, kind)[0] * np.exp(2j * np.pi * f * tt / FS)
            return np.sum(w * np.abs(g - m) ** 2)
        r = minimize(cost, [d, t, rho], method="L-BFGS-B",
                     bounds=[(d_grid[0], d_grid[-1]), (t_grid[0], t_grid[-1]), (0, rho_max)])
        d, t, rho = r.x
    m = rho * dmodel(f, d, kind)[0] * np.exp(2j * np.pi * f * t / FS)
    res = g - m
    rmse = float(np.sqrt(np.sum(w * np.abs(res) ** 2) / np.sum(w)))
    ph = np.angle(g * np.exp(-2j * np.pi * f * t / FS))
    ph_rms = float(np.degrees(np.sqrt(np.sum(w * ph ** 2) / np.sum(w))))
    return dict(d_cm=100 * float(d), tau0=float(t), rho=float(rho), rmse=rmse, ph_rms_deg=ph_rms)


def phase_slope(f, g, w=None, lo=None, hi=None, t_grid=np.arange(-3, 3.0001, 0.002)):
    """Delay (samples) maximising Re sum w g exp(-j 2 pi f t/fs) (= weighted linear-phase fit, no intercept),
    plus an LS fit of unwrapped phase with intercept. Returns (t_max, slope_ls, intercept_deg)."""
    m = np.ones_like(f, bool)
    if lo is not None:
        m &= f >= lo
    if hi is not None:
        m &= f <= hi
    ff, gg = f[m], g[m]
    ww = np.abs(gg) if w is None else w[m]
    E = np.exp(-2j * np.pi * np.outer(t_grid, ff) / FS)
    s = np.real(E @ (ww * gg / np.maximum(np.abs(gg), 1e-12)))
    t_max = float(t_grid[int(np.argmax(s))])
    # refine with a parabola
    i = int(np.argmax(s))
    if 0 < i < len(t_grid) - 1:
        a, b, c = s[i - 1], s[i], s[i + 1]
        den = a - 2 * b + c
        if den < 0:
            t_max += 0.5 * (a - c) / den * (t_grid[1] - t_grid[0])
    # LS on phase after removing t_max (small residual, no unwrap problems)
    ph = np.angle(gg * np.exp(-2j * np.pi * ff * t_max / FS))
    Aw = np.stack([2 * np.pi * ff / FS, np.ones_like(ff)], 1) * np.sqrt(ww)[:, None]
    coef, *_ = np.linalg.lstsq(Aw, ph * np.sqrt(ww), rcond=None)
    return t_max, t_max + float(coef[0]), float(np.degrees(coef[1]))


def gcc_phat(a, b, maxlag=8.0, up=64, lo=None, hi=None):
    """tau = delay of b relative to a (samples), GCC-PHAT with 1/up resolution, optional band limits."""
    n = nfft(len(a) + len(b)) * 2
    G = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    G /= np.abs(G) + 1e-12
    if lo is not None or hi is not None:
        f = np.fft.rfftfreq(n, 1 / FS)
        G[(f < (lo or 0)) | (f > (hi or FS))] = 0
    cc = np.fft.irfft(G, n * up)
    L = int(maxlag * up)
    cc = np.concatenate([cc[-L:], cc[:L + 1]])
    return float((np.argmax(cc) - L) / up)


def onset_of(name, key=None):
    """Sample index (in the 12 s recording) where the reference's t=0 lands (reference includes 0.25 s lead)."""
    s = stim(name)
    key = key or ("L" if np.any(s["L"] != 0) else "R")
    y = load(name)[:, 0]
    r = s[key]
    return xcorr_lag(bandpass(y, 150, 7000), bandpass(r, 150, 7000))


def r3(x):
    return float(np.round(x, 3))
