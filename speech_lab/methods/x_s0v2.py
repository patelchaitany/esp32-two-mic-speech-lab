"""x_s0v2: blind self-calibration v2 for a 2-mic board (experimental; does not touch methods/_core).

estimate(x, fs, info) -> calib_json dict (schema of methods/_core/API.md "calib.json"), measured on the
input itself:
  * spacing over 0.6-10 cm: free-rho fit (Huber-reweighted) of rho*sinc (3-D) and rho*J0 (2-D) to the
    real part of the skew-corrected noise coherence over 400-6500 Hz (below ~300 Hz the board's per-mic
    LF junk lowers the coherence; the INMP441 decimation pass band ends at 0.423 fs = 6.77 kHz), tonal
    bins (> 4 dB above the 9-bin median spectrum) excluded; sinc unless J0 is clearly better (weighted
    residual < 0.8 x sinc's); prior spacing if the fit RMS > 0.15 (as calib.py), at a grid edge or SE > 25 %;
  * a CONTINUOUS skew over +-2.5 samples: coarse search on the coherence phase over 150-1200 Hz (no
    wrapping and no coherence sign change for d <= 10 cm), then a weighted (Huber) least-squares phase
    slope over 150-6500 Hz with the sign of the fitted diffuse model; skew 0 if unidentifiable, including
    when the per-band slopes (500-6500 Hz) spread by more than 0.5 samples (directional noise component);
  * the HPF-corner difference (atan(fA/f) - atan(fB/f) ~ (fA - fB)/f) jointly with the skew; kept only
    when significant (|t| >= 3 and >= 1 deg at 125 Hz), else [60, 60] (= no phase);
  * per-band gain of B re A (18 one-third-octave bands as calib.BANDS_HZ) from pooled noise powers; the
    json carries the CORRECTION applied to X_B (= -(B re A));
  * the measured noise coherence gamma_n (blended with the model as calib.py does), because calib.py's
    own measured gamma_n is rotated by its blind (wrong on this board) skew;
  * jackknife standard errors over ~1 s blocks of noise, outlier blocks (directional events) removed,
    per-parameter confidences, and fallbacks: prior spacing / skew 0 / no gain when unidentifiable.
Noise frames: valid STFT frames whose 76 ms-smoothed level is within 3 dB (relaxed up to 6 dB if fewer
than 3 s qualify) of the 4 s sliding minimum in both 200-1000 Hz and 1-6.5 kHz, with a 100 ms guard
around every rejected frame (onsets, reverberant tails). Needs >= 60 noise frames (~0.5 s); the skew is
used from 1 s of noise frames on, the spacing from 1.5 s. Report: methods/x_s0v2.md.
Conventions as methods/_core (tau > 0 = Mic B late; correction X_B <- g exp(+j(2 pi f tau0 + corner))).
Options: info['x_s0v2_skew_candidates'] = [..] (samples) snaps the skew to the nearest allowed value (e.g.
[0, 0.5] reproduces calib.py's one-bus I2S rule); default None = continuous (needed on the real board).
info['x_s0v2_head_s'] = s estimates from the first s seconds only (boot-time / causal calibration).
gamma_n_for(x, fs, cfg) completes an existing json that lacks 'gamma_n' (see x_s0v2.md section 4).
Deterministic and stateless. CLI: python -m methods.x_s0v2 file.wav [spacing_prior_m]
"""
import json
import sys

import numpy as np
from scipy.ndimage import binary_dilation, minimum_filter1d

from methods._core import calib as _cb
from methods._core import common as cm

F = cm.FREQS
BANDS_HZ = _cb.BANDS_HZ
BAND_BINS = _cb.BAND_BINS

PRE_HPF_HZ, PRE_HPF_ORDER = 90.0, 4      # identical on both rows: removes the < 60 Hz incoherent junk
LEVEL_BANDS = ((200.0, 1000.0), (1000.0, 6500.0))
FLOOR_WIN_S = 4.0
THRESH_DB = (3.0, 4.5, 6.0)
MIN_SEL_S = 3.0
GUARD_S = 0.10
BLOCK_S = 1.0
MAX_BLOCKS = 24
COARSE_BAND = (150.0, 1200.0)
PHASE_BAND = (150.0, 6500.0)
FIT_BAND = (400.0, 6500.0)                # below ~300 Hz the per-mic LF junk lowers the coherence (board)
TONAL_DB = 4.0                            # bins this far above the 9-bin median spectrum are tonal (excluded)
SPREAD_MAX = 0.5                          # band-slope spread (samples) above which the skew counts as unidentifiable
FIT_RMS_MAX = 0.15                        # as calib.py: worse spacing fits fall back to the prior
GAIN_SE_REF_DB = 0.3                      # band gains with a larger jackknife SE are shrunk to the median
SKEW_MAX = 2.5                            # samples
D_GRID = np.arange(0.006, 0.1000001, 0.0001)
J0_MARGIN = 0.8
W_CAP = 25.0                              # inverse phase variance cap (|Gamma| = 0.98)
HUBER_K = 2.5
CORNER_F0 = 60.0
JSON_D_RANGE = (0.01, 0.1)                # calib.load_json accepts spacing_m only in [1, 10] cm


# ------------------------------------------------------------------ helpers
def _k(band):
    return (F >= band[0]) & (F <= band[1])


def _corner_reg(f):
    """d/dDelta of atan((f0 + D/2)/f) - atan((f0 - D/2)/f) at D = 0."""
    f = np.maximum(f, 1e-9)
    return f / (f * f + CORNER_F0 ** 2)


def _corners(delta):
    return [float(CORNER_F0 + delta / 2), float(CORNER_F0 - delta / 2)]


def _phase_model(tau_s, delta):
    fa, fb = _corners(delta)
    return 2 * np.pi * F * tau_s + _cb.corner_phase(F, fa, fb)


def prep(x, fs):
    """DC removal + identical 4th-order 90 Hz Butterworth high-pass on both rows -> STFT (2, 257, N)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError("x must be (2, T)")
    if fs != cm.FS:
        raise ValueError("x_s0v2 expects fs = 16000")
    x = x - x.mean(axis=1, keepdims=True)
    return cm.stft(cm.hpf(x, PRE_HPF_HZ, order=PRE_HPF_ORDER))


def select_noise(X):
    """Noise frames: level within thr dB of the sliding 4 s minimum in both level bands, 100 ms guard.
    Returns (mask (N,), thr used, excess dB (N,))."""
    N = X.shape[-1]
    valid = cm.valid_frames(N)
    P = 0.5 * (X[0].real ** 2 + X[0].imag ** 2 + X[1].real ** 2 + X[1].imag ** 2)
    w = max(3, int(round(FLOOR_WIN_S / cm.HOP_S)))
    excess = np.full(N, -np.inf)
    v = np.nonzero(valid)[0]
    if not len(v):
        return np.zeros(N, bool), float(THRESH_DB[-1]), excess
    for band in LEVEL_BANDS:
        e = P[_k(band)].mean(axis=0)
        e = e[np.clip(np.arange(N), v[0], v[-1])]          # pad frames take the nearest valid frame's level
        e_s, _ = cm.smooth(e, cm.LAM_76MS)
        lev = 10 * np.log10(e_s + cm.EPS)
        fl = minimum_filter1d(np.where(valid, lev, 1e6), size=w, mode="nearest")
        excess = np.maximum(excess, lev - fl)
    g = int(round(GUARD_S / cm.HOP_S))
    ok, thr = np.zeros(N, bool), THRESH_DB[-1]
    for thr in THRESH_DB:
        ok = valid & (excess < thr)
        if g and ok.any():
            ok &= ~binary_dilation(~ok, iterations=g)
        if ok.sum() * cm.HOP_S >= MIN_SEL_S:
            break
    return ok, float(thr), excess


def _blocks(X, mask):
    """Pooled (S, PA, PB) per block of selected frames (time order). Returns list of (S, PA, PB)."""
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return []
    per = int(round(BLOCK_S / cm.HOP_S))
    if len(idx) < 3 * per:                     # short inputs: >= 3 blocks of >= 0.2 s so the jackknife still works
        per = max(25, len(idx) // 3)
    per = max(per, int(np.ceil(len(idx) / MAX_BLOCKS)))
    out = []
    for i0 in range(0, len(idx), per):
        sl = idx[i0:i0 + per]
        if len(sl) < per // 2 and out:          # short tail: merge into the previous block
            S, A, B = out.pop()
            paa, pbb, pab = cm.spectra(X[:, :, sl])
            out.append((S + pab.sum(1), A + paa.sum(1), B + pbb.sum(1)))
            continue
        paa, pbb, pab = cm.spectra(X[:, :, sl])
        out.append((pab.sum(1), paa.sum(1), pbb.sum(1)))
    return out


# ------------------------------------------------------------------ fits on pooled statistics
def tonal_mask(PA, PB):
    """(257,) bool: bins whose power is > TONAL_DB above the 9-bin running median (fan / engine tones,
    usually directional), plus their neighbours."""
    from scipy.ndimage import median_filter
    lp = 10 * np.log10(np.sqrt(PA * PB) + cm.EPS)
    t = (lp - median_filter(lp, size=9, mode="nearest")) > TONAL_DB
    return t | np.roll(t, 1) | np.roll(t, -1)


def fit_spacing(g, models=("sinc", "j0"), use=None, robust=2):
    """Free-rho fit of Re Gamma (257,) over FIT_BAND (bins in `use`) for each model on D_GRID, weights
    1 / max(1 - Gamma_s^2, 0.1)^2, then `robust` Huber re-weightings on the residuals.
    Returns {model: (wres, d, rho, rms, at_edge)}; rms = unweighted RMS over the used bins."""
    k = _k(FIT_BAND) & (np.ones(cm.N_BINS, bool) if use is None else use)
    f, y = F[k], g[k]
    ys = np.convolve(np.pad(g, 2, mode="edge"), np.ones(5) / 5, mode="valid")[k]
    w0 = 1 / np.maximum(1 - np.clip(ys, -1, 1) ** 2, 0.1) ** 2
    out = {}
    for model in models:
        M = cm.diffuse_model(f[None, :], D_GRID[:, None], model)
        hub = np.ones_like(w0)
        for rep in range(robust + 1):
            w = w0 * hub
            w = w / w.sum()
            rho = np.clip((M * w) @ y / np.maximum((M * M * w).sum(1), cm.EPS), 0.5, 1.0)
            res = np.sum(w * (y[None, :] - rho[:, None] * M) ** 2, axis=1)
            i = int(np.argmin(res))
            di = D_GRID[i]
            if 0 < i < len(D_GRID) - 1:
                den = res[i - 1] - 2 * res[i] + res[i + 1]
                if den > 0:
                    di = D_GRID[i] + 0.5 * (res[i - 1] - res[i + 1]) / den * (D_GRID[1] - D_GRID[0])
            m = cm.diffuse_model(f, di, model)
            r = float(np.clip((m * w) @ y / max((m * m * w).sum(), cm.EPS), 0.5, 1.0))
            e = (y - r * m) * np.sqrt(w0)
            sc = 1.4826 * np.median(np.abs(e)) + 1e-12
            hub = np.where(np.abs(e) <= HUBER_K * sc, 1.0, HUBER_K * sc / np.maximum(np.abs(e), 1e-12))
        out[model] = (float(np.sum(w * (y - r * m) ** 2)), float(di), r,
                      float(np.sqrt(np.mean((y - r * m) ** 2))), bool(i in (0, len(D_GRID) - 1)))
    return out


def _choose(fits):
    if "j0" in fits and "sinc" in fits:
        return "j0" if fits["j0"][0] < J0_MARGIN * fits["sinc"][0] else "sinc"
    return next(iter(fits))


def coarse_skew(G):
    """Skew (samples) maximizing Re sum Gamma e^{-j 2 pi f tau} over COARSE_BAND, grid 0.005 samples."""
    k = _k(COARSE_BAND)
    grid = np.arange(-SKEW_MAX, SKEW_MAX + 1e-9, 0.005)
    J = np.real(G[k][None, :] * np.exp(-2j * np.pi * F[k][None, :] * grid[:, None] / cm.FS)).sum(1)
    return float(grid[int(np.argmax(J))])


def refine_phase(G, Mmod, tau_s, delta, fit_corner=True, iters=4, use=None):
    """Weighted Huber LS of angle(G sign(M) e^{-j model}) on [2 pi f, corner regressor] over
    PHASE_BAND bins (in `use`) with |M| >= 0.3; weights = inverse phase variance from
    min(|M|, smoothed |G|), capped. Returns (tau_s, delta, se_tau_s, se_delta, rms_deg)."""
    k = _k(PHASE_BAND) & (np.abs(Mmod) >= 0.3) & (np.ones(cm.N_BINS, bool) if use is None else use)
    if k.sum() < 5:
        return tau_s, delta, np.inf, np.inf, np.inf
    f = F[k]
    gs = np.convolve(np.pad(np.abs(G), 2, mode="edge"), np.ones(5) / 5, mode="valid")
    m2 = np.minimum(np.minimum(np.abs(Mmod[k]), gs[k]) ** 2, 0.999)
    w0 = np.minimum(m2 / (1 - m2), W_CAP)
    A = np.stack([2 * np.pi * f, _corner_reg(f)], axis=1) if fit_corner else (2 * np.pi * f)[:, None]
    hub = np.ones_like(w0)
    se = np.full(A.shape[1], np.inf)
    rms = np.inf
    for _ in range(iters):
        r = np.angle(G[k] * np.sign(Mmod[k]) * np.exp(-1j * _phase_model(tau_s, delta)[k]))
        w = w0 * hub
        AtW = A.T * w
        H = AtW @ A
        try:
            p = np.linalg.solve(H, AtW @ r)
        except np.linalg.LinAlgError:
            break
        tau_s += p[0]
        if fit_corner:
            delta = float(np.clip(delta + p[1], -120.0, 120.0))
        r = np.angle(G[k] * np.sign(Mmod[k]) * np.exp(-1j * _phase_model(tau_s, delta)[k]))
        s = np.sqrt(np.sum(w0 * r ** 2) / np.sum(w0))      # weighted residual scale (rad, at unit weight)
        z = np.abs(r) * np.sqrt(w0 / np.mean(w0)) / max(s, 1e-9)
        hub = np.where(z <= HUBER_K, 1.0, HUBER_K / np.maximum(z, 1e-9))
        sig2 = np.sum(w0 * hub * r ** 2) / max(len(r) - A.shape[1], 1)
        try:
            se = np.sqrt(np.diag(np.linalg.inv((A.T * (w0 * hub)) @ A)) * sig2)
        except np.linalg.LinAlgError:
            se = np.full(A.shape[1], np.inf)
        rms = float(np.degrees(s))
    return float(tau_s), float(delta), float(se[0]), (float(se[1]) if fit_corner else np.inf), rms


def solve(S, PA, PB, fit_corner=True):
    """Full estimate from pooled statistics. Returns dict (tau in samples, d in m, ...)."""
    G = S / np.sqrt(PA * PB + cm.EPS)
    use = ~tonal_mask(PA, PB)
    tau = coarse_skew(G) / cm.FS
    delta = 0.0
    fits = model = None
    Mmod = np.ones(cm.N_BINS)
    se_tau = se_delta = rms_ph = np.inf
    for it in range(3):
        Gr = G * np.exp(-1j * _phase_model(tau, delta))
        fits = fit_spacing(np.real(Gr), use=use)
        model = _choose(fits)
        _, d, rho, _, _ = fits[model]
        Mmod = rho * cm.diffuse_model(F, d, model)
        tau, delta, se_tau, se_delta, rms_ph = refine_phase(G, Mmod, tau, delta, fit_corner=fit_corner, use=use)
    corner_ok = bool(fit_corner and np.isfinite(se_delta) and abs(delta) >= 3 * se_delta
                     and abs(np.degrees(_cb.corner_phase(125.0, *_corners(delta)))) >= 1.0)
    if fit_corner and not corner_ok:
        delta = 0.0
        tau, _, se_tau, _, rms_ph = refine_phase(G, Mmod, tau, 0.0, fit_corner=False, use=use)
    Gr = G * np.exp(-1j * _phase_model(tau, delta))
    fits = fit_spacing(np.real(Gr), use=use)
    model = _choose(fits)
    wres, d, rho, rms, edge = fits[model]
    pa_b = np.array([PA[b].sum() for b in BAND_BINS])
    pb_b = np.array([PB[b].sum() for b in BAND_BINS])
    return {"tau": tau * cm.FS, "delta": delta, "corner_ok": corner_ok, "se_tau_lsq": se_tau * cm.FS,
            "se_delta_lsq": se_delta, "phase_rms_deg": rms_ph, "d": d, "model": model, "rho": rho,
            "fit_rms": rms, "d_edge": edge, "fits": {m: {"d": v[1], "rho": v[2], "rms": v[3], "wres": v[0]}
                                                    for m, v in fits.items()},
            "gain_corr_db": 10 * np.log10(pa_b / np.maximum(pb_b, cm.EPS)), "n_tonal": int((~use).sum()),
            "G": G, "Gr": Gr}


def _hf_coh(S, PA, PB, tau, delta):
    k = _k((4000.0, 6500.0))
    G = S / np.sqrt(PA * PB + cm.EPS) * np.exp(-1j * _phase_model(tau / cm.FS, delta))
    return float(np.mean(np.real(G[k])))


# ------------------------------------------------------------------ public entry point
def estimate(x, fs, info=None, return_diag=False):
    """calib_json dict for info['calib_json'] (see module doc). return_diag=True also returns a diag dict
    with arrays (G, Gr) for plotting."""
    info = info or {}
    prior = float(info.get("spacing_m", 0.04))
    base = {"version": 1, "spacing_m": float(np.clip(prior, *JSON_D_RANGE)), "tau0_s": 0.0,
            "hpf_corner_hz": [CORNER_F0, CORNER_F0]}
    head = info.get("x_s0v2_head_s")                   # causal / boot-time use: estimate from the first head_s only
    if head:
        x = np.asarray(x)[:, :max(1, int(round(float(head) * fs)))]
    X = prep(x, fs)
    mask, thr, _ = select_noise(X)
    n_sel = int(mask.sum())
    t_sel = n_sel * cm.HOP_S
    blocks = _blocks(X, mask)
    diag = {"noise_s": round(t_sel, 2), "thr_db": thr, "n_blocks": len(blocks), "spacing_prior_m": prior}
    if n_sel < 60 or len(blocks) < 1:
        diag.update(note="too few noise frames: prior spacing, skew 0, no gain", confidence=0.0)
        out = dict(base, confidence=0.0, x_s0v2=diag)
        return (out, diag) if return_diag else out

    tot = [sum(b[i] for b in blocks) for i in range(3)]
    res = solve(*tot)
    # outlier blocks (directional events): per-block skew (one phase step around the pooled model) and
    # HF coherence; drop blocks beyond 5 MAD (and beyond 0.1 sample / 0.05 coherence) and re-solve.
    dropped = 0
    if len(blocks) >= 5:
        Mmod = res["rho"] * cm.diffuse_model(F, res["d"], res["model"])
        tb, hb = [], []
        for S, A, B in blocks:
            Gb = S / np.sqrt(A * B + cm.EPS)
            t, _, _, _, _ = refine_phase(Gb, Mmod, res["tau"] / cm.FS, res["delta"], fit_corner=False, iters=2)
            tb.append(t * cm.FS)
            hb.append(_hf_coh(S, A, B, res["tau"], res["delta"]))
        tb, hb = np.array(tb), np.array(hb)
        mad_t = 1.4826 * np.median(np.abs(tb - np.median(tb)))
        mad_h = 1.4826 * np.median(np.abs(hb - np.median(hb)))
        bad = (np.abs(tb - np.median(tb)) > max(5 * mad_t, 0.1)) | (hb - np.median(hb) > max(5 * mad_h, 0.05))
        if bad.any() and (~bad).sum() >= 3:
            blocks = [b for b, z in zip(blocks, bad) if not z]
            dropped = int(bad.sum())
            tot = [sum(b[i] for b in blocks) for i in range(3)]
            res = solve(*tot)
        diag["block_tau_iqr"] = float(np.subtract(*np.percentile(tb, [75, 25])))
    # jackknife over blocks (corner decision held as in the pooled fit)
    nb = len(blocks)
    se_tau = se_d = None
    se_gain = np.full(len(BANDS_HZ), np.inf)
    if nb >= 3:
        jt, jd, jg = [], [], []
        for i in range(nb):
            S, A, B = (tot[j] - blocks[i][j] for j in range(3))
            r = solve(S, A, B, fit_corner=res["corner_ok"])
            jt.append(r["tau"])
            jd.append(r["d"] if r["model"] == res["model"] else r["fits"][res["model"]]["d"])
            jg.append(r["gain_corr_db"])
        jt, jd, jg = np.array(jt), np.array(jd), np.array(jg)
        c = (nb - 1) / nb
        se_tau = float(np.sqrt(c * np.sum((jt - jt.mean()) ** 2)))
        se_d = float(np.sqrt(c * np.sum((jd - jd.mean()) ** 2)))
        se_gain = np.sqrt(c * np.sum((jg - jg.mean(0)) ** 2, axis=0))
    if se_tau is None:                      # < 3 blocks: least-squares SE, inflated
        se_tau = 2 * res["se_tau_lsq"]
        se_d = 0.25 * res["d"]
    # phase linearity: skew per band (sign of the fitted diffuse model applied); a directional noise component
    # makes the slope frequency dependent (v2 dev bus: spread 1.1-1.3 samples; board fans: 0.05-0.10)
    S, A, B = tot
    G = S / np.sqrt(A * B + cm.EPS)
    Mfit = res["rho"] * cm.diffuse_model(F, res["d"], res["model"])
    slopes, spread_vals = {}, []
    for lo, hi in ((150, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6500)):
        k = _k((lo, hi))
        r = np.angle(G[k] * np.sign(Mfit[k]) * np.exp(-1j * _phase_model(res["tau"] / cm.FS, res["delta"])[k]))
        wk = np.minimum(np.abs(G[k]) ** 2 / (1 - np.minimum(np.abs(G[k]) ** 2, 0.999)), W_CAP)
        sl = float(res["tau"] + (wk * F[k] * r).sum() / (2 * np.pi * (wk * F[k] ** 2).sum()) * cm.FS)
        slopes[f"{lo}-{hi}"] = sl
        if lo >= 500 and np.mean(np.abs(Mfit[k])) >= 0.3:
            spread_vals.append(sl)
    spread = float(max(spread_vals) - min(spread_vals)) if len(spread_vals) >= 2 else np.inf
    diag["band_slopes_samples"] = slopes
    diag["slope_spread_samples"] = spread
    # identifiability and confidences
    frac = min(1.0, t_sel / MIN_SEL_S)
    skew_ok = (t_sel >= 1.0 and se_tau <= 0.15 and abs(res["tau"]) < SKEW_MAX - 0.05
               and res["phase_rms_deg"] <= 30.0 and spread <= SPREAD_MAX)
    d_ok = (t_sel >= 1.5 and res["fit_rms"] <= FIT_RMS_MAX and not res["d_edge"] and se_d <= 0.25 * res["d"])
    gain_ok = t_sel >= 1.0 and np.all(np.isfinite(res["gain_corr_db"]))
    conf_skew = (frac * np.exp(-(se_tau / 0.05) ** 2) * np.exp(-(res["phase_rms_deg"] / 15.0) ** 2)
                 * np.exp(-(spread / 0.3) ** 2)) if skew_ok else 0.0
    conf_d = frac * np.exp(-(se_d / (0.1 * res["d"])) ** 2) * np.exp(-(res["fit_rms"] / 0.1) ** 2) if d_ok else 0.0
    se_g = float(np.median(se_gain)) if np.all(np.isfinite(se_gain)) else 0.5   # LF bands are shrunk anyway
    conf_g = frac * np.exp(-(se_g / 0.3) ** 2) if gain_ok else 0.0
    conf = float((max(conf_skew, 1e-6) * max(conf_d, 1e-6) * max(conf_g, 1e-6)) ** (1 / 3))

    d_use = res["d"] if d_ok else prior
    tau_use = res["tau"] if skew_ok else 0.0
    cand = info.get("x_s0v2_skew_candidates")          # opt-in lattice prior (samples), e.g. [0, 0.5] = calib.py's rule
    if cand is not None and len(cand):
        cand = np.asarray(cand, float)
        tau_use = float(cand[np.argmin(np.abs(cand - res["tau"]))])
        diag["skew_snapped_to"] = tau_use
    delta_use = res["delta"] if (skew_ok and res["corner_ok"]) else 0.0
    # noise coherence for the methods, as calib.py step 6 (measured, 5-bin smoothed, clipped, blended)
    S, A, B = tot
    gm = np.real(S * np.exp(-1j * _phase_model(tau_use / cm.FS, delta_use))) / np.sqrt(A * B + cm.EPS)
    gm = np.clip(np.convolve(np.pad(gm, 2, mode="edge"), np.ones(5) / 5, mode="valid"), -0.3, 0.98)
    rho_use = res["rho"] if d_ok else 1.0
    model_use = res["model"] if d_ok else "sinc"
    wb = n_sel / (n_sel + 200)
    gamma_n = wb * gm + (1 - wb) * rho_use * cm.diffuse_model(F, d_use, model_use)

    out = {"version": 1,
           "spacing_m": float(np.clip(d_use, *JSON_D_RANGE)),
           "tau0_s": float(tau_use / cm.FS),
           "hpf_corner_hz": _corners(delta_use),
           "model": model_use,
           "rho": float(rho_use),
           "gamma_n": {"f_hz": [float(v) for v in F], "gamma": [float(v) for v in gamma_n]},
           "confidence": conf}
    gain_use = np.array(res["gain_corr_db"], float)
    if gain_ok:
        rel = np.isfinite(se_gain) & (se_gain <= GAIN_SE_REF_DB)
        gmed = float(np.median(gain_use[rel])) if rel.any() else float(np.median(gain_use))
        se_eff = np.where(np.isfinite(se_gain), se_gain, 10.0)
        wgt = 1 / np.maximum(se_eff, 1e-3) ** 2
        gain_use = (gain_use * wgt + gmed / GAIN_SE_REF_DB ** 2) / (wgt + 1 / GAIN_SE_REF_DB ** 2)
        out["band_hz"] = [float(v) for v in BANDS_HZ]
        out["gain_db"] = [float(v) for v in np.clip(gain_use, -6, 6)]
    diag.update({
        "skew_samples": float(res["tau"]), "skew_se_samples": float(se_tau), "skew_ok": bool(skew_ok),
        "phase_rms_deg": float(res["phase_rms_deg"]),
        "spacing_fit_m": float(res["d"]), "spacing_se_m": float(se_d), "spacing_ok": bool(d_ok),
        "model": res["model"], "rho": float(res["rho"]), "fit_rms": float(res["fit_rms"]),
        "fits": res["fits"], "spacing_clipped_to_json_range": bool(not JSON_D_RANGE[0] <= d_use <= JSON_D_RANGE[1]),
        "corner_delta_hz": float(res["delta"]), "corner_ok": bool(res["corner_ok"]),
        "corner_se_hz": float(res["se_delta_lsq"]),
        "gain_B_re_A_db": [float(-v) for v in res["gain_corr_db"]], "gain_se_db": [float(v) for v in se_gain],
        "gain_ok": bool(gain_ok), "blocks_dropped": dropped, "n_tonal_bins": res["n_tonal"],
        "gain_corr_used_db": [float(v) for v in gain_use],
        "conf_skew": float(conf_skew), "conf_spacing": float(conf_d), "conf_gain": float(conf_g),
        "confidence": conf})
    out["x_s0v2"] = {k: v for k, v in diag.items()}
    if return_diag:
        return out, dict(diag, G=res["G"], Gr=res["Gr"])
    return out


def gamma_n_for(x, fs, cfg):
    """(257,) noise coherence measured on x's noise frames AFTER the correction of an existing calib_json `cfg`
    (tau0_s; phase_deg or hpf_corner_hz), 5-bin smoothed, clipped [-0.3, 0.98], blended with rho * model(spacing_m)
    with weight N / (N + 200) as calib.py step 6. Use it to complete a json that has no 'gamma_n' (calib._override
    otherwise re-blends a measurement rotated by the BLIND skew; see x_s0v2.md section 4)."""
    d = float(cfg.get("spacing_m", 0.04))
    model = cfg.get("model", "sinc")
    rho = float(cfg.get("rho", 1.0))
    base = rho * cm.diffuse_model(F, d, model)
    X = prep(x, fs)
    mask, _, _ = select_noise(X)
    n = int(mask.sum())
    if n < 60:
        return np.clip(base, -0.3, 0.98)
    paa, pbb, pab = cm.spectra(X[:, :, mask])
    S, A, B = pab.sum(1), paa.sum(1), pbb.sum(1)
    if "phase_deg" in cfg:
        bands = np.asarray(cfg.get("band_hz", BANDS_HZ), float)
        ph = np.radians(np.interp(np.log(np.maximum(F, 1.0)), np.log(bands), np.asarray(cfg["phase_deg"], float)))
    else:
        fa, fb = cfg.get("hpf_corner_hz", [CORNER_F0, CORNER_F0])
        ph = _cb.corner_phase(F, fa, fb)
    ph = ph + 2 * np.pi * F * float(cfg.get("tau0_s", 0.0))
    gm = np.real(S * np.exp(-1j * ph)) / np.sqrt(A * B + cm.EPS)
    gm = np.clip(np.convolve(np.pad(gm, 2, mode="edge"), np.ones(5) / 5, mode="valid"), -0.3, 0.98)
    w = n / (n + 200)
    return w * gm + (1 - w) * base


def summary(cj):
    """One-line summary of a calib_json made by estimate()."""
    d = cj.get("x_s0v2", {})
    return (f"skew {d.get('skew_samples', float('nan')):+.3f} +- {d.get('skew_se_samples', float('nan')):.3f} smp "
            f"(ok={d.get('skew_ok')}), d {100 * d.get('spacing_fit_m', float('nan')):.2f} +- "
            f"{100 * d.get('spacing_se_m', float('nan')):.2f} cm {d.get('model')} rho {d.get('rho', float('nan')):.3f} "
            f"rms {d.get('fit_rms', float('nan')):.3f} (ok={d.get('spacing_ok')}), "
            f"B re A {np.median(d.get('gain_B_re_A_db', [np.nan])):+.2f} dB, corner dHz {d.get('corner_delta_hz', 0):+.1f}"
            f" (ok={d.get('corner_ok')}), noise {d.get('noise_s')} s, conf {cj.get('confidence', 0):.2f}")


if __name__ == "__main__":
    import soundfile as sf
    x, fs = sf.read(sys.argv[1], dtype="float64", always_2d=True)
    info = {"spacing_m": float(sys.argv[2]) if len(sys.argv) > 2 else 0.04}
    cj = estimate(x.T[:2], fs, info)
    print(summary(cj))
    print(json.dumps({k: v for k, v in cj.items() if k != "gamma_n"}, indent=1))
