"""S0 self-calibration (DESIGN.md 2.0, steps 1-6). API: methods/_core/API.md.

calibrate() works on the STFT of the prefiltered signal (prefilter() = identical 2nd-order
Butterworth high-pass, 120 Hz bus / 80 Hz meeting). From diffuse-dominant frames D it estimates
- a per-band gain (one-third octaves 125 Hz - 6.3 kHz) and a phase fit
  phi(f) = 2 pi f tau0 + atan(f_A/f) - atan(f_B/f) over 100-2000 Hz (skew + two HPF corners),
  combined into the correction c(f) = g(f) exp(+j (phi_corner(f) + 2 pi f tau0)) for X_B. The phase
  is fitted to sum Phi_AB over the scenario's noise frames (bus: D; meeting: VAD pauses, the frames of
  Gamma_n), and the skew is the one of the two values a one-bus I2S stereo frame allows, 0 and
  +31.25 us (Mic B = right slot, half a sample late), that is nearer to the fitted slope (API.md);
- the spacing d_hat: free-rho fit of rho*sinc or rho*J0 to Re Gamma_D over 200-7500 Hz (DESIGN:
  4000 Hz; API.md, deviations), the lower model residual wins, raised to the TDOA lower bound
  c * P99|tau| of strongly direct frames, and replaced by the prior when the fit is poor;
- the noise coherence Gamma_n(f) measured over D (bus) or VAD pauses (meeting), blended with
  rho * model(f; d_hat) with weight N / (N + 200).
apply() multiplies X_B by c(f). calibrate_online() re-estimates every block_s from past data only.
Hardware mode: info['calib_json'] (path or dict, schema in API.md) overrides the blind estimates.
"""
import json
from dataclasses import dataclass, field, replace

import numpy as np

from lab.common import C
from . import common as cm

BANDS_HZ = 1000.0 * 2.0 ** (np.arange(-9, 9) / 3.0)          # 18 one-third-octave centres, 125 Hz - 6.35 kHz
BAND_BINS = [np.nonzero((cm.FREQS >= fc * 2 ** (-1 / 6)) & (cm.FREQS < fc * 2 ** (1 / 6)))[0] for fc in BANDS_HZ]
HPF_HZ = {"bus": 120.0, "meeting": 80.0}

LAM_D = cm.LAM_76MS          # smoothing of the MSC that selects D (step 2)
LAM_SRP = cm.LAM_50MS        # smoothing of the coherence used for TDOAs (as S1)
HF_BAND = (2500.0, 6000.0)   # MSC_hf band (directness psi)
VAD_BAND = (300.0, 4000.0)
SRP_BAND = (200.0, 3500.0)
PHASE_FIT_BAND = (100.0, 2000.0)
CORNER_ONLY_BAND = (100.0, 500.0)   # corner refit band when the joint fit is inconsistent ...
PHASE_CONSISTENT_DEG = 8.0          # ... i.e. leaves a per-band residual above this
SKEW_STEP = 0.5 / cm.FS             # 31.25 us: the only non-zero skew allowed (B late by half a sample)
SPACING_FIT_BAND = (200.0, 7500.0)   # DESIGN: 200-4000 Hz; see API.md "Deviations" (model selection)
D_RANGE = (0.015, 0.080)     # spacing search range (m)
D_STEP = 1e-4
CORNER_RANGE = (30.0, 150.0)  # HPF corner search range (Hz)
GAIN_CLIP_DB = 3.0
PHASE_FAULT_DEG = 10.0
MIN_D_FRAMES = 50            # fewer diffuse frames -> keep the identity correction
FIT_RMS_MAX = 0.15           # spacing-fit RMS residual above which the prior spacing is used
PSI_DIRECT = 0.5             # absolute directness floor of the frames used for the TDOA lower bound
MIN_DIRECT_FRAMES = 200
BLEND_N0 = 200               # Gamma_n blend weight N / (N + BLEND_N0)
MEETING_PAUSE_P = 0.3        # VAD probability below which a meeting frame counts as a pause
GAP = cm.N_FFT // cm.HOP     # frames l and l +- GAP share no samples

_SL_ALL = cm.bins_between(SRP_BAND[0], HF_BAND[1])   # bins streamed with smoothing (200 Hz - 6 kHz)
_F_ALL = cm.FREQS[_SL_ALL]
_HF = (_F_ALL >= HF_BAND[0]) & (_F_ALL <= HF_BAND[1])
_SRP = (_F_ALL >= SRP_BAND[0]) & (_F_ALL <= SRP_BAND[1])
_SL_VAD = cm.bins_between(*VAD_BAND)


@dataclass
class Calib:
    """S0 result. Frequencies are common.FREQS (257 bins) unless stated; angles of corrections are
    applied to X_B (X_B <- corr * X_B)."""
    d_hat: float                 # spacing every method should use (m)
    model: str                   # diffuse-noise coherence model: 'sinc' (3-D) or 'j0' (2-D)
    rho: float                   # coherence scale in [0.5, 1] (self-noise / uncorrelated-noise cap)
    gamma_n: np.ndarray          # (257,) real noise coherence, measured blended with rho * model(f; d_hat)
    gain_db: np.ndarray          # (18,) gain correction for X_B at BANDS_HZ (dB)
    phase_deg: np.ndarray        # (18,) phase correction for X_B at BANDS_HZ (deg), HPF-corner term only
    tau0: float                  # skew of Mic B relative to Mic A (s, + = B late); corrected by apply()
    corr: np.ndarray             # (257,) complex correction c(f) for X_B (gain, corner phase and skew)
    hpf_hz: float                # prefilter corner used (Hz)
    confidence: float            # 0..1 overall reliability of the blind estimates
    source: str                  # 'blind', 'json' (calib.json used) or 'prior' (no usable data)
    diag: dict = field(default_factory=dict)

    @property
    def gain(self):
        """(257,) linear magnitude of the correction for X_B."""
        return np.abs(self.corr)

    def model_coherence(self, f=cm.FREQS):
        """rho * model(f; d_hat): the parametric part of gamma_n."""
        return self.rho * cm.diffuse_model(f, self.d_hat, self.model)


def prefilter(x, scenario):
    """S0 step 1: identical 2nd-order Butterworth high-pass on both channels (120 Hz bus, 80 Hz meeting)."""
    return cm.hpf(x, HPF_HZ[scenario])


def identity(d=0.04, scenario="bus", model="sinc", rho=1.0):
    """Calibration that changes nothing: unit correction, gamma_n = rho * model(f; d)."""
    g = rho * cm.diffuse_model(cm.FREQS, d, model)
    return Calib(d_hat=float(d), model=model, rho=float(rho), gamma_n=np.clip(g, -0.3, 0.98),
                 gain_db=np.zeros(len(BANDS_HZ)), phase_deg=np.zeros(len(BANDS_HZ)), tau0=0.0,
                 corr=np.ones(cm.N_BINS, complex), hpf_hz=HPF_HZ[scenario], confidence=0.0, source="prior",
                 diag={})


def apply(X, cal, inplace=False):
    """Return X (2, F, N) with X_B multiplied by the correction (a copy unless inplace). cal: a Calib,
    or an online schedule [(start_frame, Calib), ...] whose entries apply from start_frame on."""
    Y = X if inplace else np.array(X, copy=True)
    if isinstance(cal, Calib):
        Y[1] *= cal.corr[:, None]
        return Y
    starts = [s for s, _ in cal] + [Y.shape[-1]]
    for (s, c), e in zip(cal, starts[1:]):
        Y[1, :, s:e] *= c.corr[:, None]
    return Y


def load_json(src):
    """Read calib.json (path or dict) and check its fields (schema in API.md)."""
    if isinstance(src, dict):
        cfg = dict(src)
    else:
        with open(src) as fh:
            cfg = json.load(fh)
    if cfg.get("version", 1) != 1:
        raise ValueError("calib.json: unsupported version")
    if "spacing_m" in cfg and not 0.01 <= float(cfg["spacing_m"]) <= 0.1:
        raise ValueError("calib.json: spacing_m out of range")
    nb = len(cfg.get("band_hz", BANDS_HZ))
    for k in ("gain_db", "phase_deg"):
        if k in cfg and len(cfg[k]) != nb:
            raise ValueError(f"calib.json: {k} must have one value per band")
    return cfg


# ------------------------------------------------------------------ statistics
class _Stats:
    """Sufficient statistics of S0, accumulated over frames (offline: one block; online: many)."""

    def __init__(self):
        z = np.zeros(cm.N_BINS)
        self.S, self.PA, self.PB = z.astype(complex), z.copy(), z.copy()      # over D
        self.Sn, self.PAn, self.PBn = z.astype(complex), z.copy(), z.copy()   # over noise frames
        self.nD = self.nN = 0
        self.bands_A, self.bands_B = [], []
        self.direct_G = []
        self.feat = {"psi": [], "level": [], "e_vad": [], "valid": [], "psi_b": []}   # psi_b: online only
        self.state = None        # smoothing state carried between blocks

    def features(self, X, l0, l1):
        """Per-frame psi (MSC_hf, lam 0.9), smoothed level and VAD energy for frames l0..l1."""
        Xb = X[:, :, l0:l1]
        st = self.state or {}
        paa, pbb, pab = cm.spectra(Xb[:, _SL_ALL])
        out = []
        for key, r in zip(("aa", "bb", "ab"), (paa, pbb, pab)):
            y, st[key] = cm.smooth(r, LAM_D, st.get(key))
            out.append(y)
        lev = np.sum(np.abs(Xb[0]) ** 2 + np.abs(Xb[1]) ** 2, axis=0)
        lev_s, st["lev"] = cm.smooth(lev, LAM_D, st.get("lev"))
        self.state = st
        self.feat["psi"].append(cm.msc(*[o[_HF] for o in out]).mean(axis=0))
        self.feat["level"].append(10 * np.log10(lev_s + cm.EPS))
        self.feat["e_vad"].append(0.5 * np.mean(np.abs(Xb[0, _SL_VAD]) ** 2 + np.abs(Xb[1, _SL_VAD]) ** 2, axis=0))

    def accumulate(self, X, l0, D, noise, direct):
        """Add frames l0 + nonzero(mask) of each mask (masks are relative to l0)."""
        for mask, kind in ((D, "D"), (noise, "N")):
            idx = l0 + np.nonzero(mask)[0]
            if not len(idx):
                continue
            Xs = X[:, :, idx]
            paa, pbb, pab = cm.spectra(Xs)
            if kind == "D":
                self.S += pab.sum(1)
                self.PA += paa.sum(1)
                self.PB += pbb.sum(1)
                self.nD += len(idx)
                self.bands_A.append(np.stack([paa[b].sum(0) for b in BAND_BINS]))
                self.bands_B.append(np.stack([pbb[b].sum(0) for b in BAND_BINS]))
            else:
                self.Sn += pab.sum(1)
                self.PAn += paa.sum(1)
                self.PBn += pbb.sum(1)
                self.nN += len(idx)
        idx = l0 + np.nonzero(direct)[0]
        if len(idx):
            self.direct_G.append(self._srp_coherence(X, idx))

    def _srp_coherence(self, X, idx):
        """lam-0.85 smoothed coherence over SRP_BAND at frames idx (smoothing warmed up over 40 frames)."""
        out = []
        for l in idx:
            a = max(0, l - 40)
            paa, pbb, pab = cm.spectra(X[:, _SL_ALL, a:l + 1][:, _SRP])
            sm = [cm.smooth(r, LAM_SRP)[1] for r in (paa, pbb, pab)]
            out.append(cm.coherence(*sm))
        return np.stack(out, axis=1)


def _psi_backward(X):
    """Anti-causal (time-reversed) lam-0.9 MSC_hf for every frame of X (D selection). Near the end of X
    it starts from a single frame (MSC 1), so the last ~20 frames read high and are never in D."""
    n = X.shape[-1]
    out, state = np.empty(n), [None, None, None]
    for l1 in range(n, 0, -cm.CHUNK):
        l0 = max(0, l1 - cm.CHUNK)
        sm = []
        for i, r in enumerate(cm.spectra(X[:, _SL_ALL, l0:l1][..., ::-1])):
            y, state[i] = cm.smooth(r, LAM_D, state[i])
            sm.append(y[_HF])
        out[l0:l1] = cm.msc(*sm).mean(axis=0)[::-1]
    return out


def _leave_out(psi_f, psi_b=None):
    """D-selection statistic of frame l without frame l's own data: max(psi_f[l-GAP], psi_b[l+GAP]).
    Selecting on psi of the frame itself biases the coherence measured on D downward (and the
    spacing fit toward J0 / larger d); frames GAP apart share no samples."""
    sel = np.full(len(psi_f), np.inf)
    sel[GAP:] = psi_f[:-GAP]
    if psi_b is not None:
        fut = np.full(len(psi_f), np.inf)
        fut[:-GAP] = psi_b[GAP:]
        sel = np.where(np.isinf(sel), fut, np.where(np.isinf(fut), sel, np.maximum(sel, fut)))
    return sel


def _select(st, scenario, l0, l1, vad_p, psi_b=None):
    """Frame masks (relative to l0) of D, noise frames and strongly direct frames for frames l0..l1,
    with percentile thresholds taken over all valid frames seen so far."""
    psi = np.concatenate(st.feat["psi"])
    lev = np.concatenate(st.feat["level"])
    valid = cm.valid_frames(len(psi)) if not st.feat["valid"] else np.concatenate(st.feat["valid"])
    sel = _leave_out(psi, psi_b)
    valid &= np.isfinite(sel)
    if valid.sum() < 10:
        z = np.zeros(l1 - l0, bool)
        return z, z, z
    lv = lev[valid]
    b_lev, ok = lev[l0:l1], valid[l0:l1]
    D = ok & (sel[l0:l1] < np.percentile(sel[valid], 30)) & (b_lev > np.percentile(lv, 20))
    if scenario == "meeting":
        noise = ok & (vad_p[l0:l1] < MEETING_PAUSE_P) & (b_lev > np.percentile(lv, 5))
    else:
        noise = D
    direct = ok & (psi[l0:l1] >= max(PSI_DIRECT, np.percentile(psi[valid], 90))) & (b_lev > np.percentile(lv, 50))
    return D, noise, direct


# ------------------------------------------------------------------ estimation
def _smooth3(v):
    """3-band smoothing [1/4, 1/2, 1/4] with linear extrapolation at the edges (keeps linear trends)."""
    p = np.concatenate([[2 * v[0] - v[1]], v, [2 * v[-1] - v[-2]]])
    return 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]


def _bands_to_bins(v):
    """Interpolate per-band values to the 257 bins (linear in log f, constant outside the bands)."""
    return np.interp(np.log(np.maximum(cm.FREQS, 1.0)), np.log(BANDS_HZ), v)


def corner_phase(f, f_a, f_b):
    """Phase difference atan(f_A/f) - atan(f_B/f) of two first-order high-passes (rad); 0 at f = 0."""
    f = np.maximum(np.asarray(f, dtype=float), 1e-9)
    return np.arctan(f_a / f) - np.arctan(f_b / f)


def _fit_phase(S, band=PHASE_FIT_BAND, tau0=None):
    """Fit angle(S) = 2 pi f tau0 + atan(f_A/f) - atan(f_B/f) over `band`, weights |S|; tau0 is
    fitted when None, else held fixed. Grid search over the corners (2 Hz, then 0.25 Hz),
    closed-form weighted LS for tau0. Returns (tau0, f_A, f_B, weighted RMS residual in deg)."""
    k = (cm.FREQS >= band[0]) & (cm.FREQS <= band[1])
    f, ph, w = cm.FREQS[k], np.angle(S[k]), np.abs(S[k])
    w = w / max(w.sum(), cm.EPS)

    def search(fa, fb):
        fa, fb = np.meshgrid(fa, fb, indexing="ij")
        fa, fb = fa.ravel(), fb.ravel()
        corner = corner_phase(f[None, :], fa[:, None], fb[:, None])
        r = np.angle(np.exp(1j * (ph[None, :] - corner)))
        t = (r * w * f).sum(1) / (2 * np.pi * (w * f * f).sum()) if tau0 is None else np.full(len(fa), tau0)
        cost = (w * (1 - np.cos(r - 2 * np.pi * f * t[:, None]))).sum(1)
        i = int(np.argmin(cost))
        return fa[i], fb[i], t[i], cost[i]

    lo, hi = CORNER_RANGE
    fa, fb, _, _ = search(np.arange(lo, hi + 1e-9, 2.0), np.arange(lo, hi + 1e-9, 2.0))
    fine = lambda c: np.arange(max(lo, c - 3.0), min(hi, c + 3.0) + 1e-9, 0.25)
    fa, fb, t, cost = search(fine(fa), fine(fb))
    rms_deg = float(np.degrees(np.sqrt(2 * cost)))       # 1 - cos r ~ r^2 / 2
    return float(t), float(fa), float(fb), rms_deg


def _band_phase_residual(S, PA, PB, model_phase):
    """Per-band |angle(sum_b S e^-j model)| (deg) over the bands inside PHASE_FIT_BAND whose band
    coherence is at least 0.3. A pure skew + corner mismatch leaves only noise (< 1-2 deg); coherent
    early reflections and point sources in D leave tens of degrees (v1 rooms)."""
    out = []
    for fc, b in zip(BANDS_HZ, BAND_BINS):
        if PHASE_FIT_BAND[0] <= fc <= PHASE_FIT_BAND[1]:
            s = np.sum(S[b] * np.exp(-1j * model_phase[b]))
            if abs(s) >= 0.3 * np.sqrt(PA[b].sum() * PB[b].sum()):
                out.append(abs(np.degrees(np.angle(s))))
    return max(out) if out else np.inf


def _fit_spacing(gamma, weighted=True):
    """Fit Re Gamma_D ~ rho * model(f; d) over SPACING_FIT_BAND for both models, rho in [0.5, 1].
    Weights 1 / max(1 - Gamma^2, 0.05)^2 follow the variance of the coherence estimate; the model
    with the lower weighted residual wins. Returns (d_fit, model, rho, RMS residual of the chosen
    model, {model: RMS residual}); the RMS residuals are unweighted."""
    k = (cm.FREQS >= SPACING_FIT_BAND[0]) & (cm.FREQS <= SPACING_FIT_BAND[1])
    f, g = cm.FREQS[k], gamma[k]
    gs = np.convolve(np.pad(gamma, 2, mode="edge"), np.ones(5) / 5, mode="valid")[k]
    w = 1 / np.maximum(1 - np.clip(gs, -1, 1) ** 2, 0.05) ** 2 if weighted else np.ones_like(g)
    w = w / w.sum()
    d = np.arange(D_RANGE[0], D_RANGE[1] + 1e-9, D_STEP)
    best = {}
    for model in ("sinc", "j0"):
        M = cm.diffuse_model(f[None, :], d[:, None], model)
        rho = np.clip((M * w) @ g / np.maximum((M * M * w).sum(1), cm.EPS), 0.5, 1.0)
        res = np.sum(w * (g[None, :] - rho[:, None] * M) ** 2, axis=1)
        i = int(np.argmin(res))
        di = d[i]
        if 0 < i < len(d) - 1:                       # parabolic refinement of the minimum
            den = res[i - 1] - 2 * res[i] + res[i + 1]
            if den > 0:
                di = d[i] + 0.5 * (res[i - 1] - res[i + 1]) / den * D_STEP
        m = cm.diffuse_model(f, di, model)
        r = float(np.clip((m * w) @ g / max((m * m * w).sum(), cm.EPS), 0.5, 1.0))
        best[model] = (float(np.sum(w * (g - r * m) ** 2)), float(di), r,
                       float(np.sqrt(np.mean((g - r * m) ** 2))))
    model = min(best, key=lambda m: best[m][0])
    _, d_fit, rho, rms = best[model]
    return d_fit, model, rho, rms, {m: v[3] for m, v in best.items()}


def _tdoa_bound(G, corr, d_search):
    """c * P99 |tau| over strongly direct frames (G: smoothed coherence over SRP_BAND, uncorrected)."""
    f = _F_ALL[_SRP]
    keep = f <= min(SRP_BAND[1], 0.8 * C / (2 * d_search))
    Gc = G[keep] * np.exp(-1j * np.angle(corr[_SL_ALL][_SRP][keep]))[:, None]
    tmax = 1.2 * d_search / C
    grid = np.arange(-tmax, tmax + 1e-9, 2e-6)
    tau, _ = cm.srp_peak(cm.srp(Gc, cm.msc_weight(np.abs(Gc) ** 2), f[keep], grid), grid)
    return float(C * np.percentile(np.abs(tau), 99)), tau


def _estimate(st, d_prior, scenario, cfg=None):
    """Turn accumulated statistics into a Calib (S0 steps 3-6), then apply calib.json overrides."""
    diag = {"n_diffuse": int(st.nD), "n_noise": int(st.nN), "d_prior": float(d_prior)}
    if st.nD < MIN_D_FRAMES:
        cal = identity(d_prior, scenario)
        cal.diag = dict(diag, note="too few diffuse frames: identity correction, prior spacing")
        return _override(cal, cfg, None) if cfg else cal

    # step 3: per-band gain (median band powers over D)
    bA, bB = np.concatenate(st.bands_A, 1), np.concatenate(st.bands_B, 1)
    raw_db = 10 * np.log10(np.median(bA, 1) / np.maximum(np.median(bB, 1), cm.EPS))
    gain_db = np.clip(_smooth3(raw_db), -GAIN_CLIP_DB, GAIN_CLIP_DB)
    # steps 3-5: joint phase fit (skew + HPF corners), skew decision, corner refit, spacing fit.
    # The phase is fitted on the scenario's noise frames (API.md, deviations): D on the bus, VAD pauses
    # in the meeting (the frames of Gamma_n; talker frames that reach D bias the phase toward their
    # TDOAs). Skew: 0 or +31.25 us, whichever is nearer to the fitted slope.
    if scenario == "meeting" and st.nN >= MIN_D_FRAMES:
        S_ph, PA_ph, PB_ph, ph_frames = st.Sn, st.PAn, st.PBn, "pauses"
    else:
        S_ph, PA_ph, PB_ph, ph_frames = st.S, st.PA, st.PB, "diffuse"
    tau0_fit, f_a, f_b, ph_rms = _fit_phase(S_ph)
    model_ph = corner_phase(cm.FREQS, f_a, f_b) + 2 * np.pi * cm.FREQS * tau0_fit
    ph_resid = _band_phase_residual(S_ph, PA_ph, PB_ph, model_ph)
    tau0 = SKEW_STEP if tau0_fit >= SKEW_STEP / 2 else 0.0
    band = PHASE_FIT_BAND if ph_resid <= PHASE_CONSISTENT_DEG else CORNER_ONLY_BAND
    _, f_a, f_b, ph_rms = _fit_phase(S_ph, band, tau0=tau0)
    corr = 10 ** (_bands_to_bins(gain_db) / 20) * np.exp(1j * (corner_phase(cm.FREQS, f_a, f_b)
                                                             + 2 * np.pi * cm.FREQS * tau0))
    rot = np.exp(-1j * np.angle(corr))
    gamma_d = np.real(st.S * rot) / np.sqrt(st.PA * st.PB + cm.EPS)
    d_fit, model, rho, fit_rms, rms_models = _fit_spacing(gamma_d)
    phase_deg = np.degrees(corner_phase(BANDS_HZ, f_a, f_b))
    fit_ok = fit_rms <= FIT_RMS_MAX
    d_base = d_fit if fit_ok else float(d_prior)
    d_lb, n_direct = 0.0, sum(g.shape[1] for g in st.direct_G)
    if n_direct >= MIN_DIRECT_FRAMES:
        d_search = min(D_RANGE[1], 1.25 * max(d_base, d_prior))
        d_lb, _ = _tdoa_bound(np.concatenate(st.direct_G, 1), corr, d_search)
        d_lb = min(d_lb, D_RANGE[1])
    d_hat = max(d_base, d_lb)
    d_source = "lower_bound" if d_lb > d_base else ("fit" if fit_ok else "prior")
    # step 6: noise coherence, measured and blended with the model
    n_used = st.nN
    if n_used:
        gm = np.real(st.Sn * rot) / np.sqrt(st.PAn * st.PBn + cm.EPS)
        gm = np.clip(np.convolve(np.pad(gm, 2, mode="edge"), np.ones(5) / 5, mode="valid"), -0.3, 0.98)
    else:
        gm = np.zeros(cm.N_BINS)
    w = n_used / (n_used + BLEND_N0)
    gamma_n = w * gm + (1 - w) * rho * cm.diffuse_model(cm.FREQS, d_hat, model)

    conf = min(1.0, st.nD / 300) * np.exp(-(fit_rms / 0.1) ** 2) * np.exp(-(ph_rms / 15.0) ** 2)
    diag.update(gain_db_raw=raw_db, f_a=f_a, f_b=f_b, phase_rms_deg=ph_rms, tau0_fit=tau0_fit,
                band_phase_resid_deg=ph_resid, skew_applied=tau0 != 0.0, phase_frames=ph_frames,
                skew_margin_us=abs(tau0_fit - SKEW_STEP / 2) * 1e6,
                phase_fault=bool(np.max(np.abs(phase_deg)) > PHASE_FAULT_DEG), gamma_d=gamma_d,
                gamma_n_meas=gm, d_fit=d_fit, fit_rms=fit_rms, fit_rms_models=rms_models, d_lb=d_lb,
                n_direct=int(n_direct), d_source=d_source, blend_w=w)
    cal = Calib(d_hat=float(d_hat), model=model, rho=float(rho), gamma_n=gamma_n, gain_db=gain_db,
                phase_deg=phase_deg, tau0=float(tau0), corr=corr, hpf_hz=HPF_HZ[scenario],
                confidence=float(conf), source="blind", diag=diag)
    return _override(cal, cfg, st) if cfg else cal


def _override(cal, cfg, st):
    """calib.json overrides: spacing, skew, per-band phase and model; the gain is the json gain plus
    the blind broadband drift (median band difference, clipped to +-1 dB) when diffuse frames exist."""
    bands = np.asarray(cfg.get("band_hz", BANDS_HZ), dtype=float)
    to_bins = lambda v: np.interp(np.log(np.maximum(cm.FREQS, 1.0)), np.log(bands), np.asarray(v, float))
    to_bands = lambda v: np.interp(np.log(BANDS_HZ), np.log(bands), np.asarray(v, float))
    d_hat = float(cfg.get("spacing_m", cal.d_hat))
    tau0 = float(cfg.get("tau0_s", cal.tau0))
    if "phase_deg" in cfg:
        ph_bins, ph_bands = np.radians(to_bins(cfg["phase_deg"])), to_bands(cfg["phase_deg"])
    elif "hpf_corner_hz" in cfg:
        fa, fb = cfg["hpf_corner_hz"]
        ph_bins, ph_bands = corner_phase(cm.FREQS, fa, fb), np.degrees(corner_phase(BANDS_HZ, fa, fb))
    else:
        ph_bins, ph_bands = np.radians(_bands_to_bins(cal.phase_deg)), cal.phase_deg
    gain_db = cal.gain_db
    drift = 0.0
    if "gain_db" in cfg:
        json_db = to_bands(cfg["gain_db"])
        if st is not None and st.nD >= MIN_D_FRAMES:
            drift = float(np.clip(np.median(cal.gain_db - json_db), -1.0, 1.0))
        gain_db = json_db + drift
    model = cfg.get("model", cal.model)
    rho = float(cfg.get("rho", cal.rho))
    gamma_n = cal.gamma_n
    if "gamma_n" in cfg:
        g = cfg["gamma_n"]
        gamma_n = (np.interp(cm.FREQS, g["f_hz"], g["gamma"]) if isinstance(g, dict) else np.asarray(g, float))
    elif d_hat != cal.d_hat or model != cal.model or rho != cal.rho:
        w = cal.diag.get("blend_w", 0.0)
        gm = cal.diag.get("gamma_n_meas", np.zeros(cm.N_BINS))
        gamma_n = w * gm + (1 - w) * rho * cm.diffuse_model(cm.FREQS, d_hat, model)
    corr = 10 ** (_bands_to_bins(gain_db) / 20) * np.exp(1j * (ph_bins + 2 * np.pi * cm.FREQS * tau0))
    diag = dict(cal.diag, json_drift_db=drift, blind_d_hat=cal.d_hat, blind_tau0=cal.tau0)
    return replace(cal, d_hat=d_hat, tau0=tau0, phase_deg=np.asarray(ph_bands, float), gain_db=gain_db,
                   model=model, rho=rho, gamma_n=gamma_n, corr=corr, source="json", diag=diag)


# ------------------------------------------------------------------ public entry points
def _cfg(info):
    src = info.get("calib_json")
    return load_json(src) if src is not None else None


def calibrate(X, info):
    """Offline S0 on the whole file. X: (2, 257, N) STFT of prefilter(x). Returns a Calib."""
    scenario = info.get("scenario", "bus")
    st = _Stats()
    N = X.shape[-1]
    for l0 in range(0, N, cm.CHUNK):
        st.features(X, l0, min(N, l0 + cm.CHUNK))
    vad_p, _ = cm.speech_presence(np.concatenate(st.feat["e_vad"]))
    psi_b = _psi_backward(X)
    for l0 in range(0, N, cm.CHUNK):
        l1 = min(N, l0 + cm.CHUNK)
        D, noise, direct = _select(st, scenario, l0, l1, vad_p, psi_b)
        st.accumulate(X, l0, D, noise, direct)
    cal = _estimate(st, float(info.get("spacing_m", 0.04)), scenario, _cfg(info))
    cal.diag["vad_p"] = vad_p
    return cal


def calibrate_online(X, info, block_s=5.0):
    """Causal S0: the first block uses the prior (or calib.json); each later block uses an estimate
    from all earlier frames, refreshed every block_s. D is selected as offline (leave-own-frame-out,
    past and future MSC_hf), the future part from a backward pass over the block's own frames only, so
    burst onsets do not leak into D. Returns [(start_frame, Calib), ...]."""
    scenario = info.get("scenario", "bus")
    d_prior = float(info.get("spacing_m", 0.04))
    cfg = _cfg(info)
    first = identity(d_prior, scenario)
    sched = [(0, _override(first, cfg, None) if cfg else first)]
    st = _Stats()
    N, B = X.shape[-1], max(1, int(round(block_s / cm.HOP_S)))
    valid = cm.valid_frames(N)
    for l0 in range(0, N, B):
        l1 = min(N, l0 + B)
        st.features(X, l0, l1)
        st.feat["valid"].append(valid[l0:l1])
        st.feat["psi_b"].append(_psi_backward(X[:, :, l0:l1]))
        vad_p, _ = cm.speech_presence(np.concatenate(st.feat["e_vad"]), valid[:l1])
        D, noise, direct = _select(st, scenario, l0, l1, vad_p, np.concatenate(st.feat["psi_b"]))
        st.accumulate(X, l0, D, noise, direct)
        if l1 < N:
            sched.append((l1, _estimate(st, d_prior, scenario, cfg)))
    return sched
