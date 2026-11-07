"""S1 talker and DOA tracker, plus the S0+S1 front end (DESIGN.md 2.0). API: methods/_core/API.md.

Per frame (hop 8 ms), from the coherence smoothed with lam 0.85 over 200 Hz - min(3.5 kHz, 0.8 c/2d):
- tau: TDOA on a +-1.2 d_hat/c grid (2 us steps, parabolic refinement). Default srp_mode 'fit' fits
  Gx = q exp(+j 2 pi f tau) + (1 - q) Gamma_n per frame (common.coherent_fit) and also returns the
  coherent fraction q. srp_mode 'design' is the DESIGN 2.0 MSC-weighted steered response, kept for
  ablation: in diffuse noise its peak is pulled toward broadside by roughly the factor q (API.md).
- psi: directness, mean MSC over 2.5-6 kHz (lam 0.9); cdr_db: DOA-free CDR over conditioned bins;
  level_db and an energy VAD.
Bus partner model: energy-weighted histogram of tau over direct (q, or psi in 'design' mode, above
its 70th percentile), loud frames; tau_A >= tau_B are the two highest peaks >= 25 us apart;
posteriors from the indicator |tau - tau_k| < 20 us on frames with direct evidence, smoothed over
160 ms and scaled so that the 90th percentile reads 1 (with one peak, tau_A = tau_B: its posterior is
p_A, p_B = 0 and there is no double talk); double talk, partner angles, orientation stratum and the
'inseparable' flag. Meeting: change points and talker clusters (spk.py).
Offline mode uses whole-file statistics; online mode is causal (histogram from the first 5 s, then
5 s forgetting; thresholds from the past 20 s; S0 refreshed every 5 s).
"""
from dataclasses import dataclass, field

import numpy as np

from lab.common import C
from . import calib as cb
from . import common as cm
from . import spk

SRP_BAND = (200.0, 3500.0)
GRID_STEP = 2e-6
HF_BAND = (2500.0, 6000.0)
VAD_BAND = (300.0, 4000.0)
CDR_BAND = (1000.0, 6000.0)
CDR_GN_MAX = 0.7             # frame CDR uses bins with Gamma_n <= this (conditioned without a DOA)
LAM_SRP = cm.LAM_50MS
LAM_PSI = cm.LAM_76MS
LAM_POST = cm.LAM_160MS
HIST_STEP = 1e-6
HIST_SIGMA = 6e-6
HIST_PCT = 70                # histogram frames: directness above this percentile ...
PEAK_SEP = 25e-6
PEAK_MIN_REL = 0.15          # a second histogram peak must reach this fraction of the highest
POST_WIN = 20e-6
POST_PCT = 50                # posterior indicator: directness above this percentile
POST_SCALE_PCT = 90          # p_k = smoothed indicator / its 90th percentile (floor POST_SCALE_MIN)
POST_SCALE_MIN = 0.15
DOUBLE_TALK = 0.25
BROADSIDE_FRAC = 0.3         # both |tau_k| < 0.3 d_hat / c -> inseparable
ONLINE_INIT_S = 5.0
ONLINE_FORGET_S = 5.0
ONLINE_UPDATE_S = 0.25
ONLINE_WIN_S = 20.0

_SL = cm.bins_between(SRP_BAND[0], HF_BAND[1])
_F = cm.FREQS[_SL]
_HF = (_F >= HF_BAND[0]) & (_F <= HF_BAND[1])
_SL_VAD = cm.bins_between(*VAD_BAND)


@dataclass
class Track:
    """S1 output. Per-frame arrays have one entry per frame of common.stft (hop 8 ms)."""
    tau: np.ndarray              # TDOA estimate (s; + = Mic B later), after the S0 correction
    q: np.ndarray                # coherent (direct) fraction at tau, 0..1 ('design' mode: SRP peak)
    psi: np.ndarray              # directness: mean MSC over 2.5-6 kHz (lam 0.9), 0..1
    cdr_db: np.ndarray           # DOA-free coherent-to-diffuse ratio over conditioned bins (dB)
    level_db: np.ndarray         # 300-4000 Hz power per bin, mean of both mics (dB re full scale^2)
    vad: np.ndarray              # speech presence probability 0..1
    noise_db: np.ndarray         # VAD noise floor (dB, same scale as level_db)
    d_hat: float                 # spacing used (m)
    online: bool
    srp_mode: str
    # bus partner model: tau_A >= tau_B (A = partner nearer the Mic-A end of the axis)
    tau_A: float = np.nan        # file-level (offline) or final (online) TDOAs (s)
    tau_B: float = np.nan
    tau_A_t: np.ndarray = None   # (N,) per-frame partner TDOAs (online: NaN until the first estimate)
    tau_B_t: np.ndarray = None
    p_A: np.ndarray = None       # (N,) partner posteriors 0..1; where tau_A_t == tau_B_t (one peak) the
    p_B: np.ndarray = None       # single peak's posterior is p_A and p_B = 0
    double_talk: np.ndarray = None  # (N,) both posteriors > 0.25 (never with one peak)
    theta_A_deg: float = np.nan  # angle from the axis: 0 = beyond Mic A, 90 = broadside, 180 = beyond B
    theta_B_deg: float = np.nan
    stratum: str = ""            # 'aligned' | '45' | 'broadside' | 'side_by_side' | 'other' | 'unknown'
    inseparable: bool = False
    inseparable_why: tuple = ()
    # meeting talker identity
    cluster: np.ndarray = None   # (N,) cluster id, -1 outside speech segments
    change: np.ndarray = None    # (N,) bool talker-change proposals
    clusters: list = field(default_factory=list)
    diag: dict = field(default_factory=dict)

    @property
    def directness(self):
        """The directness used for selection: q in 'fit' mode, psi in 'design' mode."""
        return self.q if self.srp_mode == "fit" else self.psi


# ------------------------------------------------------------------ per-frame features
def _srp_setup(d_hat):
    """Mask of the TDOA bins (within _SL) and the TDOA grid for spacing d_hat."""
    srp = (_F >= SRP_BAND[0]) & (_F <= min(SRP_BAND[1], 0.8 * C / (2 * d_hat)))
    n = int(np.floor(1.2 * d_hat / C / GRID_STEP))
    return srp, GRID_STEP * np.arange(-n, n + 1)


def frame_features(X, blocks, srp_mode="fit"):
    """Stream X once; per-frame tau, q, psi, cdr_db and VAD band energy.
    blocks: [(start, end, Calib)] covering all frames (d_hat and gamma_n per block)."""
    N = X.shape[-1]
    out = {k: np.zeros(N) for k in ("tau", "q", "psi", "cdr_db", "e_vad")}
    for l0, l1, sm in cm.smoothed_spectra(X, _SL, (LAM_SRP, LAM_PSI)):
        paa, pbb, pab = sm[LAM_PSI]
        out["psi"][l0:l1] = cm.msc(paa[_HF], pbb[_HF], pab[_HF]).mean(axis=0)
        G = cm.coherence(*sm[LAM_SRP])
        for a, b, cal in blocks:
            a, b = max(a, l0), min(b, l1)
            if a >= b:
                continue
            g = G[:, a - l0:b - l0]
            gn = cal.gamma_n[_SL]
            srp, grid = _srp_setup(cal.d_hat)
            if srp_mode == "fit":
                J, Q = cm.coherent_fit(g[srp], gn[srp], _F[srp], grid)
                out["tau"][a:b], _ = cm.srp_peak(J, grid)
                out["q"][a:b] = np.clip(Q[np.arange(b - a), np.argmax(J, axis=1)], 0.0, 1.0)
            else:
                P = cm.srp(g[srp], cm.msc_weight(np.abs(g[srp]) ** 2), _F[srp], grid)
                out["tau"][a:b], out["q"][a:b] = cm.srp_peak(P, grid)
            k = (_F >= CDR_BAND[0]) & (_F <= CDR_BAND[1]) & (gn <= CDR_GN_MAX)
            cdr = cm.cdr_nodoa(g[k], gn[k][:, None]).mean(axis=0) if k.any() else np.zeros(b - a)
            out["cdr_db"][a:b] = 10 * np.log10(np.maximum(cdr, 1e-2))
        Xv = X[:, _SL_VAD, l0:l1]
        out["e_vad"][l0:l1] = 0.5 * np.mean(np.abs(Xv[0]) ** 2 + np.abs(Xv[1]) ** 2, axis=0)
    return out


# ------------------------------------------------------------------ bus partner model
def _histogram(tau, w, d):
    """Energy-weighted TDOA histogram on a 1 us grid over +-1.2 d/c, Gaussian-smoothed (6 us)."""
    tmax = 1.2 * d / C
    grid = np.arange(-tmax, tmax + HIST_STEP / 2, HIST_STEP)
    idx = np.clip(np.rint((tau + tmax) / HIST_STEP).astype(int), 0, len(grid) - 1)
    h = np.bincount(idx, weights=w, minlength=len(grid)).astype(float)
    k = np.arange(-4 * HIST_SIGMA, 4 * HIST_SIGMA + HIST_STEP / 2, HIST_STEP)
    return np.convolve(h, np.exp(-0.5 * (k / HIST_SIGMA) ** 2), mode="same"), grid


def _hist_peaks(h, grid, tmax=np.inf):
    """Highest local maximum within |tau| <= tmax plus the highest other one >= 25 us away and
    >= 15 % of it."""
    i = np.nonzero((h[1:-1] > h[:-2]) & (h[1:-1] >= h[2:]) & (np.abs(grid[1:-1]) <= tmax))[0] + 1
    if not len(i):
        return []
    i = i[np.argsort(h[i])[::-1]]
    peaks = [(float(grid[i[0]]), float(h[i[0]]))]
    for j in i[1:]:
        if abs(grid[j] - peaks[0][0]) >= PEAK_SEP and h[j] >= PEAK_MIN_REL * peaks[0][1]:
            peaks.append((float(grid[j]), float(h[j])))
            break
    return peaks


def _order(peaks):
    """(tau_A, tau_B) with tau_A >= tau_B; one peak gives tau_A = tau_B; none gives NaN."""
    if not peaks:
        return np.nan, np.nan
    t = sorted(p[0] for p in peaks)
    return t[-1], t[0]


def _hist_frames(direct, level, ref):
    """Histogram frames among `ref`: directness above its 70th percentile and level above the median.
    Weights: linear band power capped at its 95th percentile over `ref`."""
    sel = ref & (direct > np.percentile(direct[ref], HIST_PCT)) & (level > np.median(level[ref]))
    power = 10 ** (level / 10)
    return sel, np.minimum(power, np.percentile(power[ref], 95))


def angle_deg(tau, d_hat):
    """Angle from the array axis (deg; 0 = beyond Mic A) of TDOA tau."""
    return float(np.degrees(np.arccos(np.clip(C * tau / d_hat, -1.0, 1.0))))


def orientation_stratum(theta_a, theta_b):
    """E1 stratum from two partner angles (deg from the axis), folded to 0-90 deg: 'aligned' both <= 20,
    'broadside' both >= 70 (a device lying flat looks the same), '45' both 30-60, 'side_by_side' same
    side of the broadside plane and 20-40 deg apart, else 'other' ('unknown' if an angle is NaN)."""
    if np.isnan(theta_a) or np.isnan(theta_b):
        return "unknown"
    fa, fb = (min(t, 180.0 - t) for t in (theta_a, theta_b))
    if fa <= 20 and fb <= 20:
        return "aligned"
    if fa >= 70 and fb >= 70:
        return "broadside"
    if 30 <= fa <= 60 and 30 <= fb <= 60:
        return "45"
    if (theta_a - 90) * (theta_b - 90) > 0 and 20 <= abs(theta_a - theta_b) <= 40:
        return "side_by_side"
    return "other"


def inseparable_reasons(n_peaks, tau_a, tau_b, d_hat):
    """DESIGN 2.0 'partners spatially inseparable' rules that hold: one peak, peaks < 25 us apart,
    or both |tau_k| < 0.3 d_hat / c."""
    why = []
    if n_peaks < 2:
        why.append("one_peak")
    elif abs(tau_a - tau_b) < PEAK_SEP:
        why.append("close_tdoas")
    if not np.isnan(tau_a) and max(abs(tau_a), abs(tau_b)) < BROADSIDE_FRAC * d_hat / C:
        why.append("both_broadside")
    return tuple(why)


def _indicators(tau, ta_t, tb_t, high):
    """Per-frame partner indicators (A, B): direct evidence and TDOA within 20 us of tau_k. Where only one
    peak exists (tau_A = tau_B) the indicator of that peak goes to A and B's is 0."""
    near = lambda t: ((np.abs(tau - t) < POST_WIN) & high).astype(float)
    return near(ta_t), near(tb_t) * (ta_t != tb_t)


def _partners_offline(f, direct, valid, d_hat):
    """Whole-file partner model: histogram peaks, constant per-frame TDOAs and posteriors."""
    N = len(f["tau"])
    if not valid.any():                      # input shorter than one STFT frame (T < 512): no partner model
        h, grid = _histogram(np.zeros(0), np.zeros(0), d_hat)
        nan = np.full(N, np.nan)
        return [], np.nan, np.nan, nan, nan.copy(), [np.zeros(N), np.zeros(N)], {"hist": h, "hist_grid": grid}
    sel, w = _hist_frames(direct, f["level_db"], valid)
    h, grid = _histogram(f["tau"][sel], w[sel], d_hat)
    peaks = _hist_peaks(h, grid)
    ta, tb = _order(peaks)
    ta_t, tb_t = np.full(N, ta), np.full(N, tb)
    high = direct > np.percentile(direct[valid], POST_PCT)
    post = []
    for ind in _indicators(f["tau"], ta_t, tb_t, high):
        s = cm.smooth(ind, LAM_POST, 0.0)[0]
        post.append(np.clip(s / max(np.percentile(s[valid], POST_SCALE_PCT), POST_SCALE_MIN), 0.0, 1.0))
    return peaks, ta, tb, ta_t, tb_t, post, {"hist": h, "hist_grid": grid}


def _partners_online(f, direct, valid, d_t):
    """Causal partner model in 0.25 s blocks: block [a, b) uses peaks and thresholds from frames < a;
    the histogram accumulates the first 5 s, then forgets with a 5 s time constant. d_t: per-frame
    spacing of the S0 schedule (limits the peaks to +-1.2 d/c)."""
    N = len(f["tau"])
    step = max(1, int(round(ONLINE_UPDATE_S / cm.HOP_S)))
    win = int(round(ONLINE_WIN_S / cm.HOP_S))
    n_init = int(round(ONLINE_INIT_S / cm.HOP_S))
    forget = np.exp(-step * cm.HOP_S / ONLINE_FORGET_S)
    ta_t, tb_t = np.full(N, np.nan), np.full(N, np.nan)
    s, scale = np.zeros((2, N)), np.full((2, N), np.inf)
    state = [0.0, 0.0]
    h_acc, grid = _histogram(np.zeros(0), np.zeros(0), cb.D_RANGE[1])
    power = 10 ** (f["level_db"] / 10)
    frames = np.arange(N)
    peaks, ta, tb = [], np.nan, np.nan
    post_thr, sc = np.inf, [np.inf, np.inf]
    for a in range(0, N, step):
        b = min(N, a + step)
        ta_t[a:b], tb_t[a:b] = ta, tb
        high = direct[a:b] > post_thr
        for k, ind in enumerate(_indicators(f["tau"][a:b], ta_t[a:b], tb_t[a:b], high)):
            s[k, a:b], state[k] = cm.smooth(ind, LAM_POST, state[k])
            scale[k, a:b] = sc[k]
        ref = valid & (frames >= b - win) & (frames < b)
        if ref.sum() < 20:
            continue
        post_thr = np.percentile(direct[ref], POST_PCT)
        sc = [max(np.percentile(s[k][ref], POST_SCALE_PCT), POST_SCALE_MIN) for k in range(2)]
        sel = ref & (frames >= a) & (direct > np.percentile(direct[ref], HIST_PCT)) \
            & (f["level_db"] > np.median(f["level_db"][ref]))
        if b > n_init:
            h_acc *= forget
        if sel.any():
            h_acc += _histogram(f["tau"][sel], np.minimum(power[sel], np.percentile(power[ref], 95)),
                                cb.D_RANGE[1])[0]
        peaks = _hist_peaks(h_acc, grid, 1.2 * d_t[b - 1] / C)
        ta, tb = _order(peaks)
    post = [np.clip(np.where(np.isfinite(scale[k]), s[k] / scale[k], 0.0), 0.0, 1.0) for k in range(2)]
    return peaks, ta, tb, ta_t, tb_t, post, {"hist": h_acc, "hist_grid": grid}


# ------------------------------------------------------------------ public entry points
def _cal_blocks(cal, N):
    """A Calib or an online schedule -> [(start, end, Calib)]."""
    if isinstance(cal, cb.Calib):
        return [(0, N, cal)]
    starts = [st for st, _ in cal] + [N]
    return [(st, e, c) for (st, c), e in zip(cal, starts[1:]) if e > st]


def track(X, cal, info, online=False, srp_mode="fit", partners=None, talkers=None, x=None, embedder=None,
          cos_thr=spk.COS_THR):
    """S1 on a calibrated STFT X (2, 257, N): calib.apply() output, or the WPE output in the meeting.
    cal: the Calib (offline) or the schedule of calib.calibrate_online (online).
    srp_mode: 'fit' (default) or 'design' (ablation). partners / talkers: run the bus partner model /
    the meeting clustering (default: by info['scenario']). x, embedder: prefiltered Mic A signal and a
    speaker-embedding callable (e.g. spk_onnx.WeSpeakerEmbedder) for embedding-keyed clusters, whose
    cosine threshold is cos_thr."""
    scenario = info.get("scenario", "bus")
    partners = scenario == "bus" if partners is None else partners
    talkers = scenario == "meeting" if talkers is None else talkers
    N = X.shape[-1]
    blocks = _cal_blocks(cal, N)
    f = frame_features(X, blocks, srp_mode)
    valid = cm.valid_frames(N)
    vad, floor = cm.speech_presence(f["e_vad"], valid)
    f["level_db"] = 10 * np.log10(np.maximum(f["e_vad"], cm.EPS))
    d_hat = blocks[-1][2].d_hat
    trk = Track(tau=f["tau"], q=f["q"], psi=f["psi"], cdr_db=f["cdr_db"], level_db=f["level_db"], vad=vad,
                noise_db=10 * np.log10(np.maximum(floor, cm.EPS)), d_hat=d_hat, online=online, srp_mode=srp_mode)
    if partners:
        direct = trk.directness
        if online:
            d_t = np.concatenate([np.full(e - st, c.d_hat) for st, e, c in blocks])
            peaks, ta, tb, ta_t, tb_t, post, dg = _partners_online(f, direct, valid, d_t)
        else:
            peaks, ta, tb, ta_t, tb_t, post, dg = _partners_offline(f, direct, valid, d_hat)
        trk.tau_A, trk.tau_B, trk.tau_A_t, trk.tau_B_t = ta, tb, ta_t, tb_t
        one = ta_t == tb_t                   # one peak: its posterior is p_A, p_B = 0, no double talk
        trk.p_A, trk.p_B = post[0], np.where(one, 0.0, post[1])
        trk.double_talk = (trk.p_A > DOUBLE_TALK) & (trk.p_B > DOUBLE_TALK) & ~one
        if peaks:
            trk.theta_A_deg, trk.theta_B_deg = angle_deg(ta, d_hat), angle_deg(tb, d_hat)
        trk.stratum = orientation_stratum(trk.theta_A_deg, trk.theta_B_deg) if len(peaks) == 2 else "unknown"
        trk.inseparable_why = inseparable_reasons(len(peaks), ta, tb, d_hat)
        trk.inseparable = bool(trk.inseparable_why)
        trk.diag.update(dg, peaks=peaks)
    if talkers:
        trk.change = spk.change_points(trk)
        trk.cluster, trk.clusters = spk.cluster_talkers(trk, online=online, x=x, embedder=embedder, cos_thr=cos_thr)
    return trk


@dataclass
class FrontEnd:
    """S0 + S1 result of front_end()."""
    x: np.ndarray        # (2, T) prefiltered input (Mic B not corrected)
    X: np.ndarray        # (2, 257, N) STFT with the S0 correction applied to X_B
    calib: cb.Calib      # offline: whole-file calibration; online: the last block's
    track: Track
    schedule: list = None  # online only: [(start_frame, Calib), ...] as used by calib.apply


def front_end(x, fs, info, online=False, **track_kw):
    """Prefilter -> STFT -> S0 (calibrate + apply) -> S1. x: (2, T) at 16 kHz, row 0 = Mic A."""
    if fs != cm.FS:
        raise ValueError("front_end expects fs = 16000")
    xh = cb.prefilter(np.asarray(x, dtype=np.float64), info.get("scenario", "bus"))
    X = cm.stft(xh)
    if online:
        sched = cb.calibrate_online(X, info)
        X = cb.apply(X, sched, inplace=True)
        return FrontEnd(xh, X, sched[-1][1], track(X, sched, info, online=True, x=xh[0], **track_kw), sched)
    cal = cb.calibrate(X, info)
    X = cb.apply(X, cal, inplace=True)
    return FrontEnd(xh, X, cal, track(X, cal, info, x=xh[0], **track_kw))
