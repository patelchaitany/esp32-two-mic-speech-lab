"""M7: anchored cACGMM spatial clustering + floored post-mask, optionally with per-partner mask-MVDR
(DESIGN.md 3 M7, 2.1; conventions 2.0). Offline (whole file, segments <= 60 s).

Tuned defaults (PARAMS; v2 dev bus, TTS; evidence in tmp/tune:cacgmm/) and why they differ from DESIGN:
- anchors='grid': directional classes at the fixed TDOAs -d/c, 0, +d/c (d = S0's d_hat) plus the diffuse
  noise class; no S1 input. With S1's partner TDOAs as anchors (DESIGN, anchors='s1') the tuned post-mask
  lost worst-partner ESTOI (dev bus, 90 scenes, fixed-mask proxy: mean -0.008, 23 % of scenes below
  -0.01, worst -0.15): S1 had both anchors within 30 us of the truth in only 11 of the 51 scenes it did
  not flag, and a partner without an anchor falls into the noise class and is masked. With the TRUE
  anchors the same post-mask gains +0.012 (no scene below -0.01, dSNR_active +1.4 dB), so anchors='s1' is
  the mode to use once the anchors are reliable (hardware calibration, a better partner localizer). 'grid'
  keeps every directional source (partners, passengers, point noise) and removes only what the diffuse
  class explains: +0.001, dSNR_active +0.8 dB, no scene below -0.01.
- beam='none' (post-mask on calibrated Mic A): every per-partner MVDR variant lowered worst-partner ESTOI
  even with the true anchors (bench, 0 dB subset, floor-0.3 post-mask: on Mic A +0.015; after Souden
  MVDRs -0.002, steering MVDRs +0.002, covariance-whitening RTF MVDRs -0.030). Souden's
  h = Phi_N^-1 Phi_k e / tr(.) with the noisy posterior-weighted Phi_k responds (1 + l) / (2 + l) to the
  partner (l: whitened partner SNR), so soft posteriors shrink the partners by up to 6 dB, and
  posterior-switched beams filter each partner differently over time and frequency.
- Post-mask clip((1 - gamma_noise) / 0.5, 0.5, 1) instead of max(0.3, gamma_A + gamma_B): posteriors are
  soft (partner-dominated minus noise-dominated posterior 0.06-0.18 on dev), so the DESIGN mask cut the
  partners by 5-10 dB (TA_min -9.9 dB with the true anchors; G0 needs >= -3 dB). The scaled mask keeps
  TA_min near -0.5 dB; the floor 0.5 bounds the damage where the classes are wrong.
- load_partner=0.03 (sharper initial directional classes): dSNR_active +1.8 vs +1.0 dB with the true
  anchors, and +0.9 vs 0.0 dB in 'grid' mode, at equal ESTOI (0 dB subset).
Full dev bench (methods/m7_cacgmm, defaults; paired vs M0 over 30 geometries): worst-partner dESTOI
-0.0001 [-0.0010, +0.0008], dSNR_active +1.02 dB [+0.76, +1.29], NA_active +1.1 dB, TA_min -0.09 dB,
dSIR_near -0.2 dB; G0 pass, G1 fail. Not for meetings (DESIGN 2.2): it attenuates diffuse energy, which
for far talkers includes their own reverberant speech.

Model (Ito, Araki & Nakatani 2016, [49]), per frequency bin f, on z = x / ||x|| (x = [X_A, X_B]):
    p(z; B) = 1 / (2 pi^2 det B) * (z^H B^-1 z)^-2                    complex angular central Gaussian, D = 2
    E-step   gamma_tk ∝ alpha_tk det(B_k)^-1 (z_t^H B_k^-1 z_t)^-2
    M-step   B_k = 2 sum_t gamma_tk z z^H / (z^H B_k^-1 z) / sum_t gamma_tk   (B_k of the E-step on the right)
             [+ anchor prior: B_k <- (n_k B_k + nu B_k^anchor) / (n_k + nu), n_k = sum_t gamma_tk]
             eigenvalues floored at 1e-3 lambda_max, trace renormalized to 2
             alpha_k = mean_t gamma_tk; or GSS-style alpha_tk ∝ alpha_k a_tk with S1 activity a_tk
             (noise activity 1).
Anchors (center-referenced a_k = common.steer(f, tau_k)) of every directional class k (S1 partners A, B
and/or grid guards G1..): init B_k = a_k a_k^H + load_partner I, noise B_0 = Gamma_n + 0.05 I (tr = 2),
alpha = 1/K. The post-mask uses only the noise posterior: M = clip((1 - gamma_0) / mask_scale, g_min, 1),
so without MVDR the output does not depend on how the directional classes are labelled.
Partner labels (anchors 's1' / 's1+grid', needed only for the per-partner MVDRs and the A/B posteriors):
    The predicted phase 2 pi f tau_k stays valid above c/2d (DESIGN 1.1): EM, anchors and the
    permutation test use every bin. A/B are resolved by the anchors only where their steering vectors
    differ, |a_A^H a_B| / 2 = |cos(pi f (tau_A - tau_B))| <= RESOLVE_COS (fails near f = n / (tau_A - tau_B),
    including f -> 0). There the principal eigenvectors of B_A and B_B are compared with a_A and a_B and
    the classes swapped where reversed (DESIGN). In unresolved bins no spatial test can label A and B:
    the EM's partner posterior gamma_A + gamma_B is kept and split by the frame's A share in the
    resolved bins (frame-level decision carried over, DESIGN 1.1).
Beamforming (beam='mvdr', DESIGN; from the unnormalized X, per segment, partner classes only):
    Phi_k = sum_t gamma_k X X^H / sum_t gamma_k,  Phi_N = same with gamma_0, + 1e-3 tr(Phi_N)/2 I
    h_k = Phi_N^-1 Phi_k e_ref / tr(Phi_N^-1 Phi_k)   (Souden), e_ref = mic with the higher
    sum_f Phi_k,mm / sum_f Phi_N,mm over 300-4000 Hz. A Mic-B reference is re-aligned to Mic A with
    the anchor delay (Y_k *= exp(+j 2 pi f tau_k)) so the output stays aligned with x[0].
    Y_lin = sum_k gamma_k / (gamma_A + gamma_B + eps) h_k^H X;  Y = M Y_lin. beam='none': Y_lin = X_A.
S1 'inseparable' flags (anchors_from_track, anchors 's1' and 's1+grid'): peaks < 25 us apart -> one
merged partner class at their mean TDOA; fewer than two peaks, or both partners near broadside -> no
partner class: passthrough (Mic A, mask 1, the DESIGN 2.1 omni fallback) with 's1', guards only with
's1+grid'. In synthetic tests a beam plus mask on the one partner found suppressed the other by 25 dB,
and broadside pairs lost about 10 dB each with a -0.3 to -0.7 dB SNR change.

Online variant (not implemented; DESIGN M7 block-online): 2.4 s blocks with 2.4 s of past context,
1-2 EM iterations per block warm-started from the previous block's B and alpha, anchors from S1's
causal tau_A_t / tau_B_t, permutation test per block, covariances accumulated recursively over the
past (2-4 s memory) and the MVDR refreshed per block; the output then lags by one block.

API: separate(X, cal, trk, **overrides) -> Separation (posteriors, post-mask, linear and masked output
STFTs, operator W), separate_anchored(X, taus, gamma_n, guards=...) with explicit anchors, grid_taus,
apply(sep, X) for any 2-ch STFT (e.g. a separated scene component), and the building blocks features /
anchor_init / em / align_ab / mvdr. params(**DESIGN_PARAMS) is the DESIGN M7 configuration.
Shapes: X (2, 257, N) = common.stft output with S0 applied (tracker.front_end().X).
"""
from dataclasses import dataclass, field

import numpy as np

from lab.common import C
from . import common as cm

# Defaults tuned on the v2 dev bus set (tmp/tune:cacgmm evidence, see the module docstring); DESIGN = the
# DESIGN M7 value where it differs. DESIGN_PARAMS reproduces the DESIGN M7 configuration.
PARAMS = dict(
    anchors="grid",       # directional classes: 'grid' TDOA grid (no S1 input); 's1' S1 partner TDOAs (DESIGN);
                          # 's1+grid' both
    grid_n=3,             # guard classes at linspace(-1, 1, grid_n) d_hat / c (both endfires and broadside)
    iterations=20,        # offline EM iterations over a segment (DESIGN: 10-20)
    eig_floor=1e-3,       # eigenvalue floor relative to lambda_max (DESIGN)
    load_partner=0.03,    # init B_k = a_k a_k^H + 0.03 I for every directional class (DESIGN 0.1)
    load_noise=0.05,      # init B_0 = Gamma_n + 0.05 I (DESIGN)
    prior_frames=0.0,     # anchor prior weight nu in frames (0: plain ML M-step, as DESIGN)
    activity=None,        # None: alpha_k per bin (DESIGN default); 's1': alpha_tk ∝ alpha_k a_tk (GSS-style)
    act_floor=0.2,        # 's1': a_tk = act_floor + (1 - act_floor) p_k(t)
    beam="none",          # 'none': post-mask on Mic A; 'mvdr': per-partner Souden MVDRs (DESIGN); 'steer':
                          # per-partner MVDRs toward the anchor plane waves (partner classes only)
    mask_scale=0.5,       # post-mask clip((1 - gamma_noise) / mask_scale, g_min, 1) (DESIGN 1: max(g_min, .))
    g_min=0.5,            # post-mask floor (DESIGN 0.3; sweep 0.1 / 0.2 / 0.3)
    load_n=0.1,           # MVDR: Phi_N diagonal loading, times tr(Phi_N) / 2 (DESIGN 1e-3)
    ref="snr",            # 'mvdr': 'snr' mic of higher estimated SNR per partner (DESIGN); 'A' always Mic A
    max_seg_s=60.0,       # longest segment for one EM run (DESIGN: whole file up to 60 s)
    overlap_s=4.0,        # crossfade between segments of longer files
)
DESIGN_PARAMS = dict(anchors="s1", load_partner=0.1, beam="mvdr", mask_scale=1.0, g_min=0.3, load_n=1e-3)
GUARD_SEP = 30e-6         # 's1+grid': no guard class within this of a partner anchor
RESOLVE_COS = 0.8         # anchors resolve A/B where |a_A^H a_B| / 2 <= this (37 deg apart)
REF_BAND = (300.0, 4000.0)
ALIGN_F_MIN = 500.0       # resolved bins below this do not enter the frame-level A share
SHARE_EPS = 1e-6          # A share of a frame without partner posterior -> 0.5
SEL_EPS = 1e-10           # epsilon of the partner selection gamma_k / (gamma_A + gamma_B + eps)
EPS = 1e-12


@dataclass
class Feats:
    """Normalized observations z = x / ||x|| per (bin, frame): p1 = |z_A|^2, c12 = z_A conj(z_B)."""
    p1: np.ndarray        # (F, N) float
    c12: np.ndarray       # (F, N) complex
    valid: np.ndarray     # (F, N) bool: ||x||^2 above 1e-10 x the bin's mean power (zero pad -> False)


@dataclass
class EMResult:
    """em() output."""
    gamma: np.ndarray     # (K, F, N) posteriors (invalid points: the prior)
    B: np.ndarray         # (K, F, 2, 2) shape matrices, trace 2
    alpha: np.ndarray     # (K, F) class weights
    loglik: list          # mean log-likelihood per valid point, before each M-step and at the end


@dataclass
class Separation:
    """separate() result. Frame and bin layout of common.stft. Classes: partner classes ('A', 'B', or one
    merged 'AB'; none in 'grid' mode), then guard classes 'G1', 'G2', ... (directional classes on a TDOA
    grid), then 'noise' (diffuse)."""
    classes: tuple        # e.g. ('G1', 'G2', 'G3', 'noise'), ('A', 'B', 'noise'), ('AB', 'noise')
    taus: tuple           # anchor TDOA (s) of each directional class (partners first, then guards)
    n_partner: int        # number of partner classes at the front of gamma
    gamma: np.ndarray     # (K, F, N) float32 class posteriors, A/B aligned to their anchors
    mask: np.ndarray      # (F, N) float32 post-mask clip(directional / mask_scale, g_min, 1)
    W: np.ndarray         # (2, F, N) complex64 operator: Y_lin = sum_m conj(W[m]) X[m] ([1, 0]: Mic A)
    Y_k: np.ndarray       # (P, F, N) complex64 per-partner MVDR outputs aligned to Mic A (None without MVDR)
    Y_lin: np.ndarray     # (F, N) partners' beams combined by posterior (Mic A without MVDR), before the mask
    diag: dict = field(default_factory=dict)

    @property
    def Y(self):
        """(F, N) final output STFT mask * Y_lin (computed on access)."""
        return self.mask * self.Y_lin

    @property
    def directional(self):
        """(F, N) posterior of 'any directional source' = 1 - noise posterior (partner + guard classes)."""
        return 1.0 - self.gamma[-1]

    @property
    def partner(self):
        """(F, N) posterior of 'any partner' (sum of the partner classes; zero without partner classes)."""
        return self.gamma[:self.n_partner].sum(axis=0)

    @property
    def noise(self):
        """(F, N) posterior of the noise class."""
        return self.gamma[-1]


def params(**overrides):
    """PARAMS with overrides; unknown keys raise."""
    bad = set(overrides) - set(PARAMS)
    if bad:
        raise KeyError(f"unknown cACGMM parameters {sorted(bad)}")
    return {**PARAMS, **overrides}


# ------------------------------------------------------------------ 2x2 Hermitian helpers
def _parts(B):
    """(..., 2, 2) Hermitian -> (b11, b22, b12) with real diagonals."""
    return B[..., 0, 0].real, B[..., 1, 1].real, B[..., 0, 1]


def _matrix(b11, b22, b12):
    """(b11, b22, b12) -> Hermitian (..., 2, 2)."""
    B = np.empty(np.shape(b11) + (2, 2), dtype=complex)
    B[..., 0, 0], B[..., 1, 1], B[..., 0, 1], B[..., 1, 0] = b11, b22, b12, np.conj(b12)
    return B


def _eig_floor(b11, b22, b12, floor):
    """Raise the smaller eigenvalue to floor * the larger one, then normalize the trace to 2."""
    half = 0.5 * (b11 + b22)
    disc = np.sqrt((0.5 * (b11 - b22)) ** 2 + np.abs(b12) ** 2)
    lmax, lmin = half + disc, half - disc
    # B + delta * P_min with the projector P_min = (lmax I - B) / (lmax - lmin)
    g = np.maximum(floor * lmax - lmin, 0.0) / np.maximum(2 * disc, EPS)
    b11, b22, b12 = b11 + g * (lmax - b11), b22 + g * (lmax - b22), b12 - g * b12
    s = 2.0 / np.maximum(b11 + b22, EPS)
    return b11 * s, b22 * s, b12 * s


def _principal(B):
    """Unit principal eigenvector of each 2x2 Hermitian B (..., 2, 2) -> (..., 2)."""
    b11, b22, b12 = _parts(B)
    lmax = 0.5 * (b11 + b22) + np.sqrt((0.5 * (b11 - b22)) ** 2 + np.abs(b12) ** 2)
    v = np.stack([b12, lmax - b11], axis=-1)
    w = np.stack([lmax - b22, np.conj(b12)], axis=-1)
    nv, nw = np.linalg.norm(v, axis=-1), np.linalg.norm(w, axis=-1)
    v = np.where((nv >= nw)[..., None], v, w)
    n = np.maximum(np.maximum(nv, nw), EPS)[..., None]
    return np.where(n > EPS, v / n, np.array([1.0, 0.0]))


# ------------------------------------------------------------------ cACGMM
def features(X):
    """Normalized 2-ch observations of an STFT X (2, F, N)."""
    pa, pb = np.abs(X[0]) ** 2, np.abs(X[1]) ** 2
    n2 = pa + pb
    valid = n2 > np.maximum(1e-10 * n2.mean(axis=1, keepdims=True), 1e-30)
    inv = np.where(valid, 1.0 / np.where(valid, n2, 1.0), 0.0)
    p1 = np.where(valid, pa * inv, 0.5)
    return Feats(p1=p1, c12=X[0] * np.conj(X[1]) * inv, valid=valid)


def anchor_init(f, taus, gamma_n, load_partner=PARAMS["load_partner"], load_noise=PARAMS["load_noise"]):
    """DESIGN M7 initial shape matrices (K, F, 2, 2), trace 2: one class per anchor TDOA in taus
    (B_k = a_k a_k^H + load_partner I, center-referenced steering), then the noise class
    B_0 = [[1, Gamma_n], [Gamma_n, 1]] + load_noise I (gamma_n real, (F,))."""
    f = np.asarray(f, dtype=float)
    out = []
    for tau in taus:
        a = cm.steer(f, tau)
        B = a[:, :, None] * np.conj(a[:, None, :]) + load_partner * np.eye(2)
        out.append(B)
    g = np.asarray(gamma_n, dtype=float)
    diag = np.full_like(g, 1.0 + load_noise)
    out.append(_matrix(diag, diag, g.astype(complex)))
    B = np.stack(out)
    return B * (2.0 / np.trace(B, axis1=-2, axis2=-1).real)[..., None, None]


def _quad(b11, b22, b12, det, p1, c12):
    """z^H B^-1 z for one class: parameters (F,), features (F, N) -> (F, N)."""
    num = b22[:, None] * p1 + b11[:, None] * (1.0 - p1) - 2.0 * (b12.real[:, None] * c12.real
                                                                   + b12.imag[:, None] * c12.imag)
    return np.maximum(num / det[:, None], EPS)


def em(feats, B, alpha, iterations, activity=None, prior=None, prior_frames=0.0,
       eig_floor=PARAMS["eig_floor"]):
    """EM for a K-class cACGMM, every bin at once.
    feats: Feats; B: (K, F, 2, 2) initial shape matrices; alpha: (K, F) initial weights.
    activity: optional (K, N) frame activities a_tk > 0 -> alpha_tfk = alpha_fk a_tk / D_tf with
    D_tf = sum_k alpha_fk a_tk (GSS-style); alpha_fk is then updated by the MM fixed point
    alpha_fk ∝ sum_t gamma_tfk / sum_t (a_tk / D_tf) (the plain mean of gamma would feed a_tk < 1
    back every iteration and drive the class to zero).
    prior: optional (K, F, 2, 2) anchor matrices mixed into each M-step with weight prior_frames.
    Starts with an E-step and ends with one, so gamma matches the returned B and alpha."""
    p1, c12, valid = feats.p1, feats.c12, feats.valid
    K = B.shape[0]
    b11, b22, b12 = (np.array(v) for v in _parts(B))
    b11, b22, b12 = _eig_floor(b11, b22, b12, eig_floor)
    alpha = np.array(alpha, dtype=float)
    vf = valid.astype(float)
    n_valid = np.maximum(vf.sum(axis=1), EPS)
    seen = vf.sum(axis=1) > 0                  # bins without any observation keep their weights
    act = None if activity is None else np.maximum(np.asarray(activity, dtype=float), EPS)
    if prior is not None:
        pb11, pb22, pb12 = _parts(prior)
    loglik = []
    gamma = np.empty((K,) + p1.shape)
    q = np.empty_like(gamma)
    for it in range(iterations + 1):
        det = b11 * b22 - np.abs(b12) ** 2
        log_alpha = np.log(np.maximum(alpha, EPS))
        for k in range(K):
            q[k] = _quad(b11[k], b22[k], b12[k], det[k], p1, c12)
            gamma[k] = (log_alpha[k] - np.log(det[k]))[:, None] - 2.0 * np.log(q[k])
            if act is not None:
                gamma[k] += np.log(act[k])[None, :]
        if act is not None:
            D = np.maximum(alpha.T @ act, EPS)             # (F, N)
            gamma -= np.log(D)[None]
        m = gamma.max(axis=0)
        np.exp(gamma - m, out=gamma)
        s = gamma.sum(axis=0)
        gamma /= s
        loglik.append(float(((m + np.log(s)) * vf).sum() / max(vf.sum(), 1.0)))
        if it == iterations:
            break
        for k in range(K):
            g = gamma[k] * vf
            w = g / q[k]
            s11, s22, s12 = (w * p1).sum(axis=1), (w * (1.0 - p1)).sum(axis=1), (w * c12).sum(axis=1)
            sc = 2.0 / np.maximum(s11 + s22, EPS)
            nb11, nb22, nb12 = s11 * sc, s22 * sc, s12 * sc
            n_k = g.sum(axis=1)
            empty = n_k < EPS
            nb11, nb22, nb12 = (np.where(empty, o, n) for o, n in ((b11[k], nb11), (b22[k], nb22), (b12[k], nb12)))
            if prior is not None and prior_frames > 0:
                wd = n_k / (n_k + prior_frames)
                nb11 = wd * nb11 + (1 - wd) * pb11[k]
                nb22 = wd * nb22 + (1 - wd) * pb22[k]
                nb12 = wd * nb12 + (1 - wd) * pb12[k]
            b11[k], b22[k], b12[k] = _eig_floor(nb11, nb22, nb12, eig_floor)
            alpha[k] = np.where(seen, n_k / n_valid, alpha[k])
        if act is not None:
            alpha = np.where(seen, alpha * n_valid / np.maximum(act @ (vf / D).T, EPS), alpha)
            alpha /= np.maximum(alpha.sum(axis=0), EPS)
    fi, ti = np.nonzero(~valid)           # no observation (zero pad): posterior = prior
    if len(fi):
        pr = alpha[:, fi] * (1.0 if act is None else act[:, ti])
        gamma[:, fi, ti] = pr / np.maximum(pr.sum(axis=0), EPS)
    return EMResult(gamma=gamma, B=_matrix(b11, b22, b12), alpha=alpha, loglik=loglik)


# ------------------------------------------------------------------ permutation
def resolved_bins(f, tau_a, tau_b, thr=RESOLVE_COS):
    """(F,) bool: bins where the two anchors' steering vectors differ enough to label A and B."""
    return np.abs(np.cos(np.pi * np.asarray(f) * (tau_a - tau_b))) <= thr


def align_ab(gamma, B, f, tau_a, tau_b, thr=RESOLVE_COS):
    """Label classes 0 / 1 as partners A / B per bin, in place.
    Resolved bins (resolved_bins): swap where the principal eigenvectors v_k of B_k match the other
    anchor better, |v_A^H a_B|^2 + |v_B^H a_A|^2 > |v_A^H a_A|^2 + |v_B^H a_B|^2 (DESIGN M7).
    Unresolved bins (the anchors coincide there, so no spatial test can label them): keep the EM's
    partner posterior gamma_A + gamma_B and split it by the frame's A share in the resolved bins
    >= 500 Hz, r_A(t) = mean gamma_A / (mean gamma_A + mean gamma_B) (frame-level decision carried
    over, DESIGN 1.1). Returns (swapped (F,) bool, resolved (F,) bool)."""
    f = np.asarray(f, dtype=float)
    res = resolved_bins(f, tau_a, tau_b, thr)
    v = _principal(B[:2])                                         # (2, F, 2)
    a = [cm.steer(f, t) / np.sqrt(2.0) for t in (tau_a, tau_b)]    # unit steering, (F, 2)
    sim = [[np.abs(np.sum(np.conj(v[k]) * a[j], axis=-1)) ** 2 for j in range(2)] for k in range(2)]
    swap = res & (sim[0][1] + sim[1][0] > sim[0][0] + sim[1][1])
    _swap(gamma, B, swap)
    ref = res & (f >= ALIGN_F_MIN)
    un = ~res
    if ref.any() and un.any():
        rA, rB = gamma[0, ref].mean(axis=0), gamma[1, ref].mean(axis=0)
        share = (rA + SHARE_EPS) / (rA + rB + 2 * SHARE_EPS)
        tot = gamma[0, un] + gamma[1, un]
        gamma[0, un], gamma[1, un] = tot * share, tot * (1.0 - share)
    return swap, res


def _swap(gamma, B, sel):
    """Exchange classes 0 and 1 in the bins sel (F,) bool, in place."""
    if sel.any():
        gamma[0, sel], gamma[1, sel] = gamma[1, sel].copy(), gamma[0, sel].copy()
        B[0, sel], B[1, sel] = B[1, sel].copy(), B[0, sel].copy()


# ------------------------------------------------------------------ beamforming
def covariance(X, w):
    """Weighted spatial covariance sum_t w X X^H / sum_t w per bin: X (2, F, N), w (F, N) -> (F, 2, 2)."""
    s = np.maximum(w.sum(axis=1), EPS)
    paa = (w * (X[0].real ** 2 + X[0].imag ** 2)).sum(axis=1) / s
    pbb = (w * (X[1].real ** 2 + X[1].imag ** 2)).sum(axis=1) / s
    pab = (w * X[0] * np.conj(X[1])).sum(axis=1) / s
    return _matrix(paa, pbb, pab)


def mvdr(X, gamma, partners, taus, f=cm.FREQS, ref="snr", load_n=PARAMS["load_n"]):
    """Souden MVDR per partner class from posterior-weighted covariances of X (2, F, N); the last class
    of gamma (K, F, N) is noise. partners: indices of the partner classes, taus: their anchors (s).
    Returns (H (P, F, 2) with Y_k = H[k]^H X aligned to Mic A, refs (P,) reference mic per partner)."""
    PhiN = covariance(X, gamma[-1])
    PhiN = PhiN + load_n * 0.5 * np.trace(PhiN, axis1=-2, axis2=-1).real[:, None, None] * np.eye(2)
    inv = np.linalg.inv(PhiN)
    band = (f >= REF_BAND[0]) & (f <= REF_BAND[1])
    H, refs = [], []
    for k, tau in zip(partners, taus):
        Phik = covariance(X, gamma[k])
        M = inv @ Phik
        tr = np.trace(M, axis1=-2, axis2=-1)
        dk, dn = (P[band][:, [0, 1], [0, 1]].real.sum(axis=0) for P in (Phik, PhiN))
        snr = dk / np.maximum(dn, EPS)
        m = int(np.argmax(snr)) if ref == "snr" else 0
        h = M[:, :, m] / np.where(np.abs(tr) > EPS, tr, np.inf)[:, None]
        if m == 1:            # estimate at Mic B -> advance by the anchor delay to align with Mic A
            h = h * np.exp(-2j * np.pi * f * tau)[:, None]
        H.append(h)
        refs.append(m)
    return np.stack(H), np.array(refs)


def steer_mvdr(X, gamma, partners, taus, f=cm.FREQS, load_n=PARAMS["load_n"]):
    """MVDR toward each partner anchor's plane wave g = [1, exp(-j 2 pi f tau_k)] (Mic-A referenced, so
    distortionless for that plane wave at Mic A): h_k = Phi_N^-1 g / (g^H Phi_N^-1 g), Phi_N from the noise
    posterior (last class of gamma) with diagonal loading load_n tr(Phi_N) / 2. Returns H (P, F, 2)."""
    PhiN = covariance(X, gamma[-1])
    PhiN = PhiN + load_n * 0.5 * np.trace(PhiN, axis1=-2, axis2=-1).real[:, None, None] * np.eye(2)
    inv = np.linalg.inv(PhiN)
    H = []
    for tau in taus[:len(partners)]:
        g = np.stack([np.ones(len(f)), np.exp(-2j * np.pi * f * tau)], axis=-1)
        num = np.einsum("fij,fj->fi", inv, g)
        H.append(num / np.maximum(np.einsum("fi,fi->f", np.conj(g), num).real, EPS)[:, None])
    return np.stack(H)


# ------------------------------------------------------------------ pipeline
def _segments(N, max_len, overlap):
    """Frame ranges [(start, end)] of at most max_len frames overlapping by `overlap` frames."""
    if N <= max_len:
        return [(0, N)]
    n = int(np.ceil((N - overlap) / (max_len - overlap)))
    length = int(np.ceil((N + (n - 1) * overlap) / n))
    starts = [i * (length - overlap) for i in range(n)]
    return [(s, min(N, s + length)) for s in starts]


def _ramp(N, segs, i, overlap):
    """Crossfade weight of segment i over its own frames (weights of all segments sum to 1)."""
    s, e = segs[i]
    w = np.ones(e - s)
    if i > 0:
        o = segs[i - 1][1] - s
        w[:o] = (np.arange(o) + 0.5) / o
    if i < len(segs) - 1:
        o = e - segs[i + 1][0]
        w[e - s - o:] = 1.0 - (np.arange(o) + 0.5) / o
    return w


def _activity(trk, n_partner, n_guard, floor):
    """GSS-style frame activities (K, N) from S1 posteriors, a = floor + (1 - floor) p (merged class:
    p_A + p_B, clipped to 1); guard and noise classes have activity 1."""
    pa, pb = np.nan_to_num(trk.p_A), np.nan_to_num(trk.p_B)
    rows = [pa, pb] if n_partner == 2 else [np.clip(pa + pb, 0.0, 1.0)]
    return np.stack([floor + (1.0 - floor) * r for r in rows] + [np.ones(len(pa))] * (n_guard + 1))


def grid_taus(d_hat, n, avoid=(), sep=GUARD_SEP):
    """Guard-class TDOAs: linspace(-1, 1, n) d_hat / c, without points within `sep` of the `avoid` TDOAs."""
    g = np.linspace(-1.0, 1.0, n) * d_hat / C if n > 0 else np.zeros(0)
    return tuple(float(t) for t in g if all(abs(t - a) > sep for a in avoid))


def _separate_segment(X, taus, gamma_n, f, act, n_partner, p):
    """One EM (+ optional per-partner MVDR) run over a segment X (2, F, n). taus: partner anchors first,
    then guards."""
    K = len(taus) + 1
    feats = features(X)
    B0 = anchor_init(f, taus, gamma_n, p["load_partner"], p["load_noise"])
    res = em(feats, B0.copy(), np.full((K, len(f)), 1.0 / K), p["iterations"], activity=act,
             prior=B0, prior_frames=p["prior_frames"], eig_floor=p["eig_floor"])
    gamma = res.gamma
    diag = {"loglik": res.loglik}
    if n_partner == 2:
        swapped, resolved = align_ab(gamma, res.B, f, taus[0], taus[1])
        diag.update(swapped=swapped, resolved=resolved)
    H = None
    if p["beam"] == "mvdr" and n_partner:
        H, refs = mvdr(X, gamma, list(range(n_partner)), taus[:n_partner], f, p["ref"], p["load_n"])
        diag.update(refs=refs, H=H)
    elif p["beam"] == "steer" and n_partner:
        H = steer_mvdr(X, gamma, list(range(n_partner)), taus[:n_partner], f, p["load_n"])
        diag.update(H=H)
    return gamma, H, diag


def separate_anchored(X, taus, gamma_n, f=cm.FREQS, act=None, guards=(), **overrides):
    """M7 core with explicit anchors. X: (2, F, N) calibrated STFT; taus: 0, 1 or 2 partner TDOAs (s)
    (two: classes A, B; one: a merged partner class 'AB'); guards: TDOAs of extra directional classes
    (G1, G2, ...); the last class is noise. gamma_n: (F,) real noise coherence.
    act: optional (K, N) frame activities (GSS-style class priors, see em()). Returns a Separation."""
    p = params(**overrides)
    taus, guards = tuple(float(t) for t in taus), tuple(float(t) for t in guards)
    P, K = len(taus), len(taus) + len(guards) + 1
    if K < 2:
        raise ValueError("separate_anchored needs at least one partner or guard TDOA")
    F, N = X.shape[1], X.shape[2]
    max_len = max(1, int(round(p["max_seg_s"] / cm.HOP_S)))
    overlap = min(int(round(p["overlap_s"] / cm.HOP_S)), max_len // 2)
    segs = _segments(N, max_len, overlap)
    if p["beam"] not in ("none", "mvdr", "steer"):
        raise ValueError(f"beam must be 'none', 'mvdr' or 'steer', not {p['beam']!r}")
    gamma = np.zeros((K, F, N), dtype=np.float32)
    beam = p["beam"] != "none" and P > 0
    W = np.zeros((2, F, N), dtype=np.complex64)
    Y_k = np.zeros((P, F, N), dtype=np.complex64) if beam else None
    Y_lin = np.zeros((F, N), dtype=complex) if beam else X[0].copy()
    diags = []
    for i, (s, e) in enumerate(segs):
        Xs = X[:, :, s:e]
        g, H, dg = _separate_segment(Xs, taus + guards, gamma_n, f, None if act is None else act[:, s:e], P, p)
        w = _ramp(N, segs, i, overlap)
        gamma[:, :, s:e] += (g * w).astype(np.float32)
        if beam:
            # selection gamma_k / (gamma_A + gamma_B): a convex combination even where both are ~0
            sel = (g[:P] + SEL_EPS / P) / (g[:P].sum(axis=0) + SEL_EPS)
            yk = np.einsum("kfm,mft->kft", np.conj(H), Xs)
            Y_k[:, :, s:e] += (yk * w).astype(np.complex64)
            Y_lin[:, s:e] += (sel * yk).sum(axis=0) * w
            W[:, :, s:e] += (np.einsum("kft,kfm->mft", sel, H) * w).astype(np.complex64)
        diags.append(dict(dg, frames=(s, e)))
    if not beam:
        W[0] = 1.0
    mask = np.clip((1.0 - gamma[-1]) / p["mask_scale"], p["g_min"], 1.0).astype(np.float32)
    names = {2: ("A", "B"), 1: ("AB",), 0: ()}[P] + tuple(f"G{i + 1}" for i in range(len(guards)))
    return Separation(classes=names + ("noise",), taus=taus + guards, n_partner=P, gamma=gamma, mask=mask,
                      W=W, Y_k=Y_k, Y_lin=Y_lin, diag={"segments": diags, "params": p})


def anchors_from_track(trk):
    """(taus, mode) from an S1 Track (bus partner model), following its 'inseparable' reasons:
    '3class'      two TDOA peaks, not flagged -> anchors (tau_A, tau_B);
    'merged'      two peaks < 25 us apart ('close_tdoas': one cone) -> one partner class at their mean;
    'passthrough' fewer than two peaks ('one_peak': the other partner's direction is unknown, and a beam
                  plus mask on the one found suppresses it) or both partners near broadside
                  ('both_broadside': partner and diffuse classes coincide) -> Mic A (DESIGN 2.1 omni)."""
    ta, tb = trk.tau_A, trk.tau_B
    why = trk.inseparable_why
    if "one_peak" in why or "both_broadside" in why or not (np.isfinite(ta) and np.isfinite(tb)):
        return (), "passthrough"
    if "close_tdoas" in why:
        return (0.5 * (ta + tb),), "merged"
    return (ta, tb), "3class"


def passthrough(X, **overrides):
    """Identity Separation (Mic A, mask 1, partner posterior 1) for the 'passthrough' mode."""
    p = params(**overrides)
    F, N = X.shape[1], X.shape[2]
    W = np.zeros((2, F, N), dtype=np.complex64)
    W[0] = 1.0
    gamma = np.zeros((2, F, N), dtype=np.float32)
    gamma[0] = 1.0
    return Separation(classes=("AB", "noise"), taus=(), n_partner=1, gamma=gamma, mask=np.ones((F, N), np.float32),
                      W=W, Y_k=None, Y_lin=X[0].copy(), diag={"segments": [], "params": p})


def separate(X, cal, trk, **overrides):
    """M7 on the S0/S1 front end: X = tracker.front_end().X (2, 257, N), cal = its Calib (d_hat, gamma_n),
    trk = its Track (bus partner model: tau_A, tau_B, inseparable_why, p_A, p_B). Directional classes per
    p['anchors']: 'grid' guard classes only (no S1 input); 's1' S1 partner classes (DESIGN; flags ->
    merged or passthrough, anchors_from_track); 's1+grid' both. Returns a Separation with diag['mode']
    ('grid', or S1's '3class' | 'merged' | 'passthrough') and diag['inseparable_why']."""
    p = params(**overrides)
    if p["anchors"] not in ("grid", "s1", "s1+grid"):
        raise ValueError(f"anchors must be 'grid', 's1' or 's1+grid', not {p['anchors']!r}")
    taus, mode = anchors_from_track(trk) if p["anchors"] != "grid" else ((), "grid")
    guards = grid_taus(cal.d_hat, p["grid_n"], taus) if p["anchors"] != "s1" else ()
    if not taus and not guards:
        sep = passthrough(X, **overrides)
    else:
        act = None
        if p["activity"] == "s1" and taus:
            act = _activity(trk, len(taus), len(guards), p["act_floor"])
        sep = separate_anchored(X, taus, cal.gamma_n, act=act, guards=guards, **overrides)
    sep.diag.update(mode=mode, inseparable_why=trk.inseparable_why)
    return sep


def apply(sep, X, post_mask=True):
    """Apply the separation's per-bin operator (and post-mask) to another 2-ch STFT of the same shape,
    e.g. one scene component: linear in X given the masks."""
    Y = np.einsum("mft,mft->ft", np.conj(sep.W.astype(complex)), X)
    return sep.mask * Y if post_mask else Y
