"""M3: geometry-free RTF-MVDR (covariance whitening) and mask-driven MVDR (DESIGN.md M3, 2.0, 2.1).

Use:
    fe = tracker.front_end(x, fs, info, online=online)       # S0 + S1
    bm = rtf.run(fe, info, online=online, **overrides)      # M3; overrides: keys of PARAMS
    y = common.istft(bm.Y, x.shape[1])                      # linear beam output, centre-referenced

Bus, per update block of 0.5 s (DESIGN M3 and the 2.1 state machine):
1. Covariance weights. labels='posterior': L_k = {p_k > 0.85, directness above its median, no double
   talk}, L_0 = {p_A + p_B < 0.15}. labels='mask': m_k(l,f) = p_k(l) * D_k(l,f) with the CDR-like cue
   D_k = CDR_k / (1 + CDR_k) (DOA-dependent CDR, M4 conditioning), m_0 = (1 - max p) (1 - max D);
   or external masks (M7/M8) through `masks`.
2. Recursive covariances over weighted frames, Phi <- (1 - a m) Phi + a m X X^H with a = 1 / (3 s of
   frames) (memory counted in labelled frames). Online: frames before the block only; offline: forward
   plus backward recursion around the block centre (two-sided window).
3. Noise covariance Phi_w = Gamma_n + sigma_s I (noise='hybrid'; Gamma_n from S0, sigma_s = self-noise
   ratio from the quietest 5 % of frames), or the measured covariance over L_0 (noise='measured').
4. Covariance whitening (DESIGN 1.6 item 3): L = chol(Phi_w), v = principal eigenvector of
   L^-1 Phi_x L^-H, g = L v / (L v)_1 (= Phi_w u_max, not v or u_max). Fewer than 50 labelled frames, a
   bin failing the plausibility check (300-3000 Hz: |angle(G e^{+j2pi f tau_k})| < 60 deg and |G| within
   +-3 dB), a gross error anywhere (|G| beyond +-10 dB) or a whitened SNR lambda_1 / lambda_2 - 1 below
   5 dB -> model RTF exp(-j 2 pi f tau_k) (M2 steering).
5. Symmetrized RTF g~ = [1, G] / sqrt(G) (DESIGN 2.0), the phase of G unwrapped against -2 pi f tau_k,
   so a plane wave gives exactly common.steer(f, tau_k).
6. Beams, WNG floors -15 dB < 500 Hz, -10 dB to 2 kHz, -6 dB above (meeting -3 dB), WNG computed with the
   steering scaled to ||g||^2 = 2, mu bisected over log10 mu in [-9, 6] (40 steps):
   - w_k = (Phi_w + mu I)^-1 g~_k / g~_k^H (Phi_w + mu I)^-1 g~_k (beam='mvdr'), or the Souden form
     (Phi_w + mu I)^-1 Phi_x,k e_A / tr(.) re-referenced to the centre (beam='souden');
   - w_AB = (Phi_w + mu I + lam_c A A^H)^-1 lam_c A h*, A = [g~_A, g~_B], h = [1, e^{j psi}], lam_c = 1e3,
     with the M2 phase rule: psi = 180 deg in the dipole band (contiguous feasible bins ending at c/2d),
     continuation search (2.5 deg grid, +-30 deg window, then full circle, then Mic A) elsewhere; partners
     not on opposite sides: full-circle search at 1 kHz, then continuation both ways;
   - fallback (DESIGN 2.1) where S1 flags the partners inseparable (one peak, TDOAs < 25 us apart, both
     near broadside), a partner TDOA lies beyond 1.1 d_hat / c, or (same_side='fallback') both TDOAs lie
     on one side of broadside: fallback='auto' uses the broadside MVDR only when both partners are near
     broadside and Mic A (omni) otherwise; 'broadside' / 'omni' force one.
7. States A / AB / B (enter A when p_A > 0.75 and p_B < 0.2 for 100 ms; AB on double talk or doubt;
   hold when nobody talks); state A uses w_A with A's AB response phase, state B uses w_B with B's AB
   response phase; per-bin phase-continuity check (DESIGN M2); 50 ms linear weight crossfades.
   states='ab' keeps the AB beam throughout (ablation).
Meeting (optional in DESIGN): meeting='omni' (default) passes Mic A through; 'cluster' steers one RTF-MVDR
per S1 talker cluster to the active cluster (0.3 s minimum dwell, 50 ms crossfades); 'broadside' is the
fixed broadside MVDR.

The output Y = w^H X is referenced to the array centre: partner k appears tau_k / 2 later than at Mic A
(|tau_k| / 2 <= 1.4 samples at 6 cm), and the AB beam passes partner B with its h_B (-1 in the dipole band).

Public API: run(), enhance(), Beam (Y, apply(), weights(), table, entry, df, G, g, diag), and the building
blocks params(), wng_floor_db(), noise_model(), normalize_cov(), self_noise_ratio(), cw_rtf(), checked_rtf(),
plausible(), model_rtf(), sym_rtf(), mvdr(), souden(), ab_beam(), bus_state_weights(), bus_states(),
cluster_states(), df_table(), recursive_cov(), cdr_cue(), mic_a_weights().

Deviations from DESIGN (minimal; where DESIGN is silent the choice is stated):
- Continuation search: the +-30 deg window bounds the actual response phase as well as the nominal psi
  (with a large mu the soft constraints relax and the response leaves the nominal window); larger steps
  happen only through the DESIGN full-circle fallback (1-2 bins at 4 and 6 cm with the bus floors, where
  the smallest feasible rotation is 31-55 deg because the floor steps from -10 to -15 dB at 500 Hz).
- Two-talker WNG and balance use both partners (min WNG), the floor is strict (no 0.01 dB slack), DF ties
  are broken toward the neighbour's phase, and the AB weights are scaled to sqrt(|r_A| |r_B|) = 1.
- States A / B take the AB beam's actual response phase for their partner (DESIGN: the nominal h_A = 1,
  h_B = e^{j psi}; identical where the constraints hold).
- RTF symmetrization unwraps against the plane-wave model instead of along frequency (two bad bins cannot
  flip the sign of g~ above them); outside 300-3000 Hz a +-10 dB gross-error guard is added.
- Offline: two-sided recursion and a 0.1 s look-ahead for leaving A / B (offline two-pass processing).
  Online: a partner's covariance restarts when its S1 TDOA moves by more than 20 us.
- Tuned on v2 dev (bus; master/slave scoring of the y1 weights; numbers in the tune:rtf report):
  * fallback 'auto' (DESIGN 2.1 allows a broadside MVDR or omni) and the beyond-endfire TDOA check (as M2's
    implausible_reasons): with one S1 peak the other partner's direction is unknown, and the broadside beam
    (for a real Gamma_n the plain average) comb-filters endfire partners; S1 peaks piled up at its
    +-1.2 d_hat / c grid edge (40 mm spacing prior on a 60 mm array) are not plane waves at d_hat;
  * same_side='fallback' (DESIGN: full-circle search for partners on one side): S1 TDOA pairs on one side
    of broadside were mostly a found partner plus a spurious peak, and the AB beam then nulled the other
    partner; the cost with correct partner labels was 0.001 worst-partner ESTOI;
  * min_snr_db=5 (not in DESIGN): even with oracle labels, covariance-whitening RTFs biased by coherent
    interferers (engine, transients, passengers) that Phi_w does not model lost to plane-wave steering at
    the true TDOA; keeping only bins with a whitened SNR >= 5 dB recovered most of that loss;
  * meeting mode off by default: steering to S1's talker clusters raised the mean talker ESTOI but failed
    the worst-talker and 19-of-20 no-regression criteria (dev rooms r00-r04).
- Mask variant wrappers use the hybrid noise model, not DESIGN's mask-weighted measured Phi_n (partner
  leakage with S1-derived masks).
- Not specified by DESIGN: sigma_s (see self_noise_ratio), the mask-mode weights, the Souden reference
  (Mic A, then re-referenced to the centre), "100 ms" = 13 frames, meeting dwell 0.3 s and broadside
  fallback before the first cluster.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from lab.common import C
from . import calib as cb
from . import common as cm
from . import tracker as tk

A_, AB_, B_, FB_ = 0, 1, 2, 3          # bus states: partner A, both partners, partner B, fallback
STATE_NAMES = ("A", "AB", "B", "FB")

PARAMS = dict(
    labels="posterior",      # covariance weights: 'posterior' (DESIGN frame labels) or 'mask'
    p_label=0.85,            # L_k: p_k > p_label ...
    direct_pct=50.0,         # ... and S1 directness above this percentile of the valid frames
    p_noise=0.15,            # L_0: p_A + p_B < p_noise
    mask_cue="post*cdr",     # labels='mask': 'post', 'cdr' or 'post*cdr'
    lam_cdr=cm.LAM_76MS,     # coherence smoothing of the CDR cue (M4 default lambda 0.9)
    memory_s=3.0,            # effective memory of the recursions, in weighted (labelled) frames
    update_s=0.5,            # RTF and weight refresh interval
    min_frames=50,           # fewer (weighted) labelled frames: no estimate, model steering
    noise="hybrid",          # 'hybrid' (Gamma_n + sigma_s I) or 'measured' (covariance over L_0)
    sigma_s=None,            # hybrid model's white-noise ratio: None = estimated (below), or a fixed value
    quiet_pct=5.0,           # sigma_s: the quietest 5 % of the valid frames
    sigma_min=1e-4,          # floor of sigma_s (conditioning only)
    beam="mvdr",             # per-partner beam: 'mvdr' or 'souden'
    check_band=(300.0, 3000.0),
    check_deg=60.0,
    check_db=3.0,
    guard_db=10.0,           # any bin: |G| beyond this is a gross error -> model
    min_snr_db=5.0,          # keep an estimate only where the whitened SNR lambda_1 / lambda_2 - 1 reaches this
                             # (None: off, DESIGN)
    wng_db=((500.0, -15.0), (2000.0, -10.0), (np.inf, -6.0)),   # (upper edge Hz, floor dB) pairs
    lam_c=1e3,
    mu_lo=-9.0,
    mu_hi=6.0,
    mu_steps=40,
    psi_step=2.5,            # deg
    psi_win=30.0,            # deg
    bal_db=1.0,              # |r_A| / |r_B| within +-1 dB
    dipole_deg=10.0,         # dipole accepted if angle(r_B / r_A) within 180 +- this
    anchor_hz=1000.0,        # full-circle anchor for partners not on opposite sides
    p_enter=0.75,
    p_other=0.2,
    hold_s=0.1,
    lookahead_s=0.1,         # offline only: leave A / B this early (posteriors lag; 0 online)
    xfade_s=0.05,
    phase_deg=(30.0, 90.0),  # phase-continuity limits for |r| >= phase_mag and below
    phase_mag=0.5,
    fallback="auto",         # inseparable partners: 'auto' (broadside MVDR only when S1 says both partners are
                             # near broadside, else Mic A), 'broadside' or 'omni' (Mic A)
    states="s1",             # bus: 's1' (state machine A / AB / B) or 'ab' (the AB beam throughout)
    max_tdoa=1.1,            # bus: a partner TDOA beyond max_tdoa * d_hat / c is implausible -> fallback
    same_side="fallback",    # bus, partner TDOAs on the same side of broadside: 'fallback' or 'search' (DESIGN M2
                             # full-circle search at anchor_hz)
    meeting="omni",          # meeting: 'omni' (Mic A: no spatial processing), 'cluster' (RTF-MVDR steered to the
                             # active S1 cluster) or 'broadside' (fixed broadside MVDR)
    dwell_s=0.3,             # meeting: minimum time on a cluster before switching
)
SCENARIOS = {"bus": {}, "meeting": dict(wng_db=((np.inf, -3.0),))}


def params(scenario="bus", **overrides):
    """Parameter set for a scenario with keyword overrides (unknown keys raise)."""
    p = dict(PARAMS, **SCENARIOS[scenario])
    bad = set(overrides) - set(p)
    if bad:
        raise TypeError(f"unknown M3 parameters: {sorted(bad)}")
    p.update(overrides)
    return p


def wng_floor_db(spec, f=cm.FREQS):
    """(F,) WNG floor in dB from ((upper edge Hz, dB), ...) in increasing order."""
    out, lo = np.empty(len(f)), -np.inf
    for hi, db in spec:
        out[(f >= lo) & (f < hi)] = db
        lo = hi
    return out


# ------------------------------------------------------------------ 2x2 algebra (vectorized)
_I2 = np.eye(2)


def _dot(w, g):
    """w^H g over the last axis."""
    return np.conj(w[..., 0]) * g[..., 0] + np.conj(w[..., 1]) * g[..., 1]


def _n2(v):
    return v[..., 0].real ** 2 + v[..., 0].imag ** 2 + v[..., 1].real ** 2 + v[..., 1].imag ** 2


def _mv(Q, v):
    return np.stack([Q[..., 0, 0] * v[..., 0] + Q[..., 0, 1] * v[..., 1],
                     Q[..., 1, 0] * v[..., 0] + Q[..., 1, 1] * v[..., 1]], axis=-1)


def _quad(w, Q):
    """Real w^H Q w for Hermitian Q."""
    return np.real(_dot(w, _mv(Q, w)))


def _adj(Q):
    """Adjugate of 2x2 matrices: Q^-1 = adj(Q) / det(Q), and adj(Q + mu I) = adj(Q) + mu I."""
    A = np.empty_like(Q)
    A[..., 0, 0], A[..., 1, 1] = Q[..., 1, 1], Q[..., 0, 0]
    A[..., 0, 1], A[..., 1, 0] = -Q[..., 0, 1], -Q[..., 1, 0]
    return A


def _min_mu(a, b, gs, fl, p):
    """Diagonal loading by bisection (DESIGN M2/M3): the beam direction is u(mu) = a + mu b (all beams here
    have this form because adj(Q + mu I) = adj(Q) + mu I), WNG_k(mu) = 2 |u^H g_k|^2 / (||g_k||^2 ||u||^2),
    a ratio of quadratics in mu. Returns (log10 of the smallest mu in [mu_lo, mu_hi] with min_k WNG_k >= fl,
    found by mu_steps bisection steps on log10 mu; feasible = the floor is met there)."""
    n0, n1, n2 = _n2(a), np.real(_dot(a, b)), _n2(b)
    terms = []
    for g in gs:
        c = 2.0 / _n2(g)
        ag, bg = _dot(a, g), _dot(b, g)
        terms.append((c * np.abs(ag) ** 2, c * np.real(np.conj(ag) * bg), c * np.abs(bg) ** 2))

    def wng(lmu):
        mu = 10.0 ** lmu
        num = np.minimum.reduce([t0 + mu * (2 * t1 + mu * t2) for t0, t1, t2 in terms])
        return num / np.maximum(n0 + mu * (2 * n1 + mu * n2), 1e-300)

    fl = fl * (1 - 1e-9)
    lo, hi = np.full(np.shape(n0), p["mu_lo"]), np.full(np.shape(n0), p["mu_hi"])
    ok0 = wng(lo) >= fl
    for _ in range(p["mu_steps"]):
        mid = 0.5 * (lo + hi)
        good = wng(mid) >= fl
        hi, lo = np.where(good, mid, hi), np.where(good, lo, mid)
    lmu = np.where(ok0, p["mu_lo"], hi)
    return lmu, wng(lmu) >= fl


def _outer(g):
    return g[..., :, None] * np.conj(g[..., None, :])


def _herm(paa, pbb, pab):
    """(paa, pbb, pab = E{X_A X_B*}) -> Hermitian (..., 2, 2)."""
    R = np.empty(np.shape(pab) + (2, 2), complex)
    R[..., 0, 0], R[..., 1, 1] = paa, pbb
    R[..., 0, 1], R[..., 1, 0] = pab, np.conj(pab)
    return R


def normalize_cov(R):
    """Scale each 2x2 covariance to unit mean diagonal (loading and LCMV weights assume it)."""
    return R / np.maximum(0.5 * np.real(R[..., 0, 0] + R[..., 1, 1]), cm.EPS)[..., None, None]


# ------------------------------------------------------------------ noise model and RTF estimation
def noise_model(gamma_n, sigma_s):
    """Hybrid noise covariance Phi_w = Gamma_n + sigma_s I, shape gamma_n.shape + (2, 2)."""
    g = np.asarray(gamma_n, float)
    s = np.broadcast_to(np.asarray(sigma_s, float), g.shape)
    return _herm(1.0 + s, 1.0 + s, g.astype(complex))


def self_noise_ratio(X, frames_q, p_noise, cal, sigma_min=1e-4):
    """sigma_s(f): spatially white (self-)noise power over the noise power p_noise(f) (DESIGN M3, hybrid
    model; DESIGN leaves the estimator open). In the quiet frames frames_q, the uncorrelated power
    P - Re Phi_AB / gamma (gamma = S0 diffuse model at d_hat, without rho) is read where gamma >= 0.5 and
    f >= 200 Hz; its median is the white level. Coherent energy in the quiet frames only lowers it."""
    paa, pbb, pab = (r.mean(axis=-1) for r in cm.spectra(X[:, :, frames_q]))
    gam = cm.diffuse_model(cm.FREQS, cal.d_hat, cal.model)
    k = (gam >= 0.5) & (cm.FREQS >= 200.0)
    u = 0.5 * (paa + pbb) - np.real(pab) / np.maximum(gam, 0.5)
    p_self = max(float(np.median(np.maximum(u[k], 0.0))), 0.0)
    return np.clip(p_self / np.maximum(p_noise, cm.EPS), sigma_min, 1e3)


def cw_rtf(Phi_x, Phi_w):
    """Covariance-whitening RTF (DESIGN M3 step list, 1.6 item 3). Phi_x, Phi_w: (..., 2, 2) Hermitian,
    Phi_w positive definite. L = chol(Phi_w), M = L^-1 Phi_x L^-H, v = principal eigenvector of M,
    g = L v / (L v)_1. Returns (G = g_2 (...,) complex, eigenvalue ratio lambda_1 / lambda_2 (...,))."""
    L = np.linalg.cholesky(Phi_w)
    Li = np.zeros_like(L)
    Li[..., 0, 0] = 1 / L[..., 0, 0]
    Li[..., 1, 1] = 1 / L[..., 1, 1]
    Li[..., 1, 0] = -L[..., 1, 0] / (L[..., 0, 0] * L[..., 1, 1])
    M = Li @ Phi_x @ np.conj(np.swapaxes(Li, -1, -2))
    M = 0.5 * (M + np.conj(np.swapaxes(M, -1, -2)))
    lam, V = np.linalg.eigh(M)
    g = _mv(L, V[..., :, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        G = g[..., 1] / g[..., 0]
        ratio = lam[..., 1] / np.maximum(lam[..., 0], cm.EPS)
    return G, ratio


def plausible(G, f, tau, band, max_deg, max_db, guard_db):
    """True where the RTF G (Mic B over Mic A) is kept: finite and within +-guard_db everywhere; inside
    band also |angle(G exp(+j 2 pi f tau))| < max_deg and |20 log10 |G|| <= max_db (DESIGN M3)."""
    tau = np.asarray(tau, float)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = 20 * np.log10(np.abs(G))
        dev = np.degrees(np.abs(np.angle(G * np.exp(2j * np.pi * f * tau))))
    ok = np.isfinite(G) & (np.abs(lev) <= guard_db)
    inb = (f >= band[0]) & (f <= band[1])
    return ok & ~(inb & ((dev >= max_deg) | (np.abs(lev) > max_db)))


def model_rtf(f, tau):
    """Plane-wave RTF of Mic B over Mic A, exp(-j 2 pi f tau), shape tau.shape + f.shape."""
    return np.exp(-2j * np.pi * f * np.asarray(tau, float)[..., None])


def sym_rtf(G, f, tau):
    """Centre-referenced RTF g~ = [1, G] / sqrt(G) (DESIGN 2.0): the phase of G is unwrapped against the
    plane-wave model -2 pi f tau before halving, so model_rtf(f, tau) gives exactly common.steer(f, tau).
    Returns (half = sqrt(G) on that branch, g~ (..., F, 2))."""
    tau = np.asarray(tau, float)[..., None]
    ph = -2 * np.pi * f * tau + np.angle(G * np.exp(2j * np.pi * f * tau))
    half = np.sqrt(np.abs(G)) * np.exp(0.5j * ph)
    return half, np.stack([1 / half, half], axis=-1)


# ------------------------------------------------------------------ beams
def mvdr(R, g, floor_db, p=PARAMS):
    """MVDR w = (R + mu I)^-1 g / (g^H (R + mu I)^-1 g) with the smallest mu meeting WNG >= floor, WNG
    computed with g scaled to ||g||^2 = 2 (log10 mu bisected over [mu_lo, mu_hi], mu_steps steps; mu -> inf
    is delay-and-sum, WNG +3 dB, so it always succeeds). R: (..., F, 2, 2) with unit mean diagonal,
    g: (..., F, 2), floor_db: (F,). Returns (w (..., F, 2), log10 mu (..., F))."""
    a = _mv(_adj(R), g)
    lmu, _ = _min_mu(a, g, [g], 10 ** (np.asarray(floor_db) / 10), p)
    u = a + (10.0 ** lmu)[..., None] * g
    return u / np.real(_dot(g, u))[..., None], lmu


def souden(Phi_s, R, g, half, floor_db, p=PARAMS):
    """Souden MVDR h = (R + mu I)^-1 Phi_s e_A / tr((R + mu I)^-1 Phi_s) (DESIGN M3 mask variant) with the
    smallest mu meeting the WNG floor (WNG w.r.t. g scaled to ||g||^2 = 2), re-referenced from Mic A to the
    array centre by conj(sqrt(G)) (half), so its response to g~ equals its Mic-A response (real for a rank-1
    Phi_s). R: unit-mean-diagonal noise covariance; Phi_s: (..., F, 2, 2); g, half: centre-referenced RTF and
    sqrt(G) from sym_rtf()."""
    AR = _adj(R)
    a, b = _mv(AR, Phi_s[..., :, 0]), Phi_s[..., :, 0]
    lmu, _ = _min_mu(a, b, [g], 10 ** (np.asarray(floor_db) / 10), p)
    mu = 10.0 ** lmu
    tr = np.trace(AR @ Phi_s, axis1=-2, axis2=-1) + mu * np.trace(Phi_s, axis1=-2, axis2=-1)
    tr = np.where(np.abs(tr) > 1e-300, tr, 1.0)          # empty Phi_s (no weighted frames): h = 0
    return (a + mu[..., None] * b) / tr[..., None] * np.conj(half)[..., None]


def _lcmv_cands(R, gA, gB, fl, psi, p):
    """Soft LCMV for candidate phases psi (..., n): w = (R + mu I + lam_c A A^H)^-1 lam_c A h*,
    A = [gA, gB], h = [1, e^{j psi}], mu the smallest value (bisection) with min_k WNG_k >= floor.
    R (..., 2, 2), gA/gB (..., 2), fl (...,) linear. Returns w (..., n, 2), r_A, r_B, df, ok (..., n) with
    ok = floor met and |r_A| / |r_B| within +-bal_db; WNG, DF and the balance use the steering scaled to
    ||g||^2 = 2, DF = mean_k |r_k|^2 / w^H R w."""
    lam = p["lam_c"]
    R_ = R[..., None, :, :]
    P = R_ + lam * (_outer(gA) + _outer(gB))[..., None, :, :]
    gA_, gB_ = gA[..., None, :], gB[..., None, :]
    rhs = lam * (gA_ + np.exp(-1j * psi)[..., None] * gB_)
    a = _mv(_adj(P), rhs)
    lmu, feas = _min_mu(a, rhs, [gA_, gB_], np.asarray(fl)[..., None], p)
    mu = 10.0 ** lmu
    det = np.real(P[..., 0, 0] * P[..., 1, 1] - P[..., 0, 1] * P[..., 1, 0]) \
        + mu * np.real(P[..., 0, 0] + P[..., 1, 1]) + mu * mu
    w = (a + mu[..., None] * rhs) / det[..., None]
    rA, rB = _dot(w, gA_), _dot(w, gB_)
    nA, nB = 2.0 / _n2(gA_) * np.abs(rA) ** 2, 2.0 / _n2(gB_) * np.abs(rB) ** 2
    df = 0.5 * (nA + nB) / np.maximum(_quad(w, R_), 1e-300)
    bal = np.abs(10 * np.log10(np.maximum(nA, 1e-300) / np.maximum(nB, 1e-300)))
    return w, rA, rB, df, feas & (bal <= p["bal_db"])


def _mic_a(gA, gB):
    """Mic A alone, scaled to response 1 for partner A. Returns (w, psi = angle of B's response)."""
    w = np.zeros(gA.shape, complex)
    w[..., 0] = np.conj(1 / gA[..., 0])
    return w, np.angle(gB[..., 0] / gA[..., 0])


def _wrap(x):
    return np.angle(np.exp(1j * x))


def _pick(ok, df, act, prev):
    """Index of the best candidate per row: largest DF among ok; ties (within 1e-9) go to the actual
    phase nearest prev, then to the positive side (exact ties occur for mirror-symmetric partners).
    Returns (index, found)."""
    score = np.where(ok, df, -np.inf)
    top = score.max(axis=-1, keepdims=True)
    delta = _wrap(act - prev[:, None])
    key = np.where(ok & (score >= top * (1 - 1e-9)), np.abs(delta) - 1e-6 * delta, np.inf)
    return np.argmin(key, axis=-1), np.isfinite(top[:, 0])


def ab_beam(R, gA, gB, floor_db, opposite, k_top, p=PARAMS):
    """Two-talker beam per block with the M2 phase rule on centre-referenced steering (DESIGN M2, M3).
    R (nb, F, 2, 2) unit-mean-diagonal noise covariance; gA, gB (nb, F, 2); floor_db (F,); opposite (nb,)
    bool (tau_A > 0 > tau_B); k_top (nb,) bin of c / 2 d_hat. Dipole band (opposite partners): psi = 180 deg
    on the contiguous bins ending at k_top where the dipole meets the floor (balance +-1 dB, phase within
    +-dipole_deg). Elsewhere a continuation search bin by bin, away from the band: the candidate with the
    largest DF that meets the floor and the balance within +-psi_win of the neighbour (nominal psi and
    actual response phase), else the full circle, else Mic A alone. Partners not on opposite sides (or no
    dipole at k_top): full-circle search at anchor_hz, continued both ways. Weights are scaled so that
    sqrt(|r_A| |r_B|) = 1 (WNG and DF unchanged). Returns (w (nb, F, 2), psi (nb, F) rad = angle(r_B / r_A),
    mode (nb, F): 0 dipole, 1 window, 2 full circle, 3 Mic A)."""
    nb, F = gA.shape[:2]
    fl = 10 ** (np.asarray(floor_db) / 10)
    W = np.zeros((nb, F, 2), complex)
    psi = np.zeros((nb, F))
    mode = np.zeros((nb, F), int)
    wD, rAD, rBD, _, okD = _lcmv_cands(R, gA, gB, np.broadcast_to(fl, (nb, F)), np.full((nb, F, 1), np.pi), p)
    wD, okD = wD[..., 0, :], okD[..., 0]
    okD &= np.abs(_wrap(np.angle(rBD[..., 0] / rAD[..., 0]) - np.pi)) <= np.radians(p["dipole_deg"])
    full = np.radians(np.arange(0.0, 360.0, p["psi_step"]))
    win = np.radians(np.arange(-p["psi_win"], p["psi_win"] + 1e-9, p["psi_step"]))

    def search(b, k, prev, window=True):
        """Bin k, blocks b: window around prev, else the full circle, else Mic A -> (w, psi, mode)."""
        out_w, out_p, out_m = np.zeros((len(b), 2), complex), np.zeros(len(b)), np.full(len(b), 3)
        for m in ((1, 2) if window else (2,)):
            idx = np.nonzero(out_m == 3)[0]
            if not len(idx):
                break
            cands = prev[idx, None] + win[None, :] if m == 1 else np.broadcast_to(full, (len(idx), len(full)))
            w, rA, rB, df, ok = _lcmv_cands(R[b[idx], k], gA[b[idx], k], gB[b[idx], k], np.full(len(idx), fl[k]),
                                            cands, p)
            act = np.angle(rB / rA)
            if m == 1:      # with a large mu the response phase can leave the nominal window
                ok &= np.abs(_wrap(act - prev[idx, None])) <= np.radians(p["psi_win"]) + 1e-9
            j, hit = _pick(ok, df, act, prev[idx])
            r = np.arange(len(idx))[hit]
            out_w[idx[hit]], out_p[idx[hit]], out_m[idx[hit]] = w[r, j[hit]], act[r, j[hit]], m
        miss = out_m == 3
        if miss.any():
            out_w[miss], out_p[miss] = _mic_a(gA[b[miss], k], gB[b[miss], k])
        return out_w, out_p, out_m

    use_dip = opposite & okD[np.arange(nb), k_top]
    start_dn, start_up, prev = np.empty(nb, int), np.empty(nb, int), np.full(nb, np.pi)
    for b in np.nonzero(use_dip)[0]:
        lo = k_top[b]
        while lo > 0 and okD[b, lo - 1]:
            lo -= 1
        W[b, lo:k_top[b] + 1], psi[b, lo:k_top[b] + 1] = wD[b, lo:k_top[b] + 1], np.pi
        start_dn[b], start_up[b] = lo - 1, k_top[b] + 1
    other = np.nonzero(~use_dip)[0]
    if len(other):
        ka = int(np.argmin(np.abs(cm.FREQS - p["anchor_hz"])))
        W[other, ka], psi[other, ka], mode[other, ka] = search(other, ka, prev[other], window=False)
        prev[other] = psi[other, ka]
        start_dn[other], start_up[other] = ka - 1, ka + 1
    for ks, starts, sel in ((range(int(start_dn.max()), -1, -1), start_dn, np.greater_equal),
                            (range(int(start_up.min()), F), start_up, np.less_equal)):
        cur = prev.copy()
        for k in ks:
            b = np.nonzero(sel(starts, k))[0]
            if len(b):
                W[b, k], psi[b, k], mode[b, k] = search(b, k, cur[b])
                cur[b] = cur[b] + _wrap(psi[b, k] - cur[b])
    s = np.sqrt(np.abs(_dot(W, gA)) * np.abs(_dot(W, gB)))
    return W / np.where(s > cm.EPS, s, 1.0)[..., None], psi, mode


def mic_a_weights(shape):
    """Weights selecting Mic A unchanged, shape + (2,)."""
    w = np.zeros(tuple(shape) + (2,), complex)
    w[..., 0] = 1.0
    return w


def bus_state_weights(wA, wB, wAB, gA, gB, p=PARAMS):
    """State weights (DESIGN M2 state machine): state A = w_A rotated to A's AB response phase, state B =
    w_B rotated to B's AB response phase (h_B* w_B in DESIGN terms). Per-bin phase-continuity check for the
    partner not targeted: if its response r in state A (B) differs in phase from its AB response by more
    than 30 deg while |r| >= 0.5 (relative to the AB response), or 90 deg below, use w_AB in that bin.
    Returns (WA, WB, replaced_A, replaced_B)."""
    rA, rB = _dot(wAB, gA), _dot(wAB, gB)
    WA = wA * np.exp(-1j * np.angle(rA))[..., None]
    WB = wB * np.exp(-1j * np.angle(rB))[..., None]
    out = []
    for Wk, g_other, r_other in ((WA, gB, rB), (WB, gA, rA)):
        r = _dot(Wk, g_other)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = r / r_other
        d = np.degrees(np.abs(np.angle(rel)))
        big = np.abs(rel) >= p["phase_mag"]
        bad = np.where(big, d > p["phase_deg"][0], d > p["phase_deg"][1]) | ~np.isfinite(rel)
        out.append((np.where(bad[..., None], wAB, Wk), bad))
    (WA, badA), (WB, badB) = out
    return WA, WB, badA, badB


def df_table(W, gamma_n, targets):
    """Directivity factor mean_k |w^H g_k|^2 (g_k scaled to ||g||^2 = 2) / (w^H Gamma w) against the noise
    coherence Gamma = [[1, Gamma_n], [Gamma_n, 1]] (for M4's beam correction); targets: list of (..., F, 2)."""
    G = noise_model(gamma_n, 0.0)
    num = np.mean([2.0 * np.abs(_dot(W, g)) ** 2 / _n2(g) for g in targets], axis=0)
    return num / np.maximum(_quad(W, G), cm.EPS)


# ------------------------------------------------------------------ covariance recursions
def recursive_cov(X, m, alpha, evals, backward=False, resets=None, chunk=512):
    """Weighted recursion Phi <- (1 - alpha m) Phi + alpha m X X^H over the frames of X (2, F, N) with
    weights m, (N,) or (F, N) in [0, 1]; frames with zero weight are skipped (memory counts weighted
    frames). Returns the states at the frames `evals` as (paa, pbb, pab), each (len(evals), F), and the
    weight sums since the last reset (len(evals), F). Forward: state after the frames <= e; backward: after
    the frames >= e. resets (forward only): frames before which the state is cleared."""
    N, F = X.shape[-1], X.shape[1]
    m = np.asarray(m, float)
    evals = np.asarray(evals, int)
    if backward:
        X, m, evals = X[..., ::-1], m[..., ::-1], N - 1 - evals
        resets = None
    act = np.nonzero(m.reshape(-1, N).max(axis=0) > 0)[0]
    events = sorted([(e + 0.5, 0, i) for i, e in enumerate(evals)]          # at equal times: snapshot, then
                    + [(r - 0.5, 1, -1) for r in ([] if resets is None else resets)])   # reset
    out = [np.zeros((len(evals), F)), np.zeros((len(evals), F)), np.zeros((len(evals), F), complex)]
    cnt = np.zeros((len(evals), F))
    state = [np.zeros(F), np.zeros(F), np.zeros(F, complex)]
    n_sum = np.zeros(F)
    i0 = 0
    for t, kind, i in events:
        i_end = int(np.searchsorted(act, t))
        while i0 < i_end:
            fr = act[i0:min(i_end, i0 + chunk)]
            mm = m[:, fr] if m.ndim == 2 else np.broadcast_to(m[fr], (F, len(fr)))
            la = np.log1p(-alpha * mm)
            later = np.cumsum(la[:, ::-1], axis=1)[:, ::-1] - la       # sum of log(1 - a m) over later frames
            c = alpha * mm * np.exp(later)
            dec = np.exp(la.sum(axis=1))
            state = [dec * s + np.sum(c * r, axis=1) for s, r in zip(state, cm.spectra(X[:, :, fr]))]
            n_sum = n_sum + mm.sum(axis=1)
            i0 += len(fr)
        if kind == 1:
            state = [np.zeros(F), np.zeros(F), np.zeros(F, complex)]
            n_sum = np.zeros(F)
        else:
            for j in range(3):
                out[j][i] = state[j]
            cnt[i] = n_sum
    return out, cnt


def _held(c, n_hold, n_look):
    """True where c holds on every frame of [l - n_hold + 1, l + n_look] (False before the start; the last
    value is extended past the end)."""
    c = np.asarray(c, bool)
    pad = np.concatenate([np.zeros(n_hold - 1, bool), c, np.full(n_look, c[-1] if len(c) else False)])
    cs = np.concatenate([[0], np.cumsum(pad)])
    n = n_hold + n_look
    return (cs[n:] - cs[:-n]) == n


def bus_states(p_A, p_B, sep, p=PARAMS, online=False):
    """Per-frame state (A_, AB_, B_, FB_): A when p_A > 0.75 and p_B < 0.2 for 100 ms (offline also for the
    next lookahead_s), B likewise, AB on double talk or doubt, the previous state when nobody talks
    (both posteriors < 0.2), FB where sep is False (inseparable partners)."""
    n_hold = max(1, int(np.ceil(p["hold_s"] / cm.HOP_S - 1e-9)))        # at least hold_s (13 frames)
    n_look = 0 if online else int(np.ceil(p["lookahead_s"] / cm.HOP_S - 1e-9))
    cA = (p_A > p["p_enter"]) & (p_B < p["p_other"])
    cB = (p_B > p["p_enter"]) & (p_A < p["p_other"])
    quiet = (p_A < p["p_other"]) & (p_B < p["p_other"])
    st = np.where(_held(cA, n_hold, n_look), A_, np.where(_held(cB, n_hold, n_look), B_,
                                                            np.where(quiet, -1, AB_)))
    idx = np.where(st >= 0, np.arange(len(st)), -1)
    idx = np.maximum.accumulate(idx)
    st = np.where(idx >= 0, st[np.maximum(idx, 0)], AB_)
    return np.where(sep, st, FB_)


def cluster_states(cluster, n_dwell):
    """Meeting: per-frame target = active S1 cluster, held through -1 frames, switched only after the current
    target has been on for n_dwell frames; -1 before the first cluster."""
    st = np.full(len(cluster), -1)
    cur, since = -1, 0
    for l, c in enumerate(cluster):
        since += 1
        if c >= 0 and c != cur and (cur < 0 or since >= n_dwell):
            cur, since = int(c), 0
        st[l] = cur
    return st


# ------------------------------------------------------------------ result container and application
@dataclass
class Beam:
    """M3 result. Y = w^H X (257, N) with w the crossfaded weights of entry[l] (moving average over
    n_x frames of table[entry[l']], i.e. linear 50 ms ramps). apply() re-applies the same time-varying
    weights to any (2, 257, N) STFT (master/slave evaluation, DESIGN 4.2)."""
    Y: np.ndarray                 # (257, N) linear beam output (centre-referenced)
    table: np.ndarray             # (E, 257, 2) weight sets
    entry: np.ndarray             # (N,) index into table per frame (before crossfading)
    n_x: int                      # crossfade length (frames)
    state: np.ndarray             # (N,) bus: A_/AB_/B_/FB_; meeting: cluster id (-1 = fallback)
    block: np.ndarray             # (N,) update block of each frame
    scenario: str
    online: bool
    G: dict = field(default_factory=dict)        # target -> (n_blocks, 257) RTF (Mic B / Mic A), after checks
    g: dict = field(default_factory=dict)        # target -> (n_blocks, 257, 2) centre-referenced steering
    df: np.ndarray = None         # (E, 257) directivity factor of each table entry vs Gamma_n (M4 correction)
    diag: dict = field(default_factory=dict)

    def weights(self, l0, l1):
        """Effective (crossfaded) weights for frames l0..l1-1 -> (l1 - l0, 257, 2)."""
        n = self.n_x
        idx = self.entry[np.clip(np.arange(l0 - n + 1, l1), 0, len(self.entry) - 1)]
        cs = np.cumsum(np.concatenate([np.zeros((1,) + self.table.shape[1:], complex), self.table[idx]]), axis=0)
        return (cs[n:] - cs[:-n]) / n

    def apply(self, X, chunk=2048):
        """Y = w^H X for any 2-channel STFT X (2, 257, N) with the same frames."""
        N = X.shape[-1]
        Y = np.empty(X.shape[1:], complex)
        for l0 in range(0, N, chunk):
            l1 = min(N, l0 + chunk)
            W = self.weights(l0, l1)
            Y[:, l0:l1] = (np.conj(W[..., 0]) * X[0, :, l0:l1].T + np.conj(W[..., 1]) * X[1, :, l0:l1].T).T
        return Y

    def steered(self, X, state):
        """Bus: w^H X with one state's weights throughout (A_, AB_, B_ or FB_), e.g. the beam steered to
        partner A for M9 self-enrollment or M6's two-channel reference; same blocks and block crossfades."""
        if self.scenario != "bus":
            raise ValueError("steered() needs the bus states")
        return replace(self, entry=self.block * 4 + state).apply(X)

    def df_frames(self, l0=0, l1=None):
        """Per-frame DF of the active entry (no crossfade blending), (257, l1 - l0)."""
        l1 = len(self.entry) if l1 is None else l1
        return self.df[self.entry[l0:l1]].T


# ------------------------------------------------------------------ labels, masks, per-block context
def _causal_pct(x, ok, pct, step, win):
    """Per-frame percentile of x over the valid past frames (window win frames), refreshed every step
    frames; +inf until 20 valid past frames exist."""
    N = len(x)
    out = np.full(N, np.inf)
    for a in range(0, N, step):
        lo = max(0, a - win)
        v = x[lo:a][ok[lo:a]]
        if len(v) >= 20:
            out[a:a + step] = np.percentile(v, pct)
    return out


def cdr_cue(X, gamma_n, tau, lam=cm.LAM_76MS):
    """CDR-like directness D = CDR / (1 + CDR) per bin and frame (F, N) for the TDOA tau (scalar or per
    frame; NaN gives 0): DOA-dependent CDR (common.cdr_doa) on the lam-smoothed coherence; bins with
    |Gamma_s - Gamma_n| < 0.2 take the frame median over the conditioned bins in 1-4 kHz (DESIGN M4)."""
    f = cm.FREQS[:, None]
    N = X.shape[-1]
    out = np.empty((cm.N_BINS, N))
    band = ((cm.FREQS >= 1000.0) & (cm.FREQS <= 4000.0))[:, None]
    tau = np.broadcast_to(np.asarray(tau, float), (N,))
    gn = np.asarray(gamma_n, float)[:, None]
    for l0, l1, sm in cm.smoothed_spectra(X, slice(None), (lam,)):
        t = tau[l0:l1]
        Gs = cm.plane_coherence(f, np.nan_to_num(t)[None, :])
        cdr = cm.cdr_doa(cm.coherence(*sm[lam]), gn, Gs)
        d = cdr / (1.0 + cdr)
        cond = np.abs(Gs - gn) >= 0.2
        with np.errstate(all="ignore"):
            med = np.nan_to_num(np.nanmedian(np.where(cond & band, d, np.nan), axis=0))
        out[:, l0:l1] = np.where(cond, d, med[None, :]) * np.isfinite(t)[None, :]
    return out


def _cal_at(fe, starts):
    """Calib in effect at each block start (offline: fe.calib; online: the S0 schedule)."""
    if fe.schedule is None:
        return [fe.calib] * len(starts)
    s0 = np.array([s for s, _ in fe.schedule])
    return [fe.schedule[int(np.searchsorted(s0, s, side="right") - 1)][1] for s in starts]


def _sigma_s(fe, X, valid, noise_frames, cals, starts, p, online):
    """sigma_s per block: offline from the whole file; online per S0 schedule entry from the frames before
    the first block that uses it. Noise power from per-block sums (one chunked pass, no STFT copies)."""
    N, nb = X.shape[-1], len(starts)
    blk = np.repeat(np.arange(nb), np.diff(np.r_[starts, N]))
    sums = np.zeros((2, nb, cm.N_BINS))                  # power over noise frames / over valid frames
    cnts = np.zeros((2, nb))
    for l0 in range(0, N, cm.CHUNK):
        l1 = min(N, l0 + cm.CHUNK)
        P = 0.5 * (np.abs(X[0, :, l0:l1]) ** 2 + np.abs(X[1, :, l0:l1]) ** 2)
        for i, sel in enumerate((noise_frames[l0:l1] & valid[l0:l1], valid[l0:l1])):
            np.add.at(sums[i], blk[l0:l1][sel], P[:, sel].T)
            np.add.at(cnts[i], blk[l0:l1][sel], 1.0)
    lev = fe.track.level_db
    out = np.full((nb, cm.N_BINS), p["sigma_min"])
    ids = [id(c) for c in cals]
    for key in dict.fromkeys(ids):
        bi = [i for i, k in enumerate(ids) if k == key]
        upto = bi[0] if online else nb                   # blocks [0, upto) are usable
        ok = valid.copy()
        ok[starts[upto] if upto < nb else N:] = False
        if ok.sum() < 100:
            continue
        i = 0 if cnts[0, :upto].sum() >= p["min_frames"] else 1
        p_n = sums[i, :upto].sum(axis=0) / cnts[i, :upto].sum()
        q = np.nonzero(ok & (lev <= np.percentile(lev[ok], p["quiet_pct"])))[0]
        out[bi] = self_noise_ratio(X, q, p_n, cals[bi[0]], p["sigma_min"])
    return out


def _block_cov(X, m, alpha, starts, ends, online, resets=None):
    """Covariance (nb, F, 2, 2) and weight count (nb, F) per block. Online: the recursion state after the
    frames before the block (count since the last reset); offline: forward state at the block centre plus
    the backward state after it (two-sided window; count = all weights of the file)."""
    if online:
        (a, b, c), n = recursive_cov(X, m, alpha, starts - 1, resets=resets)
        return _herm(a, b, c), n
    ev = (starts + ends - 1) // 2
    (a, b, c), _ = recursive_cov(X, m, alpha, ev)
    (a2, b2, c2), _ = recursive_cov(X, m, alpha, ev + 1, backward=True)
    tot = np.asarray(m, float).reshape(-1, X.shape[-1]).sum(axis=-1)
    return _herm(a + a2, b + b2, c + c2), np.broadcast_to(tot, (len(starts), cm.N_BINS)).copy()


def checked_rtf(Phi_x, n, Rw, tau, p):
    """RTF (Mic B over Mic A) per block for one target: covariance whitening where at least min_frames
    weighted frames exist and the plausibility check passes, the model exp(-j 2 pi f tau) elsewhere.
    Returns (G (nb, F), fraction kept in the check band over the estimated blocks)."""
    f = cm.FREQS
    load = 1e-12 * np.real(Phi_x[..., 0, 0] + Phi_x[..., 1, 1]) + 1e-30
    G, ratio = cw_rtf(Phi_x + load[..., None, None] * _I2, Rw)
    est = n >= p["min_frames"]
    keep = est & plausible(G, f, tau, p["check_band"], p["check_deg"], p["check_db"], p["guard_db"])
    if p["min_snr_db"] is not None:                 # lambda_1 / lambda_2 - 1: whitened SNR of the labelled frames
        keep &= ratio - 1.0 >= 10 ** (p["min_snr_db"] / 10)
    band = (f >= p["check_band"][0]) & (f <= p["check_band"][1])
    frac = float(keep[:, band].sum() / max(est[:, band].sum(), 1))
    return np.where(keep, G, model_rtf(f, tau)), frac


def _resets(tau_t, tol=tk.POST_WIN):
    """Online partner TDOA track -> frames where it first appears or moves by more than tol (reset)."""
    out, ref = [], np.nan
    for l in np.nonzero(np.isfinite(tau_t))[0]:
        if not abs(tau_t[l] - ref) <= tol:
            out.append(int(l))
            ref = tau_t[l]
    return out


# ------------------------------------------------------------------ main entry
def run(fe, info, online=None, masks=None, **overrides):
    """M3 on a front end (tracker.front_end output); online defaults to whether fe has an S0 schedule.
    masks (bus, labels='mask'): optional external masks {'A', 'B', 'noise'}: (F, N) each (M7/M8), used
    instead of the posterior x CDR masks. overrides: keys of PARAMS. Returns a Beam."""
    scenario = info.get("scenario", "bus")
    online = fe.schedule is not None if online is None else online
    p = params(scenario, **overrides)
    if scenario == "bus" and fe.track.p_A is not None:
        return _run_bus(fe, p, online, masks)
    return _run_meeting(fe, p, online)


def _setup(fe, p, online):
    """Shared per-file context: blocks, calibrations, valid and directness-high frames, constants."""
    N = fe.X.shape[-1]
    step = max(1, int(round(p["update_s"] / cm.HOP_S)))
    starts = np.arange(0, N, step)
    ends = np.minimum(starts + step, N)
    valid = cm.valid_frames(N)
    direct = fe.track.directness
    if online:
        thr = _causal_pct(direct, valid, p["direct_pct"], step, int(round(tk.ONLINE_WIN_S / cm.HOP_S)))
    else:                                     # input shorter than one frame (T < 512): no labelled frames
        thr = np.percentile(direct[valid], p["direct_pct"]) if valid.any() else np.inf
    return dict(N=N, starts=starts, ends=ends, block=np.repeat(np.arange(len(starts)), ends - starts),
                cals=_cal_at(fe, starts), valid=valid, high=valid & (direct > thr),
                alpha=cm.HOP_S / p["memory_s"], fl=wng_floor_db(p["wng_db"]),
                n_x=max(1, int(round(p["xfade_s"] / cm.HOP_S))))


def _noise_cov(fe, ctx, m0, p, online):
    """Unit-mean-diagonal noise covariance per block, (nb, F, 2, 2), and sigma_s (nb, F)."""
    X, starts = fe.X, ctx["starts"]
    noise_frames = np.asarray(m0).reshape(-1, ctx["N"]).max(axis=0) > 0.5
    if p["sigma_s"] is None:
        sig = _sigma_s(fe, X, ctx["valid"], noise_frames, ctx["cals"], starts, p, online)
    else:
        sig = np.full((len(starts), cm.N_BINS), float(p["sigma_s"]))
    Rw = noise_model(np.stack([c.gamma_n for c in ctx["cals"]]), sig)
    if p["noise"] == "measured":
        Rn, n0 = _block_cov(X, m0, ctx["alpha"], starts, ctx["ends"], online)
        Rw = np.where((n0 >= p["min_frames"])[..., None, None], normalize_cov(Rn) + 1e-6 * _I2, Rw)
    return normalize_cov(Rw), sig


def _bus_weights(fe, ctx, p, masks, online):
    """Covariance weights (A, B, noise): posterior labels, posterior x CDR masks, or external masks."""
    trk, valid, N = fe.track, ctx["valid"], ctx["N"]
    if p["labels"] == "posterior":
        lab = lambda pk: (valid & ctx["high"] & (pk > p["p_label"]) & ~trk.double_talk).astype(float)
        return lab(trk.p_A), lab(trk.p_B), (valid & (trk.p_A + trk.p_B < p["p_noise"])).astype(float)
    if masks is not None:
        return tuple(np.clip(np.asarray(masks[k], float), 0, 1) * valid for k in ("A", "B", "noise"))
    use_cdr, use_post = "cdr" in p["mask_cue"], "post" in p["mask_cue"]
    gn = fe.calib.gamma_n
    DA = cdr_cue(fe.X, gn, trk.tau_A_t, p["lam_cdr"]) if use_cdr else np.ones((1, N))
    DB = cdr_cue(fe.X, gn, trk.tau_B_t, p["lam_cdr"]) if use_cdr else np.ones((1, N))
    pA, pB = (trk.p_A, trk.p_B) if use_post else (np.ones(N), np.ones(N))
    mA, mB = pA * DA * valid, pB * DB * valid
    m0 = (1 - np.maximum(trk.p_A, trk.p_B)) * (1 - np.maximum(DA, DB) if use_cdr else 1.0) * valid
    return mA, mB, np.broadcast_to(m0, mA.shape).copy()


def _run_bus(fe, p, online, masks):
    X, trk, ctx = fe.X, fe.track, _setup(fe, p, online)
    starts, ends, block, f = ctx["starts"], ctx["ends"], ctx["block"], cm.FREQS
    nb, fl = len(starts), ctx["fl"]
    at = starts if online else np.zeros(nb, int)
    tA, tB = np.asarray(trk.tau_A_t, float)[at], np.asarray(trk.tau_B_t, float)[at]
    d_hat = np.array([c.d_hat for c in ctx["cals"]])
    if online:
        why = [tk.inseparable_reasons(1 if a == b else 2, a, b, d) if np.isfinite(a) and np.isfinite(b)
               else ("no_tdoa",) for a, b, d in zip(tA, tB, d_hat)]
    else:
        why = [tuple(trk.inseparable_why) if np.isfinite(trk.tau_A) else ("no_tdoa",)] * nb
    lim = p["max_tdoa"] * d_hat / C                     # no plane wave at d_hat has a larger TDOA
    why = [w + (("beyond_endfire",) if np.isfinite(a) and max(abs(a), abs(b)) > m else ())
           for w, a, b, m in zip(why, tA, tB, lim)]
    if p["same_side"] == "fallback":                    # two-talker beam only for partners on opposite sides
        why = [w + (("same_side",) if not w and not a > 0 > b else ()) for w, a, b in zip(why, tA, tB)]
    sep_b = np.array([not w for w in why])
    broad = np.array([p["fallback"] == "broadside" or (p["fallback"] == "auto" and "both_broadside" in w)
                      for w in why])
    tA, tB = np.nan_to_num(tA), np.nan_to_num(tB)
    mA, mB, m0 = _bus_weights(fe, ctx, p, masks, online)
    Rw, sig = _noise_cov(fe, ctx, m0, p, online)
    diag, G, g, half, cov = {}, {}, {}, {}, {}
    for key, m, tau, tau_t in (("A", mA, tA, trk.tau_A_t), ("B", mB, tB, trk.tau_B_t)):
        resets = _resets(np.asarray(tau_t, float)) if online else None
        Phi, n = _block_cov(X, m, ctx["alpha"], starts, ends, online, resets)
        G[key], diag[f"rtf_kept_{key}"] = checked_rtf(Phi, n, Rw, tau, p)
        half[key], g[key] = sym_rtf(G[key], f, tau)
        cov[key] = (Phi, n)
        diag[f"weight_sum_{key}"] = float(np.asarray(m).reshape(-1, ctx["N"]).mean(axis=0).sum())
        diag[f"blocks_estimated_{key}"] = int((n.max(axis=1) >= p["min_frames"]).sum())
    gA, gB = g["A"], g["B"]
    a0 = np.broadcast_to(cm.steer(f, 0.0), gA.shape)
    WF = mic_a_weights((nb, cm.N_BINS))
    if broad.any():
        WF[broad] = mvdr(Rw[broad], a0[broad], fl, p)[0]
    WA, WB, wAB = WF.copy(), WF.copy(), WF.copy()           # inseparable blocks only use the fallback
    psi, mode = np.full((nb, cm.N_BINS), np.nan), np.full((nb, cm.N_BINS), -1)
    badA, badB = np.zeros((nb, cm.N_BINS), bool), np.zeros((nb, cm.N_BINS), bool)
    k_top = np.array([int(np.argmin(np.abs(f - C / (2 * d)))) for d in d_hat])
    opposite = (tA > 0) & (tB < 0)
    s = np.nonzero(sep_b)[0]
    if len(s):
        if p["beam"] == "souden":
            ws = []
            for k in ("A", "B"):
                w = souden(cov[k][0][s], Rw[s], g[k][s], half[k][s], fl, p)
                low = cov[k][1][s] < p["min_frames"]
                if low.any():
                    w[low] = mvdr(Rw[s], g[k][s], fl, p)[0][low]
                ws.append(w)
            wA, wB = ws
        else:
            wA, wB = mvdr(Rw[s], gA[s], fl, p)[0], mvdr(Rw[s], gB[s], fl, p)[0]
        wAB[s], psi[s], mode[s] = ab_beam(Rw[s], gA[s], gB[s], fl, opposite[s], k_top[s], p)
        WA[s], WB[s], badA[s], badB[s] = bus_state_weights(wA, wB, wAB[s], gA[s], gB[s], p)
    if p["states"] == "ab":
        state = np.where(sep_b[block], AB_, FB_)
    else:
        state = bus_states(trk.p_A, trk.p_B, sep_b[block], p, online)
    gn = np.stack([c.gamma_n for c in ctx["cals"]])
    df = np.stack([df_table(WA, gn, [gA]), df_table(wAB, gn, [gA, gB]), df_table(WB, gn, [gB]),
                   df_table(WF, gn, [a0])], axis=1).reshape(nb * 4, cm.N_BINS)
    bm = Beam(Y=None, table=np.stack([WA, wAB, WB, WF], axis=1).reshape(nb * 4, cm.N_BINS, 2),
              entry=block * 4 + state, n_x=ctx["n_x"], state=state, block=block, scenario="bus",
              online=online, G=G, g=g, df=df)
    bm.Y = bm.apply(X)
    kd = (f >= 300.0) & (f <= 0.8 * C / (2 * np.median(d_hat)))
    diag.update(n_blocks=nb, separable_blocks=int(sep_b.sum()), broadside_fallback_blocks=int(broad[~sep_b].sum()),
                opposite_blocks=int(opposite.sum()),
                tau_A=tA, tau_B=tB, sigma_s_median=float(np.median(sig)), psi_deg=np.degrees(psi),
                state_frac={n: float(np.mean(state == i)) for i, n in enumerate(STATE_NAMES)},
                ab_mode_frac={n: float(np.mean(mode == i)) for i, n in enumerate(("dipole", "window", "full",
                                                                                   "mic_a"))},
                phase_check_replaced=(float(badA[:, kd].mean()), float(badB[:, kd].mean())),
                k_top=k_top, opposite=opposite,
                labels=p["labels"], noise=p["noise"], beam=p["beam"])
    bm.diag = diag
    return bm


def _cluster_tau(trk, m, alpha, at, online, cid):
    """TDOA of a meeting cluster at block starts `at`: offline the S1 cluster summary; online a running mean
    of tau over the cluster's labelled frames before each block (else the median of its earlier frames,
    else broadside)."""
    mine = trk.cluster == cid
    if not online:
        d = next((c for c in trk.clusters if c["id"] == cid), None)
        return np.full(len(at), d["tau"] if d else float(np.median(trk.tau[mine])))
    idx = np.nonzero(m > 0)[0]
    run_mean = cm.smooth(trk.tau[idx], 1.0 - alpha)[0] if len(idx) else np.zeros(0)
    out = np.zeros(len(at))
    for i, a in enumerate(at):
        j = np.searchsorted(idx, a) - 1
        if j >= 0:
            out[i] = run_mean[j]
        else:
            early = trk.tau[:a][mine[:a]]
            out[i] = float(np.median(early)) if len(early) else 0.0
    return out


def _run_meeting(fe, p, online):
    X, trk, ctx = fe.X, fe.track, _setup(fe, p, online)
    starts, ends, block, f, N = ctx["starts"], ctx["ends"], ctx["block"], cm.FREQS, ctx["N"]
    nb, fl = len(starts), ctx["fl"]
    steer = p["meeting"] == "cluster" and trk.cluster is not None
    cl = trk.cluster if steer else np.full(N, -1)
    st = cluster_states(cl, max(1, int(round(p["dwell_s"] / cm.HOP_S))))
    Rw, sig = _noise_cov(fe, ctx, (ctx["valid"] & (trk.vad < 0.3)).astype(float), p, online)
    gn = np.stack([c.gamma_n for c in ctx["cals"]])
    a0 = np.broadcast_to(cm.steer(f, 0.0), (nb, cm.N_BINS, 2))
    WF = mic_a_weights((nb, cm.N_BINS)) if p["meeting"] == "omni" else mvdr(Rw, a0, fl, p)[0]
    tables, dfs = [WF], [df_table(WF, gn, [a0])]
    entry = np.where(st < 0, block, -1)
    off, G, g, kept = nb, {}, {}, {}
    for c in sorted({int(c) for c in st if c >= 0}):
        used = np.unique(block[st == c])                      # weights only where the cluster is steered to
        m = (ctx["high"] & (cl == c)).astype(float)
        tau = _cluster_tau(trk, m, ctx["alpha"], starts[used], online, c)
        Phi, n = _block_cov(X, m, ctx["alpha"], starts[used], ends[used], online)
        G[c], kept[c] = checked_rtf(Phi, n, Rw[used], tau, p)
        g[c] = sym_rtf(G[c], f, tau)[1]
        w = mvdr(Rw[used], g[c], fl, p)[0]
        tables.append(w)
        dfs.append(df_table(w, gn[used], [g[c]]))
        sel = st == c
        entry[sel] = off + np.searchsorted(used, block[sel])
        off += len(used)
    bm = Beam(Y=None, table=np.concatenate(tables), entry=entry, n_x=ctx["n_x"], state=st, block=block,
              scenario="meeting", online=online, G=G, g=g, df=np.concatenate(dfs))
    bm.Y = bm.apply(X)
    bm.diag = dict(n_blocks=nb, n_clusters=len(G), rtf_kept=kept, sigma_s_median=float(np.median(sig)),
                   fallback_frac=float(np.mean(st < 0)))
    return bm


def enhance(x, fs, info, online=False, return_stages=False, **overrides):
    """Method body for the wrappers: S0 + S1 front end -> M3 -> ISTFT. Returns the linear beam output
    (centre-referenced, see the module docstring), or {'out', 'pre_agc'} (identical: no level control).
    A meeting with meeting='omni' (the default) returns S0's prefiltered Mic A without running S0 / S1."""
    scenario = info.get("scenario", "bus")
    if scenario == "meeting" and params(scenario, **overrides)["meeting"] == "omni":
        if fs != cm.FS:
            raise ValueError("enhance expects fs = 16000")
        y = cb.prefilter(np.asarray(x, dtype=np.float64), scenario)[0]
        return {"out": y, "pre_agc": y} if return_stages else y
    fe = tk.front_end(np.asarray(x, dtype=np.float64), fs, info, online=online)
    Y = run(fe, info, online=online, **overrides).Y
    del fe                                   # frees the input STFT before the (frame-materializing) ISTFT
    y = cm.istft(Y, np.shape(x)[1])
    return {"out": y, "pre_agc": y} if return_stages else y
