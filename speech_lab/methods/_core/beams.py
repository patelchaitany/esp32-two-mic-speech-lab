"""M1 delay-and-sum and M2 WNG-constrained superdirective / two-talker beams (DESIGN.md M1, M2, 2.1, 2.2).

Conventions (common.py, API.md): X is the S0-corrected STFT (2, 257, N), row 0 = Mic A; tau > 0 means Mic B
hears later; center-referenced steering a(f, tau) = [e^{+j pi f tau}, e^{-j pi f tau}]. A beam is y = w^H x per
bin, so a source at tau with w^H a = 1 comes out as its signal at the array centre (tau/2 after Mic A, at most
1.4 samples at 6 cm). WNG = |w^H a|^2 / w^H w and DF = |w^H a|^2 / w^H Gamma w, with Gamma = [[1, g], [g*, 1]]
and g = S0's measured noise coherence Calib.gamma_n ('measured', default) or rho * sinc/J0 at d_hat ('model').

Weight design (vectorized over bins, closed-form 2x2 algebra):
- das_weights (M1): w = a / 2.
- mvdr_weights (M2 single target): w = (Gamma + mu I)^-1 a / (a^H (Gamma + mu I)^-1 a) with the smallest mu that
  meets the WNG floor (bisection of log10 mu over [-9, 6], 40 steps). mu -> inf is delay-and-sum (+3 dB), so
  any floor <= 0 dB is met.
- two_talker (M2 AB beam): soft LCMV w = (Gamma + mu I + lam A A^H)^-1 lam A h*, A = [a_A, a_B],
  h = [1, e^{j psi}], lam = 1e3, mu by the same bisection, for psi on a 2.5 deg grid in every bin. A candidate
  qualifies only if it meets the floor for both partners (min_k |r_k|^2 / w^H w) and |r_A| / |r_B| is within
  +-1 dB; weights below the floor are never kept. Phase rule: one qualified candidate per bin, chosen by a
  Viterbi pass that maximizes the summed DF (dB) subject to bin-to-bin steps <= 30 deg of the ACTUAL response
  phase arg(r_B / r_A) (a step beyond 30 deg costs >= 1000 dB, so it is taken only where no path avoids it);
  Mic A alone where nothing qualifies. For partners symmetric about broadside this keeps h = [1, -1] on the
  dipole band and rotates smoothly away from it where the floor binds, as DESIGN M2 prescribes.
  Deviation: DESIGN's greedy search (fixed 180 deg band, then bin by bin within +-30 deg of the neighbour's
  grid psi) let the actual response jump by up to 179 deg between bins (mu > 0 moves arg(r_B / r_A) away from
  psi), could not keep B.1's 30 deg bound at the band edges, and lost up to 1.8 dB AI-DF for asymmetric pairs
  (60/100 deg); the Viterbi path is within 0.05 dB of the per-bin optimum in every tested geometry
  (tests/test_beams.py, test_design_rule). Weights are scaled to r_A = 1; h_B = r_B is partner B's AB response.
- Every design checks the WNG of what it returns and raises if a bin is below its floor.

Time-varying beams (Beam): a bank of weight sets plus per-frame coefficients (linear 50 ms crossfades,
crossfade()); Beam.apply(X) -> Y (257, N), Beam.weights(l0, l1) -> (257, n, 2), Beam.df(cal) -> 1 / W^H Gamma W
for the M4 beam correction. Pipelines on the S0/S1 front end (tracker.front_end):
- das_beam (M1): broadside average, or steered to the active partner (bus) / active talker (meeting).
- bus_beam (M2 bus): the AB beam throughout (default) or states A / AB / B from S1's posteriors with the per-bin
  phase-continuity substitution; the DESIGN 2.1 fallback when S1 flags the partners inseparable or their TDOAs
  fail the gate below.
- meeting_beam (M2 meeting): MVDR steered to the active talker on a 64-point TDOA grid, or three fixed beams
  (+axis, broadside, -axis) with best-beam selection.

Tuned on the v2 dev set (TTS; tune:beams). Deviations from DESIGN, each measured against M0 (Mic A after S0's
high-pass), paired over the 30 dev bus geometries / dev meeting rooms:
- Partner-TDOA gate (implausible_reasons): S1 peaks beyond 1.1 d_hat / c, or both on one side of broadside, fall
  back to Mic A. The former were histogram pile-ups at S1's +-1.2 d_hat / c grid edge (40 mm prior on 60 mm arrays)
  or a missed skew, and the AB beam built on them cut the real partners by 10-19 dB (worst-partner dESTOI to
  -0.26); every same-side S1 pair on dev was wrong (a partner plus a passenger or noise source).
- Bus states default 'ab': S1's posteriors and TDOAs are wrong often enough on v2 that single-target states put
  the active partner into a rear lobe (DESIGN: "prefer AB when uncertain"). The DESIGN state machine stays as
  states='s1', with stricter entry (p > 0.9 held 300 ms instead of 0.75 / 100 ms).
- Bus WNG floor -10 dB below 500 Hz (DESIGN -15 dB): never worse on dev, and it amplifies uncorrelated noise
  (handling, wind) by 5 dB less.
- Steered modes (M1, M2 meeting) return to a neutral beam (Mic A alone) after 0.5 s without talker evidence and
  start there; DESIGN's hold kept the beam on the previous talker, so a soft talker S1 never localizes was cut by
  up to 11 dB.
- Steering evidence near the coherent-noise TDOA is ignored (noise_tdoa): in quiet dev rooms the projector fan
  dominates the idle frames, and S1's TDOA for a weak talker reads the fan's, so the beam turned to the fan while
  the talker spoke (4.5 % of talkers lost more than 0.03 ESTOI). Talkers within 40 us of it get the neutral beam.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lab.common import C
from . import calib as cb
from . import common as cm
from . import tracker as tk

BUS_FLOOR_DB = (-10.0, -10.0, -6.0)    # WNG floors below 500 Hz, 500 Hz - 2 kHz, above 2 kHz (DESIGN: -15/-10/-6)
BUS_FLOOR_EDGES_HZ = (500.0, 2000.0)
MEETING_FLOOR_DB = -3.0
WNG_TOL_DB = 1e-6                      # numerical tolerance of the floor check

MU_LOG10 = (-9.0, 6.0)                 # bisection interval of log10 mu
MU_STEPS = 40
LAM_C = 1e3                            # soft-constraint weight of the two-talker LCMV
PSI_STEP = 2.5                         # deg, phase grid of the two-talker search
PSI_WINDOW = 30.0                      # deg, max bin-to-bin step of the AB response phase
STEP_PEN = 100.0                       # dB per degree of a step beyond PSI_WINDOW (+10 deg fixed: unavoidable only)
BALANCE_DB = 1.0                       # |r_A| / |r_B| within +-1 dB
CONT_DEG = (30.0, 90.0)                # phase-continuity limits for responses >= / < CONT_MAG
CONT_MAG = 0.5                         # -6 dB

RAMP_S = 0.05                          # weight crossfade
STATE_HOLD_S = 0.3                     # bus 's1' states: entry condition held this long (DESIGN: 100 ms)
P_ENTER = 0.9                          # ... p_k above this and the other below P_OTHER (DESIGN: 0.75)
P_OTHER = 0.2
ACTIVE_P = 0.25                        # M1 bus: a partner is active above this posterior
DAS_HOLD_S = 0.3                       # M1 steering hold
STEER_HOLD_S = 0.4                     # M2 meeting steering hold (DESIGN: 300-500 ms)
STEER_WIN_S = 0.3                      # active-talker TDOA: weighted median over this window
STEER_MIN_W = 1.0                      # ... of at least this much weight (sum of vad * directness)
STEER_HYST = 10e-6                     # re-steer only for TDOA changes above this (grid step 4.4 us at 4 cm)
STEER_RELEASE_S = 0.5                  # back to the neutral beam after this long without talker evidence
NEUTRAL = "omni"                       # neutral beam of the steered modes: Mic A alone
SPEECH_P = 0.5
NOISE_P = 0.2                          # idle frames for the coherent-noise TDOA: vad below this
NOISE_EXCL = 40e-6                     # steering evidence this close to the coherent-noise TDOA is ignored ...
NOISE_MIN = 0.3                        # ... when +-20 us around it hold this fraction of the idle directness mass
NOISE_BLOCK_S = 1.0                    # causal coherent-noise TDOA: update interval and ...
NOISE_FORGET_S = 20.0                  # ... forgetting time constant
N_GRID = 64
GRID_SPAN = 1.2                        # TDOA grid +-1.2 d_hat / c (as S1)
SELECT_HOLD_S = 0.5                    # fixed-beam selection
SELECT_BAND = (300.0, 3000.0)
SELECT_LAM = cm.LAM_160MS              # smoothing of the selection power
TAU_Q = 1e-6                           # TDOA quantization of cached designs (S1's histogram step)
MAX_TDOA = 1.1                         # partner TDOAs beyond MAX_TDOA * d_hat / c are implausible (not in DESIGN)
STATES = ("A", "AB", "B")

AI_FC = np.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
                  7900.0])
AI_W = np.array([.0083, .0095, .0150, .0289, .0440, .0578, .0653, .0711, .0818, .0844, .0882, .0898, .0868, .0844,
                 .0771, .0527, .0364, .0185])


# ------------------------------------------------------------------ floors, noise model, beam metrics
def wng_floor(scenario, f=cm.FREQS, meeting_db=MEETING_FLOOR_DB):
    """Per-bin WNG floor (dB): bus -15 dB below 500 Hz, -10 dB up to 2 kHz, -6 dB above; meeting -3 dB."""
    f = np.asarray(f, dtype=float)
    if scenario == "meeting":
        return np.full(f.shape, float(meeting_db))
    lo, hi = BUS_FLOOR_EDGES_HZ
    return np.where(f < lo, BUS_FLOOR_DB[0], np.where(f <= hi, BUS_FLOOR_DB[1], BUS_FLOOR_DB[2]))


def noise_coherence(cal, noise="measured"):
    """Gamma_AB(f) (257,) for the designs: S0's measured gamma_n, or the model rho * sinc/J0 at d_hat."""
    if noise == "measured":
        return np.asarray(cal.gamma_n, dtype=float)
    if noise == "model":
        return cal.model_coherence()
    raise ValueError(f"unknown noise model {noise!r}")


def _abs2(z):
    return z.real ** 2 + z.imag ** 2


def response(w, a):
    """w^H a over the last axis."""
    return np.sum(np.conj(w) * a, axis=-1)


def white_gain(w):
    """w^H w over the last axis."""
    return np.sum(_abs2(w), axis=-1)


def noise_power(w, g):
    """w^H Gamma w with Gamma = [[1, g], [g*, 1]]."""
    return white_gain(w) + 2.0 * np.real(np.conj(w[..., 0]) * g * w[..., 1])


def wng_db(w, a):
    """White-noise gain |w^H a|^2 / w^H w in dB."""
    return 10 * np.log10(_abs2(response(w, a)) / white_gain(w))


def df_db(w, a, g):
    """Directivity factor |w^H a|^2 / w^H Gamma w in dB."""
    return 10 * np.log10(_abs2(response(w, a)) / noise_power(w, g))


def ai_index(db, f=cm.FREQS):
    """Speech-weighted average of a per-bin dB quantity (ANSI S3.5 band importance at the nearest bins,
    DESIGN 1.2 AI-DI)."""
    k = np.argmin(np.abs(np.asarray(f)[:, None] - AI_FC[None, :]), axis=0)
    return float(np.sum(AI_W * np.asarray(db)[k]))


def _check_floor(wng, floor_db, what):
    """WNG must never be violated silently (DESIGN M2, B.1)."""
    margin = np.asarray(wng) - np.asarray(floor_db)
    if not np.all(margin >= -WNG_TOL_DB):
        raise FloatingPointError(f"{what}: WNG below the floor in {int(np.sum(~(margin >= -WNG_TOL_DB)))} bins "
                                 f"(worst margin {np.nanmin(margin):.3f} dB)")


# ------------------------------------------------------------------ weight design
def das_weights(f, tau):
    """Delay-and-sum w = a(f, tau) / 2 (M1): (F, 2), unit response toward tau."""
    return cm.steer(f, tau) / 2.0


def _solve(p, s, q, b):
    """Solve [[p, q], [q*, s]] w = b elementwise (p, s real; b (..., 2))."""
    det = p * s - _abs2(q)
    return np.stack([(s * b[..., 0] - q * b[..., 1]) / det, (p * b[..., 1] - np.conj(q) * b[..., 0]) / det], axis=-1)


def _bisect(solve, gain, floor, args):
    """Smallest mu = 10^x, x in MU_LOG10 (MU_STEPS bisection steps), with gain(w, *args) >= floor elementwise,
    w = solve(mu, *args). args: arrays shaped floor.shape (+ a trailing axis of 2); only the elements that miss
    the floor at the smallest mu are bisected. Returns (w, ok, mu); ok is False where w misses the floor."""
    mu = np.full(floor.shape, 10.0 ** MU_LOG10[0])
    w = solve(mu, *args)
    need = ~(gain(w, *args) >= floor)
    if need.any():
        sub = [x[need] for x in args]
        fl = floor[need]
        lo = np.full(fl.shape, MU_LOG10[0])
        hi = np.full(fl.shape, MU_LOG10[1])
        for _ in range(MU_STEPS):
            mid = 0.5 * (lo + hi)
            good = gain(solve(10.0 ** mid, *sub), *sub) >= fl
            hi = np.where(good, mid, hi)
            lo = np.where(good, lo, mid)
        mu[need] = 10.0 ** hi
        w[need] = solve(mu[need], *sub)
    return w, gain(w, *args) >= floor, mu


def _mvdr_solve(mu, a, g):
    v = _solve(1.0 + mu, 1.0 + mu, g, a)
    return v / response(a, v)[..., None]


def _mvdr_gain(w, a, g):
    return _abs2(response(w, a)) / white_gain(w)


def mvdr_weights(a, g, floor_db):
    """WNG-constrained superdirective MVDR (DESIGN M2): w = (Gamma + mu I)^-1 a / (a^H (Gamma + mu I)^-1 a) with
    the smallest mu meeting the floor. a (..., 2) steering, g Gamma_AB and floor_db (dB) broadcast to a.shape[:-1].
    Returns (w (..., 2), mu)."""
    a = np.asarray(a, dtype=complex)
    shape = a.shape[:-1]
    g = np.broadcast_to(np.asarray(g), shape)
    floor_db = np.broadcast_to(np.asarray(floor_db, dtype=float), shape)
    w, _, mu = _bisect(_mvdr_solve, _mvdr_gain, 10.0 ** (floor_db / 10), (a, g))
    _check_floor(wng_db(w, a), floor_db, "MVDR")
    return w, mu


def lcmv2_weights(a_A, a_B, h_B, g, floor_db, lam=LAM_C):
    """Two-talker soft LCMV (DESIGN M2): w = (Gamma + mu I + lam A A^H)^-1 lam A h*, A = [a_A, a_B], h = [1, h_B],
    with the smallest mu meeting the WNG floor for both partners (min_k |r_k|^2 / w^H w). All arguments
    broadcast over the leading axes (steering vectors carry a trailing axis of 2).
    Returns (w, ok); ok is False where no mu meets the floor (those weights must not be used)."""
    shape = np.broadcast_shapes(np.shape(a_A)[:-1], np.shape(a_B)[:-1], np.shape(h_B), np.shape(g),
                                np.shape(floor_db))
    aA = np.broadcast_to(a_A, shape + (2,))
    aB = np.broadcast_to(a_B, shape + (2,))
    p = 1.0 + lam * (_abs2(aA[..., 0]) + _abs2(aB[..., 0]))
    s = 1.0 + lam * (_abs2(aA[..., 1]) + _abs2(aB[..., 1]))
    q = g + lam * (aA[..., 0] * np.conj(aA[..., 1]) + aB[..., 0] * np.conj(aB[..., 1]))
    b = lam * (aA + aB * np.conj(h_B)[..., None])
    floor = np.broadcast_to(10.0 ** (np.asarray(floor_db, dtype=float) / 10), shape)
    w, ok, _ = _bisect(_lcmv_solve, _lcmv_gain, floor, (p, s, q, b, aA, aB))
    return w, ok


def _lcmv_solve(mu, p, s, q, b, aA, aB):
    return _solve(p + mu, s + mu, q, b)


def _lcmv_gain(w, p, s, q, b, aA, aB):
    return np.minimum(_abs2(response(w, aA)), _abs2(response(w, aB))) / white_gain(w)


@dataclass
class TwoTalker:
    """AB beam of one partner pair: one weight vector per bin (DESIGN M2 two-talker beam and phase rule)."""
    w: np.ndarray          # (F, 2) weights scaled to r_A = w^H a_A = 1
    h_B: np.ndarray        # (F,) partner B's response r_B = w^H a_B (the state machine's h_B)
    kind: np.ndarray       # (F,) 0 dipole (h = [1, -1]), 1 other qualified candidate, 2 Mic A alone
    band: tuple            # (f_lo, f_hi) Hz: DESIGN dipole band (opposite sides, dipole qualifies), or None
    wng_db: np.ndarray     # (F,) min over the partners of |r_k|^2 / w^H w
    df_db: np.ndarray      # (F,) 0.5 (|r_A|^2 + |r_B|^2) / w^H Gamma w

    @property
    def psi_deg(self):
        """(F,) phase of h_B, unwrapped along frequency (deg)."""
        return np.degrees(np.unwrap(np.angle(self.h_B)))


def _circ_deg(x, y):
    """Circular distance |x - y| in degrees."""
    return np.abs((x - y + 180.0) % 360.0 - 180.0)


def _phase_path(score, phase, fallback_phase):
    """Viterbi choice of one candidate per bin maximizing sum(score) minus a penalty for every bin-to-bin phase
    step beyond PSI_WINDOW (STEP_PEN * (10 + excess in deg), so such a step is taken only where no path avoids
    it). score (F, P) in dB (-inf = not qualified), phase (F, P) deg; an extra node P (Mic A alone, phase
    fallback_phase, 0 dB) is allowed only in bins where nothing qualifies. Returns (F,) chosen node per bin."""
    F, P = score.shape
    S = np.concatenate([score, np.where(np.isfinite(score).any(axis=1), -np.inf, 0.0)[:, None]], axis=1)
    PH = np.concatenate([phase, fallback_phase[:, None]], axis=1)
    back = np.zeros((F, P + 1), dtype=int)
    V = S[0]
    rows = np.arange(P + 1)
    for k in range(1, F):
        excess = _circ_deg(PH[k][:, None], PH[k - 1][None, :]) - PSI_WINDOW
        tot = V[None, :] - np.where(excess > 1e-6, STEP_PEN * (10.0 + excess), 0.0)
        back[k] = np.argmax(tot, axis=1)
        V = S[k] + tot[rows, back[k]]
    path = np.empty(F, dtype=int)
    path[-1] = int(np.argmax(V))
    for k in range(F - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


def two_talker(f, tau_A, tau_B, g, floor_db, d_hat):
    """AB beam keeping partners A (tau_A) and B (tau_B) with responses 1 and h_B (DESIGN M2 phase rule; see the
    module docstring). f (F,) ascending; g (F,) Gamma_AB; floor_db (F,) or scalar; d_hat only sets the
    reported DESIGN dipole band."""
    if not (np.isfinite(tau_A) and np.isfinite(tau_B)) or tau_A == tau_B:
        raise ValueError("two_talker needs two distinct finite TDOAs")
    f = np.asarray(f, dtype=float)
    F = len(f)
    g = np.broadcast_to(np.asarray(g), f.shape)
    floor_db = np.broadcast_to(np.asarray(floor_db, dtype=float), f.shape)
    aA, aB = cm.steer(f, tau_A), cm.steer(f, tau_B)
    grid = np.arange(0.0, 360.0, PSI_STEP)
    with np.errstate(divide="ignore", invalid="ignore"):
        W, ok = lcmv2_weights(aA[:, None], aB[:, None], np.exp(1j * np.radians(grid))[None, :], g[:, None],
                              floor_db[:, None])
        rA, rB = response(W, aA[:, None]), response(W, aB[:, None])
        bal = np.abs(10 * np.log10(_abs2(rA) / _abs2(rB)))
        score = 10 * np.log10(0.5 * (_abs2(rA) + _abs2(rB)) / noise_power(W, g[:, None]))
        phase = np.degrees(np.angle(rB / rA))          # actual response phase of each candidate
        qual = ok & (bal <= BALANCE_DB) & np.isfinite(score) & np.isfinite(phase)
    mic_a_phase = np.degrees(np.pi * f * (tau_B - tau_A))
    path = _phase_path(np.where(qual, score, -np.inf), np.where(qual, phase, 0.0), mic_a_phase)
    i180 = int(round(180.0 / PSI_STEP))
    kind = np.where(path == len(grid), 2, np.where(path == i180, 0, 1))
    band = None
    f_top = min(C / (2.0 * d_hat), 1.0 / abs(tau_A - tau_B), f[-1])
    k = int(np.searchsorted(f, f_top, side="right")) - 1
    if tau_A * tau_B < 0 and qual[k, i180]:
        k_lo = k
        while k_lo > 0 and qual[k_lo - 1, i180]:
            k_lo -= 1
        band = (float(f[k_lo]), float(f[k]))
    w_mic_a = np.stack([np.exp(1j * np.pi * f * tau_A), np.zeros(F)], axis=-1)     # r_A = 1, r_B = e^{j mic_a_phase}
    w = np.where((kind < 2)[:, None], W[np.arange(F), np.minimum(path, len(grid) - 1)], w_mic_a)
    w = w / np.conj(response(w, aA))[:, None]
    h_B = response(w, aB)
    wng = 10 * np.log10(np.minimum(1.0, _abs2(h_B)) / white_gain(w))
    _check_floor(wng, floor_db, "two-talker beam")
    df = 10 * np.log10(0.5 * (1.0 + _abs2(h_B)) / noise_power(w, g))
    return TwoTalker(w=w, h_B=h_B, kind=kind, band=band, wng_db=wng, df_db=df)


def _discontinuous(r, h):
    """DESIGN M2 phase-continuity rule for the partner a single-target state does not target: its response r
    there against its AB response h. Limit 30 deg where |r| or |r/h| >= 0.5 (-6 dB), 90 deg elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = r / h
        strict = (np.abs(r) >= CONT_MAG) | (np.abs(rel) >= CONT_MAG)
        return ~(np.abs(np.degrees(np.angle(rel))) <= np.where(strict, CONT_DEG[0], CONT_DEG[1]))


@dataclass
class BusWeights:
    """M2 bus weight sets of one configuration (calibration block, partner TDOAs)."""
    w: np.ndarray                    # (3, F, 2): states A (w_A), AB (w_AB), B (h_B* w_B), after the substitution
    tau: tuple                       # (tau_A, tau_B) in s
    fallback: str = ""               # '' (two-talker set) or 'broadside' / 'omni' (all three states equal)
    why: tuple = ()                  # S1 inseparable reasons that triggered the fallback
    ab: TwoTalker = None
    subst: np.ndarray = None         # (2, F) bool: bins where state A / state B use w_AB (phase continuity)
    df_db: np.ndarray = None         # (3, F) DF of each state toward its partner(s), against the design Gamma


def bus_weights(tau_A, tau_B, g, d_hat, floor_db, why=(), fallback="auto", f=cm.FREQS):
    """State weights for partners at tau_A, tau_B (DESIGN M2 bus). why: S1 inseparable reasons; if any, every
    state uses the DESIGN 2.1 fallback: 'broadside' (broadside MVDR, which for a real Gamma is the plain average),
    'omni' (Mic A alone) or 'auto' (broadside when both partners are near broadside, else omni)."""
    F = len(f)
    if why:
        mode = fallback if fallback != "auto" else ("broadside" if "both_broadside" in why else "omni")
        if mode == "broadside":
            w = mvdr_weights(cm.steer(f, 0.0), g, floor_db)[0]
        elif mode == "omni":
            w = np.stack([np.ones(F), np.zeros(F)], axis=-1).astype(complex)
            _check_floor(np.zeros(F), floor_db, "omni fallback")          # Mic A alone: WNG 0 dB
        else:
            raise ValueError(f"unknown fallback {fallback!r}")
        return BusWeights(w=np.stack([w, w, w]), tau=(tau_A, tau_B), fallback=mode, why=tuple(why))
    aA, aB = cm.steer(f, tau_A), cm.steer(f, tau_B)
    wA = mvdr_weights(aA, g, floor_db)[0]
    wB = mvdr_weights(aB, g, floor_db)[0]
    ab = two_talker(f, tau_A, tau_B, g, floor_db, d_hat)
    wB = np.conj(ab.h_B)[:, None] * wB                       # response h_B to partner B, as in state AB
    sub = np.stack([_discontinuous(response(wA, aB), ab.h_B), _discontinuous(response(wB, aA), 1.0)])
    wA = np.where(sub[0][:, None], ab.w, wA)
    wB = np.where(sub[1][:, None], ab.w, wB)
    dfs = np.stack([df_db(wA, aA, g), ab.df_db, df_db(wB, aB, g)])
    return BusWeights(w=np.stack([wA, ab.w, wB]), tau=(tau_A, tau_B), ab=ab, subst=sub, df_db=dfs)


# ------------------------------------------------------------------ time-varying beams
@dataclass
class Beam:
    """Time-varying beam: per frame l the weights are W(l) = sum_m val[l, m] * bank[idx[l, m]] (257, 2) and the
    output is Y(l) = W(l)^H X(l) per bin (linear, so masks can be applied to Y afterwards)."""
    bank: np.ndarray                 # (K, 257, 2) complex weight sets
    idx: np.ndarray                  # (N, M) int: bank entries mixed in each frame
    val: np.ndarray                  # (N, M) float: their coefficients (each row sums to 1)
    labels: list = field(default_factory=list)   # (K,) what each bank entry is
    diag: dict = field(default_factory=dict)

    def weights(self, l0=0, l1=None):
        """Per-bin weights of frames l0..l1-1: (257, n, 2) complex."""
        l1 = len(self.idx) if l1 is None else l1
        return np.einsum("lm,lmfc->flc", self.val[l0:l1], self.bank[self.idx[l0:l1]])

    def apply(self, X, chunk=1024):
        """Beam output Y = W^H X (257, N) of the calibrated STFT X (2, 257, N)."""
        N = X.shape[-1]
        if N != len(self.idx):
            raise ValueError(f"beam has {len(self.idx)} frames, X has {N}")
        Y = np.empty(X.shape[1:], dtype=complex)
        for l0 in range(0, N, chunk):
            l1 = min(N, l0 + chunk)
            W = self.weights(l0, l1)
            Y[:, l0:l1] = np.conj(W[..., 0]) * X[0, :, l0:l1] + np.conj(W[..., 1]) * X[1, :, l0:l1]
        return Y

    def df(self, cal, noise="measured", chunk=1024):
        """1 / (W^H Gamma W) per bin and frame (257, N): the beam's directivity factor for its unit-response
        target, i.e. the M4 beam correction (masks.cdr_gain(df=...)). cal: Calib or online schedule."""
        N = len(self.idx)
        out = np.empty((cm.N_BINS, N))
        for a, b, c in _blocks(cal, N):
            g = noise_coherence(c, noise)[:, None]
            for l0 in range(a, b, chunk):
                l1 = min(b, l0 + chunk)
                out[:, l0:l1] = 1.0 / noise_power(self.weights(l0, l1), g)
        return out


def crossfade(target, n_ramp):
    """Per-frame bank index (N,) -> (idx, val) (N, M) with linear crossfades of n_ramp frames: the target's
    coefficient rises by 1/n_ramp per frame and the others shrink in proportion, so the coefficients always sum
    to 1 and a single switch is a plain linear crossfade; a target that changes mid-ramp starts from its current
    coefficient."""
    target = np.asarray(target, dtype=int)
    N = len(target)
    step = 1.0 / max(1, int(n_ramp))
    cur = {int(target[0]): 1.0}
    changes = [(0, cur)]
    for l in range(N):
        t = int(target[l])
        c = cur.get(t, 0.0)
        if c < 1.0:
            c1 = 1.0 if c + step > 1.0 - 1e-9 else c + step
            s = (1.0 - c1) / (1.0 - c)
            cur = {k: v * s for k, v in cur.items() if k != t and v * s > 0.0}
            cur[t] = c1
            changes.append((l, cur))
    M = max(len(d) for _, d in changes)
    idx = np.zeros((N, M), dtype=int)
    val = np.zeros((N, M))
    ends = [l for l, _ in changes[1:]] + [N]
    for (l, d), e in zip(changes, ends):
        idx[l:e, :len(d)] = list(d.keys())
        val[l:e, :len(d)] = list(d.values())
    return idx, val


def hold_switch(target, n_hold, init=-1, n_release=None):
    """Follow a per-frame target index (-1 = no new evidence: keep the current one), switching at most once per
    n_hold frames (minimum dwell between two targets); with n_release, return to -1 after n_release frames
    without evidence and take the next target at once. Frames before the first target keep init."""
    out = np.empty(len(target), dtype=int)
    cur, last, quiet = int(init), -int(n_hold), 0
    for l, t in enumerate(np.asarray(target, dtype=int)):
        if t < 0:
            quiet += 1
            if n_release is not None and quiet >= n_release:
                cur = -1
        else:
            quiet = 0
            if t != cur and (cur < 0 or l - last >= n_hold):
                cur, last = int(t), l
        out[l] = cur
    return out


def _frames(seconds):
    return max(1, int(np.ceil(seconds / cm.HOP_S - 1e-9)))


def _run_length(c):
    """Length of the run of True ending at each element (0 where False)."""
    c = np.asarray(c, dtype=bool)
    n = np.arange(1, len(c) + 1)
    return np.where(c, n - np.maximum.accumulate(np.where(c, 0, n)), 0)


def _blocks(cal, N):
    """A Calib or an online schedule [(start_frame, Calib), ...] -> [(start, end, Calib)]."""
    if isinstance(cal, cb.Calib):
        return [(0, N, cal)]
    starts = [s for s, _ in cal] + [N]
    return [(s, e, c) for (s, c), e in zip(cal, starts[1:]) if e > s]


def _floor(scenario, floor_db):
    return wng_floor(scenario) if floor_db is None else np.broadcast_to(np.asarray(floor_db, float), cm.FREQS.shape)


def _runs(*keys):
    """[(start, end)] of the runs over which every per-frame key array is constant."""
    N = len(keys[0])
    cut = np.zeros(N, dtype=bool)
    cut[0] = True
    for k in keys:
        k = np.asarray(k)
        cut[1:] |= k[1:] != k[:-1]
    s = np.flatnonzero(cut)
    return list(zip(s, list(s[1:]) + [N]))


def _qtau(t):
    """TDOA quantized to TAU_Q steps as an int key; NaN -> a sentinel."""
    t = np.asarray(t, dtype=float)
    return np.where(np.isfinite(t), np.rint(np.nan_to_num(t) / TAU_Q), np.iinfo(np.int64).min).astype(np.int64)


def _tau_of(q):
    return np.nan if q == np.iinfo(np.int64).min else float(q) * TAU_Q


def _partner_taus(trk, N):
    if trk.p_A is None:
        raise ValueError("this beam needs S1's bus partner model (tracker.track(..., partners=True))")
    ta = trk.tau_A_t if trk.tau_A_t is not None else np.full(N, trk.tau_A)
    tb = trk.tau_B_t if trk.tau_B_t is not None else np.full(N, trk.tau_B)
    return ta, tb


def implausible_reasons(tau_A, tau_B, d_hat, max_tdoa=MAX_TDOA, same_side=False):
    """Beam-side check of S1's two partner TDOAs (not in DESIGN), reasons to use the DESIGN 2.1 fallback:
    'beyond_endfire': a TDOA beyond max_tdoa * d_hat / c (None: no check). No plane wave at the estimated spacing
    has one; on v2 dev these were S1 histogram peaks piled up at its +-1.2 d_hat / c grid edge (40 mm prior on a
    60 mm array) or a missed skew, and an AB beam constrained there nulled the real partners by 10-19 dB.
    'same_side': both TDOAs on one side of broadside (unless same_side=True). With the true TDOAs such a pair
    gains little (+0.01 worst-partner ESTOI); on v2 dev all 13 such S1 pairs were wrong (a partner plus a
    passenger or noise source) and cost 0.033 (M2) / 0.020 (M1) on average."""
    if not (np.isfinite(tau_A) and np.isfinite(tau_B)):
        return ()
    why = []
    if max_tdoa is not None and max(abs(tau_A), abs(tau_B)) > max_tdoa * d_hat / C:
        why.append("beyond_endfire")
    if not same_side and tau_A * tau_B > 0:
        why.append("same_side")
    return tuple(why)


# ------------------------------------------------------------------ M2 bus
def bus_states(p_A, p_B, hold_s=STATE_HOLD_S, enter=P_ENTER, other=P_OTHER, init=1):
    """DESIGN M2 bus states per frame (0 = A, 1 = AB, 2 = B) from S1's posteriors. Enter A once p_A > enter and
    p_B < other has held for hold_s (B symmetric); leave A for AB as soon as p_B >= other (double talk or
    uncertainty), B likewise; otherwise keep the state, also while nobody talks. Starts in AB."""
    n = _frames(hold_s)
    run_a = _run_length((p_A > enter) & (p_B < other))
    run_b = _run_length((p_B > enter) & (p_A < other))
    out = np.empty(len(p_A), dtype=np.int8)
    s = init
    for l in range(len(p_A)):
        if run_a[l] >= n:
            s = 0
        elif run_b[l] >= n:
            s = 2
        elif (s == 0 and p_B[l] >= other) or (s == 2 and p_A[l] >= other):
            s = 1
        out[l] = s
    return out


def bus_beam(X, cal, trk, scenario="bus", noise="measured", fallback="auto", floor_db=None, ramp_s=RAMP_S,
             hold_s=STATE_HOLD_S, enter=P_ENTER, other=P_OTHER, states="ab", max_tdoa=MAX_TDOA, same_side=False):
    """M2 bus beam (DESIGN M2, 2.1): states A / AB / B from S1's posteriors (bus_states), weights w_A, w_AB,
    h_B* w_B with the per-bin phase-continuity substitution, 50 ms linear crossfades; the DESIGN 2.1 fallback
    when the partners are inseparable or their TDOAs fail implausible_reasons. X: calibrated STFT
    (2, 257, N); cal: Calib or online schedule (Gamma_n, d_hat per block); trk: S1 Track with the partner model
    (offline: constant TDOAs; online: per-frame, a new design per change of the 1 us-quantized TDOAs or of the
    calibration block, crossfaded like a state change). states: 's1' (the state machine) or 'ab' (the AB beam
    throughout). Returns a Beam; diag: 'states' (N,), 'state_frac', 'configs' (one dict per design),
    'fallback_frac'."""
    N = X.shape[-1]
    blocks = _blocks(cal, N)
    floor = _floor(scenario, floor_db)
    if states == "s1":
        states = bus_states(trk.p_A, trk.p_B, hold_s, enter, other)
    elif states == "ab":
        states = np.ones(N, dtype=np.int8)
    else:
        raise ValueError(f"unknown states {states!r}")
    ta, tb = _partner_taus(trk, N)
    blk = np.repeat(np.arange(len(blocks)), [e - s for s, e, _ in blocks])
    qa, qb = _qtau(ta), _qtau(tb)
    bank, labels, base, configs = [], [], {}, []
    target = np.empty(N, dtype=int)
    fb = np.zeros(N, dtype=bool)
    for s, e in _runs(blk, qa, qb):
        key = (int(blk[s]), int(qa[s]), int(qb[s]))
        if key not in base:
            c = blocks[key[0]][2]
            t_a, t_b = _tau_of(key[1]), _tau_of(key[2])
            n_peaks = 0 if np.isnan(t_a) else (1 if t_a == t_b else 2)
            why = tk.inseparable_reasons(n_peaks, t_a, t_b, c.d_hat)
            why = why or implausible_reasons(t_a, t_b, c.d_hat, max_tdoa, same_side)
            bw = bus_weights(t_a, t_b, noise_coherence(c, noise), c.d_hat, floor, why, fallback)
            base[key] = len(bank)
            bank.extend(bw.w)
            labels.extend((name, key) for name in STATES)
            configs.append(_bus_diag(bw, int(s), key[0]))
        target[s:e] = base[key] + states[s:e].astype(int)       # int8 states would overflow past 42 designs
        fb[s:e] = bool(configs[base[key] // 3]["fallback"])
    idx, val = crossfade(target, _frames(ramp_s))
    diag = {"states": states, "state_frac": {n: float(np.mean(states == i)) for i, n in enumerate(STATES)},
            "configs": configs, "fallback_frac": float(fb.mean())}
    return Beam(np.stack(bank), idx, val, labels, diag)


def _bus_diag(bw, start, block):
    d = {"start": start, "block": block, "tau_A": bw.tau[0], "tau_B": bw.tau[1], "fallback": bw.fallback,
         "why": bw.why}
    if bw.ab is not None:
        d.update(band=bw.ab.band, kind_counts=np.bincount(bw.ab.kind, minlength=3).tolist(),
                 n_subst=bw.subst.sum(axis=1).tolist(), ai_df_db=[ai_index(x) for x in bw.df_db])
    return d


# ------------------------------------------------------------------ steering (M1, M2 meeting)
def _weighted_median(x, w, n, causal, min_w):
    """Weighted median of x over windows of n frames (trailing if causal, else centred); NaN where the window's
    total weight is below min_w."""
    pad = (n - 1, 0) if causal else ((n - 1) // 2, n // 2)
    X = sliding_window_view(np.pad(np.nan_to_num(x), pad), n)
    Wt = sliding_window_view(np.pad(w, pad), n)
    o = np.argsort(X, axis=1)
    xs, cw = np.take_along_axis(X, o, 1), np.cumsum(np.take_along_axis(Wt, o, 1), axis=1)
    k = np.argmax(cw >= 0.5 * cw[:, -1:], axis=1)
    return np.where(cw[:, -1] >= min_w, xs[np.arange(len(xs)), k], np.nan)


def noise_tdoa(trk, causal=False):
    """TDOA of the strongest coherent noise source (e.g. a projector fan) per frame, and whether it dominates:
    the mode of S1's tau over idle frames (vad < NOISE_P), weighted by directness on a 2 us grid (10 us boxcar),
    'dominant' when +-20 us around it hold NOISE_MIN of the weight. Offline: one value from the whole file;
    causal: from the idle frames before each NOISE_BLOCK_S block, forgetting with NOISE_FORGET_S.
    Returns (tau_n (N,), dominant (N,) bool)."""
    N = len(trk.tau)
    idle = (trk.vad < NOISE_P) & cm.valid_frames(N)
    half = 150                                          # grid +-300 us covers S1's +-1.2 d_hat / c up to 8.5 cm
    idx = np.clip(np.rint(np.nan_to_num(trk.tau) / 2e-6).astype(int) + half, 0, 2 * half)
    w = np.where(idle, trk.directness, 0.0)

    def mode(h):
        k = int(np.argmax(np.convolve(h, np.ones(5), mode="same")))    # peak of the 10 us-smoothed histogram
        tot = h.sum()
        return (k - half) * 2e-6, tot > 0 and h[max(0, k - 10):k + 11].sum() >= NOISE_MIN * tot

    if not causal:
        t, dom = mode(np.bincount(idx, weights=w, minlength=2 * half + 1))
        return np.full(N, t), np.full(N, dom)
    nb = max(1, int(round(NOISE_BLOCK_S / cm.HOP_S)))
    decay = np.exp(-NOISE_BLOCK_S / NOISE_FORGET_S)
    tau_n, dom = np.zeros(N), np.zeros(N, dtype=bool)
    H = np.zeros(2 * half + 1)
    for a in range(0, N, nb):
        b = min(N, a + nb)
        tau_n[a:b], dom[a:b] = mode(H) if H.any() else (0.0, False)
        H = H * decay + np.bincount(idx[a:b], weights=w[a:b], minlength=2 * half + 1)
    return tau_n, dom


def active_talker_tdoa(trk, win_s=STEER_WIN_S, causal=False, min_w=STEER_MIN_W, speech_p=SPEECH_P,
                       noise_excl=NOISE_EXCL):
    """TDOA of the active talker per frame (s; NaN = no evidence): weighted median of S1's tau over win_s
    (trailing if causal, else centred), weights vad * directness on speech frames (vad >= speech_p). Frames
    whose tau lies within noise_excl of a dominant coherent noise source (noise_tdoa; None: no exclusion) are
    no evidence: a weak talker's S1 TDOA sits on the noise source (the projector fan on dev)."""
    w = np.where((trk.vad >= speech_p) & cm.valid_frames(len(trk.tau)), trk.vad * trk.directness, 0.0)
    if noise_excl is not None:
        tau_n, dom = noise_tdoa(trk, causal)
        w = np.where(dom & (np.abs(trk.tau - tau_n) <= noise_excl), 0.0, w)
    return _weighted_median(trk.tau, w, _frames(win_s), causal, min_w)


def tdoa_grid(d_hat, n=N_GRID):
    """n TDOAs evenly over +-1.2 d_hat / c (s)."""
    return GRID_SPAN * d_hat / C * np.linspace(-1.0, 1.0, n)


def hold_tdoa(tau, n_hold, hyst=STEER_HYST, init=np.nan, n_release=None):
    """Held steering TDOA per frame (NaN = the neutral beam): move to tau (NaN = no evidence) when it differs
    from the current value by more than hyst, at most once per n_hold frames (minimum dwell between talkers);
    with n_release, return to NaN after n_release frames without evidence, and take the next talker at once.
    init: before the first talker."""
    out = np.empty(len(tau))
    cur, last, quiet = float(init), -int(n_hold), 0
    for l, t in enumerate(np.asarray(tau, dtype=float)):
        if np.isnan(t):
            quiet += 1
            if n_release is not None and quiet >= n_release:
                cur = np.nan
        else:
            quiet = 0
            if np.isnan(cur) or (abs(t - cur) > hyst and l - last >= n_hold):
                cur, last = t, l
        out[l] = cur
    return out


def _steer_ids(trk, blocks, n_grid, win_s, causal, hold_s, release_s, noise_excl):
    """Per-frame steering id block * n_grid + grid index of the held active-talker TDOA (-1: the neutral beam),
    and the per-block grids (+-1.2 d_hat / c of each calibration block)."""
    tau = hold_tdoa(active_talker_tdoa(trk, win_s, causal, noise_excl=noise_excl), _frames(hold_s),
                    n_release=None if release_s is None else _frames(release_s))
    ids = np.full(len(tau), -1, dtype=int)
    grids = []
    for i, (a, b, c) in enumerate(blocks):
        grids.append(tdoa_grid(c.d_hat, n_grid))
        t = tau[a:b]
        ok = np.isfinite(t)
        ids[a:b][ok] = i * n_grid + np.argmin(np.abs(grids[-1][None, :] - t[ok, None]), axis=1)
    return ids, grids


def neutral_weights(neutral, f=cm.FREQS):
    """Weights of the neutral beam used without talker evidence: 'omni' (Mic A alone) or 'broadside' (DAS)."""
    if neutral == "omni":
        return np.stack([np.ones(len(f)), np.zeros(len(f))], axis=-1).astype(complex)
    if neutral == "broadside":
        return das_weights(f, 0.0)
    raise ValueError(f"unknown neutral beam {neutral!r}")


def _bank_from_ids(held, make):
    """Bank of make(id) for every id used in the held sequence -> (bank, labels, per-frame bank index)."""
    uniq, inv = np.unique(held, return_inverse=True)
    return np.stack([make(int(u)) for u in uniq]), [int(u) for u in uniq], inv.reshape(-1)


def das_beam(X, cal, trk, scenario, mode="steered", hold_s=DAS_HOLD_S, ramp_s=RAMP_S, win_s=STEER_WIN_S,
             causal=None, n_grid=N_GRID, release_s=STEER_RELEASE_S, neutral=NEUTRAL, max_tdoa=MAX_TDOA,
             same_side=False, noise_excl=NOISE_EXCL):
    """M1 delay-and-sum (DESIGN M1). mode 'broadside': (X_A + X_B) / 2 (M1a); 'steered' (M1b): a(f, tau)^H X / 2
    toward the active partner (bus: the higher posterior above 0.25, steered to its S1 TDOA, only with two
    partner peaks that pass implausible_reasons) or the active talker (meeting: active_talker_tdoa, re-steered
    when it moves by more than 10 us, 64-point grid), held >= hold_s, 50 ms crossfades; the neutral beam (Mic A
    alone) before the first one and after release_s without evidence (None: hold the last one).
    causal: trailing TDOA window (default: trk.online)."""
    N = X.shape[-1]
    f = cm.FREQS
    if mode == "broadside":
        return Beam(das_weights(f, 0.0)[None], np.zeros((N, 1), int), np.ones((N, 1)), ["broadside"], {})
    if mode != "steered":
        raise ValueError(f"unknown M1 mode {mode!r}")
    blocks = _blocks(cal, N)
    if scenario == "bus":
        ta, tb = _partner_taus(trk, N)
        d_t = np.concatenate([np.full(b - a, c.d_hat) for a, b, c in blocks])
        lim = np.inf if max_tdoa is None else max_tdoa * d_t / C        # implausible_reasons, per frame
        two = np.isfinite(ta) & (ta != tb) & (np.maximum(np.abs(ta), np.abs(tb)) <= lim) & (same_side | (ta * tb <= 0))
        act = np.where(two & (np.maximum(trk.p_A, trk.p_B) > ACTIVE_P), np.where(trk.p_A >= trk.p_B, 0, 1), -1)
        who = hold_switch(act, _frames(hold_s), n_release=None if release_s is None else _frames(release_s))
        tau = np.where(who == 0, ta, np.where(who == 1, tb, np.nan))
        bank, keys, target = _bank_from_ids(_qtau(tau), lambda q: neutral_weights(neutral, f) if np.isnan(_tau_of(q))
                                            else das_weights(f, q * TAU_Q))
        labels = [(neutral, None) if np.isnan(_tau_of(k)) else ("das", k * TAU_Q) for k in keys]
        diag = {"partner": who}
    else:
        held, grids = _steer_ids(trk, blocks, n_grid, win_s, trk.online if causal is None else causal, hold_s,
                                 release_s, noise_excl)
        bank, keys, target = _bank_from_ids(held, lambda i: neutral_weights(neutral, f) if i < 0 else
                                            das_weights(f, grids[i // n_grid][i % n_grid]))
        labels = [("das", grids[k // n_grid][k % n_grid]) if k >= 0 else (neutral, None) for k in keys]
        diag = {"steer_id": held}
    idx, val = crossfade(target, _frames(ramp_s))
    return Beam(bank, idx, val, labels, diag)


# ------------------------------------------------------------------ M2 meeting
def meeting_beam(X, cal, trk, scenario="meeting", mode="steered", noise="measured", floor_db=None, n_grid=N_GRID,
                 hold_s=None, win_s=STEER_WIN_S, ramp_s=RAMP_S, causal=None, release_s=STEER_RELEASE_S,
                 neutral=NEUTRAL, noise_excl=NOISE_EXCL):
    """M2 meeting beam (DESIGN M2, 2.2), WNG floor -3 dB (scenario 'bus': the bus floors).
    mode 'steered': superdirective MVDR toward the active talker's TDOA (active_talker_tdoa, re-steered when it
    moves by more than 10 us, 64-point grid over +-1.2 d_hat / c per calibration block, designs cached), held
    >= hold_s (default 0.4 s), 50 ms crossfades; the neutral beam (Mic A alone) before the first talker and
    after release_s without talker evidence (None: hold the last talker).
    mode 'fixed3': MVDR beams at tau = +d_hat/c, 0, -d_hat/c; on speech frames (vad >= 0.5) the beam with the
    highest smoothed (160 ms) 300-3000 Hz output power is selected, held >= hold_s (default 0.5 s). In a
    reverberant room this rule (DESIGN's) picks the least directive beam, broadside (= M1a), nearly always.
    causal: trailing TDOA window (default: trk.online). Returns a Beam; diag: 'steer_id' or 'select'."""
    N = X.shape[-1]
    f = cm.FREQS
    blocks = _blocks(cal, N)
    floor = _floor(scenario, floor_db)
    gs = [noise_coherence(c, noise) for _, _, c in blocks]
    if mode == "steered":
        held, grids = _steer_ids(trk, blocks, n_grid, win_s, trk.online if causal is None else causal,
                                 STEER_HOLD_S if hold_s is None else hold_s, release_s, noise_excl)

        def make(i):
            if i < 0:
                return neutral_weights(neutral, f)
            return mvdr_weights(cm.steer(f, grids[i // n_grid][i % n_grid]), gs[i // n_grid], floor)[0]

        bank, keys, target = _bank_from_ids(held, make)
        labels = [("mvdr", grids[k // n_grid][k % n_grid]) if k >= 0 else (neutral, None) for k in keys]
        diag = {"steer_id": held}
    elif mode == "fixed3":
        taus = [(c.d_hat / C, 0.0, -c.d_hat / C) for _, _, c in blocks]
        fixed = [np.stack([mvdr_weights(cm.steer(f, t), g, floor)[0] for t in tt]) for tt, g in zip(taus, gs)]
        sl = cm.bins_between(*SELECT_BAND)
        P = np.empty((3, N))
        for (a, b, _), wb in zip(blocks, fixed):
            for l0 in range(a, b, cm.CHUNK):
                l1 = min(b, l0 + cm.CHUNK)
                P[:, l0:l1] = np.sum(_abs2(np.einsum("kfc,cfl->kfl", np.conj(wb[:, sl]), X[:, sl, l0:l1])), axis=1)
        P = cm.smooth(P, SELECT_LAM)[0]
        sel = np.where(trk.vad >= SPEECH_P, np.argmax(P, axis=0), -1)
        held = hold_switch(sel, _frames(SELECT_HOLD_S if hold_s is None else hold_s), init=1)
        blk = np.repeat(np.arange(len(blocks)), [e - s for s, e, _ in blocks])
        bank = np.concatenate(fixed)
        labels = [("mvdr", t) for tt in taus for t in tt]
        target = 3 * blk + held
        diag = {"select": held}
    else:
        raise ValueError(f"unknown M2 meeting mode {mode!r}")
    idx, val = crossfade(target, _frames(ramp_s))
    return Beam(bank, idx, val, labels, diag)
