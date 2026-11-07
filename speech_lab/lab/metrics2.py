"""v2 scoring (DESIGN 4.2-4.4): Hagerman run plans and decoding, per-talker fidelity, levels, composites.

Definitions are documented in BENCH2.md "Metrics". Conventions used everywhere:
- A method is run once per Hagerman run (plan()); its outputs are decoded into separated components
  (decode()) for two stages: 'pre' (pre-AGC; the output itself when the method returns no stages) and
  'out' (the final output).
- Levels are mean-square levels in the scene SNR band (150-7000 Hz) at Mic A digital scale. A level change
  compares a decoded output component with the same groups' comp2 at Mic A over the same samples and is
  clipped to +-CAP_DB.
- Fidelity (SDR through a 512-tap allowed-distortion FIR, ESTOI) is computed on the normal output y1
  against a talker's own early image, per talker, so polarity and linear filtering of each talker are free.
  Nothing is sign-flipped globally.
"""
import warnings

import numpy as np
from pystoi import stoi
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, correlate, sosfiltfilt

from . import metrics as v1
from .common import FS, active_rms, db, rms
from .scene2 import Sparse

BAND_HZ = (150.0, 7000.0)       # level band: the scene SNR band (BENCH2.md "SNR rule")
CAP_DB = 60.0                   # level changes are clipped to +-CAP_DB
RATIO_CAP_DB = 150.0            # cap for SDR and the Hagerman consistency figures
FIR_TAPS, FIR_LEAD = 512, 128   # allowed-distortion FIR: lags -128 .. +383 samples
MIN_SPEECH = FS                 # at least 1 s of audio for a per-talker ESTOI / SDR / level
WIN = FS                        # 1 s windows (B2 balance, B5 RDA)
MIN_WIN_ACT = FS // 4           # a talker counts in a window with >= 0.25 s of activity in it
IDLE_GUARD = (int(0.1 * FS), int(0.3 * FS))  # idle: >= 0.1 s before and >= 0.3 s after target speech
STARTUP = int(0.5 * FS)         # idle samples in the first 0.5 s are skipped (method start-up)
MIN_IDLE = FS                   # NA_idle and NM need at least 1 s of idle samples
MAX_LAG, LAG_TOL = 1600, 2      # delay search +-100 ms (safety net); contract |lag| <= 2 samples
LAG_BUDGET = 30 * FS            # speech per talker used for the delay search
LAG_BAND_HZ = (1000.0, 4000.0)  # delay search band (see find_delay)
DIST_BINS = (("near", 0.0, 2.0), ("mid", 2.0, 4.0), ("far", 4.0, np.inf))  # meeting bins (m)
SOFT_DB = 55.5                  # meeting talkers at or below this dB SPL at 1 m are 'soft' (E7)
TURN_GAP = FS                   # same-talker segments less than 1 s apart form one turn
SETTLE_WIN, SETTLE_HOP, SETTLE_TOL = int(0.4 * FS), int(0.05 * FS), 3.0
SETTLE_MIN_TURN, SETTLE_FINAL = 2 * FS, FS
PASSENGER_ESTOI_MAX = 0.25      # focus composite: nearest passengers not followable
RDA_MIN = 10.0                  # focus composite: RDA P10 (dB)
TA_MIN = -3.0                   # focus composite and G0: TA_k (dB)
ESTOI_DROP_TOL = -0.03          # meeting T1: per-talker dESTOI
ESTOI_TIE = 1e-6                # composites: ESTOI_out >= ESTOI_in up to numerical ties
LEVEL_TOL = 3.0                 # meeting T1: |L_k - median| (dB); T2 margin
BUS_SPLITS = {"near": "passenger_near", "probe": "probe", "cone": "passenger_cone", "conv": "conv_pair",
              "far": "babble_far", "engine": "engine_road"}
EXTRA_SPLITS = ("cone", "conv", "far", "engine")   # only with extra=True (DESIGN 4.2: a 20% subset)
PENDING = {"wer": "pending: no ASR model downloaded yet (B8/T6 hook: score_wer)",
           "dnsmos": "pending: no DNSMOS model downloaded yet (B9/T9 hook: score_dnsmos)"}
_SOS = butter(4, BAND_HZ, btype="band", fs=FS, output="sos")
_SOS_LAG = butter(4, LAG_BAND_HZ, btype="band", fs=FS, output="sos")
_NAN = float("nan")


# ----------------------------------------------------------------------------- small helpers
def inband(x):
    """In-band (150-7000 Hz, zero-phase 4th-order Butterworth, as the scene SNR rule) copy, float32."""
    return sosfiltfilt(_SOS, np.asarray(x, np.float64)).astype(np.float32)


def _msq(x, mask=None):
    """Mean square of x (over mask), float64 accumulation; NaN when empty."""
    v = (x if mask is None else x[mask]).astype(np.float64)
    return float(v @ v / len(v)) if len(v) else _NAN


def _change(out_f, in_f, mask=None):
    """Level change (dB, clipped to +-CAP_DB) of out_f relative to in_f; NaN without input power."""
    p_in, p_out = _msq(in_f, mask), _msq(out_f, mask)
    if not p_in > 1e-30:
        return _NAN
    return float(np.clip(10 * np.log10((p_out + 1e-30) / p_in), -CAP_DB, CAP_DB))


def _ratio_db(num, den):
    """10 log10(num/den) for energies, capped at RATIO_CAP_DB; NaN when num (the reference) is silent."""
    if not num > 0:
        return _NAN
    return float(min(10 * np.log10(num / max(den, 1e-300)), RATIO_CAP_DB))


def _energy(x):
    x = np.asarray(x, np.float64)
    return float(x @ x)


def _active_db(x):
    """Active speech level (dB): RMS over the 20 ms frames within 30 dB of the loudest (lab.common.active_rms)."""
    return float(db(active_rms(np.asarray(x, np.float64))))


def _finite(v):
    return [x for x in v if np.isfinite(x)]


def _nanmin(v):
    v = _finite(v)
    return float(min(v)) if v else _NAN


def _nanmedian(v):
    v = _finite(v)
    return float(np.median(v)) if v else _NAN


def _nanmean(v):
    v = _finite(v)
    return float(np.mean(v)) if v else _NAN


def _strict_min(v):
    """min(v), or NaN when any value is NaN."""
    return float(min(v)) if all(np.isfinite(v)) else _NAN


def _crit(ok, *vals):
    """Composite criterion: 1.0 / 0.0, or NaN when any input is NaN."""
    return _NAN if not all(np.isfinite(vals)) else float(bool(ok(*vals)))


def _all_pass(crit):
    vals = list(crit.values())
    return _NAN if not all(np.isfinite(vals)) else float(all(v == 1.0 for v in vals))


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        elif b > a:
            out.append([a, b])
    return [(int(a), int(b)) for a, b in out]


def _from_mask(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _total(iv):
    return sum(b - a for a, b in iv)


def _concat(x, iv):
    """Concatenation of x[a:b] over intervals (x: ndarray or Sparse), float64."""
    return np.concatenate([np.asarray(x[a:b], np.float64) for a, b in iv]) if iv else np.zeros(0)


def _band_segments(x, iv, sos=None, margin=1600):
    """x[a:b] for each interval, band-pass filtered (default: in-band) with `margin` samples of context."""
    out = []
    for a, b in iv:
        lo, hi = max(a - margin, 0), min(b + margin, len(x))
        seg = np.asarray(x[lo:hi], np.float64)
        out.append((inband(seg) if sos is None else sosfiltfilt(sos, seg))[a - lo:b - lo].astype(np.float64))
    return out


def _shift(y, lag):
    """y advanced by lag samples (lag > 0: y was late), zero-filled."""
    out = np.zeros_like(y)
    if lag > 0:
        out[:len(y) - lag] = y[lag:]
    elif lag < 0:
        out[-lag:] = y[:len(y) + lag]
    else:
        out[:] = y
    return out


# ----------------------------------------------------------------------------- fidelity measures
def sdr_fir(y, ref, iv, taps=FIR_TAPS, lead=FIR_LEAD):
    """In-band BSS-Eval SDR (dB) of y against ref over intervals iv, with a `taps`-tap allowed-distortion FIR.

    Both signals are band-limited to BAND_HZ (as the scene SNR), then y is projected (least squares) onto
    ref filtered by a FIR at lags -lead .. taps-1-lead; SDR is the projection-to-residual energy ratio. Each
    interval is followed by `taps` zeros, so the FIR never mixes intervals and the Gram matrix is exactly
    Toeplitz (mir_eval's zero-padded convention). Polarity, delays inside the lag range and spectral tilt
    are free. NaN with less than MIN_SPEECH samples."""
    if _total(iv) < MIN_SPEECH:
        return _NAN
    gap = np.zeros(taps)
    yc = np.concatenate([p for s in _band_segments(y, iv) for p in (s, gap)])
    rc = np.concatenate([p for s in _band_segments(ref, iv) for p in (s, gap)])
    nfft = next_fast_len(len(yc) + taps)
    R = rfft(rc, nfft)
    ac = irfft(R * np.conj(R), nfft)[:taps]
    if not ac[0] > 0:
        return _NAN
    b = irfft(rfft(yc, nfft) * np.conj(R), nfft)[np.arange(-lead, taps - lead) % nfft]
    ac[0] *= 1 + 1e-9  # tiny diagonal loading for band-limited references
    h = solve_toeplitz(ac, b)
    target = float(h @ b)  # = ||projection||^2 at the least-squares solution
    resid = max(_energy(yc) - target, 1e-30)
    return float(min(10 * np.log10(max(target, 1e-30) / resid), RATIO_CAP_DB))


def estoi(y, ref, iv):
    """ESTOI (pystoi, extended=True) of y against ref on the concatenation of intervals iv; NaN when shorter
    than MIN_SPEECH or when too little speech remains after pystoi's silent-frame removal."""
    if _total(iv) < MIN_SPEECH:
        return _NAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = stoi(_concat(ref, iv), _concat(y, iv), FS, extended=True)
    return _NAN if v == 1e-5 else float(v)


def find_delay(pairs, max_lag=MAX_LAG):
    """Common delay (samples; > 0: output late) of separated target components against their references.

    pairs: [(component, reference, intervals)]; the same component object may appear in several pairs.
    Both signals are band-passed to LAG_BAND_HZ, where the S0 high-pass (<= 120 Hz) shifts phase by under
    0.5 samples; full-band correlation peaks move by several samples with that filter's low-frequency phase
    lead. The lag maximizes the sum over pairs of the squared normalized cross-correlation (polarity-free),
    using up to LAG_BUDGET samples per pair. Returns (lag, found); found is False (lag 0) when no pair
    correlates above 0.1."""
    acc = np.zeros(2 * max_lag + 1)
    filtered = {}
    for comp, ref, iv in pairs:
        if id(comp) not in filtered:
            filtered[id(comp)] = sosfiltfilt(_SOS_LAG, np.asarray(comp, np.float64))
        comp = filtered[id(comp)]
        xc = np.zeros(2 * max_lag + 1)
        ey = er = 0.0
        budget = LAG_BUDGET
        clipped = []
        for a, b in iv:
            b = min(b, a + budget)
            if b <= a:
                break
            budget -= b - a
            clipped.append((a, b))
        for (a, b), r in zip(clipped, _band_segments(ref, clipped, _SOS_LAG)):
            lo, hi = a - max_lag, b + max_lag
            w = np.zeros(hi - lo)
            s0, s1 = max(lo, 0), min(hi, len(comp))
            w[s0 - lo:s1 - lo] = comp[s0:s1]
            xc += correlate(w, r, mode="valid", method="fft")
            ey += _energy(comp[a:b])
            er += _energy(r)
        if ey > 0 and er > 0:
            acc += xc ** 2 / (ey * er)
    if not acc.max() > 0.01:
        return 0, False
    return int(np.argmax(acc)) - max_lag, True


# ----------------------------------------------------------------------------- Hagerman runs
def partner_groups(sc):
    """(group A, group B): the two 'partner' groups, A first by name (partner_A, partner_B)."""
    p = sorted(g for g, r in sc["groups"].items() if r == "partner")
    if len(p) != 2:
        raise ValueError(f"bus scene needs exactly 2 partner groups, found {p}")
    return p[0], p[1]


def _groups_with(sc, role):
    return tuple(g for g, r in sc["groups"].items() if r == role)


def plan(sc, extra=False):
    """Hagerman runs and decoders for a scene (DESIGN 4.2).

    Returns (runs, comps). runs = [(name, flipped groups)], runs[0] = ('y1', ()) is the unmodified mix.
    comps = {component: ([(coef, run), ...], input groups)}; input groups is None for the closure term.
    Bus: the 4 DESIGN runs over partner A, partner B and everything else N; a 2-run flip of the nearest
    passengers (B4, B5) and of a probe (B7) when present; extra=True adds 2-run flips of the cone passenger,
    conversing pairs, far babble and engine/road, and the target-flip consistency run F(A + B - N).
    Meeting: 2 runs, all talkers S versus everything else N; extra=True adds the consistency run F(-S + N)."""
    groups = tuple(sc["comp2"])
    if sc["meta"]["scenario"] == "bus":
        A, B = partner_groups(sc)
        N = tuple(g for g in groups if g not in (A, B))
        runs = [("y1", ()), ("y2", (B,) + N), ("y3", (A,) + N), ("y4", (A, B))]

        def four(a, b, c, d):
            return [(a / 4, "y1"), (b / 4, "y2"), (c / 4, "y3"), (d / 4, "y4")]

        comps = {"A": (four(1, 1, -1, -1), (A,)), "B": (four(1, -1, 1, -1), (B,)), "N": (four(1, -1, -1, 1), N),
                 "closure": (four(1, 1, 1, 1), None)}
        for key, role in BUS_SPLITS.items():
            gs = _groups_with(sc, role)
            if gs and (extra or key not in EXTRA_SPLITS):
                runs.append((key, gs))
                comps[key] = ([(0.5, "y1"), (-0.5, key)], gs)
        if "near" in comps:
            near = comps["near"][1]
            comps["rest"] = (comps["N"][0] + [(-0.5, "y1"), (0.5, "near")], tuple(g for g in N if g not in near))
        if extra:
            runs.append(("cons", N))
            comps["S_nf"] = ([(0.5, "y1"), (0.5, "cons")], (A, B))
            comps["S_tf"] = ([(0.5, "y1"), (-0.5, "y4")], (A, B))
        return runs, comps
    S = _groups_with(sc, "talker")
    N = tuple(g for g in groups if g not in S)
    runs = [("y1", ()), ("y2", N)]
    comps = {"S": ([(0.5, "y1"), (0.5, "y2")], S), "N": ([(0.5, "y1"), (-0.5, "y2")], N)}
    if extra:
        runs.append(("cons", S))
        comps["S_tf"] = ([(0.5, "y1"), (-0.5, "cons")], S)
    return runs, comps


def decode(outs, terms, stage):
    """Linear combination sum(coef * outs[run][stage]) of Hagerman run outputs, float64."""
    y = None
    for k, r in terms:
        v = k * np.asarray(outs[r][stage], np.float64)
        y = v if y is None else y + v
    return y


# ----------------------------------------------------------------------------- per-scene input side
class SceneCtx:
    """Input-side facts of one scene shared by every method scored on it: talker activity, in-band Mic-A
    inputs and memoized input-side fidelity (raw Mic A against each reference)."""

    def __init__(self, sc):
        self.sc = sc
        self.T = int(sc["mix"].shape[1])
        self.scenario = sc["meta"]["scenario"]
        self.xa = np.asarray(sc["mix"][0], np.float64)
        iv = {}
        for a, b, tk in sc["segs"]:
            iv.setdefault(tk, []).append((a, b))
        self.iv = {tk: _merge(v) for tk, v in iv.items()}
        self.n_act = np.zeros(self.T, np.int16)
        for v in self.iv.values():
            for a, b in v:
                self.n_act[a:b] += 1
        self.talkers = sc["meta"].get("talkers", {})
        self._memo = {}

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def mask(self, iv):
        m = np.zeros(self.T, bool)
        for a, b in iv:
            m[a:b] = True
        return m

    def active(self, tids):
        """Bool mask where any of `tids` is active."""
        return self.mask([p for t in tids for p in self.iv.get(t, [])])

    def alone(self, tid):
        """Intervals where `tid` is the only active talker (every talker with segs counts), memoized."""
        return self.memo(("alone", tid), lambda: _from_mask(self.mask(self.iv.get(tid, [])) & (self.n_act == 1)))

    def without(self, tid, others):
        """Intervals where `tid` is active and none of `others` is."""
        return _from_mask(self.mask(self.iv.get(tid, [])) & ~self.active(others))

    def tids_of(self, groups):
        return sorted(t for t, g in self.sc["talker_group"].items() if g in groups)

    def mic_a(self, groups):
        """In-band Mic-A sum of the comp2 groups (float32), memoized."""
        def make():
            y = np.zeros(self.T)
            for g in groups:
                v = self.sc["comp2"][g]
                if isinstance(v, Sparse):
                    for s, blk in v.blocks:
                        y[s:s + blk.shape[-1]] += blk[0]
                else:
                    y += v[0]
            return inband(y)
        return self.memo(("mic_a", tuple(groups)), make)

    def fidelity_in(self, tid, iv, key):
        """(SDR, ESTOI) of raw Mic A against tid's early image over iv, memoized under (tid, key)."""
        ref = self.sc["refs"][tid]
        return self.memo(("fid_in", tid, key), lambda: (sdr_fir(self.xa, ref, iv), estoi(self.xa, ref, iv)))

    def dist(self, tid):
        return float(self.talkers.get(tid, {}).get("dist_m", _NAN))


class _Comps:
    """Decoded components per stage, plus memoized in-band versions."""

    def __init__(self, outs, comps):
        self.outs, self.comps = outs, comps
        self.same = all(o["pre"] is o["out"] for o in outs.values())
        self._raw, self._f = {}, {}

    def has(self, c):
        return c in self.comps

    def raw(self, c, stage):
        stage = "out" if self.same else stage
        if (c, stage) not in self._raw:
            self._raw[(c, stage)] = decode(self.outs, self.comps[c][0], stage)
        return self._raw[(c, stage)]

    def f(self, c, stage):
        stage = "out" if self.same else stage
        if (c, stage) not in self._f:
            self._f[(c, stage)] = inband(self.raw(c, stage))
        return self._f[(c, stage)]

    def groups(self, c):
        return self.comps[c][1]


# ----------------------------------------------------------------------------- windows and noise
def _win_pow(xf, mask):
    """Per 1 s window: (sum of xf^2 over mask, number of mask samples)."""
    n = -(-len(xf) // WIN)
    pad = n * WIN - len(xf)
    p = np.pad(np.where(mask, xf.astype(np.float64) ** 2, 0.0), (0, pad)).reshape(n, WIN).sum(1)
    c = np.pad(mask, (0, pad)).reshape(n, WIN).sum(1)
    return p, c


def _win_change(out_f, in_f, mask):
    """Per 1 s window: level change (dB, clipped) where the mask covers >= MIN_WIN_ACT samples, else NaN."""
    po, c = _win_pow(out_f, mask)
    pi, _ = _win_pow(in_f, mask)
    ok = (c >= MIN_WIN_ACT) & (pi > 1e-30)
    v = np.clip(10 * np.log10((po + 1e-30) / np.where(ok, pi, 1.0)), -CAP_DB, CAP_DB)
    return np.where(ok, v, np.nan)


def _idle(ctx, active):
    """Samples at least IDLE_GUARD away from `active` (before, after) and after STARTUP."""
    pre, post = IDLE_GUARD
    edge = np.zeros(ctx.T + 1, np.int32)
    for a, b in _from_mask(active):
        edge[max(a - pre, 0)] += 1
        edge[min(b + post, ctx.T)] -= 1
    idle = np.cumsum(edge[:-1]) == 0
    idle[:STARTUP] = False
    return idle


def _noise_terms(ctx, C, act):
    """NA_active and NA_idle (pre stage), NM (final output) and its pre-stage value (B3, T4)."""
    n_in = ctx.mic_a(C.groups("N"))
    idle = _idle(ctx, act)
    if idle.sum() < MIN_IDLE:
        idle[:] = False  # too little idle time: NA_idle and NM are NaN
    na = {st: (-_change(C.f("N", st), n_in, act), -_change(C.f("N", st), n_in, idle)) for st in ("pre", "out")}
    return {"na_active": na["pre"][0], "na_idle": na["pre"][1], "nm": na["out"][1] - na["out"][0],
            "nm_pre": na["pre"][1] - na["pre"][0], "na_active_out": na["out"][0], "na_idle_out": na["out"][1],
            "idle_s": float(idle.sum()) / FS}


# ----------------------------------------------------------------------------- scoring entry point
def _delay(ctx, outs, comps):
    """find_delay() on the separated target components of the final output (bus: each partner over its
    speech; meeting: all speech, per talker where the talker is alone)."""
    sc = ctx.sc
    if ctx.scenario == "bus":
        pairs = [(decode(outs, comps[c][0], "out"), sc["refs"][t], ctx.iv.get(t, []))
                 for c, g in zip("AB", partner_groups(sc)) for t in ctx.tids_of((g,))]
    else:
        s_out = decode(outs, comps["S"][0], "out")
        pairs = [(s_out, sc["refs"][t], ctx.alone(t)) for t in ctx.tids_of(comps["S"][1])]
    return find_delay(pairs)


def score(sc, outs, comps, ctx=None):
    """All metrics of one method on one scene from its Hagerman run outputs.

    outs: {run: {'out': (T,), 'pre': (T,)}} for every run of plan(sc, ...); comps: plan()'s decoders.
    Returns a JSON-able dict with keys delay, delay_found, aligned, m (decision and report metrics), legacy
    (v1-style, never used for decisions), talkers (per-talker details) and pending (WER/DNSMOS hooks).
    A delay beyond the +-2-sample contract is reported and compensated before scoring (safety net)."""
    ctx = ctx or SceneCtx(sc)
    lag, found = _delay(ctx, outs, comps)
    aligned = abs(lag) > LAG_TOL
    if aligned:
        shifted = {}
        for r, o in outs.items():
            out = _shift(o["out"], lag)
            shifted[r] = {"out": out, "pre": out if o["pre"] is o["out"] else _shift(o["pre"], lag)}
        outs = shifted
    C = _Comps(outs, comps)
    res = (_score_bus if ctx.scenario == "bus" else _score_meeting)(ctx, C, outs["y1"]["out"])
    res.update(delay=int(lag), delay_found=bool(found), aligned=bool(aligned), pending=dict(PENDING))
    m = res["m"]
    m.update(delay=float(lag), delay_ok=float(abs(lag) <= LAG_TOL))
    if C.has("closure"):
        m["closure_db"] = _ratio_db(_energy(outs["y1"]["out"]), _energy(C.raw("closure", "out")))
    if C.has("S_tf"):
        ref = C.raw("S_nf" if C.has("S_nf") else "S", "out")
        m["consistency_db"] = _ratio_db(_energy(ref), _energy(ref - C.raw("S_tf", "out")))
    return res


# ----------------------------------------------------------------------------- bus (DESIGN 4.3)
def partner_fidelity(ctx, y):
    """B1 for output y: per-partner SDR / ESTOI against the partner's early image over the samples where
    that partner talks and the other partner is silent (all of its speech if that is under 1 s)."""
    sc = ctx.sc
    A, B = partner_groups(sc)
    (tA,), (tB,) = ctx.tids_of((A,)), ctx.tids_of((B,))
    m = {}
    for k, t, o in (("A", tA, tB), ("B", tB, tA)):
        iv = ctx.without(t, [o])
        if _total(iv) < MIN_SPEECH:
            iv = ctx.iv.get(t, [])
        ref = sc["refs"][t]
        m[f"sdr_{k}"] = sdr_fir(y, ref, iv)
        m[f"estoi_{k}"] = estoi(y, ref, iv)
        m[f"sdr_in_{k}"], m[f"estoi_in_{k}"] = ctx.fidelity_in(t, iv, "partner")
    m["estoi_wrst"] = _nanmin([m["estoi_A"], m["estoi_B"]])
    m["estoi_wrst_in"] = _nanmin([m["estoi_in_A"], m["estoi_in_B"]])
    m["d_estoi_wrst"] = m["estoi_wrst"] - m["estoi_wrst_in"]
    m["d_estoi_min"] = _strict_min([m["estoi_A"] - m["estoi_in_A"], m["estoi_B"] - m["estoi_in_B"]])
    m["sdr_wrst"] = _nanmin([m["sdr_A"], m["sdr_B"]])
    m["d_sdr_min"] = _strict_min([m["sdr_A"] - m["sdr_in_A"], m["sdr_B"] - m["sdr_in_B"]])
    return m


def _score_bus(ctx, C, y):
    sc = ctx.sc
    gA, gB = partner_groups(sc)
    tid = {"A": ctx.tids_of((gA,)), "B": ctx.tids_of((gB,))}
    grp = {"A": (gA,), "B": (gB,)}
    m = partner_fidelity(ctx, y)
    act_k = {k: ctx.active(tid[k]) for k in "AB"}
    act = act_k["A"] | act_k["B"]
    # B2: partner level change before AGC (and after), balance over 1 s windows
    ta_w = {}
    for k in "AB":
        p_in = ctx.mic_a(grp[k])
        m[f"ta_{k}"] = _change(C.f(k, "pre"), p_in, act_k[k])
        m[f"ta_out_{k}"] = _change(C.f(k, "out"), p_in, act_k[k])
        ta_w[k] = _win_change(C.f(k, "pre"), p_in, act_k[k])
    m["ta_min"] = _strict_min([m["ta_A"], m["ta_B"]])
    m["ta_mean"] = (m["ta_A"] + m["ta_B"]) / 2
    m["ta_mean_out"] = (m["ta_out_A"] + m["ta_out_B"]) / 2
    m["balance"] = abs(m["ta_A"] - m["ta_B"])
    dev = []
    for k, o in (("A", "B"), ("B", "A")):
        other = _nanmedian(ta_w[o])
        dev += _finite([abs(v - other) for v in ta_w[k]])
    m["balance_med"] = float(np.median(dev)) if dev else _NAN
    m["balance_p90"] = float(np.percentile(dev, 90)) if dev else _NAN
    # B3: noise during partner speech and in pauses
    m.update(_noise_terms(ctx, C, act))
    m["dsnr_active"] = m["na_active"] + m["ta_mean"]
    # B4: SIR change per interferer group during partner speech (pre-AGC)
    for c in ("near", "rest") + EXTRA_SPLITS:
        if C.has(c) and C.groups(c):
            m[f"dsir_{c}"] = m["ta_mean"] - _change(C.f(c, "pre"), ctx.mic_a(C.groups(c)), act)
    # B5: relative distractor attenuation per 1 s window (final output)
    near_t = ctx.tids_of(C.groups("near")) if C.has("near") else []
    if near_t:
        d_near = _win_change(C.f("near", "out"), ctx.mic_a(C.groups("near")), ctx.active(near_t))
        ta_ws = np.stack([_win_change(C.f(k, "out"), ctx.mic_a(grp[k]), act_k[k]) for k in "AB"])
        cnt = np.isfinite(ta_ws).sum(0)
        ta_p = np.where(cnt > 0, np.nansum(ta_ws, 0) / np.maximum(cnt, 1), m["ta_mean_out"])
        rda = (ta_p - d_near)[np.isfinite(d_near)]
        if len(rda):
            m.update(rda_mean=float(rda.mean()), rda_p10=float(np.percentile(rda, 10)), rda_n=float(len(rda)))
    # B6: passenger intelligibility on y1 where the passenger talks and both partners are silent (other
    # passengers are background there: strictly-alone passenger speech hardly exists in the bus scenes)
    talkers = {}
    pax = {"pass": sorted(near_t, key=ctx.dist)[:3], "conv": ctx.tids_of(_groups_with(sc, "conv_pair"))}
    for tag, tids in pax.items():
        e_out, e_in = [], []
        for t in tids:
            iv = ctx.without(t, tid["A"] + tid["B"])
            e_out.append(estoi(y, sc["refs"][t], iv))
            e_in.append(ctx.memo(("pax_in", t), lambda: estoi(ctx.xa, sc["refs"][t], iv)))
            talkers[t] = {"role": "near" if tag == "pass" else "conv", "dist_m": ctx.dist(t),
                          "scored_s": _total(iv) / FS, "estoi": e_out[-1], "estoi_in": e_in[-1]}
        m[f"{tag}_estoi_med"] = _nanmedian(e_out)
        m[f"{tag}_estoi_in_med"] = _nanmedian(e_in)
        m[f"{tag}_drop_med"] = _nanmedian([i - o for i, o in zip(e_in, e_out)])
    # B7: probe level change relative to the partners (final output)
    if C.has("probe"):
        pt = ctx.tids_of(C.groups("probe"))
        p_in = ctx.mic_a(C.groups("probe"))
        p_act = ctx.active(pt) if pt else np.abs(p_in) > 0
        m["probe_rel_db"] = _change(C.f("probe", "out"), p_in, p_act) - m["ta_mean_out"]
        m["probe_dist_m"] = ctx.dist(pt[0]) if pt else _NAN
    for k in "AB":
        talkers[tid[k][0]] = {"role": "partner", "dist_m": ctx.dist(tid[k][0]),
                              **{q: m[f"{q}_{k}"] for q in ("sdr", "sdr_in", "estoi", "estoi_in", "ta", "ta_out")}}
    # composite: two-talker focus rate (DESIGN 4.3)
    crit = {"focus_c1": _crit(lambda d: d >= -ESTOI_TIE, m["d_estoi_min"]),
            "focus_c2": _crit(lambda e: e <= PASSENGER_ESTOI_MAX, m["pass_estoi_med"]),
            "focus_c3": _crit(lambda r: r >= RDA_MIN, m.get("rda_p10", _NAN)),
            "focus_c4": _crit(lambda t: t >= TA_MIN, m["ta_min"])}
    m.update(crit, focus_pass=_all_pass(crit))
    return {"m": m, "legacy": _legacy_bus(ctx, y, tid["A"][0], tid["B"][0]), "talkers": talkers}


def _legacy_bus(ctx, y, tA, tB):
    """v1 columns (lab.metrics.score_bus against the summed partner reference): report only."""
    sc = ctx.sc
    segs = [(a, b, 0 if t == tA else 1) for a, b, t in sc["segs"] if t in (tA, tB)]
    v1scene = {"ref_early": np.asarray(sc["refs"][tA], np.float64) + np.asarray(sc["refs"][tB], np.float64),
               "mix": sc["mix"], "segs": segs}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r = v1.score_bus(np.asarray(y, np.float64), v1scene)
    return {"dsisdr_sum": r["si_sdr_imp"], "br": r["bg_reduction_db"], "pesq": r["pesq"], "pesq_in": r["pesq_in"],
            "estoi_sum": r["estoi"], "estoi_sum_in": r["estoi_in"]}


# ----------------------------------------------------------------------------- meeting (DESIGN 4.4)
def _turns(ctx):
    """[(talker, start, end, returning)]: same-talker segments < TURN_GAP apart merged, in time order."""
    turns = []
    for a, b, tk in sorted(ctx.sc["segs"]):
        if turns and turns[-1][0] == tk and a - turns[-1][2] < TURN_GAP:
            turns[-1][2] = max(turns[-1][2], b)
        else:
            turns.append([tk, a, b])
    seen, out = set(), []
    for tk, a, b in turns:
        out.append((tk, a, b, tk in seen))
        seen.add(tk)
    return out


def _settling(out_f, in_f, a, b):
    """T5 for one turn [a, b): seconds from the onset until the 400 ms output/input gain stays within
    +-SETTLE_TOL of the turn's final gain (median over windows starting in its last second). Windows more
    than 20 dB below the turn's loudest input window are ignored. NaN for turns under 2 s."""
    if b - a < SETTLE_MIN_TURN:
        return _NAN
    co = np.concatenate([[0.0], np.cumsum(out_f[a:b].astype(np.float64) ** 2)])
    ci = np.concatenate([[0.0], np.cumsum(in_f[a:b].astype(np.float64) ** 2)])
    st = np.arange(0, b - a - SETTLE_WIN + 1, SETTLE_HOP)
    po, pi = co[st + SETTLE_WIN] - co[st], ci[st + SETTLE_WIN] - ci[st]
    act = pi > pi.max() * 1e-2
    g = 10 * np.log10((po + 1e-30) / (pi + 1e-30))
    last = act & (st >= (b - a) - SETTLE_FINAL)
    if not last.any():
        return _NAN
    bad = np.flatnonzero(act & (np.abs(g - np.median(g[last])) > SETTLE_TOL))
    if not len(bad):
        return 0.0
    later = np.flatnonzero(act & (np.arange(len(st)) > bad[-1]))
    return float(st[later[0]] / FS) if len(later) else float((b - a) / FS)


def _bin(d):
    for name, lo, hi in DIST_BINS:
        if lo <= d < hi:
            return name
    return "?"


def _score_meeting(ctx, C, y):
    sc = ctx.sc
    S = C.groups("S")
    tids = ctx.tids_of(S)
    s_in, s_pre, s_out = ctx.mic_a(S), C.f("S", "pre"), C.f("S", "out")
    talkers, skipped = {}, []
    for t in tids:
        iv = ctx.alone(t)
        if _total(iv) < MIN_SPEECH:
            skipped.append(t)
            continue
        mk = ctx.mask(iv)
        ref = sc["refs"][t]
        sdr_in, e_in = ctx.fidelity_in(t, iv, "alone")
        r = {"dist_m": ctx.dist(t), "level_db": float(ctx.talkers.get(t, {}).get("level_db", _NAN)),
             "scored_s": _total(iv) / FS, "estoi": estoi(y, ref, iv), "estoi_in": e_in, "sdr": sdr_fir(y, ref, iv),
             "sdr_in": sdr_in, "ta": _change(s_pre, s_in, mk), "level_out_db": _active_db(_concat(s_out, iv))}
        r.update(bin=_bin(r["dist_m"]), d_estoi=r["estoi"] - e_in)
        talkers[t] = r
    v = list(talkers.values())
    de = [r["d_estoi"] for r in v]
    m = {"n_talkers": float(len(v)), "n_talkers_skipped": float(len(skipped)),
         "estoi_mean": _nanmean([r["estoi"] for r in v]), "estoi_in_mean": _nanmean([r["estoi_in"] for r in v]),
         "d_estoi_mean": _nanmean(de), "estoi_wrst": _nanmin([r["estoi"] for r in v]),
         "estoi_wrst_in": _nanmin([r["estoi_in"] for r in v]), "d_estoi_min": _nanmin(de),
         "sdr_mean": _nanmean([r["sdr"] for r in v]), "sdr_in_mean": _nanmean([r["sdr_in"] for r in v])}
    m["d_estoi_wrst"] = m["estoi_wrst"] - m["estoi_wrst_in"]
    m["frac_estoi_ok"] = float(np.mean([x >= ESTOI_DROP_TOL for x in _finite(de)])) if _finite(de) else _NAN
    # T2 / T7 by distance bin
    for name, _, _ in DIST_BINS:
        rows = [r for r in v if r["bin"] == name]
        m[f"ta_med_{name}"] = _nanmedian([r["ta"] for r in rows])
        m[f"d_estoi_{name}"] = _nanmean([r["d_estoi"] for r in rows])
    m["ta_wrst"] = _nanmin([r["ta"] for r in v])
    present = [n for n, _, _ in DIST_BINS if np.isfinite(m[f"ta_med_{n}"])]
    m["t2_margin"] = m[f"ta_med_{present[-1]}"] - m[f"ta_med_{present[0]}"] if len(present) >= 2 else _NAN
    m["t2_ok"] = _crit(lambda x: x >= -LEVEL_TOL, m["t2_margin"])
    # T3 level evenness (post-AGC, separated speech)
    lv = np.array([r["level_out_db"] for r in v])
    med = float(np.median(lv)) if len(lv) else _NAN
    for r in v:
        r["level_rel_db"] = r["level_out_db"] - med
    m["lvl_std"] = float(np.std(lv)) if len(lv) else _NAN
    m["lvl_range"] = float(lv.max() - lv.min()) if len(lv) else _NAN
    m["lvl_soft_rel"] = _nanmean([r["level_rel_db"] for r in v if r["level_db"] <= SOFT_DB])
    # T1 capture rate (its WER part is pending)
    cap = [r["d_estoi"] >= ESTOI_DROP_TOL and abs(r["level_rel_db"]) <= LEVEL_TOL
           for r in v if np.isfinite(r["d_estoi"])]
    m["capture_rate"] = float(np.mean(cap)) if cap else _NAN
    # T4 noise, pumping across turns; T5 settling; per-turn level stability
    m.update(_noise_terms(ctx, C, ctx.n_act > 0))
    m["dsnr_active"] = m["na_active"] + _nanmean([r["ta"] for r in v])
    n_in, n_out = ctx.mic_a(C.groups("N")), C.f("N", "out")
    turn_lv, pump, settle = {}, [], []
    for t, a, b, returning in _turns(ctx):
        g = _change(n_out[a:b], n_in[a:b])
        if np.isfinite(g):
            pump.append(g)
        if t in talkers:
            if b - a >= MIN_SPEECH:
                turn_lv.setdefault(t, []).append(_active_db(s_out[a:b]))
            settle.append((returning, _settling(s_out, s_in, a, b)))
    m["turn_lvl_std"] = _nanmean([float(np.std(x)) for x in turn_lv.values() if len(x) >= 2])
    m["pump_std"] = float(np.std(pump)) if len(pump) >= 2 else _NAN
    m["pump_step_max"] = float(np.max(np.abs(np.diff(pump)))) if len(pump) >= 2 else _NAN
    s_all = _finite([s for _, s in settle])
    m["settle_med"] = float(np.median(s_all)) if s_all else _NAN
    m["settle_p90"] = float(np.percentile(s_all, 90)) if s_all else _NAN
    m["settle_first_med"] = _nanmedian([s for r, s in settle if not r])
    m["settle_return_med"] = _nanmedian([s for r, s in settle if r])
    # composite: even-capture rate (DESIGN 4.4)
    crit = {"even_c1": _crit(lambda c: c >= 1.0, m["capture_rate"]),
            "even_c2": _crit(lambda d: d >= -ESTOI_TIE, m["d_estoi_wrst"]), "even_c3": m["t2_ok"]}
    m.update(crit, even_pass=_all_pass(crit))
    return {"m": m, "legacy": _legacy_meeting(ctx, y, tids), "talkers": talkers, "skipped_talkers": skipped}


def _legacy_meeting(ctx, y, tids):
    """v1-style columns on the noisy output (report only): loudness std across talkers, gap SNR and SI-SDR
    improvement, per talker over the samples where the talker is alone."""
    sc = ctx.sc
    loud, dsi = [], []
    for t in tids:
        iv = ctx.alone(t)
        if _total(iv) < MIN_SPEECH:
            continue
        yt, rt = _concat(y, iv), _concat(sc["refs"][t], iv)
        loud.append(db(active_rms(yt)))
        dsi.append(v1.si_sdr(yt, rt) - ctx.memo(("si_in", t), lambda: v1.si_sdr(_concat(ctx.xa, iv), rt)))
    idle = _idle(ctx, ctx.n_act > 0)
    gap = float(np.median(loud) - db(rms(y[idle]))) if loud and idle.sum() >= MIN_SPEECH else _NAN
    return {"loud_std": float(np.std(loud)) if loud else _NAN, "gap_snr": gap, "dsisdr": _nanmean(dsi)}


# ----------------------------------------------------------------------------- pending external scorers
def score_wer(y, sc):
    """B8/T6 hook: per-talker WER with a small ONNX ASR in the scoring venv. Not available until the ASR
    model download is approved; returns None, and result files carry the 'pending' marker."""
    return None


def score_dnsmos(y, sc):
    """B9/T9 hook: DNSMOS P.835 (report only, never a tuning target). Not available until the model download
    is approved; returns None."""
    return None
