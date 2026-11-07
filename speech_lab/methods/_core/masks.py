"""M4 CDR coherence post-filter and M5 two-hypothesis phase fence (DESIGN.md M4, M5, 2.1).

Both estimate real per-bin gains G (257, N) in [floor, 1] from the calibrated two-channel STFT X
(S0 output: X_B corrected); M5, and M4 when given hypotheses, also use per-frame TDOA hypotheses from S1.
A stack multiplies them with its other masks and applies the product ONCE to its linear signal; the
DESIGN 2.1 time rule (instant rise, fall <= 4.8 dB per 8 ms hop) belongs to that combined mask:
ns.limit_gain(m, None, FALL_DB).

M4, Schwarz & Kellermann 2015 [34]. Per bin and frame, with Phi smoothed recursively (lam):
    Gx = Phi_AB / sqrt(Phi_AA Phi_BB), |Gx| <= 0.999;  Gn = S0 gamma_n, clipped to <= gn_max = 0.98 for
        prop1/prop2 (DESIGN: 0.95, which biases prop1 upward in conditioned bins whose true Gn exceeds
        0.95) and to <= 0.95 for prop3 (common.cdr_nodoa; see cdr_estimate)
    hypotheses tau_k (Gs = exp(+j 2 pi f tau_k)), prop1: CDR_k = max(0, Re{Gs*(Gn - Gx)} / (Re{Gs* Gx} - 1))
        (option 'prop2': the robust variant, eq. 20-21); CDR = max over the frame's hypotheses;
    no hypothesis (taus None, or NaN in that frame): the DOA-free prop3 (eq. 25);
    conditioning, cond > 0: bins with |Gs - Gn| < cond (DOA-free: 1 - Gn < cond) take the frame's
        fill_pct-th percentile of the conditioned bins in 1-4 kHz, per hypothesis; cond 0 = none;
    CDR_out = CDR * DF_beam (beam correction [35]); G = max(G_min, 1 - sqrt(mu / (CDR_out + 1)));
    3-bin frequency median.
M5: e_k = wrap(arg Phi_AB - 2 pi f tau_k), Phi_AB smoothed with lam 0.7;
    eta = max_k 1 / (1 + gamma e_k^2) for f in [1 kHz, min(4 kHz, 0.8 c / 2 d_hat)]; outside that band
    1 ('one') or the frame's band median of eta ('median', the DESIGN bus rule); frames without a
    hypothesis get 1; G = max(floor, eta); 3-bin frequency median.

Defaults CDR_PARAMS / FENCE_PARAMS were tuned on the v2 dev bus set (TTS, 90 scenes; evidence in
tmp/tune:masks). The low-level functions (cdr_estimate, fence_mask) keep the DESIGN values as their
signature defaults. Deviations from the DESIGN starting points, each measured on dev:
1. M4 conditioning off (cond 0). With cond 0.2 and the frame-median fill, every partner lost 6-7 dB
   (worst-partner TA -7.5 dB at mu 0.5): speech is sparse in 1-4 kHz, so the frame median is a noise
   statistic and the fill gates the partners' low band with the noise. Raw estimates in ill-conditioned
   bins keep the partners (TA -2.5 dB) and give the best worst-partner ESTOI; a high fill percentile
   (90-100) or passing ill-conditioned bins unchanged was worse on ESTOI.
2. M4 gain law mu 0.5, lam 0.8 (DESIGN start 1.3, 0.9): mu 1.3 cut the partners by 14 dB and lowered
   ESTOI; lam 0.8 beat 0.68, 0.9 and 0.95. G_min 0.15 is kept but is inactive: at mu 0.5 the law's own
   minimum is 1 - sqrt(0.5) = 0.29 (-10.7 dB). One set for all spacings (25/40/60 mm had the same optimum;
   S0's spacing falls back to the 40 mm prior on most 60 mm arrays, so a d_hat-keyed set could not be
   selected reliably anyway).
3. M5 outside rule 'one' on the bus as well (DESIGN: band median): the band median of a sparse-speech
   frame is low, so 'median' gated the partners' whole spectrum (TA -3.6 dB, worst-partner dESTOI
   -0.015 against M0). gamma 16 with floor 0.5 (DESIGN 4 / 0.3): a narrow fence that never cuts by more
   than 6 dB was the least harmful setting with S1's hypotheses (gamma 4 / floor 0.3: -0.013).
Not changed here, but decisive: the S1 partner TDOAs are often wrong on v2 (both partners within 20 us in
7 of 30 dev scenes at 0 dB). The DOA-dependent M4 and M5 then attenuate the partners, so m4_cdr runs
DOA-free on the bus and M5 stays slightly below M0 in worst-partner ESTOI; with the true TDOAs (a
diagnostic) both would help (see the m4_cdr / m5_fence docstrings).

Stack use (bus):
    fe = tracker.front_end(x, fs, info); cal = fe.schedule or fe.calib
    G_cdr = cdr_gain(fe.X, cal, None, df=beam.df(cal))            # DOA-free (recommended with today's S1)
    G_fen = fence_gain(fe.X, cal, partner_taus(fe.track, cal))     # (2, N) hypotheses, NaN where inseparable
    m = ns.limit_gain(np.maximum(G_cdr * G_fen * G_ns, m_floor), None, FALL_DB); Y = m * Y_lin
Tests: `.venv/bin/python -m tests.test_masks` (Appendix B.2, B.4).
"""
import numpy as np
from scipy.ndimage import median_filter

from lab.common import C
from . import calib as cb
from . import common as cm
from . import tracker as tr

CDR_PARAMS = dict(            # tuned on v2 dev (module docstring); DESIGN start values in comments
    lam=0.8,                  # smoothing of Phi (DESIGN 0.9; sweep 0.68 / 0.8 / 0.9)
    mu=0.5,                   # spectral-subtraction overestimation (DESIGN 1.3; sweep 0.5 ... 2.0)
    g_min=0.15,               # gain floor (sweep 0.1 / 0.15 / 0.3); inactive while 1 - sqrt(mu) > g_min
    cond=0.0,                 # conditioning |Gs - Gn| >= cond (DESIGN 0.2); 0 = raw estimate in every bin
    gn_max=0.98,              # Gn clip (DESIGN 0.95; 0.98 = S0's own clip, see module docstring)
    fill_band=(1000.0, 4000.0),
    fill_pct=50.0,            # fill statistic (cond > 0): percentile of the conditioned bins (50 = DESIGN)
    estimator="prop1",        # DOA-dependent estimator: 'prop1' (DESIGN) or 'prop2' (robust variant)
    median_bins=3,            # frequency median width (1 = off)
)
FENCE_PARAMS = dict(          # tuned on v2 dev (module docstring); DESIGN start values in comments
    lam=0.7,                  # smoothing of Phi_AB
    gamma=16.0,               # rad^-2 (DESIGN 4; sweep 1 / 4 / 16)
    floor=0.5,                # (DESIGN 0.3)
    band=(1000.0, 4000.0),
    alias_frac=0.8,           # upper band edge <= alias_frac * c / (2 d_hat)
    outside="one",            # bins outside the band: 'one' or 'median' (DESIGN bus rule)
    median_bins=3,
)
FALL_DB = 4.8                 # DESIGN 2.1: max fall of the combined mask per 8 ms hop
GX_MAX = 0.999                # |Gx| clip (as common.cdr_doa / cdr_nodoa)


def _params(defaults, overrides):
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    return dict(defaults, **overrides)


def _blocks(cal, n):
    """A Calib or an online schedule [(start_frame, Calib), ...] -> [(start, end, Calib)]."""
    if isinstance(cal, cb.Calib):
        return [(0, n, cal)]
    starts = [s for s, _ in cal] + [n]
    return [(s, e, c) for (s, c), e in zip(cal, starts[1:]) if e > s]


def _stream(X, cal, lam):
    """Yield (a, b, Calib, (paa, pbb, pab)) over frames [a, b): chunks of the recursively smoothed
    spectra (lam) split at the calibration blocks."""
    blocks = _blocks(cal, X.shape[-1])
    for l0, l1, sm in cm.smoothed_spectra(X, slice(None), (lam,)):
        for s, e, c in blocks:
            a, b = max(s, l0), min(e, l1)
            if a < b:
                yield a, b, c, tuple(v[:, a - l0:b - l0] for v in sm[lam])


def _taus(taus, n):
    """None, (K,) constants, (K, 1) or (K, N) per frame -> None or a (K, N) float array."""
    if taus is None:
        return None
    t = np.asarray(taus, dtype=float)
    return np.broadcast_to(t[:, None] if t.ndim == 1 else t, (t.shape[0], n))


def _freq_median(G, n):
    return G if n <= 1 else median_filter(G, size=(n, 1), mode="nearest")


# ------------------------------------------------------------------ hypotheses from S1
def partner_taus(trk, cal, fallback=True):
    """Bus hypotheses (2, N): S1's per-frame partner TDOAs [tau_A_t, tau_B_t] in s.
    fallback=True (DESIGN 2.1): frames where the partners are spatially inseparable (one peak, peaks
    < 25 us apart, or both |tau_k| < 0.3 d_hat/c, the S1 rule with the frame's d_hat) are NaN, so M4
    uses its DOA-free estimator and M5 is off there. All NaN if S1 ran no partner model."""
    n = len(trk.tau)
    if trk.tau_A_t is None:
        return np.full((2, n), np.nan)
    H = np.stack([trk.tau_A_t, trk.tau_B_t]).astype(float)
    if fallback:
        d = np.empty(n)
        for s, e, c in _blocks(cal, n):
            d[s:e] = c.d_hat
        ta, tb = H
        with np.errstate(invalid="ignore"):
            ok = (ta - tb >= tr.PEAK_SEP) & (np.maximum(np.abs(ta), np.abs(tb)) >= tr.BROADSIDE_FRAC * d / C)
        H[:, ~ok] = np.nan
    return H


def talker_taus(trk):
    """Meeting hypothesis (1, N): TDOA of the frame's S1 talker cluster, held through pauses; NaN
    before the first labeled frame (or without clusters)."""
    n = len(trk.tau)
    if trk.cluster is None or not trk.clusters:
        return np.full((1, n), np.nan)
    ids = np.array([c["id"] for c in trk.clusters])
    tau_of = np.full(ids.max() + 1, np.nan)
    tau_of[ids] = [c["tau"] for c in trk.clusters]
    last = np.maximum.accumulate(np.where(trk.cluster >= 0, np.arange(n), -1))
    held = np.where(last >= 0, trk.cluster[np.maximum(last, 0)], -1)
    return np.where(held >= 0, tau_of[np.maximum(held, 0)], np.nan)[None, :]


# ------------------------------------------------------------------ M4: CDR estimation and gain
def _clip_mag(G, m):
    a = np.abs(G)
    return np.where(a > m, G * (m / np.maximum(a, cm.EPS)), G)


def cdr_prop1(Gx, Gn, Gs):
    """DOA-dependent CDR ([34] prop1, eq. 19), linear: max(0, Re{Gs*(Gn - Gx)} / (Re{Gs* Gx} - 1)),
    with |Gx| <= 0.999. Gn is used as given (common.cdr_doa is the same with Gn clipped to 0.95)."""
    Gx = _clip_mag(Gx, GX_MAX)
    c = np.conj(Gs)
    return np.maximum(0.0, np.real(c * (Gn - Gx)) / np.minimum(np.real(c * Gx) - 1.0, -1e-6))


def cdr_prop2(Gx, Gn, Gs):
    """Robust DOA-dependent CDR ([34] prop2 with its bias compensation, eq. 20-21), linear >= 0:
    |Gs*(Gn - Gx)| / (1 - Re{Gs* Gx}) * (1 - Gn Re{Gs}) / |Gn - Gs|, with |Gx| <= 0.999."""
    Gx = _clip_mag(Gx, GX_MAX)
    c = np.conj(Gs)
    raw = np.abs(c * (Gn - Gx)) / np.maximum(1.0 - np.real(c * Gx), 1e-6)
    return raw * (1.0 - Gn * np.real(Gs)) / np.maximum(np.abs(Gn - Gs), 1e-6)


def _col_pct(a, m, q):
    """Per-column q-th percentile (linear interpolation, as np.percentile) of a (F, L) over the rows
    where m (F, L) is True; NaN where there are none."""
    s = np.sort(np.where(m, a, np.inf), axis=0)
    n = m.sum(axis=0)
    pos = np.maximum(n - 1, 0) * (q / 100.0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    cols = np.arange(a.shape[1])
    with np.errstate(invalid="ignore"):
        v = s[lo, cols] + (pos - lo) * (s[hi, cols] - s[lo, cols])
    return np.where(n > 0, v, np.nan)


def fill_unconditioned(cdr, ok, band, pct=50.0):
    """Replace ill-conditioned bins (ok False) by the frame's pct-th percentile (50: the DESIGN median)
    of the conditioned bins in `band` ((F,) bool); of all conditioned bins if `band` has none; by +inf
    (no attenuation) if the frame has no conditioned bin at all."""
    fill = _col_pct(cdr, ok & band[:, None], pct)
    miss = np.isnan(fill)
    if miss.any():
        fill[miss] = _col_pct(cdr[:, miss], ok[:, miss], pct)
    return np.where(ok, cdr, np.where(np.isnan(fill), np.inf, fill)[None, :])


def cdr_estimate(Gx, gn, taus=None, f=cm.FREQS, cond=0.2, gn_max=0.98, fill_band=(1000.0, 4000.0),
                 fill_pct=50.0, estimator="prop1"):
    """Conditioned CDR (F, L), linear, from the complex coherence Gx (F, L) and the real noise coherence
    gn (F,), clipped to <= gn_max. taus: None or TDOA hypotheses (s), (K,), (K, 1) or (K, L); per frame
    the CDR is the max over its finite hypotheses (prop1 / prop2), or the DOA-free prop3 where none is
    finite. Ill-conditioned bins: fill_unconditioned(..., fill_band, fill_pct) per hypothesis (cond 0:
    none). prop3 is common.cdr_nodoa, which clips Gn to 0.95 (DESIGN value) whatever gn_max is: with cond 0
    that clip acts below about 0.5-1.2 kHz, where Gn exceeds 0.95; on v2 dev a 0.98 clip there cost 0.6 dB
    of partner level at equal ESTOI."""
    doa_cdr = {"prop1": cdr_prop1, "prop2": cdr_prop2}[estimator]
    taus = _taus(taus, Gx.shape[1])
    gnc = np.minimum(gn, gn_max)[:, None]
    band = (f >= fill_band[0]) & (f <= fill_band[1])
    out = np.zeros(Gx.shape)
    doa = np.zeros(Gx.shape[1], bool) if taus is None else np.isfinite(taus).any(axis=0)
    if not doa.all():
        g = Gx[:, ~doa]
        ok = np.broadcast_to(1.0 - gnc >= cond, g.shape)
        out[:, ~doa] = fill_unconditioned(cm.cdr_nodoa(g, gnc), ok, band, fill_pct)
    for t in ([] if taus is None else taus):
        cols = np.isfinite(t)
        if not cols.any():
            continue
        g = Gx[:, cols]
        Gs = cm.plane_coherence(f[:, None], t[None, cols])
        ck = fill_unconditioned(doa_cdr(g, gnc, Gs), np.abs(Gs - gnc) >= cond, band, fill_pct)
        out[:, cols] = np.maximum(out[:, cols], ck)
    return out


def gain_from_cdr(cdr, mu=1.3, g_min=0.15, df=None):
    """Spectral-subtraction gain max(G_min, 1 - sqrt(mu / (CDR * DF + 1))); df: beam DF (F,) or
    (F, L), 1/(w^H Gamma_n w) of the weights in use (None = 1, a single mic)."""
    c = cdr if df is None else cdr * df
    return np.maximum(g_min, 1.0 - np.sqrt(mu / (c + 1.0)))


def _cdr_blocks(X, cal, taus, p):
    """Yield (a, b, CDR) per chunk x calibration block (CDR before the beam correction)."""
    taus = _taus(taus, X.shape[-1])
    for a, b, c, sm in _stream(X, cal, p["lam"]):
        yield a, b, cdr_estimate(cm.coherence(*sm), c.gamma_n, None if taus is None else taus[:, a:b],
                                 cm.FREQS, p["cond"], p["gn_max"], p["fill_band"], p["fill_pct"], p["estimator"])


def cdr_map(X, cal, taus=None, **overrides):
    """CDR (257, N), linear, before the beam correction and the gain (diagnostics, M6 cues).
    Arguments as cdr_gain()."""
    p = _params(CDR_PARAMS, overrides)
    out = np.empty(X.shape[1:])
    for a, b, cdr in _cdr_blocks(X, cal, taus, p):
        out[:, a:b] = cdr
    return out


def cdr_gain(X, cal, taus=None, df=None, **overrides):
    """M4 gain G_cdr (257, N) in [g_min, 1].
    X: calibrated STFT (2, 257, N) (S0 applied); cal: its Calib or online schedule (gamma_n, d_hat);
    taus: None (DOA-free everywhere), (K,) constants or (K, N) per-frame hypotheses in s with NaN where
    unusable (bus: partner_taus()); df: beam directivity factor (257,) or (257, N), None = 1;
    overrides: CDR_PARAMS keys. No time smoothing (apply FALL_DB to the combined mask)."""
    p = _params(CDR_PARAMS, overrides)
    df = None if df is None else np.asarray(df, dtype=float)
    G = np.empty(X.shape[1:])
    for a, b, cdr in _cdr_blocks(X, cal, taus, p):
        dfb = None if df is None else (df[:, None] if df.ndim == 1 else df[:, a:b])
        G[:, a:b] = gain_from_cdr(cdr, p["mu"], p["g_min"], dfb)
    return _freq_median(G, p["median_bins"])


# ------------------------------------------------------------------ M5: phase fence
def fence_band(d_hat, band=(1000.0, 4000.0), alias_frac=0.8, f=cm.FREQS):
    """(F,) bool: bins with band[0] <= f <= min(band[1], alias_frac * c / (2 d_hat))."""
    return (f >= band[0]) & (f <= min(band[1], alias_frac * C / (2.0 * d_hat)))


def fence_mask(pab, taus, d_hat, f=cm.FREQS, gamma=4.0, band=(1000.0, 4000.0), alias_frac=0.8,
               outside="median"):
    """eta (F, L) from the smoothed cross-spectrum pab (F, L) and hypotheses taus (K, L) (NaN = none):
    max_k 1 / (1 + gamma e_k^2) with e_k = wrap(arg pab - 2 pi f tau_k) inside fence_band(); outside
    it the frame's band median ('median') or 1 ('one'); 1 in frames without a finite hypothesis."""
    inb = fence_band(d_hat, band, alias_frac, f)
    pb, fb = pab[inb], f[inb][:, None]
    eta = np.full(pb.shape, np.nan)
    for t in taus:
        e = np.angle(pb * np.exp(-2j * np.pi * fb * t[None, :]))     # NaN where t is NaN
        eta = np.fmax(eta, 1.0 / (1.0 + gamma * e * e))
    has = np.isfinite(taus).any(axis=0)
    out = np.ones(pab.shape)
    out[np.ix_(inb, has)] = eta[:, has]
    if outside == "median":
        out[np.ix_(~inb, has)] = np.median(eta[:, has], axis=0)[None, :]
    elif outside != "one":
        raise ValueError(f"unknown outside rule {outside!r}")
    return out


def fence_gain(X, cal, taus, **overrides):
    """M5 gain G_fence (257, N) in [floor, 1]. X, cal as cdr_gain(); taus: (K,) or (K, N) hypotheses
    in s, NaN = none (bus: partner_taus(), meeting: talker_taus()); overrides: FENCE_PARAMS keys
    (outside='one' for the meeting). No time smoothing (apply FALL_DB to the combined mask)."""
    p = _params(FENCE_PARAMS, overrides)
    taus = _taus(taus, X.shape[-1])
    if taus is None:
        taus = np.full((1, X.shape[-1]), np.nan)
    G = np.empty(X.shape[1:])
    for a, b, c, (_, _, pab) in _stream(X, cal, p["lam"]):
        G[:, a:b] = fence_mask(pab, taus[:, a:b], c.d_hat, cm.FREQS, p["gamma"], p["band"],
                               p["alias_frac"], p["outside"])
    return _freq_median(np.maximum(G, p["floor"]), p["median_bins"])
