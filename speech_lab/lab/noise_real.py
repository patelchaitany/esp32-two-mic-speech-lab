"""Real recorded multichannel noise: DEMAND TBUS, PCAFETER and OMEETING (DESIGN.md 4.1 E6(a)).

DEMAND (J. Thiemann, N. Ito, E. Vincent, 2013; zenodo.org/records/1227121): 16 single-channel 16 kHz WAV
files per environment (chNN.wav, 300 s). Licence: see DATA.md (Zenodo lists CC BY 4.0, the bundled
DEMAND.pdf v1.0 says CC BY-SA 3.0); attribute the authors.

Array geometry. DEMAND.pdf v1.0 section 3.1 and Fig. 1: 16 microphones in 4 staggered rows, 5 cm from each
microphone to its immediate neighbours, in a horizontal plane (tripod, mics about 1.4 m above the floor),
built to +-2 mm. Channels are numbered row by row (1-4, 5-8, 9-12, 13-16); rows 2 and 4 are shifted by
half a spacing, so neighbouring rows are 5*sqrt(3)/2 = 4.33 cm apart and the largest distance is 21.8 cm.
design_xy() gives these positions. fit_geometry() recovers the geometry from the recordings themselves:
the real part of each pair's coherence in diffuse-dominated blocks is fitted with rho*sinc(2fd/c), the
120 fitted distances are compared with the layout (grid spacing fitted) and embedded by classical MDS.
FIT holds the result (python -m lab.noise_real --analyze); the 33 nearest-neighbour pairs (5 cm, closest
to our 4 cm device) are listed by pairs().

Channel gains. The electret capsules are not calibrated against each other (DEMAND.pdf 4.2). Per-channel
offsets (GAIN_DB, -1.4 to +1.9 dB) agree across the three environments (median 0.17 dB, max 0.76 dB) and
vary little over 125-4000 Hz (largest band-to-band spread of a channel 0.9 dB in PCAFETER, 2.3 dB in TBUS),
so they are microphone sensitivities; real_noise() removes them by default (equalize=True), because the
scene renderer adds its own E8 gain mismatch.

Levels are digital (dBFS = 20*log10(rms) re full scale 1.0): DEMAND is not SPL-calibrated. For scenes use a
'field' source, which sets the level in dB SPL (BENCH2.md 'Scene format'):
    {"type": "field", "level_db": 72.0, "weighting": "A",
     "signal": {"kind": "func", "func": "lab.noise_real:field", "params": {"env": "TBUS"}}}
(field_source() builds this dict). Note the pair carries the coherence of its own 5 cm spacing, whatever
spacing the scene geometry has.

API
    ENVS                                       ('TBUS', 'PCAFETER', 'OMEETING')
    load_env(env, start=0, stop=None, channels=None, equalize=False) -> float32 (C, n), default (16, 4800064)
    pairs(max_dist=0.06, min_dist=0.0) -> [{pair: (a, b), design_m, fitted_m, angle_deg}, ...]
    real_noise(env, n_samples, rng, pair=None, equalize=True, xfade_s=1.0, hp_hz=None, return_info=False)
        -> float32 (2, n) [, info]
    field(T, fs, rng, env='TBUS', pair=None, equalize=True, hp_hz=None) -> float64 (2, T), Mic A RMS 1
    field_source(env, level_db, pair=None, weighting='A', hp_hz=None) -> recipe source dict
    design_xy(spacing=0.05) -> (16, 2) m;  design_distances() -> (16, 16) m;  n_samples_of(env)
    fit_geometry(envs=ENVS, seconds=None) -> dict;  analyze(env, seconds=None) -> dict
    block_spectra / diffuse_blocks / coherence / fit_distance(s) / block_diffuseness / gain_table: analysis
CLI: python -m lab.noise_real --analyze [--seconds S] [--json PATH]   (about 1 min, all three environments)

Infrasound: most of the TBUS power is below 20 Hz (peak near 13 Hz), so Z-weighted levels describe
infrasound, not the audible noise; use A-weighted levels or hp_hz (DATA.md has the numbers).
"""
import argparse
import functools
import json
import os

import numpy as np
import soundfile as sf
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.signal import butter, sosfiltfilt, welch
from scipy.special import j0

from .common import C, FS

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIRS = {"TBUS": "data/demand_tbus/TBUS", "PCAFETER": "data/demand_pcafeter/PCAFETER",
            "OMEETING": "data/demand_omeeting/OMEETING"}
ENVS = tuple(ENV_DIRS)
N_CH = 16
SPACING = 0.05  # documented nearest-neighbour distance (DEMAND.pdf 3.1)

# Analysis settings (fit_geometry / analyze)
NFFT, HOP, BLOCK_S = 512, 256, 2.0
FIT_BAND = (150.0, 7000.0)       # Hz, coherence fit band
DIFFUSE_PCT, LEVEL_PCT = 30, 20  # diffuse blocks: mean MSC (1-6 kHz) <= p30 and level >= p20 (as S0 step 2)

# Per-channel sensitivity offsets in dB (channel level minus mean over channels), median over the three
# environments and the 125-4000 Hz octave bands. Produced by `python -m lab.noise_real --analyze`.
GAIN_DB = None  # filled by _install_constants() from _GAIN_DB_TABLE
# Geometry fit (fit_geometry on the full recordings of all three environments), see DATA.md:
# {'spacing_m': fitted grid spacing, 'dist_m': 16x16 fitted distances}. Filled from _FIT_UPPER.
FIT = None


# ----------------------------------------------------------------------------- geometry
def design_xy(spacing=SPACING):
    """(16, 2) documented mic positions in m: row r (0-3, channels 4r+1..4r+4), x = spacing*(c + (r%2)/2),
    y = -r*spacing*sqrt(3)/2. Channel 1 at the origin."""
    r, c = np.divmod(np.arange(N_CH), 4)
    return np.stack([spacing * (c + 0.5 * (r % 2)), -r * spacing * np.sqrt(3) / 2], axis=1)


def _pdist(xy):
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))


def design_distances(spacing=SPACING):
    """(16, 16) documented inter-mic distances in m (index = channel - 1)."""
    return _pdist(design_xy(spacing))


def _check_pair(pair):
    a, b = (int(v) for v in pair)
    if not (1 <= a <= N_CH and 1 <= b <= N_CH and a != b):
        raise ValueError(f"pair {pair}: need two different DEMAND channel numbers 1..16")
    return a, b


def pairs(max_dist=0.06, min_dist=0.0):
    """Channel pairs (1-based DEMAND channel numbers, a < b) with documented distance in [min_dist, max_dist] m.

    Each entry: pair (a, b), design_m (documented), fitted_m (coherence fit FIT['dist_m'], combined over the
    three environments), angle_deg (direction a->b in the array plane, 0 = along the rows).
    The default lists the 33 nearest-neighbour (5 cm) pairs: 12 along rows and 21 diagonal."""
    xy, D = design_xy(), design_distances()
    out = []
    for a in range(1, N_CH + 1):
        for b in range(a + 1, N_CH + 1):
            d = D[a - 1, b - 1]
            if min_dist - 1e-9 <= d <= max_dist + 1e-9:
                v = xy[b - 1] - xy[a - 1]
                out.append({"pair": (a, b), "design_m": round(float(d), 5),
                            "fitted_m": FIT["dist_m"][a - 1][b - 1] if FIT else None,
                            "angle_deg": round(float(np.degrees(np.arctan2(v[1], v[0]))), 1)})
    return sorted(out, key=lambda p: (p["design_m"], p["pair"]))


# ----------------------------------------------------------------------------- loading
def _env(env):
    e = str(env).upper()
    if e not in ENV_DIRS:
        raise ValueError(f"unknown DEMAND environment {env!r}; have {ENVS}")
    return e


def channel_path(env, ch):
    """Absolute path of channel `ch` (1..16) of an environment."""
    return os.path.join(LAB, ENV_DIRS[_env(env)], f"ch{int(ch):02d}.wav")


@functools.lru_cache(maxsize=8)
def n_samples_of(env):
    """Common length (samples) of the 16 channel files of an environment (4800064 = 300.004 s)."""
    ns = {sf.info(channel_path(env, c)).frames for c in range(1, N_CH + 1)}
    fss = {sf.info(channel_path(env, c)).samplerate for c in range(1, N_CH + 1)}
    if fss != {FS}:
        raise ValueError(f"{env}: sample rates {fss}")
    return min(ns)


def _gain(ch, equalize):
    return 10 ** (-GAIN_DB[ch - 1] / 20) if equalize else 1.0


def load_env(env, start=0, stop=None, channels=None, equalize=False):
    """float32 (C, n): channels (1-based list, default all 16 -> row k = channel k+1) of samples [start, stop).
    equalize=True removes the per-channel sensitivity offsets GAIN_DB."""
    env = _env(env)
    stop = n_samples_of(env) if stop is None else int(stop)
    if not 0 <= int(start) < stop <= n_samples_of(env):
        raise ValueError(f"{env}: bad sample range [{start}, {stop}) (length {n_samples_of(env)})")
    chans = list(range(1, N_CH + 1)) if channels is None else [int(c) for c in channels]
    out = np.empty((len(chans), stop - int(start)), np.float32)
    for k, c in enumerate(chans):
        x, fs = sf.read(channel_path(env, c), start=int(start), stop=stop, dtype="float32")
        if fs != FS:
            raise ValueError(f"{env} ch{c}: fs {fs}")
        out[k] = x * np.float32(_gain(c, equalize))
    return out


def real_noise(env, n_samples, rng, pair=None, equalize=True, xfade_s=1.0, hp_hz=None, return_info=False):
    """float32 (2, n_samples) of real noise: row 0 = channel pair[0] (Mic A), row 1 = pair[1] (Mic B).

    env: 'TBUS' | 'PCAFETER' | 'OMEETING' (case-insensitive). rng: numpy Generator or int seed; all random
    choices come from it in a fixed order (pair, order, offsets), so the output is deterministic.
    pair: (a, b) DEMAND channel numbers in any order, or None: one of the 33 nearest-neighbour (5 cm) pairs
    drawn from rng, in random order. The segment starts at a uniformly drawn offset. A request longer than
    the recording (300 s) chains random segments with xfade_s equal-power (sin/cos) crossfades, so the
    content repeats after about 5 min. Levels stay as recorded (dBFS), with channel sensitivities equalized
    unless equalize=False. hp_hz: optional zero-phase 4th-order Butterworth high-pass, identical on both rows
    (coherence unchanged), e.g. 20 Hz to drop the infrasound that dominates TBUS power (see DATA.md).
    return_info=True also returns {env, pair, segments [(offset, length)], design_m, fitted_m, gain_db}."""
    env = _env(env)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    if pair is None:
        nn = pairs()
        a, b = nn[int(rng.integers(len(nn)))]["pair"]
        if rng.random() < 0.5:
            a, b = b, a
    else:
        a, b = _check_pair(pair)
    n, N = int(n_samples), n_samples_of(env)
    if n <= 0:
        raise ValueError("n_samples must be positive")
    xf = max(1, int(round(xfade_s * FS)))
    out = np.zeros((2, n), np.float32)
    segs, pos = [], 0
    while pos < n:
        first = not segs
        L = min(N, n - pos + (0 if first else xf))
        off = int(rng.integers(0, N - L + 1))
        seg = load_env(env, off, off + L, (a, b), equalize)
        segs.append((off, L))
        if first:
            out[:, :L] = seg
            pos = L
        else:
            t = (np.arange(xf) + 0.5) / xf
            fo, fi = np.cos(0.5 * np.pi * t).astype(np.float32), np.sin(0.5 * np.pi * t).astype(np.float32)
            out[:, pos - xf:pos] = out[:, pos - xf:pos] * fo + seg[:, :xf] * fi
            out[:, pos:pos + L - xf] = seg[:, xf:]
            pos += L - xf
    if hp_hz:
        sos = butter(4, float(hp_hz), "highpass", fs=FS, output="sos")
        out = sosfiltfilt(sos, out, axis=-1).astype(np.float32)
    if not return_info:
        return out
    D = design_distances()
    info = {"env": env, "pair": (a, b), "segments": segs, "design_m": round(float(D[a - 1, b - 1]), 5),
            "fitted_m": FIT["dist_m"][a - 1][b - 1] if FIT else None,
            "gain_db": [float(-GAIN_DB[c - 1]) if equalize else 0.0 for c in (a, b)]}
    return out, info


def field(T, fs, rng, env="TBUS", pair=None, equalize=True, hp_hz=None):
    """scene2 'field' signal (kind 'func'): real_noise scaled so Mic A (row 0) has RMS 1, float64 (2, T).
    pair may be a JSON list. Deterministic given rng. The unit level includes any infrasound, so give the
    recipe source weighting 'A' (field_source does) or set hp_hz."""
    if int(fs) != FS:
        raise ValueError(f"fs {fs} != {FS}")
    x = real_noise(env, int(T), rng, None if pair is None else tuple(pair), equalize,
                   hp_hz=hp_hz).astype(np.float64)
    return x / (np.sqrt(np.mean(x[0] ** 2)) + 1e-20)


def field_source(env, level_db, pair=None, weighting="A", hp_hz=None):
    """Recipe source dict for scene2: a 'field' of real DEMAND noise at level_db (dB SPL, A-weighted by
    default) at Mic A. pair=None lets every scene draw its own 5 cm pair from the render rng."""
    params = {"env": _env(env)}
    if pair is not None:
        params["pair"] = list(_check_pair(pair))
    if hp_hz:
        params["hp_hz"] = float(hp_hz)
    return {"type": "field", "level_db": float(level_db), "weighting": weighting,
            "signal": {"kind": "func", "func": "lab.noise_real:field", "params": params}}


# ----------------------------------------------------------------------------- analysis
def _freqs():
    return np.fft.rfftfreq(NFFT, 1 / FS)


@functools.lru_cache(maxsize=3)
def block_spectra(env, seconds=None):
    """Cross-spectra of all 16 channels summed per BLOCK_S block (Hann, NFFT 512, hop 256, per-block mean
    removed), raw channel gains. Returns (S complex64 (B, 16, 16, F), level dB (B, 16)). Cached (3 entries)."""
    env = _env(env)
    stop = n_samples_of(env) if seconds is None else min(n_samples_of(env), int(seconds * FS))
    x = load_env(env, 0, stop)
    win = np.hanning(NFFT + 1)[:-1]
    nb = int(BLOCK_S * FS)
    B = x.shape[1] // nb
    nfr = 1 + (nb - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(nfr)[:, None]
    S = np.zeros((B, N_CH, N_CH, NFFT // 2 + 1), np.complex64)
    lev = np.zeros((B, N_CH))
    for b in range(B):
        seg = x[:, b * nb:(b + 1) * nb].astype(np.float64)
        seg -= seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg[:, idx] * win, axis=-1)
        S[b] = np.einsum("inf,jnf->ijf", X, X.conj())
        lev[b] = 10 * np.log10(np.mean(seg ** 2, axis=1) + 1e-30)
    S.flags.writeable = False
    return S, lev


def diffuse_blocks(S, lev, pct=DIFFUSE_PCT, level_pct=LEVEL_PCT):
    """Boolean mask of diffuse-dominated blocks: mean MSC over all 120 pairs in 1-6 kHz at or below its
    pct-th percentile, and mean channel level at or above its level_pct-th percentile (excludes blocks
    dominated by sensor self-noise). Same idea as S0 step 2 (DESIGN.md 2.0)."""
    f = _freqs()
    band = (f >= 1000) & (f <= 6000)
    iu = np.triu_indices(N_CH, 1)
    d = np.real(np.einsum("biif->bif", S[..., band]))
    msc = np.abs(S[:, iu[0], iu[1]][..., band]) ** 2 / (d[:, iu[0]] * d[:, iu[1]] + 1e-30)
    score = msc.mean(axis=(1, 2))
    L = lev.mean(axis=1)
    return (score <= np.percentile(score, pct)) & (L >= np.percentile(L, level_pct))


def coherence(S, sel=None):
    """Complex coherence (16, 16, F) of the blocks selected by mask `sel` (all if None)."""
    P = (S if sel is None else S[np.asarray(sel)]).sum(axis=0)
    d = np.real(np.einsum("iif->if", P))
    return P / np.sqrt(d[:, None, :] * d[None, :, :] + 1e-30)


def _model(fm, dgrid, model):
    arg = 2 * fm[None, :] * dgrid[:, None] / C
    return np.sinc(arg) if model == "sinc" else j0(np.pi * arg)


def fit_distance(g, model="sinc", band=FIT_BAND, dgrid=None):
    """Fit Re g(f) (one pair's coherence on the analysis grid) with rho*model(f; d): sinc(2fd/c) (3-D diffuse)
    or J0(2 pi f d/c) (2-D). Grid search over d (1-35 cm, 0.25 mm), rho in [0, 1] by least squares.
    Returns (d m, rho, rmse)."""
    f = _freqs()
    m = (f >= band[0]) & (f <= band[1])
    dgrid = np.arange(0.01, 0.35, 0.00025) if dgrid is None else dgrid
    M = _model(f[m], dgrid, model)
    y = np.real(g[m])
    rho = np.clip((M @ y) / (M ** 2).sum(axis=1), 0.0, 1.0)
    e = np.sqrt(np.mean((y[None, :] - rho[:, None] * M) ** 2, axis=1))
    k = int(np.argmin(e))
    return float(dgrid[k]), float(rho[k]), float(e[k])


def fit_distances(G, model="sinc"):
    """Fit all 120 pairs of a (16, 16, F) coherence: (D, rho, rmse), each (16, 16) symmetric."""
    D, R, E = (np.zeros((N_CH, N_CH)) for _ in range(3))
    for i in range(N_CH):
        for j in range(i + 1, N_CH):
            D[i, j], R[i, j], E[i, j] = fit_distance(G[i, j], model)
            D[j, i], R[j, i], E[j, i] = D[i, j], R[i, j], E[i, j]
    return D, R, E


def _mds(D, dim=2):
    n = len(D)
    J = np.eye(n) - 1.0 / n
    w, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    o = np.argsort(w)[::-1]
    return V[:, o[:dim]] * np.sqrt(np.maximum(w[o[:dim]], 0)), w[o]


def _align(P, Q):
    """Orthogonal Procrustes (rotation or reflection, no scaling) of points P onto Q; returns aligned P."""
    p0, q0 = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - p0).T @ (Q - q0))
    return (P - p0) @ (U @ Vt) + q0


def _blind_assignment(P, Q, steps=720):
    """Match MDS points P to layout positions Q without using channel labels: search rotations (and the
    reflection), Hungarian matching of nearest positions, refine with Procrustes. Returns (assignment
    channel->position index, rms position error m)."""
    p = P - P.mean(0)
    q0 = Q.mean(0)
    best = (np.inf, None)
    for refl in (1.0, -1.0):
        for th in np.linspace(0, 2 * np.pi, steps, endpoint=False):
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1.0, refl])
            A = p @ R.T + q0
            cost = ((A[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
            r, c = linear_sum_assignment(cost)
            tot = cost[r, c].sum()
            if tot < best[0]:
                best = (tot, c.copy())
    assign = best[1]
    Aa = _align(P, Q[assign])
    return assign, float(np.sqrt(np.mean(((Aa - Q[assign]) ** 2).sum(-1))))


def _grid_fit(D, W=None):
    """Fit the staggered layout to fitted distances D: one spacing s (least squares on all pairs), and a
    two-parameter version (column spacing a, row spacing b). Returns dict."""
    iu = np.triu_indices(N_CH, 1)
    U = design_distances(1.0)[iu]
    d = D[iu]
    w = np.ones_like(d) if W is None else W[iu]
    s = float(np.sum(w * d * U) / np.sum(w * U * U))
    r, c = np.divmod(np.arange(N_CH), 4)

    def resid(p):
        xy = np.stack([p[0] * (c + 0.5 * (r % 2)), -r * p[1]], axis=1)
        return np.sqrt(w) * (_pdist(xy)[iu] - d)

    ab = least_squares(resid, [s, s * np.sqrt(3) / 2]).x
    return {"spacing_m": round(s, 5), "col_spacing_m": round(float(ab[0]), 5),
            "row_spacing_m": round(float(ab[1]), 5),
            "row_spacing_design_m": round(float(SPACING * np.sqrt(3) / 2), 5)}


def fit_geometry(envs=ENVS, seconds=None, model="sinc"):
    """Recover the array geometry from the recordings (see module docstring).

    Per environment: coherence over diffuse_blocks, per-pair fit (fit_distances). Combined distance per pair:
    mean of the environments' fits weighted by 1/rmse^2. Then: grid fit (spacing), per-pair error against the
    fitted grid, MDS embedding aligned to the layout (per-channel position error), and a blind label-free
    matching of the MDS points to the layout positions, which confirms the channel numbering up to the
    layout's 180-degree symmetry."""
    iu = np.triu_indices(N_CH, 1)
    per_env, Ds, Ws = {}, [], []
    for env in envs:
        S, lev = block_spectra(env, seconds)
        sel = diffuse_blocks(S, lev)
        D, R, E = fit_distances(coherence(S, sel), model)
        per_env[_env(env)] = {"blocks": int(len(sel)), "diffuse_blocks": int(sel.sum()),
                              "rmse_mean": round(float(E[iu].mean()), 4), "rho_mean": round(float(R[iu].mean()), 3),
                              "D": D, "rmse": E}
        Ds.append(D)
        Ws.append(1.0 / np.maximum(E, 1e-3) ** 2)
    Ds, Ws = np.array(Ds), np.array(Ws)
    Dc = (Ds * Ws).sum(0) / Ws.sum(0)
    np.fill_diagonal(Dc, 0.0)
    grid = _grid_fit(Dc)
    Dg = design_distances(grid["spacing_m"])
    err = Dc - Dg
    nn = np.isclose(design_distances()[iu], SPACING, atol=1e-6)
    xy, ev = _mds(Dc)
    Q = design_xy(grid["spacing_m"])
    pos_err = np.sqrt(((_align(xy, Q) - Q) ** 2).sum(-1))
    assign, blind_rms = _blind_assignment(xy, Q)
    sym = np.array([N_CH - 1 - k for k in range(N_CH)])  # 180-degree rotation maps channel k to 17-k
    out = {
        "envs": list(per_env), "seconds": seconds, "model": model, "grid": grid,
        "dist_m": np.round(Dc, 4).tolist(),
        "err_vs_grid_cm": {"all_max": round(float(np.abs(err[iu]).max() * 100), 2),
                           "all_rms": round(float(np.sqrt(np.mean(err[iu] ** 2)) * 100), 2),
                           "nn_max": round(float(np.abs(err[iu][nn]).max() * 100), 2),
                           "nn_mean_fitted_cm": round(float(Dc[iu][nn].mean() * 100), 2),
                           "nn_range_cm": [round(float(Dc[iu][nn].min() * 100), 2),
                                           round(float(Dc[iu][nn].max() * 100), 2)]},
        "mds": {"eig_ratio_3_2": round(float(ev[2] / ev[1]), 4),
                "pos_err_cm_mean": round(float(pos_err.mean() * 100), 2),
                "pos_err_cm_max": round(float(pos_err.max() * 100), 2),
                "blind_assignment": [int(a) + 1 for a in assign],
                "blind_matches_docs": bool(np.array_equal(assign, np.arange(N_CH)) or np.array_equal(assign, sym)),
                "blind_rms_cm": round(blind_rms * 100, 2)},
        "per_env": {e: {k: v for k, v in d.items() if k not in ("D", "rmse")} | {
            "nn_mean_fitted_cm": round(float(d["D"][iu][nn].mean() * 100), 2),
            "max_err_vs_design_cm": round(float(np.abs(d["D"][iu] - design_distances()[iu]).max() * 100), 2)}
            for e, d in per_env.items()},
    }
    return out


def a_weight_db(f):
    """IEC 61672 A-weighting in dB (0 dB at 1 kHz)."""
    f2 = np.asarray(f, float) ** 2
    ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                   * (f2 + 12194.0 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def block_diffuseness(env, seconds=None, seed=0):
    """Short-term (BLOCK_S) diffuseness of the 33 nearest-neighbour pairs against a matched null.

    For every block and pair: rmse of Re(coherence) against sinc(2fd/c) at the pair's fitted distance over
    200-7000 Hz, and mean MSC over 2.5-6 kHz. Null: complex Gaussian STFT frames with the same per-frame,
    per-bin power (mean of the two channels) and exactly that sinc coherence, so the estimator spread caused
    by non-stationary levels is included (independent frames: a stationary white check gives the same median
    rmse as overlapped frames, 0.054 vs 0.055). Returns dict of arrays (B, 33): rmse, msc, rmse_null, msc_null."""
    env = _env(env)
    stop = n_samples_of(env) if seconds is None else min(n_samples_of(env), int(seconds * FS))
    x = load_env(env, 0, stop)
    rng = np.random.default_rng(seed)
    f = _freqs()
    m, hf = (f >= 200) & (f <= 7000), (f >= 2500) & (f <= 6000)
    nn = [p["pair"] for p in pairs()]
    I, J = np.array([a - 1 for a, _ in nn]), np.array([b - 1 for _, b in nn])
    s = np.sinc(2 * f[None, :] * np.array([FIT["dist_m"][i][j] for i, j in zip(I, J)])[:, None] / C)  # (33, F)
    win = np.hanning(NFFT + 1)[:-1]
    nb = int(BLOCK_S * FS)
    nfr = 1 + (nb - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(nfr)[:, None]

    def stats(X0, X1):  # (33, nfr, F) each
        P01 = np.sum(X0 * X1.conj(), axis=1)
        g = P01 / np.sqrt(np.sum(np.abs(X0) ** 2, 1) * np.sum(np.abs(X1) ** 2, 1) + 1e-30)
        return (np.sqrt(np.mean((np.real(g[:, m]) - s[:, m]) ** 2, axis=1)),
                np.mean(np.abs(g[:, hf]) ** 2, axis=1))

    out = {k: [] for k in ("rmse", "msc", "rmse_null", "msc_null")}
    for b in range(x.shape[1] // nb):
        seg = x[:, b * nb:(b + 1) * nb].astype(np.float64)
        seg -= seg.mean(axis=1, keepdims=True)
        X = np.fft.rfft(seg[:, idx] * win, axis=-1)            # (16, nfr, F)
        r, q = stats(X[I], X[J])
        pw = np.sqrt(0.5 * (np.abs(X[I]) ** 2 + np.abs(X[J]) ** 2))
        w0 = (rng.standard_normal(pw.shape) + 1j * rng.standard_normal(pw.shape)) / np.sqrt(2)
        w1 = (rng.standard_normal(pw.shape) + 1j * rng.standard_normal(pw.shape)) / np.sqrt(2)
        sb = s[:, None, :]
        r0, q0 = stats(pw * w0, pw * (sb * w0 + np.sqrt(np.maximum(1 - sb ** 2, 0)) * w1))
        for k, v in zip(out, (r, q, r0, q0)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def analyze(env, seconds=None):
    """Level, spectrum tilt and coherence-vs-diffuse-model statistics of one environment (dict).

    Levels (dBFS, raw channels, power-averaged over the 16 channels) from a Welch PSD (8192-point, about
    2 Hz bins): Z (all), A-weighted, and the 150-7000 Hz band used by the scene SNR rule; the share below
    20 Hz (infrasound); octave bands re total; tilt_db_per_oct = slope of the PSD over the 125-4000 Hz
    octave bands (pink noise -3, white 0). Block (2 s) percentiles of the 150-7000 Hz level.
    Coherence of the 33 nearest-neighbour (5 cm) pairs, all blocks (long term): rmse of Re(coherence)
    against sinc(2fd/c) and J0(2 pi f d/c) at the documented 5 cm per band, mean |Im| (a diffuse field has
    a real coherence), per-pair fits (d, rho, rmse); the 100-7000 Hz rmse also at each pair's fitted distance.
    Short term: block_diffuseness() against a matched ideal-diffuse null."""
    env = _env(env)
    S, lev = block_spectra(env, seconds)
    stop = n_samples_of(env) if seconds is None else min(n_samples_of(env), int(seconds * FS))
    x = load_env(env, 0, stop)
    fw, Pw = welch(x, fs=FS, nperseg=8192, axis=-1)
    Pw = Pw.mean(axis=0)
    del x
    tot = Pw[fw > 0].sum()
    level = 10 * np.log10(np.mean(10 ** (lev / 10)))              # time-domain power, all blocks/channels
    level_a = level + 10 * np.log10(np.sum(Pw * 10 ** (a_weight_db(np.maximum(fw, 1e-3)) / 10)) / tot)
    band = (fw >= 150) & (fw <= 7000)
    level_band = level + 10 * np.log10(Pw[band].sum() / tot)
    octs = [63, 125, 250, 500, 1000, 2000, 4000]
    band_db, dens_db = [], []
    for fc in octs:
        mm = (fw >= fc / np.sqrt(2)) & (fw < fc * np.sqrt(2))
        band_db.append(10 * np.log10(Pw[mm].sum() / tot))
        dens_db.append(10 * np.log10(Pw[mm].mean()))
    tilt = float(np.polyfit(np.log2(octs[1:]), np.array(dens_db[1:]), 1)[0])
    f = _freqs()
    iu = np.triu_indices(N_CH, 1)
    auto = np.real(np.einsum("biif->bif", S))                     # (B, 16, F)
    mb = (f >= 150) & (f <= 7000)
    frac = auto[:, :, mb].sum(-1).mean(-1) / auto.sum(-1).mean(-1)   # band share of each block's power
    blk_band = 10 * np.log10(np.mean(10 ** (lev / 10), axis=1) * frac)
    # channel offsets by octave band (consistency of sensitivities)
    psd_ch = auto.sum(axis=0)
    offs = []
    for fc in octs[1:]:
        mm = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        L = 10 * np.log10(psd_ch[:, mm].sum(axis=1))
        offs.append(L - L.mean())
    offs = np.array(offs)                                         # (6 bands, 16)
    # long-term coherence of the 5 cm pairs vs the diffuse models
    nn = [(a - 1, b - 1) for a, b in (p["pair"] for p in pairs())]
    G = coherence(S)
    s5, jj = np.sinc(2 * f * SPACING / C), j0(2 * np.pi * f * SPACING / C)
    coh_bands = {}
    for lo, hi in [(50, 200), (200, 500), (500, 2000), (2000, 4000), (4000, 7000), (100, 7000)]:
        mm = (f >= lo) & (f <= hi)
        re = np.array([np.real(G[i, j, mm]) for i, j in nn])
        coh_bands[f"{lo}-{hi}"] = {
            "rmse_sinc": round(float(np.sqrt(np.mean((re - s5[mm]) ** 2))), 3),
            "rmse_j0": round(float(np.sqrt(np.mean((re - jj[mm]) ** 2))), 3),
            "mean_re": round(float(re.mean()), 3), "mean_sinc": round(float(s5[mm].mean()), 3),
            "mean_abs_im": round(float(np.mean([np.abs(np.imag(G[i, j, mm])) for i, j in nn])), 3)}
    mm = (f >= 100) & (f <= 7000)
    sfit = np.array([np.sinc(2 * f[mm] * FIT["dist_m"][i][j] / C) for i, j in nn])
    re = np.array([np.real(G[i, j, mm]) for i, j in nn])
    coh_bands["100-7000"]["rmse_sinc_fitted_d"] = round(float(np.sqrt(np.mean((re - sfit) ** 2))), 3)
    fits = np.array([fit_distance(G[i, j], "sinc") for i, j in nn])
    fits_j = np.array([fit_distance(G[i, j], "j0") for i, j in nn])
    bd = block_diffuseness(env, seconds)
    p95_r = np.percentile(bd["rmse_null"], 95)
    p95_q = np.percentile(bd["msc_null"], 95)
    return {
        "env": env, "seconds": round(len(S) * BLOCK_S, 1),
        "level_dbfs_z": round(float(level), 2), "level_dbfs_a": round(float(level_a), 2),
        "level_dbfs_150_7000": round(float(level_band), 2),
        "below_20hz_share": round(float(Pw[(fw > 0) & (fw < 20)].sum() / tot), 3),
        "block_level_150_7000_p10_p50_p90": [round(float(np.percentile(blk_band, q)), 1) for q in (10, 50, 90)],
        "octave_band_db_re_total": {str(fc): round(float(v), 1) for fc, v in zip(octs, band_db)},
        "tilt_db_per_oct": round(tilt, 2),
        "channel_offset_db": [round(float(v), 2) for v in np.median(offs, axis=0)],
        "channel_offset_band_spread_db": round(float(np.max(offs.max(axis=0) - offs.min(axis=0))), 2),
        "coh5_bands": coh_bands,
        "coh5_fit_sinc": {"d_cm_mean": round(float(fits[:, 0].mean() * 100), 2),
                          "d_cm_range": [round(float(fits[:, 0].min() * 100), 2),
                                         round(float(fits[:, 0].max() * 100), 2)],
                          "rho_mean": round(float(fits[:, 1].mean()), 3),
                          "rmse_mean": round(float(fits[:, 2].mean()), 3)},
        "coh5_fit_j0": {"d_cm_mean": round(float(fits_j[:, 0].mean() * 100), 2),
                        "rho_mean": round(float(fits_j[:, 1].mean()), 3),
                        "rmse_mean": round(float(fits_j[:, 2].mean()), 3)},
        "block_rmse_p10_p50_p90": [round(float(np.percentile(bd["rmse"], q)), 3) for q in (10, 50, 90)],
        "block_rmse_null_p50_p95": [round(float(np.median(bd["rmse_null"])), 3), round(float(p95_r), 3)],
        # share of (block, pair) cases beyond the matched null's 95th percentile (0.05 if ideally diffuse)
        "frac_non_diffuse_rmse": round(float(np.mean(bd["rmse"] > p95_r)), 3),
        "block_msc_2500_6000_p10_p50_p90": [round(float(np.percentile(bd["msc"], q)), 3) for q in (10, 50, 90)],
        "block_msc_null_p50_p95": [round(float(np.median(bd["msc_null"])), 3), round(float(p95_q), 3)],
        "frac_non_diffuse_msc": round(float(np.mean(bd["msc"] > p95_q)), 3),
    }


def gain_table(envs=ENVS, seconds=None):
    """Per-channel sensitivity offsets (dB): median over environments and 125-4000 Hz octave bands."""
    offs = []
    for env in envs:
        S, _ = block_spectra(env, seconds)
        f = _freqs()
        psd_ch = np.real(np.einsum("biif->if", S))
        for fc in (125, 250, 500, 1000, 2000, 4000):
            m = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
            L = 10 * np.log10(psd_ch[:, m].sum(axis=1))
            offs.append(L - L.mean())
    return np.round(np.median(np.array(offs), axis=0), 2)


# ----------------------------------------------------------------------------- fitted constants
# Both produced by `python -m lab.noise_real --analyze` on the full 300 s of all three environments
# (2026-11-02); tests/test_noise_real.py re-fits on 120 s and checks agreement.
# GAIN_DB: gain_table(), channel k at index k-1 (dB re the 16-channel mean).
_GAIN_DB_TABLE = [-0.65, 0.10, 0.73, -0.42, 1.89, 0.25, -0.15, 0.21, 1.70, -1.42, -1.15, -0.33, 0.29, -0.14,
                  -0.51, -0.49]
# fit_geometry()['dist_m'] upper triangle (i < j, row-major, channels 1..16), m; grid spacing of the fit.
_FIT_SPACING_M = 0.05081
_FIT_UPPER = [
    0.0529, 0.1020, 0.1551, 0.0500, 0.0915, 0.1386, 0.1869, 0.0927, 0.1028, 0.1330, 0.1788, 0.1359, 0.1506, 0.1838,
    0.2230, 0.0525, 0.1035, 0.0480, 0.0525, 0.0915, 0.1396, 0.1003, 0.0914, 0.1022, 0.1331, 0.1379, 0.1366, 0.1503,
    0.1816, 0.0546, 0.0865, 0.0469, 0.0499, 0.0911, 0.1293, 0.0983, 0.0897, 0.0996, 0.1527, 0.1357, 0.1347, 0.1501,
    0.1338, 0.0856, 0.0476, 0.0509, 0.1626, 0.1316, 0.0987, 0.0879, 0.1847, 0.1558, 0.1357, 0.1368, 0.0555, 0.1029,
    0.1558, 0.0526, 0.0535, 0.0939, 0.1411, 0.0906, 0.1026, 0.1328, 0.1780, 0.0543, 0.1027, 0.0877, 0.0526, 0.0533,
    0.0920, 0.1000, 0.0906, 0.1012, 0.1326, 0.0538, 0.1277, 0.0871, 0.0499, 0.0517, 0.1308, 0.0998, 0.0907, 0.1011,
    0.1725, 0.1344, 0.0863, 0.0483, 0.1627, 0.1318, 0.0999, 0.0886, 0.0542, 0.1025, 0.1545, 0.0485, 0.0890, 0.1387,
    0.1840, 0.0545, 0.1020, 0.0494, 0.0495, 0.0885, 0.1370, 0.0548, 0.0885, 0.0484, 0.0505, 0.0887, 0.1306, 0.0845,
    0.0516, 0.0496, 0.0541, 0.1017, 0.1526, 0.0525, 0.1013, 0.0536]


def _install_constants():
    global GAIN_DB, FIT
    GAIN_DB = np.asarray(_GAIN_DB_TABLE, float)
    D = np.zeros((N_CH, N_CH))
    D[np.triu_indices(N_CH, 1)] = _FIT_UPPER
    D = D + D.T
    FIT = {"spacing_m": _FIT_SPACING_M, "dist_m": D.round(4).tolist()}


_install_constants()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DEMAND real-noise loader: geometry fit and environment report")
    ap.add_argument("--analyze", action="store_true", help="fit the geometry and analyze TBUS/PCAFETER/OMEETING")
    ap.add_argument("--seconds", type=float, default=None, help="use only the first S seconds of each recording")
    ap.add_argument("--json", default=None, help="also write the report to this JSON path")
    a = ap.parse_args()
    if a.analyze:
        rep = {"gain_db": gain_table(seconds=a.seconds).tolist(), "geometry": fit_geometry(seconds=a.seconds),
               "envs": {e: analyze(e, a.seconds) for e in ENVS}}
        print(json.dumps({k: v for k, v in rep["geometry"].items() if k != "dist_m"}, indent=1))
        print(json.dumps(rep["envs"], indent=1))
        print("gain_db", rep["gain_db"])
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(rep, fh, indent=1)
