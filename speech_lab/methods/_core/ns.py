"""M6: spatially gated MMSE-SPP noise tracking + LSA noise suppression (DESIGN.md M6, 2.1, 2.2).

Noise PSD (Gerkmann & Hendriks 2012, [43]), per bin k and frame l, with |Y|^2 the periodogram:
    P     = 1 / (1 + (1 + xi_H1) * exp(-(|Y|^2 / s2(l-1)) * xi_H1 / (1 + xi_H1)))
    Pbar  = alpha_p * Pbar + (1 - alpha_p) * P;  if Pbar > 0.99: P = min(P, 0.99)   (stagnation guard)
    s2(l) = alpha * s2(l-1) + (1 - alpha) * ((1 - P) * |Y|^2 + P * s2(l-1))       (per 8 ms hop)
Gain (Ephraim & Malah LSA, [44]), decision-directed a-priori SNR:
    gamma = |Y|^2 / s2(l);  xi = max(alpha_dd * Gr(l-1)^2 * gamma(l-1) + (1 - alpha_dd) * max(gamma - 1, 0), xi_min)
    Gr    = xi / (1 + xi) * exp(E1(xi * gamma / (1 + xi)) / 2)       (Gr unclipped in the recursion)
    G     = min(Gr, 1), floored at -15 dB (bus) / -12 dB (meeting); instant rise, fall <= 4.8 dB per hop.
In a stack whose masks are combined once (DESIGN 2.1), take the raw gain with floor_db=None,
fall_db=None and apply limit_gain() to the combined mask.

Tuned values (v2 dev set, tune:ns; DESIGN value in brackets): xi_H1 5 dB [15], alpha 0.97 [0.9],
alpha_dd 0.7 [0.98]; floors, alpha_p, fall limit as DESIGN. With the DESIGN values the tracker absorbs
low-band speech at bus SNRs and the slow decision-directed recursion holds speech gains down: partners
lost 6-7 dB (TA_min -7.2 dB), failing G0. The tuned tracker settles at about -2 dB re stationary
Gaussian noise ([-0.9 dB]) and follows a +10 dB noise rise in about 3 s ([0.4 s]); falls in about 1 s.
DESIGN_PARAMS restores the DESIGN set (ablations).

Gating (optional) freezes the noise update while the wanted speech is present:
    gate="hold"  (bus default): where the target cue (per frame or per bin, e.g. band_mean(G_cdr *
                 G_fence)) exceeds 0.6, hold s2 in bins >= 500 Hz; lower bins always update (engine).
    gate="pause" (DESIGN meeting rule; not the default): update only once a speech pause has lasted
                 300 ms. The VAD is the given per-frame speech probability (> 0.5 = speech), else an
                 internal one: the SPP of an ungated tracker with xi_H1 = vad_xi_h1_db (15 dB, the
                 value vad_thr is calibrated for) averaged over 300-4000 Hz, > 0.25 = speech.
    gate="none"  (meeting default): plain tracker. On the 20 dev meetings the tuned plain tracker beat
                 the pause gate (mean dESTOI +0.005, NA_active +0.7 dB, NM -0.2 dB, pumping -0.1 dB),
                 and an oracle-VAD pause gate was no better than the internal one.
Any hold run longer than hold_max_s (5 s) is released until the cue drops, so a stuck cue cannot
freeze the estimate (with a stuck cue and a +10 dB noise rise the frozen estimate stayed 10 dB low).
Without a cue the plain tracker runs. Works on any single-channel STFT laid out like lab.dsp.stft
(F x N, hop 128 at 16 kHz); the first `lead_frames` frames, which overlap lab.dsp.stft's leading zero
pad, keep the initial estimate. Everything is causal except the initial noise PSD, which looks ahead
init_s = 2 s (online variant: init_s=0.1).
"""
import numpy as np
from scipy.special import exp1

FS = 16000
N_FFT = 512
HOP = 128
_TINY = 1e-30

PARAMS = dict(
    xi_h1_db=5.0,       # a-priori SNR of the SPP model's speech hypothesis ([43] and DESIGN: 15 dB)
    alpha=0.97,         # noise PSD smoothing per 8 ms hop (DESIGN: 0.9)
    alpha_p=0.9,        # SPP smoothing of the stagnation guard (DESIGN; 0.95 measured identical)
    p_max=0.99,         # stagnation guard cap
    alpha_dd=0.7,       # decision-directed smoothing (DESIGN: 0.98)
    xi_min_db=-25.0,
    floor_db=-15.0,     # gain floor; None = no floor
    fall_db=4.8,        # max gain fall per hop in dB; None = unlimited
    rise_db=None,       # max gain rise per hop in dB; None = instant
    gate="hold",        # "hold" | "pause" | "none"
    gate_thr=0.6,       # cue threshold: target present (hold) / speech (pause, external VAD)
    gate_f_lo=500.0,    # hold gate acts only at and above this frequency (Hz)
    gate_band=(1000.0, 4000.0),   # band for reducing a per-bin cue to a frame cue (pause gate)
    pause_s=0.3,        # pause gate: minimum pause before the noise update resumes
    vad_band=(300.0, 4000.0),     # internal VAD: band of the SPP average
    vad_thr=0.25,       # internal VAD: speech if the band-mean SPP exceeds this (noise only: ~0.14)
    vad_xi_h1_db=15.0,  # internal VAD: SPP prior of its ungated tracker (vad_thr is calibrated for 15 dB)
    hold_max_s=5.0,     # longest continuous hold per bin; then the plain update resumes (None = no limit)
    init_s=2.0,         # initial noise PSD from the quietest frames of the first init_s seconds
    init_frac=0.25,
    lead_frames=N_FFT // HOP,
)
SCENARIOS = {
    "bus": dict(floor_db=-15.0, gate="hold", gate_thr=0.6),
    "meeting": dict(floor_db=-12.0, gate="none", gate_thr=0.5),
}
DESIGN_PARAMS = dict(xi_h1_db=15.0, alpha=0.9, alpha_dd=0.98)   # DESIGN M6 values (ablation overrides)


def params(scenario="bus", **overrides):
    """Parameter set for a scenario ('bus' or 'meeting'), with keyword overrides."""
    p = dict(PARAMS, **SCENARIOS[scenario])
    unknown = set(overrides) - set(p)
    if unknown:
        raise TypeError(f"unknown NS parameters: {sorted(unknown)}")
    p.update(overrides)
    return p


def bin_freqs(F, fs=FS):
    """Centre frequencies (Hz) of the F bins of a real STFT."""
    return np.arange(F) * fs / (2.0 * (F - 1))


def band_mean(M, f, band):
    """Mean of a (F, N) map over the bins with band[0] <= f <= band[1] -> (N,)."""
    sel = (f >= band[0]) & (f <= band[1])
    return np.asarray(M)[sel].mean(axis=0)


def init_noise(P, n_frames, frac=0.25):
    """Initial noise PSD (F,): mean periodogram over the quietest `frac` of the first n_frames
    frames, ranked by mean log power (robust to speech at the start; no Gaussian-specific quantile
    correction, so tonal noise is not over-estimated)."""
    seg = P[:, :max(int(n_frames), 1)]
    k = max(1, int(round(frac * seg.shape[1])))
    rank = np.argsort(np.mean(np.log(seg + _TINY), axis=0), kind="stable")
    return np.maximum(seg[:, rank[:k]].mean(axis=1), _TINY)


def track_noise(P, noise0, hold=None, xi_h1_db=PARAMS["xi_h1_db"], alpha=PARAMS["alpha"],
                alpha_p=PARAMS["alpha_p"], p_max=PARAMS["p_max"]):
    """MMSE-SPP noise PSD tracker [43] on a periodogram P (F, N).

    hold: optional bool array broadcastable to (F, N); where True the tracker state (noise PSD and
    smoothed SPP) is frozen for that bin and frame. Returns (noise PSD after each frame's update,
    SPP) as (F, N) arrays.
    """
    F, N = P.shape
    xi = 10.0 ** (xi_h1_db / 10.0)
    c0, c1 = 1.0 + xi, xi / (1.0 + xi)
    hold = None if hold is None else np.broadcast_to(hold, (F, N))
    s2 = np.maximum(np.asarray(noise0, dtype=np.float64), _TINY).copy()
    pbar = np.zeros(F)
    noise = np.empty((F, N))
    spp = np.empty((F, N))
    for l in range(N):
        y = P[:, l]
        p = 1.0 / (1.0 + c0 * np.exp(-c1 * y / s2))
        pb = alpha_p * pbar + (1.0 - alpha_p) * p
        p = np.where(pb > p_max, np.minimum(p, p_max), p)
        new = alpha * s2 + (1.0 - alpha) * ((1.0 - p) * y + p * s2)
        if hold is not None:
            h = hold[:, l]
            new = np.where(h, s2, new)
            pb = np.where(h, pbar, pb)
        s2 = np.maximum(new, _TINY)
        pbar = pb
        noise[:, l] = s2
        spp[:, l] = p
    return noise, spp


def lsa_gain(P, noise, alpha_dd=PARAMS["alpha_dd"], xi_min_db=PARAMS["xi_min_db"], floor_db=PARAMS["floor_db"],
             fall_db=PARAMS["fall_db"], rise_db=None):
    """Ephraim-Malah LSA gain [44] with decision-directed a-priori SNR -> G (F, N) in [floor, 1].

    The decision-directed recursion uses the unclipped LSA estimate (as in OM-LSA); the applied gain
    is clipped to 1 and passed through limit_gain(floor_db, fall_db, rise_db).
    """
    F, N = P.shape
    xi_min = 10.0 ** (xi_min_db / 10.0)
    G = np.empty((F, N))
    amp2 = np.ones(F)      # Gr(l-1)^2 * gamma(l-1); 1 = neutral start
    for l in range(N):
        gam = P[:, l] / noise[:, l]
        xi = np.maximum(alpha_dd * amp2 + (1.0 - alpha_dd) * np.maximum(gam - 1.0, 0.0), xi_min)
        g_raw = xi / (1.0 + xi) * np.exp(0.5 * exp1(np.maximum(xi * gam / (1.0 + xi), 1e-12)))
        amp2 = g_raw * g_raw * gam
        G[:, l] = np.minimum(g_raw, 1.0)
    return limit_gain(G, floor_db, fall_db, rise_db)


def limit_gain(G, floor_db=-15.0, fall_db=4.8, rise_db=None):
    """Floor a gain (F, N) and bound its change per frame: fall <= fall_db, rise <= rise_db
    (None = unlimited; DESIGN 2.1 rule: instant rise, fall <= 4.8 dB per 8 ms hop). Returns a copy."""
    g = np.maximum(np.asarray(G, dtype=np.float64), 0.0 if floor_db is None else 10.0 ** (floor_db / 20.0))
    if fall_db is None and rise_db is None:
        return g
    fall = 0.0 if fall_db is None else 10.0 ** (-fall_db / 20.0)
    rise = np.inf if rise_db is None else 10.0 ** (rise_db / 20.0)
    for l in range(1, g.shape[1]):
        g[:, l] = np.clip(g[:, l], g[:, l - 1] * fall, g[:, l - 1] * rise)
    return g


def cue_hold(cue, f, thr=0.6, f_lo=500.0):
    """Hold mask for gate='hold': cue (N,) per frame or (F, N) per bin > thr, only where f >= f_lo."""
    c = np.asarray(cue, dtype=np.float64)
    h = c[None, :] > thr if c.ndim == 1 else c > thr
    return h & (f >= f_lo)[:, None]


def pause_open(speech, n_min):
    """True where the current run of non-speech frames (this frame included) is >= n_min frames."""
    speech = np.asarray(speech, dtype=bool)
    idx = np.arange(len(speech))
    last = np.maximum.accumulate(np.where(speech, idx, -1))
    return idx - last >= n_min


def cap_runs(h, n_max):
    """Bool array h (..., N) with every run of True along the last axis cut to its first n_max frames
    (a hold longer than that is released until the cue drops)."""
    h = np.asarray(h, dtype=bool)
    idx = np.arange(h.shape[-1])
    last_off = np.maximum.accumulate(np.where(h, -1, idx), axis=-1)
    return h & (idx - last_off <= n_max)


def suppress(Y, scenario="bus", target_prob=None, fs=FS, hop=HOP, **overrides):
    """Noise suppression of a single-channel STFT Y (F, N).

    target_prob: optional cue in [0, 1], per frame (N,) or per bin (F, N): probability that wanted
    speech is present (gate='hold': target cue; gate='pause': VAD; ignored with gate='none', the
    meeting default). Keyword overrides go to params().
    Returns dict: G (gain, F x N), S = G * Y (enhanced STFT), noise (PSD estimate after each frame),
    spp (speech presence probability), hold (bool (F, N), True where the update was frozen).
    """
    p = params(scenario, **overrides)
    Y = np.asarray(Y)
    P = (Y.real ** 2 + Y.imag ** 2).astype(np.float64)
    F, N = P.shape
    f = bin_freqs(F, fs)
    lead = min(int(p["lead_frames"]), N)
    noise0 = init_noise(P[:, lead:] if lead < N else P, round(p["init_s"] * fs / hop), p["init_frac"])
    track = {k: p[k] for k in ("xi_h1_db", "alpha", "alpha_p", "p_max")}
    hold = np.zeros((F, N), dtype=bool)
    hold[:, :lead] = True

    gate = None
    if p["gate"] == "hold" and target_prob is not None:
        gate = cue_hold(target_prob, f, p["gate_thr"], p["gate_f_lo"])
    elif p["gate"] == "pause":
        if target_prob is None:
            _, spp0 = track_noise(P, noise0, hold, **dict(track, xi_h1_db=p["vad_xi_h1_db"]))
            speech = band_mean(spp0, f, p["vad_band"]) > p["vad_thr"]
        else:
            c = np.asarray(target_prob, dtype=np.float64)
            speech = (c if c.ndim == 1 else band_mean(c, f, p["gate_band"])) > p["gate_thr"]
        gate = ~pause_open(speech, int(round(p["pause_s"] * fs / hop)))[None, :]
    elif p["gate"] not in ("hold", "none"):
        raise ValueError(f"unknown gate {p['gate']!r}")
    if gate is not None:
        if p["hold_max_s"] is not None:
            gate = cap_runs(np.broadcast_to(gate, (F, N)), int(round(p["hold_max_s"] * fs / hop)))
        hold |= gate

    noise, spp = track_noise(P, noise0, hold, **track)
    G = lsa_gain(P, noise, p["alpha_dd"], p["xi_min_db"], p["floor_db"], p["fall_db"], p["rise_db"])
    return {"G": G, "S": G * Y, "noise": noise, "spp": spp, "hold": hold}
