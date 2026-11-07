"""M11 leveling (DESIGN.md M11; bus mode: 2.1 'M11-bus'; meeting: 2.2). Entry point for stacks: level().

Chain, each block usable on its own: speech-gated, noise-capped AGC with per-key gain memory (Leveler)
-> off-attenuation in pauses (off_attenuation) -> -1 dBTP true-peak limiter (limiter). dugan_automix()
is the multi-node gain-sharing mixer (one node passes through unchanged).

Parameters: params(scenario, online) = PARAMS (the DESIGN values) + SCENARIOS[scenario] (tuned on the v2
dev set) + OFFLINE[scenario] when offline. Levels are band levels of the AGC input per frame of
common.stft (hop 8 ms; meeting 150-7000 Hz, bus 300-4000 Hz), in dBFS re a full-scale sine (AES17:
10 log10(mean square) + 3.01 dB). Per frame l, gains in dB:
- p(l), target-speech probability: S1 VAD (meeting) or min(1, p_A + p_B) (bus; never the VAD, which
  also fires on bystanders). Keys k(l) >= 0 name what gets its own gain ('auto'): meeting one key per S1
  speech segment ('segment'; the DP-means talker clusters merge talkers and measured worse), bus one
  pooled key. Options: 'cluster' (S1 clusters: the DESIGN gain memory, for embedding clusters), 'pooled'
  (no memory), 'tdoa' (|tau| bins), 'partner' (bus A/B), or an int array.
- Level update (AGC2 rule): every frame of a run with p >= p_update and one key that lasts >= min_run_s
  (meeting 0.5 / 50 ms; bus 0.8 / 120 ms; DESIGN 0.95 / 120 ms); online the run's first frames take
  effect when the run is long enough, offline as they come. Leaky power mean over level_tau_s of gated
  speech: num <- lam num + P, den <- lam den + 1, L_k = num / den. Meeting: the tracked noise power on
  the same frames is subtracted (noise_sub_db: by at most 10 dB). Offline meeting: every key starts
  from its first-pass level over the whole file (prime_s) and level_tau_s is 30 s, so each S1 segment
  gets one static gain; online a segment starts from the room-average gain and follows a 0.5 s leaky
  level.
- Noise N(l): leaky mean in dB (1 s) over pause frames (p < 0.3 from 300 ms after speech to 32 ms
  before it, so reverberant tails and onsets stay out), held otherwise; or given by the caller (e.g.
  the residual noise after M6).
- Target g_k = clip(min(T - L_k, max(noise_max_db - N, 0), G_room + boost_max_db), -10, +45), T = -26 dBFS.
  Bus: the DESIGN noise cap (output noise <= -50 dBFS; boost only, never forced attenuation, as AGC2).
  Meeting: no absolute cap (on raw meeting input every talker is below its 24 dB SNR, so it pins all
  talkers to one gain) but at most 6 dB above the room-average gain G_room (offline: first pass, median
  over keys; online: running median of the known keys), which bounds the noise boost of any one key.
- Applied gain: held in pauses (p < 0.5 or no key). Within a key it slews toward g_k (meeting 12 dB/s,
  bus 6 dB/s; 20 dB/s during a new key's first 0.5 s). A change of key is confirmed by the new
  key's first level update (switch='speech'; 'label' switches known keys at their label onset). It then
  crossfades (linear in dB, 75 ms) to the stored g_k of a known key, or to a prior for a new key: the
  median stored gain of known keys in the same 10 us |tau| bin when the mic axis lies along the table
  (axis_along_table), otherwise the room-average gain (online before any key is known: its first level).
Off-attenuation (off_db; DESIGN -3 dB, tuned 0 for both scenarios): after 300 ms without speech (falling
at 10 dB/s), restored within 20 ms (offline with a 20 ms look-ahead). Limiter: 4x-oversampled true-peak
detector, 5 ms look-ahead, 50 ms release.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, firwin, lfilter, resample_poly, sosfilt

from lab import dsp
from . import common as cm

AES17_DB = 10 * math.log10(2.0)                                        # a full-scale sine is 0 dBFS
_BAND_NORM = cm.N_FFT * float(np.sum(dsp._win(cm.N_FFT) ** 2)) / 2.0   # sum_band |X|^2 -> mean square
_TP_BLOCK, _TP_MARGIN = 1 << 18, 64        # true-peak oversampling chunks; margin > the FIR half-length (48)

PARAMS = dict(
    target_db=-26.0,          # T: target speech level (band level, dBFS)
    gain_min_db=-10.0,
    gain_max_db=45.0,
    noise_max_db=-50.0,       # noise cap: output noise level at most this (dBFS), boost only
    boost_max_db=math.inf,    # relative cap: gain at most this far above the room-average gain
    band_hz=(300.0, 4000.0),  # level band
    p_update=0.95,            # level update: p >= p_update ...
    min_run_s=0.12,           # ... for at least this long (AGC2: 12 frames of 10 ms)
    confirm_s=0.25,           # online: no update until this long after an S1 change proposal
    level_tau_s=1.0,          # leaky RMS time constant (seconds of gated speech)
    noise_sub_db=0.0,         # >0: a key's level is its gated power minus the tracked noise power,
                              # lowered by at most this many dB (0: noisy power as is)
    prime_s=0.0,              # offline: seed each key with its whole-file level (weight in s; 0: off)
    p_speech=0.5,             # speech: the gain may move and the off-attenuation is released
    p_pause=0.3,              # noise level measured below this ...
    noise_hold_s=0.3,         # ... this long after speech (reverberant tails excluded) ...
    noise_guard_s=0.032,      # ... and this long before speech (onsets excluded)
    noise_tau_s=1.0,
    noise_init_s=3.0,         # initial noise level from the pauses of the first 3 s
    rate_db_s=6.0,            # max gain change within a talker
    fast_rate_db_s=20.0,      # ... for a new key, during its first fast_s of adaptation
    fast_s=0.5,
    xfade_s=0.075,            # crossfade to a stored or prior gain on a change of key (DESIGN 50-100 ms)
    switch="speech",          # change of key on confirmed speech; 'label': known keys at their label onset
    axis_along_table=False,   # new-key prior from the |tau| bin (mic axis along the table, DESIGN 5)
    tau_bin_s=10e-6,
    onset_s=0.25,             # a new key's |tau| comes from its first 250 ms (also the partner-key hold)
    off_db=-3.0,              # off-attenuation in pauses (0 disables)
    off_after_s=0.3,
    off_fall_db_s=10.0,
    off_restore_s=0.02,
    ceiling_dbtp=-1.0,        # limiter
    oversample=4,
    lookahead_s=0.005,
    release_s=0.05,
)
# Tuned on the v2 dev meetings (tune:agc report): level the 150-7000 Hz band (the T3 band and close to
# BS.1770), gate on p >= 0.5 runs of >= 50 ms with the noise power subtracted, no absolute noise cap
# (on raw meeting input it pins every talker to one gain) but at most 6 dB above the room-average gain,
# no off-attenuation (it lands on soft talkers that the VAD misses). Online: faster leaky levels.
SCENARIOS = {
    "meeting": dict(band_hz=(150.0, 7000.0), p_update=0.5, min_run_s=0.05, noise_sub_db=10.0,
                    noise_max_db=math.inf, boost_max_db=6.0, off_db=0.0, level_tau_s=0.5, rate_db_s=12.0),
    "bus": dict(p_update=0.8, off_db=0.0),
}
# Offline (two-pass) additions: static per-segment levels from the first pass.
OFFLINE = {"meeting": dict(prime_s=math.inf, level_tau_s=30.0), "bus": {}}
# keys='auto': one key per S1 speech segment in the meeting (the DP-means talker clusters merge talkers
# and measured worse, offline and online; use keys='cluster' with embedding clusters), pooled on the bus.
AUTO_KEYS = {"meeting": "segment", "bus": "pooled"}


def params(scenario="meeting", online=False, **overrides):
    """Parameter set for 'meeting' or 'bus' (offline adds OFFLINE[scenario]) with keyword overrides."""
    p = {**PARAMS, **SCENARIOS[scenario], **({} if online else OFFLINE[scenario])}
    unknown = set(overrides) - set(p)
    if unknown:
        raise TypeError(f"unknown AGC parameters: {sorted(unknown)}")
    p.update(overrides)
    return p


def dbfs(P):
    """Mean square -> dBFS re a full-scale sine."""
    return 10 * np.log10(np.maximum(P, 1e-30)) + AES17_DB


# ------------------------------------------------------------------ per-frame inputs
def band_power(y=None, Y=None, band=PARAMS["band_hz"]):
    """(N,) mean square of the band per common.stft frame, from y (T,) or its common.stft Y (257, N)."""
    sl = cm.bins_between(*band)
    if Y is not None:
        Yb = Y[sl]
        return (Yb.real ** 2 + Yb.imag ** 2).sum(axis=0) / _BAND_NORM
    xp = np.pad(np.asarray(y, np.float64), (cm.N_FFT, 2 * cm.N_FFT))
    n = cm.n_frames(len(y))
    w = dsp._win(cm.N_FFT)
    out = np.empty(n)
    for l0 in range(0, n, cm.CHUNK):
        l1 = min(n, l0 + cm.CHUNK)
        fr = sliding_window_view(xp[l0 * cm.HOP:(l1 - 1) * cm.HOP + cm.N_FFT], cm.N_FFT)[::cm.HOP]
        X = np.fft.rfft(fr * w, axis=-1)[:, sl]
        out[l0:l1] = (X.real ** 2 + X.imag ** 2).sum(axis=1) / _BAND_NORM
    return out


def target_prob(trk, scenario):
    """(N,) target-speech probability from an S1 Track: VAD (meeting) or min(1, p_A + p_B) (bus)."""
    if scenario == "bus":
        return np.minimum(1.0, np.asarray(trk.p_A) + np.asarray(trk.p_B))
    return np.asarray(trk.vad, np.float64)


def _wmedian(v, w):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def tdoa_keys(trk, bin_s=PARAMS["tau_bin_s"]):
    """'TDOA-keyed memory' ablation (DESIGN 4.8): every run of one S1 cluster label is keyed by the
    bin_s bin of its directness-weighted median |tau| (frames after the first 48 ms)."""
    lab = np.asarray(trk.cluster, int)
    keys = np.full(len(lab), -1)
    edges = np.flatnonzero(np.diff(np.r_[-2, lab, -2]) != 0)
    for a, b in zip(edges[:-1], edges[1:]):
        if lab[a] >= 0:
            s = a + 6 if b - a > 12 else a
            keys[a:b] = int(_wmedian(np.abs(trk.tau[s:b]), trk.directness[s:b] + 1e-6) // bin_s)
    return keys


def segment_keys(trk):
    """Meeting default: one key per S1 speech segment (a run of one cluster label, split at S1 change
    proposals), numbered in time order. Offline (prime_s) every segment gets the static gain of its own
    gated level, a per-turn stand-in for the DESIGN's per-speaker offline gains when diarization merges
    talkers (DP-means clusters); online a segment starts from the room-average gain and adapts."""
    lab = np.asarray(trk.cluster, int)
    sp = lab >= 0
    ch = np.zeros(len(lab), bool) if getattr(trk, "change", None) is None else np.asarray(trk.change, bool)
    start = sp & (np.r_[True, ~sp[:-1]] | np.r_[True, lab[1:] != lab[:-1]] | ch)
    return np.where(sp, np.cumsum(start) - 1, -1)


def partner_keys(p_A, p_B, p_speech=0.5, hold_s=0.25):
    """Bus option: key 0 (partner A) or 1 (B) on frames with p_A + p_B >= p_speech, else -1. The key
    follows the larger posterior but switches only after the other partner has led for hold_s."""
    n_hold = round(hold_s / cm.HOP_S)
    keys = np.full(len(p_A), -1)
    cur, run = -1, 0
    for l, (a, b) in enumerate(zip(np.asarray(p_A).tolist(), np.asarray(p_B).tolist())):
        if a + b < p_speech:
            continue
        lead = 0 if a >= b else 1
        run = run + 1 if lead != cur else 0
        if cur < 0 or run >= n_hold:
            cur, run = lead, 0
        keys[l] = cur
    return keys


def make_keys(mode, trk, scenario, p, prm):
    """Per-frame int keys (-1 = none). mode: an array (used as is), 'auto' (AUTO_KEYS when S1 clusters
    exist: meeting 'segment', bus 'pooled'; else 'pooled'), 'cluster', 'segment', 'pooled' (one key on
    speech frames), 'tdoa', 'partner'."""
    if not isinstance(mode, str):
        return np.asarray(mode, int)
    if mode == "auto":
        has_clusters = trk is not None and getattr(trk, "cluster", None) is not None
        mode = AUTO_KEYS[scenario] if has_clusters else "pooled"
    if mode == "pooled":
        return np.where(p >= prm["p_speech"], 0, -1)
    if mode == "cluster":
        return np.asarray(trk.cluster, int)
    if mode == "segment":
        return segment_keys(trk)
    if mode == "tdoa":
        return tdoa_keys(trk, prm["tau_bin_s"])
    if mode == "partner":
        return partner_keys(trk.p_A, trk.p_B, prm["p_speech"], prm["onset_s"])
    raise ValueError(f"unknown key mode {mode!r}")


def update_gate(p, keys, prm, change=None, online=False):
    """Level-update frames (AGC2 rule): every frame of a run of p >= p_update with one key (>= 0) that
    lasts >= min_run_s. A run restarts at a change of key and at S1 change proposals (`change`); online,
    frames within confirm_s after a proposal are excluded (S1 online labels lag a change by 250 ms).
    Returns (upd, vis, first), each (N,): upd bool; vis int, the frame at which an update takes effect
    (online: the frame that completes min_run_s for a run's first frames, else the frame itself;
    offline: always the frame itself, as the run is known to complete); first int, the first frame of
    the update frame's run (-1 elsewhere)."""
    N = len(p)
    n_run = max(1, round(prm["min_run_s"] / cm.HOP_S))
    n_conf = round(prm["confirm_s"] / cm.HOP_S) if online and change is not None else 0
    ok = ((np.asarray(p) >= prm["p_update"]) & (keys >= 0) & cm.valid_frames(N)).tolist()
    kl = keys.tolist()
    ch = np.zeros(N, bool).tolist() if change is None else np.asarray(change, bool).tolist()
    upd, vis, first = np.zeros(N, bool), np.arange(N), np.full(N, -1)
    start, since, prev = -1, n_conf, None
    for l in range(N):
        if ch[l]:
            start, since = -1, 0
        if kl[l] != prev:
            start, prev = -1, kl[l]
        since += 1
        if not (ok[l] and since > n_conf):
            start = -1
            continue
        if start < 0:
            start = l
        if l - start + 1 == n_run:
            upd[start:l + 1], first[start:l + 1] = True, start
            if online:
                vis[start:l + 1] = l
        elif l - start + 1 > n_run:
            upd[l], first[l] = True, start
    return upd, vis, first


def noise_level(P, p, prm):
    """(N,) residual-noise band level (dBFS): leaky mean in dB (noise_tau_s) over strict pause frames,
    i.e. p < p_pause on the frame, for noise_hold_s before it (reverberant tails of a turn stay out) and
    for noise_guard_s after it (onsets whose VAD has not risen yet stay out). The estimate at frame l uses
    frames up to l - guard (causal with that delay) and starts from the median over the strict pauses of
    the first noise_init_s (else the 10th percentile there)."""
    N = len(P)
    g, h = round(prm["noise_guard_s"] / cm.HOP_S), round(prm["noise_hold_s"] / cm.HOP_S)
    valid = cm.valid_frames(N)
    pause = (np.asarray(p) < prm["p_pause"]) & valid
    strict = minimum_filter1d(pause.astype(np.uint8), size=g + h + 1, origin=(h - g) // 2,
                              mode="constant").astype(bool)
    L = dbfs(P)
    first = valid & (np.arange(N) < round(prm["noise_init_s"] / cm.HOP_S) + cm.N_FFT // cm.HOP)
    if (first & strict).any():
        s = float(np.median(L[first & strict]))
    else:
        s = float(np.percentile(L[first] if first.any() else L, 10))
    lam = math.exp(-cm.HOP_S / prm["noise_tau_s"])
    out = np.empty(N)
    Ld, sd = np.r_[np.zeros(g), L].tolist(), np.r_[np.zeros(g, bool), strict].tolist()
    for l in range(N):
        if sd[l]:
            s = lam * s + (1.0 - lam) * Ld[l]
        out[l] = s
    return out


def sub_power(p, pn, sub_db):
    """Speech power estimate: noisy power p minus noise power pn, at least p lowered by sub_db (0: p)."""
    return max(p - pn, p * 10.0 ** (-sub_db / 10.0)) if sub_db > 0 else p


def room_gain(P, keys, upd, p, prm, noise_db=None):
    """Offline first pass: room-average gain (dB, before the noise cap) = T minus the median over keys
    of each key's mean gated level (all speech frames if nothing is gated; noise power subtracted as in
    the keys' levels when noise_sub_db > 0 and noise_db is given); None without speech."""
    Pn = np.zeros(len(P)) if noise_db is None else 10.0 ** ((np.asarray(noise_db, float) - AES17_DB) / 10.0)
    sets = [upd & (keys == k) for k in np.unique(keys[upd])]
    if not sets:
        sp = (np.asarray(p) >= prm["p_speech"]) & cm.valid_frames(len(P))
        if not sp.any():
            return None
        sets = [sp]
    lv = [sub_power(P[m].mean(), Pn[m].mean(), prm["noise_sub_db"]) for m in sets]
    return prm["target_db"] - float(np.median(dbfs(np.array(lv))))


# ------------------------------------------------------------------ speaker-keyed AGC
@dataclass
class KeyState:
    """Memory of one talker key."""
    num: float = 0.0            # leaky sums of gated band power and of frame counts: level = num / den
    den: float = 0.0
    nnum: float = 0.0           # leaky sum of the tracked noise power on the same frames
    n_update: int = 0
    fast: int = 0               # frames of fast adaptation left
    prior_db: float = math.nan  # gain the key started from (NaN: none, waited for its first level)
    prior_src: str = ""         # 'tau_bin', 'room', 'room_known' or 'first_level'
    taus: list = field(default_factory=list)   # |tau| and directness of gated frames
    ws: list = field(default_factory=list)

    def level_db(self, sub_db=0.0):
        """Gated band level (dBFS); with sub_db > 0 the noise power is subtracted (see sub_power)."""
        if self.den <= 0:
            return math.nan
        return 10 * math.log10(max(sub_power(self.num, self.nnum, sub_db) / self.den, 1e-30)) + AES17_DB

    def tau_abs(self):
        return _wmedian(np.array(self.taus), np.array(self.ws) + 1e-6) if self.taus else math.nan


class Leveler:
    """Speaker-keyed AGC: per-key levels (the gain memory) and the applied gain. The state persists, so
    run() can continue on the next file of a session."""

    def __init__(self, prm, room_gain_db=None):
        self.prm = prm
        self.room = room_gain_db      # offline first-pass room-average gain (dB, before the cap)
        self.mem = {}
        self.gain = None              # applied gain (dB)
        self.active = -1              # key of the current or last talker
        self.ref = None               # online room-average reference: median raw gain of the known keys

    def _clip(self, g, cap):
        ref = self.room if self.room is not None else self.ref
        if ref is not None:
            g = min(g, ref + self.prm["boost_max_db"])
        return min(max(min(g, cap), self.prm["gain_min_db"]), self.prm["gain_max_db"])

    def _raw(self, st):
        return self.prm["target_db"] - st.level_db(self.prm["noise_sub_db"])

    def target(self, st, cap):
        """Target gain (dB) of a key with a level, under the noise cap `cap` (dB) and boost_max_db above
        the room-average gain."""
        return self._clip(self._raw(st), cap)

    def prime(self, P, keys, upd, noise_db):
        """Offline first pass: seed every key that has level updates with its whole-file gated level,
        weighted as prime_s seconds of gated speech (or all of its updates if fewer), so that its first
        turn already gets its own gain; the updates of the second pass then continue the leaky mean."""
        Pn = 10.0 ** ((np.asarray(noise_db, float) - AES17_DB) / 10.0)
        n_max = self.prm["prime_s"] / cm.HOP_S
        for k in np.unique(keys[upd]).tolist():
            m = upd & (keys == k)
            n = min(float(m.sum()), n_max)
            self.mem[k] = KeyState(num=float(P[m].mean()) * n, den=n, nnum=float(Pn[m].mean()) * n,
                                   prior_src="first_pass")

    def prior(self, tau_abs, cap):
        """(gain, source) for a new key: |tau|-bin median (axis along the table), else room average."""
        known = [s for s in self.mem.values() if s.den > 0]
        b = self.prm["tau_bin_s"]
        if self.prm["axis_along_table"] and math.isfinite(tau_abs):
            same = [self.target(s, cap) for s in known
                    if math.isfinite(s.tau_abs()) and s.tau_abs() // b == tau_abs // b]
            if same:
                return float(np.median(same)), "tau_bin"
        if self.room is not None:
            return self._clip(self.room, cap), "room"
        if known:
            return float(np.median([self.target(s, cap) for s in known])), "room_known"
        return math.nan, "first_level"

    def run(self, P, p, keys, noise_db, gate, tau=None, w=None, online=False, cap_noise_db=None):
        """Applied AGC gain per frame (dB). P: band power (N,); p: target-speech probability; keys: int
        (N,); noise_db: (N,) noise level (dBFS); gate: update_gate() output; tau, w: per-frame TDOA and
        directness (|tau| of each key, for the |tau|-bin prior); cap_noise_db: (N,) noise level behind
        the absolute cap (default noise_db).
        A change of talker takes effect when the new key's speech is confirmed (its first level update),
        so a noise burst that S1 labels as a talker never moves the gain. With prm['switch'] == 'label',
        a known key also takes over at its label onset on a speech frame (DESIGN-literal)."""
        prm, hop = self.prm, cm.HOP_S
        lam = math.exp(-hop / prm["level_tau_s"])
        n_xf, n_fast = max(1, round(prm["xfade_s"] / hop)), round(prm["fast_s"] / hop)
        n_on = round(prm["onset_s"] / hop)
        slow, fast = prm["rate_db_s"] * hop, prm["fast_rate_db_s"] * hop
        nc = noise_db if cap_noise_db is None else cap_noise_db
        cap = np.maximum(prm["noise_max_db"] - np.asarray(nc, float), 0.0).tolist()
        Nl = (10.0 ** ((np.asarray(noise_db, float) - AES17_DB) / 10.0)).tolist()   # noise mean square
        speech = (np.asarray(p) >= prm["p_speech"]).tolist()
        Pl, kl = np.asarray(P, float).tolist(), np.asarray(keys, int).tolist()
        tl = None if tau is None else np.abs(tau).tolist()
        wl = None if w is None else np.asarray(w, float).tolist()
        upd, vis, first = gate
        uf = np.flatnonzero(upd)
        uv, uf = vis[uf].tolist(), uf.tolist()
        on_label = prm["switch"] == "label"

        def onset_tau(batch):
            """|tau| of a new key: its confirmed frames (online) or its run's first onset_s (offline)."""
            if tl is None:
                return math.nan
            j0 = first[batch[0]]
            idx = batch if online else [j for j in range(j0, min(len(kl), j0 + n_on)) if first[j] == j0]
            return _wmedian(np.array([tl[j] for j in idx]), np.array([wl[j] for j in idx]) + 1e-6)

        def switch(k, st, l, batch):
            """Crossfade toward the stored gain of a known key or a new key's prior."""
            if st.den > 0:
                dest = self.target(st, cap[l])
            else:
                st.fast = n_fast
                st.prior_db, st.prior_src = self.prior(onset_tau(batch), cap[l])
                dest = st.prior_db
            self.active = k
            return None if math.isnan(dest) else (g, dest, 0)

        if self.gain is None:
            self.gain = 0.0 if self.room is None else self._clip(self.room, cap[0])
        g, xf, ptr = self.gain, None, 0
        out = np.empty(len(Pl))
        for l, k in enumerate(kl):
            if on_label and k >= 0 and k != self.active and speech[l] and k in self.mem and self.mem[k].den > 0:
                xf = switch(k, self.mem[k], l, None)
            if ptr < len(uf) and uv[ptr] <= l:
                b0 = ptr
                while ptr < len(uf) and uv[ptr] <= l:
                    ptr += 1
                batch = uf[b0:ptr]                     # one run's frames: a single key
                kb = kl[batch[0]]
                st = self.mem.setdefault(kb, KeyState())
                if kb != self.active:
                    xf = switch(kb, st, l, batch)
                boot = st.den == 0 and math.isnan(st.prior_db)
                for j in batch:
                    st.num, st.den = lam * st.num + Pl[j], lam * st.den + 1.0
                    st.nnum = lam * st.nnum + Nl[j]
                    if tl is not None:
                        st.taus.append(tl[j])
                        st.ws.append(wl[j])
                st.n_update += len(batch)
                if self.room is None:
                    self.ref = float(np.median([self._raw(s) for s in self.mem.values() if s.den > 0]))
                if boot:                               # no prior: start from the first confirmed level
                    xf = (g, self.target(st, cap[l]), 0)
            if xf is not None:
                i = xf[2] + 1
                g = xf[0] + (xf[1] - xf[0]) * i / n_xf
                xf = None if i >= n_xf else (xf[0], xf[1], i)
            elif k >= 0 and k == self.active and speech[l] and self.mem[k].den > 0:
                st = self.mem[k]
                r = fast if st.fast > 0 else slow
                g += min(max(self.target(st, cap[l]) - g, -r), r)
                st.fast = max(0, st.fast - 1)
            out[l] = g
        self.gain = g
        return out

    def summary(self, cap=0.0):
        """Per-key memory as a JSON-able dict (gains under the noise cap `cap`)."""
        return {int(k): {"level_db": s.level_db(self.prm["noise_sub_db"]),
                         "gain_db": self.target(s, cap) if s.den > 0 else math.nan,
                         "n_update": s.n_update, "tau_us": 1e6 * s.tau_abs(), "prior_db": s.prior_db,
                         "prior_src": s.prior_src} for k, s in self.mem.items()}


# ------------------------------------------------------------------ off-attenuation, gain application
def off_attenuation(speech, prm, lookahead=False):
    """(N,) off-attenuation (dB, <= 0): falls toward off_db at off_fall_db_s once off_after_s has passed
    without speech; restored at |off_db| per off_restore_s. lookahead (offline): restored by the onset."""
    N, off = len(speech), prm["off_db"]
    if off >= 0:
        return np.zeros(N)
    sp = np.asarray(speech, bool)
    if lookahead:
        c = np.r_[0, np.cumsum(sp)]
        n_la = math.ceil(prm["off_restore_s"] / cm.HOP_S)
        sp = c[np.minimum(np.arange(N) + n_la + 1, N)] > c[:N]
    idx = np.arange(N)
    last = np.maximum.accumulate(np.where(sp, idx, -N - 10 ** 6))
    tgt = np.where(idx - last >= round(prm["off_after_s"] / cm.HOP_S), off, 0.0).tolist()
    fall, rise = prm["off_fall_db_s"] * cm.HOP_S, -off * cm.HOP_S / prm["off_restore_s"]
    out, g = np.empty(N), 0.0
    for l, t in enumerate(tgt):
        g = min(max(t, g - fall), g + rise)
        out[l] = g
    return out


def frame_gain(gain_db, T):
    """Per-frame gain (dB) -> per-sample linear gain (T,), interpolated between frame centres."""
    return 10.0 ** (np.interp(np.arange(T), cm.frame_centers(len(gain_db)), gain_db) / 20.0)


# ------------------------------------------------------------------ true-peak limiter
def _tp_filter(oversample):
    """Interpolation FIR of the true-peak detector: 48 input samples each side, Kaiser beta 9; flat
    (< 0.01 dB) to 7.5 kHz with >= 90 dB image rejection from 8.5 kHz at fs = 16 kHz. resample_poly's
    default (10 samples each side) under-reads 6 / 7.5 kHz peaks by 0.2 / 0.7 dB."""
    return firwin(96 * oversample + 1, 1.0 / oversample, window=("kaiser", 9.0))


def _interval_peaks(y, oversample):
    """(T,) max |.| of the oversampled signal over [n, n+1) (resample_poly, processed in chunks)."""
    T = len(y)
    h = _tp_filter(oversample)
    out = np.empty(T)
    for a in range(0, T, _TP_BLOCK):
        b = min(T, a + _TP_BLOCK)
        lo, hi = max(0, a - _TP_MARGIN), min(T, b + _TP_MARGIN)
        u = resample_poly(y[lo:hi], oversample, 1, window=h)[(a - lo) * oversample:(b - lo) * oversample]
        out[a:b] = np.abs(u).reshape(b - a, oversample).max(axis=1)
    return out


def true_peak_db(y, oversample=4):
    """True-peak level (dBTP) with an oversampled peak meter (BS.1770 style)."""
    y = np.asarray(y, np.float64)
    return float(20 * np.log10(max(_interval_peaks(y, oversample).max(), 1e-30))) if len(y) else -np.inf


def limiter(y, ceiling_dbtp=-1.0, oversample=4, lookahead_s=0.005, release_s=0.05, fs=cm.FS, margin_db=0.05):
    """Look-ahead true-peak limiter -> (out, gain (T,)).
    Needed attenuation per sample from the oversampled peaks on both sides of it; held over the
    look-ahead window [n, n + L]; exponential release in dB (A[n] = max(A_hold[n], a A[n-1])); then a
    box average over [n - L, n]. Each averaged value is at most the gain needed at n, so the samples
    stay under the ceiling; margin_db covers the inter-sample effect of a gain that varies within the
    interpolation filter (seen at abrupt full-scale starts: +0.008 dB). The gain is exactly 1 away
    from limiting."""
    y = np.asarray(y, np.float64)
    T = len(y)
    iv = _interval_peaks(y, oversample)
    need = 20 * np.log10(np.maximum(np.maximum(iv, np.r_[0.0, iv[:-1]]), 1e-30)) - ceiling_dbtp + margin_db
    if not T or need.max() <= 0:
        return y.copy(), np.ones(T)
    L = max(0, round(lookahead_s * fs))
    att = np.r_[np.zeros(L), np.maximum(need, 0.0)]            # L virtual samples before the start
    hold = maximum_filter1d(att, size=L + 1, origin=-((L + 1) // 2), mode="constant")
    n = np.arange(T + L) / (release_s * fs)
    with np.errstate(divide="ignore"):
        A = np.exp(np.maximum.accumulate(np.log(hold) + n) - n)
    A[A < 1e-6] = 0.0
    c = np.r_[0.0, np.cumsum(1.0 - 10.0 ** (-A / 20.0))]
    g = 1.0 - (c[L + 1:] - c[:T]) / (L + 1)
    return y * g, g


# ------------------------------------------------------------------ multi-node automix
def dugan_automix(sigs, fs=cm.FS, band=(300.0, 4000.0), smooth_s=0.02, attack_s=0.01, release_s=0.2,
                  block=16):
    """Dugan gain-sharing automix of K time-aligned node signals (K, T) -> (mix (T,), gains (K, T)).
    g_k = sqrt(P_k / sum_j P_j), P = band power smoothed over smooth_s. The gains are smoothed (attack_s
    when rising, release_s when falling) on a grid of `block` samples and renormalized to sum g^2 = 1,
    so uncorrelated background stays at the level of one open mic. One node passes through unchanged."""
    x = np.atleast_2d(np.asarray(sigs, np.float64))
    K, T = x.shape
    if K == 1:
        return x[0].copy(), np.ones((1, T))
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    lam = math.exp(-1.0 / (smooth_s * fs))
    P = lfilter([1 - lam], [1, -lam], sosfilt(sos, x, axis=-1) ** 2, axis=-1)[:, block - 1::block]
    raw = np.sqrt((P + 1e-30) / (P.sum(axis=0) + K * 1e-30))
    a_att, a_rel = math.exp(-block / (attack_s * fs)), math.exp(-block / (release_s * fs))
    g = np.empty_like(raw)
    cur = np.full(K, 1 / math.sqrt(K))
    for i in range(raw.shape[1]):
        a = np.where(raw[:, i] > cur, a_att, a_rel)
        cur = a * cur + (1 - a) * raw[:, i]
        cur /= math.sqrt(float(cur @ cur))
        g[:, i] = cur
    t = np.arange(block - 1, T, block)
    gains = np.stack([np.interp(np.arange(T), t, gk) for gk in g])
    gains /= np.sqrt((gains ** 2).sum(axis=0))
    return (gains * x).sum(axis=0), gains


# ------------------------------------------------------------------ entry point for stacks
@dataclass
class AgcResult:
    """level() output. Frame arrays follow the common.stft grid (frame l centred on sample l*128 - 256)."""
    out: np.ndarray        # (T,) leveled, off-attenuated and limited output
    pre: np.ndarray        # (T,) AGC input: the 'pre_agc' stage
    gain: np.ndarray       # (T,) float32 total per-sample gain (out = pre * gain up to float32 rounding)
    agc_db: np.ndarray     # (N,) AGC gain per frame (dB)
    off_db: np.ndarray     # (N,) off-attenuation per frame (dB)
    level_db: np.ndarray   # (N,) input band level (dBFS)
    noise_db: np.ndarray   # (N,) noise level behind the cap (dBFS)
    p: np.ndarray          # (N,) target-speech probability used
    keys: np.ndarray       # (N,) talker keys used (-1: none)
    update: np.ndarray     # (N,) bool level-update frames
    leveler: Leveler       # per-key memory; pass to level(..., leveler=) to continue a session
    diag: dict


def level(y, trk=None, scenario="meeting", *, Y=None, p=None, keys="auto", noise_db=None, online=False,
          leveler=None, **overrides):
    """M11 on any enhanced single-channel signal y (T,) at 16 kHz, time-aligned with the mics.

    trk: S1 Track on the same frames (tracker.track / front_end); uses vad (meeting) or p_A, p_B (bus),
         cluster, change, tau and directness. Without trk: an energy VAD on y and one pooled key.
    Y: common.stft(y) if already computed (saves one STFT). p: (N,) target-speech probability override.
    keys: int (N,) array or a make_keys() mode ('auto', 'segment', 'cluster', 'pooled', 'tdoa', 'partner').
    noise_db: residual-noise band level of y (dBFS, scalar or (N,)), e.g. from M6; default: tracked in
         pauses of y. online: causal variant (no first pass, so no room-average gain, static per-key
         levels or file-median noise behind the absolute cap; change confirmation; no off look-ahead).
    leveler: a previous result's Leveler (speaker memory across files). overrides: PARAMS keys."""
    prm = params(scenario, online, **overrides)
    y = np.asarray(y, np.float64)
    P = band_power(y, Y, prm["band_hz"])
    N = len(P)
    if p is None:
        p = cm.speech_presence(P)[0] if trk is None else target_prob(trk, scenario)
    p = np.asarray(p, np.float64)
    keys = make_keys(keys, trk, scenario, p, prm)
    change = None if trk is None else getattr(trk, "change", None)
    gate = update_gate(p, keys, prm, change, online)
    upd = gate[0]
    nz = noise_level(P, p, prm) if noise_db is None else np.broadcast_to(np.asarray(noise_db, float), (N,))
    if leveler is None:
        leveler = Leveler(prm, None if online else room_gain(P, keys, upd, p, prm, nz))
        if not online and prm["prime_s"] > 0:
            leveler.prime(P, keys, upd, nz)
    tau = w = None
    if trk is not None and getattr(trk, "tau", None) is not None:
        tau, w = np.asarray(trk.tau, float), np.asarray(trk.directness, float)
    # offline the absolute cap uses the file's median noise level: a cap that follows the frame noise
    # estimate makes a capped gain follow the noise (measured on the bus: RDA P10 -0.35 dB, NM -0.2 dB)
    nz_cap = nz if online else np.full(N, float(np.median(nz)))
    agc_db = leveler.run(P, p, keys, nz, gate, tau, w, online, nz_cap)
    off_db = off_attenuation(p >= prm["p_speech"], prm, lookahead=not online)
    g = frame_gain(agc_db + off_db, len(y))
    out, g_lim = limiter(y * g, prm["ceiling_dbtp"], prm["oversample"], prm["lookahead_s"], prm["release_s"])
    diag = {"room_gain_db": leveler.room, "n_update": int(upd.sum()), "n_keys": len(leveler.mem),
            "limited_frac": float(np.mean(g_lim < 1.0)), "limiter_min_db": float(20 * np.log10(g_lim.min())),
            "keys": leveler.summary(float(max(prm["noise_max_db"] - nz_cap[-1], 0.0)))}
    return AgcResult(out=out, pre=y, gain=(g * g_lim).astype(np.float32), agc_db=agc_db, off_db=off_db,
                     level_db=dbfs(P), noise_db=np.asarray(nz, float), p=p, keys=keys, update=upd,
                     leveler=leveler, diag=diag)
