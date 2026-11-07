"""Speaker identity for S1 (DESIGN.md 2.0, 'Meeting talker identity'). API: methods/_core/API.md.

- Embedder hook: any callable y (1-D float, 16 kHz) -> unit-norm embedding. OnnxEmbedder wraps a
  future ONNX speaker-embedding model (e.g. a WeSpeaker / 3D-Speaker export, <= 30 MB) with Kaldi-style
  80-dim log-mel features; it is unvalidated until a model is approved (no model is installed).
- With an embedder: one embedding per 1.5 s of confident speech (p_s >= 0.95), online clustering by
  cosine similarity to centroids (join if >= 0.6, else a new cluster, at most 30).
- Fallback: DP-means on per-segment (TDOA, median CDR dB), thresholds 15 us and 4 dB, at most 12 clusters
  (online: one pass; offline: batch iterations). Expect merged talkers (20 seats ~ 8 TDOA groups).
- change_points(): a TDOA or CDR shift sustained for 250 ms, or speech onset after a 300 ms pause.
"""
import numpy as np

from . import common as cm

TAU_THR = 15e-6
CDR_THR = 4.0
MAX_DP = 12
COS_THR = 0.6
MAX_COS = 30
EMB_S = 1.5
EMB_P = 0.95
SPEECH_P = 0.5
PAUSE_S = 0.3
CHANGE_S = 0.25
MIN_SEG_S = 0.2
ONSET_SKIP = 6               # frames (48 ms) skipped at a segment start: coherence smoothing transient
ONLINE_CDR_SLACK = 2.0       # online decisions after 250 ms use 2 x the CDR threshold


# ------------------------------------------------------------------ embedding hook
def fbank(y, fs=cm.FS, n_mels=80):
    """Kaldi-style log-mel filterbank (25 ms Hamming frames, 10 ms hop, pre-emphasis 0.97),
    mean-normalized over time. Returns (frames, n_mels) float32."""
    n, hop, nfft = int(0.025 * fs), int(0.010 * fs), 512
    y = np.asarray(y, dtype=np.float64) * 32768.0
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    fr = np.lib.stride_tricks.sliding_window_view(y, n)[::hop].copy()
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= 0.97 * fr[:, :-1]
    spec = np.abs(np.fft.rfft(fr * np.hamming(n), nfft)) ** 2
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    edges = np.linspace(mel(20.0), mel(fs / 2 - 400.0), n_mels + 2)
    fm = mel(np.fft.rfftfreq(nfft, 1 / fs))
    lo, c, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    W = np.maximum(0.0, np.minimum((fm - lo) / (c - lo), (hi - fm) / (hi - c)))
    feats = np.log(np.maximum(spec @ W.T, 1e-10))
    return (feats - feats.mean(axis=0)).astype(np.float32)


class OnnxEmbedder:
    """Speaker-embedding hook: OnnxEmbedder(path)(y) -> unit-norm embedding (onnxruntime, CPU).
    Expects a model with one input of shape (1, frames, 80) fbank features."""

    def __init__(self, path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, y):
        e = self.sess.run(None, {self.inp: fbank(y)[None]})[0].ravel().astype(np.float64)
        return e / (np.linalg.norm(e) + 1e-12)


# ------------------------------------------------------------------ clusterers
class DPMeans:
    """Online DP-means on (tau, cdr_db): a point joins the nearest centroid if
    max(|d tau| / 15 us, |d cdr| / 4 dB) <= 1, otherwise opens a new cluster (at most 12)."""

    def __init__(self, tau_thr=TAU_THR, cdr_thr=CDR_THR, max_clusters=MAX_DP):
        self.scale = np.array([tau_thr, cdr_thr])
        self.max = max_clusters
        self.c = np.zeros((0, 2))
        self.n = np.zeros(0)

    def nearest(self, feat, slack=(1.0, 1.0)):
        """(index, normalized distance) of the nearest centroid, or (-1, inf); slack widens the
        (tau, cdr) thresholds."""
        if not len(self.c):
            return -1, np.inf
        d = np.max(np.abs(self.c - feat) / (self.scale * np.asarray(slack)), axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def assign(self, feat, weight=1.0, update=True, slack=(1.0, 1.0)):
        """Cluster index for feat (opening a new cluster if needed); updates its centroid if update."""
        feat = np.asarray(feat, dtype=float)
        i, d = self.nearest(feat, slack)
        if i < 0 or (d > 1.0 and len(self.c) < self.max):
            self.c = np.vstack([self.c, feat])
            self.n = np.append(self.n, 0.0)
            i = len(self.c) - 1
        if update:
            self.update(i, feat, weight)
        return i

    def update(self, i, feat, weight=1.0):
        """Move centroid i toward feat (running weighted mean)."""
        self.n[i] += weight
        self.c[i] += (np.asarray(feat, dtype=float) - self.c[i]) * weight / self.n[i]


class CosineClusterer:
    """Online clustering of unit-norm embeddings: join the most similar centroid if cos >= 0.6,
    else open a new cluster (at most 30)."""

    def __init__(self, thr=COS_THR, max_clusters=MAX_COS):
        self.thr, self.max = thr, max_clusters
        self.sum = []

    def assign(self, e, weight=1.0, update=True):
        e = np.asarray(e, dtype=float)
        if self.sum:
            C = np.array([s / (np.linalg.norm(s) + 1e-12) for s in self.sum])
            sim = C @ e
            i = int(np.argmax(sim))
            if sim[i] >= self.thr or len(self.sum) >= self.max:
                if update:
                    self.sum[i] = self.sum[i] + weight * e
                return i
        self.sum.append(weight * e)
        return len(self.sum) - 1


def dp_means_batch(feats, weights, tau_thr=TAU_THR, cdr_thr=CDR_THR, max_clusters=MAX_DP, iters=10):
    """Offline DP-means: online pass, then reassign / recompute centroids until stable."""
    dp = DPMeans(tau_thr, cdr_thr, max_clusters)
    lab = np.array([dp.assign(f, w) for f, w in zip(feats, weights)], dtype=int)
    for _ in range(iters):
        C = np.array([np.average(feats[lab == k], axis=0, weights=weights[lab == k]) if np.any(lab == k)
                      else dp.c[k] for k in range(len(dp.c))])
        new = np.argmin(np.max(np.abs(feats[:, None, :] - C[None]) / dp.scale, axis=2), axis=1)
        dp.c = C
        if np.array_equal(new, lab):
            break
        lab = new
    used = np.unique(lab)
    remap = {k: i for i, k in enumerate(used)}
    return np.array([remap[k] for k in lab], dtype=int), dp.c[used]


# ------------------------------------------------------------------ segmentation
def _speech_mask(vad):
    """Speech frames with gaps shorter than PAUSE_S bridged."""
    sp = vad > SPEECH_P
    gap = int(round(PAUSE_S / cm.HOP_S))
    idx = np.nonzero(sp)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            sp[a:b] = True
    return sp


def change_points(trk):
    """(N,) bool talker-change proposals: speech onset after a >= 300 ms pause, or the median TDOA /
    median CDR of the last 250 ms of speech moving more than 15 us / 4 dB away from the current
    segment's for a full 250 ms. Causal (marks the frame where the shift began)."""
    sp = _speech_mask(trk.vad)
    N, w = len(sp), int(round(CHANGE_S / cm.HOP_S))
    ch = np.zeros(N, bool)
    seg_start, run = -1, 0
    for l in range(N):
        if not sp[l]:
            seg_start, run = -1, 0
            continue
        if seg_start < 0:
            seg_start, run = l, 0
            ch[l] = True
            continue
        if l - seg_start < 2 * w:
            continue
        ref = slice(seg_start + ONSET_SKIP, l - w)
        cur = slice(l - w, l + 1)
        shift = (abs(np.median(trk.tau[cur]) - np.median(trk.tau[ref])) > TAU_THR
                 or abs(np.median(trk.cdr_db[cur]) - np.median(trk.cdr_db[ref])) > CDR_THR)
        run = run + 1 if shift else 0
        if run >= w:
            c = l - w
            ch[c] = True
            seg_start, run = c, 0
    return ch


def _segments(trk):
    """Speech segments [(a, b)) split at change points; segments shorter than MIN_SEG_S are dropped."""
    sp = _speech_mask(trk.vad)
    cut = trk.change | np.r_[True, ~sp[:-1]]
    segs, a = [], None
    for l in range(len(sp) + 1):
        if l == len(sp) or not sp[l] or (cut[l] and a is not None and l > a):
            if a is not None and (l - a) * cm.HOP_S >= MIN_SEG_S:
                segs.append((a, l))
            a = None
        if l < len(sp) and sp[l] and a is None:
            a = l
    return segs


def _seg_feature(trk, a, b):
    """(directness-weighted median TDOA, median CDR dB) of frames a..b, skipping the first
    ONSET_SKIP frames (the 50 ms smoothing transient) when the span allows."""
    a = a + ONSET_SKIP if b - a > 2 * ONSET_SKIP else a
    t, w = trk.tau[a:b], trk.directness[a:b] + 1e-6
    o = np.argsort(t)
    cw = np.cumsum(w[o])
    return np.array([t[o][np.searchsorted(cw, 0.5 * cw[-1])], float(np.median(trk.cdr_db[a:b]))])


def _regions(mask):
    """[(a, b)) runs of True in a boolean mask."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _emb_labels(trk, x, embedder, cl, online):
    """Embedding route: inside each speech region, one embedding per 1.5 s of confident speech
    (p_s >= 0.95), clustered online by cosine similarity. Offline, a chunk's label covers the region
    from the end of the previous chunk (or the region start) on; online it takes over when the chunk
    is complete. Regions without 1.5 s of confident speech keep the last label.
    Returns per-frame labels and [(a, b, cluster)] spans."""
    N = len(trk.vad)
    lab = np.full(N, -1, dtype=int)
    per = int(round(EMB_S / cm.HOP_S))
    cen = cm.frame_centers(N)
    spans, prev = [], -1
    for a, b in _regions(_speech_mask(trk.vad)):
        conf = np.nonzero(trk.vad[a:b] >= EMB_P)[0] + a
        chunks = [conf[i:i + per] for i in range(0, len(conf) - per + 1, per)]
        lab[a:b] = prev
        start = a
        for j, ch in enumerate(chunks):
            y = np.concatenate([x[max(0, c - cm.HOP // 2):max(0, c + cm.HOP // 2)] for c in cen[ch]])
            k = cl.assign(embedder(y))
            done = int(ch[-1]) + 1
            end = b if j == len(chunks) - 1 else done
            lab[(done if online and prev >= 0 else start):end] = k
            spans.append((start, end, k))
            start, prev = end, k
    return lab, spans


def cluster_talkers(trk, online=False, x=None, embedder=None, cos_thr=COS_THR):
    """Per-frame cluster labels (-1 outside speech and before the first cluster) and a summary list.
    With an embedder and the time signal x: the embedding route (_emb_labels, cosine threshold
    cos_thr). Otherwise DP-means on per-segment (TDOA, CDR): offline batch over all segments; online
    a segment keeps the previous label for its first 250 ms, then takes the cluster of its first
    250 ms (CDR threshold doubled, because the CDR of an onset is still rising)."""
    N = len(trk.vad)
    if embedder is not None and x is not None:
        lab, spans = _emb_labels(trk, x, embedder, CosineClusterer(cos_thr), online)
    else:
        lab, spans = np.full(N, -1, dtype=int), []
        segs = _segments(trk)
        feats = [_seg_feature(trk, a, b) for a, b in segs]
        if online:
            cl, prev = DPMeans(), -1
            w = int(round(CHANGE_S / cm.HOP_S))
            for (a, b), f in zip(segs, feats):
                at = min(b, a + w)
                k = cl.assign(_seg_feature(trk, a, at), update=False, slack=(1.0, ONLINE_CDR_SLACK))
                cl.update(k, f, b - a)
                lab[a:at], lab[at:b] = (prev, k) if prev >= 0 else (k, k)
                spans.append((a, b, k))
                prev = k
        elif segs:
            ks, _ = dp_means_batch(np.array(feats), np.array([b - a for a, b in segs], dtype=float))
            for (a, b), k in zip(segs, ks):
                lab[a:b] = k
                spans.append((a, b, int(k)))
    summary = []
    for k in sorted({k for _, _, k in spans}):
        sp = [(a, b) for a, b, kk in spans if kk == k]
        f = np.array([_seg_feature(trk, a, b) for a, b in sp])
        w = np.array([b - a for a, b in sp], dtype=float)
        summary.append({"id": int(k), "tau": float(np.average(f[:, 0], weights=w)),
                        "cdr_db": float(np.average(f[:, 1], weights=w)), "frames": int(w.sum()), "segments": len(sp)})
    return lab, summary
