"""x_meeting_v3_stack: configurable final meeting pipeline family (final:meeting, 2026-11-03; study/final/MEETING_V3.md).
Not a method by itself (enhance() runs DEFAULT); the variants methods/x_meeting_v3_<tag>.py call enhance_cfg().

DESIGN 2.2 chain, every stage switchable through a cfg dict:
  S0   'core': calib.py blind S0 (80 Hz HPF, per-band EQ, skew in {0, +0.5}); 'v2': x_s0v2 blind self-calibration
       (any spacing / skew) when info has no calib_json. A given calib_json is always used (gamma_n added by
       x_combo_common.gn_fix when missing).
  M10  two-channel offline WPE (methods._core.wpe defaults) on the calibrated STFT, both channels kept (wpe=False skips).
  S1   tracker.track on the WPE output (tau, DP-means clusters, change proposals, VAD).
  M2   beam: 'steered' (WNG >= -3 dB superdirective MVDR toward the active talker, 400 ms hold, 50 ms crossfades:
       beams.meeting_beam), 'fixed3' (3 fixed beams + DESIGN best-beam selection) or 'none' (WPE Mic A).
  NS   'm6' (MMSE-SPP/LSA, floor -12 dB, noise update only after >= 300 ms of S1-VAD pause), 'gtcrn' / 'dpdfnet'
       (M8 with the attenuation limit nn_atten_db, replacing M6), 'm6+gtcrn' (M8 GTCRN after M6), 'none'.
  M11  agc.level (speech-gated, noise-capped, tuned meeting parameters, -1 dBTP limiter); keys: 'segment' (default:
       one static gain per S1 speech segment), 'cluster' (DP-means TDOA/CDR clusters), 'emb' (WeSpeaker ResNet34
       embedding clusters, spk_onnx, cosine threshold THETA_S = 0.375, embedded on the WPE Mic A signal), 'embhyb'
       (embedding cluster where the speech region had its own >= 1.5 s embedding chunk, else the region's own S1
       segment key); cfg['agc']: agc.level parameter overrides (default none: the tuned meeting set).
       Returns {'out', 'pre_agc'}; pre_agc is the NS output, the signal entering M11.
run_all(x, fs, info, cfgs) computes several variants on one input and shares every identical prefix (front end,
beam, NS), so a variant's output equals enhance_cfg(x, fs, info, cfg) bit for bit (deterministic, stateless: the
only module-level object is the ONNX speaker-model session, which holds no signal state).
"""
import numpy as np

from methods import x_combo_common as xc
from methods import x_s0v2
from methods._core import agc, beams, calib, common as cm, nn, ns, spk, tracker, wpe

DEFAULT = dict(s0="core", wpe=True, beam="steered", ns="m6", nn_atten_db=12.0, keys="segment", agc={})
_EMB = {}


def _embedder():
    if "w" not in _EMB:
        from methods._core import spk_onnx
        _EMB["w"] = spk_onnx.WeSpeakerEmbedder(threads=1)
        _EMB["thr"] = spk_onnx.THETA_S
    return _EMB["w"], _EMB["thr"]


def full_cfg(cfg):
    c = dict(DEFAULT)
    c.update(cfg or {})
    return c


class Front:
    def __init__(self, x, fs, info, s0="core", use_wpe=True):
        self.scen = info.get("scenario", "meeting")
        x = np.asarray(x, np.float64)
        self.T = x.shape[-1]
        if s0 == "v2" and info.get("calib_json") is None:
            info = dict(info, calib_json=x_s0v2.estimate(x, fs, info))
        info = xc.gn_fix(x, fs, info)
        self.fs = fs
        self.xh = calib.prefilter(x, self.scen)
        X0 = cm.stft(self.xh)
        self.cal = calib.calibrate(X0, info)
        X = calib.apply(X0, self.cal)
        self.Xw = wpe.wpe_offline(X, fs=fs) if use_wpe else X
        self.trk = tracker.track(self.Xw, self.cal, info, x=self.xh[0])
        self._yA = None

    def mic_a(self):
        """WPE (or plain) Mic A, time domain: the signal the speaker embeddings see."""
        if self._yA is None:
            self._yA = cm.istft(self.Xw[0], self.T)
        return self._yA


def beam_stft(fe, mode):
    if mode == "none":
        return fe.Xw[0]
    b = beams.meeting_beam(fe.Xw, fe.cal, fe.trk, scenario=fe.scen, mode=mode)
    return b.apply(fe.Xw)


def ns_out(fe, Yb, kind, atten_db):
    T = fe.T
    if kind == "none":
        return cm.istft(Yb, T)
    if kind in ("m6", "m6+gtcrn"):
        y = cm.istft(ns.suppress(Yb, fe.scen, target_prob=fe.trk.vad, fs=fe.fs)["S"], T)
        if kind == "m6":
            return y
        return nn.post_filter(y, "gtcrn", atten_db=atten_db)["y"]
    if kind in ("gtcrn", "dpdfnet"):
        return nn.post_filter(cm.istft(Yb, T), kind, atten_db=atten_db, X_lab=Yb)["y"]
    raise ValueError(f"unknown ns {kind!r}")


def emb_keys(fe, hybrid):
    """Per-frame AGC keys from WeSpeaker clusters of the WPE Mic A signal (spk._emb_labels, offline)."""
    emb, thr = _embedder()
    lab, spans = spk._emb_labels(fe.trk, fe.mic_a(), emb, spk.CosineClusterer(thr), False)
    if not hybrid:
        return lab
    own = np.zeros(len(lab), bool)
    for a, b, _ in spans:
        own[a:b] = True
    seg = agc.segment_keys(fe.trk)
    off = int(lab.max()) + 1 if lab.size and lab.max() >= 0 else 0
    return np.where(lab < 0, -1, np.where(own, lab, np.where(seg >= 0, seg + off, -1)))


def level(fe, y, keys, key_arr=None, overrides=None):
    k = key_arr if key_arr is not None else keys
    res = agc.level(y, fe.trk, fe.scen, Y=cm.stft(y), keys=k, **(overrides or {}))
    return res.out, res.pre


def run_all(x, fs, info, cfgs, timing=None):
    """{tag: cfg} -> {tag: {'out', 'pre_agc'}}, sharing identical prefixes. timing: optional dict, filled with
    {tag: seconds of the stages on that variant's own path} (shared stages counted for every variant using them)."""
    import time
    cfgs = {t: full_cfg(c) for t, c in cfgs.items()}
    memo, cost, out = {}, {}, {}

    def get(key, fn):
        if key not in memo:
            t0 = time.perf_counter()
            memo[key] = fn()
            cost[key] = time.perf_counter() - t0
        return memo[key]

    for tag, c in cfgs.items():
        fk = ("front", c["s0"], c["wpe"])
        fe = get(fk, lambda: Front(x, fs, info, c["s0"], c["wpe"]))
        bk = fk + (c["beam"],)
        Yb = get(bk, lambda: beam_stft(fe, c["beam"]))
        nk = bk + (c["ns"], c["nn_atten_db"] if c["ns"] not in ("m6", "none") else None)
        y_ns = get(nk, lambda: ns_out(fe, Yb, c["ns"], c["nn_atten_db"]))
        path = [fk, bk, nk]
        karr = None
        if c["keys"] in ("emb", "embhyb"):
            kk = fk + ("keys", c["keys"])
            karr = get(kk, lambda: emb_keys(fe, c["keys"] == "embhyb"))
            path.append(kk)
        ak = ("agc", tag)
        y, pre = get(ak, lambda: level(fe, y_ns, c["keys"], karr, c["agc"]))
        path.append(ak)
        out[tag] = {"out": y, "pre_agc": pre}
        if timing is not None:
            timing[tag] = timing.get(tag, 0.0) + sum(cost[k] for k in path)
    if timing is not None:
        timing["_stages"] = {"/".join(map(str, k)): v for k, v in cost.items()}
    return out


def enhance_cfg(x, fs, info, cfg, return_stages=False):
    r = run_all(x, fs, info, {"v": cfg})["v"]
    return xc.stages(r["out"], r["pre_agc"], return_stages)


def enhance(x, fs, info, return_stages=False):
    return enhance_cfg(x, fs, info, DEFAULT, return_stages)
