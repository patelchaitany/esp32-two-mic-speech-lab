"""x_bus_v3_stack: config-driven final BUS pipeline family (DESIGN.md 2.1), night-B final:bus. Not a method by
itself: enhance() runs DEFAULT; methods/x_bus_v3*.py bind a config (report: study/final/BUS_V3.md).

Chain (every stage switchable through CFG keys):
  calib  'auto': info['calib_json'] if given (gamma_n repaired as x_combo_common.gn_fix), else a blind x_s0v2
         estimate from this very input (continuous skew, any spacing; stateless: each call re-estimates);
         's0': the default blind S0 (skew in {0, +0.5} samples, D_RANGE 1.5-8 cm).
  S0 + S1 tracker.front_end (offline, partner model on).
  spatial 'm3': RTF-MVDR (methods._core.rtf.run, its tuned defaults and inseparable fallback: Mic A or broadside);
          'm2': M2 two-talker WNG-constrained beam (beams.bus_beam, states 'ab', same-side / endfire gates);
          'mica': no spatial stage (S0-prefiltered Mic A) = the DESIGN inseparable fallback.      -> Y_lin
  masks, each estimated from LINEAR signals (the calibrated mic pair X or Y_lin), combined ONCE:
          G_cdr  M4 DOA-free CDR gain from the mic pair (masks.CDR_PARAMS, tuned);
          G_fen  M5 phase fence under S1's partner TDOAs (masks.FENCE_PARAMS);
          G_ns   M6 MMSE-SPP/LSA gain on Y_lin with the noise update frozen where the pair's direct-to-total
                 ratio CDR/(1+CDR) > 0.6 (spatial gating, ns 'hold' gate), no own floor / time rule;
          G_tse  M9 union of both partners (tse.enhance, info['enroll'] when given, else self-enrollment; the
                 mask |Y9| / |X_A| floored at -TSE_LIMIT dB);
          m_c = max(floor, product), instant rise, fall <= 4.8 dB per 8 ms hop (ns.limit_gain).
  M8 (optional) on the TARGET stream istft(m_c Y_lin): GTCRN or DPDFNet-4 raw lab-grid mask, floored at -nn_limit dB,
          m = max(floor, m_c * G_nn) with the same time rule.
  split remix: T = m Y_lin; R = (1 - m) Y_lin -> M6 (bus defaults) -> slow 2:1 compression of its level around its
          10 s running level (1 s detector, gain -12..+6 dB) -> mixed at rho = remix_db (default -15 dB). -> y_pre
  M11-bus: agc.level(y_pre, S1 track, 'bus'): level updates only when p_A + p_B > 0.8, noise cap, -1 dBTP limiter.
Returns y or {'out': y, 'pre_agc': y_pre} (y_pre enters the AGC, the last level-control stage).
Deterministic and stateless; float64 inside; output length = input length.
"""
import numpy as np

from methods import x_combo_common as xc
from methods import x_s0v2
from methods._core import agc, beams, common as cm, masks, nn, ns, rtf, tracker

DEFAULT = dict(
    calib="auto",        # 'auto' (x_s0v2 when no calib_json) | 's0'
    spatial="m3",        # 'm3' | 'm2' | 'mica'
    cdr=True,
    cdr_df=False,        # True: M4 CDR scaled by the M3 beam's directivity factor (DESIGN M4 correction; m3 only)
    fence=False,
    ns=True,
    ns_cue="cdr",        # 'cdr' (per-bin direct-to-total ratio of the pair) | 'post' (p_A + p_B) | None
    tse=False,
    tse_limit_db=20.0,
    nn=None,             # None | 'gtcrn' | 'dpdfnet'
    nn_input="target",   # 'target' (M8 on m_c * Y_lin) | 'lin' (M8 on Y_lin, mask joins the product)
    nn_limit_db=20.0,
    floor_db=-15.0,      # floor of the combined mask
    fall_db=4.8,
    remix_db=-15.0,      # None = no residual path
    agc=True,
)
CUE_THR = 0.6
COMP = dict(ratio=2.0, fast_s=1.0, slow_s=10.0, min_db=-12.0, max_db=6.0, band=(300.0, 4000.0))


def config(**kw):
    bad = set(kw) - set(DEFAULT)
    if bad:
        raise KeyError(f"unknown x_bus_v3 config keys {sorted(bad)}")
    return dict(DEFAULT, **kw)


def latency_ms(cfg):
    """Algorithmic latency of the online form of the chain: 32 ms STFT + M8 model delay + 5 ms limiter look-ahead.
    (The bench runs the OFFLINE form: whole-file S0/S1/M3 statistics.)"""
    lat = 1000.0 * cm.N_FFT / cm.FS + (5.0 if cfg["agc"] else 0.0)
    if cfg["nn"]:
        lat += nn.DELAY_FRAMES[cfg["nn"]] * 1000.0 * cm.HOP / cm.FS
    if cfg["tse"]:
        lat = max(lat, 856.0)
    return lat


def _calib_info(x, fs, info, cfg):
    if cfg["calib"] == "s0":
        return info
    if info.get("calib_json") is None:
        return dict(info, calib_json=x_s0v2.estimate(x, fs, info))
    return xc.gn_fix(x, fs, info)


def _spatial(fe, info, cfg):
    kind = cfg["spatial"]
    if kind == "m3":
        b = rtf.run(fe, info)
        return b.Y, {"df": b.df_frames() if getattr(b, "df", None) is not None else None}
    if kind == "m2":
        b = beams.bus_beam(fe.X, fe.calib, fe.track, scenario="bus", states="ab")
        return b.apply(fe.X), {"fallback_frac": float(b.diag.get("fallback_frac", np.nan))}
    if kind == "mica":
        return fe.X[0].copy(), {}
    raise ValueError(f"unknown spatial stage {kind!r}")


def _band_power(Y, band):
    sl = cm.bins_between(*band)
    return np.mean(np.abs(Y[sl]) ** 2, axis=0) + 1e-20


def _slow_compress(R):
    """Per-frame gain (N,) of a slow 2:1 compressor of R's 300-4000 Hz level around its 10 s running level."""
    P = _band_power(R, COMP["band"])
    fast, _ = cm.smooth(P, cm.lam(COMP["fast_s"]), init=float(P[: max(1, len(P) // 50)].mean()))
    slow, _ = cm.smooth(P, cm.lam(COMP["slow_s"]), init=float(P.mean()))
    g_db = -(1.0 - 1.0 / COMP["ratio"]) * 10.0 * np.log10(np.maximum(fast, 1e-20) / np.maximum(slow, 1e-20))
    return 10.0 ** (np.clip(g_db, COMP["min_db"], COMP["max_db"]) / 20.0)


def run(x, fs, info, cfg, return_stages=False):
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[-1]
    scen = info.get("scenario", "bus")
    if T < cm.N_FFT:                                       # nothing to estimate: prefiltered Mic A
        from methods._core import calib
        y = calib.prefilter(x, scen)[0] if T else np.zeros(0)
        return {"out": y, "pre_agc": y} if return_stages else y
    info2 = _calib_info(x, fs, info, cfg)
    fe = tracker.front_end(x, fs, info2, partners=True, talkers=False)
    cal = fe.calib
    Y, sdiag = _spatial(fe, info2, cfg)
    f = cm.FREQS
    G = np.ones(Y.shape)
    if cfg["cdr"]:
        G *= masks.cdr_gain(fe.X, cal, None, df=sdiag.get("df") if cfg["cdr_df"] else None)
    if cfg["fence"]:
        G *= masks.fence_gain(fe.X, cal, masks.partner_taus(fe.track, cal))
    if cfg["ns"]:
        cue = None
        if cfg["ns_cue"] == "cdr":
            cdr = masks.cdr_map(fe.X, cal)
            cue = cdr / (1.0 + cdr)
        elif cfg["ns_cue"] == "post":
            cue = np.clip(fe.track.p_A + fe.track.p_B, 0.0, 1.0)
        G *= ns.suppress(Y, "bus", target_prob=cue, floor_db=None, fall_db=None)["G"]
    if cfg["tse"]:
        from methods._core import tse
        y9 = np.asarray(tse.enhance(x, fs, info, mode="explicit", return_stages=False), np.float64)
        m9 = np.clip(np.abs(cm.stft(y9)) / np.maximum(np.abs(fe.X[0]), 1e-10), 0.0, 1.0)
        G *= np.maximum(m9, 10.0 ** (-cfg["tse_limit_db"] / 20.0))
    if cfg["nn"] and cfg["nn_input"] == "lin":
        pf = nn.post_filter_stft(Y, T, cfg["nn"], atten_db=None)
        G *= np.maximum(pf["mask_raw"], nn.floor_of(cfg["nn_limit_db"]))
    m = ns.limit_gain(G, cfg["floor_db"], cfg["fall_db"])
    if cfg["nn"] and cfg["nn_input"] == "target":
        pf = nn.post_filter_stft(m * Y, T, cfg["nn"], atten_db=None)
        m = ns.limit_gain(m * np.maximum(pf["mask_raw"], nn.floor_of(cfg["nn_limit_db"])), cfg["floor_db"],
                          cfg["fall_db"])
    Yt = m * Y
    if cfg["remix_db"] is not None:
        R = (1.0 - m) * Y
        Rn = ns.suppress(R, "bus")["S"]
        Yt = Yt + (10.0 ** (cfg["remix_db"] / 20.0)) * _slow_compress(Rn)[None, :] * Rn
    y_pre = cm.istft(Yt, T)
    if cfg["agc"]:
        y = agc.level(y_pre, fe.track, "bus").out
    else:
        y = y_pre
    y = np.nan_to_num(np.asarray(y, np.float64))
    if return_stages:
        return {"out": y, "pre_agc": y_pre}
    return y


def enhance(x, fs, info, return_stages=False):
    return run(x, fs, info, DEFAULT, return_stages)
