"""Real-device evaluation scenes built from the hw/realset takes (+ simulated bus passengers / noise fields).

  .venv/bin/python -I hw/realset/build/realscenes.py build      # writes hw/realset/scenes/*.npz + scenes.json
  .venv/bin/python -I hw/realset/build/realscenes.py check      # sum(comp2) == mix, levels, realized SNRs

Storage: the components are stored once at unit gain (bus_base, bus_L<k>, meet_base, enroll) and every scene is a
set of group gains in scenes.json; load_scene() composes the scene dict that lab.metrics2 expects
(BENCH2.md 'Scene dict'): id, fs, mix, comp2, groups, refs, talker_group, segs, meta.

Conventions: ch0 = Mic A, ch1 = Mic B, float32, digital full scale 1.0, everything high-passed exactly like the takes
(4th-order 80 Hz zero-phase Butterworth, hw/realset.py images()). tau > 0 = Mic B hears later.
Simulated parts go through the 'device chain' of the session calibration: 1st-order 60 Hz high-pass per mic (the
calibration's nominal corner), Mic B level per band = -calib gain_db, Mic B shifted by tau0 (= -1.015 samples: B
leads), then the 80 Hz zero-phase high-pass.
"""
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # hw/realset
HW = os.path.dirname(ROOT)
LAB = os.path.dirname(HW)
OUT = os.path.join(ROOT, "scenes")
for p in (HW, LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import butter, lfilter, sosfiltfilt  # noqa: E402

FS, C = 16000, 343.0
XF = 800                                          # 50 ms crossfade
EARLY = 800                                       # early image: direct path + 50 ms (as scene2)
BUS_TAKES = ["A1", "B1", "A2", "B2", "A3", "B3"]
MEET_TAKES = ["M1", "M2", "M3", "M4", "M5", "M6", "MF1", "MF2"]
QUIET_DB = -12.0
BUS_SNRS = (-5.0, 0.0, 5.0)
N_PAX = {1: 3, 2: 4, 3: 5}                         # passengers per layout
PAX_TO_BUS_DB = 0.0                                # passengers (sum) re bus noise, in band, partner-active samples
OMEETING_DBFS = -55.0                              # in-band level of the synthesized OMEETING field at Mic A
_HP80 = butter(4, 80, "high", fs=FS, output="sos")
_BAND = butter(4, [150, 7000], "band", fs=FS, output="sos")
BANDS_HZ = 1000.0 * 2.0 ** (np.arange(-9, 9) / 3.0)


def _calib():
    with open(os.path.join(ROOT, "calib_session.json")) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- device chain
def frac_delay(x, tau, pad=64):
    """Delay x (..., T) by tau samples (may be negative / fractional) as linear phase, zero-padded both sides."""
    T = x.shape[-1]
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    n = int(2 ** np.ceil(np.log2(xp.shape[-1])))
    f = np.fft.rfftfreq(n)
    y = np.fft.irfft(np.fft.rfft(xp, n) * np.exp(-2j * np.pi * f * tau), n)
    return y[..., pad:pad + T]


def device_chain(x, cal=None, mic_only_a=False):
    """Acoustic 2-ch signal (2, T) at the mic positions -> what the board + take processing would store."""
    cal = cal or _calib()
    x = np.asarray(x, np.float64)
    b, a = butter(1, cal["hpf_corner_hz"][0], "high", fs=FS)
    if x.ndim == 1:                                   # Mic A only (references)
        return sosfiltfilt(_HP80, lfilter(b, a, x))
    y = np.stack([lfilter(b, a, x[0]), lfilter(*butter(1, cal["hpf_corner_hz"][1], "high", fs=FS), x[1])])
    T = y.shape[1]
    pad = 64
    n = int(2 ** np.ceil(np.log2(T + 2 * pad)))
    f = np.fft.rfftfreq(n, 1 / FS)
    g_b = -np.interp(np.log(np.maximum(f, 1.0)), np.log(BANDS_HZ), np.asarray(cal["gain_db"], float))
    tau = cal["tau0_s"] * FS
    YB = np.fft.rfft(np.pad(y[1], (pad, pad)), n) * 10 ** (g_b / 20) * np.exp(-2j * np.pi * (f / FS) * tau)
    y[1] = np.fft.irfft(YB, n)[pad:pad + T]
    return sosfiltfilt(_HP80, y, axis=-1)


def diffuse_field(n1, n2, d, model="sinc", nfft=512):
    """Habets & Gannot (2007/2008) coherence mixing: two independent noises -> 2-ch field with coherence
    sinc(2 f d / c) (3-D diffuse) at spacing d. STFT with sqrt-Hann windows, 50 % overlap."""
    hop = nfft // 2
    T = len(n1)
    w = np.sqrt(np.hanning(nfft + 1)[:-1])
    n_fr = (T + nfft) // hop + 1
    L = (n_fr - 1) * hop + nfft
    x = np.zeros((2, L + nfft))
    src = np.stack([np.pad(n1, (nfft, L - T)), np.pad(n2, (nfft, L - T))])
    f = np.fft.rfftfreq(nfft, 1 / FS)
    if model != "sinc":
        raise ValueError("only the 3-D diffuse (sinc) model is implemented")
    g = np.sinc(2 * f * d / C)
    a2 = np.sqrt(np.clip(1 - g ** 2, 0, 1))
    for i in range(n_fr):
        s = i * hop
        S = np.fft.rfft(src[:, s:s + nfft] * w, axis=-1)
        # the second noise contributes only its phase, at the first one's magnitude: with non-stationary noise
        # (two different stretches of DEMAND) plain mixing biases the coherence upward (measured: a 1.19 cm
        # target fitted as 0.97 cm); with equal magnitudes E[X1 X2*] / sqrt(E|X1|^2 E|X2|^2) = g exactly
        S2 = np.abs(S[0]) * np.exp(1j * np.angle(S[1]))
        X = np.stack([S[0], g * S[0] + a2 * S2])
        x[:, s:s + nfft] += np.fft.irfft(X, nfft, axis=-1) * w
    return x[:, nfft:nfft + T]


def inband_pow(x, mask=None):
    y = sosfiltfilt(_BAND, np.asarray(x, np.float64))
    return float(np.mean(y[mask] ** 2) if mask is not None else np.mean(y ** 2))


# ----------------------------------------------------------------------------- takes
def take_parts(take):
    """image (2, N), noise (2, N), early Mic-A image (N,), onset sample, early direct index, recording (2, N)."""
    from analyze import stretch
    z = np.load(os.path.join(ROOT, "img", take + ".npz"))
    st = np.load(os.path.join(ROOT, "stim", take + ".npz"))
    img, noise, h, onset = z["image"].T.astype(np.float64), z["noise"].T.astype(np.float64), z["h"].astype(np.float64), int(z["onset"])
    n0 = onset - 256 - int(0.25 * FS)
    s2 = stretch(st["played"].astype(np.float64), 17e-6)
    p = int(np.argmax(np.abs(h).sum(1)))
    early = np.zeros(img.shape[1])
    seg = np.convolve(s2, h[:p + EARLY, 0])[:max(0, img.shape[1] - n0)]
    early[n0:n0 + len(seg)] = seg
    full = np.zeros(img.shape[1])
    seg = np.convolve(s2, h[:, 0])[:max(0, img.shape[1] - n0)]
    full[n0:n0 + len(seg)] = seg
    err = 10 * np.log10(np.sum((full - img[0]) ** 2) / np.sum(img[0] ** 2) + 1e-30)
    return {"image": img, "noise": noise, "early": early, "onset": onset, "direct_idx": p,
            "rebuild_err_db": float(err), "recording": z["recording"].T.astype(np.float64)}


def concat(parts, windows):
    """Equal-power (sin/cos) crossfade concatenation. parts: list of dicts of arrays (..., N_k); windows: list of
    (a, b) cuts in each take. Returns (dict of arrays (..., T), positions [scene start of each cut])."""
    lens = [b - a for a, b in windows]
    pos = [0]
    for L in lens[:-1]:
        pos.append(pos[-1] + L - XF)
    T = pos[-1] + lens[-1]
    t = (np.arange(XF) + 0.5) / XF
    fin, fout = np.sin(np.pi / 2 * t), np.cos(np.pi / 2 * t)
    out = {}
    for k, (pt, (a, b), p0) in enumerate(zip(parts, windows, pos)):
        w = np.ones(b - a)
        if k > 0:
            w[:XF] = fin
        if k < len(parts) - 1:
            w[-XF:] = fout
        for key, v in pt.items():
            if key not in out:
                out[key] = np.zeros(v.shape[:-1] + (T,))
            out[key][..., p0:p0 + b - a] += v[..., a:b] * w
    return out, pos


# ----------------------------------------------------------------------------- corpora / noise
def _babble(rng, corpus, spk, T, max_utt_s=6.0):
    """Passenger speech over [0, T): (dry (T,), events [(start, end, utt_id)])."""
    utts = corpus.select(speaker=spk)
    order = rng.permutation(len(utts))
    x = np.zeros(T)
    ev = []
    t = int(rng.uniform(0.0, 2.0) * FS)
    k = 0
    while t < T - FS // 2:
        u = utts[order[k % len(order)]]
        k += 1
        a = np.asarray(corpus.audio(u.id), np.float64)
        n = min(len(a), int(max_utt_s * FS), T - t)
        if len(a) > n:
            o = int(rng.integers(0, len(a) - n + 1))
            a = a[o:o + n]
        else:
            a = a[:n]
        fl = min(320, n // 4)
        a = a.copy()
        a[:fl] *= np.linspace(0, 1, fl)
        a[-fl:] *= np.linspace(1, 0, fl)
        x[t:t + n] += a
        ev.append((t, t + n, u.id))
        t += n + int(rng.uniform(0.4, 2.5) * FS)
    return x, ev


def _demand_pair(env, T, rng, cha, chb):
    base = os.path.join(LAB, "data", f"demand_{env.lower()}", env)
    n_tot = sf.info(os.path.join(base, "ch01.wav")).frames
    o1 = int(rng.integers(0, n_tot - T))
    o2 = (o1 + int(rng.integers(20 * FS, n_tot - T - 20 * FS))) % (n_tot - T)  # >= 20 s apart: independent
    a = sf.read(os.path.join(base, f"ch{cha:02d}.wav"), start=o1, frames=T, dtype="float64")[0]
    b = sf.read(os.path.join(base, f"ch{chb:02d}.wav"), start=o2, frames=T, dtype="float64")[0]
    return a, b, {"env": env, "channels": [cha, chb], "offsets": [o1, o2]}


# ----------------------------------------------------------------------------- builders
def build_bus():
    from lab import corpus2, scene2
    from lab.common import PA_TO_DIG, spl_to_pa
    cal = _calib()
    d = cal["spacing_m"]
    parts = [take_parts(t) for t in BUS_TAKES]
    wins = [(p["onset"] - int(0.3 * FS), p["onset"] + int(4.45 * FS)) for p in parts]
    zero = None
    comps = []
    for t, p in zip(BUS_TAKES, parts):
        zero = np.zeros_like(p["image"])
        side = "A" if t.startswith("A") else "B"
        comps.append({f"img_{side}": p["image"], f"img_{'B' if side == 'A' else 'A'}": zero, "noise": p["noise"],
                      f"ref_{side}": p["early"], f"ref_{'B' if side == 'A' else 'A'}": np.zeros_like(p["early"])})
    cat, pos = concat(comps, wins)
    T = cat["noise"].shape[1]
    segs = []
    for t, p0 in zip(BUS_TAKES, pos):
        a = p0 + int(0.3 * FS)
        segs.append((a, min(a + int(4.2 * FS), T), "pA" if t.startswith("A") else "pB"))
    np.savez(os.path.join(OUT, "bus_base.npz"), partner_A=cat["img_A"].astype(np.float32),
             partner_B=cat["img_B"].astype(np.float32), device=cat["noise"].astype(np.float32),
             ref_pA=cat["ref_A"].astype(np.float32), ref_pB=cat["ref_B"].astype(np.float32))
    act = np.zeros(T, bool)
    for a, b, _ in segs:
        act[a:b] = True
    p_t = inband_pow(cat["img_A"][0] + cat["img_B"][0], act)
    corpus = corpus2.load_corpus("libri")
    babble_spk = corpus.speakers_for("babble", "test")
    layouts, scenes = {}, []
    for L in (1, 2, 3):
        rng = np.random.default_rng([20261103, L])
        dims = [round(float(v), 3) for v in np.array([11.5, 2.5, 2.3]) * rng.uniform(0.95, 1.05, 3)]
        rt60 = round(float(rng.uniform(0.35, 0.55)), 3)
        center = np.array([rng.uniform(3.0, dims[0] - 3.0), rng.uniform(0.8, dims[1] - 0.8), rng.uniform(0.9, 1.1)])
        az = float(rng.uniform(0, 360))
        mics = scene2.place_array(center, d, az, 0.0)
        pax, tries = [], 0
        while len(pax) < N_PAX[L] and tries < 10000:
            tries += 1
            r, phi = rng.uniform(0.8, 3.0), rng.uniform(0, 2 * np.pi)
            p = center + np.array([r * np.cos(phi), r * np.sin(phi), 0.0])
            p[2] = rng.uniform(1.05, 1.65)
            if not (0.3 <= p[0] <= dims[0] - 0.3 and 0.2 <= p[1] <= dims[1] - 0.2):
                continue
            if any(np.linalg.norm(p - q) < 0.6 for q in pax):
                continue
            pax.append(p)
        dirs = []
        for p in pax:                                   # passengers face a random horizontal direction
            a_ = rng.uniform(0, 2 * np.pi)
            dirs.append(scene2.facing(p, p + np.array([np.cos(a_), np.sin(a_), 0.0]), 0.75))
        seed = 7000 + L
        rirs, _, params = scene2.compute_rirs(dims, rt60, [p.tolist() for p in pax], mics, seed,
                                              tail="synthetic", directivity=dirs)
        spk = [babble_spk[i] for i in rng.choice(len(babble_spk), N_PAX[L], replace=False)]
        store, meta_pax, pax_segs = {}, {}, []
        pax_sum = np.zeros(T)
        for k, (p, s) in enumerate(zip(pax, spk)):
            tid = f"pax{k + 1}"
            dry, ev = _babble(np.random.default_rng([20261103, L, k]), corpus, s, T)
            lvl = float(rng.uniform(62.0, 72.0))
            dry = dry * spl_to_pa(lvl) * PA_TO_DIG
            truth = scene2.source_truth(p, mics)
            ac = np.stack([np.convolve(dry, rirs[k, c])[:T] for c in range(2)])
            sig = device_chain(ac, cal)
            da = truth["direct_a"]
            ref = device_chain(np.convolve(dry, rirs[k, 0, :int(np.ceil(da)) + EARLY])[:T], cal)
            store[f"near_{k + 1}"] = sig.astype(np.float32)
            store[f"ref_{tid}"] = ref.astype(np.float32)
            pax_sum += sig[0]
            sh = int(round(da))
            pax_segs += [(min(a + sh, T), min(b + sh, T), tid) for a, b, _ in ev if a + sh < T]
            meta_pax[tid] = {"group": f"near_{k + 1}", "speaker": s, "level_db": round(lvl, 2),
                             "pos": [round(float(v), 3) for v in p], "directivity": dirs[k],
                             "utts": [u for _, _, u in ev], **truth}
        n1, n2, nmeta = _demand_pair("TBUS", T, rng, 1, 16)
        bus = device_chain(diffuse_field(n1, n2, d), cal)
        # bus noise level: passengers (sum) : bus = PAX_TO_BUS_DB in band over partner-active samples
        p_pax = inband_pow(pax_sum, act)
        p_bus = inband_pow(bus[0], act)
        bus *= np.sqrt(p_pax / (p_bus * 10 ** (PAX_TO_BUS_DB / 10)))
        store["bus_noise"] = bus.astype(np.float32)
        np.savez(os.path.join(OUT, f"bus_L{L}.npz"), **store)
        p_n = inband_pow(pax_sum + bus[0], act)
        layouts[f"L{L}"] = {"dims": dims, "rt60": rt60, "center": center.round(3).tolist(), "array_azimuth_deg": az,
                            "mics": mics, "spacing_m": d, "rir": params, "passengers": meta_pax,
                            "pax_segs": sorted(pax_segs), "bus_noise": dict(nmeta, model="sinc", spacing_m=d),
                            "pax_to_bus_db": PAX_TO_BUS_DB}
        for snr in BUS_SNRS:
            g = float(np.sqrt(p_t / (p_n * 10 ** (snr / 10))))
            for dev in ("recorded", "quiet"):
                scenes.append({"id": f"bus/L{L}_snr{snr:+.0f}_{dev}", "set": "bus", "layout": f"L{L}", "snr_db": snr,
                               "device_noise": dev, "gain_sim": g,
                               "gain_device": 1.0 if dev == "recorded" else 10 ** (QUIET_DB / 20)})
    base = {"takes": BUS_TAKES, "windows": wins, "positions": pos, "segs": segs, "T": T,
            "rebuild_err_db": {t: p["rebuild_err_db"] for t, p in zip(BUS_TAKES, parts)},
            "onsets": {t: p["onset"] for t, p in zip(BUS_TAKES, parts)}}
    return {"bus_base": base, "layouts": layouts}, scenes


def build_meeting():
    cal = _calib()
    d = cal["spacing_m"]
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    parts = [take_parts(t) for t in MEET_TAKES]
    rng = np.random.default_rng([20261103, 99])
    pauses = rng.uniform(0.3, 0.8, len(MEET_TAKES) - 1)
    pre = [0.3] + [None] * (len(MEET_TAKES) - 1)
    post = [None] * (len(MEET_TAKES) - 1) + [0.45]
    for k, pz in enumerate(pauses):                      # split each pause (incl. the crossfade) over both margins
        tot = pz + XF / FS
        post[k] = max(0.1, tot / 2)
        pre[k + 1] = tot - post[k]
    wins = [(p["onset"] - int(round(a * FS)), p["onset"] + int(4.2 * FS) + int(round(b * FS)))
            for p, a, b in zip(parts, pre, post)]
    comps = []
    for k, p in enumerate(parts):
        dct = {"noise": p["noise"]}
        for j in range(len(parts)):
            dct[f"img_{j}"] = p["image"] if j == k else np.zeros_like(p["image"])
            dct[f"ref_{j}"] = p["early"] if j == k else np.zeros_like(p["early"])
        comps.append(dct)
    cat, pos = concat(comps, wins)
    T = cat["noise"].shape[1]
    segs, talkers, store = [], {}, {}
    for k, (t, p0, a) in enumerate(zip(MEET_TAKES, pos, pre)):
        s0 = p0 + int(round(a * FS))
        segs.append((s0, min(s0 + int(4.2 * FS), T), t))
        store[f"talker_{k:02d}"] = cat[f"img_{k}"].astype(np.float32)
        store[f"ref_{t}"] = cat[f"ref_{k}"].astype(np.float32)
        mt = man[t]
        talkers[t] = {"group": f"talker_{k:02d}", "speaker": mt["speaker"], "side": mt["side"],
                      "rel_level_db": mt["level_db"], "level_db": 60.0 + mt["level_db"],
                      "dist_m": 4.0 if mt["kind"] == "far" else 0.5, "kind": mt["kind"], "utts": mt["utts"]}
    store["device"] = cat["noise"].astype(np.float32)
    n1, n2, nmeta = _demand_pair("OMEETING", T, rng, 1, 16)
    om = device_chain(diffuse_field(n1, n2, d), cal)
    om *= 10 ** (OMEETING_DBFS / 20) / np.sqrt(inband_pow(om[0]))
    store["omeeting"] = om.astype(np.float32)
    np.savez(os.path.join(OUT, "meet_base.npz"), **store)
    base = {"takes": MEET_TAKES, "windows": wins, "positions": pos, "pauses_s": pauses.round(3).tolist(),
            "segs": segs, "T": T, "talkers": talkers, "omeeting": dict(nmeta, model="sinc", spacing_m=d,
                                                                         level_dbfs_inband=OMEETING_DBFS),
            "rebuild_err_db": {t: p["rebuild_err_db"] for t, p in zip(MEET_TAKES, parts)}}
    scenes = [{"id": "meeting/recorded", "set": "meeting", "device_noise": "recorded", "gain_device": 1.0,
               "omeeting": False},
              {"id": "meeting/quiet", "set": "meeting", "device_noise": "quiet", "gain_device": 10 ** (QUIET_DB / 20),
               "omeeting": False},
              {"id": "meeting/quiet_omeeting", "set": "meeting", "device_noise": "quiet",
               "gain_device": 10 ** (QUIET_DB / 20), "omeeting": True}]
    return {"meet_base": base}, scenes


def build_enroll():
    out, meta = {}, {}
    for spk, takes in (("A", ("AE1", "AE2")), ("B", ("BE1", "BE2"))):
        clean = [np.load(os.path.join(ROOT, "stim", t + ".npz"))["dry"].astype(np.float64) for t in takes]
        dev = []
        for t in takes:
            z = np.load(os.path.join(ROOT, "img", t + ".npz"))
            on = int(z["onset"])
            dev.append(z["recording"][on - int(0.25 * FS):on + int(4.45 * FS), 0].astype(np.float64))
        out[f"clean_{spk}"] = np.concatenate(clean).astype(np.float32)
        out[f"device_{spk}"] = np.concatenate(dev).astype(np.float32)
        meta[spk] = {"takes": list(takes), "clean_s": len(out[f"clean_{spk}"]) / FS,
                     "device_s": len(out[f"device_{spk}"]) / FS}
    np.savez(os.path.join(OUT, "enroll.npz"), **out)
    return meta


def build():
    os.makedirs(OUT, exist_ok=True)
    info_bus, sc_bus = build_bus()
    info_meet, sc_meet = build_meeting()
    enr = build_enroll()
    doc = {"version": 1, "fs": FS, "calib": _calib(), "quiet_db": QUIET_DB, "bus": info_bus["bus_base"],
           "layouts": info_bus["layouts"], "meeting": info_meet["meet_base"], "enroll": enr,
           "scenes": sc_bus + sc_meet}
    with open(os.path.join(OUT, "scenes.json"), "w") as fh:
        json.dump(doc, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"built {len(doc['scenes'])} scenes into {OUT}")
    return doc


# ----------------------------------------------------------------------------- loading
_DOC, _NPZ = None, {}


def _doc():
    global _DOC
    if _DOC is None:
        with open(os.path.join(OUT, "scenes.json")) as fh:
            _DOC = json.load(fh)
    return _DOC


def _npz(name):
    if name not in _NPZ:
        z = np.load(os.path.join(OUT, name + ".npz"))
        _NPZ[name] = {k: z[k] for k in z.files}
    return _NPZ[name]


def list_scenes(set_name=None):
    return [s["id"] for s in _doc()["scenes"] if set_name in (None, "all", s["set"])]


def scene_card(scene_id):
    return dict(next(s for s in _doc()["scenes"] if s["id"] == scene_id))


def load_enroll(kind="clean"):
    z = _npz("enroll")
    return {"A": np.array(z[f"{kind}_A"], np.float64), "B": np.array(z[f"{kind}_B"], np.float64)}


def load_scene(scene_id):
    """Scene dict in the lab.metrics2 format (BENCH2.md 'Scene dict'); arrays are float32, mix = sum(comp2)."""
    doc = _doc()
    card = scene_card(scene_id)
    comp2, groups, refs, tg, segs = {}, {}, {}, {}, []
    talkers = {}
    if card["set"] == "bus":
        base, lay = _npz("bus_base"), doc["layouts"][card["layout"]]
        L = _npz(f"bus_{card['layout']}")
        comp2["partner_A"], groups["partner_A"] = base["partner_A"], "partner"
        comp2["partner_B"], groups["partner_B"] = base["partner_B"], "partner"
        refs["pA"], refs["pB"] = base["ref_pA"], base["ref_pB"]
        tg.update(pA="partner_A", pB="partner_B")
        talkers["pA"] = {"group": "partner_A", "speaker": "121", "dist_m": float("nan"), "takes": ["A1", "A2", "A3"]}
        talkers["pB"] = {"group": "partner_B", "speaker": "1089", "dist_m": float("nan"), "takes": ["B1", "B2", "B3"]}
        g = np.float32(card["gain_sim"])
        for tid, m in lay["passengers"].items():
            grp = m["group"]
            comp2[grp], groups[grp] = L[grp] * g, "passenger_near"
            refs[tid] = L[f"ref_{tid}"] * g
            tg[tid] = grp
            talkers[tid] = {k: m[k] for k in ("group", "speaker", "level_db", "dist_m", "theta_deg", "tau_us")}
        comp2["bus_noise"], groups["bus_noise"] = L["bus_noise"] * g, "engine_road"
        comp2["device"], groups["device"] = base["device"] * np.float32(card["gain_device"]), "device_noise"
        segs = [tuple(s) for s in doc["bus"]["segs"]] + [tuple(s) for s in lay["pax_segs"]]
        meta = {"scenario": "bus", "snr_db": card["snr_db"], "layout": card["layout"], "rt60": lay["rt60"],
                "device_noise": card["device_noise"]}
    else:
        base, mb = _npz("meet_base"), doc["meeting"]
        for t, m in mb["talkers"].items():
            comp2[m["group"]], groups[m["group"]] = base[m["group"]], "talker"
            refs[t] = base[f"ref_{t}"]
            tg[t] = m["group"]
            talkers[t] = dict(m)
        if card.get("omeeting"):
            comp2["omeeting"], groups["omeeting"] = base["omeeting"], "hvac"
        comp2["device"], groups["device"] = base["device"] * np.float32(card["gain_device"]), "device_noise"
        segs = [tuple(s) for s in mb["segs"]]
        meta = {"scenario": "meeting", "device_noise": card["device_noise"], "omeeting": bool(card.get("omeeting"))}
    comp2 = {k: np.ascontiguousarray(v, np.float32) for k, v in comp2.items()}
    T = next(iter(comp2.values())).shape[1]
    mix = np.zeros((2, T), np.float32)
    for v in comp2.values():
        mix += v
    segs = sorted((int(a), int(b), str(t)) for a, b, t in segs if b > a)
    meta.update(scene_id=scene_id, set=card["set"], n_samples=T, duration_s=T / FS, fs=FS,
                spacing_m=doc["calib"]["spacing_m"], talkers=talkers, card=card, corpus="real")
    sc = {"id": scene_id, "fs": FS, "mix": mix, "comp2": comp2, "groups": groups,
          "refs": {k: np.ascontiguousarray(v, np.float32) for k, v in refs.items()}, "talker_group": tg,
          "segs": segs, "meta": meta}
    for v in list(sc["comp2"].values()) + list(sc["refs"].values()) + [mix]:
        v.setflags(write=False)
    return sc


def check():
    """Consistency report: sum(comp2) == mix, realized SNR, levels per component (dBFS in band at Mic A)."""
    rep = {}
    for sid in list_scenes():
        sc = load_scene(sid)
        T = sc["mix"].shape[1]
        acc = np.zeros((2, T), np.float32)
        for v in sc["comp2"].values():
            acc += v
        act = np.zeros(T, bool)
        tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] in ("partner", "talker")]
        for a, b, t in sc["segs"]:
            if t in tids:
                act[a:b] = True
        lv = {g: round(10 * np.log10(inband_pow(v[0], act) + 1e-30), 1) for g, v in sc["comp2"].items()}
        r = {"T_s": round(T / FS, 2), "sum_err": float(np.max(np.abs(acc - sc["mix"]))), "lvl_dbfs_active": lv}
        if sc["meta"]["scenario"] == "bus":
            tgt = sc["comp2"]["partner_A"][0] + sc["comp2"]["partner_B"][0]
            nz = sum(v[0] for g, v in sc["comp2"].items() if sc["groups"][g] in ("passenger_near", "engine_road"))
            r["snr_rule_db"] = round(10 * np.log10(inband_pow(tgt, act) / inband_pow(nz, act)), 2)
            allnz = nz + sc["comp2"]["device"][0]
            r["snr_incl_device_db"] = round(10 * np.log10(inband_pow(tgt, act) / inband_pow(allnz, act)), 2)
        else:
            nz = sum(v[0] for g, v in sc["comp2"].items() if sc["groups"][g] != "talker")
            for t in tids:
                m = np.zeros(T, bool)
                for a, b, tt in sc["segs"]:
                    if tt == t:
                        m[a:b] = True
                r.setdefault("talker_snr_db", {})[t] = round(10 * np.log10(
                    inband_pow(sc["comp2"][sc["talker_group"][t]][0], m) / inband_pow(nz, m)), 1)
        rep[sid] = r
    with open(os.path.join(OUT, "check.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    for k, v in rep.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "lvl_dbfs_active"}))
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
        check()
    elif cmd == "check":
        check()
