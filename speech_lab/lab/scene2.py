"""v2 scene format and rendering core (DESIGN 4.1 E5-E9, 4.2). Format and API: BENCH2.md "Scene format".

A scene is a geometry (RIRs for every point source plus positions: scenes2/<set>/<geom_id>.geom.{npz,json})
and one recipe per corpus (scenes2/<set>/<name>.<corpus>.recipe.json: utterances, timings, noise seeds,
levels, impairments). Scenes render deterministically on load; rendered audio is never stored.
"""
import argparse
import bisect
import copy
import functools
import importlib
import json
import os
import time
import zlib

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.signal import butter, lfilter, oaconvolve, sosfiltfilt
from scipy.special import j0

from . import corpus2
from .common import C, FS, PA_TO_DIG, rms, spl_to_pa
from .scenes import coherent_noise

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(LAB, "scenes2")
FRAC = pra.constants.get("frac_delay_length") // 2  # every pyroomacoustics RIR starts 40 samples late
EARLY = int(0.050 * FS)  # early-image window after the direct path (DESIGN 4.2)
PAD_PRE, PAD_POST = 512, 1600  # island padding: non-causal skew delay, 100 ms high-pass tail
NTAPS = 2048  # noise-shaping FIR length (7.8 Hz grid)
TAILS = ("synthetic", "ism", "none")


# ----------------------------------------------------------------------------- geometry helpers
def place_array(center, spacing, azimuth_deg, elevation_deg=0.0):
    """[mic A, mic B]: A = center + (d/2)·u, B = center − (d/2)·u, u from azimuth/elevation (DESIGN 2.0)."""
    az, el = np.radians([azimuth_deg, elevation_deg])
    u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    c = np.asarray(center, float)
    return [(c + u * spacing / 2).tolist(), (c - u * spacing / 2).tolist()]


def facing(pos, target, p=0.75):
    """Directivity spec of a source at `pos` facing `target` (p: 1 omni, 0.75 sub-cardioid, 0.5 cardioid)."""
    v = np.asarray(target, float) - np.asarray(pos, float)
    return {"azimuth_deg": float(np.degrees(np.arctan2(v[1], v[0]))),
            "colatitude_deg": float(np.degrees(np.arccos(v[2] / np.linalg.norm(v)))), "p": p}


def source_truth(pos, mics):
    """Distances, angle from the mic axis (folded to 0-90°), exact TDOA (>0: Mic B later), direct-path sample."""
    a, b = (np.asarray(m, float) for m in mics)
    p = np.asarray(pos, float)
    u = (a - b) / np.linalg.norm(a - b)
    v = p - (a + b) / 2
    th = float(np.degrees(np.arccos(np.clip(v @ u / np.linalg.norm(v), -1, 1))))
    ra, rb = np.linalg.norm(p - a), np.linalg.norm(p - b)
    return {"dist_m": round(float(np.linalg.norm(v)), 6), "dist_a_m": round(float(ra), 6),
            "theta_deg": round(th, 4), "fold_deg": round(min(th, 180 - th), 4),
            "tau_us": round(float((rb - ra) / C * 1e6), 4), "direct_a": round(float(ra / C * FS + FRAC), 4)}


def _array_truth(mics):
    a, b = (np.asarray(m, float) for m in mics)
    u = (a - b) / np.linalg.norm(a - b)
    return {"spacing_m": round(float(np.linalg.norm(a - b)), 6), "center": ((a + b) / 2).round(6).tolist(),
            "array_azimuth_deg": round(float(np.degrees(np.arctan2(u[1], u[0])) % 360), 4),
            "array_elevation_deg": round(float(np.degrees(np.arcsin(u[2]))), 4)}


# ----------------------------------------------------------------------------- RIRs
def _directivity(d):
    if d is None:
        return None
    o = DirectionVector(azimuth=d["azimuth_deg"], colatitude=d.get("colatitude_deg", 90.0), degrees=True)
    return CardioidFamily(o, p=d.get("p", 0.75))


def add_image(room, pos, gain, mirror_axis=2):
    """Append an explicit image of the room's source inside an `edit_images` hook (e.g. the E7 table image).

    gain: broadband reflection factor. mirror_axis: normal axis of the mirror plane; the emission direction
    used for source directivity is the image-to-mic direction with that component negated (pyroomacoustics'
    own convention for its images)."""
    src = room.sources[0]
    p = np.asarray(pos, float)[:, None]
    d = room.mic_array.R - p
    d = d / np.linalg.norm(d, axis=0)
    if mirror_axis is not None:
        d[mirror_axis] *= -1
    oxyz = np.zeros((3, 1), dtype=src.orders_xyz.dtype)
    if mirror_axis is not None:
        oxyz[mirror_axis] = 1
    src.images = np.hstack([src.images, p.astype(src.images.dtype)])
    src.damping = np.hstack([src.damping, np.full((src.damping.shape[0], 1), gain, src.damping.dtype)])
    src.directions = np.concatenate([src.directions, d.T[:, :, None].astype(src.directions.dtype)], axis=2)
    src.orders = np.append(src.orders, 1).astype(src.orders.dtype)
    src.orders_xyz = np.hstack([src.orders_xyz, oxyz])
    src.walls = np.append(src.walls, -1).astype(src.walls.dtype)
    src.generators = np.append(src.generators, -1)
    vis = room.visibility[0]
    room.visibility[0] = np.hstack([vis, np.ones((vis.shape[0], 1), vis.dtype)])


def compute_rirs(dims, rt60, srcs, mics, seed, max_order=18, tail="synthetic", directivity=None,
                 edit_images=None, rand_disp=0.05, exact_direct=True):
    """RIRs of point sources `srcs` at the 2 `mics` in a shoebox room: (rirs (S, 2, L) float32, store, params).

    tail: 'synthetic' = image sources up to max_order + stochastic tail with sinc (diffuse) coherence, as in
          v1 lab.scenes.hybrid_rirs; 'ism' = full-order image sources, order from Sabine (E6b; slow);
          'none' = image sources up to max_order only.
    directivity: per source None (omni) or {'azimuth_deg', 'colatitude_deg' (90), 'p' (0.75)} (E5; image
          sources only, shoebox only in pyroomacoustics 0.10.1).
    edit_images(room, k): hook called for source k after room.image_source_model() and before
          room.compute_rir(); the room holds only that source (room.sources[0]). Edit its .damping or call
          add_image() there (E7).
    exact_direct: undo the randomized-ISM jitter (up to rand_disp per axis) of the order-0 image so the
          direct path comes from the true position. Seeded with pra.random.seed (E9): same seed, same bits.
    `store` is what the geometry file keeps (the image-source part only for 'synthetic'; the tail is
    re-synthesized from the seed on load) and `params` the JSON-able settings.
    """
    if tail not in TAILS:
        raise ValueError(f"tail must be one of {TAILS}")
    pra.random.seed(numpy=seed, libroom=seed)
    e_abs, sabine_order = pra.inverse_sabine(rt60, dims)
    order = int(sabine_order) if tail == "ism" else int(max_order)
    dmax = max(np.linalg.norm(np.subtract(s, m)) for s in srcs for m in mics)
    n_rir = int(np.ceil((dmax / C + 1.1 * rt60) * FS)) + FRAC
    params = {"dims": [float(v) for v in dims], "rt60": float(rt60), "tail": tail, "max_order": order,
              "e_abs": float(e_abs), "rand_disp": float(rand_disp), "exact_direct": bool(exact_direct),
              "edited": edit_images is not None, "seed": int(seed), "n_rir": n_rir}
    if tail == "synthetic":
        params["n_x"] = min(int(order * min(dims) / C * FS), n_rir)  # image sources ~complete up to here
    n_keep = params.get("n_x", n_rir)
    store = np.zeros((len(srcs), 2, n_keep), np.float32)
    for k, s in enumerate(srcs):  # one room per source keeps full-order ISM memory bounded
        room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=order,
                           use_rand_ism=rand_disp > 0, max_rand_disp=max(rand_disp, 1e-9))
        room.add_source([float(v) for v in s], directivity=_directivity(directivity[k] if directivity else None))
        room.add_microphone_array(np.asarray(mics, float).T)
        room.image_source_model()
        src = room.sources[0]
        if exact_direct:
            src.images[:, src.orders == 0] = src.position[:, None]
        if edit_images is not None:
            edit_images(room, k)
        room.compute_rir()
        for m in range(2):
            r = room.rir[m][0][:n_keep]
            store[k, m, :len(r)] = r
    return (_hybrid(store, params, mics) if tail == "synthetic" else store), store, params


def _hybrid(h_ism, p, mics):
    """Image-source part (S, 2, n_x) + energy-matched diffuse-coherent stochastic tail -> (S, 2, n_rir)."""
    S, _, n_x = h_ism.shape
    L = p["n_rir"]
    w0 = max(n_x - int(0.03 * FS), 0)
    env = np.exp(-6.908 / p["rt60"] * (np.arange(L) - n_x) / FS)
    xf = np.clip((np.arange(L) - (n_x - int(0.01 * FS))) / (0.01 * FS), 0, 1)  # 10 ms crossfade
    spacing = float(np.linalg.norm(np.subtract(mics[0], mics[1])))
    out = np.zeros((S, 2, L), np.float32)
    for k in range(S):
        h = np.zeros((2, L))
        h[:, :n_x] = h_ism[k]
        e_ism = np.mean(h[:, w0:n_x] ** 2) + 1e-20
        amp = np.sqrt(e_ism / (np.mean(env[w0:n_x] ** 2) + 1e-20))
        tail = coherent_noise(L, spacing, np.random.default_rng([p["seed"], 0x7A11, k]))
        out[k] = h * (1 - xf) + tail * env * amp * xf
    return out


# ----------------------------------------------------------------------------- files
def _write_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _set_dir(set_name, root):
    return os.path.join(root or ROOT, set_name)


def _import(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def make_geometry(set_name, geom_id, scenario, dims, rt60, mics, sources, seed, generator,
                  meta=None, root=None, **rir_opts):
    """Compute RIRs for `sources`, write <set>/<geom_id>.geom.{npz,json}, return the geometry dict.

    sources: [{'name', 'pos': [x, y, z], 'directivity': None | spec}] (names unique within the geometry).
    generator: 'module:function' making recipes, fn(geom, corpus, **variant) -> recipe dict.
    meta: generator-specific JSON (strata, bins, ...); rir_opts: compute_rirs keyword options.
    """
    names = [s["name"] for s in sources]
    if len(set(names)) != len(names):
        raise ValueError("source names must be unique")
    # positions are rounded to µm first, so the stored JSON regenerates the stored RIRs bit for bit
    mics = [[round(float(v), 6) for v in m] for m in mics]
    sources = [dict(s, pos=[round(float(v), 6) for v in s["pos"]], directivity=s.get("directivity"))
               for s in sources]
    _, store, params = compute_rirs(dims, rt60, [s["pos"] for s in sources], mics, seed,
                                    directivity=[s["directivity"] for s in sources], **rir_opts)
    geom = {"geom_id": geom_id, "set": set_name, "scenario": scenario, "generator": generator, "seed": int(seed),
            "fs": FS, "room": params, "mics": mics, **_array_truth(mics),
            "sources": [{"name": s["name"], "pos": s["pos"], "directivity": s["directivity"],
                         **source_truth(s["pos"], mics)} for s in sources],
            "meta": meta or {}}
    d = _set_dir(set_name, root)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, geom_id + ".geom")
    tmp = f"{base}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, store=store)
    os.replace(tmp, base + ".npz")
    _write_json(base + ".json", geom)
    clear_cache()
    return geom


@functools.lru_cache(maxsize=256)
def _geom_json(root, set_name, geom_id):
    with open(os.path.join(_set_dir(set_name, root), geom_id + ".geom.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=4)
def _geom_rirs(root, set_name, geom_id):
    geom = _geom_json(root, set_name, geom_id)
    with np.load(os.path.join(_set_dir(set_name, root), geom_id + ".geom.npz")) as z:
        store = z["store"]
    rirs =_hybrid(store, geom["room"], geom["mics"]) if geom["room"]["tail"] == "synthetic" else store
    rirs.flags.writeable = False
    return rirs


def load_geometry(set_name, geom_id, root=None):
    """(geometry dict, RIRs (S, 2, L) float32 read-only), cached per process."""
    root = root or ROOT
    return copy.deepcopy(_geom_json(root, set_name, geom_id)), _geom_rirs(root, set_name, geom_id)


def _recipe_path(scene_id, corpus, root):
    set_name, name = scene_id.split("/")
    return os.path.join(_set_dir(set_name, root), f"{name}.{corpus}.recipe.json")


def add_scene(set_name, name, geom, variant, corpus="tts", root=None):
    """Make scene '<set>/<name>' for `corpus` with the geometry's generator and save its recipe.

    variant: JSON dict of every corpus-independent choice (seed, SNR, ...) passed to the generator as
    keyword arguments; it is stored so the scene can be re-made for another corpus."""
    corp = corpus2.load_corpus(corpus)
    rec = dict(_import(geom["generator"])(copy.deepcopy(geom), corp, **variant))
    scene_id = f"{set_name}/{name}"
    rec.update(scene_id=scene_id, set=set_name, geom_id=geom["geom_id"], scenario=geom["scenario"],
               corpus=corpus, variant=variant)
    _validate(rec, geom, corp)
    os.makedirs(_set_dir(set_name, root), exist_ok=True)
    _write_json(_recipe_path(scene_id, corpus, root), rec)
    _load_cached.cache_clear()
    return scene_id


def _recipes_of(scene_id, root):
    set_name, name = scene_id.split("/")
    d = _set_dir(set_name, root)
    pre, suf = name + ".", ".recipe.json"
    return sorted(f[len(pre):-len(suf)] for f in os.listdir(d) if f.startswith(pre) and f.endswith(suf)
                  and "." not in f[len(pre):-len(suf)])


def ensure_recipe(scene_id, corpus="tts", root=None):
    """The recipe of `scene_id` for `corpus`; if missing, re-made from an existing recipe's variant and the
    same geometry by the geometry's generator (how scenes switch corpus, e.g. TTS -> LibriSpeech)."""
    path = _recipe_path(scene_id, corpus, root)
    if not os.path.exists(path):
        have = _recipes_of(scene_id, root)
        if not have:
            raise FileNotFoundError(f"no recipe for scene {scene_id}")
        with open(_recipe_path(scene_id, "tts" if "tts" in have else have[0], root)) as f:
            base = json.load(f)
        geom = _geom_json(root or ROOT, base["set"], base["geom_id"])
        add_scene(base["set"], scene_id.split("/")[1], geom, base["variant"], corpus, root)
    with open(path) as f:
        return json.load(f)


def draw_impairments(rng, skew_us=None):
    """E8 truth: per-mic gain ±1 dB, first-order high-pass corner 50-80 Hz, Mic-B skew 0 or 31.25 µs."""
    return {"gain_db": rng.uniform(-1, 1, 2).round(4).tolist(), "hp_hz": rng.uniform(50, 80, 2).round(3).tolist(),
            "skew_us": float(rng.choice([0.0, 31.25]) if skew_us is None else skew_us)}


# ----------------------------------------------------------------------------- recipe validation
SOURCE_KINDS = {"point": ("speech", "noise", "harmonic", "func"), "diffuse": ("noise",),
                "uncorrelated": ("noise",), "field": ("func",)}


def _validate(rec, geom, corp):
    for k in ("n_samples", "seed", "impair", "groups"):
        if k not in rec:
            raise ValueError(f"recipe {rec.get('scene_id')}: missing '{k}'")
    T = int(rec["n_samples"])
    names = [g["name"] for g in rec["groups"]]
    if len(set(names)) != len(names) or "sensor" in names:
        raise ValueError("group names must be unique and not 'sensor'")
    srcs = {s["name"] for s in geom["sources"]}
    talkers = set()
    for g in rec["groups"]:
        if not isinstance(g.get("role"), str) or not g.get("sources"):
            raise ValueError(f"group {g['name']}: needs a role and sources")
        for s in g["sources"]:
            kind = s["signal"]["kind"]
            if kind not in SOURCE_KINDS.get(s["type"], ()):
                raise ValueError(f"group {g['name']}: signal '{kind}' not allowed for type '{s['type']}'")
            if s["type"] == "point" and s["src"] not in srcs:
                raise ValueError(f"group {g['name']}: unknown source '{s['src']}'")
            if kind == "speech":
                unknown = [u for u, _, _ in s["signal"]["events"] if u not in corp]
                if unknown:
                    raise ValueError(f"group {g['name']}: utterances not in corpus {corp.name}: {unknown[:3]}")
                spk = {corp.utt(u).speaker for u, _, _ in s["signal"]["events"]}
                if len(spk) > 1:
                    raise ValueError(f"group {g['name']}: one speaker per speech source, got {spk}")
                if any(not 0 <= int(st) < T for _, st, _ in s["signal"]["events"]):
                    raise ValueError(f"group {g['name']}: event start outside the scene")
            tk = s.get("talker")
            if tk is not None:
                if kind != "speech" or tk in talkers:
                    raise ValueError(f"talker '{tk}': must be unique and on a speech source")
                talkers.add(tk)
    for r in rec.get("rules", []):
        if r["type"] != "snr" or not set(r["target"]) | set(r["noise"]) <= set(names):
            raise ValueError(f"bad rule {r}")


# ----------------------------------------------------------------------------- signals
def a_weight(f):
    """A-weighting amplitude response (1 at 1 kHz)."""
    def ra(f2):
        return 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                         * (f2 + 12194.0 ** 2))
    return ra(np.asarray(f, float) ** 2) / ra(1e6)


def _a_ratio(x):
    """A-weighted over unweighted power of a signal."""
    P = np.abs(rfft(x, axis=-1)) ** 2
    w = a_weight(rfftfreq(x.shape[-1], 1 / FS)) ** 2
    return float(np.sum(P * w) / (np.sum(P) + 1e-30))


def _amp(shape, f):
    """Amplitude of a spectral shape [[f_hz, dB], ...] (log-frequency interpolation, flat outside)."""
    if shape is None:
        return np.ones_like(f)
    pts = np.asarray(shape, float)
    lf = np.log10(np.maximum(f, pts[0, 0]))
    return 10 ** (np.interp(lf, np.log10(pts[:, 0]), pts[:, 1]) / 20)


def _fir(H):
    """Linear-phase FIR (NTAPS) with zero-phase amplitude response H on the rfft grid of NTAPS."""
    return np.roll(irfft(H, NTAPS), NTAPS // 2) * np.hanning(NTAPS)


def _shaped(w, H):
    """Filter white rows w (k, n + NTAPS - 1) to (k, n) with amplitude response H."""
    return oaconvolve(w, _fir(H)[None, :], mode="valid", axes=1)


def _noise(n, rng, shape, k=1):
    """k independent rows of Gaussian noise with spectral `shape`, unit RMS each."""
    x = _shaped(rng.standard_normal((k, n + NTAPS - 1)), _amp(shape, rfftfreq(NTAPS, 1 / FS)))
    return x / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))


def _diffuse(n, rng, shape, model, spacing):
    """2-ch noise with diffuse coherence sinc(2fd/c) ('sinc', 3-D) or J0(2πfd/c) ('j0', 2-D), unit RMS."""
    f = rfftfreq(NTAPS, 1 / FS)
    g = np.sinc(2 * f * spacing / C) if model == "sinc" else j0(2 * np.pi * f * spacing / C)
    S = _amp(shape, f)
    w = rng.standard_normal((2, n + NTAPS - 1))
    x0 = _shaped(w[:1], S)[0]
    x1 = _shaped(w[:1], S * g)[0] + _shaped(w[1:], S * np.sqrt(np.maximum(1 - g ** 2, 0)))[0]
    return np.stack([x0, x1]) / rms(x0)


def _harmonic(n, rng, sig):
    """Engine / fan / hum: harmonic series on a slowly modulated f0, plus optional shaped noise; unit RMS."""
    t = np.arange(n) / FS
    f0 = sig["f0"] + sig.get("f0_dev", 0.0) * np.sin(2 * np.pi * sig.get("f0_rate", 0.05) * t
                                                      + rng.uniform(0, 2 * np.pi))
    ph = 2 * np.pi * np.cumsum(f0) / FS
    x = np.zeros(n)
    for k in range(1, int(sig.get("n_harm", 10)) + 1):
        if k * f0.max() < 0.48 * FS:
            x += np.sin(k * ph + rng.uniform(0, 2 * np.pi)) / k ** sig.get("rolloff", 1.0)
    x /= rms(x)
    if sig.get("noise_db") is not None:
        x = x + 10 ** (sig["noise_db"] / 20) * _noise(n, rng, sig.get("noise_shape"))[0]
    return x / rms(x)


def _envelope(n, kind):
    """Burst envelope with peak 1: 'decay' (2 ms attack, exponential decay to -43 dB) or 'gate' (10 ms fades)."""
    i = np.arange(n, dtype=float)
    if kind == "decay":
        a = int(0.002 * FS)
        return np.minimum(i / a, 1.0) * np.exp(-np.maximum(i - a, 0) / (n / 5))
    r = min(int(0.01 * FS), n // 2)
    e = np.ones(n)
    if r > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
        e[:r], e[n - r:] = ramp, ramp[::-1]
    return e


def _spans(sig, T):
    if not sig.get("bursts"):
        return [(0, T, 0.0)]
    return [(int(s), min(int(n), T - int(s)), float(g)) for s, n, g in sig["bursts"] if int(s) < T]


def _dry(sig, T, rng, corp, k=1, spacing=None, model=None):
    """Unit-level blocks [(start, (k, n) float64)] of a signal spec (level semantics in BENCH2.md)."""
    kind = sig["kind"]
    if kind == "speech":
        return [(int(st), corp.audio(u).astype(np.float64)[None, :] * 10 ** (g / 20)) for u, st, g in sig["events"]]
    if kind == "func":
        x = np.asarray(_import(sig["func"])(T, FS, rng, **sig.get("params", {})), float)
        return [(0, x.reshape(k, T))]
    out = []
    for s, n, g in _spans(sig, T):
        if n <= 0:
            continue
        if model is not None:
            x = _diffuse(n, rng, sig.get("shape"), model, spacing)
        elif kind == "noise":
            x = _noise(n, rng, sig.get("shape"), k)
        else:
            x = _harmonic(n, rng, sig)[None, :]
        if sig.get("bursts"):
            x = x * _envelope(n, sig.get("env", "decay")) * 10 ** (g / 20)
        out.append((s, x))
    return out


def _level(src, blocks):
    """Amplitude factor turning unit-level blocks into Pa, A-weighting aware. pyroomacoustics 0.10.1 RIRs
    have direct-path gain 1/r (not 1/(4πr)), so for point sources this is the on-axis level at 1 m."""
    a = spl_to_pa(src["level_db"])
    if src.get("weighting", "Z") == "A" and blocks:
        a /= np.sqrt(_a_ratio(np.concatenate([b[0] for _, b in blocks])))
    return a


# ----------------------------------------------------------------------------- accumulation and impairments
class _Acc:
    """Pieces (start, (C, n) float64) of a (C, T) signal, merged on demand into padded islands."""

    def __init__(self, C, T):
        self.C, self.T, self.parts = C, T, []

    def add(self, start, x):
        x = np.asarray(x, float).reshape(self.C, -1)
        n = min(x.shape[1], self.T - start)
        if n > 0:
            self.parts.append((int(start), x[:, :n]))

    def channel(self, m):
        y = np.zeros(self.T)
        for s, x in self.parts:
            y[s:s + x.shape[1]] += x[m]
        return y

    def islands(self):
        if len(self.parts) == 1 and self.parts[0][0] == 0 and self.parts[0][1].shape[1] == self.T:
            return [self.parts[0]]
        iv = sorted((max(0, s - PAD_PRE), min(self.T, s + x.shape[1] + PAD_POST)) for s, x in self.parts)
        merged = []
        for a, b in iv:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        starts = [a for a, _ in merged]
        bufs = [np.zeros((self.C, b - a)) for a, b in merged]
        for s, x in self.parts:
            i = bisect.bisect_right(starts, s) - 1
            bufs[i][:, s - starts[i]:s - starts[i] + x.shape[1]] += x
        return list(zip(starts, bufs))


class _Chain:
    """E8 per-mic chain: first-order high-pass (causal, bilinear), gain, Mic-B fractional delay (skew)."""

    def __init__(self, imp):
        self.on = imp is not None
        if self.on:
            self.ba = [butter(1, f, "highpass", fs=FS) for f in imp["hp_hz"]]
            self.g = 10 ** (np.asarray(imp["gain_db"], float) / 20)
            self.skew = imp["skew_us"] * 1e-6

    def apply(self, b, mics):
        if not self.on:
            return b
        y = np.empty_like(b)
        for i, m in enumerate(mics):
            x = lfilter(*self.ba[m], b[i]) * self.g[m]
            if m == 1 and self.skew:
                nf = next_fast_len(len(x) + PAD_PRE)
                f = rfftfreq(nf, 1 / FS)
                x = irfft(rfft(x, nf) * np.exp(-2j * np.pi * f * self.skew), nf)[:len(x)]
            y[i] = x
        return y


class Sparse:
    """Read-only, mostly-zero float32 signal of shape (T,) or (C, T) kept as sorted non-overlapping blocks.

    Acts like an ndarray where it matters: np.asarray(x) or x.dense() materialize it, x[c] is channel c
    (Sparse), x[a:b], x[c, a:b] and x[:, a:b] return dense slices, and arithmetic or any other ndarray
    attribute (x.sum(), x.astype(...)) works on the dense array. x.blocks = [(start, array)]."""

    dtype = np.dtype(np.float32)

    def __init__(self, shape, blocks):
        self.shape, self.blocks = tuple(shape), blocks

    ndim = property(lambda self: len(self.shape))
    size = property(lambda self: int(np.prod(self.shape)))

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"Sparse(shape={self.shape}, blocks={len(self.blocks)})"

    def support(self):
        return [(s, s + b.shape[-1]) for s, b in self.blocks]

    def window(self, t0, t1):
        out = np.zeros(self.shape[:-1] + (t1 - t0,), np.float32)
        for s, b in self.blocks:
            a, e = max(s, t0), min(s + b.shape[-1], t1)
            if a < e:
                out[..., a - t0:e - t0] = b[..., a - s:e - s]
        return out

    def dense(self):
        return self.window(0, self.shape[-1])

    def __array__(self, dtype=None, copy=None):
        d = self.dense()
        return d if dtype is None else d.astype(dtype)

    def __iter__(self):
        return iter(self.dense())

    def __getitem__(self, idx):
        if self.ndim == 2 and isinstance(idx, (int, np.integer)):
            return Sparse(self.shape[1:], [(s, b[idx]) for s, b in self.blocks])
        if self.ndim == 1:
            ch, t = Ellipsis, idx
        else:
            ch, t = idx if isinstance(idx, tuple) and len(idx) == 2 else (None, None)
        if isinstance(t, slice) and t.step in (None, 1):
            t0, t1, _ = t.indices(self.shape[-1])
            return self.window(t0, max(t0, t1))[ch]
        return self.dense()[idx]

    def __getattr__(self, name):
        if name.startswith("__") or name in ("blocks", "shape"):
            raise AttributeError(name)
        return getattr(self.dense(), name)

    def __add__(self, o):
        return self.dense() + _dense(o)

    def __radd__(self, o):
        return _dense(o) + self.dense()

    def __sub__(self, o):
        return self.dense() - _dense(o)

    def __rsub__(self, o):
        return _dense(o) - self.dense()

    def __mul__(self, o):
        return self.dense() * _dense(o)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self.dense() / _dense(o)

    def __neg__(self):
        return -self.dense()


def _dense(o):
    return o.dense() if isinstance(o, Sparse) else o


def _finish(acc, chain, mics, gain):
    """Impair, scale to digital float32 and pack a group (or ref): dense if islands cover > 50% of T."""
    isl = acc.islands()
    acc.parts = []
    is_full = len(isl) == 1 and isl[0][0] == 0 and isl[0][1].shape[1] == acc.T
    if not is_full and sum(b.shape[1] for _, b in isl) > 0.5 * acc.T:
        full = np.zeros((acc.C, acc.T))
        for s, b in isl:
            full[:, s:s + b.shape[1]] += b
        isl = [(0, full)]
    blocks = []
    for s, b in isl:
        y = chain.apply(b, mics)
        y *= PA_TO_DIG * gain
        y = y.astype(np.float32)
        y = y[0] if acc.C == 1 else y
        y.flags.writeable = False
        blocks.append((s, y))
    shape = (acc.T,) if acc.C == 1 else (acc.C, acc.T)
    if len(blocks) == 1 and blocks[0][0] == 0 and blocks[0][1].shape[-1] == acc.T:
        return blocks[0][1]
    return Sparse(shape, blocks)


def _add_into(out, v, sign=1.0):
    """out (float32, 2-D) += sign·v, in place; v is an ndarray or a Sparse."""
    for s, b in (v.blocks if isinstance(v, Sparse) else [(0, v)]):
        seg = out[:, s:s + b.shape[-1]]
        if sign > 0:
            seg += b
        else:
            seg -= b


def signed_mix(scene, flip=(), drop=()):
    """Float32 sum of comp2 in canonical order with groups in `flip` negated and groups in `drop` left out.

    With no flip/drop it equals scene['mix'] bit for bit. Hagerman bus runs (DESIGN 4.2): y2 = F(A − B − N) is
    F(signed_mix(sc, flip=[g for g in sc['comp2'] if g != 'partner_A']))."""
    T = scene["mix"].shape[1]
    out = np.zeros((2, T), np.float32)
    for g, v in scene["comp2"].items():
        if g not in drop:
            _add_into(out, v, -1.0 if g in flip else 1.0)
    return out


# ----------------------------------------------------------------------------- rendering
def _rng(seed, *keys):
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(k).encode()) for k in keys])


def _band_power(x, mask):
    sos = butter(4, [150, 7000], btype="band", fs=FS, output="sos")
    return float(np.mean(sosfiltfilt(sos, x)[mask] ** 2))


def render(geom, rirs, recipe, impair=True):
    """Render a recipe on its geometry into the scene dict (keys in BENCH2.md "Scene format")."""
    T, seed = int(recipe["n_samples"]), recipe["seed"]
    corp = corpus2.load_corpus(recipe["corpus"])
    sources = geom["sources"]
    src_ix = {s["name"]: k for k, s in enumerate(sources)}
    chain = _Chain(recipe["impair"] if impair else None)
    rule_groups = {g for r in recipe.get("rules", []) for g in r["target"] + r["noise"]}
    roles, talker_group, talkers, segs = {}, {}, {}, []
    pending, finished, ref_accs = {}, {}, {}
    for g in recipe["groups"]:
        name = g["name"]
        roles[name] = g["role"]
        acc = _Acc(2, T)
        for si, s in enumerate(g["sources"]):
            rng = _rng(seed, name, si)
            if s["type"] == "point":
                k = src_ix[s["src"]]
                h = rirs[k].astype(np.float64)
                blocks = _dry(s["signal"], T, rng, corp)
                a = _level(s, blocks)
                tk = s.get("talker")
                if tk is not None:
                    d0 = int(round(sources[k]["direct_a"]))
                    he = h[:1, :int(np.ceil(sources[k]["direct_a"])) + EARLY]
                    ref_accs[tk] = _Acc(1, T)
                    talker_group[tk] = name
                    utts = [e[0] for e in s["signal"]["events"]]
                    talkers[tk] = {"group": name, "src": s["src"],
                                   "speaker": corp.utt(utts[0]).speaker if utts else None,
                                   "level_db": s["level_db"], "utts": utts,
                                   **{q: sources[k][q] for q in ("dist_m", "theta_deg", "fold_deg", "tau_us")}}
                for st, x in blocks:
                    acc.add(st, oaconvolve(x * a, h, axes=1))
                    if tk is not None:
                        ref_accs[tk].add(st, oaconvolve(x * a, he, axes=1))
                        if st + d0 < T:
                            segs.append((st + d0, min(st + x.shape[1] + d0, T), tk))
            else:
                model = s.get("model", "sinc") if s["type"] == "diffuse" else None
                blocks = _dry(s["signal"], T, rng, corp, k=2, spacing=geom["spacing_m"], model=model)
                a = _level(s, blocks)
                for st, x in blocks:
                    acc.add(st, x * a)
        if name in rule_groups:
            pending[name] = acc
        else:  # finish early to keep memory low (long meetings)
            finished[name] = _finish(acc, chain, (0, 1), 1.0)
    gains = {name: 1.0 for name in roles}
    rule_log = []
    # rules see the acoustic (pre-microphone) signal at Mic A, so impair on/off keeps the acoustic scene
    mic_a = {g: pending[g].channel(0) for g in pending}
    for r in recipe.get("rules", []):
        mask = _active(segs, talker_group, r["target"], T)
        pt = _band_power(sum(mic_a[g] * gains[g] for g in r["target"]), mask)
        pn = _band_power(sum(mic_a[g] * gains[g] for g in r["noise"]), mask)
        x = np.sqrt(pt / max(pn, 1e-30) / 10 ** (r["snr_db"] / 10))
        if r.get("scale", "noise") == "noise":
            for g in r["noise"]:
                gains[g] *= x
        else:
            for g in r["target"]:
                gains[g] /= x
        rule_log.append(dict(r, gain_db=round(float(20 * np.log10(x)), 4)))
    del mic_a
    for name, acc in pending.items():
        finished[name] = _finish(acc, chain, (0, 1), gains[name])
    comp2 = {name: finished[name] for name in roles}
    sensor = _Acc(2, T)
    dba = recipe.get("sensor_dba", 33.0)
    w = a_weight(rfftfreq(4096, 1 / FS))
    sensor.add(0, _rng(seed, "sensor").standard_normal((2, T)) * spl_to_pa(dba) / np.sqrt(np.mean(w ** 2)))
    comp2["sensor"] = _finish(sensor, chain, (0, 1), 1.0)
    roles["sensor"] = "sensor"
    refs = {tk: _finish(ref_accs[tk], chain, (0,), gains[talker_group[tk]]) for tk in ref_accs}
    mix = np.zeros((2, T), np.float32)
    for v in comp2.values():
        _add_into(mix, v)
    mix.flags.writeable = False
    for lg in rule_log:  # realized SNR (in-band, Mic A, digital, over target-active samples)
        mask = _active(segs, talker_group, lg["target"], T)
        pt, pn = (_band_power(sum(np.asarray(comp2[g][0], np.float64) for g in lg[key]), mask)
                  for key in ("target", "noise"))
        lg["realized_snr_db"] = round(float(10 * np.log10(pt / max(pn, 1e-30))), 3)
    peak = float(np.abs(mix).max())
    imp = dict(recipe["impair"], applied=bool(impair)) if impair else \
        {"gain_db": [0.0, 0.0], "hp_hz": [0.0, 0.0], "skew_us": 0.0, "applied": False}
    core = {"scene_id": recipe["scene_id"], "set": recipe["set"], "scenario": recipe["scenario"],
            "corpus": recipe["corpus"], "geom_id": recipe["geom_id"], "generator": geom["generator"],
            "n_samples": T, "duration_s": T / FS, "fs": FS,
            **{k: geom[k] for k in ("spacing_m", "mics", "center", "array_azimuth_deg", "array_elevation_deg")},
            "room": geom["room"], "rt60": geom["room"]["rt60"], "sources": geom["sources"], "talkers": talkers,
            "impair": imp, "sensor_dba": dba, "rules": rule_log,
            "group_gain_db": {g: round(float(20 * np.log10(v)), 4) for g, v in gains.items()},
            "peak": peak, "clipped": peak >= 1.0, "seeds": {"geometry": geom["seed"], "recipe": seed},
            "variant": recipe.get("variant", {}), "recipe_meta": recipe.get("meta", {})}
    meta = {**geom.get("meta", {}), **recipe.get("variant", {}), **core}
    return {"id": recipe["scene_id"], "fs": FS, "mix": mix, "comp2": comp2, "groups": roles, "refs": refs,
            "talker_group": talker_group, "segs": sorted(segs, key=lambda q: (q[0], q[2])), "meta": meta}


def _active(segs, talker_group, groups, T):
    """Samples where a talker of `groups` is active (all samples if none)."""
    mask = np.zeros(T, bool)
    for s0, s1, tk in segs:
        if talker_group[tk] in groups:
            mask[s0:s1] = True
    if not mask.any():
        mask[:] = True
    return mask


# ----------------------------------------------------------------------------- public loading API
@functools.lru_cache(maxsize=2)
def _load_cached(scene_id, corpus, impair, root):
    rec = ensure_recipe(scene_id, corpus, root)
    geom, rirs = load_geometry(rec["set"], rec["geom_id"], root)
    return render(geom, rirs, rec, impair)


def load_scene(scene_id, corpus="tts", impair=True, root=None):
    """Render scene '<set>/<name>' with `corpus` (recipe made on first use if missing). impair=False renders
    ideal mics (no E8 gain/high-pass/skew; sensor noise kept). Arrays are read-only; the last 2 renders are
    cached per process."""
    sc = _load_cached(scene_id, corpus, bool(impair), root or ROOT)
    return dict(sc, comp2=dict(sc["comp2"]), groups=dict(sc["groups"]), refs=dict(sc["refs"]),
                talker_group=dict(sc["talker_group"]), segs=list(sc["segs"]), meta=copy.deepcopy(sc["meta"]))


def clear_cache():
    _load_cached.cache_clear()
    _geom_json.cache_clear()
    _geom_rirs.cache_clear()


def scene_card(scene_id, root=None):
    """Small dict used for filtering: geometry meta, variant and core facts of a scene (no rendering)."""
    root = root or ROOT
    have = _recipes_of(scene_id, root)
    with open(_recipe_path(scene_id, "tts" if "tts" in have else have[0], root)) as f:
        rec = json.load(f)
    geom = _geom_json(root, rec["set"], rec["geom_id"])
    return {**geom.get("meta", {}), **rec.get("variant", {}), "scene_id": scene_id, "set": rec["set"],
            "scenario": rec["scenario"], "geom_id": rec["geom_id"], "corpora": have, "spacing_m": geom["spacing_m"],
            "rt60": geom["room"]["rt60"], "tail": geom["room"]["tail"], "duration_s": rec["n_samples"] / FS}


def _match(value, want):
    if callable(want):
        return bool(want(value))
    if isinstance(want, (list, tuple, set, frozenset)):
        return value in want
    return value == want


def list_scenes(set_name, scenario=None, corpus=None, root=None, **filters):
    """Sorted scene ids '<set>/<name>' of a set. corpus: only scenes that already have that recipe.
    filters: key=value (equality), key=(v1, v2) (membership) or key=callable, on scene_card() fields."""
    d = _set_dir(set_name, root)
    if not os.path.isdir(d):
        return []
    names = sorted({f.split(".")[0] for f in os.listdir(d) if f.endswith(".recipe.json")})
    out = []
    for name in names:
        card = scene_card(f"{set_name}/{name}", root)
        if corpus is not None and corpus not in card["corpora"]:
            continue
        if scenario is not None and card["scenario"] != scenario:
            continue
        if all(_match(card.get(k), v) for k, v in filters.items()):
            out.append(f"{set_name}/{name}")
    return out


# ----------------------------------------------------------------------------- demo (used by the tests)
def demo_geometry(set_name="demo", root=None, seed=7, tail="synthetic"):
    """Tiny bus-like geometry: partners A/B facing each other near the mic axis, engine at the rear."""
    dims, center = [11.5, 2.5, 2.3], [5.0, 1.25, 1.0]
    mics = place_array(center, 0.04, azimuth_deg=20.0)
    az = np.radians(20.0)
    pa = [center[0] + 0.45 * np.cos(az), center[1] + 0.45 * np.sin(az), 1.1]
    pb = [center[0] - 0.6 * np.cos(az + 0.25), center[1] - 0.6 * np.sin(az + 0.25), 1.1]
    srcs = [{"name": "pA", "pos": pa, "directivity": facing(pa, pb)},
            {"name": "pB", "pos": pb, "directivity": facing(pb, pa)},
            {"name": "engine", "pos": [11.1, 1.25, 0.5]}]
    fold = max(source_truth(p, mics)["fold_deg"] for p in (pa, pb))
    return make_geometry(set_name, f"demo_bus3_{tail}", "bus", dims, 0.4, mics, srcs, seed,
                         generator="lab.scene2:demo_recipe",
                         meta={"stratum": "aligned" if fold <= 20 else "other", "purpose": "demo"},
                         root=root, tail=tail)


def demo_recipe(geom, corpus, seed=0, snr_db=0.0, seconds=10.0, skew_us=31.25):
    """Generator example: two alternating partners, engine + diffuse road noise at `snr_db`, handling bumps.

    Speech choices use their own generator, so non-speech parts are identical for every corpus."""
    rs, rn = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    T = int(seconds * FS)
    spk = [str(s) for s in rs.choice(corpus.speakers_for("partner"), 2, replace=False)]
    events, t, i = {0: [], 1: []}, int(0.3 * FS), 0
    pools = [[u for u in corpus.select(speaker=s) if u.duration < 3.5] for s in spk]
    while True:
        u = pools[i % 2][int(rs.integers(len(pools[i % 2])))]
        n = int(u.duration * FS)
        if t + n > T - int(0.3 * FS):
            break
        events[i % 2].append([u.id, t, 0.0])
        t += n + int(rs.uniform(0.2, 0.6) * FS)
        i += 1

    def partner(k, name):
        return {"name": name, "role": "partner", "sources": [{"type": "point", "src": ["pA", "pB"][k],
                "talker": ["pA", "pB"][k], "level_db": 68.0, "signal": {"kind": "speech", "events": events[k]}}]}

    road = [[40, 0], [100, -6], [400, -10], [800, -10], [2000, -22], [8000, -40]]
    bumps = [[int(rn.integers(0, T - 4000)), int(rn.integers(800, 3000)), float(rn.uniform(-6, 0))] for _ in range(3)]
    groups = [partner(0, "partner_A"), partner(1, "partner_B"),
              {"name": "engine_road", "role": "engine_road", "sources": [
                  {"type": "point", "src": "engine", "level_db": 80.0,
                   "signal": {"kind": "harmonic", "f0": float(rn.uniform(36, 44)), "f0_dev": 4.0, "f0_rate": 0.05,
                              "n_harm": 24, "noise_db": -3.0, "noise_shape": [[50, 0], [300, -3], [600, -30]]}},
                  {"type": "diffuse", "model": "sinc", "level_db": 68.0, "signal": {"kind": "noise", "shape": road}}]},
              {"name": "handling", "role": "handling", "sources": [
                  {"type": "uncorrelated", "level_db": 70.0,
                   "signal": {"kind": "noise", "shape": [[20, 0], [400, 0], [700, -40]], "bursts": bumps,
                              "env": "gate"}}]}]
    return {"n_samples": T, "seed": int(rn.integers(2 ** 31)), "impair": draw_impairments(rn, skew_us),
            "sensor_dba": 33.0, "groups": groups,
            "rules": [{"type": "snr", "target": ["partner_A", "partner_B"], "noise": ["engine_road"],
                       "snr_db": float(snr_db)}],
            "meta": {"speakers": spk}}


def build_demo(root=None):
    """Write the demo geometry and scene 'demo/bus3_snr+0' (about 0.3 MB)."""
    geom = demo_geometry(root=root)
    return add_scene("demo", "bus3_snr+0", geom, {"seed": 3, "snr_db": 0.0}, root=root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v2 scene core: build the demo scene or list a set")
    ap.add_argument("--demo", action="store_true", help="build scenes2/demo and time one render")
    ap.add_argument("--list", metavar="SET", help="list scene ids of a set")
    a = ap.parse_args()
    if a.demo:
        sid = build_demo()
        t0 = time.time()
        sc = load_scene(sid)
        print(f"{sid}: rendered {sc['meta']['duration_s']:.1f} s in {time.time() - t0:.2f} s, groups {list(sc['groups'])}")
    if a.list:
        print("\n".join(list_scenes(a.list)))
