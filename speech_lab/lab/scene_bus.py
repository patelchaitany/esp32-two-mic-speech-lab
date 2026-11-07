"""v2 bus scene generator: two partners talking in a crowded bus (DESIGN 4.1 E1-E9, 4.2).

Builds bus geometries with lab.scene2 (cabin, device, partners, passengers, engine, wheel arches, doors)
and makes their recipes (turn-taking speech, Lombard levels, engine/road noise, door/brake transients,
handling bursts, the SNR rule). Sets, layout rules and validation numbers: BENCH2.md "Bus scenes".

CLI (from the lab folder):
  .venv/bin/python -m lab.scene_bus build dev test smoke -j 4
  .venv/bin/python -m lab.scene_bus validate dev -j 3
  .venv/bin/python -m lab.scene_bus design test
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.fft import next_fast_len
from scipy.signal import butter, lfilter, sosfiltfilt

from . import scene2 as s2
from .common import C, FS, PA_TO_DIG

GENERATOR = "lab.scene_bus:make_recipe"
STRATA = ("aligned", "45", "broadside", "side_by_side", "lap_flat")
FOLD_RANGE = {"aligned": (0.0, 20.0), "45": (30.0, 60.0), "broadside": (70.0, 90.0)}
SPACINGS = (0.025, 0.04, 0.06)
TAILS = ("synthetic", "ism")
DIST_BINS = ((0.3, 0.45), (0.45, 0.65), (0.65, 1.0))
BIN_NAMES = ("0.3-0.45", "0.45-0.65", "0.65-1.0")
DIST_DESIGN = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))  # (bin A, bin B), opposed strata
RT60_RANGE, RT60_STEP = (0.15, 0.6), 0.075
LOMBARD = (62.0, 72.0)  # dB SPL at 1 m on-axis: partners and every passenger (E2, E4)
SEATED, STANDING = (1.05, 1.32), (1.45, 1.72)  # mouth heights (m)
MIN_SEP = 0.42  # minimum mouth-to-mouth distance (m)
LAP_TAU = 0.28  # lap-flat partners: |tau| <= 0.28 d/c (DESIGN: < 0.3 d/c)
SNRS, SWEEP_SNRS = (-5, 0, 5), (-10, -5, 0, 5, 10)
SETS = {"dev": {"seed0": 10000, "sweep": False, "snrs": SNRS, "seconds": 24.0},
        "test": {"seed0": 20000, "sweep": True, "snrs": SNRS, "seconds": 24.0},
        "smoke": {"seed0": 30000, "sweep": False, "snrs": (0,), "seconds": 12.0}}
PER_STRATUM = 6  # replicates per stratum: spacing (3) x tail (2)
RULE_ROLES = ("passenger_near", "passenger_cone", "conv_pair", "babble_far", "engine_road")
TRANSIENT_CAP = 103.0  # dB SPL at the device under a transient's envelope peak (peaks stay below -6 dBFS)
OCTAVES = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)  # Hz, band gains of the noise sources
# spectral shapes [[f_hz, dB], ...]; point-source noise rolls off below 30 Hz, where the image-source model
# has a large near-DC gain that a real (leaky, flexible) cabin does not
ENGINE_NOISE = [[15, -20], [35, 0], [150, -2], [400, -12], [1500, -30], [8000, -50]]
WHEEL = [[15, -20], [35, 0], [100, 0], [300, -3], [700, -4], [1200, -8], [2500, -18], [5000, -30], [8000, -40]]
ROAD_DIFFUSE = [[30, 0], [80, -2], [200, -8], [500, -14], [1000, -20], [3000, -32], [8000, -45]]
HISS = [[100, -24], [500, -8], [2000, 0], [6000, -3], [8000, -8]]
THUMP = [[15, -20], [40, 0], [150, 0], [400, -12], [1500, -30], [8000, -50]]
HANDLING = [[20, 0], [150, 0], [350, -3], [500, -12], [700, -30], [1000, -50]]


# ----------------------------------------------------------------------------- design
def set_design(set_name):
    """Design rows of a set (one per geometry), balanced within every stratum (see BENCH2.md)."""
    spec = SETS[set_name]
    if set_name == "smoke":
        fixed = [("aligned", 0.04, 0.30, (1, 0), False, 31.25, "sinc"),
                 ("side_by_side", 0.06, 0.45, None, False, 0.0, "j0"),
                 ("lap_flat", 0.025, 0.20, None, True, 31.25, "sinc")]
        return [_row(set_name, i, st, sp, "synthetic", rt, bins, hand, False, skew, road, spec, n_pass=16, n_pairs=1)
                for i, (st, sp, rt, bins, hand, skew, road) in enumerate(fixed)]
    rng = np.random.default_rng([spec["seed0"], 0xDE5])
    rows = []
    for si, stratum in enumerate(STRATA):
        dist = rng.permutation(len(DIST_DESIGN))
        third = [rng.permutation(3) for _ in TAILS]  # RT60 third of each spacing, per tail
        half = [rng.permutation(2) for _ in range(3)]  # which tail takes the lower half of a third
        hand = {int(rng.integers(3)), 3 + int(rng.integers(3))}
        sweep = {int(rng.integers(3)), 3 + int(rng.integers(3))} if spec["sweep"] else set()
        skew = set(rng.permutation(PER_STRATUM)[:3].tolist())
        flip = int(rng.integers(2))  # diffuse road model: each spacing gets one sinc and one j0 room
        for k in range(PER_STRATUM):
            sp, ti = k % 3, k // 3
            sub = 2 * third[ti][sp] + half[third[ti][sp]][ti]
            lo = RT60_RANGE[0] + sub * RT60_STEP
            rt60 = round(float(rng.uniform(lo + 0.001, lo + RT60_STEP - 0.001)), 3)
            bins = DIST_DESIGN[dist[k]] if stratum in FOLD_RANGE else None
            rows.append(_row(set_name, k * len(STRATA) + si, stratum, SPACINGS[sp], TAILS[ti], rt60, bins,
                             k in hand, k in sweep, 31.25 if k in skew else 0.0, ("sinc", "j0")[(sp + ti + flip) % 2],
                             spec, replicate=k))
    return sorted(rows, key=lambda r: r["idx"])


def _row(set_name, idx, stratum, spacing, tail, rt60, bins, handling, sweep, skew, road, spec,
         replicate=0, n_pass=None, n_pairs=None):
    sub = int(np.clip(np.floor((rt60 - RT60_RANGE[0]) / RT60_STEP + 1e-9), 0, 5))
    lo = RT60_RANGE[0] + sub * RT60_STEP
    return {"set": set_name, "idx": idx, "geom_id": f"bus_{set_name}_g{idx:02d}", "seed": spec["seed0"] + idx,
            "stratum": stratum, "replicate": replicate, "spacing": spacing, "tail": tail, "rt60": rt60,
            "rt60_bin": f"{lo:.3f}-{lo + RT60_STEP:.3f}", "dist_bins": bins, "handling": handling, "sweep": sweep,
            "skew_us": skew, "road_model": road, "n_pass": n_pass, "n_pairs": n_pairs, "seconds": spec["seconds"],
            "snrs": SWEEP_SNRS if sweep else spec["snrs"]}


def scene_name(idx, snr):
    return f"bus_g{idx:02d}_snr{int(snr):+d}"


# ----------------------------------------------------------------------------- layout helpers
def classify(theta_a, theta_b):
    """E1 stratum from two partner angles to the mic axis (deg, 0-180): 'aligned' (both folded <= 20),
    'broadside' (both >= 70; a device lying flat looks the same), '45' (both 30-60), 'side_by_side' (same
    side of the broadside plane, 20-40 deg apart) or 'other'. Same rule as methods/_core/tracker."""
    fa, fb = (min(t, 180.0 - t) for t in (theta_a, theta_b))
    if fa <= 20 and fb <= 20:
        return "aligned"
    if fa >= 70 and fb >= 70:
        return "broadside"
    if 30 <= fa <= 60 and 30 <= fb <= 60:
        return "45"
    if (theta_a - 90) * (theta_b - 90) > 0 and 20 <= abs(theta_a - theta_b) <= 40:
        return "side_by_side"
    return "other"


def _frame(az_deg, el_deg=0.0):
    """Mic axis u (Mic B -> Mic A, as lab.scene2.place_array), horizontal normal h and w = u x h."""
    a, e = np.radians([az_deg, el_deg])
    u = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    h = np.array([-np.sin(a), np.cos(a), 0.0])
    return u, h, np.cross(u, h)


def _on_cone(c, frame, theta_deg, r, z, side):
    """Point at distance r from c, at angle theta_deg from the axis and height z, on the `side` (+1/-1) of h;
    None if no such point exists."""
    u, h, w = frame
    th = np.radians(theta_deg)
    if np.sin(th) < 1e-6:
        return None
    s = ((z - c[2]) / r - np.cos(th) * u[2]) / (np.sin(th) * w[2])
    if abs(s) > 1:
        return None
    return c + r * (np.cos(th) * u + np.sin(th) * (side * np.sqrt(1 - s * s) * h + s * w))


def _ring(rng, c, r, z):
    """Point at distance r from c and height z, at a random azimuth; None if |z - c_z| >= r."""
    dz = z - c[2]
    if abs(dz) >= r:
        return None
    rho, a = np.sqrt(r * r - dz * dz), rng.uniform(0, 2 * np.pi)
    return c + np.array([rho * np.cos(a), rho * np.sin(a), dz])


def _inside(p, dims, mx=0.3, my=0.2):
    return mx <= p[0] <= dims[0] - mx and my <= p[1] <= dims[1] - my and 0.3 <= p[2] <= dims[2] - 0.25


def _height(rng, p_stand):
    return float(rng.uniform(*(STANDING if rng.random() < p_stand else SEATED)))


def _theta(p, c, u):
    v = np.asarray(p) - c
    return float(np.degrees(np.arccos(np.clip(v @ u / np.linalg.norm(v), -1, 1))))


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def _random_facing(rng):
    return {"azimuth_deg": round(float(rng.uniform(0, 360)), 3), "colatitude_deg": 90.0, "p": 0.75}


# ----------------------------------------------------------------------------- partner layouts per stratum
def _opposed(rng, stratum, dims, bins):
    """Aligned / 45 / broadside: partners on opposite sides of the device, facing each other across it."""
    lo, hi = FOLD_RANGE[stratum]
    for _ in range(20000):
        r = [rng.uniform(*DIST_BINS[b]) for b in bins]
        c = np.array([rng.uniform(1.0, dims[0] - 1.0), rng.uniform(0.35, dims[1] - 0.35), rng.uniform(0.85, 1.2)])
        az = rng.uniform(0, 360)
        fr = _frame(az)
        fa, fb = rng.uniform(lo, hi, 2)
        th = (fa, 180 - fb) if rng.random() < 0.5 else (180 - fa, fb)
        side = rng.choice((-1.0, 1.0))
        pa = _on_cone(c, fr, th[0], r[0], _height(rng, 0.15), side)
        pb = _on_cone(c, fr, th[1], r[1], _height(rng, 0.15), -side)
        if pa is None or pb is None or not (_inside(pa, dims) and _inside(pb, dims)):
            continue
        if _angle(pa - c, pb - c) >= 110 and np.linalg.norm(pa - pb) >= MIN_SEP:
            return c, az, 0.0, pa, pb, {}
    raise RuntimeError(f"no {stratum} layout")


def _side_by_side(rng, dims):
    """Two seated (or standing) neighbours, device in front of them; accepted when the angles to the axis
    satisfy the side_by_side rule and the pair subtends 20-40 deg at the device."""
    for _ in range(50000):
        fa = np.radians(rng.choice((0.0, 180.0)) + rng.uniform(-15, 15))
        fwd, lat = np.array([np.cos(fa), np.sin(fa), 0.0]), np.array([-np.sin(fa), np.cos(fa), 0.0])
        sep = rng.uniform(0.42, 0.62)
        z = rng.uniform(1.45, 1.68) if rng.random() < 0.15 else rng.uniform(1.05, 1.3)
        m = np.array([rng.uniform(1.2, dims[0] - 1.2), rng.uniform(0.3 + sep / 2, dims[1] - 0.3 - sep / 2), z])
        s = rng.choice((-1.0, 1.0))
        pa, pb = m + s * lat * sep / 2, m - s * lat * sep / 2
        pb[2] += rng.uniform(-0.08, 0.08)
        c = m + fwd * rng.uniform(0.3, 0.95) + lat * rng.uniform(-0.25, 0.25)
        c[2] = rng.uniform(0.85, 1.15)
        az = rng.uniform(0, 360)
        u = _frame(az)[0]
        ra, rb = np.linalg.norm(pa - c), np.linalg.norm(pb - c)
        gam = _angle(pa - c, pb - c)
        if not (0.3 <= ra <= 1.0 and 0.3 <= rb <= 1.0 and 20 <= gam <= 40):
            continue
        if classify(_theta(pa, c, u), _theta(pb, c, u)) == "side_by_side" and _inside(pa, dims) \
                and _inside(pb, dims) and _inside(c, dims):
            return c, az, 0.0, pa, pb, {"pose": "standing" if z > 1.4 else "seated"}
    raise RuntimeError("no side_by_side layout")


def _lap_flat(rng, dims, spacing):
    """Device lying in the owner's lap (axis tilted by the lap, +-10 deg); the other partner sits beside the
    owner or stands in front. Accepted when both |tau| <= LAP_TAU * d/c (partners above the device)."""
    lim = LAP_TAU * spacing / C * 1e6
    for _ in range(50000):
        fa = np.radians(rng.choice((0.0, 180.0)) + rng.uniform(-15, 15))
        fwd, lat = np.array([np.cos(fa), np.sin(fa), 0.0]), np.array([-np.sin(fa), np.cos(fa), 0.0])
        mo = np.array([rng.uniform(1.2, dims[0] - 1.2), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(1.05, 1.28)])
        c = mo + fwd * rng.uniform(0.0, 0.2) + lat * rng.uniform(-0.06, 0.06)
        c[2] = mo[2] - rng.uniform(0.42, 0.62)
        standing = rng.random() < 0.3
        if standing:
            mx = c + fwd * rng.uniform(0.25, 0.5) + lat * rng.uniform(-0.3, 0.3)
            mx[2] = rng.uniform(1.42, 1.62)
        else:
            mx = mo + lat * rng.choice((-1.0, 1.0)) * rng.uniform(0.42, 0.6) + fwd * rng.uniform(-0.08, 0.08)
            mx[2] = rng.uniform(1.05, 1.28)
        az, el = rng.uniform(0, 360), rng.uniform(-10, 10)
        owner_a = rng.random() < 0.5
        pa, pb = (mo, mx) if owner_a else (mx, mo)
        if not (0.5 <= c[2] <= 0.78 and _inside(pa, dims) and _inside(pb, dims) and _inside(c, dims)):
            continue
        mics = s2.place_array(c, spacing, az, el)
        ta, tb = s2.source_truth(pa, mics), s2.source_truth(pb, mics)
        if max(abs(ta["tau_us"]), abs(tb["tau_us"])) <= lim and all(0.3 <= t["dist_m"] <= 1.0 for t in (ta, tb)):
            return c, az, el, pa, pb, {"lap_owner": "A" if owner_a else "B",
                                       "pose": "standing" if standing else "seated"}
    raise RuntimeError("no lap_flat layout")


# ----------------------------------------------------------------------------- full layout
def _noise_sources(rng, dims):
    """Engine at the rear, four wheel arches, front and middle door (right-hand side)."""
    L, W, _ = dims
    xf, xr, zw = rng.uniform(2.0, 2.8), L - rng.uniform(2.6, 3.4), rng.uniform(0.3, 0.45)
    return {"engine": [L - rng.uniform(0.25, 0.45), W / 2 + rng.uniform(-0.4, 0.4), rng.uniform(0.45, 0.8)],
            "wheel_fl": [xf, 0.15, zw], "wheel_fr": [xf, W - 0.15, zw],
            "wheel_rl": [xr, 0.15, zw], "wheel_rr": [xr, W - 0.15, zw],
            "door_front": [rng.uniform(0.6, 1.3), W - 0.1, rng.uniform(0.9, 1.4)],
            "door_mid": [L * rng.uniform(0.45, 0.6), W - 0.1, rng.uniform(0.9, 1.4)]}


def _people(rng, c, fr, dims, pa, pb, n_pass, n_pairs):
    """Near passengers, conversing pairs, the cone passenger and far babble around device c; None if
    they do not fit."""
    taken = [pa, pb]

    def ok(p):
        return p is not None and _inside(p, dims) and all(np.linalg.norm(p - q) >= MIN_SEP for q in taken)

    def place(draw, rmin=0.0, rmax=np.inf, tries=4000):
        """First draw() that is free and rmin <= |p - c| <= rmax (appended to taken), or None."""
        for _ in range(tries):
            p = draw()
            if ok(p) and rmin <= np.linalg.norm(p - c) <= rmax:
                taken.append(p)
                return p
        return None

    near = [place(lambda k=k: _ring(rng, c, rng.uniform(*((0.6, 0.85) if k == 0 else (0.6, 1.4))),
                                    _height(rng, 0.3))) for k in range(3)]
    conv = []
    for _ in range(n_pairs):
        a = place(lambda: _ring(rng, c, rng.uniform(0.6, 1.2), _height(rng, 0.3)))
        if a is None:
            return None
        conv.append((a, place(lambda: _ring(rng, a, rng.uniform(0.45, 1.0), _height(rng, 0.3)), 0.6, 1.2)))
    th_a = _theta(pa, c, fr[0])
    side_a = 1.0 if (pa - c) @ fr[1] >= 0 else -1.0

    def cone_draw():
        th, r, z = th_a + rng.uniform(-2, 2), rng.uniform(1.5, 2.0), _height(rng, 0.35)
        p = _on_cone(c, fr, th, r, z, side_a)
        return p if ok(p) else _on_cone(c, fr, th, r, z, -side_a)
    cone = place(cone_draw)
    if any(p is None for p in near) or any(b is None for _, b in conv) or cone is None:
        return None
    r_far = max(1.2, max(np.linalg.norm(p - c) for p in near + [q for pr in conv for q in pr]) + 0.05)
    far = [place(lambda: np.array([rng.uniform(0.6, dims[0] - 0.6), rng.uniform(0.25, dims[1] - 0.25),
                                   _height(rng, 0.3)]), r_far) for _ in range(n_pass - 4 - 2 * n_pairs)]
    if any(p is None for p in far):
        return None
    return near, conv, cone, far


def sample_layout(row):
    """Room, device and every source of one design row (no RIRs): {dims, mics, sources, meta}."""
    rng = np.random.default_rng([row["seed"], 0xB05])
    dims = [round(float(v), 3) for v in np.array([11.5, 2.5, 2.3]) * rng.uniform(0.95, 1.05, 3)]
    n_pass = row["n_pass"] or int(rng.integers(16, 26))
    n_pairs = row["n_pairs"] or int(rng.integers(1, 3))
    noise = {k: np.array(v) for k, v in _noise_sources(rng, dims).items()}
    st, d = row["stratum"], row["spacing"]
    for _ in range(500):
        if st in FOLD_RANGE:
            c, az, el, pa, pb, extra = _opposed(rng, st, dims, row["dist_bins"])
        elif st == "side_by_side":
            c, az, el, pa, pb, extra = _side_by_side(rng, dims)
        else:
            c, az, el, pa, pb, extra = _lap_flat(rng, dims, d)
        if min(np.linalg.norm(c - q) for q in noise.values()) < 0.6:
            continue
        ppl = _people(rng, c, _frame(az, el), dims, pa, pb, n_pass, n_pairs)
        if ppl is not None and all(min(np.linalg.norm(p - q) for q in noise.values()) >= 0.4
                                   for p in [pa, pb, ppl[2]] + ppl[0] + ppl[3] + [x for pr in ppl[1] for x in pr]):
            break
    else:
        raise RuntimeError(f"no layout for {row['geom_id']}")
    near, conv, cone, far = ppl
    rd = lambda p: [round(float(v), 6) for v in p]  # noqa: E731  (µm, as make_geometry stores them)
    mics = [rd(m) for m in s2.place_array(c, d, az, el)]
    pa, pb = rd(pa), rd(pb)
    src = [{"name": "pA", "pos": pa, "directivity": s2.facing(pa, pb)},
           {"name": "pB", "pos": pb, "directivity": s2.facing(pb, pa)}]
    src += [{"name": f"n{k + 1}", "pos": rd(p), "directivity": _random_facing(rng)} for k, p in enumerate(near)]
    src.append({"name": "cone", "pos": rd(cone), "directivity": _random_facing(rng)})
    for k, (a, b) in enumerate(conv):
        a, b = rd(a), rd(b)
        src += [{"name": f"c{k + 1}a", "pos": a, "directivity": s2.facing(a, b)},
                {"name": f"c{k + 1}b", "pos": b, "directivity": s2.facing(b, a)}]
    src += [{"name": f"f{k + 1:02d}", "pos": rd(p), "directivity": _random_facing(rng)} for k, p in enumerate(far)]
    src += [{"name": k, "pos": rd(v), "directivity": None} for k, v in noise.items()]
    tr = {s["name"]: s2.source_truth(s["pos"], mics) for s in src}
    ta, tb = tr["pA"], tr["pB"]
    why = []
    if abs(ta["tau_us"] - tb["tau_us"]) < 25:
        why.append("close_tdoas")
    if max(abs(ta["tau_us"]), abs(tb["tau_us"])) < 0.3 * d / C * 1e6:
        why.append("both_broadside")
    bins = [next(i for i, (lo, hi) in enumerate(DIST_BINS) if t["dist_m"] <= hi or i == 2) for t in (ta, tb)]
    v = np.prod(dims)
    meta = {"stratum": st, "replicate": row["replicate"], "spacing_mm": int(round(d * 1000)),
            "rt60_bin": row["rt60_bin"], "dist_A_m": ta["dist_m"], "dist_B_m": tb["dist_m"],
            "dist_bin_A": BIN_NAMES[bins[0]], "dist_bin_B": BIN_NAMES[bins[1]],
            "partner_dist_max_m": max(ta["dist_m"], tb["dist_m"]),
            "theta_A_deg": ta["theta_deg"], "theta_B_deg": tb["theta_deg"],
            "fold_A_deg": ta["fold_deg"], "fold_B_deg": tb["fold_deg"],
            "tau_A_us": ta["tau_us"], "tau_B_us": tb["tau_us"],
            "partner_angle_deg": round(_angle(np.subtract(pa, c), np.subtract(pb, c)), 3),
            "angle_class": classify(ta["theta_deg"], tb["theta_deg"]),
            "inseparable": bool(why), "inseparable_why": why,
            "device_height_m": round(float(c[2]), 4), **extra,
            "n_passengers": n_pass, "n_conv_pairs": n_pairs, "n_far": len(far),
            "nearest_passenger_m": min(tr[s["name"]]["dist_m"] for s in src[2:2 + n_pass]),
            "cone_dist_m": tr["cone"]["dist_m"],
            "cone_dtheta_deg": round(tr["cone"]["theta_deg"] - ta["theta_deg"], 3),
            "handling": row["handling"], "sweep": row["sweep"], "skew_us": row["skew_us"],
            "road_model": row["road_model"], "rc_sabine_m": round(float(0.057 * np.sqrt(v / row["rt60"])), 3),
            "people": {"partners": ["pA", "pB"], "near": [f"n{k + 1}" for k in range(3)], "cone": "cone",
                       "conv": [[f"c{k + 1}a", f"c{k + 1}b"] for k in range(n_pairs)],
                       "far": [f"f{k + 1:02d}" for k in range(len(far))]},
            "noise_src": {"engine": "engine", "wheels": [k for k in noise if k.startswith("wheel")],
                          "doors": [k for k in noise if k.startswith("door")]}}
    return {"dims": dims, "mics": mics, "sources": src, "meta": meta}


# ----------------------------------------------------------------------------- measured acoustics
def _decay(h, lo, hi):
    """Decay time (s, extrapolated to 60 dB) of the Schroeder curve of h between lo and hi dB; nan if the
    curve does not reach hi before the last 5% of h."""
    e = np.cumsum(h[::-1] ** 2)[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-300)
    i0, i1 = int(np.argmax(edc <= lo)), int(np.argmax(edc <= hi))
    if i1 <= i0 + 8 or i1 > 0.95 * len(h):
        return float("nan")
    return float(-60.0 / np.polyfit(np.arange(i0, i1) / FS, edc[i0:i1], 1)[0])


def _octave_masks(f):
    return [(f >= fc / np.sqrt(2)) & (f < (FS / 2 if fc == OCTAVES[-1] else fc * np.sqrt(2))) for fc in OCTAVES]


def _signal_gain(sig, band_gain):
    """Gain (dB) from a source's level_db to its level at Mic A for the dry signal `sig` (noise shape or
    harmonic series), weighting the source's octave-band gains by the signal's band powers."""
    f = np.fft.rfftfreq(8192, 1 / FS)
    if sig["kind"] == "harmonic":
        P = np.zeros_like(f)
        for k in range(1, int(sig["n_harm"]) + 1):
            P[np.argmin(np.abs(f - k * sig["f0"]))] += k ** (-2.0 * sig.get("rolloff", 1.0))
    else:
        P = s2._amp(sig["shape"], f) ** 2
    w = np.array([P[b].sum() for b in _octave_masks(f)])
    return float(10 * np.log10(np.sum(w * 10 ** (np.asarray(band_gain) / 10)) / w.sum()))


def measure_acoustics(rirs, sources, mics):
    """Mic-A acoustics of a geometry: decay times and effective critical distance r·10^(DRR/20) (median over
    the far passengers, who stand clear of walls), partner DRR (direct path +-0.5 ms), every source's
    energy gain 10log10(sum h^2) (level at Mic A minus level_db for a steady broadband source) and, for the
    omni noise sources, the same gain per octave band (OCTAVES; the top band runs to 8 kHz)."""
    tr = [s2.source_truth(s["pos"], mics) for s in sources]
    h = np.asarray(rirs[:, 0], np.float64)

    def drr(k):
        d0 = int(round(tr[k]["direct_a"]))
        e_dir = np.sum(h[k, max(d0 - 8, 0):d0 + 9] ** 2)
        return 10 * np.log10(e_dir / (np.sum(h[k] ** 2) - e_dir))
    far = [k for k, s in enumerate(sources) if s["name"].startswith("f") and s["name"][1:].isdigit()]
    start = [max(int(tr[k]["direct_a"]) - 40, 0) for k in range(len(sources))]
    dec = {q: float(np.nanmedian([_decay(h[k, start[k]:], lo, hi) for k in far]))
           for q, (lo, hi) in {"edt_s": (0, -10), "t20_s": (-5, -25), "t30_s": (-5, -35)}.items()}
    rc = np.median([tr[k]["dist_m"] * 10 ** (drr(k) / 20) for k in far])
    names = [s["name"] for s in sources]
    n = next_fast_len(max(h.shape[1], FS))
    bands = _octave_masks(np.fft.rfftfreq(n, 1 / FS))
    band_gain = {}
    for k in range(len(sources)):
        if sources[k]["directivity"] is None:  # noise sources: octave-band gains calibrate the transients
            H2 = np.abs(np.fft.rfft(h[k], n)) ** 2
            band_gain[names[k]] = [round(float(10 * np.log10(H2[b].mean())), 2) for b in bands]
    return {**{k: round(v, 4) for k, v in dec.items()}, "rc_eff_m": round(float(rc), 3),
            "drr_A_db": round(float(drr(names.index("pA"))), 2), "drr_B_db": round(float(drr(names.index("pB"))), 2),
            "gain_db": {nm: round(float(10 * np.log10(np.sum(h[k] ** 2))), 2) for k, nm in enumerate(names)},
            "band_gain_db": band_gain}


# ----------------------------------------------------------------------------- recipe
class _Pool:
    """A speaker's utterances [(id, n_samples)], drawn in random order without repeats until used up."""

    def __init__(self, utts, rng):
        self.utts, self.rng, self.order = utts, rng, []

    def take(self, max_n=None):
        """Next utterance no longer than max_n samples, or None if none fits."""
        if max_n is not None and min(n for _, n in self.utts) > max_n:
            return None
        while True:
            if not self.order:
                self.order = [int(i) for i in self.rng.permutation(len(self.utts))]
            for i, j in enumerate(self.order):
                if max_n is None or self.utts[j][1] <= max_n:
                    return self.utts[self.order.pop(i)]
            self.order = []


def _utts(corpus, spk, lo, hi, style=None):
    """[(id, n_samples)] of a speaker in [lo, hi] s, preferring `style`; else the 12 shortest."""
    us = (style and corpus.select(speaker=spk, style=style, min_dur=lo, max_dur=hi)) \
        or corpus.select(speaker=spk, min_dur=lo, max_dur=hi) \
        or sorted(corpus.select(speaker=spk), key=lambda u: u.duration)[:12]
    return [(u.id, int(round(u.duration * FS))) for u in us]


def _gain(rs):
    return round(float(rs.uniform(-1.5, 1.5)), 2)


def _dialogue(rs, pools, t0, t1, bc=None):
    """Alternating turns of two talkers in [t0, t1) (samples): 1-2 utterances per turn; transitions overlap
    (double talk, 20%), leave a short gap (60%) or a long pause (20%); with `bc` the listener drops a short
    utterance (back-channel stand-in) into 40% of the turns longer than 2.5 s."""
    ev, busy = ([], []), [0, 0]
    who, t = int(rs.integers(2)), t0
    while t < t1:
        t = max(t, busy[who] + int(0.1 * FS))
        start = end = t
        for _ in range(1 + int(rs.random() < 0.35)):
            u = pools[who].take(t1 - t)
            if u is None:
                break
            ev[who].append([u[0], int(t), _gain(rs)])
            end = t + u[1]
            t = end + int(rs.uniform(0.1, 0.35) * FS)
        if end == start:
            break
        busy[who], dur = end, end - start
        if bc is not None and dur > 2.5 * FS and rs.random() < 0.4:
            b = bc[1 - who].take(int(0.4 * dur))
            bt = max(start + int(rs.uniform(0.3, 0.55) * dur), busy[1 - who] + int(0.1 * FS))
            if b is not None and bt + b[1] <= end:
                ev[1 - who].append([b[0], int(bt), round(float(rs.uniform(-8, -4)), 2)])
                busy[1 - who] = bt + b[1]
        r = rs.random()
        gap = rs.uniform(-0.7, -0.15) if r < 0.2 else rs.uniform(0.1, 0.6) if r < 0.8 else rs.uniform(0.8, 2.0)
        t = max(start + dur // 2, end + int(gap * FS))
        who = 1 - who
    return ev


def _spurts(rs, pool, t1, activity):
    """Talk spurts of 1-3 utterances, then pauses giving about `activity` speech fraction; starts at
    0-3 s and may run past t1 (the renderer trims at the scene end)."""
    ev, t = [], int(rs.uniform(0, 3.0) * FS)
    while t < t1:
        spoken = 0
        for _ in range(1 + int(rs.integers(3))):
            u = pool.take()
            ev.append([u[0], int(t), _gain(rs)])
            spoken += u[1]
            t += u[1] + int(rs.uniform(0.15, 0.5) * FS)
            if t >= t1:
                break
        t += int(spoken * (1 - activity) / activity * rs.uniform(0.5, 1.5))
    return ev


def partner_speakers(corpus, split):
    """Partner speakers of a split. Speakers labelled with the split are used alone; a corpus without
    split labels (TTS) is halved deterministically (alternating within gender), so dev and test differ."""
    spk = corpus.speakers_for("partner", split)
    own = [s for s in spk if corpus.speakers[s]["split"] == split]
    if len(own) >= 2:
        return own
    if len(spk) >= 4:
        spk = sorted(spk, key=lambda s: (corpus.speakers[s]["gender"], s))
        return spk[1::2] if split == "test" else spk[0::2]
    return spk


def _fast_peak_db(n, env):
    """Max of the 125 ms ('F') time-weighted power of a burst envelope, dB re its peak."""
    a = np.exp(-1 / (0.125 * FS))
    e2 = np.concatenate([s2._envelope(n, env) ** 2, np.zeros(FS // 2)])
    return float(10 * np.log10(lfilter([1 - a], [1, -a], e2).max()))


def _transients(rn, T, band_gains, doors, wheels):
    """1-3 door or brake events with fast-weighted maximum level (LZFmax) 85-97 dB SPL at the device ([96]),
    set through the source's measured octave-band gains. The level under the envelope peak is capped at
    TRANSIENT_CAP dB SPL at the device, so short thumps may stay below the target, far from full scale."""
    n = 1 + int(rn.integers(3))
    while True:
        starts = np.sort(rn.uniform(0.8, T / FS - 1.6, n))
        if n == 1 or np.diff(starts).min() >= 2.0:
            break
    out, srcs = [], []
    for st in starts:
        kind = ("door_hiss", "door_thump", "brake_squeal", "brake_hiss")[int(rn.integers(4))]
        src = (doors if kind.startswith("door") else wheels)[int(rn.integers(2 if kind.startswith("door") else 4))]
        length = int(rn.uniform(*{"door_hiss": (0.4, 1.0), "door_thump": (0.12, 0.3), "brake_squeal": (0.4, 1.5),
                                  "brake_hiss": (0.3, 0.8)}[kind]) * FS)
        target = float(rn.uniform(85, 97))
        env = "gate" if kind == "brake_squeal" else "decay"
        burst = [[int(st * FS), length, 0.0]]
        if kind == "brake_squeal":
            sig = {"kind": "harmonic", "f0": round(float(rn.uniform(1200, 3200)), 1), "n_harm": 3, "rolloff": 1.5,
                   "bursts": burst, "env": env}
        else:
            sig = {"kind": "noise", "shape": THUMP if kind == "door_thump" else HISS, "bursts": burst, "env": env}
        peak = min(target - _fast_peak_db(length, env), TRANSIENT_CAP)  # envelope-peak level at the device
        srcs.append({"type": "point", "src": src, "level_db": round(peak - _signal_gain(sig, band_gains[src]), 2),
                     "signal": sig})
        out.append({"kind": kind, "src": src, "start": int(st * FS), "length": length,
                    "lfmax_target_db": round(target, 2), "capped": bool(peak >= TRANSIENT_CAP)})
    return srcs, out


def _handling(rn, T):
    """2-5 spatially uncorrelated bursts below 500 Hz (E6c): taps (decay) and rubs (gate), 72-84 dB SPL peak."""
    level = round(float(rn.uniform(72, 84)), 2)
    bursts = {"decay": [], "gate": []}
    for _ in range(2 + int(rn.integers(4))):
        env = "gate" if rn.random() < 0.5 else "decay"
        bursts[env].append([int(rn.uniform(0.5, T / FS - 1.5) * FS), int(rn.uniform(0.15, 1.2) * FS),
                            round(float(rn.uniform(-6, 0)), 2)])
    srcs = [{"type": "uncorrelated", "level_db": level,
             "signal": {"kind": "noise", "shape": HANDLING, "bursts": sorted(b), "env": env}}
            for env, b in bursts.items() if b]
    return srcs, {"level_db": level, "bursts": {k: sorted(v) for k, v in bursts.items()}}


def make_recipe(geom, corpus, seed, snr_db, seconds=24.0, lead_in=1.5, recorded=None):
    """Recipe of one bus scene (generator contract of lab.scene2; keyword args = the variant).

    Partners: alternating turns from `lead_in` on, Lombard levels 62-72 dB SPL at 1 m. Passengers: talk
    spurts at the same levels. SNR rule: in-band SNR at Mic A between the partners and every continuous
    interferer (passengers, pairs, far babble, engine/road) = snr_db; transients and handling keep their
    physical levels. recorded: optional E6(a) hook {"files": [...], "level_db": dBA, "replace": bool} adding
    recorded 2-channel noise (lab.scene_bus:recorded_noise) to engine_road, replacing the synthetic
    engine, wheels and road noise if "replace"."""
    m = geom["meta"]
    ppl = m["people"]
    T = int(round(seconds * FS))
    split = "test" if geom["set"] == "test" else "dev"
    rs_spk, rs_tl = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    rn_lvl, rn = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 4])
    # corpus-independent draws (fixed order and count): levels, activities, noise, impairments, render seed
    talkers = ["pA", "pB"] + ppl["near"] + [ppl["cone"]] + [t for pr in ppl["conv"] for t in pr] + ppl["far"]
    level = {t: round(float(rn_lvl.uniform(*LOMBARD)), 2) for t in talkers}
    act = {t: round(float(rn_lvl.uniform(*((0.15, 0.5) if t in ppl["far"] else (0.25, 0.55)))), 3)
           for t in ppl["near"] + [ppl["cone"]] + ppl["far"]}
    eng = {"f0": round(float(rn.uniform(28, 45)), 2), "f0_dev": round(float(rn.uniform(1.5, 5.0)), 2),
           "f0_rate": round(float(rn.uniform(0.02, 0.08)), 3), "rolloff": round(float(rn.uniform(0.8, 1.2)), 3),
           "level_dba": round(float(rn.uniform(66, 74)), 2)}
    wheel_db = [round(float(rn.uniform(60, 66)), 2) for _ in m["noise_src"]["wheels"]]
    road_db = round(float(rn.uniform(58, 66)), 2)
    tr_srcs, tr_meta = _transients(rn, T, m["acoustics"]["band_gain_db"], m["noise_src"]["doors"],
                                   m["noise_src"]["wheels"])
    hd_srcs, hd_meta = _handling(rn, T) if m["handling"] else ([], None)
    impair = s2.draw_impairments(np.random.default_rng([seed, 5]), skew_us=m["skew_us"])
    render_seed = int(np.random.default_rng([seed, 6]).integers(2 ** 31))
    # speakers and timelines (corpus-dependent)
    p_spk = [str(s) for s in rs_spk.choice(partner_speakers(corpus, split), 2, replace=False)]
    bab = [str(s) for s in rs_spk.permutation(corpus.speakers_for("babble", split))]
    spk = {"pA": p_spk[0], "pB": p_spk[1], **{t: bab[i % len(bab)] for i, t in enumerate(talkers[2:])}}
    pools = {t: _Pool(_utts(corpus, spk[t], 0.9, 6.5, "dialogue") if t in ("pA", "pB")
                      else _utts(corpus, spk[t], 0.8, 8.0), rs_tl) for t in talkers}
    bc = [_Pool(sorted(_utts(corpus, spk[t], 0.5, 1.8, "dialogue"), key=lambda u: u[1])[:3], rs_tl)
          for t in ("pA", "pB")]  # the 3 shortest lines stand in for back-channels
    t0 = int((lead_in + rs_tl.uniform(0, 0.4)) * FS)
    ev = dict(zip(("pA", "pB"), _dialogue(rs_tl, [pools["pA"], pools["pB"]], t0, T - int(0.25 * FS), bc)))
    for t in ppl["near"] + [ppl["cone"]]:
        ev[t] = _spurts(rs_tl, pools[t], T - int(0.3 * FS), act[t])
    for a, b in ppl["conv"]:
        off = int(rs_tl.uniform(0, 2.0) * FS)
        ev[a], ev[b] = _dialogue(rs_tl, [pools[a], pools[b]], off, T - int(0.3 * FS))
    for t in ppl["far"]:
        ev[t] = _spurts(rs_tl, pools[t], T - int(0.3 * FS), act[t])

    def talker(t, with_id=True):
        s = {"type": "point", "src": t, "level_db": level[t], "signal": {"kind": "speech", "events": ev[t]}}
        return dict(s, talker=t) if with_id else s
    groups = [{"name": "partner_A", "role": "partner", "sources": [talker("pA")]},
              {"name": "partner_B", "role": "partner", "sources": [talker("pB")]}]
    groups += [{"name": f"near_{k + 1}", "role": "passenger_near", "sources": [talker(t)]}
               for k, t in enumerate(ppl["near"])]
    groups.append({"name": "cone", "role": "passenger_cone", "sources": [talker(ppl["cone"])]})
    groups += [{"name": f"conv_{k + 1}", "role": "conv_pair", "sources": [talker(a), talker(b)]}
               for k, (a, b) in enumerate(ppl["conv"])]
    groups.append({"name": "babble_far", "role": "babble_far",
                   "sources": [talker(t, False) for t in ppl["far"] if ev[t]]})
    road = [{"type": "point", "src": "engine", "level_db": eng["level_dba"], "weighting": "A",
             "signal": {"kind": "harmonic", "n_harm": 40, "noise_db": -6.0, "noise_shape": ENGINE_NOISE,
                        **{k: eng[k] for k in ("f0", "f0_dev", "f0_rate", "rolloff")}}}]
    road += [{"type": "point", "src": w, "level_db": lv, "weighting": "A", "signal": {"kind": "noise", "shape": WHEEL}}
             for w, lv in zip(m["noise_src"]["wheels"], wheel_db)]
    road.append({"type": "diffuse", "model": m["road_model"], "level_db": road_db, "weighting": "A",
                 "signal": {"kind": "noise", "shape": ROAD_DIFFUSE}})
    if recorded:
        field = {"type": "field", "level_db": float(recorded.get("level_db", 72.0)), "weighting": "A",
                 "signal": {"kind": "func", "func": "lab.scene_bus:recorded_noise",
                            "params": {"files": list(recorded["files"]),
                                       "channels": list(recorded.get("channels", (0, 1)))}}}
        road = [field] if recorded.get("replace") else road + [field]
    groups.append({"name": "engine_road", "role": "engine_road", "sources": road})
    groups.append({"name": "transient", "role": "transient", "sources": tr_srcs})
    if hd_srcs:
        groups.append({"name": "handling", "role": "handling", "sources": hd_srcs})
    rule_noise = [g["name"] for g in groups if g["role"] in RULE_ROLES]
    return {"n_samples": T, "seed": render_seed, "impair": impair, "sensor_dba": 33.0, "groups": groups,
            "rules": [{"type": "snr", "target": ["partner_A", "partner_B"], "noise": rule_noise,
                       "snr_db": float(snr_db), "scale": "noise"}],
            "meta": {"speakers": spk, "lombard_db": level, "activity": act, "lead_in_s": lead_in, "engine": eng,
                     "wheel_dba": wheel_db, "road_dba": road_db, "transients": tr_meta, "handling": hd_meta,
                     "recorded": recorded}}


def recorded_noise(T, fs, rng, files, channels=(0, 1)):
    """E6(a) hook ('func' signal of a 'field' source): (2, T) excerpt of recorded 2-channel noise at a random
    offset, looped if shorter than T, unit RMS over both channels. files: one multichannel file (pick
    `channels`) or two mono files, paths relative to the lab folder, sample rate fs."""
    lab = os.path.dirname(s2.ROOT)
    xs = []
    for f in files:
        x, sr = sf.read(os.path.join(lab, f), dtype="float32", always_2d=True)
        if sr != fs:
            raise ValueError(f"{f}: fs {sr} != {fs}")
        xs.append(x.T)
    x = xs[0][list(channels)] if len(xs) == 1 else np.stack([v[0][:min(len(w[0]) for w in xs)] for v in xs])
    x = np.roll(x.astype(np.float64), -int(rng.integers(x.shape[1])), axis=1)
    x = np.tile(x, (1, int(np.ceil(T / x.shape[1]))))[:, :T]
    return x / np.sqrt(np.mean(x ** 2))


# ----------------------------------------------------------------------------- build
def build_geometry(row, root=None):
    """Geometry (RIRs + measured acoustics) and all scenes of one design row; returns a log dict."""
    t0 = time.time()
    lay = sample_layout(row)
    pos, dirs = [s["pos"] for s in lay["sources"]], [s["directivity"] for s in lay["sources"]]
    rirs, _, params = s2.compute_rirs(lay["dims"], row["rt60"], pos, lay["mics"], row["seed"], directivity=dirs,
                                      tail=row["tail"])
    meta = dict(lay["meta"], acoustics=measure_acoustics(rirs, lay["sources"], lay["mics"]))
    del rirs
    geom = s2.make_geometry(row["set"], row["geom_id"], "bus", lay["dims"], row["rt60"], lay["mics"], lay["sources"],
                            row["seed"], GENERATOR, meta=meta, root=root, tail=row["tail"])
    t1 = time.time()
    ids = [s2.add_scene(row["set"], scene_name(row["idx"], snr), geom,
                        {"seed": row["seed"], "snr_db": float(snr), "seconds": row["seconds"]}, root=root)
           for snr in row["snrs"]]
    return {"geom_id": row["geom_id"], "tail": row["tail"], "rt60": row["rt60"], "order": params["max_order"],
            "n_src": len(pos), "geom_s": round(t1 - t0, 2), "scenes_s": round(time.time() - t1, 2), "scenes": ids}


def _expected_cost(row):
    return row["rt60"] ** 3 if row["tail"] == "ism" else 0.0


def build_sets(names, jobs=4, root=None):
    rows = [r for n in names for r in set_design(n)]
    rows.sort(key=_expected_cost, reverse=True)  # slow full-order ISM rooms first
    t0 = time.time()
    with Pool(jobs) as p:
        logs = p.map(_build_one, [(r, root) for r in rows], chunksize=1)
    wall = time.time() - t0
    for n in names:
        d = os.path.join(root or s2.ROOT, n)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.startswith("bus_"))
        lg = [g for g in logs if g["geom_id"].startswith(f"bus_{n}_")]
        print(f"{n}: {len(lg)} geometries, {sum(len(g['scenes']) for g in lg)} scenes, "
              f"{size / 1e6:.1f} MB; geometry build {sum(g['geom_s'] for g in lg):.0f} s CPU "
              f"(max {max(g['geom_s'] for g in lg):.1f} s), recipes {sum(g['scenes_s'] for g in lg):.0f} s CPU")
    print(f"wall time {wall:.0f} s with {jobs} workers")
    return logs


def _build_one(args):
    return build_geometry(*args)


# ----------------------------------------------------------------------------- validation
def _inband(x):
    return sosfiltfilt(butter(4, [150, 7000], "band", fs=FS, output="sos"), x)


def _mic_a(sc, groups):
    """Sum of the Mic-A signals of `groups` (float64)."""
    return sum(np.asarray(sc["comp2"][g][0], np.float64) for g in groups)


def _dba(x):
    """A-weighted level (dB SPL) of a digital Mic-A signal."""
    p = np.asarray(x, np.float64) / PA_TO_DIG
    return float(10 * np.log10(np.mean(p ** 2) * s2._a_ratio(p) / 4e-10 + 1e-30))


def _lfmax(x):
    """Max 'F' (125 ms) time-weighted level, dB SPL, of a digital Mic-A signal (unweighted)."""
    p = np.asarray(x, np.float64) / PA_TO_DIG
    a = np.exp(-1 / (0.125 * FS))
    return float(10 * np.log10(lfilter([1 - a], [1, -a], p ** 2).max() / 4e-10))


GEOM_KEYS = ("stratum", "spacing_mm", "dist_A_m", "dist_B_m", "dist_bin_A", "dist_bin_B", "theta_A_deg",
             "theta_B_deg", "fold_A_deg", "fold_B_deg", "tau_A_us", "tau_B_us", "partner_angle_deg", "angle_class",
             "inseparable", "cone_dist_m", "cone_dtheta_deg", "nearest_passenger_m", "n_passengers", "handling")


def validate_geometry(row, root=None):
    """Render every scene of a design row (impaired and ideal mics) and measure what BENCH2.md reports."""
    root = root or s2.ROOT
    geom, _ = s2.load_geometry(row["set"], row["geom_id"], root=root)
    m = geom["meta"]
    out = {"geom_id": row["geom_id"], **{k: m[k] for k in GEOM_KEYS}, "rt60": geom["room"]["rt60"],
           "tail": geom["room"]["tail"], "order": geom["room"]["max_order"], "rc_sabine_m": m["rc_sabine_m"],
           **{k: m["acoustics"][k] for k in ("edt_s", "t20_s", "t30_s", "rc_eff_m", "drr_A_db", "drr_B_db")},
           "scenes": []}
    for snr in row["snrs"]:
        sid = f"{row['set']}/{scene_name(row['idx'], snr)}"
        s2.clear_cache()
        t0 = time.time()
        sc = s2.load_scene(sid, root=root)
        dt = time.time() - t0
        ideal = s2.load_scene(sid, impair=False, root=root)
        rule = sc["meta"]["rules"][0]
        mask = np.zeros(sc["mix"].shape[1], bool)
        for a, b, tk in sc["segs"]:
            mask[a:b] |= tk in ("pA", "pB")
        tgt = _inband(_mic_a(sc, rule["target"]))[mask]
        rest = _inband(_mic_a(sc, [g for g in sc["comp2"] if g not in rule["target"]]))[mask]
        pn = {g: np.mean(_inband(_mic_a(ideal, [g]))[mask] ** 2) for g in rule["noise"]}
        share = {role: sum(v for g, v in pn.items() if ideal["groups"][g] == role) / sum(pn.values())
                 for role in RULE_ROLES}
        out["scenes"].append({
            "scene_id": sid, "snr_db": rule["snr_db"], "realized_snr_db": rule["realized_snr_db"],
            "ideal_snr_db": ideal["meta"]["rules"][0]["realized_snr_db"],
            "all_inclusive_snr_db": round(float(10 * np.log10(np.mean(tgt ** 2) / np.mean(rest ** 2))), 3),
            "noise_gain_db": rule["gain_db"], "noise_dba": round(_dba(_mic_a(ideal, rule["noise"])), 2),
            "partner_dba": round(_dba(_mic_a(ideal, rule["target"])[mask]), 2),
            "share": {k: round(float(v), 3) for k, v in share.items()}, "peak": sc["meta"]["peak"],
            "clipped": sc["meta"]["clipped"], "render_s": round(dt, 3)})
    # signals that do not depend on the SNR: transients, handling, turn-taking (last scene rendered)
    tr = _mic_a(ideal, ["transient"])
    out["transients"] = []
    for e in ideal["meta"]["recipe_meta"]["transients"]:
        seg = tr[e["start"]:e["start"] + e["length"] + FS // 2]
        out["transients"].append({"kind": e["kind"], "target": e["lfmax_target_db"], "capped": e["capped"],
                                  "measured": round(_lfmax(seg), 2),
                                  "peak_dbfs": round(float(20 * np.log10(np.abs(seg).max())), 2)})
    if "handling" in ideal["comp2"]:
        hd = np.asarray(ideal["comp2"]["handling"], np.float64)
        on = np.abs(hd[0]) > 0
        f = np.fft.rfftfreq(hd.shape[1], 1 / FS)
        P = np.abs(np.fft.rfft(hd[0])) ** 2
        out["handling_stats"] = {"corr": round(float(np.corrcoef(hd[0][on], hd[1][on])[0, 1]), 4),
                                 "frac_below_500": round(float(P[f < 500].sum() / P.sum()), 4)}
    events = [e for g in s2.ensure_recipe(sid, root=root)["groups"] if g["role"] == "partner"
              for e in g["sources"][0]["signal"]["events"]]
    out["turns"] = dict(_turn_stats(ideal["segs"], len(tr), ideal["meta"]["recipe_meta"]["lead_in_s"]),
                        n_backchannel=sum(e[2] <= -4 for e in events))
    return out


def _turn_stats(segs, T, lead_in):
    """Partner activity, double talk and onset after the lead-in; near passengers' speech while both partners
    are silent, and while no other ref talker (partner, near, cone or pair) speaks at all."""
    act = {}
    for s0, s1, tk in segs:
        act.setdefault(tk, np.zeros(T, bool))[s0:s1] = True
    a, b = act["pA"], act["pB"]
    n_on = sum(v.astype(int) for v in act.values())
    first = min(s0 for s0, _, tk in segs if tk in ("pA", "pB"))
    near = [act.get(t, np.zeros(T, bool)) for t in ("n1", "n2", "n3")]
    return {"act_A": round(float(a.mean()), 3), "act_B": round(float(b.mean()), 3),
            "double_talk": round(float((a & b).sum() / max((a | b).sum(), 1)), 3),
            "n_A": sum(tk == "pA" for *_, tk in segs), "n_B": sum(tk == "pB" for *_, tk in segs),
            "first_partner_s": round(first / FS, 3), "lead_in_ok": bool(first >= lead_in * FS),
            "near_no_partner_s": [round(float((v & ~a & ~b).sum() / FS), 2) for v in near],
            "near_solo_s": [round(float((v & (n_on == 1)).sum() / FS), 2) for v in near]}


def _validate_one(args):
    return validate_geometry(*args)


def validate_sets(names, jobs=3, root=None, out_json=None):
    """Validate sets and print the BENCH2.md summary; returns the per-geometry records."""
    with Pool(jobs) as p:
        recs = p.map(_validate_one, [(r, root) for n in names for r in set_design(n)], chunksize=1)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(recs, f, indent=1)
    summarize(recs)
    return recs


def _rng_txt(v, fmt="{:.1f}"):
    v = np.asarray(v, float)
    return f"{fmt.format(np.min(v))}..{fmt.format(np.max(v))} (median {fmt.format(np.median(v))})"


def summarize(recs):
    """Print the validation tables (angles, TDOA, distances, acoustics, SNR, levels, cost)."""
    print("\n== partner angles per stratum (folded, deg), angle-rule class, inseparable truth flags")
    for st in STRATA:
        g = [r for r in recs if r["stratum"] == st]
        extra = ""
        if st == "side_by_side":
            extra = (f"; |theta_A-theta_B| {_rng_txt([abs(r['theta_A_deg'] - r['theta_B_deg']) for r in g])}"
                     f"; subtended {_rng_txt([r['partner_angle_deg'] for r in g])}")
        if g:
            print(f"{st:13s} n={len(g):2d} fold_A {_rng_txt([r['fold_A_deg'] for r in g])} fold_B "
                  f"{_rng_txt([r['fold_B_deg'] for r in g])} class {sorted({r['angle_class'] for r in g})} "
                  f"inseparable {sum(r['inseparable'] for r in g)}{extra}")
    print("\n== TDOA truth |tau| (us) per stratum and spacing, and |tau|/(d/c)")
    for st in STRATA:
        for mm in (25, 40, 60):
            g = [r for r in recs if r["stratum"] == st and r["spacing_mm"] == mm]
            t = np.array([abs(r[k]) for r in g for k in ("tau_A_us", "tau_B_us")])
            if g:
                print(f"{st:13s} {mm} mm n={len(g)}: tau {_rng_txt(t)}  norm "
                      f"{_rng_txt(t / (mm * 1e-3 / C * 1e6), '{:.2f}')}")
    print("\n== partner distance bins (count of partners) per stratum")
    for st in STRATA:
        g = [r for r in recs if r["stratum"] == st]
        cnt = {b: sum((r["dist_bin_A"] == b) + (r["dist_bin_B"] == b) for r in g) for b in BIN_NAMES}
        if g:
            print(f"{st:13s} {cnt}  dist {_rng_txt([r[k] for r in g for k in ('dist_A_m', 'dist_B_m')], '{:.2f}')}")
    print("\n== interferers: nearest passenger (m), cone passenger distance (m) and angle offset (deg), count")
    print(f"nearest {_rng_txt([r['nearest_passenger_m'] for r in recs], '{:.2f}')}; cone "
          f"{_rng_txt([r['cone_dist_m'] for r in recs], '{:.2f}')}, dtheta "
          f"{_rng_txt([r['cone_dtheta_deg'] for r in recs], '{:.2f}')}; passengers "
          f"{_rng_txt([r['n_passengers'] for r in recs], '{:.0f}')}")
    print("\n== measured acoustics vs nominal RT60 (Mic A, median over the far passengers)")
    for tail in TAILS:
        g = sorted((r for r in recs if r["tail"] == tail), key=lambda r: r["rt60"])
        for q in ("edt_s", "t20_s", "t30_s"):
            ratio = [r[q] / r["rt60"] for r in g if np.isfinite(r[q])]
            print(f"{tail:9s} {q}/RT60 {_rng_txt(ratio, '{:.2f}')} (n={len(ratio)})")
        print(f"{tail:9s} rc_eff/rc_sabine {_rng_txt([r['rc_eff_m'] / r['rc_sabine_m'] for r in g], '{:.2f}')}; "
              f"ISM order {_rng_txt([r['order'] for r in g], '{:.0f}')}")
        for r in g[::max(1, len(g) // 5)]:
            print(f"   {r['geom_id']}: RT60 {r['rt60']:.3f} EDT {r['edt_s']:.3f} T20 {r['t20_s']:.3f} "
                  f"T30 {r['t30_s']:.3f} rc_eff {r['rc_eff_m']:.2f} (Sabine {r['rc_sabine_m']:.2f}) "
                  f"DRR A/B {r['drr_A_db']:+.1f}/{r['drr_B_db']:+.1f} dB")
    sc = [s for r in recs for s in r["scenes"]]
    print("\n== SNR (in-band 150-7000 Hz, Mic A, partner-active samples), minus the target")
    for k, label in (("realized_snr_db", "recorded mics"), ("ideal_snr_db", "ideal mics"),
                     ("all_inclusive_snr_db", "everything else (+ transients, handling, sensor)")):
        print(f"{label}: {_rng_txt([s[k] - s['snr_db'] for s in sc], '{:+.3f}')} dB")
    for snr in sorted({s["snr_db"] for s in sc}):
        g = [s for s in sc if s["snr_db"] == snr]
        print(f"snr {snr:+5.1f}: n={len(g):3d} noise gain {_rng_txt([s['noise_gain_db'] for s in g])} dB; noise "
              f"{_rng_txt([s['noise_dba'] for s in g])} dBA; partners {_rng_txt([s['partner_dba'] for s in g])} dBA")
    print("noise composition (share of in-band rule-noise power, partner-active): "
          + ", ".join(f"{k} {_rng_txt([s['share'][k] for s in sc], '{:.2f}')}" for k in RULE_ROLES))
    tr = [t for r in recs for t in r["transients"]]
    print(f"\n== transients: n={len(tr)}, capped {sum(t['capped'] for t in tr)}, peak "
          f"{_rng_txt([t['peak_dbfs'] for t in tr])} dBFS; LZFmax measured - target per kind (uncapped):")
    for kind in sorted({t["kind"] for t in tr}):
        d = [t["measured"] - t["target"] for t in tr if t["kind"] == kind and not t["capped"]]
        if d:
            print(f"   {kind:12s} n={len(d):2d} {_rng_txt(d, '{:+.1f}')} dB, 5-95% "
                  f"{np.percentile(d, 5):+.1f}..{np.percentile(d, 95):+.1f}")
    hd = [r["handling_stats"] for r in recs if "handling_stats" in r]
    if hd:
        print(f"== handling: n={len(hd)} geometries, inter-mic correlation "
              f"{_rng_txt([h['corr'] for h in hd], '{:.3f}')}, power below 500 Hz "
              f"{_rng_txt([h['frac_below_500'] for h in hd], '{:.3f}')}")
    tt = [r["turns"] for r in recs]
    print(f"== turns: utterances A {_rng_txt([t['n_A'] for t in tt], '{:.0f}')}, B "
          f"{_rng_txt([t['n_B'] for t in tt], '{:.0f}')}; back-channels "
          f"{_rng_txt([t['n_backchannel'] for t in tt], '{:.0f}')}; partner activity "
          f"{_rng_txt([t[k] for t in tt for k in ('act_A', 'act_B')], '{:.2f}')}; double talk "
          f"{_rng_txt([t['double_talk'] for t in tt], '{:.2f}')}")
    print(f"   first partner onset {_rng_txt([t['first_partner_s'] for t in tt], '{:.2f}')} s, lead-in ok "
          f"{all(t['lead_in_ok'] for t in tt)}; near passenger speech with both partners silent "
          f"{_rng_txt([s for t in tt for s in t['near_no_partner_s']], '{:.1f}')} s, with no other ref talker "
          f"{_rng_txt([s for t in tt for s in t['near_solo_s']], '{:.1f}')} s")
    print(f"== peak |mix| {_rng_txt([s['peak'] for s in sc], '{:.3f}')}; clipped {sum(s['clipped'] for s in sc)}; "
          f"render {_rng_txt([s['render_s'] for s in sc], '{:.2f}')} s per scene")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="v2 bus scenes: build, validate or show the design of sets")
    ap.add_argument("cmd", choices=("build", "validate", "design"))
    ap.add_argument("sets", nargs="+", choices=tuple(SETS))
    ap.add_argument("-j", type=int, default=4, help="worker processes (at most 4)")
    ap.add_argument("--json", help="validate: write per-geometry records to this file")
    a = ap.parse_args()
    if a.cmd == "build":
        build_sets(a.sets, min(a.j, 4))
    elif a.cmd == "validate":
        validate_sets(a.sets, min(a.j, 4), out_json=a.json)
    else:
        for n in a.sets:
            for r in set_design(n):
                print(json.dumps({k: r[k] for k in ("geom_id", "seed", "stratum", "spacing", "tail", "rt60",
                                                    "dist_bins", "handling", "sweep", "skew_us", "road_model",
                                                    "snrs")}))


if __name__ == "__main__":
    main()
