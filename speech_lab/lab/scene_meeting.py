"""v2 meeting scene generator (DESIGN §4.1 E5, E7–E9; §4.4). Usage, sets and limits: BENCH2.md "Meeting scenes".

One scene per room, built on lab.scene2:
- room about 12 × 7 × 3 m (±8 %), RT60 0.5–0.9 s, an 8 × 2 m table (top 0.72–0.76 m) along the room's x axis;
- the device at the table centre, mics 1 / 2 / 10 / 20 cm above the top, mic axis along the table or random;
- 20 talkers: table seats (long sides and ends), a second row of chairs (one in a far corner, 5.5–6.2 m away)
  and, in half the rooms, a standing presenter facing away from the device; sub-cardioid directivity (E5);
- about 3 min of conversation: every talker has ≥ 3 turns, plus interruptions and back-channels
  (10–15 % overlapped speech); 2–3 soft talkers at 52–55 dB SPL at 1 m, the others at 57–66;
- diffuse HVAC at 35 / 40 / 45 dBA, a projector fan with 120 Hz harmonics, sensor noise; 2 s of noise only
  at the start and (after the reverberation has decayed) at the end.

E7 table reflection (first order): `table_hook` edits the image sources between image_source_model() and
compute_rir(). It adds an explicit table image of every source (mirror in the tabletop plane, amplitude 0.9),
visible at a mic when the specular point lies on the tabletop, and hides the first-order floor image at a mic
when the floor-reflected path crosses the tabletop.

CLI: python -m lab.scene_meeting --build smoke dev test [-j 3] | --report SET [--corpus tts] | --b10
"""
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

from . import scene2 as s2
from .common import C, FS, PA_TO_DIG

GENERATOR = "lab.scene_meeting:make_recipe"
N_TALKERS = 20
TABLE_SIZE = (8.0, 2.0)  # m; long axis along room x
TABLE_REFL = 0.9         # broadband amplitude reflection factor of the tabletop (E7)
TALKER_P = 0.75          # talker directivity p + (1 - p)·cos (sub-cardioid, E5)
SPACINGS = (0.025, 0.04, 0.06)
MIC_HEIGHTS_CM = (1, 2, 10, 20)
HVAC_DBA = (35.0, 40.0, 45.0)
DIRECT_WIN = 40          # ±2.5 ms direct window for DRR; contains the table image (< 0.2 ms after the direct path)
LEAD_S = 2.0             # noise only before the first and after the last speech (plus RT60 at the end)
# Sets: seeds are base + room index, so dev, test and smoke never share a room.
SETS = {"smoke": {"n": 2, "base": 490_000, "target_s": 40.0, "min_turns": 1},
        "dev": {"n": 20, "base": 410_000, "target_s": 180.0, "min_turns": 3},
        "test": {"n": 20, "base": 430_000, "target_s": 180.0, "min_turns": 3}}
# Conversation model (overlap calibrated on seeds outside every set; BENCH2.md "Meeting scenes").
P_TWO_UTTS = 0.2         # a turn whose first utterance is < 2.5 s continues with a second one
P_INTERRUPT = 0.15       # share of turn changes that start before the previous turn has ended
BC_EVERY_S = 28.0        # one back-channel per this many seconds of conversation


# ----------------------------------------------------------------------------- E7 table reflection
def _inside(p, table):
    return table["x"][0] <= p[0] <= table["x"][1] and table["y"][0] <= p[1] <= table["y"][1]


def _cross(p, q, z):
    """Point where segment p→q meets the horizontal plane at height z, or None."""
    dp, dq = p[2] - z, q[2] - z
    if dp * dq > 0 or dp == dq:
        return None
    return p + (q - p) * (dp / (dp - dq))


def table_paths(src, mic, table):
    """E7 flags of one source/mic pair.

    table_image: source and mic are above the tabletop and the specular point of the tabletop reflection is on it.
    floor_under: the floor reflection's specular point lies under the table (DESIGN E7 rule).
    floor_blocked: either leg of the floor-reflected path crosses the tabletop (the rule used; it includes
    floor_under, and for a mic above the table centre it holds for every source in the room)."""
    s, m = np.asarray(src, float), np.asarray(mic, float)
    zt = table["z"]
    spec_t = _cross(s * [1, 1, -1] + [0, 0, 2 * zt], m, zt)
    spec_f = _cross(s * [1, 1, -1], m, 0.0)
    under = spec_f is not None and _inside(spec_f, table)
    legs = [] if spec_f is None else [_cross(s, spec_f, zt), _cross(spec_f, m, zt)]
    return {"table_image": bool(s[2] > zt and m[2] > zt and spec_t is not None and _inside(spec_t, table)),
            "floor_under": bool(under),
            "floor_blocked": bool(under or any(q is not None and _inside(q, table) for q in legs))}


def table_hook(table, refl=TABLE_REFL, add_table=True, zero_floor=True):
    """`edit_images` hook for s2.compute_rirs implementing E7 at first order.

    Per mic it hides the floor image (orders_xyz [0, 0, -1]) where table_paths() says the path is blocked
    (visibility, so a mic-specific decision is exact), and appends the table image (s2.add_image: amplitude
    `refl`, pyroomacoustics' directivity convention), visible where its specular point lies on the table."""
    def hook(room, k):
        src = room.sources[0]
        flags = [table_paths(src.position, m, table) for m in room.mic_array.R.T]
        if zero_floor:
            floor = (src.orders == 1) & (src.orders_xyz[2] == -1)
            for i, f in enumerate(flags):
                if f["floor_blocked"]:
                    room.visibility[0][i, floor] = False
        if add_table and any(f["table_image"] for f in flags):
            s = np.asarray(src.position, float)
            s2.add_image(room, [s[0], s[1], 2 * table["z"] - s[2]], refl, mirror_axis=2)
            room.visibility[0][:, -1] = [f["table_image"] for f in flags]
    return hook


def rebuild_rirs(geom):
    """A meeting geometry's RIRs recomputed from its JSON alone (the hook's parameters are meta['table'])."""
    r, t = geom["room"], geom["meta"]["table"]
    return s2.compute_rirs(r["dims"], r["rt60"], [s["pos"] for s in geom["sources"]], geom["mics"], geom["seed"],
                           max_order=r["max_order"], tail=r["tail"],
                           directivity=[s["directivity"] for s in geom["sources"]],
                           edit_images=table_hook(t, t["refl"]) if t["on"] else None,
                           rand_disp=r["rand_disp"], exact_direct=r["exact_direct"])[0]


def notch_check(dist=3.0, mouth=0.42, mic=0.10, table_z=0.74):
    """DESIGN B.10: (measured, predicted) first comb-notch frequency in Hz of the direct path plus the table image,
    for a talker `dist` m away (horizontally, across the table), mouth and mics `mouth` / `mic` m above the top.
    measured is None when no notch lies below 7.9 kHz."""
    c = [10.0, 10.0, table_z + mic]
    table = {"x": [6.0, 14.0], "y": [9.0, 11.0], "z": table_z}
    mics = s2.place_array(c, 0.04, 0.0)  # axis across the talker direction: both mics equidistant
    src = [c[0], c[1] + dist, table_z + mouth]
    h = s2.compute_rirs([20.0, 20.0, 6.0], 1.0, [src], mics, 1, max_order=0, tail="none", rand_disp=0.0,
                        edit_images=table_hook(table))[0][0, 0].astype(np.float64)
    n = 1 << 16
    f = rfftfreq(n, 1 / FS)
    H = 20 * np.log10(np.abs(rfft(h, n)) + 1e-12)
    band = np.flatnonzero((f > 300) & (f < 7900))
    deep = H[band] < H[band].max() - 10
    loc = (H[band][1:-1] < H[band][:-2]) & (H[band][1:-1] <= H[band][2:]) & deep[1:-1]
    found = band[1:-1][loc]
    img = [src[0], src[1], 2 * table_z - src[2]]
    delta = np.linalg.norm(np.subtract(img, mics[0])) - np.linalg.norm(np.subtract(src, mics[0]))
    return (float(f[found[0]]) if len(found) else None), float(C / (2 * delta))


# ----------------------------------------------------------------------------- room design and layout
def set_design(set_name):
    """Room designs of a meeting set: factors balanced across rooms (shuffled), RT60 stratified in 0.5–0.9 s."""
    cfg = SETS[set_name]
    n, rng = cfg["n"], np.random.default_rng([cfg["base"], 7])

    def balanced(levels):
        lv = [levels[i] for i in rng.permutation(len(levels))]
        return [lv[i % len(lv)] for i in rng.permutation(n)]

    rt60 = 0.5 + 0.4 * (rng.permutation(n) + rng.uniform(size=n)) / n
    cols = {"spacing_m": balanced(SPACINGS), "mic_height_cm": balanced(MIC_HEIGHTS_CM),
            "hvac_dba": balanced(HVAC_DBA), "axis_mode": balanced(("table", "random")),
            "presenter": balanced((True, False)), "tail": balanced(("synthetic", "ism"))}
    return [{"set": set_name, "index": i, "seed": cfg["base"] + i, "name": f"meeting_r{i:02d}",
             "geom_id": f"meeting_{set_name}_r{i:02d}", "rt60": round(float(rt60[i]), 4), "table": True,
             "target_s": cfg["target_s"], "min_turns": cfg["min_turns"], **{k: v[i] for k, v in cols.items()}}
            for i in range(n)]


def room_layout(d):
    """Room, table, device, talkers and fan of design `d`, deterministic in d['seed'].

    Returns {dims, table, mics, sources (t00..t19 shuffled, then 'fan'), seat {tid: 'table'|'row2'|'presenter'}}."""
    rg = np.random.default_rng([d["seed"], 11])
    lx, ly, lz = dims = [12.0 * rg.uniform(0.92, 1.08), 7.0 * rg.uniform(0.92, 1.08), 3.0 * rg.uniform(0.92, 1.08)]
    cx, cy, zt = lx / 2 + rg.uniform(-0.3, 0.3), ly / 2 + rg.uniform(-0.2, 0.2), rg.uniform(0.72, 0.76)
    hl, hw = TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2
    table = {"on": bool(d.get("table", True)), "x": [cx - hl, cx + hl], "y": [cy - hw, cy + hw], "z": zt,
             "refl": TABLE_REFL}
    dev = np.array([cx + rg.uniform(-0.1, 0.1), cy + rg.uniform(-0.1, 0.1), zt + d["mic_height_cm"] / 100])
    az = rg.choice([0.0, 180.0]) + rg.uniform(-3, 3) if d["axis_mode"] == "table" else rg.uniform(0, 360)
    mics = s2.place_array(dev, d["spacing_m"], float(az) % 360)

    def clamp(p):  # mouths stay 0.4 m from the walls
        return [float(np.clip(p[0], 0.4, lx - 0.4)), float(np.clip(p[1], 0.4, ly - 0.4)), float(p[2])]

    def mouth():
        return rg.uniform(1.12, 1.25)

    seats = [clamp([cx - 3.5 + j + rg.uniform(-0.15, 0.15), cy + sy * (hw + rg.uniform(0.35, 0.5)), mouth()])
             for j in range(8) for sy in (-1, 1)]
    seats += [clamp([cx + sx * (hl + rg.uniform(0.35, 0.5)), cy + sy * 0.5 + rg.uniform(-0.1, 0.1), mouth()])
              for sx in (-1, 1) for sy in (-1, 1)]
    row_y = {sy: cy + sy * (hw + rg.uniform(1.3, 1.8)) for sy in (-1, 1)}
    row2 = [clamp([cx - 4.5 + j + rg.uniform(-0.2, 0.2), row_y[sy] + rg.uniform(-0.1, 0.1), mouth()])
            for sy in (-1, 1) for j in range(10)]
    front = 1.0 if lx - dev[0] > dev[0] else -1.0  # the end with more room: presenter and far chair
    placed = []  # (kind, pos, directivity)
    if d["presenter"]:
        p = clamp([cx + front * rg.uniform(5.0, 5.5), cy + rg.uniform(-1.2, 1.2), rg.uniform(1.55, 1.7)])
        away = np.degrees(np.arctan2(p[1] - dev[1], p[0] - dev[0])) + rg.uniform(-20, 20)
        placed.append(("presenter", p, {"azimuth_deg": float(away % 360), "colatitude_deg": 90.0, "p": TALKER_P}))
    # far corner chair: 5.5-6.2 m from the device, on the other side of the table axis than the presenter
    sy = rg.choice([-1.0, 1.0])
    if d["presenter"] and np.sign(placed[0][1][1] - cy) == sy:
        sy = -sy
    z, r, y = mouth(), rg.uniform(5.5, 6.2), rg.uniform(1.6, 2.6)
    xmax = (lx - 0.4 - dev[0]) if front > 0 else dev[0] - 0.4
    ymax = (ly - 0.4 - dev[1]) if sy > 0 else dev[1] - 0.4
    dz2 = (z - dev[2]) ** 2
    x = min(np.sqrt(max(r * r - y * y - dz2, 0.0)), xmax)
    y = min(ymax, max(y, np.sqrt(max(r * r - x * x - dz2, 0.0))))
    placed.append(("row2", clamp([dev[0] + front * x, dev[1] + sy * y, z]), None))
    # other second-row chairs (0.8 m clear of the far chair and the presenter), then table seats
    n_row2 = int(rg.integers(3, 7))
    free = [p for p in row2 if min(np.hypot(p[0] - q[1][0], p[1] - q[1][1]) for q in placed) >= 0.8]
    placed += [("row2", free[i], None) for i in rg.choice(len(free), n_row2 - 1, replace=False)]
    placed += [("table", seats[i], None) for i in rg.choice(len(seats), N_TALKERS - len(placed), replace=False)]
    sources, seat = [], {}
    for i, k in enumerate(rg.permutation(N_TALKERS)):
        kind, pos, dr = placed[k]
        if dr is None:  # seated talkers face a point on the table's long axis
            dr = s2.facing(pos, [cx + rg.uniform(-3, 3), cy, pos[2]], TALKER_P)
        sources.append({"name": f"t{i:02d}", "pos": pos, "directivity": dr})
        seat[f"t{i:02d}"] = kind
    fan = [cx + rg.uniform(-1.5, 1.5), cy + rg.uniform(-0.6, 0.6), lz - rg.uniform(0.3, 0.5)]
    sources.append({"name": "fan", "pos": [float(v) for v in fan], "directivity": None})
    return {"dims": [float(v) for v in dims], "table": {k: (float(v) if k == "z" else v) for k, v in table.items()},
            "mics": mics, "sources": sources, "seat": seat}


# ----------------------------------------------------------------------------- acoustic truth
def dist_bin(r):
    """DESIGN T2 distance bin of a talker."""
    return "<2" if r < 2 else ("2-4" if r <= 4 else ">4")


def decay_time(h, lo=-5.0, hi=-35.0):
    """Reverberation time (s) from the Schroeder curve of h, line fit between lo and hi dB: T30 by default,
    EDT with (0, -10)."""
    edc = np.cumsum(h[::-1] ** 2)[::-1]
    edc = 10 * np.log10(edc / edc[0] + 1e-30)
    i = np.flatnonzero((edc <= lo) & (edc >= hi))
    return float(-60.0 / np.polyfit(i / FS, edc[i], 1)[0])


def acoustic_truth(geom, rirs):
    """Per source at Mic A: DRR (direct ±2.5 ms, which includes the table image), C50, EDT and T30."""
    out = {}
    for k, s in enumerate(geom["sources"]):
        h = rirs[k, 0].astype(np.float64)
        d0 = int(round(s["direct_a"]))
        e = h ** 2
        a, b = max(d0 - DIRECT_WIN, 0), d0 + DIRECT_WIN + 1
        out[s["name"]] = {"drr_db": round(float(10 * np.log10(e[a:b].sum() / e[b:].sum())), 3),
                          "c50_db": round(float(10 * np.log10(e[:d0 + s2.EARLY].sum() / e[d0 + s2.EARLY:].sum())), 3),
                          "edt_s": round(decay_time(h[a:], 0.0, -10.0), 4), "t30_s": round(decay_time(h[a:]), 4)}
    return out


def _room_truth(geom, rirs, lay):
    """Geometry-meta additions: per-talker distance, angles, directivity toward the device, E7 flags, DRR
    (measured, and the DESIGN §1.4 theory for an omni talker without a table)."""
    ac = acoustic_truth(geom, rirs)
    dev = np.asarray(geom["center"])
    rc = 0.057 * np.sqrt(np.prod(geom["room"]["dims"]) / geom["room"]["rt60"])  # DESIGN §1.4, omni talker
    tg = {}
    for s in geom["sources"]:
        if s["name"] == "fan":
            continue
        p = np.asarray(s["pos"])
        v = dev - p
        az, col = np.radians([s["directivity"]["azimuth_deg"], s["directivity"]["colatitude_deg"]])
        face = np.array([np.sin(col) * np.cos(az), np.sin(col) * np.sin(az), np.cos(col)])
        off = float(np.degrees(np.arccos(np.clip(face @ v / np.linalg.norm(v), -1, 1))))
        fl = [table_paths(p, m, lay["table"]) for m in geom["mics"]] if lay["table"]["on"] else []
        tg[s["name"]] = {
            "seat": lay["seat"][s["name"]], "dist_m": s["dist_m"], "dist_bin": dist_bin(s["dist_m"]),
            "theta_deg": s["theta_deg"], "fold_deg": s["fold_deg"], "tau_us": s["tau_us"],
            "azimuth_deg": round(float(np.degrees(np.arctan2(-v[1], -v[0])) % 360), 3),
            "elevation_deg": round(float(np.degrees(np.arcsin(-v[2] / np.linalg.norm(v)))), 3),
            "mouth_above_table_m": round(float(p[2] - lay["table"]["z"]), 4),
            "off_axis_deg": round(off, 2), "dir_gain_db": round(float(20 * np.log10(
                TALKER_P + (1 - TALKER_P) * np.cos(np.radians(off)))), 3),
            "table_image": bool(fl) and all(f["table_image"] for f in fl),
            "floor_blocked": bool(fl) and all(f["floor_blocked"] for f in fl),
            "floor_under": bool(fl) and all(f["floor_under"] for f in fl),
            **ac[s["name"]], "drr_theory_db": round(float(20 * np.log10(rc / s["dist_m"])), 3)}
    d = [t["dist_m"] for t in tg.values()]
    return {"talker_geo": tg, "fan_truth": ac["fan"], "min_dist_m": min(d), "max_dist_m": max(d),
            "n_bin": dict(Counter(t["dist_bin"] for t in tg.values())), "rc_m": round(float(rc), 4),
            "t30_s": round(float(np.median([t["t30_s"] for t in tg.values()])), 4),
            "edt_s": round(float(np.median([t["edt_s"] for t in tg.values()])), 4),
            "n_table_image": sum(t["table_image"] for t in tg.values()),
            "n_floor_zeroed": sum(t["floor_blocked"] for t in tg.values()),
            "n_floor_under": sum(t["floor_under"] for t in tg.values())}


def _write_json(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def build_room(d, root=None, corpus="tts"):
    """Write the geometry (RIRs; layout, E7 and acoustic truth in meta) and the scene of room design `d`."""
    lay = room_layout(d)
    meta = {"purpose": "meeting", "room_index": d["index"], "spacing_mm": int(round(d["spacing_m"] * 1000)),
            "mic_height_cm": d["mic_height_cm"], "axis_mode": d["axis_mode"], "presenter": d["presenter"],
            "table": lay["table"], "n_row2": sum(k == "row2" for k in lay["seat"].values())}
    hook = table_hook(lay["table"], lay["table"]["refl"]) if lay["table"]["on"] else None
    geom = s2.make_geometry(d["set"], d["geom_id"], "meeting", lay["dims"], d["rt60"], lay["mics"], lay["sources"],
                            d["seed"], GENERATOR, meta=meta, root=root, tail=d["tail"], edit_images=hook)
    _, rirs = s2.load_geometry(d["set"], d["geom_id"], root)
    geom["meta"].update(_room_truth(geom, rirs, lay))
    _write_json(os.path.join(root or s2.ROOT, d["set"], d["geom_id"] + ".geom.json"), geom)
    s2.clear_cache()
    variant = {"seed": d["seed"], "hvac_dba": d["hvac_dba"], "target_s": d["target_s"], "min_turns": d["min_turns"]}
    return s2.add_scene(d["set"], d["name"], geom, variant, corpus, root)


# ----------------------------------------------------------------------------- conversation
class _Picker:
    """Utterance choice: not yet used by this talker, then least-used text in the scene, then style 'meeting'."""

    def __init__(self, rs):
        self.rs, self.text, self.used = rs, Counter(), defaultdict(set)

    def choose(self, tid, cands):
        fresh = [u for u in cands if u.id not in self.used[tid]] or cands
        best = min((self.text[u.text], u.style != "meeting") for u in fresh)
        opts = [u for u in fresh if (self.text[u.text], u.style != "meeting") == best]
        return opts[int(self.rs.integers(len(opts)))]

    def commit(self, tid, u):
        self.text[u.text] += 1
        self.used[tid].add(u.id)


def _sequence(counts, rt):
    """Turn order: talkers drawn with probability ∝ remaining turns, never the same talker twice in a row."""
    left, seq = dict(counts), []
    while any(left.values()):
        cand = [t for t in sorted(left) if left[t] and (not seq or t != seq[-1])]
        if not cand:  # only the last talker has turns left: put them where the neighbours differ
            t = seq[-1]
            slots = [i for i in range(1, len(seq)) if t not in (seq[i - 1], seq[i])]
            seq.insert(slots[int(rt.integers(len(slots)))], t)
            left[t] -= 1
            continue
        w = np.array([left[t] for t in cand], float)
        t = cand[int(rt.choice(len(cand), p=w / w.sum()))]
        seq.append(t)
        left[t] -= 1
    return seq


def _free_parts(span, others):
    """Parts of interval `span` not covered by any interval in `others`."""
    parts = [list(span)]
    for a, b in sorted(others):
        nxt = []
        for p0, p1 in parts:
            if b <= p0 or a >= p1:
                nxt.append([p0, p1])
            else:
                if a > p0:
                    nxt.append([p0, a])
                if b < p1:
                    nxt.append([b, p1])
        parts = nxt
    return parts


def overlap_ratio(intervals, step=0.01):
    """Share of speech-active time with two or more talkers active (intervals in seconds)."""
    end = max(b for a, b in intervals)
    n = np.zeros(int(np.ceil(end / step)) + 1, int)
    for a, b in intervals:
        n[int(round(a / step)):int(round(b / step))] += 1
    return float((n >= 2).sum() / max((n >= 1).sum(), 1))


def _conversation(geom, corpus, spk, rs, rt, target_s, min_turns):
    """Events per talker and the turn table of one meeting (model in BENCH2.md "Meeting scenes")."""
    tids = sorted(spk)
    conv_s = target_s - 2 * LEAD_S - geom["room"]["rt60"]
    short = conv_s < len(tids) * min_turns * 2.5  # smoke scenes: each talker's shortest utterances
    pools = {}
    for t in tids:
        us = sorted(corpus.select(speaker=spk[t]), key=lambda u: (u.duration, u.id))
        turn = us[:3] if short else ([u for u in us if 1.0 <= u.duration <= 6.0] or us)
        pools[t] = {"turn": turn, "bc": us[:4]}
    # extra turns (activity weights) until the expected length reaches the target
    counts = {t: min_turns for t in tids}
    mean_d = {t: float(np.mean([u.duration for u in pools[t]["turn"]])) + 0.45 for t in tids}
    weights = rt.dirichlet(np.ones(len(tids)))
    while sum(counts[t] * mean_d[t] for t in tids) < conv_s:
        counts[tids[int(rt.choice(len(tids), p=weights))]] += 1
    pick, events, iv = _Picker(rs), {t: [] for t in tids}, []  # iv: [start_s, end_s, tid, kind, utts]

    def place(t, utts, start, gain, kind, pause=0.0):
        x = start
        for j, u in enumerate(utts):
            x += pause if j else 0.0
            events[t].append([u.id, int(round(x * FS)), round(float(gain), 2)])
            x += u.duration
        iv.append([start, x, t, kind, [u.id for u in utts]])

    for i, t in enumerate(_sequence(counts, rt)):
        r = rt.random(6)  # second utterance, interruption, gap, overlap, level jitter, pause
        utts = [pick.choose(t, pools[t]["turn"])]
        pick.commit(t, utts[0])
        if not short and r[0] < P_TWO_UTTS and utts[0].duration < 2.5:
            utts.append(pick.choose(t, pools[t]["turn"]))
            pick.commit(t, utts[1])
        kind, start = "turn", LEAD_S
        if i:
            prev, before = iv[-1], max([e[1] for e in iv[:-1]], default=0.0)
            if r[1] < P_INTERRUPT:  # start inside the previous turn's tail, never over a third talker
                kind, start = "interrupt", max(prev[0] + 0.5, prev[1] - (0.3 + 0.9 * r[3]), before + 0.05)
            else:
                start = max(prev[1], before) + 0.1 + min(-0.35 * np.log1p(-r[2]), 1.9)
        place(t, utts, start, 2 * r[4] - 1, kind, 0.15 + 0.3 * r[5])
    # back-channels: a talker's short utterance, softer, inside a part of another talker's turn nobody else uses
    main = list(iv)
    for _ in range(int(round((main[-1][1] - LEAD_S) / BC_EVERY_S))):
        r = rt.random(3)
        t = tids[int(r[0] * len(tids))]
        u = pick.choose(t, pools[t]["bc"])
        for h in (main[j] for j in rt.permutation(len(main))):
            parts = [] if h[2] == t else [p for p in _free_parts(h[:2], [e[:2] for e in iv if e is not h])
                                          if p[1] - p[0] >= u.duration + 0.5]
            if parts:
                p0, p1 = parts[0]
                pick.commit(t, u)
                place(t, [u], p0 + 0.25 + r[1] * (p1 - p0 - u.duration - 0.5), -2 - 4 * r[2], "backchannel")
                break
    iv.sort(key=lambda e: e[0])
    return events, iv


def make_recipe(geom, corpus, seed, hvac_dba=40.0, target_s=180.0, min_turns=3):
    """Meeting recipe (lab.scene2 generator contract; variant = seed, hvac_dba, target_s, min_turns).

    Random streams: rn (corpus-independent: talker levels, HVAC and fan, impairments, render seed), rs (speakers
    and utterances) and rt (conversation structure). Talker t<k> is group talker_<k>; groups hvac and fan."""
    rs, rn, rt = (np.random.default_rng([seed, k]) for k in (1, 2, 3))
    tids = sorted(s["name"] for s in geom["sources"] if s["name"] != "fan")
    soft = set(rn.choice(len(tids), int(rn.choice([2, 3])), replace=False).tolist())
    level = {t: round(float(rn.uniform(52, 55) if i in soft else rn.uniform(57, 66)), 2) for i, t in enumerate(tids)}
    slope = round(float(rn.uniform(-6, -4)), 3)  # HVAC spectrum above 125 Hz, dB per octave
    hvac_shape = [[31.5, 0.0], [63.0, -2.0], [125.0, -5.0]] + [[125.0 * 2 ** k, round(-5 + slope * k, 3)]
                                                                for k in range(1, 7)]
    fan_db = round(float(rn.uniform(43, 47)), 2)
    fan = {"kind": "harmonic", "f0": 120.0, "f0_dev": 0.3, "f0_rate": 0.02, "n_harm": 12,
           "rolloff": round(float(rn.uniform(0.8, 1.2)), 3), "noise_db": round(float(rn.uniform(-3, 3)), 2),
           "noise_shape": [[100, -12], [250, 0], [600, -2], [1500, -8], [4000, -18], [8000, -30]]}
    impair = s2.draw_impairments(rn)
    render_seed = int(rn.integers(2 ** 31))
    split = "test" if geom["set"] == "test" else "dev"
    pool = corpus.speakers_for("talker", split)
    if len(pool) < len(tids):
        raise ValueError(f"corpus {corpus.name}: {len(pool)} '{split}' talkers, a meeting needs {len(tids)}")
    spk = dict(zip(tids, (str(v) for v in rs.permutation(pool)[:len(tids)])))
    events, iv = _conversation(geom, corpus, spk, rs, rt, target_s, min_turns)
    lag = max(s["direct_a"] for s in geom["sources"]) / FS
    T = int(np.ceil((max(e[1] for e in iv) + lag + geom["room"]["rt60"] + LEAD_S) * FS))
    groups = [{"name": f"talker_{t[1:]}", "role": "talker", "sources": [
        {"type": "point", "src": t, "talker": t, "level_db": level[t],
         "signal": {"kind": "speech", "events": sorted(events[t], key=lambda e: e[1])}}]} for t in tids]
    groups += [{"name": "hvac", "role": "hvac", "sources": [
                   {"type": "diffuse", "model": "sinc", "level_db": float(hvac_dba), "weighting": "A",
                    "signal": {"kind": "noise", "shape": hvac_shape}}]},
               {"name": "fan", "role": "fan", "sources": [
                   {"type": "point", "src": "fan", "level_db": fan_db, "signal": fan}]}]
    n_main = Counter(e[2] for e in iv if e[3] != "backchannel")
    texts = Counter(corpus.utt(u).text for e in iv for u in e[4])
    meta = {"speakers": spk, "level_db": level, "soft": sorted(tids[i] for i in soft),
            "hvac": {"dba": float(hvac_dba), "slope_db_oct": slope}, "fan": {"level_db": fan_db, "f0": 120.0},
            "turns": [[e[2], e[3], round(e[0], 3), round(e[1], 3), e[4]] for e in iv],
            "n_turns": {t: n_main[t] for t in tids}, "n_interrupts": sum(e[3] == "interrupt" for e in iv),
            "n_backchannels": sum(e[3] == "backchannel" for e in iv),
            "overlap_ratio": round(overlap_ratio([e[:2] for e in iv]), 4),
            "text_repeats": sum(n - 1 for n in texts.values())}
    return {"n_samples": T, "seed": render_seed, "impair": impair, "sensor_dba": 33.0, "groups": groups,
            "meta": meta}


# ----------------------------------------------------------------------------- sets and validation report
def _build_one(d, root):
    t0 = time.time()
    return build_room(d, root), time.time() - t0


def build_set(set_name, workers=3, root=None):
    """Build every room of a set in `workers` processes (full image-source rooms first: they are slowest)."""
    designs = sorted(set_design(set_name), key=lambda d: (d["tail"] != "ism", -d["rt60"]))
    with ProcessPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(_build_one, d, root) for d in designs]):
            sid, dt = fut.result()
            print(f"built {sid} in {dt:.1f} s", flush=True)


def _level(x, weighted=False):
    """dB SPL (A-weighted if `weighted`) of a Pa signal."""
    p = float(np.mean(x ** 2))
    if weighted:
        P = np.abs(rfft(x)) ** 2
        p *= float(np.sum(P * s2.a_weight(rfftfreq(len(x), 1 / FS)) ** 2) / np.sum(P))
    return 10 * np.log10(p / 4e-10 + 1e-30)


def scene_stats(sid, corpus="tts", root=None):
    """Measured facts of one meeting scene (rendered without E8 impairments, so levels are acoustic)."""
    s2.clear_cache()
    t0 = time.time()
    sc = s2.load_scene(sid, corpus, impair=False, root=root)
    render_s = time.time() - t0
    m, T = sc["meta"], sc["mix"].shape[1]
    count = np.zeros(T, int)
    for a, b, _ in sc["segs"]:
        count[a:b] += 1
    sos = butter(4, [150, 7000], btype="band", fs=FS, output="sos")
    noise = sum(np.asarray(sc["comp2"][g][0], np.float64) for g in ("hvac", "fan", "sensor")) / PA_TO_DIG
    noise_band = sosfiltfilt(sos, noise)
    talk = {}
    for tk, g in sc["talker_group"].items():
        own = np.zeros(T, bool)
        for a, b, k in sc["segs"]:
            if k == tk:
                own[a:b] = True
        solo = own & (count == 1) if (own & (count == 1)).any() else own
        x = np.asarray(sc["comp2"][g][0], np.float64) / PA_TO_DIG
        snr = np.mean(sosfiltfilt(sos, x)[solo] ** 2) / np.mean(noise_band[solo] ** 2)
        talk[tk] = {"level_db": _level(x[solo]), "snr_db": 10 * np.log10(snr)}
    lv = {g: [_level(np.asarray(sc["comp2"][g][c], np.float64) / PA_TO_DIG, w)  # [A Z, A dBA, B Z, B dBA]
              for c in (0, 1) for w in (False, True)] for g in ("hvac", "fan", "sensor")}
    seg0, seg1 = min(a for a, _, _ in sc["segs"]), max(b for _, b, _ in sc["segs"])
    return {"id": sid, "render_s": render_s, "duration_s": T / FS, "meta": m, "talk": talk, "noise": lv,
            "noise_dba": _level(noise, True), "overlap": float((count >= 2).sum() / (count >= 1).sum()),
            "lead_s": seg0 / FS, "tail_s": (T - seg1) / FS, "peak": m["peak"]}


def report(set_name, corpus="tts", root=None):
    """Print the validation summary of a meeting set (geometry truth, levels, conversation, render time)."""
    sids = s2.list_scenes(set_name, "meeting", root=root)
    st = [scene_stats(s, corpus, root) for s in sids]
    tg = [(s["meta"], t, s["talk"][k]) for s in st for k, t in s["meta"]["talker_geo"].items()]
    q = lambda v: f"{np.median(v):6.2f} [{np.min(v):6.2f}, {np.max(v):6.2f}]"  # noqa: E731
    print(f"{set_name}: {len(st)} meeting scenes, corpus {corpus}  (median [min, max])")
    for tail in ("synthetic", "ism"):
        r = [(t["t30_s"] / m["rt60"], t["edt_s"] / m["rt60"]) for m, t, _ in tg if m["room"]["tail"] == tail]
        if r:
            print(f"  {tail:9s} tail ({len(r)} talker RIRs): T30 / RT60 {q([v[0] for v in r])}, "
                  f"EDT / RT60 {q([v[1] for v in r])}")
    mx = [s["meta"]["max_dist_m"] for s in st]
    print(f"  talker distance: min {min(s['meta']['min_dist_m'] for s in st):.2f} m, max per room {q(mx)} m, "
          f"rooms with max >= 5.5 m: {sum(v >= 5.5 for v in mx)}/{len(mx)}")
    for b in ("<2", "2-4", ">4"):
        sel = [(t, x) for _, t, x in tg if t["dist_bin"] == b]
        print(f"  bin {b:>3} m: n {len(sel):3d}  DRR {q([t['drr_db'] for t, _ in sel])} dB  "
              f"(theory {np.median([t['drr_theory_db'] for t, _ in sel]):6.2f})  "
              f"C50 {q([t['c50_db'] for t, _ in sel])} dB  level {q([x['level_db'] for _, x in sel])} dB SPL  "
              f"SNR {q([x['snr_db'] for _, x in sel])} dB")
    for h in MIC_HEIGHTS_CM:
        v = [t["drr_db"] - t["drr_theory_db"] for m, t, _ in tg if m["mic_height_cm"] == h]
        if v:
            print(f"  mic {h:2d} cm above the table: DRR - theory {q(v)} dB")
    spread = [np.ptp([x["level_db"] for x in s["talk"].values()]) for s in st]
    print(f"  talker level spread per scene (max - min, Mic A): {q(spread)} dB")
    for g in ("hvac", "fan", "sensor"):
        print(f"  {g:6s} Mic A dB(Z) {q([s['noise'][g][0] for s in st])}  dBA {q([s['noise'][g][1] for s in st])}")
    err = [s["noise"]["hvac"][2 * c + 1] - s["meta"]["hvac_dba"] for s in st for c in (0, 1)]
    print(f"  HVAC dBA - target, both mics: {q(err)} dB; noise floor at Mic A {q([s['noise_dba'] for s in st])} dBA")
    rm = [s["meta"]["recipe_meta"] for s in st]
    print(f"  duration {q([s['duration_s'] for s in st])} s, turns per talker >= "
          f"{min(min(r['n_turns'].values()) for r in rm)}, interrupts {q([r['n_interrupts'] for r in rm])}, "
          f"back-channels {q([r['n_backchannels'] for r in rm])}, overlap (segs) {q([s['overlap'] for s in st])}")
    print(f"  noise-only lead {min(s['lead_s'] for s in st):.2f} s, tail {min(s['tail_s'] for s in st):.2f} s; "
          f"peak {max(s['peak'] for s in st):.3f}; render time {q([s['render_s'] for s in st])} s")
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v2 meeting scenes: build sets, print a validation report, B.10 check")
    ap.add_argument("--build", nargs="+", choices=list(SETS), help="build these sets into scenes2/")
    ap.add_argument("-j", type=int, default=3, help="worker processes for --build (at most 4)")
    ap.add_argument("--report", choices=list(SETS), help="validation summary of a built set")
    ap.add_argument("--corpus", default="tts")
    ap.add_argument("--b10", action="store_true", help="table comb-notch check (DESIGN B.10, §5)")
    a = ap.parse_args()
    for name in a.build or []:
        t0 = time.time()
        build_set(name, min(a.j, 4))
        print(f"{name}: {time.time() - t0:.0f} s")
    if a.report:
        report(a.report, a.corpus)
    if a.b10:
        for dist in (1.5, 3.0):
            for h in MIC_HEIGHTS_CM:
                meas, pred = notch_check(dist, 0.42, h / 100)
                print(f"talker {dist} m, mouth 0.42 m, mics {h:2d} cm: first notch "
                      f"{'none < 7.9 kHz' if meas is None else f'{meas:7.0f} Hz'} (predicted {pred:7.0f} Hz)")
