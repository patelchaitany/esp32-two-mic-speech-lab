"""Paired spacing scene set: identical bus and meeting layouts rendered with the current board's 12 mm array
and with a 40 mm array (what moving the mics would buy). Natural speech (LibriSpeech, test-split speakers),
the real board's impairments on every scene. Report and usage: study/SCENES.md.

Layout identity
- Bus: every layout is drawn once with lab.scene_bus.sample_layout at 40 mm (the lap-flat acceptance rule
  depends on the spacing); the 12 mm geometry re-places the two mics about the same centre and axis.
- Meeting: lab.scene_meeting.room_layout, whose random draws do not depend on the spacing, called with each
  spacing (asserted: identical room, table and talkers).
- Both spacings share the RIR seed (same randomized image sources and stochastic-tail seed), the recipe
  (speech, levels, noise and render seeds, impairments; asserted equal) and the scene length. Only the two mic
  positions differ. The bus recipe sets transient source levels from the 40 mm geometry's measured band gains
  (meta 'acoustics'; the 12 mm geometry's own measurement is meta 'acoustics_own').

Impairments on every scene (hw/HW_TESTS.md): Mic B skew -63.125 us (-1.01 samples, B early), gain [0, -1.3] dB
(B 1.3 dB low), first-order high-pass corners drawn as the bench draws them (50-80 Hz per mic). Truth: scene meta
'impair'; recipe meta 'factory_calib' = a calib_json dict for methods/_core/calib.py (true spacing, tau0, HPF
corners, gain correction for B), also written to scenes3/spacing/calib/<scene name>.json.

CLI (from the lab folder; -I because the corpus audio lives in data/):
  OMP_NUM_THREADS=1 .venv/bin/python -I study/scene_spacing.py build [-j 2]
  OMP_NUM_THREADS=1 .venv/bin/python -I study/scene_spacing.py validate [-j 2]
"""
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LAB not in sys.path:
    sys.path.insert(0, LAB)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

from lab import scene2 as s2  # noqa: E402
from lab import scene_bus as sb  # noqa: E402
from lab import scene_meeting as smt  # noqa: E402
from lab.common import C, FS  # noqa: E402

ROOT = os.path.join(LAB, "scenes3")
SET = "spacing"
CORPUS = "libri"
SPACINGS_MM = (12, 40)
LAYOUT_MM = 40                      # bus layouts are drawn at this spacing and reused for the others
SPEAKER_SPLIT = "test"              # LibriSpeech test-split partners, babble and meeting talkers (task spec)
BUS_GEN = "study.scene_spacing:bus_recipe"
MEETING_GEN = "study.scene_spacing:meeting_recipe"
BUS_SEED0 = 61000                   # unused by every scenes2 set (dev 10000, test 20000, smoke 30000)
MEETING_SEED0 = 481000              # unused (meeting dev 410000, test 430000, tuning 470000+, smoke 490000)
BUS_SNRS = (-5, 0)
BUS_SECONDS = 24.0
BUS_PER_STRATUM = 2
MEETING_TARGET_S = 100.0            # the generator's target length; min_turns 1 lets it reach ~100 s
MEETING_MIN_TURNS = 1
# The real board (hw/HW_TESTS.md, hw/analysis.json): skew of Mic B -1.01 samples after the current boot
# (-1.09..-1.11 at the first boot), B 1.3 dB below A (direct sound 150-7000 Hz).
BOARD = {"name": "esp32c3_2x_inmp441", "skew_samples": -1.01, "skew_us": -1.01 / FS * 1e6, "gain_db": [0.0, -1.3],
         "hp": "bench draw: 50-80 Hz per mic (lab.scene2.draw_impairments)",
         "source": "hw/HW_TESTS.md (8 esptool resets: -1.01 +- 0.01 samples; band gains -0.8..-1.9 dB, mean -1.3)"}
N_BANDS = 18                        # calib.json gain_db: one-third octaves 125 Hz - 6.35 kHz


# ----------------------------------------------------------------------------- impairments and calib truth
def board_impair(drawn):
    """The board's E8 truth: drawn HPF corners kept (the bench's), gain and skew replaced by the board's."""
    return {"gain_db": [float(v) for v in BOARD["gain_db"]], "hp_hz": [float(v) for v in drawn["hp_hz"]],
            "skew_us": round(float(BOARD["skew_us"]), 4)}


def factory_calib(spacing_m, impair):
    """calib_json (methods/_core/API.md schema) with the scene's truth: spacing, tau0 (+ = B late), the two HPF
    corners and the per-band gain correction applied to Mic B (+1.3 dB: B is 1.3 dB low). No 'model': the
    diffuse-field model is a property of the room, not of the device."""
    g = float(impair["gain_db"][0] - impair["gain_db"][1])
    return {"version": 1, "spacing_m": round(float(spacing_m), 6), "tau0_s": float(impair["skew_us"]) * 1e-6,
            "hpf_corner_hz": [float(v) for v in impair["hp_hz"]], "gain_db": [round(g, 4)] * N_BANDS,
            "note": "factory calibration = scene truth (study/scene_spacing.py)"}


def _finish_recipe(rec, geom, speaker_split):
    rec["impair"] = board_impair(rec["impair"])
    rec.setdefault("meta", {})
    rec["meta"].update(board=dict(BOARD), speaker_split=speaker_split, layout_id=geom["meta"].get("layout_id"),
                       factory_calib=factory_calib(geom["spacing_m"], rec["impair"]))
    return rec


# ----------------------------------------------------------------------------- generators (scene2 contract)
def bus_recipe(geom, corpus, seed, snr_db, seconds=BUS_SECONDS, speaker_split=SPEAKER_SPLIT):
    """lab.scene_bus.make_recipe with speakers from `speaker_split` (scene_bus picks the test split only for
    set 'test') and the board's impairments. Corpus-independent draws are unchanged."""
    rec = sb.make_recipe(dict(geom, set=speaker_split), corpus, seed, snr_db, seconds)
    return _finish_recipe(rec, geom, speaker_split)


def meeting_recipe(geom, corpus, seed, hvac_dba=40.0, target_s=MEETING_TARGET_S, min_turns=MEETING_MIN_TURNS,
                   speaker_split=SPEAKER_SPLIT):
    """lab.scene_meeting.make_recipe with `speaker_split` talkers and the board's impairments. The scene
    length is recomputed from the array centre (the core uses Mic A's direct path, which moves with the
    spacing), so both spacings of a room have the same length."""
    rec = smt.make_recipe(dict(geom, set=speaker_split), corpus, seed, hvac_dba, target_s, min_turns)
    end_s = max(t[3] for t in rec["meta"]["turns"])
    lag_s = np.ceil(max(s["dist_m"] for s in geom["sources"]) / C * 1000) / 1000 + s2.FRAC / FS + 0.01
    T = int(np.ceil((end_s + lag_s + geom["room"]["rt60"] + smt.LEAD_S) * FS))
    rec["n_samples"] = max(T, max(int(e[1]) + 1 for g in rec["groups"] for s in g["sources"]
                                  for e in s["signal"].get("events", [])))
    return _finish_recipe(rec, geom, speaker_split)


# ----------------------------------------------------------------------------- design
def bus_design():
    """10 layouts: 2 per E1 stratum. Replicates differ in RT60 half (0.15-0.375 / 0.375-0.6 s), handling
    (one of the two), diffuse road model (one sinc, one j0) and, in the opposed strata, the distance-bin pair."""
    spec = {"seed0": BUS_SEED0, "sweep": False, "snrs": BUS_SNRS, "seconds": BUS_SECONDS}
    rng = np.random.default_rng([BUS_SEED0, 0x5ACE])
    rows = []
    for si, stratum in enumerate(sb.STRATA):
        pairs = rng.permutation(len(sb.DIST_DESIGN))[:BUS_PER_STRATUM]
        half = rng.permutation(2)
        hand = int(rng.integers(2))
        flip = int(rng.integers(2))
        for k in range(BUS_PER_STRATUM):
            idx = k * len(sb.STRATA) + si
            lo = 0.15 + half[k] * 0.225
            rt60 = round(float(rng.uniform(lo + 0.001, lo + 0.224)), 3)
            bins = sb.DIST_DESIGN[pairs[k]] if stratum in sb.FOLD_RANGE else None
            row = sb._row(SET, idx, stratum, LAYOUT_MM / 1000, "synthetic", rt60, bins, k == hand, False,
                          round(BOARD["skew_us"], 4), ("sinc", "j0")[(k + flip) % 2], spec, replicate=k)
            row["layout_id"] = f"L{idx:02d}"
            rows.append(row)
    return sorted(rows, key=lambda r: r["idx"])


def meeting_design():
    """4 rooms (RT60 stratified over 0.5-0.9 s); mic height, HVAC, axis mode, presenter balanced; one room
    with the full image-source tail. Spacing is set per build (12 / 40 mm)."""
    rng = np.random.default_rng([MEETING_SEED0, 7])
    rt60 = 0.5 + 0.4 * (np.arange(4) + rng.uniform(size=4)) / 4
    fixed = [(1, 40.0, "table", True, "synthetic"), (2, 35.0, "random", False, "ism"),
             (10, 45.0, "table", False, "synthetic"), (2, 40.0, "random", True, "synthetic")]
    return [{"set": SET, "index": i, "seed": MEETING_SEED0 + i, "layout_id": f"R{i:02d}", "rt60": round(float(rt60[i]), 4),
             "table": True, "target_s": MEETING_TARGET_S, "min_turns": MEETING_MIN_TURNS, "mic_height_cm": mh,
             "hvac_dba": hv, "axis_mode": ax, "presenter": pr, "tail": tl}
            for i, (mh, hv, ax, pr, tl) in enumerate(fixed)]


def bus_scene_name(row, mm, snr):
    return f"bus_{row['layout_id']}_d{mm}_snr{int(snr):+d}"


def bus_geom_id(row, mm):
    return f"bus_{SET}_{row['layout_id']}_d{mm}"


def meeting_scene_name(d, mm):
    return f"meeting_{d['layout_id']}_d{mm}"


def meeting_geom_id(d, mm):
    return f"meeting_{SET}_{d['layout_id']}_d{mm}"


# ----------------------------------------------------------------------------- build
def _replace_mics(mics, spacing_m):
    """Same centre and axis, new spacing (positions rounded to 1 um as make_geometry stores them)."""
    a, b = (np.asarray(m, float) for m in mics)
    c, u = (a + b) / 2, (a - b) / np.linalg.norm(a - b)
    az, el = float(np.degrees(np.arctan2(u[1], u[0]))), float(np.degrees(np.arcsin(np.clip(u[2], -1, 1))))
    return [[round(float(v), 6) for v in m] for m in s2.place_array(c, spacing_m, az, el)]


def _bus_meta(meta40, mics, sources, spacing_m):
    """Geometry meta for another spacing: the spacing-dependent truth recomputed on the new mics."""
    m = copy.deepcopy(meta40)
    tr = {s["name"]: s2.source_truth(s["pos"], mics) for s in sources}
    ta, tb = tr["pA"], tr["pB"]
    why = []
    if abs(ta["tau_us"] - tb["tau_us"]) < 25:
        why.append("close_tdoas")
    if max(abs(ta["tau_us"]), abs(tb["tau_us"])) < 0.3 * spacing_m / C * 1e6:
        why.append("both_broadside")
    m.update(spacing_mm=int(round(spacing_m * 1000)), dist_A_m=ta["dist_m"], dist_B_m=tb["dist_m"],
             partner_dist_max_m=max(ta["dist_m"], tb["dist_m"]), theta_A_deg=ta["theta_deg"],
             theta_B_deg=tb["theta_deg"], fold_A_deg=ta["fold_deg"], fold_B_deg=tb["fold_deg"],
             tau_A_us=ta["tau_us"], tau_B_us=tb["tau_us"], angle_class=sb.classify(ta["theta_deg"], tb["theta_deg"]),
             inseparable=bool(why), inseparable_why=why, cone_dist_m=tr["cone"]["dist_m"],
             cone_dtheta_deg=round(tr["cone"]["theta_deg"] - ta["theta_deg"], 3))
    return m


def _write_geom_json(geom):
    s2._write_json(os.path.join(ROOT, SET, geom["geom_id"] + ".geom.json"), geom)
    s2.clear_cache()


def _write_calib(scene_id):
    rec = s2.ensure_recipe(scene_id, CORPUS, ROOT)
    d = os.path.join(ROOT, SET, "calib")
    os.makedirs(d, exist_ok=True)
    s2._write_json(os.path.join(d, scene_id.split("/")[1] + ".json"), rec["meta"]["factory_calib"])


def build_bus_layout(row):
    """Both spacings of one bus layout: geometries (RIRs, truth, acoustics) and the SNR scenes."""
    t0 = time.time()
    s2.clear_cache()
    lay = sb.sample_layout(row)  # drawn at LAYOUT_MM
    srcs = lay["sources"]
    out = {"layout_id": row["layout_id"], "stratum": row["stratum"], "scenes": [], "geoms": {}}
    meta40 = None
    for mm in sorted(SPACINGS_MM, key=lambda v: v != LAYOUT_MM):  # layout spacing first
        mics = lay["mics"] if mm == LAYOUT_MM else _replace_mics(lay["mics"], mm / 1000)
        base = dict(lay["meta"], layout_id=row["layout_id"], layout_spacing_mm=LAYOUT_MM, board=dict(BOARD),
                    skew_us=round(BOARD["skew_us"], 4), speaker_split=SPEAKER_SPLIT)
        meta = base if mm == LAYOUT_MM else _bus_meta(meta40, mics, srcs, mm / 1000)
        geom = s2.make_geometry(SET, bus_geom_id(row, mm), "bus", lay["dims"], row["rt60"], mics, srcs, row["seed"],
                                BUS_GEN, meta=meta, root=ROOT, tail=row["tail"])
        _, rirs = s2.load_geometry(SET, geom["geom_id"], root=ROOT)
        own = sb.measure_acoustics(rirs, geom["sources"], geom["mics"])
        if mm == LAYOUT_MM:
            geom["meta"]["acoustics"] = own
            meta40 = copy.deepcopy(geom["meta"])
        else:  # transient source levels follow the layout geometry's band gains: identical recipes
            geom["meta"]["acoustics"] = copy.deepcopy(meta40["acoustics"])
            geom["meta"]["acoustics_own"] = own
            geom["meta"]["acoustics_ref_mm"] = LAYOUT_MM
        _write_geom_json(geom)
        out["geoms"][mm] = geom["geom_id"]
        for snr in row["snrs"]:
            sid = s2.add_scene(SET, bus_scene_name(row, mm, snr), geom,
                               {"seed": row["seed"], "snr_db": float(snr), "seconds": row["seconds"],
                                "speaker_split": SPEAKER_SPLIT}, corpus=CORPUS, root=ROOT)
            _write_calib(sid)
            out["scenes"].append(sid)
    out["build_s"] = round(time.time() - t0, 1)
    return out


def build_meeting_room(d):
    """Both spacings of one meeting room (scene_meeting.build_room with this module's generator)."""
    t0 = time.time()
    s2.clear_cache()
    out = {"layout_id": d["layout_id"], "scenes": [], "geoms": {}}
    ref = None
    for mm in SPACINGS_MM:
        dd = dict(d, spacing_m=mm / 1000, name=meeting_scene_name(d, mm), geom_id=meeting_geom_id(d, mm))
        lay = smt.room_layout(dd)
        key = json.dumps([lay["dims"], lay["table"], lay["sources"], lay["seat"]], sort_keys=True)
        if ref is None:
            ref = key
        elif key != ref:
            raise RuntimeError(f"{d['layout_id']}: room layout depends on the spacing")
        meta = {"purpose": "meeting", "room_index": d["index"], "layout_id": d["layout_id"],
                "spacing_mm": mm, "mic_height_cm": d["mic_height_cm"], "axis_mode": d["axis_mode"],
                "presenter": d["presenter"], "table": lay["table"],
                "n_row2": sum(k == "row2" for k in lay["seat"].values()), "board": dict(BOARD),
                "speaker_split": SPEAKER_SPLIT}
        hook = smt.table_hook(lay["table"], lay["table"]["refl"]) if lay["table"]["on"] else None
        geom = s2.make_geometry(SET, dd["geom_id"], "meeting", lay["dims"], d["rt60"], lay["mics"], lay["sources"],
                                d["seed"], MEETING_GEN, meta=meta, root=ROOT, tail=d["tail"], edit_images=hook)
        _, rirs = s2.load_geometry(SET, dd["geom_id"], root=ROOT)
        geom["meta"].update(smt._room_truth(geom, rirs, lay))
        _write_geom_json(geom)
        out["geoms"][mm] = geom["geom_id"]
        sid = s2.add_scene(SET, dd["name"], geom, {"seed": d["seed"], "hvac_dba": d["hvac_dba"],
                                                   "target_s": d["target_s"], "min_turns": d["min_turns"],
                                                   "speaker_split": SPEAKER_SPLIT}, corpus=CORPUS, root=ROOT)
        _write_calib(sid)
        out["scenes"].append(sid)
    out["build_s"] = round(time.time() - t0, 1)
    return out


def _build_job(job):
    kind, row = job
    try:
        return build_bus_layout(row) if kind == "bus" else build_meeting_room(row)
    except Exception as e:  # report, do not kill the pool
        import traceback
        return {"layout_id": row["layout_id"], "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}


def recipe_identity(scene_a, scene_b):
    """True when two recipes are identical apart from ids, the true spacing in the calib truth and layout tags."""
    ra, rb = (copy.deepcopy(s2.ensure_recipe(s, CORPUS, ROOT)) for s in (scene_a, scene_b))
    for r in (ra, rb):
        for k in ("scene_id", "geom_id"):
            r.pop(k)
        r["meta"]["factory_calib"].pop("spacing_m")
    return ra == rb


def build(jobs=2, only=None):
    os.makedirs(os.path.join(ROOT, SET), exist_ok=True)
    work = [("bus", r) for r in bus_design()] + [("meeting", d) for d in meeting_design()]
    if only:
        work = [w for w in work if w[1]["layout_id"] in only]
    work.sort(key=lambda w: (w[0] != "meeting", w[1].get("tail") != "ism"))  # slow rooms first
    t0 = time.time()
    if jobs <= 1:
        logs = [_build_job(w) for w in work]
    else:
        with get_context("spawn").Pool(jobs) as p:
            logs = p.map(_build_job, work, chunksize=1)
    bad = [g for g in logs if "error" in g]
    for g in logs:
        print(json.dumps({k: v for k, v in g.items() if k != "tb"}), flush=True)
    ident = {}
    for g in logs:
        if "error" in g:
            continue
        by = {}
        for sid in g["scenes"]:
            key = sid.replace("_d12", "_dXX").replace("_d40", "_dXX")
            by.setdefault(key, []).append(sid)
        for key, sids in by.items():
            ident[key] = recipe_identity(*sids) if len(sids) == 2 else None
    print(f"built {len(logs) - len(bad)} layouts ({len(bad)} failed) in {time.time() - t0:.0f} s; recipes identical "
          f"across spacings: {sum(v is True for v in ident.values())}/{len(ident)}", flush=True)
    for g in bad:
        print(g["tb"])
    with open(os.path.join(ROOT, SET, "build_log.json"), "w") as f:
        json.dump({"logs": [{k: v for k, v in g.items() if k != "tb"} for g in logs], "recipe_identity": ident,
                   "wall_s": round(time.time() - t0, 1)}, f, indent=1)
    return logs


# ----------------------------------------------------------------------------- validation
def _inband(x):
    return sb._inband(np.asarray(x, np.float64))


def _hp_phase(f, fc):
    from scipy.signal import butter, freqz
    b, a = butter(1, fc, "highpass", fs=FS)
    return freqz(b, a, worN=f, fs=FS)[1]


def _welch_xy(x, y, nper=1024):
    from scipy.signal import csd, welch
    f, sxy = csd(x, y, fs=FS, nperseg=nper)
    _, sxx = welch(x, fs=FS, nperseg=nper)
    _, syy = welch(y, fs=FS, nperseg=nper)
    return f, sxy, sxx, syy


def coherence_truth(xab, imp, nper=1024):
    """Complex coherence of a 2-ch signal with the known chain (HPF phases, skew) removed: (f, gamma)."""
    f, sab, saa, sbb = _welch_xy(xab[0], xab[1], nper)  # S_AB = E[A conj(B)] (scipy: conj(x) y -> swap)
    g = np.conj(sab) / np.sqrt(saa * sbb + 1e-300)       # scipy csd(x, y) = E[conj(X) Y]
    ha, hb = _hp_phase(np.maximum(f, 1.0), imp["hp_hz"][0]), _hp_phase(np.maximum(f, 1.0), imp["hp_hz"][1])
    # observed gamma = gamma_ac * exp(j(arg Ha - arg Hb)) * exp(+j 2 pi f skew)
    g = g * np.exp(-1j * (np.angle(ha) - np.angle(hb))) * np.exp(-2j * np.pi * f * imp["skew_us"] * 1e-6)
    return f, g


def chain_fit(x_imp, x_ideal, imp):
    """Per-mic transfer function impaired/ideal (Welch): gain at 0.5-4 kHz and skew from the phase slope over
    0.2-6 kHz after removing the known first-order HPF phase."""
    out = {}
    for m in range(2):
        f, sxy, sxx, _ = _welch_xy(x_ideal[m], x_imp[m], 2048)
        H = sxy / (sxx + 1e-300)
        hp = _hp_phase(np.maximum(f, 1.0), imp["hp_hz"][m])
        band = (f >= 500) & (f <= 4000)
        gain_db = float(10 * np.log10(np.mean(np.abs(H[band]) ** 2 / np.abs(hp[band]) ** 2)))
        sel = (f >= 200) & (f <= 6000)
        ph = np.unwrap(np.angle(H[sel] / hp[sel]))
        w = np.abs(H[sel])
        slope = float(np.sum(w * f[sel] * ph) / np.sum(w * f[sel] ** 2))  # phase = -2 pi f delay
        out["AB"[m]] = {"gain_db": round(gain_db, 3), "delay_us": round(-slope / (2 * np.pi) * 1e6, 3)}
    return out


def tdoa_phase(xab, imp, f_lo=300.0, f_hi=3000.0):
    """Inter-mic delay (us, + = B later) of a single-source 2-ch signal from the cross-spectrum phase slope
    (weighted LS through 0) over f_lo-f_hi, after removing the known HPF phase difference."""
    f, sab, saa, sbb = _welch_xy(xab[0], xab[1], 1024)
    cab = np.conj(sab)  # E[A conj(B)]: phase = arg A - arg B = +2 pi f tau (B later by tau)
    ha, hb = _hp_phase(np.maximum(f, 1.0), imp["hp_hz"][0]), _hp_phase(np.maximum(f, 1.0), imp["hp_hz"][1])
    cab = cab * np.exp(-1j * (np.angle(ha) - np.angle(hb)))
    sel = (f >= f_lo) & (f <= f_hi)
    ph = np.unwrap(np.angle(cab[sel]))
    w = np.abs(cab[sel])
    return float(np.sum(w * f[sel] * ph) / np.sum(w * f[sel] ** 2) / (2 * np.pi) * 1e6)


COH_F = (250, 500, 1000, 2000, 3000, 4000, 6000)


def _coh_at(f, g, fs_list=COH_F, half_oct=True):
    out = {}
    for fc in fs_list:
        sel = (f >= fc / 2 ** (1 / 6)) & (f <= fc * 2 ** (1 / 6))
        out[str(fc)] = round(float(np.mean(g[sel].real)), 3)
    return out


def validate_scene(sid):
    """Measured facts of one scene (impaired and ideal renders)."""
    s2.clear_cache()
    t0 = time.time()
    sc = s2.load_scene(sid, CORPUS, impair=True, root=ROOT)
    dt = time.time() - t0
    m = sc["meta"]
    imp = m["impair"]
    d = m["spacing_m"]
    rec = {"scene": sid, "scenario": m["scenario"], "layout_id": m.get("layout_id"), "spacing_mm": m["spacing_mm"],
           "duration_s": round(m["duration_s"], 2), "render_s": round(dt, 2), "peak": round(m["peak"], 4),
           "clipped": m["clipped"], "impair": {k: imp[k] for k in ("gain_db", "hp_hz", "skew_us")}}
    T = sc["mix"].shape[1]
    if m["scenario"] == "bus":
        rule = m["rules"][0]
        rec.update(stratum=m["stratum"], snr_db=rule["snr_db"], realized_snr_db=rule["realized_snr_db"],
                   rule_gain_db=rule["gain_db"], tau_A_us=m["tau_A_us"], tau_B_us=m["tau_B_us"],
                   dist_A_m=m["dist_A_m"], dist_B_m=m["dist_B_m"], fold_A_deg=m["fold_A_deg"],
                   fold_B_deg=m["fold_B_deg"], road_model=m["road_model"], handling=m["handling"],
                   speakers=[m["recipe_meta"]["speakers"][t] for t in ("pA", "pB")])
        mask = np.zeros(T, bool)
        for a, b, tk in sc["segs"]:
            mask[a:b] |= tk in ("pA", "pB")
        tgt = sum(np.asarray(sc["comp2"][g][0], np.float64) for g in rule["target"])
        noise = sum(np.asarray(sc["comp2"][g], np.float64) for g in rule["noise"])
        rec["partner_dbfs"] = round(float(10 * np.log10(np.mean(_inband(tgt)[mask] ** 2))), 2)
        rec["rule_noise_dbfs"] = round(float(10 * np.log10(np.mean(_inband(noise[0])[mask] ** 2))), 2)
        rest = sum(np.asarray(sc["comp2"][g][0], np.float64) for g in sc["comp2"] if g not in rule["target"])
        rec["all_inclusive_snr_db"] = round(float(10 * np.log10(np.mean(_inband(tgt)[mask] ** 2)
                                                               / np.mean(_inband(rest)[mask] ** 2))), 2)
        f, g = coherence_truth(noise, imp)
        rec["coh_rule_noise"] = _coh_at(f, g)
        f, g = coherence_truth(np.asarray(sc["comp2"]["babble_far"], np.float64), imp)
        rec["coh_babble_far"] = _coh_at(f, g)
        # partner TDOA from its own component (direct path dominates the 0.3-1 m partners)
        for k, grp in (("A", "partner_A"), ("B", "partner_B")):
            x = np.asarray(sc["comp2"][grp], np.float64)
            rec[f"tdoa_meas_{k}_us"] = round(tdoa_phase(x, imp), 2)
            rec[f"tdoa_truth_{k}_us"] = round(m[f"tau_{k}_us"] + imp["skew_us"], 2)
        ideal = s2.load_scene(sid, CORPUS, impair=False, root=ROOT)
        rec["ideal_snr_db"] = ideal["meta"]["rules"][0]["realized_snr_db"]
        rec["chain"] = chain_fit(np.asarray(sc["mix"], np.float64), np.asarray(ideal["mix"], np.float64), imp)
        rec["partner_dba"] = round(sb._dba(sb._mic_a(ideal, rule["target"])[mask]), 2)
        rec["rule_noise_dba"] = round(sb._dba(sb._mic_a(ideal, rule["noise"])), 2)
    else:
        rm = m["recipe_meta"]
        segs = sc["segs"]
        cnt = np.zeros(T, int)
        for a, b, _ in segs:
            cnt[a:b] += 1
        alone = {}
        for tk in sc["talker_group"]:
            own = np.zeros(T, bool)
            for a, b, k in segs:
                if k == tk:
                    own[a:b] = True
            alone[tk] = float((own & (cnt == 1)).sum() / FS)
        al = np.array(list(alone.values()))
        rec.update(n_talkers=len(alone), alone_s_min=round(float(al.min()), 2), alone_s_median=round(float(np.median(al)), 2),
                   n_talkers_alone_lt1s=int((al < 1).sum()), n_talkers_alone_lt3s=int((al < 3).sum()),
                   overlap_ratio=rm["overlap_ratio"], n_turns_total=sum(rm["n_turns"].values()),
                   hvac_dba=m["hvac_dba"], mic_height_cm=m["mic_height_cm"], axis_mode=m["axis_mode"],
                   presenter=m["presenter"], tail=m["room"]["tail"], rt60=m["rt60"])
        f, g = coherence_truth(np.asarray(sc["comp2"]["hvac"], np.float64), imp)
        rec["coh_hvac"] = _coh_at(f, g)
        noise = sum(np.asarray(sc["comp2"][g], np.float64) for g in ("hvac", "fan", "sensor"))
        f, g = coherence_truth(noise, imp)
        rec["coh_noise"] = _coh_at(f, g)
        talk = sum(np.asarray(sc["comp2"][g][0], np.float64) for g, r in sc["groups"].items() if r == "talker")
        act = cnt > 0
        rec["speech_dbfs"] = round(float(10 * np.log10(np.mean(_inband(talk)[act] ** 2))), 2)
        rec["noise_dbfs"] = round(float(10 * np.log10(np.mean(_inband(noise[0]) ** 2))), 2)
        # nearest talker's TDOA from its own component (direct path + table image)
        geo = m["talker_geo"]
        tk = min(geo, key=lambda t: geo[t]["dist_m"])
        x = np.asarray(sc["comp2"][sc["talker_group"][tk]], np.float64)
        rec["nearest_talker"] = tk
        rec["tdoa_meas_near_us"] = round(tdoa_phase(x, imp), 2)
        rec["tdoa_truth_near_us"] = round(geo[tk]["tau_us"] + imp["skew_us"], 2)
        ideal = s2.load_scene(sid, CORPUS, impair=False, root=ROOT)
        rec["chain"] = chain_fit(np.asarray(sc["mix"], np.float64), np.asarray(ideal["mix"], np.float64), imp)
    th = {"sinc": lambda f: np.sinc(2 * f * d / C)}
    rec["coh_model_sinc"] = {str(fc): round(float(th["sinc"](fc)), 3) for fc in COH_F}
    from scipy.special import j0
    rec["coh_model_j0"] = {str(fc): round(float(j0(2 * np.pi * fc * d / C)), 3) for fc in COH_F}
    return rec


def _val_job(sid):
    try:
        return validate_scene(sid)
    except Exception as e:
        import traceback
        return {"scene": sid, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}


def validate(jobs=2, scenes=None, out=None):
    sids = scenes or s2.list_scenes(SET, root=ROOT)
    t0 = time.time()
    if jobs <= 1:
        recs = [_val_job(s) for s in sids]
    else:
        with get_context("spawn").Pool(jobs) as p:
            recs = p.map(_val_job, sids, chunksize=1)
    out = out or os.path.join(LAB, "study", "scene_spacing_validation.json")
    with open(out, "w") as f:
        json.dump({"wall_s": round(time.time() - t0, 1), "scenes": recs}, f, indent=1)
    print(f"validated {len(recs)} scenes in {time.time() - t0:.0f} s -> {out}")
    for r in recs:
        if "error" in r:
            print(r["scene"], r["error"])
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description="paired 12 / 40 mm spacing scene set (study/SCENES.md)")
    ap.add_argument("cmd", choices=("build", "validate", "design"))
    ap.add_argument("-j", type=int, default=2, help="worker processes (default 2)")
    ap.add_argument("--only", help="comma list of layout ids (L00..L09, R00..R03)")
    ap.add_argument("--scenes", help="validate: comma list of scene ids (default: all)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.j, a.only.split(",") if a.only else None)
    elif a.cmd == "validate":
        validate(a.j, a.scenes.split(",") if a.scenes else None)
    else:
        for r in bus_design():
            print(json.dumps({k: r[k] for k in ("layout_id", "seed", "stratum", "rt60", "dist_bins", "handling",
                                                 "road_model", "snrs")}))
        for d in meeting_design():
            print(json.dumps(d))


if __name__ == "__main__":
    main()
