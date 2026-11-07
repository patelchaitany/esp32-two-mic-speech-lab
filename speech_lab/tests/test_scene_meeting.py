"""Tests for the v2 meeting scene generator (lab.scene_meeting). Run: .venv/bin/python -m tests.test_scene_meeting

Builds two rooms in a temporary root (scenes2/ is not touched): 'full', a 3 min scene with the synthetic tail,
and 'short', a smoke-length scene with the full image-source tail.
"""
import atexit
import json
import os
import shutil
import tempfile
import time

import numpy as np
from scipy.signal import oaconvolve
from scipy.stats import spearmanr

from lab import corpus2
from lab import scene2 as s2
from lab import scene_meeting as sm
from lab.common import FS, PA_TO_DIG, spl_to_pa

_ROOT = []


def _design(i, **kw):
    d = dict(sm.set_design("smoke")[i])
    d.update(set="t", **kw)
    return d


DESIGNS = {"full": _design(0, name="full", geom_id="g_full", tail="synthetic", target_s=180.0, min_turns=3,
                           presenter=True, axis_mode="table", mic_height_cm=1),
           "short": _design(1, name="short", geom_id="g_short", tail="ism", rt60=0.5, mic_height_cm=10)}


def _root():
    """Temporary scenes root holding rooms 'full' and 'short' (built once per process)."""
    if not _ROOT:
        _ROOT.append(tempfile.mkdtemp(prefix="meeting_test_"))
        atexit.register(shutil.rmtree, _ROOT[0], True)
        for d in DESIGNS.values():
            sm.build_room(d, _ROOT[0])
    return _ROOT[0]


def _scene(name, impair=True, corpus="tts"):
    return s2.load_scene(f"t/{name}", corpus=corpus, impair=impair, root=_root())


def _recipe(name, corpus="tts"):
    return json.load(open(os.path.join(_root(), "t", f"{name}.{corpus}.recipe.json")))


def test_b10_table_notch():
    """DESIGN B.10: talker 3 m away, mouth 0.42 m and mics 10 cm above the table: first notch 6.2 kHz ± 5 %."""
    meas, pred = sm.notch_check(3.0, 0.42, 0.10)
    assert abs(meas / 6200 - 1) < 0.05 and abs(meas / pred - 1) < 0.02, (meas, pred)
    # DESIGN §5: 20 cm -> 3.1 kHz; 1.5 m away -> 3.2 / 1.6 kHz; within 2 cm the first notch is above 15 kHz
    for dist, h, want in ((3.0, 0.20, 3100), (1.5, 0.10, 3200), (1.5, 0.20, 1600)):
        meas, pred = sm.notch_check(dist, 0.42, h)
        assert abs(meas / want - 1) < 0.05 and abs(meas / pred - 1) < 0.02, (dist, h, meas, pred)
    meas, pred = sm.notch_check(1.5, 0.42, 0.02)
    assert meas is None and pred > 15000


def test_floor_image_zeroed_and_table_image_added():
    """B.10 second part: the hook removes exactly the floor image and adds the table image, as predicted."""
    dims, zt = [8.0, 6.0, 3.0], 0.74
    table = {"x": [0.0, 8.0], "y": [2.0, 4.0], "z": zt}
    mics = s2.place_array([4.0, 3.0, zt + 0.10], 0.04, 0.0)
    src = [4.3, 5.2, zt + 0.45]  # seated across the table edge, 2.2 m away
    kw = dict(max_order=1, tail="none", rand_disp=0.0)
    r0, _, par = s2.compute_rirs(dims, 0.4, [src], mics, 1, **kw)
    rt = s2.compute_rirs(dims, 0.4, [src], mics, 1, edit_images=sm.table_hook(table, zero_floor=False), **kw)[0]
    rtf = s2.compute_rirs(dims, 0.4, [src], mics, 1, edit_images=sm.table_hook(table), **kw)[0]
    h0, ht, htf = (r[0, 0].astype(np.float64) for r in (r0, rt, rtf))
    d_dir = s2.source_truth(src, mics)["direct_a"]

    def arrival(img):
        return np.linalg.norm(np.subtract(img, mics[0])) / s2.C * FS + s2.FRAC

    def energy(h, c):
        return np.sum(h[int(c) - 20:int(c) + 21] ** 2)

    def check(diff, img, amp2):  # one image: arrival at the predicted delay, energy re the direct path
        k = int(np.argmax(np.abs(diff)))
        assert abs(k - arrival(img)) <= 1, (k, arrival(img))
        rel = amp2 * (np.linalg.norm(np.subtract(src, mics[0])) / np.linalg.norm(np.subtract(img, mics[0]))) ** 2
        assert abs(energy(diff, arrival(img)) / energy(h0, d_dir) / rel - 1) < 0.05
        far = np.abs(np.arange(len(diff)) - arrival(img)) > 45  # nothing else changed (filter leakage aside)
        assert np.sum(diff[far] ** 2) < 1e-3 * np.sum(diff[~far] ** 2)
    check(ht - h0, [src[0], src[1], 2 * zt - src[2]], sm.TABLE_REFL ** 2)  # table image added, factor 0.9
    check(ht - htf, [src[0], src[1], -src[2]], 1 - par["e_abs"])           # floor image removed


def test_path_rules():
    """table_paths: floor paths to a mic above the table centre are always blocked; edge cases are not."""
    table = {"x": [0.0, 8.0], "y": [0.0, 2.0], "z": 0.74}
    mic = [4.0, 1.0, 0.84]
    rng = np.random.default_rng(0)
    for _ in range(200):  # any mouth in a 12 x 7 m room around the table, seated or standing
        s = [rng.uniform(-2, 10), rng.uniform(-2.5, 4.5), rng.uniform(1.1, 1.7)]
        if sm._inside(s, table):
            continue
        f = sm.table_paths(s, mic, table)
        assert f["floor_blocked"] and f["table_image"], s
    # mic 20 cm above the table edge, source 4 m away on that side: the floor path passes beside the table
    # and the tabletop specular point lies beyond the edge
    f = sm.table_paths([4.0, 6.0, 1.2], [4.0, 1.95, 0.94], table)
    assert not (f["floor_blocked"] or f["floor_under"] or f["table_image"])
    # high mic near the table end, source far beyond it: the table specular point falls off the table
    f = sm.table_paths([12.0, 1.0, 1.24], [7.9, 1.0, 0.94], table)
    assert not f["table_image"]


def test_set_design_balanced():
    seeds = {}
    for name, cfg in sm.SETS.items():
        des = sm.set_design(name)
        assert len(des) == cfg["n"] and len({d["name"] for d in des}) == cfg["n"]
        seeds[name] = {d["seed"] for d in des}
        if cfg["n"] < 20:
            continue
        for key, levels in (("spacing_m", sm.SPACINGS), ("mic_height_cm", sm.MIC_HEIGHTS_CM),
                            ("hvac_dba", sm.HVAC_DBA), ("axis_mode", ("table", "random")),
                            ("presenter", (True, False)), ("tail", ("synthetic", "ism"))):
            n = [sum(d[key] == v for d in des) for v in levels]
            assert sum(n) == cfg["n"] and max(n) - min(n) <= 1, (name, key, n)
        rt = np.sort([d["rt60"] for d in des])
        assert np.all((rt >= 0.5 + 0.4 * np.arange(20) / 20) & (rt <= 0.5 + 0.4 * np.arange(1, 21) / 20))
    assert not (seeds["dev"] & seeds["test"]) and not (seeds["smoke"] & (seeds["dev"] | seeds["test"]))
    assert sm.set_design("dev") == sm.set_design("dev")


def test_layouts():
    """Every dev and test layout: room, table, device, 20 talkers out to 5.5-6.2 m, presenter facing away."""
    for d in sm.set_design("dev") + sm.set_design("test"):
        lay = sm.room_layout(d)
        lx, ly, lz = lay["dims"]
        assert 11.04 <= lx <= 12.96 and 6.44 <= ly <= 7.56 and 2.76 <= lz <= 3.24
        t = lay["table"]
        assert np.isclose(t["x"][1] - t["x"][0], 8.0) and np.isclose(t["y"][1] - t["y"][0], 2.0)
        assert 0.72 <= t["z"] <= 0.76 and t["refl"] == 0.9
        a, b = np.asarray(lay["mics"])
        c = (a + b) / 2
        assert np.isclose(np.linalg.norm(a - b), d["spacing_m"]) and np.isclose(c[2] - t["z"], d["mic_height_cm"] / 100)
        assert abs(c[0] - np.mean(t["x"])) <= 0.1 + 1e-9 and abs(c[1] - np.mean(t["y"])) <= 0.1 + 1e-9
        if d["axis_mode"] == "table":
            az = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0])) % 180
            assert min(az, 180 - az) <= 3 + 1e-6
        talkers = lay["sources"][:-1]
        assert [s["name"] for s in talkers] == [f"t{i:02d}" for i in range(20)] and lay["sources"][-1]["name"] == "fan"
        kinds = list(lay["seat"].values())
        assert kinds.count("presenter") == int(d["presenter"]) and 3 <= kinds.count("row2") <= 6
        pos = np.array([s["pos"] for s in lay["sources"]])
        assert np.all(pos[:, :2] >= 0.4 - 1e-9) and np.all(pos[:, 0] <= lx - 0.4 + 1e-9)
        assert np.all(pos[:, 1] <= ly - 0.4 + 1e-9) and np.all(pos[:, 2] < lz)
        hz = np.hypot(*(pos[:20, None, :2] - pos[None, :20, :2]).transpose(2, 0, 1)) + 10 * np.eye(20)
        assert hz.min() >= 0.5, hz.min()
        dist = [s2.source_truth(p, lay["mics"])["dist_m"] for p in pos[:20]]
        assert 5.5 <= max(dist) <= 6.3 and min(dist) >= 1.3, (min(dist), max(dist))
        for s in talkers:
            assert not sm._inside(s["pos"], t) and s["directivity"]["p"] == sm.TALKER_P
            assert 1.1 <= s["pos"][2] <= 1.7 and (s["pos"][2] >= 1.5) == (lay["seat"][s["name"]] == "presenter")
            if lay["seat"][s["name"]] == "presenter":
                az = np.radians(s["directivity"]["azimuth_deg"])
                v = c[:2] - np.asarray(s["pos"][:2])
                assert np.degrees(np.arccos(np.dot([np.cos(az), np.sin(az)], v / np.linalg.norm(v)))) >= 159.9


def test_geometry_truth_and_rebuild():
    """Meta has the E7 parameters and per-talker truth; the RIRs rebuild bit for bit from the JSON alone."""
    root = _root()
    for name, d in DESIGNS.items():
        geom, rirs = s2.load_geometry("t", d["geom_id"], root=root)
        m = geom["meta"]
        assert geom["room"]["edited"] and m["table"]["on"] and m["mic_height_cm"] == d["mic_height_cm"]
        assert np.array_equal(sm.rebuild_rirs(geom), rirs), name
        tg = m["talker_geo"]
        assert len(tg) == 20 and m["n_table_image"] == 20 and m["n_floor_zeroed"] == 20
        assert m["max_dist_m"] >= 5.5 and set(m["n_bin"]) <= {"<2", "2-4", ">4"}
        dist, drr = [t["dist_m"] for t in tg.values()], [t["drr_db"] for t in tg.values()]
        assert spearmanr(dist, drr)[0] < -0.8
        ratio_t30 = np.array([t["t30_s"] for t in tg.values()]) / geom["room"]["rt60"]
        ratio_edt = np.array([t["edt_s"] for t in tg.values()]) / geom["room"]["rt60"]
        if d["tail"] == "synthetic":  # the tail decays at RT60 by construction
            assert np.all(np.abs(ratio_t30 - 1) < 0.1), ratio_t30
        else:  # specular full-order ISM in a flat room: early decay at RT60, slower late decay (BENCH2.md)
            assert np.all(np.abs(np.median(ratio_edt) - 1) < 0.25) and np.all((ratio_t30 > 1.2) & (ratio_t30 < 2.0))


def test_conversation_plan():
    rec = _recipe("full")
    m, T = rec["meta"], rec["n_samples"]
    assert 150 * FS <= T <= 260 * FS, T / FS
    assert min(m["n_turns"].values()) >= 3 and m["n_interrupts"] >= 3 and m["n_backchannels"] >= 3
    assert 0.07 <= m["overlap_ratio"] <= 0.18, m["overlap_ratio"]
    lv = m["level_db"]
    assert len(m["soft"]) in (2, 3) and all(52 <= lv[t] <= 55 for t in m["soft"])
    assert all(57 <= v <= 66 for t, v in lv.items() if t not in m["soft"])
    turns = m["turns"]
    assert min(e[2] for e in turns) >= sm.LEAD_S
    grid = np.zeros(int(T / FS * 100) + 1, int)
    for e in turns:
        grid[int(round(e[2] * 100)):int(round(e[3] * 100))] += 1
    assert grid.max() <= 2  # never three talkers at once
    corp = corpus2.load_corpus("tts")
    for g in rec["groups"][:20]:
        src = g["sources"][0]
        ev = src["signal"]["events"]
        assert len({u for u, _, _ in ev}) == len(ev)  # no talker repeats a sentence
        assert {corp.utt(u).speaker for u, _, _ in ev} == {m["speakers"][src["talker"]]}
    for g in rec["groups"][:20]:  # back-channels are 2-6 dB softer, turns vary by ±1 dB
        tk = g["sources"][0]["talker"]
        bc = [(u, e[2]) for e in turns if e[1] == "backchannel" and e[0] == tk for u in e[4]]
        for u, st, gdb in g["sources"][0]["signal"]["events"]:
            is_bc = any(u == v and abs(st / FS - t0) < 0.002 for v, t0 in bc)
            assert (-6 <= gdb <= -2) if is_bc else (-1 <= gdb <= 1), (tk, u, gdb)
    assert len(set(m["speakers"].values())) == 20


def test_render_groups_and_levels():
    sc = _scene("full", impair=False)
    names = [f"talker_{i:02d}" for i in range(20)]
    assert list(sc["comp2"]) == names + ["hvac", "fan", "sensor"]
    assert all(sc["groups"][g] == "talker" for g in names) and sc["groups"]["hvac"] == "hvac"
    assert set(sc["refs"]) == {f"t{i:02d}" for i in range(20)} and len({k for _, _, k in sc["segs"]}) == 20
    assert np.array_equal(sum(np.asarray(v) for v in sc["comp2"].values()), sc["mix"])
    assert all(isinstance(sc["comp2"][g], s2.Sparse) for g in names)
    m, T = sc["meta"], sc["mix"].shape[1]
    assert min(a for a, _, _ in sc["segs"]) >= 2 * FS and max(b for _, b, _ in sc["segs"]) <= T - 2 * FS
    for c in (0, 1):  # HVAC A-weighted level at each mic
        x = np.asarray(sc["comp2"]["hvac"][c], np.float64) / PA_TO_DIG
        assert abs(sm._level(x, True) - m["hvac_dba"]) < 0.5
    rirs = s2.load_geometry("t", DESIGNS["full"]["geom_id"], root=_root())[1]
    src = {s["name"]: k for k, s in enumerate(m["sources"])}
    rec, corp = _recipe("full"), corpus2.load_corpus("tts")
    for g in rec["groups"][:20]:  # early image of each talker's first utterance: level_db at 1 m, gain_db, RIR
        sp = g["sources"][0]
        k = src[sp["src"]]
        h = rirs[k, 0, :int(np.ceil(m["sources"][k]["direct_a"])) + s2.EARLY].astype(np.float64)
        u, st, gdb = sp["signal"]["events"][0]
        dry = corp.audio(u).astype(np.float64) * 10 ** (gdb / 20) * spl_to_pa(sp["level_db"])
        want = oaconvolve(dry, h)[:len(dry)] * PA_TO_DIG
        got = np.asarray(sc["refs"][sp["talker"]][st:st + len(dry)], np.float64)
        assert np.sqrt(np.mean((got - want) ** 2) / np.mean(want ** 2)) < 0.01, sp["talker"]


def test_determinism():
    """Same design, other root: identical geometry, RIRs and recipe; re-rendering is bit-identical (E9)."""
    other = tempfile.mkdtemp(prefix="meeting_test2_")
    try:
        d = DESIGNS["short"]
        sm.build_room(d, other)
        for f in (f"{d['geom_id']}.geom.json", f"{d['name']}.tts.recipe.json"):
            assert json.load(open(os.path.join(_root(), "t", f))) == json.load(open(os.path.join(other, "t", f)))
        a = s2.load_geometry("t", d["geom_id"], root=_root())[1]
        b = s2.load_geometry("t", d["geom_id"], root=other)[1]
        assert np.array_equal(a, b)
    finally:
        shutil.rmtree(other, True)
    x = _scene("short")
    s2.clear_cache()
    y = _scene("short")
    assert x["mix"] is not y["mix"] and np.array_equal(x["mix"], y["mix"]) and x["segs"] == y["segs"]


def test_corpus_switch():
    """A LibriSpeech-style manifest drops in: same geometry, noise, levels and impairments; new speakers."""
    c = corpus2.load_corpus("tts")
    root = _root()
    spk = c.speakers_for("talker")
    man = {"name": "fake20", "source": "libri",
           "speakers": {f"s_{v}": {"pool": "target", "split": "dev"} for v in spk},
           "utterances": [{"id": f"s_{u.id}", "speaker": f"s_{u.speaker}", "text": u.text, "path": u.path,
                           "duration": u.duration} for v in spk for u in c.select(speaker=v)]}
    path = os.path.join(root, "fake20.json")
    json.dump(man, open(path, "w"))
    corpus2.register("fake20", path)
    sc = _scene("short", corpus="fake20")
    a, b = _recipe("short"), _recipe("short", "fake20")
    keep = lambda r: [g for g in r["groups"] if g["role"] != "talker"]  # noqa: E731
    assert keep(a) == keep(b) and a["impair"] == b["impair"] and a["seed"] == b["seed"]
    assert a["meta"]["level_db"] == b["meta"]["level_db"]
    assert set(b["meta"]["speakers"].values()) <= set(man["speakers"]) and sc["meta"]["corpus"] == "fake20"
    del man["speakers"][f"s_{spk[0]}"]
    man["utterances"] = [u for u in man["utterances"] if u["speaker"] != f"s_{spk[0]}"]
    json.dump(man, open(path, "w"))
    corpus2.register("fake20", path)
    geom = s2.load_geometry("t", DESIGNS["short"]["geom_id"], root=root)[0]
    try:
        sm.make_recipe(geom, corpus2.load_corpus("fake20"), 1)
    except ValueError:
        return
    raise AssertionError("a corpus with 19 talkers was accepted")


def test_render_time():
    s2.clear_cache()
    t0 = time.time()
    sc = _scene("full")
    dt = time.time() - t0
    print(f"    full meeting: {sc['meta']['duration_s']:.0f} s of audio rendered in {dt:.2f} s")
    assert dt < 8.0


if __name__ == "__main__":
    fails = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.2f} s)")
        except Exception as e:  # report every failure, then exit non-zero
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
