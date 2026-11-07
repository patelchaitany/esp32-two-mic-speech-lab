"""Tests for the v2 bus scene generator (lab.scene_bus). Run: .venv/bin/python -m tests.test_scene_bus

Layout tests sample every dev/test/smoke design row without RIRs. Scene tests build the smoke geometries
(and one small full-order ISM room) in a temporary root, so scenes2/ is not touched.
"""
import atexit
import collections
import copy
import itertools
import json
import os
import shutil
import tempfile
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

from lab import corpus2
from lab import scene2 as s2
from lab import scene_bus as sb
from lab.common import C, FS

_ROOT = []
ROWS = sb.set_design("dev") + sb.set_design("test") + sb.set_design("smoke")


def _root():
    """Temporary scenes root holding the smoke set (built once per process)."""
    if not _ROOT:
        _ROOT.append(tempfile.mkdtemp(prefix="scene_bus_test_"))
        atexit.register(shutil.rmtree, _ROOT[0], True)
        for row in sb.set_design("smoke"):
            sb.build_geometry(row, root=_ROOT[0])
    return _ROOT[0]


def _scene(i, impair=True, corpus="tts"):
    return s2.load_scene(f"smoke/{sb.scene_name(i, 0)}", corpus=corpus, impair=impair, root=_root())


def _layouts():
    if not hasattr(_layouts, "cache"):
        _layouts.cache = [(r, sb.sample_layout(r)) for r in ROWS]
    return _layouts.cache


def test_design_balanced():
    for name in ("dev", "test"):
        d = sb.set_design(name)
        assert len(d) == 30 and len({r["seed"] for r in d}) == 30
        for st in sb.STRATA:
            g = [r for r in d if r["stratum"] == st]
            assert len(g) == 6
            assert sorted((r["spacing"], r["tail"]) for r in g) == sorted((s, t) for s in sb.SPACINGS for t in sb.TAILS)
            subs = sorted(int(np.floor((r["rt60"] - 0.15) / 0.075)) for r in g)
            assert subs == list(range(6)), (name, st, subs)  # every RT60 sub-bin once per stratum
            assert all(0.15 <= r["rt60"] <= 0.6 for r in g)
            assert sorted(r["tail"] for r in g if r["handling"]) == ["ism", "synthetic"]
            assert sum(r["skew_us"] == 31.25 for r in g) == 3
            assert collections.Counter(r["road_model"] for r in g) == {"sinc": 3, "j0": 3}
            assert all({r["road_model"] for r in g if r["spacing"] == s} == {"sinc", "j0"} for s in sb.SPACINGS)
            if st in sb.FOLD_RANGE:
                assert sorted(r["dist_bins"] for r in g) == sorted(sb.DIST_DESIGN)
            sweep = [r for r in g if r["sweep"]]
            assert len(sweep) == (2 if name == "test" else 0) and sorted(r["tail"] for r in sweep) == \
                (["ism", "synthetic"] if name == "test" else [])
            assert all(tuple(r["snrs"]) == (sb.SWEEP_SNRS if r["sweep"] else sb.SNRS) for r in g)
    assert not {r["seed"] for r in sb.set_design("dev")} & {r["seed"] for r in sb.set_design("test")}
    assert sb.scene_name(7, -5) == "bus_g07_snr-5" and sb.scene_name(12, 0) == "bus_g12_snr+0"
    assert sb.set_design("dev") == sb.set_design("dev")  # deterministic


def test_layouts_meet_strata():
    for row, lay in _layouts():
        m, dims = lay["meta"], lay["dims"]
        tag = row["geom_id"]
        assert all(abs(v / n - 1) <= 0.05 for v, n in zip(dims, (11.5, 2.5, 2.3))), tag
        assert m["angle_class"] == ("broadside" if row["stratum"] == "lap_flat" else row["stratum"]), tag
        assert 0.3 <= m["dist_A_m"] <= 1.0 and 0.3 <= m["dist_B_m"] <= 1.0, tag
        if row["dist_bins"] is not None:
            assert (m["dist_bin_A"], m["dist_bin_B"]) == tuple(sb.BIN_NAMES[b] for b in row["dist_bins"]), tag
        mics = np.array(lay["mics"])
        el = np.degrees(np.arcsin((mics[0, 2] - mics[1, 2]) / np.linalg.norm(mics[0] - mics[1])))
        assert abs(np.linalg.norm(mics[0] - mics[1]) - row["spacing"]) < 1e-5, tag
        if row["stratum"] == "lap_flat":
            assert abs(el) <= 10.01 and 0.5 <= m["device_height_m"] <= 0.78, tag
            assert max(abs(m["tau_A_us"]), abs(m["tau_B_us"])) < 0.3 * row["spacing"] / C * 1e6, tag
            assert m["inseparable"], tag
        else:
            assert abs(el) < 1e-3 and 0.85 <= m["device_height_m"] <= 1.2, tag
        if row["stratum"] == "side_by_side":
            assert 20 <= m["partner_angle_deg"] <= 40, tag
            assert (m["theta_A_deg"] - 90) * (m["theta_B_deg"] - 90) > 0, tag
        elif row["stratum"] != "lap_flat":
            assert m["partner_angle_deg"] >= 110, tag  # facing each other across the device
        assert 16 <= m["n_passengers"] <= 25 and m["n_conv_pairs"] in (1, 2), tag
        assert 1.5 <= m["cone_dist_m"] <= 2.0 and abs(m["cone_dtheta_deg"]) <= 2.0, tag
        src = {s["name"]: s for s in lay["sources"]}
        tr = {k: s2.source_truth(s["pos"], lay["mics"]) for k, s in src.items()}
        near = [tr[k]["dist_m"] for k in m["people"]["near"]]
        conv = [tr[k]["dist_m"] for pr in m["people"]["conv"] for k in pr]
        far = [tr[k]["dist_m"] for k in m["people"]["far"]]
        assert min(near) >= 0.6 and min(near) <= 0.85 and max(near) <= 1.4, tag
        assert all(0.6 <= r <= 1.2 for r in conv), tag
        assert min(far) >= max(1.2, max(near + conv)), tag
        assert len(near) + len(conv) + len(far) + 1 == m["n_passengers"], tag
        people = ["pA", "pB", "cone"] + m["people"]["near"] + m["people"]["far"] + \
            [k for pr in m["people"]["conv"] for k in pr]
        pos = np.array([src[k]["pos"] for k in people])
        dmin = min(np.linalg.norm(pos[i] - pos[j]) for i in range(len(pos)) for j in range(i))
        assert dmin >= sb.MIN_SEP - 1e-5, (tag, dmin)
        allpos = np.array([s["pos"] for s in lay["sources"]])
        assert (allpos > 0.05).all() and (allpos < np.array(dims) - 0.05).all(), tag
        for a, b in m["people"]["conv"]:  # pairs face each other, everybody else has some facing
            assert src[a]["directivity"] == s2.facing(src[a]["pos"], src[b]["pos"]), tag
        assert src["pA"]["directivity"] == s2.facing(src["pA"]["pos"], src["pB"]["pos"]), tag
        assert all(src[k]["directivity"] is not None for k in people), tag


def test_layout_deterministic():
    row = sb.set_design("dev")[3]
    a, b = sb.sample_layout(row), sb.sample_layout(copy.deepcopy(row))
    assert json.dumps(a) == json.dumps(b)
    c = sb.sample_layout(dict(row, seed=row["seed"] + 1000))
    assert json.dumps(a["sources"]) != json.dumps(c["sources"])


def test_scene_structure():
    for i, row in enumerate(sb.set_design("smoke")):
        sc = _scene(i)
        m = sc["meta"]
        groups = list(sc["groups"])
        want = ["partner_A", "partner_B", "near_1", "near_2", "near_3", "cone", "conv_1", "babble_far",
                "engine_road", "transient"] + (["handling"] if row["handling"] else []) + ["sensor"]
        assert groups == want, groups
        roles = collections.Counter(sc["groups"].values())
        assert roles["partner"] == 2 and roles["passenger_near"] == 3 and roles["conv_pair"] == 1
        assert sc["talker_group"] == {"pA": "partner_A", "pB": "partner_B", "n1": "near_1", "n2": "near_2",
                                      "n3": "near_3", "cone": "cone", "c1a": "conv_1", "c1b": "conv_1"}
        assert set(sc["refs"]) == set(sc["talker_group"])
        assert np.array_equal(sum(np.asarray(v) for v in sc["comp2"].values()), sc["mix"])
        assert sc["mix"].shape == (2, int(row["seconds"] * FS)) and not m["clipped"] and m["peak"] < 0.5
        assert m["stratum"] == row["stratum"] and m["snr_db"] == 0.0 and abs(m["spacing_m"] - row["spacing"]) < 1e-5
        assert m["impair"]["skew_us"] == row["skew_us"] and m["room"]["tail"] == "synthetic"
        assert min(a for a, _, tk in sc["segs"] if tk in ("pA", "pB")) >= 1.5 * FS
        assert m["rules"][0]["noise"] == ["near_1", "near_2", "near_3", "cone", "conv_1", "babble_far", "engine_road"]
        for tk, t in m["talkers"].items():
            assert sb.LOMBARD[0] <= t["level_db"] <= sb.LOMBARD[1], tk
        tp = {tk: m["talkers"][tk]["speaker"] for tk in ("pA", "pB")}
        assert set(tp.values()) <= set(sb.partner_speakers(corpus2.load_corpus("tts"), "dev"))


def _inband_power(x, mask):
    return np.mean(sosfiltfilt(butter(4, [150, 7000], "band", fs=FS, output="sos"), x)[mask] ** 2)


def test_snr_exact_and_scaling():
    sc = _scene(0, impair=False)
    rule = sc["meta"]["rules"][0]
    mask = np.zeros(sc["mix"].shape[1], bool)
    for a, b, tk in sc["segs"]:
        if tk in ("pA", "pB"):
            mask[a:b] = True
    pt = _inband_power(sum(np.asarray(sc["comp2"][g][0], np.float64) for g in rule["target"]), mask)
    pn = _inband_power(sum(np.asarray(sc["comp2"][g][0], np.float64) for g in rule["noise"]), mask)
    assert abs(10 * np.log10(pt / pn) - 0.0) < 0.01 and abs(rule["realized_snr_db"]) < 0.01
    assert abs(_scene(0)["meta"]["rules"][0]["realized_snr_db"]) < 0.3  # recorded mics (E8 chain)
    # another SNR of the same geometry: same signals, only the rule's noise groups move
    root, row = _root(), sb.set_design("smoke")[0]
    geom, _ = s2.load_geometry("smoke", row["geom_id"], root=root)
    sid = s2.add_scene("smoke", "bus_g00_snr-7", geom, {"seed": row["seed"], "snr_db": -7.0,
                                                        "seconds": row["seconds"]}, root=root)
    lo = s2.load_scene(sid, impair=False, root=root)
    for g in sc["comp2"]:
        a, b = np.asarray(sc["comp2"][g], np.float64), np.asarray(lo["comp2"][g], np.float64)
        if g in rule["noise"]:
            assert np.allclose(b, a * 10 ** (7 / 20), rtol=1e-4, atol=1e-9), g
        else:
            assert np.array_equal(a, b), g
    assert abs(lo["meta"]["rules"][0]["realized_snr_db"] + 7.0) < 0.01
    os.remove(os.path.join(root, "smoke", "bus_g00_snr-7.tts.recipe.json"))


def test_transients_handling_levels():
    for i, row in enumerate(sb.set_design("smoke")):
        sc = _scene(i, impair=False)
        tr = np.asarray(sc["comp2"]["transient"][0], np.float64)
        for e in sc["meta"]["recipe_meta"]["transients"]:
            seg = tr[e["start"]:e["start"] + e["length"] + FS // 2]
            lf = sb._lfmax(seg)
            if e["capped"]:  # the cap holds the level below the target (up to the calibration error)
                assert lf <= e["lfmax_target_db"] + 3.0, (e, lf)
            else:
                assert abs(lf - e["lfmax_target_db"]) < 4.0, (e, lf)
            assert 85 <= e["lfmax_target_db"] <= 97
        if row["handling"]:
            hd = np.asarray(sc["comp2"]["handling"], np.float64)
            on = np.abs(hd[0]) > 0
            assert abs(np.corrcoef(hd[0][on], hd[1][on])[0, 1]) < 0.1
            f = np.fft.rfftfreq(hd.shape[1], 1 / FS)
            P = np.abs(np.fft.rfft(hd[0])) ** 2
            assert P[f < 500].sum() / P.sum() > 0.98
        else:
            assert "handling" not in sc["comp2"]


def test_turn_taking():
    """Partners alternate (runs of at most 2 utterances), start after the 1.5 s lead-in, overlap sometimes,
    and get back-channel stand-ins; checked over many recipes without rendering."""
    corp = corpus2.load_corpus("tts")
    geom, _ = s2.load_geometry("smoke", sb.set_design("smoke")[0]["geom_id"], root=_root())
    geom = dict(geom, set="dev")
    overlaps = bc = 0
    for seed in range(20):
        rec = sb.make_recipe(geom, corp, seed=seed, snr_db=0.0, seconds=24.0)
        ev = []
        for g in rec["groups"][:2]:
            tk = g["sources"][0]["talker"]
            for u, st, gain in g["sources"][0]["signal"]["events"]:
                ev.append((st, st + int(round(corp.utt(u).duration * FS)), tk, gain))
        ev.sort()
        assert ev[0][0] >= 1.5 * FS and ev[-1][1] <= 24 * FS
        main = [e for e in ev if e[3] > -4]
        bc += len(ev) - len(main)
        runs = [len(list(g)) for _, g in itertools.groupby(e[2] for e in main)]
        assert len(runs) >= 3 and max(runs) <= 2, runs
        overlaps += any(b[0] < a[1] for a, b in zip(main, main[1:]) if a[2] != b[2])
        for tk in ("pA", "pB"):  # nobody overlaps with themselves
            own = [e for e in ev if e[2] == tk]
            assert all(b[0] >= a[1] for a, b in zip(own, own[1:])), tk
    assert overlaps >= 5 and bc >= 5, (overlaps, bc)


def test_corpus_switch_and_partner_split():
    """A LibriSpeech-style manifest with dev/test partner splits: only speech changes, splits are honored."""
    c = corpus2.load_corpus("tts")
    spk = {"d1": ("Daniel", "dev"), "d2": ("Karen", "dev"), "t1": ("Moira", "test"), "t2": ("Rishi", "test")}
    spk.update({f"b{k}": (v, "dev") for k, v in enumerate(c.speakers_for("babble")[:10])})
    utts, speakers = [], {}
    for new, (old, split) in spk.items():
        pool = "babble" if new.startswith("b") else "target"
        speakers[new] = {"pool": pool, "split": split, "gender": c.speakers[old]["gender"]}
        for u in c.select(speaker=old):
            utts.append({"id": f"{new}-{u.id}", "speaker": new, "text": u.text, "path": u.path,
                         "duration": u.duration})
    root = _root()
    path = os.path.join(root, "fake_bus_manifest.json")
    with open(path, "w") as f:
        json.dump({"name": "fakebus", "source": "libri", "speakers": speakers, "utterances": utts}, f)
    corpus2.register("fakebus", path)
    fake = corpus2.load_corpus("fakebus")
    assert sb.partner_speakers(fake, "dev") == ["d1", "d2"] and sb.partner_speakers(fake, "test") == ["t1", "t2"]
    tts = corpus2.load_corpus("tts")
    assert not set(sb.partner_speakers(tts, "dev")) & set(sb.partner_speakers(tts, "test"))
    assert len(sb.partner_speakers(tts, "dev")) == len(sb.partner_speakers(tts, "test")) == 4
    sid = f"smoke/{sb.scene_name(2, 0)}"
    a, b = (s2.ensure_recipe(sid, k, root=root) for k in ("tts", "fakebus"))
    keep = ("engine_road", "transient", "handling")
    assert [g for g in a["groups"] if g["name"] in keep] == [g for g in b["groups"] if g["name"] in keep]
    assert a["impair"] == b["impair"] and a["seed"] == b["seed"] and a["rules"] == b["rules"]
    assert a["meta"]["lombard_db"] == b["meta"]["lombard_db"]
    sc = s2.load_scene(sid, corpus="fakebus", root=root)
    who = {t: v["speaker"] for t, v in sc["meta"]["talkers"].items()}
    assert {who["pA"], who["pB"]} == {"d1", "d2"}
    assert all(v.startswith("b") for t, v in who.items() if t not in ("pA", "pB"))
    assert np.array_equal(sum(np.asarray(v) for v in sc["comp2"].values()), sc["mix"])


def test_recorded_noise_hook():
    """E6(a): a field source from 2-channel recordings (two corpus files stand in for a recording)."""
    c = corpus2.load_corpus("tts")
    files = [u.path for u in c.select(speaker="Daniel")[:2]]
    T = 3 * FS
    x1 = sb.recorded_noise(T, FS, np.random.default_rng(5), files)
    x2 = sb.recorded_noise(T, FS, np.random.default_rng(5), files)
    assert x1.shape == (2, T) and np.array_equal(x1, x2) and abs(np.mean(x1 ** 2) - 1) < 1e-9
    root, row = _root(), sb.set_design("smoke")[1]
    geom, _ = s2.load_geometry("smoke", row["geom_id"], root=root)
    rec = {"files": files, "level_db": 70.0, "replace": True}
    sid = s2.add_scene("smoke", "bus_g01_snr+0_rec", geom, {"seed": row["seed"], "snr_db": 0.0,
                                                           "seconds": row["seconds"], "recorded": rec}, root=root)
    sc = s2.load_scene(sid, impair=False, root=root)
    road = s2.ensure_recipe(sid, root=root)["groups"]
    road = next(g for g in road if g["name"] == "engine_road")
    assert [s["type"] for s in road["sources"]] == ["field"]
    assert abs(sc["meta"]["rules"][0]["realized_snr_db"]) < 0.01
    os.remove(os.path.join(root, "smoke", "bus_g01_snr+0_rec.tts.recipe.json"))


def test_acoustics_and_ism_tail():
    """Stored acoustics match the stored RIRs; a full-order ISM room builds, with the measured decay
    (EDT near nominal, T20 longer: double-slope decay of the specular cabin) recorded in meta."""
    root = _root()
    row = sb.set_design("smoke")[0]
    geom, rirs = s2.load_geometry("smoke", row["geom_id"], root=root)
    again = sb.measure_acoustics(rirs, geom["sources"], geom["mics"])
    assert again == geom["meta"]["acoustics"]
    ism = dict(sb.set_design("smoke")[2], tail="ism", rt60=0.2, geom_id="bus_smoke_ism", idx=50, snrs=(0,))
    t0 = time.time()
    log = sb.build_geometry(ism, root=root)
    g, r = s2.load_geometry("smoke", "bus_smoke_ism", root=root)
    ac = g["meta"]["acoustics"]
    assert g["room"]["tail"] == "ism" and g["room"]["max_order"] == log["order"] > 18
    assert 0.6 < ac["edt_s"] / 0.2 < 1.6 and 1.0 < ac["t20_s"] / 0.2 < 3.0, ac
    print(f"    ISM room RT60 0.2 s: order {log['order']}, EDT {ac['edt_s']:.3f} T20 {ac['t20_s']:.3f} s, "
          f"built in {time.time() - t0:.1f} s")
    sc = s2.load_scene("smoke/bus_g50_snr+0", root=root)
    assert np.array_equal(sum(np.asarray(v) for v in sc["comp2"].values()), sc["mix"])


def test_render_time():
    s2.clear_cache()
    t0 = time.time()
    sc = _scene(1)
    dt = time.time() - t0
    n = len(sc["meta"]["sources"])
    print(f"    smoke render {sc['meta']['duration_s']:.0f} s audio, {n} sources in {dt:.2f} s")
    assert dt < 3.0


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
