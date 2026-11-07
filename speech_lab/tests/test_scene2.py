"""Tests for the v2 scene core (lab.scene2, lab.corpus2). Run: .venv/bin/python -m tests.test_scene2

Builds the demo geometry (bus-like room, 3 sources) in a temporary root, so scenes2/ is not touched.
"""
import atexit
import json
import os
import pickle
import shutil
import tempfile
import time

import numpy as np
from scipy.signal import butter, csd, freqz, welch

from lab import corpus2
from lab import scene2 as s2
from lab.common import FS, PA_TO_DIG, active_rms

_ROOT = []


def _root():
    """Temporary scenes root holding the demo scene (built once per process)."""
    if not _ROOT:
        _ROOT.append(tempfile.mkdtemp(prefix="scene2_test_"))
        atexit.register(shutil.rmtree, _ROOT[0], True)
        s2.build_demo(root=_ROOT[0])
    return _ROOT[0]


def _scene(impair=True, corpus="tts"):
    return s2.load_scene("demo/bus3_snr+0", corpus=corpus, impair=impair, root=_root())


def test_corpus_tts():
    c = corpus2.load_corpus("tts")
    partners, talkers, babble = (c.speakers_for(r) for r in ("partner", "talker", "babble"))
    assert len(partners) == 8 and len(talkers) == 20 and len(babble) >= 25
    assert set(partners) <= set(talkers) and not set(talkers) & set(babble)
    names = set(c.speakers)
    assert not names & {"Fred", "Kathy", "Ralph", "Junior"}
    assert not any("(English (" in v for v in names)
    for u in c.utts:
        assert u.duration > 0.3 and u.pool == c.speakers[u.speaker]["pool"] and u.source == "tts"
    x = c.audio(c.select(speaker="Samantha", style="dialogue")[0].id)
    assert x.dtype == np.float32 and not x.flags.writeable


def test_rir_seeding_bit_identical():
    dims, mics = [11.5, 2.5, 2.3], s2.place_array([5.0, 1.2, 1.0], 0.04, 10.0)
    srcs = [[5.4, 1.3, 1.2], [3.0, 0.6, 1.3]]
    d = [s2.facing(srcs[0], srcs[1]), None]
    r1, st1, p1 = s2.compute_rirs(dims, 0.4, srcs, mics, seed=11, directivity=d)
    np.random.seed(123)  # unrelated global state must not matter
    s2.pra.random.seed(999)
    r2, st2, _ = s2.compute_rirs(dims, 0.4, srcs, mics, seed=11, directivity=d)
    r3, _, _ = s2.compute_rirs(dims, 0.4, srcs, mics, seed=12, directivity=d)
    assert np.array_equal(r1, r2) and np.array_equal(st1, st2)
    assert not np.array_equal(r1, r3)
    assert r1.dtype == np.float32 and r1.shape == (2, 2, p1["n_rir"])
    # the direct path comes from the true position despite the randomized-ISM jitter
    for k, s in enumerate(srcs):
        d0 = s2.source_truth(s, mics)["direct_a"]
        assert abs(int(np.argmax(np.abs(r1[k, 0, :int(d0) + 30]))) - d0) <= 1


def test_geometry_store_roundtrip():
    root = _root()
    sc = _scene()
    geom, rirs = s2.load_geometry("demo", sc["meta"]["geom_id"], root=root)
    srcs = [s["pos"] for s in geom["sources"]]
    r, _, _ = s2.compute_rirs(geom["room"]["dims"], geom["room"]["rt60"], srcs, geom["mics"], geom["seed"],
                              directivity=[s["directivity"] for s in geom["sources"]])
    assert np.array_equal(r, rirs), "stored geometry (with re-synthesized tail) differs from a fresh compute"
    assert not rirs.flags.writeable


def test_sum_comp2_equals_mix():
    for impair in (True, False):
        sc = _scene(impair)
        total = sum(np.asarray(v) for v in sc["comp2"].values())
        assert total.dtype == np.float32 and total.shape == sc["mix"].shape
        assert np.abs(total - sc["mix"]).max() <= 1e-6
        assert np.array_equal(total, sc["mix"])  # canonical (dict) order is bit-exact
        assert list(sc["comp2"])[-1] == "sensor" and sc["groups"]["sensor"] == "sensor"
        assert set(sc["comp2"]) == set(sc["groups"])


def test_render_determinism():
    a = _scene()
    s2.clear_cache()
    b = _scene()
    assert a["mix"] is not b["mix"]
    assert np.array_equal(a["mix"], b["mix"])
    for g in a["comp2"]:
        assert np.array_equal(np.asarray(a["comp2"][g]), np.asarray(b["comp2"][g]))
    for k in a["refs"]:
        assert np.array_equal(np.asarray(a["refs"][k]), np.asarray(b["refs"][k]))
    assert a["segs"] == b["segs"] and a["meta"] == b["meta"]


def test_impairment_truth():
    raw, imp = _scene(False), _scene(True)
    truth = imp["meta"]["impair"]
    assert truth["applied"] and truth["skew_us"] == 31.25 and not raw["meta"]["impair"]["applied"]
    assert all(-1 <= g <= 1 for g in truth["gain_db"]) and all(50 <= f <= 80 for f in truth["hp_hz"])
    # transfer function recorded/acoustic per mic (Welch, so signal-end effects do not bias it)
    for g in ("partner_A", "engine_road", "handling", "sensor"):
        x, y = np.asarray(raw["comp2"][g], np.float64), np.asarray(imp["comp2"][g], np.float64)
        for m in range(2):
            f, pxx = welch(x[m], FS, nperseg=4096)
            _, pxy = csd(x[m], y[m], FS, nperseg=4096)
            b, a = butter(1, truth["hp_hz"][m], "highpass", fs=FS)
            H = freqz(b, a, worN=f, fs=FS)[1] * 10 ** (truth["gain_db"][m] / 20)
            if m == 1:
                H = H * np.exp(-2j * np.pi * f * truth["skew_us"] * 1e-6)
            band = (f > 40) & (f < 7000)
            strong = band & (pxx > np.percentile(pxx[band], 50))
            err = np.abs(pxy[strong] / pxx[strong] - H[strong]) / np.abs(H[strong])
            assert np.median(err) < 2e-3 and np.percentile(err, 90) < 1e-2, (g, m, np.median(err))
            if m == 1:  # without the skew term the phase would be off by 2πf·31.25 µs (11.25° at 1 kHz)
                no_skew = H * np.exp(2j * np.pi * f * truth["skew_us"] * 1e-6)
                assert np.median(np.abs(pxy[strong] / pxx[strong] - no_skew[strong])) > 10 * np.median(err)


def test_refs_aligned_with_direct_path():
    sc = _scene()
    rec = json.load(open(os.path.join(_root(), "demo", "bus3_snr+0.tts.recipe.json")))
    c = corpus2.load_corpus("tts")
    T = sc["mix"].shape[1]
    nf = 1 << int(np.ceil(np.log2(2 * T)))
    for tk, g in sc["talker_group"].items():
        ref = np.asarray(sc["refs"][tk], np.float64)
        comp = np.asarray(sc["comp2"][g][0], np.float64)
        cc = np.fft.irfft(np.fft.rfft(comp, nf) * np.conj(np.fft.rfft(ref, nf)), nf)
        lag = int(np.argmax(np.concatenate([cc[-50:], cc[:51]]))) - 50
        assert abs(lag) <= 1, (tk, lag)
        # the ref sits at the direct-path delay of the dry speech, and segs say where
        dry = np.zeros(T)
        src = next(s for grp in rec["groups"] for s in grp["sources"] if s.get("talker") == tk)
        for u, st, gdb in src["signal"]["events"]:
            x = c.audio(u)[:T - st]
            dry[st:st + len(x)] += x
        cc = np.fft.irfft(np.fft.rfft(ref, nf) * np.conj(np.fft.rfft(dry, nf)), nf)
        direct = next(s["direct_a"] for s in sc["meta"]["sources"] if s["name"] == src["src"])
        assert abs(int(np.argmax(cc[:400])) - direct) <= 1.5, (tk, int(np.argmax(cc[:400])), direct)
        mine = [(a, b) for a, b, k in sc["segs"] if k == tk]
        assert mine and all(a - st == round(direct) for (a, _), (_, st, _) in zip(mine, src["signal"]["events"]))
        inside = np.zeros(T, bool)
        for a, b in mine:
            inside[a:b] = True
        assert np.mean(ref[inside] ** 2) > 1e3 * np.mean(ref[~inside] ** 2)


def test_levels_and_rules():
    sc = _scene(impair=False)
    m = sc["meta"]
    assert m["rules"][0]["snr_db"] == 0.0 and abs(m["rules"][0]["realized_snr_db"]) < 0.01
    # sensor self-noise is 33 dBA SPL-equivalent per mic
    x = np.asarray(sc["comp2"]["sensor"], np.float64) / PA_TO_DIG
    for m_ in range(2):
        lvl = 10 * np.log10(np.mean(x[m_] ** 2) * s2._a_ratio(x[m_]) / (20e-6) ** 2)
        assert abs(lvl - 33.0) < 0.2, lvl
    # a talker's early image has the level set in the recipe: level_db at 1 m, 1/r spreading
    t = m["talkers"]["pA"]
    ref = np.asarray(sc["refs"]["pA"], np.float64) / PA_TO_DIG
    a, b, _ = next(q for q in sc["segs"] if q[2] == "pA")
    lvl = 20 * np.log10(active_rms(ref[a:b]) / 20e-6)
    expect = t["level_db"] - 20 * np.log10(t["dist_m"])
    assert expect - 1 < lvl < expect + 4, (lvl, expect)  # + early reflections, directivity on-axis


def test_a_weighted_and_diffuse_noise():
    """level_db with weighting 'A' is the A-weighted level; diffuse noise has the model coherence."""
    rng = np.random.default_rng(0)
    f = np.fft.rfftfreq(1024, 1 / FS)
    models = {"sinc": np.sinc(2 * f * 0.04 / s2.C), "j0": s2.j0(2 * np.pi * f * 0.04 / s2.C)}
    for model, g in models.items():
        src = {"type": "diffuse", "level_db": 40.0, "weighting": "A",
               "signal": {"kind": "noise", "shape": [[30, 0], [1000, -15], [8000, -35]]}}
        blocks = s2._dry(src["signal"], 20 * FS, rng, None, k=2, spacing=0.04, model=model)
        x = blocks[0][1] * s2._level(src, blocks)
        for m in range(2):
            lvl = 10 * np.log10(np.mean(x[m] ** 2) * s2._a_ratio(x[m]) / (20e-6) ** 2)
            assert abs(lvl - 40.0) < 0.3, (model, m, lvl)
        X0, X1 = (np.fft.rfft(v[:len(v) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024), axis=1) for v in x)
        gam = np.real(np.mean(X0 * np.conj(X1), 0)) / np.sqrt(np.mean(abs(X0) ** 2, 0) * np.mean(abs(X1) ** 2, 0))
        for lo, hi in ((200, 1000), (1000, 3000), (3000, 7000)):  # unbiased; spread is estimation noise
            b = (f > lo) & (f < hi)
            assert np.median(np.abs(gam - g)[b]) < 0.03 and abs(np.mean((gam - g)[b])) < 0.01, (model, lo)


def test_signed_mix_and_hagerman():
    sc = _scene()
    assert np.array_equal(s2.signed_mix(sc), sc["mix"])
    assert np.array_equal(s2.signed_mix(sc, flip=list(sc["comp2"])), -sc["mix"])
    others = [g for g in sc["comp2"] if g not in ("partner_A", "partner_B")]
    A, B = (np.asarray(sc["comp2"][g], np.float64) for g in ("partner_A", "partner_B"))
    N = sum(np.asarray(sc["comp2"][g], np.float64) for g in others)
    y1 = s2.signed_mix(sc).astype(np.float64)
    y2 = s2.signed_mix(sc, flip=["partner_B"] + others).astype(np.float64)
    y3 = s2.signed_mix(sc, flip=["partner_A"] + others).astype(np.float64)
    y4 = s2.signed_mix(sc, flip=["partner_A", "partner_B"]).astype(np.float64)
    for est, ref in (((y1 + y2 - y3 - y4) / 4, A), ((y1 - y2 + y3 - y4) / 4, B), ((y1 - y2 - y3 + y4) / 4, N)):
        assert np.abs(est - ref).max() < 1e-6
    assert np.array_equal(s2.signed_mix(sc, drop=others), (sc["comp2"]["partner_A"] + sc["comp2"]["partner_B"]))


def test_sparse_behaves_like_array():
    sc = _scene()
    sparse = [v for v in list(sc["comp2"].values()) + list(sc["refs"].values()) if isinstance(v, s2.Sparse)]
    assert sparse, "demo should contain at least one Sparse component"
    for v in sparse:
        d = np.asarray(v)
        assert d.shape == v.shape and d.dtype == np.float32
        if v.ndim == 2:
            assert np.array_equal(v[1][100:5000], d[1, 100:5000]) and np.array_equal(v[1, 100:5000], d[1, 100:5000])
            assert np.array_equal(v[:, 7:9], d[:, 7:9]) and len(v) == 2
        else:
            assert np.array_equal(v[3000:9000], d[3000:9000])
        assert np.array_equal(v + 0, d) and np.isclose(v.sum(), d.sum()) and np.array_equal(-v, -d)
        assert np.array_equal(np.asarray(v, np.float64), d.astype(np.float64))
        assert np.array_equal(np.asarray(pickle.loads(pickle.dumps(v))), d)


def test_list_and_card():
    root = _root()
    sid = "demo/bus3_snr+0"
    assert s2.list_scenes("demo", root=root) == [sid]
    assert s2.list_scenes("demo", scenario="bus", stratum="aligned", snr_db=(0.0, 5.0), root=root) == [sid]
    assert s2.list_scenes("demo", stratum="broadside", root=root) == []
    assert s2.list_scenes("demo", scenario="meeting", root=root) == []
    assert s2.list_scenes("demo", rt60=lambda v: v < 0.5, root=root) == [sid]
    assert s2.list_scenes("nope", root=root) == []
    card = s2.scene_card(sid, root=root)
    assert card["spacing_m"] == 0.04 and card["tail"] == "synthetic" and "tts" in card["corpora"]


def test_corpus_switch_same_geometry():
    """A LibriSpeech-style manifest drops in; the scene re-renders on the same geometry with the new corpus."""
    c = corpus2.load_corpus("tts")
    root = _root()
    spk = {"spk1": "Daniel", "spk2": "Karen", "bab1": "Anna"}
    utts = []
    for new, old in spk.items():
        for u in c.select(speaker=old)[:6]:
            utts.append({"id": f"{new}-{u.id}", "speaker": new, "text": u.text, "path": u.path, "duration": u.duration})
    man = {"name": "fake", "source": "libri",
           "speakers": {"spk1": {"pool": "target", "split": "dev", "gender": "M"},
                        "spk2": {"pool": "target", "split": "test", "gender": "F"},
                        "bab1": {"pool": "babble", "split": "dev"}},
           "utterances": utts}
    path = os.path.join(root, "fake_manifest.json")
    json.dump(man, open(path, "w"))
    corpus2.register("fake", path)
    fake = corpus2.load_corpus("fake")
    assert fake.speakers_for("partner") == ["spk1", "spk2"] and fake.speakers_for("partner", "dev") == ["spk1"]
    assert fake.speakers_for("babble") == ["bab1"] and fake.source == "libri"
    a, b = _scene(corpus="tts"), _scene(corpus="fake")
    assert os.path.exists(os.path.join(root, "demo", "bus3_snr+0.fake.recipe.json"))
    assert a["meta"]["geom_id"] == b["meta"]["geom_id"] and b["meta"]["corpus"] == "fake"
    assert {t["speaker"] for t in b["meta"]["talkers"].values()} == {"spk1", "spk2"}
    ra, rb = (json.load(open(os.path.join(root, "demo", f"bus3_snr+0.{k}.recipe.json"))) for k in ("tts", "fake"))
    noise = lambda r: [g for g in r["groups"] if g["role"] != "partner"]  # noqa: E731
    assert noise(ra) == noise(rb) and ra["impair"] == rb["impair"] and ra["seed"] == rb["seed"]
    assert not np.array_equal(a["mix"], b["mix"])
    assert np.array_equal(sum(np.asarray(v) for v in b["comp2"].values()), b["mix"])


def test_directivity_and_image_hook():
    dims, mics = [8.0, 6.0, 3.0], s2.place_array([4.0, 3.0, 1.0], 0.04, 0.0)
    src = [5.0, 3.0, 1.0]  # 1 m from the array center along +x
    toward = {"azimuth_deg": 180.0, "colatitude_deg": 90.0, "p": 0.75}
    away = {"azimuth_deg": 0.0, "colatitude_deg": 90.0, "p": 0.75}
    kw = dict(max_order=0, tail="none", rand_disp=0.0)
    r_to = s2.compute_rirs(dims, 0.3, [src], mics, 1, directivity=[toward], **kw)[0]
    r_aw = s2.compute_rirs(dims, 0.3, [src], mics, 1, directivity=[away], **kw)[0]
    ratio = np.abs(r_aw[0, 0]).max() / np.abs(r_to[0, 0]).max()
    assert abs(ratio - 0.5) < 0.01, ratio  # p + (1 - p)·cos(180°)
    img = [5.0, 3.0, -1.0]  # mirror image in the floor plane z = 0

    def hook(room, k):
        assert k == 0 and len(room.sources) == 1
        s2.add_image(room, img, 0.9)
    r_im = s2.compute_rirs(dims, 0.3, [src], mics, 1, edit_images=hook, **kw)[0][0, 0].astype(float)
    d_dir = s2.source_truth(src, mics)["direct_a"]
    d_img = np.linalg.norm(np.subtract(img, mics[0])) / s2.C * FS + s2.FRAC
    k = int(np.argmax(np.abs(r_im[int(d_dir) + 20:]))) + int(d_dir) + 20
    assert abs(k - d_img) <= 1, (k, d_img)
    e = lambda c: np.sum(r_im[int(c) - 20:int(c) + 21] ** 2)  # noqa: E731
    expect = (0.9 * np.linalg.norm(np.subtract(src, mics[0])) / np.linalg.norm(np.subtract(img, mics[0]))) ** 2
    assert abs(e(d_img) / e(d_dir) / expect - 1) < 0.05


def test_tail_modes():
    dims, mics = [6.0, 4.0, 2.7], s2.place_array([3.0, 2.0, 1.0], 0.04, 30.0)

    def t60(h):
        edc = np.cumsum((h ** 2)[::-1])[::-1]
        edc = 10 * np.log10(edc / edc[0] + 1e-30)
        return 3 * (np.argmax(edc < -25) - np.argmax(edc < -5)) / FS
    for tail in ("synthetic", "ism"):
        r = s2.compute_rirs(dims, 0.35, [[1.5, 1.0, 1.2]], mics, 5, tail=tail)[0]
        assert 0.26 < t60(r[0, 0].astype(float)) < 0.44, (tail, t60(r[0, 0].astype(float)))


def test_bad_recipe_rejected():
    geom, _ = s2.load_geometry("demo", "demo_bus3_synthetic", root=_root())
    rec = json.load(open(os.path.join(_root(), "demo", "bus3_snr+0.tts.recipe.json")))
    rec["groups"][0]["sources"][0]["src"] = "nowhere"
    try:
        s2._validate(rec, geom, corpus2.load_corpus("tts"))
    except ValueError:
        return
    raise AssertionError("unknown source accepted")


def test_render_time():
    s2.clear_cache()
    t0 = time.time()
    sc = _scene()
    dt = time.time() - t0
    print(f"    demo render {sc['meta']['duration_s']:.0f} s audio in {dt:.2f} s")
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
