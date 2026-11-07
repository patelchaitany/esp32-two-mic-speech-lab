"""Tests for the LibriSpeech table, splits, enrollment and corpus2 drop-in (lab.corpus_libri).
Run: .venv/bin/python -m tests.test_corpus_libri   (about 10 s; needs `python -m lab.corpus_libri --build` once)
"""
import atexit
import json
import os
import shutil
import tempfile
import time

import numpy as np
import soundfile as sf

from lab import corpus2
from lab import corpus_libri as cl
from lab.common import FS, active_rms

LAB = cl.LAB


def _table():
    if not hasattr(_table, "rows"):
        _table.rows = cl.load_table()
    return _table.rows


def test_files_written():
    for name in ("utterances.tsv", "speakers.tsv", "splits.json", "enrollment.json"):
        assert os.path.exists(os.path.join(LAB, cl.OUT_DIR, name)), name
    assert os.path.exists(os.path.join(LAB, cl.MANIFEST))


def test_table_matches_disk():
    """Every transcript line on disk is in the table once, with existing paths, SPEAKERS.TXT gender and the
    chapter of its folder; no audio was copied (paths point into data/)."""
    rows = _table()
    disk = cl.list_utterances("test-clean") + cl.list_utterances("test-other")
    assert len(rows) == len(disk) == 2620 + 2939
    assert [r["id"] for r in rows] == sorted(u[0] for u in disk)
    spk = {**cl.read_speakers_txt("test-clean"), **cl.read_speakers_txt("test-other")}
    for r in rows:
        assert r["path"].startswith("data/librispeech_test_") and r["path"].endswith(r["id"] + ".flac")
        assert os.path.exists(os.path.join(LAB, r["path"])), r["path"]
        assert r["gender"] == spk[r["speaker"]]["gender"] and r["subset"] == spk[r["speaker"]]["subset"]
        assert r["id"].split("-")[:2] == [r["speaker"], r["chapter"]] and r["text"]
        assert 0 < r["active_s"] <= r["duration_s"] and r["use"] in ("scene", "enroll")


def test_durations_match_audio():
    for r in _table():
        info = sf.info(os.path.join(LAB, r["path"]))
        assert info.samplerate == FS and info.channels == 1
        assert abs(info.frames / FS - r["duration_s"]) < 1e-4, r["id"]
    man = json.load(open(os.path.join(LAB, cl.MANIFEST)))
    dur = {r["id"]: r["duration_s"] for r in _table()}
    assert all(abs(u["duration"] - dur[u["id"]]) < 1e-9 for u in man["utterances"])


def test_measurements_reproducible():
    """Re-analysing a seeded sample of files gives the stored VAD / level values exactly."""
    rows = _table()
    rng = np.random.default_rng(0)
    for k in rng.choice(len(rows), 25, replace=False):
        r = rows[int(k)]
        m = cl.analyze_file(r["path"])
        for key in ("duration_s", "active_s", "active_rms_db", "norm_rms_db", "floor_db", "snr_db", "f0_hz",
                    "clip_n"):
            assert abs(m[key] - r[key]) < 1e-9, (r["id"], key, m[key], r[key])


def test_splits():
    sp, spk = cl.load_splits(), cl.load_speakers()
    tc = set(cl.read_speakers_txt("test-clean"))
    to = set(cl.read_speakers_txt("test-other"))
    td, tt, bd, bt = (set(sp[p][s]) for p in ("target", "babble") for s in ("dev", "test"))
    assert len(td) == len(tt) == 20 and not td & tt and td | tt == tc
    assert not bd & bt and bd | bt == to and (len(bd), len(bt)) == (16, 17)
    assert not (td | tt) & (bd | bt)
    for ids in (td, tt):
        assert sorted(spk[s]["gender"] for s in ids) == ["F"] * 10 + ["M"] * 10
    assert sum(spk[s]["gender"] == "F" for s in bd) == 8 and sum(spk[s]["gender"] == "M" for s in bd) == 8
    for s, v in spk.items():
        assert v["split"] == ("dev" if s in td | bd else "test") and v["pool"] == ("target" if s in tc else "babble")
    # the split is a deterministic function of the stored F0 and the seed
    for pool in ("target", "babble"):
        again = cl.split_speakers({s: v for s, v in spk.items() if v["pool"] == pool})
        assert all(again[s] == spk[s]["split"] for s in again)
    # every utterance carries its speaker's pool and split
    assert all(r["split"] == spk[r["speaker"]]["split"] and r["pool"] == spk[r["speaker"]]["pool"] for r in _table())


def test_enrollment_reserved():
    en, spk = cl.load_enrollment(), cl.load_speakers()
    rows = {r["id"]: r for r in _table()}
    target = {s for s, v in spk.items() if v["pool"] == "target"}
    assert set(en) == target
    ids = cl.enrollment_ids()
    assert ids == {i for i, r in rows.items() if r["use"] == "enroll"}
    for s, v in en.items():
        assert len(v["samples"]) == 2 and v["split"] == spk[s]["split"]
        seen = set()
        for smp in v["samples"]:
            assert 8.0 <= smp["active_s"] <= 15.0, (s, smp["ids"])
            assert abs(smp["active_s"] - sum(rows[i]["active_s"] for i in smp["ids"])) < 0.011
            assert all(rows[i]["speaker"] == s and rows[i]["chapter"] == smp["chapter"] for i in smp["ids"])
            assert all(rows[i]["clip_n"] == 0 for i in smp["ids"]), (s, smp["ids"])  # no clipped enrollment
            assert not seen & set(smp["ids"])
            seen |= set(smp["ids"])
            if len(smp["ids"]) > 1:  # runs only for the speaker without long single utterances
                assert s == "61" and [int(i.split("-")[2]) for i in smp["ids"]] == list(
                    range(int(smp["ids"][0].split("-")[2]), int(smp["ids"][0].split("-")[2]) + len(smp["ids"])))
        # enrollment chapter holds less than about half of the remaining scene material when possible
        if spk[s]["n_chapters"] > 1:
            assert spk[s]["enroll_chapter_scene_share"] < 0.52, (s, spk[s]["enroll_chapter_scene_share"])
        else:
            assert spk[s]["enroll_chapter_scene_share"] == 1.0
    # single-utterance samples everywhere except speaker 61
    assert sum(len(smp["ids"]) for v in en.values() for smp in v["samples"]) == len(ids) == 83


def test_dropin_manifest_and_corpus2():
    """corpora/libri/manifest.json is in the BENCH2.md format and loads as corpus2 corpus 'libri' without
    any enrollment utterance."""
    man = json.load(open(os.path.join(LAB, cl.MANIFEST)))
    assert man["name"] == "libri" and man["source"] == "libri"
    assert set(man) == {"name", "source", "speakers", "utterances"}
    for u in man["utterances"]:
        assert set(u) == {"id", "speaker", "text", "path", "duration"} and isinstance(u["duration"], float)
    for s, v in man["speakers"].items():
        assert v["pool"] in ("target", "babble") and v["split"] in ("dev", "test") and v["gender"] in ("F", "M")
    corpus2.load_corpus.cache_clear()
    c = corpus2.load_corpus("libri")
    sp, ids = cl.load_splits(), cl.enrollment_ids()
    assert len(c.speakers) == 73 and len(c.utts) == 5559 - len(ids)
    assert not any(i in c for i in ids)
    for role, pool in (("partner", "target"), ("talker", "target"), ("babble", "babble")):
        for split in ("dev", "test"):
            assert c.speakers_for(role, split) == sorted(sp[pool][split]), (role, split)
    for s in sp["target"]["dev"][:3]:
        got = {u.id for u in c.select(speaker=s)}
        assert got and not got & ids
    x = c.audio(c.utts[0].id)
    assert x.dtype == np.float32 and abs(active_rms(x.astype(np.float64)) - 1.0) < 1e-3


def test_enrollment_audio():
    en = cl.load_enrollment()
    for s in ("61", "1089"):
        xs = cl.enrollment_audio(s)
        assert len(xs) == 2
        for x, smp in zip(xs, en[s]["samples"]):
            assert x.dtype == np.float32 and abs(len(x) / FS - smp["duration_s"]) < 1e-3
            assert abs(active_rms(x.astype(np.float64)) - 1.0) < 1e-3
    raw = cl.enrollment_audio("1089", normalize=False)[0]
    ref, _ = sf.read(os.path.join(LAB, en["1089"]["samples"][0]["paths"][0]), dtype="float32")
    assert np.allclose(raw, ref, atol=1e-7)


def test_scene_render_with_libri():
    """load_scene(..., corpus='libri') re-renders the scene2 demo geometry with LibriSpeech partners."""
    from lab import scene2 as s2
    root = tempfile.mkdtemp(prefix="libri_test_")
    atexit.register(shutil.rmtree, root, True)
    sid = s2.build_demo(root=root)
    sc = s2.load_scene(sid, corpus="libri", root=root)
    assert sc["meta"]["corpus"] == "libri" and os.path.exists(
        os.path.join(root, "demo", sid.split("/")[1] + ".libri.recipe.json"))
    spk = cl.load_speakers()
    ids = cl.enrollment_ids()
    for t in sc["meta"]["talkers"].values():
        assert spk[t["speaker"]]["pool"] == "target" and not set(t["utts"]) & ids
    assert np.array_equal(sum(np.asarray(v) for v in sc["comp2"].values()), sc["mix"])


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
