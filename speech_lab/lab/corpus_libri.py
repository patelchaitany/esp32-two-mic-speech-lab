"""LibriSpeech test-clean / test-other: utterance table, speaker splits, enrollment reservation and the
lab.corpus2 drop-in manifest (DESIGN.md 4.1 E0 natural speech, M9 explicit enrollment, 2.0 S1 embeddings).

Audio is never copied or converted: every path points into data/ (relative to the lab folder).

Build (about 20 s with 3 worker processes; deterministic, rewrites the files below):
    .venv/bin/python -m lab.corpus_libri --build
Summary of the written files:
    .venv/bin/python -m lab.corpus_libri --summary

API (after a build)
    load_table() -> [row dict (UTT_COLUMNS)];  load_speakers() -> {speaker: dict (SPK_COLUMNS)}
    load_splits() -> splits.json dict;  load_enrollment() -> {speaker: {..., samples: [...]}}
    enrollment_ids() -> set of reserved utterance ids;  enrollment_audio(speaker, normalize=True) -> [2 x float32]
    lab.corpus2.load_corpus('libri') -> Corpus (scene utterances only); scene2.load_scene(sid, corpus='libri')

Files written (paths relative to the lab folder):
  corpus_libri/utterances.tsv   all 5559 utterances sorted by id (UTT_COLUMNS), including the reserved
                                enrollment ones (use = 'enroll'); tab-separated, one header line
  corpus_libri/speakers.tsv     all 73 speakers (SPK_COLUMNS)
  corpus_libri/splits.json      speaker splits and how they were made
  corpus_libri/enrollment.json  2 reserved enrollment samples per target speaker
  corpora/libri/manifest.json   drop-in for lab.corpus2 ('Scene format' in BENCH2.md): corpus 'libri';
                                enrollment utterances are left out, so scenes can never use them

Pools and splits (speaker level, all disjoint):
  target  test-clean, 40 speakers: bus partners and meeting talkers. dev 20 (10 F / 10 M), test 20 (10 F / 10 M).
  babble  test-other, 33 speakers: every bus interferer. dev 16 (8 F / 8 M), test 17 (9 F / 8 M).
  Within each gender, speakers are sorted by median F0 and taken in consecutive pairs; a seeded coin flip
  (SPLIT_SEED) sends one of each pair to dev and the other to test, so both splits span the same voice range.
  An unpaired speaker (odd count) goes to test.

Enrollment (target speakers only): 2 samples each with ENROLL_ACTIVE_S active speech, taken from the
speaker's chapter with the least material that still offers two such utterances, so most of the speaker's
scene material comes from other chapters (recording sessions). Within that chapter the two with the highest
estimated SNR are taken; clipped files (clip_n > 0) are never used. They are excluded from the drop-in
manifest (use 'enroll' in the table).
A sample is one utterance, except for speaker 61, whose longest utterance has only 7.3 s of active speech:
each of its samples is a run of consecutive utterances (choose_enrollment step 3).
11 of the 40 target speakers have a single chapter, so their enrollment shares the chapter of all their scene
material (speakers.tsv enroll_chapter_scene_share = 1.0).

Energy VAD (vad()): 20 ms frames (as lab.common.active_rms); a frame is active if its level is within 30 dB of
the loudest frame AND at least 10 dB above the utterance noise floor (5th percentile of frame levels, digital
silence excluded). active_s = 0.02 s x active frames; active_rms_db = RMS over the active frames.
norm_rms_db = lab.common.active_rms (the -30 dB rule only): the level lab.corpus2 normalizes every utterance by.
Levels are 20*log10(rms) re digital full scale 1.0 (a full-scale sine is -3.01 dB).
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

from .common import FS, active_rms

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSETS = {"test-clean": "data/librispeech_test_clean/LibriSpeech",
           "test-other": "data/librispeech_test_other/LibriSpeech"}
POOL_OF_SUBSET = {"test-clean": "target", "test-other": "babble"}
OUT_DIR = "corpus_libri"
MANIFEST = os.path.join("corpora", "libri", "manifest.json")
CORPUS_NAME = "libri"

SPLIT_SEED = 20261102
ENROLL_N = 2
ENROLL_ACTIVE_S = (8.0, 15.0)

VAD_FRAME_S = 0.02
VAD_REL_DB = -30.0
VAD_FLOOR_PCT = 5.0
VAD_FLOOR_MARGIN_DB = 10.0
SILENCE_DB = -120.0  # frames below this are digital silence (excluded from the floor estimate)

UTT_COLUMNS = ["id", "subset", "speaker", "gender", "chapter", "pool", "split", "use", "duration_s", "active_s",
               "active_rms_db", "norm_rms_db", "floor_db", "snr_db", "peak_dbfs", "clip_n", "f0_hz", "n_voiced",
               "path", "text"]
SPK_COLUMNS = ["speaker", "subset", "gender", "pool", "split", "minutes_txt", "n_utts", "total_s", "active_s",
               "n_scene_utts", "scene_s", "n_chapters", "chapters", "f0_hz", "f0_gender", "enroll_chapter",
               "enroll_chapter_scene_share"]
_FLOAT = {"duration_s", "active_s", "active_rms_db", "norm_rms_db", "floor_db", "snr_db", "peak_dbfs", "f0_hz",
          "minutes_txt", "total_s", "scene_s", "enroll_chapter_scene_share"}
_INT = {"n_voiced", "clip_n", "n_utts", "n_scene_utts", "n_chapters"}
CLIP_LEVEL = 0.999  # |sample| at or above this counts as clipped (clip_n): 11 files, all speaker 2094


# ----------------------------------------------------------------------------- metadata parsing
def _txt_rows(path):
    """Rows of a LibriSpeech pipe-separated metadata file (';' lines are comments)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith(";"):
                yield [c.strip() for c in line.rstrip("\n").split("|")]


def read_speakers_txt(subset):
    """{speaker: {gender, subset, minutes}} for the speakers of `subset` (from its SPEAKERS.TXT)."""
    out = {}
    for r in _txt_rows(os.path.join(LAB, SUBSETS[subset], "SPEAKERS.TXT")):
        if r[2] == subset:
            out[r[0]] = {"gender": r[1], "subset": r[2], "minutes": float(r[3])}
    return out


def read_chapters_txt(subset):
    """{chapter: {reader, minutes, book_id}} for the chapters of `subset` (from its CHAPTERS.TXT)."""
    out = {}
    for r in _txt_rows(os.path.join(LAB, SUBSETS[subset], "CHAPTERS.TXT")):
        if r[3] == subset:
            out[r[0]] = {"reader": r[1], "minutes": float(r[2]), "book_id": r[5]}
    return out


def list_utterances(subset):
    """[(id, speaker, chapter, path relative to LAB, transcript)] of a subset, sorted by id."""
    root = os.path.join(SUBSETS[subset], subset)
    out = []
    for spk in sorted(os.listdir(os.path.join(LAB, root))):
        sdir = os.path.join(root, spk)
        if not os.path.isdir(os.path.join(LAB, sdir)):
            continue
        for ch in sorted(os.listdir(os.path.join(LAB, sdir))):
            cdir = os.path.join(sdir, ch)
            trans = os.path.join(LAB, cdir, f"{spk}-{ch}.trans.txt")
            if not os.path.exists(trans):
                continue
            with open(trans, encoding="utf-8") as fh:
                for line in fh:
                    uid, _, text = line.strip().partition(" ")
                    if uid:
                        out.append((uid, spk, ch, os.path.join(cdir, uid + ".flac"), text))
    return sorted(out)


# ----------------------------------------------------------------------------- signal analysis
def _frame_db(x, n):
    k = len(x) // n
    if k == 0:
        return np.array([10 * np.log10(np.mean(np.square(x)) + 1e-30)])
    return 10 * np.log10(np.mean(np.square(x[: k * n].reshape(k, n)), axis=1) + 1e-30)


def vad(x, fs=FS):
    """Energy VAD on 20 ms frames. Returns (active mask per frame, frame levels dB, floor dB, peak dB)."""
    lev = _frame_db(np.asarray(x, np.float64), int(VAD_FRAME_S * fs))
    peak = float(lev.max())
    live = lev[lev > SILENCE_DB]
    floor = float(np.percentile(live, VAD_FLOOR_PCT)) if len(live) else SILENCE_DB
    act = (lev >= peak + VAD_REL_DB) & (lev >= floor + VAD_FLOOR_MARGIN_DB)
    if not act.any():
        act = lev >= peak + VAD_REL_DB
    return act, lev, floor, peak


def f0_frames(x, fs=FS, fmin=60.0, fmax=400.0, thr=0.5):
    """F0 (Hz) of voiced 40 ms frames (20 ms hop): normalized autocorrelation with window correction
    (Boersma 1993), frames within 25 dB of the loudest, voiced if the ACF peak > thr."""
    n, h = int(0.04 * fs), int(0.02 * fs)
    if len(x) < n:
        return np.zeros(0)
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, nfft)) ** 2)[:n]
    rw /= rw[0]
    fr = np.lib.stride_tricks.sliding_window_view(np.asarray(x, np.float64), n)[::h]
    lev = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-30)
    fr = fr[lev > lev.max() - 25.0]
    fr = (fr - fr.mean(axis=1, keepdims=True)) * w
    r = np.fft.irfft(np.abs(np.fft.rfft(fr, nfft, axis=1)) ** 2, axis=1)[:, :n]
    lo, hi = int(fs / fmax), int(np.ceil(fs / fmin))
    r = r[:, : hi + 2] / (r[:, :1] + 1e-30) / rw[: hi + 2]
    k = lo + np.argmax(r[:, lo: hi + 1], axis=1)
    pk = r[np.arange(len(k)), k]
    # parabolic interpolation of the peak lag
    a, b, c = r[np.arange(len(k)), k - 1], pk, r[np.arange(len(k)), k + 1]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    f0 = fs / (k + np.clip(delta, -0.5, 0.5))
    return f0[pk > thr]


def analyze_file(path):
    """Per-utterance measurements (see module docstring) of one audio file (path relative to LAB)."""
    x, fs = sf.read(os.path.join(LAB, path), dtype="float64", always_2d=True)
    if fs != FS:
        raise ValueError(f"{path}: fs {fs} != {FS}")
    x = x[:, 0]
    act, lev, floor, _ = vad(x, fs)
    a_db = float(10 * np.log10(np.mean(10 ** (lev[act] / 10))))
    f0 = f0_frames(x, fs)
    return {"duration_s": round(len(x) / fs, 4), "active_s": round(float(act.sum()) * VAD_FRAME_S, 2),
            "active_rms_db": round(a_db, 2), "norm_rms_db": round(float(20 * np.log10(active_rms(x, fs) + 1e-30)), 2),
            "floor_db": round(max(floor, SILENCE_DB), 2), "snr_db": round(a_db - max(floor, SILENCE_DB), 2),
            "peak_dbfs": round(float(20 * np.log10(np.abs(x).max() + 1e-30)), 2),
            "clip_n": int(np.sum(np.abs(x) >= CLIP_LEVEL)),
            "f0_hz": round(float(np.median(f0)), 1) if len(f0) else 0.0, "n_voiced": int(len(f0))}


def _weighted_median(v, w):
    o = np.argsort(v)
    cw = np.cumsum(np.asarray(w, float)[o])
    return float(np.asarray(v)[o][np.searchsorted(cw, cw[-1] / 2)])


# ----------------------------------------------------------------------------- splits and enrollment
def split_speakers(spk, seed=SPLIT_SEED):
    """{speaker: 'dev'|'test'} for one pool. spk: {speaker: {gender, f0_hz}}. Per gender: sort by median F0,
    pair neighbours, seeded coin flip per pair; an unpaired last speaker goes to test."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in ("F", "M"):
        ids = sorted((s for s in spk if spk[s]["gender"] == g), key=lambda s: (spk[s]["f0_hz"], s))
        for i in range(0, len(ids) - 1, 2):
            a, b = ids[i], ids[i + 1]
            if rng.random() < 0.5:
                a, b = b, a
            out[a], out[b] = "dev", "test"
        if len(ids) % 2:
            out[ids[-1]] = "test"
    return out


def choose_enrollment(rows):
    """The ENROLL_N enrollment samples of one speaker (rows = that speaker's utterances): a list of samples,
    each a list of utterance rows (normally one utterance; see step 3).

    1. Chapters are tried from the least total material upwards; the first chapter offering ENROLL_N single
       utterances with ENROLL_ACTIVE_S active speech supplies them (highest SNR first, then id).
    2. Otherwise such single utterances are pooled over chapters in the same order.
    3. Only if a speaker has fewer than ENROLL_N such utterances at all (test-clean speaker 61: the longest
       has 7.3 s of active speech), runs of 2-4 consecutive utterances of one chapter whose summed active
       speech is in ENROLL_ACTIVE_S are added, chapters in the same order, highest mean SNR first.
    Clipped utterances (clip_n > 0) are never used for enrollment."""
    lo, hi = ENROLL_ACTIVE_S
    by_ch = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_ch.setdefault(r["chapter"], []).append(r)
    order = sorted(by_ch, key=lambda c: (sum(r["duration_s"] for r in by_ch[c]), c))
    ok = lambda a: lo <= a <= hi  # noqa: E731
    rank = lambda s: (-float(np.mean([r["snr_db"] for r in s])), s[0]["id"])  # noqa: E731
    singles = {c: sorted(([r] for r in by_ch[c] if ok(r["active_s"]) and not r["clip_n"]), key=rank) for c in order}
    for c in order:
        if len(singles[c]) >= ENROLL_N:
            return singles[c][:ENROLL_N]
    picked = [s for c in order for s in singles[c]][:ENROLL_N]
    if len(picked) == ENROLL_N:
        return picked
    used = {r["id"] for s in picked for r in s}
    for c in order:
        runs = []
        us = by_ch[c]
        for i in range(len(us)):
            for k in range(2, 5):
                run = us[i:i + k]
                if len(run) == k and ok(sum(r["active_s"] for r in run)) and not any(r["clip_n"] for r in run):
                    runs.append(run)
                    break
        for run in sorted(runs, key=rank):
            if len(picked) < ENROLL_N and not used & {r["id"] for r in run}:
                picked.append(run)
                used |= {r["id"] for r in run}
    return picked


# ----------------------------------------------------------------------------- build
def build(workers=3, verbose=True):
    """Measure every utterance, make splits and enrollment, write all files. Returns a summary dict."""
    t0 = time.time()
    rows, spk_txt = [], {}
    for subset in SUBSETS:
        spk_txt.update(read_speakers_txt(subset))
        for uid, spk, ch, path, text in list_utterances(subset):
            rows.append({"id": uid, "subset": subset, "speaker": spk, "chapter": ch, "path": path, "text": text})
    rows.sort(key=lambda r: r["id"])
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 3))) as ex:
        meas = list(ex.map(analyze_file, [r["path"] for r in rows], chunksize=16))
    for r, m in zip(rows, meas):
        r.update(m)
        r["gender"] = spk_txt[r["speaker"]]["gender"]
        r["pool"] = POOL_OF_SUBSET[r["subset"]]
    if verbose:
        print(f"measured {len(rows)} utterances in {time.time() - t0:.1f} s")

    # speaker table, F0 and splits
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)
    spk = {}
    for s, rs in sorted(by_spk.items()):
        v = [r for r in rs if r["n_voiced"] > 0]
        f0 = _weighted_median([r["f0_hz"] for r in v], [r["n_voiced"] for r in v]) if v else 0.0
        spk[s] = {"speaker": s, "subset": rs[0]["subset"], "gender": spk_txt[s]["gender"], "pool": rs[0]["pool"],
                  "minutes_txt": spk_txt[s]["minutes"], "f0_hz": round(f0, 1),
                  "f0_gender": "F" if f0 >= 160.0 else "M"}
    split = {}
    for pool in ("target", "babble"):
        split.update(split_speakers({s: v for s, v in spk.items() if v["pool"] == pool}))
    for r in rows:
        r["split"] = split[r["speaker"]]
        r["use"] = "scene"

    # enrollment (target speakers)
    enroll = {}
    for s, rs in sorted(by_spk.items()):
        if spk[s]["pool"] != "target":
            continue
        pick = choose_enrollment(rs)
        for sample in pick:
            for r in sample:
                r["use"] = "enroll"
        enroll[s] = pick
    for s, rs in sorted(by_spk.items()):
        sc = [r for r in rs if r["use"] == "scene"]
        chs = {}
        for r in rs:
            chs[r["chapter"]] = chs.get(r["chapter"], 0.0) + r["duration_s"]
        e_ch = sorted({r["chapter"] for sample in enroll.get(s, []) for r in sample})
        share = (sum(r["duration_s"] for r in sc if r["chapter"] in e_ch) / max(sum(r["duration_s"] for r in sc), 1e-9)
                 if e_ch else 0.0)
        spk[s].update({"split": split[s], "n_utts": len(rs), "total_s": round(sum(r["duration_s"] for r in rs), 2),
                       "active_s": round(sum(r["active_s"] for r in rs), 2), "n_scene_utts": len(sc),
                       "scene_s": round(sum(r["duration_s"] for r in sc), 2), "n_chapters": len(chs),
                       "chapters": ",".join(f"{c}:{v:.1f}" for c, v in sorted(chs.items())),
                       "enroll_chapter": "+".join(e_ch), "enroll_chapter_scene_share": round(share, 3)})

    _write_all(rows, spk, split, enroll)
    summ = summary(rows, spk)
    summ["build_s"] = round(time.time() - t0, 1)
    if verbose:
        print(json.dumps(summ, indent=1))
    return summ


def _write_tsv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        w.writerow(columns)
        for r in rows:
            w.writerow([r[c] for c in columns])
    os.replace(tmp, path)


def _write_json(path, obj, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def _write_all(rows, spk, split, enroll):
    out = os.path.join(LAB, OUT_DIR)
    _write_tsv(os.path.join(out, "utterances.tsv"), UTT_COLUMNS, rows)
    _write_tsv(os.path.join(out, "speakers.tsv"), SPK_COLUMNS, [spk[s] for s in sorted(spk)])
    splits = {"seed": SPLIT_SEED,
              "method": "per pool and gender: sort speakers by median F0, pair neighbours, seeded coin flip per "
                        "pair (one to dev, one to test); an odd last speaker goes to test",
              "target": {"subset": "test-clean", "roles": ["partner", "talker"]},
              "babble": {"subset": "test-other", "roles": ["babble"]}}
    for pool in ("target", "babble"):
        for sp in ("dev", "test"):
            ids = sorted(s for s in spk if spk[s]["pool"] == pool and split[s] == sp)
            splits[pool][sp] = ids
            splits[pool][f"{sp}_gender"] = {g: sum(spk[s]["gender"] == g for s in ids) for g in ("F", "M")}
    _write_json(os.path.join(out, "splits.json"), splits)
    en = {"rule": f"{ENROLL_N} samples per target speaker, each {ENROLL_ACTIVE_S[0]:g}-{ENROLL_ACTIVE_S[1]:g} s of "
                  "active speech (lab.corpus_libri.vad): single utterances from the speaker's chapter with the "
                  "least material that offers two, highest SNR first (choose_enrollment). A sample is a run of "
                  "consecutive utterances only when the speaker has no two long enough single utterances "
                  "(speaker 61). Never used by scenes: absent from corpora/libri/manifest.json.",
          "enroll_chapter_scene_share": "fraction of the speaker's remaining (scene) seconds that come from the "
                                        "enrollment chapter(s); 1.0 for speakers with a single chapter",
          "speakers": {}}
    for s, pick in sorted(enroll.items()):
        en["speakers"][s] = {
            "split": split[s], "gender": spk[s]["gender"], "chapter": spk[s]["enroll_chapter"],
            "enroll_chapter_scene_share": spk[s]["enroll_chapter_scene_share"],
            "samples": [_sample_entry(smp) for smp in pick]}
    _write_json(os.path.join(out, "enrollment.json"), en)
    _write_json(os.path.join(LAB, MANIFEST), to_manifest(rows, spk), indent=None)


def _sample_entry(smp):
    """JSON entry of one enrollment sample (a list of consecutive utterance rows, usually one)."""
    act = sum(r["active_s"] for r in smp)
    lev = 10 * np.log10(sum(r["active_s"] * 10 ** (r["active_rms_db"] / 10) for r in smp) / max(act, 1e-9))
    return {"ids": [r["id"] for r in smp], "paths": [r["path"] for r in smp], "chapter": smp[0]["chapter"],
            "duration_s": round(sum(r["duration_s"] for r in smp), 4), "active_s": round(act, 2),
            "active_rms_db": round(float(lev), 2), "snr_db": round(float(np.mean([r["snr_db"] for r in smp])), 2),
            "text": " ".join(r["text"] for r in smp)}


def to_manifest(rows, spk):
    """lab.corpus2 drop-in manifest dict (BENCH2.md 'Scene format'): scene utterances only (no enrollment)."""
    speakers = {}
    for s in sorted(spk):
        v = spk[s]
        speakers[s] = {"pool": v["pool"], "split": v["split"], "gender": v["gender"], "quality": "natural",
                       "roles": ["partner", "talker"] if v["pool"] == "target" else ["babble"]}
    utts = [{"id": r["id"], "speaker": r["speaker"], "text": r["text"], "path": r["path"],
             "duration": float(r["duration_s"])} for r in sorted(rows, key=lambda r: r["id"]) if r["use"] == "scene"]
    return {"name": CORPUS_NAME, "source": "libri", "speakers": speakers, "utterances": utts}


def summary(rows=None, spk=None):
    """Counts and durations per pool / split / use, from the given rows or the written table."""
    rows = rows if rows is not None else load_table()
    spk = spk if spk is not None else load_speakers()
    out = {}
    for pool in ("target", "babble"):
        for sp in ("dev", "test"):
            ss = [s for s in spk.values() if s["pool"] == pool and s["split"] == sp]
            rr = [r for r in rows if r["pool"] == pool and r["split"] == sp]
            sc = [r for r in rr if r["use"] == "scene"]
            out[f"{pool}/{sp}"] = {
                "speakers": len(ss), "F": sum(s["gender"] == "F" for s in ss), "M": sum(s["gender"] == "M" for s in ss),
                "utts": len(rr), "scene_utts": len(sc), "enroll_utts": len(rr) - len(sc),
                "scene_hours": round(sum(r["duration_s"] for r in sc) / 3600, 3),
                "scene_active_hours": round(sum(r["active_s"] for r in sc) / 3600, 3)}
    return out


# ----------------------------------------------------------------------------- loaders
def _typed(row):
    out = {}
    for k, v in row.items():
        out[k] = float(v) if k in _FLOAT else int(v) if k in _INT else v
    return out


def _read_tsv(name):
    path = os.path.join(LAB, OUT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run .venv/bin/python -m lab.corpus_libri --build")
    with open(path, encoding="utf-8", newline="") as fh:
        return [_typed(r) for r in csv.DictReader(fh, delimiter="\t")]


def load_table():
    """All utterances (dicts with UTT_COLUMNS, numbers typed), sorted by id. use: 'scene' | 'enroll'."""
    return _read_tsv("utterances.tsv")


def load_speakers():
    """{speaker: dict with SPK_COLUMNS}."""
    return {r["speaker"]: r for r in _read_tsv("speakers.tsv")}


def load_splits():
    """splits.json: {'target': {'dev': [...], 'test': [...], ...}, 'babble': {...}, 'seed', 'method'}."""
    with open(os.path.join(LAB, OUT_DIR, "splits.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_enrollment():
    """{speaker: {split, gender, chapter, enroll_chapter_scene_share,
                  samples: [{ids, paths, chapter, duration_s, active_s, active_rms_db, snr_db, text}, x2]}}
    for the 40 target speakers. A sample is one utterance (len(ids) == 1) except for speaker 61 (see
    choose_enrollment)."""
    with open(os.path.join(LAB, OUT_DIR, "enrollment.json"), encoding="utf-8") as fh:
        return json.load(fh)["speakers"]


def enrollment_ids():
    """Set of all reserved enrollment utterance ids (never in corpus2's 'libri' corpus)."""
    return {i for v in load_enrollment().values() for smp in v["samples"] for i in smp["ids"]}


def enrollment_audio(speaker, normalize=True):
    """The 2 reserved enrollment samples of a target speaker: list of float32 (T,) at 16 kHz. A multi-utterance
    sample is the plain concatenation of its consecutive utterances (each keeps its own edge silences).
    normalize=True scales each sample to unit active RMS like lab.corpus2 audio (lab.common.active_rms);
    False returns the recorded level.

    For M9 explicit enrollment: a scene's talker -> speaker map is scene['meta']['talkers'][tid]['speaker']."""
    out = []
    for smp in load_enrollment()[str(speaker)]["samples"]:
        parts = []
        for p in smp["paths"]:
            x, fs = sf.read(os.path.join(LAB, p), dtype="float64", always_2d=True)
            if fs != FS:
                raise ValueError(f"{p}: fs {fs}")
            parts.append(x[:, 0])
        x = np.concatenate(parts)
        if normalize:
            x = x / (active_rms(x) + 1e-12)
        out.append(x.astype(np.float32))
    return out


def write_manifest():
    """(Re)write corpora/libri/manifest.json from the table (no audio analysis); returns its path."""
    rows, spk = load_table(), load_speakers()
    _write_json(os.path.join(LAB, MANIFEST), to_manifest(rows, spk), indent=None)
    return os.path.join(LAB, MANIFEST)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LibriSpeech table, splits, enrollment and corpus2 drop-in")
    ap.add_argument("--build", action="store_true", help="measure all utterances and write every file")
    ap.add_argument("--workers", type=int, default=3, help="worker processes for --build (max 3)")
    ap.add_argument("--manifest", action="store_true", help="only rewrite corpora/libri/manifest.json")
    ap.add_argument("--summary", action="store_true", help="print counts per pool/split")
    a = ap.parse_args()
    if a.build:
        build(a.workers)
    if a.manifest:
        print(write_manifest())
    if a.summary:
        print(json.dumps(summary(), indent=1))
