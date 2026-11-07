"""Real-device test set: speech played through the MacBook speakers, recorded by the ESP32 (each sound <= 4.5 s).

Each take is recorded alone. Its clean "image" at the two mics (speech through the real room and mics, no noise) is
recovered from the same take by fitting a 4096-tap two-channel response from the played signal to the recording
(the fit residual during speech equals the take's own room/fan noise, so the image is accurate to well below the noise).

  .venv/bin/python -I hw/realset.py stim            build hw/realset/stim/<take>.wav (48 kHz) + .npz (16 kHz dry)
  .venv/bin/python -I hw/realset.py record [take..] play + record (12 s each) into hw/realset/rec/<take>.wav (+ .json)
  .venv/bin/python -I hw/realset.py images [take..] fit images into hw/realset/img/<take>.npz
Takes: bus partners A (LibriSpeech 121, F, left speaker) and B (1089, M, right); enrollment takes AE*/BE* from the
reserved enrollment utterances; meeting talkers M1-M6 at different levels; MF1/MF2 = "far talkers" (speech pre-convolved
with a simulated 4 m, RT60 0.8 s room response before playback).
"""
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LAB = os.path.dirname(HERE)
ROOT = os.path.join(HERE, "realset")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

FS, FS_PLAY, LEAD, CLIP = 16000, 48000, 0.25, 4.2

# take: (speaker, side, level_db, kind)  kind: scene | enroll | far
TAKES = {
    "A1": ("121", "L", 0, "scene"), "A2": ("121", "L", 0, "scene"), "A3": ("121", "L", 0, "scene"),
    "B1": ("1089", "R", 0, "scene"), "B2": ("1089", "R", 0, "scene"), "B3": ("1089", "R", 0, "scene"),
    "AE1": ("121", "L", 0, "enroll"), "AE2": ("121", "L", 0, "enroll"),
    "BE1": ("1089", "R", 0, "enroll"), "BE2": ("1089", "R", 0, "enroll"),
    "M1": ("3575", "L", 0, "scene"), "M2": ("2300", "R", -6, "scene"), "M3": ("1995", "L", -12, "scene"),
    "M4": ("7729", "R", -3, "scene"), "M5": ("4970", "L", -9, "scene"), "M6": ("1320", "R", -15, "scene"),
    "MF1": ("6829", "L", -6, "far"), "MF2": ("260", "R", -6, "far"),
}


def db(x):
    return 10 ** (x / 20)


def fade(x, t_in=0.02, t_out=0.05):
    x = x.copy()
    a, b = int(t_in * FS), int(t_out * FS)
    x[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    x[-b:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(b) / b)
    return x


def frame_db(x, n=320):
    e = np.array([np.mean(x[i:i + n] ** 2) for i in range(0, len(x) - n, n)]) + 1e-12
    return 10 * np.log10(e)


def best_window(x, length):
    e = frame_db(x)
    act = e > e.max() - 30
    w = int(length * FS / 320)
    if len(act) <= w:
        return x[:int(length * FS)], float(act.mean())
    score = np.convolve(act, np.ones(w), "valid")
    i = int(np.argmax(score))
    return x[i * 320:i * 320 + int(length * FS)], float(score[i] / w)


def normalize(x, level_db):
    e = frame_db(x)
    act = np.sqrt(np.mean(10 ** (e[e > e.max() - 30] / 10)))
    x = x * db(-22.0) / act
    if np.abs(x).max() > 0.5:
        x *= 0.5 / np.abs(x).max()
    return x * db(level_db)


def far_rir(seed=7):
    """Mono response of a 12 x 8 x 3.5 m room, RT60 ~0.8 s, talker 4 m from the listener."""
    import pyroomacoustics as pra
    rng = np.random.default_rng(seed)
    dims = [12.0, 8.0, 3.5]
    e_abs, max_order = pra.inverse_sabine(0.8, dims)
    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=min(max_order, 40))
    room.add_source([3.0, 4.0 + rng.uniform(-0.2, 0.2), 1.2])
    room.add_microphone([7.0, 4.0, 1.2])
    room.compute_rir()
    h = np.asarray(room.rir[0][0], float)
    return h / np.abs(h).max()


def build_stimuli():
    os.makedirs(os.path.join(ROOT, "stim"), exist_ok=True)
    with open(os.path.join(LAB, "corpus_libri/utterances.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    enroll = json.load(open(os.path.join(LAB, "corpus_libri/enrollment.json")))["speakers"]
    used, manifest = set(), {}
    hfar = None
    for take, (spk, side, level, kind) in TAKES.items():
        if kind == "enroll":
            k = int(take[-1]) - 1
            sample = enroll[spk]["samples"][k]
            x = np.concatenate([sf.read(os.path.join(LAB, p), dtype="float32")[0] for p in sample["paths"]])
            ids = sample["ids"]
            seg, frac = best_window(x, CLIP)
        else:
            cands = [r for r in rows if r["speaker"] == spk and r["use"] == "scene" and r["id"] not in used
                     and float(r["duration_s"]) >= 6 and float(r["snr_db"]) >= 25 and int(r["clip_n"]) == 0]
            cands.sort(key=lambda r: -float(r["snr_db"]))
            best = None
            for r in cands[:6]:
                x, fs = sf.read(os.path.join(LAB, r["path"]), dtype="float32")
                assert fs == FS
                length = 3.4 if kind == "far" else CLIP
                seg_, frac_ = best_window(x, length)
                if best is None or frac_ > best[0]:
                    best = (frac_, r["id"], seg_)
            frac, uid, seg = best
            used.add(uid)
            ids = [uid]
        dry = normalize(fade(seg.astype(np.float64)), level)
        played = dry
        if kind == "far":
            if hfar is None:
                hfar = far_rir()
            played = np.convolve(dry, hfar)[:int(CLIP * FS)]
            played = fade(normalize(played, level))
        n16 = int(LEAD * FS) + len(played)
        sig = np.pad(played, (int(LEAD * FS), 0))
        L = sig if side == "L" else np.zeros(n16)
        R = sig if side == "R" else np.zeros(n16)
        play = np.stack([resample_poly(L, 3, 1), resample_poly(R, 3, 1)], 1)
        assert np.abs(play).max() < 0.9 and len(play) / FS_PLAY - LEAD <= 4.5 + 1e-6
        sf.write(os.path.join(ROOT, "stim", take + ".wav"), play.astype(np.float32), FS_PLAY, subtype="PCM_24")
        np.savez(os.path.join(ROOT, "stim", take + ".npz"), played=sig.astype(np.float32),
                 dry=np.pad(dry, (int(LEAD * FS), 0)).astype(np.float32))
        manifest[take] = {"speaker": spk, "side": side, "level_db": level, "kind": kind, "utts": ids,
                          "active_frac": round(frac, 3), "sound_s": round(len(played) / FS, 2)}
        print(take, manifest[take])
    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


def record(takes):
    from esp32 import play_record
    os.makedirs(os.path.join(ROOT, "rec"), exist_ok=True)
    for i, take in enumerate(takes):
        if i:
            time.sleep(1.5)
        fs, x, meta = play_record(os.path.join(ROOT, "stim", take + ".wav"), 12)
        sf.write(os.path.join(ROOT, "rec", take + ".wav"), x, fs, subtype="FLOAT")
        meta["take"] = take
        meta["wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(ROOT, "rec", take + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(take, "recorded", flush=True)


def fit_image(y, s, K=4096, alpha=17e-6, pre=256):
    """Clean two-channel image of the played signal s inside recording y (both 16 kHz)."""
    from analyze import bandpass, stretch, xcorr_lag
    s2 = stretch(s, alpha)
    n0 = xcorr_lag(bandpass(y[:, 0], 150, 7000), bandpass(s2, 150, 7000)) - pre
    N = 1 << int(np.ceil(np.log2(len(y) + len(s2))))
    S = np.fft.rfft(s2, N)
    Y = np.fft.rfft(y, N, axis=0)
    H = Y * np.conj(S)[:, None] / (np.abs(S)[:, None] ** 2 + 1e-3 * np.mean(np.abs(S) ** 2))
    h = np.roll(np.fft.irfft(H, N, axis=0), -n0, axis=0)[:K]
    taper = np.ones(K)
    taper[-512:] = np.hanning(1024)[512:]
    h *= taper[:, None]
    img = np.zeros_like(y)
    for c in range(2):
        seg = np.convolve(s2, h[:, c])[:max(0, len(y) - n0)]
        img[n0:n0 + len(seg), c] = seg
    return img, h, n0


def images(takes):
    from analyze import bandpass, load  # noqa: F401
    from scipy.signal import butter, sosfiltfilt
    hp = butter(4, 80, "high", fs=FS, output="sos")
    os.makedirs(os.path.join(ROOT, "img"), exist_ok=True)
    report = {}
    for take in takes:
        x, fs = sf.read(os.path.join(ROOT, "rec", take + ".wav"), dtype="float64")
        y = sosfiltfilt(hp, x - x.mean(0), axis=0)
        st = np.load(os.path.join(ROOT, "stim", take + ".npz"))
        img, h, n0 = fit_image(y, st["played"].astype(np.float64))
        noise = y - img
        on = n0 + 256 + int(LEAD * FS)
        act = slice(on, on + int(CLIP * FS))
        quiet = slice(on + int(4.7 * FS), len(y) - FS // 4)
        p = lambda z: float(np.mean(z ** 2))
        rep = {"onset_sample": int(on), "image_dbfs": round(10 * np.log10(p(img[act, 0])), 1),
               "residual_during_speech_dbfs": round(10 * np.log10(p(noise[act, 0])), 1),
               "noise_only_dbfs": round(10 * np.log10(p(noise[quiet, 0])), 1)}
        rep["snr_at_micA_db"] = round(rep["image_dbfs"] - rep["noise_only_dbfs"], 1)
        np.savez(os.path.join(ROOT, "img", take + ".npz"), recording=y.astype(np.float32), image=img.astype(np.float32),
                 noise=noise.astype(np.float32), h=h.astype(np.float32), onset=on, fs=FS)
        report[take] = rep
        print(take, rep, flush=True)
    path = os.path.join(ROOT, "images.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(report)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "stim":
        build_stimuli()
    elif cmd == "record":
        record(args or list(TAKES))
    elif cmd == "images":
        images(args or list(TAKES))
