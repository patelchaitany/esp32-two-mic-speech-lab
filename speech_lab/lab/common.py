"""Shared constants and helpers for the 2-mic speech lab."""
import json
import numpy as np
import soundfile as sf

FS = 16000
C = 343.0
# INMP441: -26 dBFS (full-scale sine = 0 dBFS) at 94 dB SPL (1 Pa rms).
PA_TO_DIG = 10 ** (-26 / 20) / np.sqrt(2)


def spl_to_pa(db):
    return 20e-6 * 10 ** (db / 20)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-30))


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-30))


def active_rms(x, fs=FS, frame=0.02, rel_thresh_db=-30):
    """RMS over active frames (frames within rel_thresh_db of the loudest), like a light ITU P.56."""
    n = int(frame * fs)
    k = len(x) // n
    if k == 0:
        return rms(x)
    fr = np.sqrt(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1) + 1e-30)
    act = fr > fr.max() * 10 ** (rel_thresh_db / 20)
    return float(np.sqrt(np.mean(fr[act] ** 2)))


def read_wav(path):
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    return x, fs


def write_wav(path, x, fs=FS):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] < x.shape[1]:
        x = x.T
    sf.write(path, np.clip(x, -1, 1), fs, subtype="PCM_16")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)
