"""Build the speaker test sounds (each 4.5 s or shorter) and their 16 kHz references.

Writes hw/stim/<name>.wav (48 kHz stereo, played by afplay; L/R = MacBook left/right speaker)
and hw/stim/<name>.npz (16 kHz references: 'L', 'R', plus per-component sources).
Run: .venv/bin/python -I hw/stimuli.py
"""
import csv
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
OUT = os.path.join(HERE, "stim")
FS, FS_PLAY = 16000, 48000
LEAD = 0.25            # silence before the sound, so the speaker amp is awake
CLIP_S = 4.2           # speech/noise length
SPEECH_RMS_DB = -22.0  # active-speech RMS at 0 dB
SPEECH_PEAK = 0.50     # -6 dBFS ceiling, leaves headroom for the summed mixtures
NOISE_RMS_DB = -24.0


def db(x):
    return 10 ** (x / 20)


def fade(x, fs, t_in=0.02, t_out=0.05):
    x = x.copy()
    n_in, n_out = int(t_in * fs), int(t_out * fs)
    x[:n_in] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    x[-n_out:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n_out) / n_out)
    return x


def ess(f1=50.0, f2=7800.0, T=3.0, fs=FS_PLAY, amp=db(-10)):
    """Exponential sine sweep (Farina 2000)."""
    t = np.arange(int(T * fs)) / fs
    L = T / np.log(f2 / f1)
    return amp * fade(np.sin(2 * np.pi * f1 * L * (np.exp(t / L) - 1)), fs, 0.02, 0.05)


def frame_db(x, fs, win=0.02):
    n = int(win * fs)
    e = np.array([np.mean(x[i:i + n] ** 2) for i in range(0, len(x) - n, n)]) + 1e-12
    return 10 * np.log10(e), n


def pick_speech(rows, speaker):
    """Best 4.2 s window (most active frames) from the speaker's clean, long scene utterances."""
    best = None
    cands = [r for r in rows if r["speaker"] == speaker and r["use"] == "scene" and float(r["duration_s"]) >= 6
             and float(r["snr_db"]) >= 30 and int(r["clip_n"]) == 0]
    for r in sorted(cands, key=lambda r: -float(r["snr_db"]))[:8]:
        x, fs = sf.read(os.path.join(LAB, r["path"]), dtype="float32")
        assert fs == FS
        e, n = frame_db(x, fs)
        active = e > e.max() - 30
        w = int(CLIP_S * fs / n)
        score = np.convolve(active, np.ones(w), "valid")
        i = int(np.argmax(score))
        frac = score[i] / w
        if best is None or frac > best[0]:
            best = (frac, r["id"], x[i * n:i * n + int(CLIP_S * fs)])
    frac, uid, x = best
    e, _ = frame_db(x, FS)
    act_rms = np.sqrt(np.mean(10 ** (e[e > e.max() - 30] / 10)))
    x = x * db(SPEECH_RMS_DB) / act_rms
    if np.abs(x).max() > SPEECH_PEAK:
        x *= SPEECH_PEAK / np.abs(x).max()
    return fade(x, FS), {"speaker": speaker, "utt": uid, "active_frac": round(float(frac), 3)}


def noise_clip(ch, start_s=100.0):
    x, fs = sf.read(os.path.join(LAB, f"data/demand_tbus/TBUS/ch{ch:02d}.wav"), dtype="float32",
                    start=int(start_s * FS), frames=int(CLIP_S * FS))
    assert fs == FS
    x = x - x.mean()
    return fade(x * db(NOISE_RMS_DB) / np.sqrt(np.mean(x ** 2)), FS)


def write(name, L16, R16, sources, meta, sweep48=None):
    """L16/R16: 16 kHz channel signals (None = silent). sweep48: exact 48 kHz signal for sweeps."""
    n16 = int(round(LEAD * FS)) + max(len(s) for s in (L16, R16) if s is not None)
    pad = lambda s: np.zeros(n16, np.float32) if s is None else np.pad(s, (int(round(LEAD * FS)), n16 - len(s) - int(round(LEAD * FS))))
    L, R = pad(L16), pad(R16)
    if sweep48 is not None:
        n48 = int(round(LEAD * FS_PLAY)) + len(sweep48)
        s48 = np.pad(sweep48, (int(round(LEAD * FS_PLAY)), 0))
        play = np.stack([s48 if L16 is not None else np.zeros(n48), s48 if R16 is not None else np.zeros(n48)], 1)
    else:
        play = np.stack([resample_poly(L, 3, 1), resample_poly(R, 3, 1)], 1)
    assert np.abs(play).max() < 0.9, (name, np.abs(play).max())
    sound_s = (len(play) / FS_PLAY) - LEAD
    assert sound_s <= 4.5 + 1e-6, (name, sound_s)
    sf.write(os.path.join(OUT, name + ".wav"), play.astype(np.float32), FS_PLAY, subtype="PCM_24")
    np.savez(os.path.join(OUT, name + ".npz"), L=L.astype(np.float32), R=R.astype(np.float32),
             **{k: pad(v).astype(np.float32) for k, v in sources.items()})
    meta.update(name=name, lead_s=LEAD, sound_s=round(sound_s, 3), fs_ref=FS, fs_play=FS_PLAY,
                peak_dbfs=round(float(20 * np.log10(np.abs(play).max())), 1))
    return meta


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(LAB, "corpus_libri/utterances.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    A, mA = pick_speech(rows, "121")    # F, test split, has enrollment clips
    B, mB = pick_speech(rows, "1089")   # M, test split, has enrollment clips
    C, mC = pick_speech(rows, "3575")   # F, test split, played 15 dB quieter
    C = C * db(-15)
    N1, N4 = noise_clip(1), noise_clip(4)
    sw48 = ess()
    sw16 = resample_poly(sw48, 1, 3).astype(np.float32)
    noise_meta = {"noise": "DEMAND TBUS ch01 (L) + ch04 (R), 100.0-104.2 s"}
    sweep_meta = {"sweep": "ESS 50-7800 Hz, 3.0 s, -10 dBFS"}
    manifest = [
        write("sweep_L", sw16, None, {"sweep": sw16}, dict(sweep_meta, plays="left speaker"), sw48),
        write("sweep_R", None, sw16, {"sweep": sw16}, dict(sweep_meta, plays="right speaker"), sw48),
        write("talkA_L", A, None, {"A": A}, dict(mA, plays="talker A on left")),
        write("talkB_R", None, B, {"B": B}, dict(mB, plays="talker B on right")),
        write("talkAB_LR", A, B, {"A": A, "B": B}, dict(A=mA, B=mB, plays="A left + B right together")),
        write("noise_LR", N1, N4, {"N_L": N1, "N_R": N4}, dict(noise_meta, plays="bus noise on both")),
        write("talkA_noise", A + N1, N4, {"A": A, "N_L": N1, "N_R": N4},
              dict(noise_meta, A=mA, plays="A left + bus noise both, 0 dB at source")),
        write("talkC_R_quiet", None, C, {"C": C}, dict(mC, plays="talker C on right, 15 dB quieter")),
    ]
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for m in manifest:
        print(f"{m['name']:14s} sound {m['sound_s']:.2f} s  peak {m['peak_dbfs']:6.1f} dBFS  {m['plays']}")


if __name__ == "__main__":
    main()
