"""Play each test sound through the Mac speakers while the ESP32 records 12 s, and save the recordings.

Run: .venv/bin/python -I hw/session.py [name ...]   (no names = full session; 'ambient' = no sound)
Writes hw/rec/<name>.wav (float32, 16 kHz, ch0 = Mic A, ch1 = Mic B) and hw/rec/<name>.json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from esp32 import play_record  # noqa: E402

SESSION = ["ambient", "sweep_L", "sweep_R", "sweep_L2", "talkA_L", "talkB_R", "talkAB_LR", "noise_LR",
           "talkA_noise", "talkC_R_quiet"]
STIM_OF = {"sweep_L2": "sweep_L"}


def dbfs(v):
    return round(float(20 * np.log10(max(v, 1e-12))), 1)


def run(name, seconds=12):
    stim = None if name == "ambient" else os.path.join(HERE, "stim", STIM_OF.get(name, name) + ".wav")
    fs, x, meta = play_record(stim, seconds)
    sf.write(os.path.join(HERE, "rec", name + ".wav"), x, fs, subtype="FLOAT")
    xc = x - x.mean(0)
    meta.update(name=name, stim=STIM_OF.get(name, name) if stim else None, wall_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                peak_dbfs=[dbfs(np.abs(xc[:, c]).max()) for c in range(2)],
                rms_dbfs=[dbfs(np.sqrt(np.mean(xc[:, c] ** 2))) for c in range(2)],
                dc=[float(x[:, c].mean()) for c in range(2)])
    with open(os.path.join(HERE, "rec", name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{name:14s} peak A/B {meta['peak_dbfs']} dBFS  rms A/B {meta['rms_dbfs']} dBFS  "
          f"stream {meta['stream_s']:.2f} s  play+{meta.get('play_after_rec_s', 0) * 1000:.0f} ms", flush=True)
    return meta


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "rec"), exist_ok=True)
    for i, n in enumerate(sys.argv[1:] or SESSION):
        if i:
            time.sleep(1.5)
        run(n)
