"""Overnight calibration drift watch (no sound is played).

Every PERIOD seconds: record 12 s of room noise, fit the diffuse-field model (spacing, skew, rho) and the noise level per mic,
append one JSON line to hw/rec2/drift_watch.jsonl. Skips a cycle while hw/HW_BUSY exists (another hardware job owns the port).
Run: .venv/bin/python -I hw/drift_watch.py [period_s] [cycles]
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
from geomcheck import diffuse_fit, load  # noqa: E402

OUT = os.path.join(HERE, "rec2")
BUSY = os.path.join(HERE, "HW_BUSY")


def main():
    period = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for i in range(cycles):
        t0 = time.time()
        if os.path.exists(BUSY):
            print("busy, skipped", flush=True)
        else:
            try:
                fs, x, meta = play_record(None, 12)
                wav = os.path.join(OUT, f"drift_{i:02d}.wav")
                sf.write(wav, x, fs, subtype="FLOAT")
                y = load(wav)[fs:11 * fs]
                rec = {"i": i, "time": time.strftime("%H:%M:%S"), **diffuse_fit(wav),
                       "noise_dbfs": [round(float(10 * np.log10(np.mean(y[:, c] ** 2))), 1) for c in range(2)],
                       "load": os.getloadavg()[0]}
            except Exception as e:  # noqa: BLE001
                rec = {"i": i, "time": time.strftime("%H:%M:%S"), "error": repr(e)}
            with open(os.path.join(OUT, "drift_watch.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
