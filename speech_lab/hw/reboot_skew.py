"""Is the inter-channel skew the same after every ESP32 reboot? (no sound is played)

Each cycle: hard-reset the board with esptool (read-mac, then hard reset into the app), wait for it to boot,
record 12 s of room noise, fit the diffuse-field model (spacing d, skew, rho).
Run: .venv/bin/python -I hw/reboot_skew.py [cycles]   -> hw/rec2/reboot_skew.json
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import soundfile as sf  # noqa: E402
from esp32 import play_record  # noqa: E402
from geomcheck import diffuse_fit  # noqa: E402

ESPTOOL = os.path.expanduser("~/Library/Arduino15/packages/esp32/tools/esptool_py/5.3.1/esptool")
OUT = os.path.join(HERE, "rec2")


def port():
    for _ in range(40):
        p = sorted(glob.glob("/dev/cu.usbmodem*"))
        if p:
            return p[0]
        time.sleep(0.25)
    raise RuntimeError("ESP32 port did not come back")


def main():
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OUT, exist_ok=True)
    results = []
    for i in range(cycles):
        r = subprocess.run([ESPTOOL, "--chip", "esp32c3", "--port", port(), "--after", "hard-reset", "read-mac"],
                           capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0
        time.sleep(3.0)  # USB re-enumeration and app boot
        fs, x, meta = play_record(None, 12, port())
        wav = os.path.join(OUT, f"reboot_{i:02d}.wav")
        sf.write(wav, x, fs, subtype="FLOAT")
        fit = diffuse_fit(wav)
        res = {"cycle": i, "esptool_ok": ok, **fit, "time": time.strftime("%H:%M:%S")}
        results.append(res)
        print(json.dumps(res), flush=True)
        with open(os.path.join(OUT, "reboot_skew.json"), "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
