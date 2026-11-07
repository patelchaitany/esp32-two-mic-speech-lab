"""Re-estimate skew/spacing/gain with x_s0v2 on every board noise take (drift watch, resets, ambient). Prints JSON lines."""
import glob, json, os, sys
LAB = sys.argv[1]
sys.path.insert(0, LAB)
import numpy as np, soundfile as sf
from methods import x_s0v2
files = sorted(glob.glob(os.path.join(LAB, "hw/rec2/drift_*.wav"))) + sorted(glob.glob(os.path.join(LAB, "hw/rec2/reboot_*.wav"))) \
    + [os.path.join(LAB, p) for p in ("hw/rec/ambient.wav", "hw/rec2/s2_ambient.wav", "hw/rec2/s3_ambient_after.wav", "hw/rec2/s4_ambient_0330.wav")]
for f in files:
    if not os.path.exists(f):
        continue
    x, fs = sf.read(f, always_2d=True)
    x = x.T.astype(np.float64)[:, fs: 11 * fs]
    cj = x_s0v2.estimate(x, fs, {"scenario": "bus", "spacing_m": 0.0125})
    g = np.array(cj.get("gain_db", [np.nan]))
    print(json.dumps({"file": os.path.basename(f), "mtime": os.path.getmtime(f), "tau_samples": round(cj["tau0_s"] * fs, 4),
                      "d_cm": round(100 * cj["spacing_m"], 3), "gain_med_db": round(float(np.nanmedian(g)), 2)}), flush=True)
