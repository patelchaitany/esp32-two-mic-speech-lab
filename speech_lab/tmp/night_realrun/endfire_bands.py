"""Per-octave level change of the endfire beam (bus floors, toward +d/c) on the REAL partners and REAL device noise."""
import json, os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw"), os.path.dirname(os.path.abspath(__file__))]
import numpy as np
from scipy.signal import butter, sosfiltfilt
import realeval, realscenes as rs
from group_atten import apply
from lab import metrics2
from methods import x_combo_common as xc
from methods._core import beams, common as cm, tracker
FS = 16000
BANDS = [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]
sid = sys.argv[1]
sc = rs.load_scene(sid); ctx = metrics2.SceneCtx(sc)
x = np.asarray(sc["mix"], np.float64)
info = xc.gn_fix(x, FS, realeval.make_info("bus", "measured", None))
fe = tracker.front_end(x, FS, info, partners=False, talkers=False)
cal = fe.calib
a = cm.steer(cm.FREQS, float(cal.d_hat) / cm.C)
w, _ = beams.mvdr_weights(a, beams.noise_coherence(cal, "measured"), beams.wng_floor("bus"))
tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] == "partner"]
act = ctx.active(tids)
out = {}
for g in ("partner_A", "partner_B", "device", "bus_noise", "near_1"):
    ref, y = apply(sc["comp2"][g], cal, w)
    row = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        r, o = sosfiltfilt(sos, ref)[act], sosfiltfilt(sos, y)[act]
        row.append(float(10 * np.log10(np.mean(o ** 2) / np.mean(r ** 2))))
    out[g] = row
    print(f"{g:10s}", " ".join(f"{lo}-{hi}:{v:+.1f}" for (lo, hi), v in zip(BANDS, row)), flush=True)
wng = beams.wng_db(w, a)
print("WNG dB at 250/500/1k/2k/4k/6k Hz:", " ".join(f"{wng[int(f / 31.25)]:+.1f}" for f in (250, 500, 1000, 2000, 4000, 6000)))
print("DF dB  at 250/500/1k/2k/4k/6k Hz:", " ".join(f"{beams.df_db(w, a, beams.noise_coherence(cal, 'measured'))[int(f / 31.25)]:+.1f}" for f in (250, 500, 1000, 2000, 4000, 6000)))
json.dump({"scene": sid, "bands": BANDS, "dL": out, "wng_db": [float(v) for v in wng]}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "endfire_bands.json"), "w"), indent=1)
