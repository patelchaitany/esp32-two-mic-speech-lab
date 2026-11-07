"""Meeting version of group_atten.py: per-group level change of the fixed endfire MVDR (meeting floor -3 dB, and the
bus floors for comparison) over talker-active samples; talkers and device noise are REAL, omeeting is synthetic."""
import json, os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw"), os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import realeval, realscenes as rs
from group_atten import lvl
from lab import metrics2
from methods import x_combo_common as xc
from methods._core import beams, calib, common as cm, tracker
FS = 16000
res = {}
for sid in sys.argv[1:]:
    sc = rs.load_scene(sid); ctx = metrics2.SceneCtx(sc)
    x = np.asarray(sc["mix"], np.float64)
    info = xc.gn_fix(x, FS, realeval.make_info("meeting", "measured", None))
    fe = tracker.front_end(x, FS, info, partners=False, talkers=False)
    cal = fe.calib
    tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] == "talker"]
    act = ctx.active(tids)
    for fl in ("meeting", "bus"):
        a = cm.steer(cm.FREQS, float(cal.d_hat) / cm.C)
        w, _ = beams.mvdr_weights(a, beams.noise_coherence(cal, "measured"), beams.wng_floor(fl))
        row = {}
        for g, role in sc["groups"].items():
            x2 = np.asarray(sc["comp2"][g], np.float64)
            xh = calib.prefilter(x2, "meeting")
            X = calib.apply(cm.stft(xh), cal)
            y = cm.istft(np.conj(w[:, 0])[:, None] * X[0] + np.conj(w[:, 1])[:, None] * X[1], x2.shape[-1])
            tk = [t for t, gg in sc["talker_group"].items() if gg == g]
            m = ctx.active(tk) if tk else act
            row[g] = lvl(y, m) - lvl(xh[0], m)
        res.setdefault(sid, {})[fl] = row
        print(sid, fl, " ".join(f"{g}:{v:+.1f}" for g, v in row.items()), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_atten_meet.json"), "w"), indent=1)
