"""Endfire beam per-group level change under the three calibration variants (+ x_s0v2 blind self-calibration).
Same method as group_atten.py; shows what a calibration error does to the fixed beam on the REAL components."""
import json, os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw"), os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import realeval, realscenes as rs
from group_atten import apply, lvl
from lab import metrics2
from methods import x_combo_common as xc, x_combo_s1beam as sb
from methods._core import beams, common as cm, tracker
FS = 16000
res = {}
for sid in sys.argv[1:]:
    sc = rs.load_scene(sid); ctx = metrics2.SceneCtx(sc)
    x = np.asarray(sc["mix"], np.float64)
    tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] == "partner"]
    act = ctx.active(tids)
    for var in ("measured", "blind", "wrong4cm", "x_s0v2"):
        info = realeval.make_info("bus", "wrong4cm" if var == "x_s0v2" else var, None)
        info = sb.calibrated_info(x, FS, info) if var in ("measured", "x_s0v2") else info
        fe = tracker.front_end(x, FS, info, partners=False, talkers=False)
        cal = fe.calib
        a = cm.steer(cm.FREQS, float(cal.d_hat) / cm.C)
        w, _ = beams.mvdr_weights(a, beams.noise_coherence(cal, "measured"), beams.wng_floor("bus"))
        row = {"d_hat_cm": 100 * float(cal.d_hat), "tau0_smp": float(cal.tau0) * FS}
        for g, role in sc["groups"].items():
            ref, y = apply(sc["comp2"][g], cal, w)
            row[g] = lvl(y, act) - lvl(ref, act)
        _, ymix = apply(x, cal, w)
        b1 = metrics2.partner_fidelity(ctx, ymix)
        row["d_estoi_min"] = float(min(b1["estoi_A"] - b1["estoi_in_A"], b1["estoi_B"] - b1["estoi_in_B"]))
        res.setdefault(sid, {})[var] = row
        print(sid, var, " ".join(f"{k}:{v:+.2f}" for k, v in row.items()), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_atten_var.json"), "w"), indent=1)
