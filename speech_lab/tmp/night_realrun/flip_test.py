"""'Device rotated by 180 deg' test on REAL data: swap the two channels (partners now at the B-side endfire, tau = -d/c)
and swap the calibration accordingly (tau0 -> -tau0, B gain correction -> -correction). Then the fixed beam toward
+d/c points AWAY from the partners; toward -d/c points at them. Linear per-group levels + B1 ESTOI as group_atten.py."""
import json, os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw"), os.path.dirname(os.path.abspath(__file__))]
import numpy as np
import realeval, realscenes as rs
from group_atten import apply, lvl
from lab import metrics2
from methods import x_combo_common as xc
from methods._core import beams, common as cm, tracker
FS = 16000
res = {}
for sid in sys.argv[1:]:
    sc = rs.load_scene(sid); ctx = metrics2.SceneCtx(sc)
    tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] == "partner"]
    act = ctx.active(tids)
    for flip in (False, True):
        x = np.asarray(sc["mix"], np.float64)
        info = realeval.make_info("bus", "measured", None)
        if flip:
            x = x[::-1].copy()
            cj = dict(info["calib_json"]); cj["tau0_s"] = -cj["tau0_s"]; cj["gain_db"] = [-g for g in cj["gain_db"]]
            info["calib_json"] = cj
        info = xc.gn_fix(x, FS, info)
        fe = tracker.front_end(x, FS, info, partners=True, talkers=False)
        cal = fe.calib
        for side in (+1.0, -1.0):
            a = cm.steer(cm.FREQS, side * float(cal.d_hat) / cm.C)
            w, _ = beams.mvdr_weights(a, beams.noise_coherence(cal, "measured"), beams.wng_floor("bus"))
            row = {"S1_tauA": float(fe.track.tau_A) * FS, "S1_tauB": float(fe.track.tau_B) * FS}
            for g, role in sc["groups"].items():
                c2 = np.asarray(sc["comp2"][g], np.float64)
                c2 = c2[::-1].copy() if flip else c2
                ref, y = apply(c2, cal, w)
                row[g] = lvl(y, act) - lvl(ref, act)
            _, ymix = apply(x, cal, w)
            b1 = metrics2.partner_fidelity(ctx, ymix)
            row["d_estoi_min"] = float(min(b1["estoi_A"] - b1["estoi_in_A"], b1["estoi_B"] - b1["estoi_in_B"]))
            key = f"{'flipped' if flip else 'normal'}_steer{'+' if side > 0 else '-'}dc"
            res.setdefault(sid, {})[key] = row
            print(sid, key, " ".join(f"{k}:{v:+.2f}" for k, v in row.items()), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flip_test.json"), "w"), indent=1)
