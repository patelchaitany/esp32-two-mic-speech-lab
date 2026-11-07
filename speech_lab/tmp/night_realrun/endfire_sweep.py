"""Endfire-beam design sweep on the real bus scenes (real:run diagnostic, linear => exact per-group levels).
Grid: steering tau = frac * d_hat/c (frac 0, 0.5, 0.75, 1.0, 1.25) x WNG floor (bus default -15/-10/-6 dB, flat -3, -6,
-10, -15 dB). Per config: partner level change (REAL partners), device-noise change (REAL fans + mic noise), passenger
and bus-noise change (simulated), all over partner-active samples, and B1 ESTOI of the beam output of the mix.
  .venv/bin/python -I tmp/night_realrun/endfire_sweep.py bus/L1_snr+0_recorded ...
"""
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw"), os.path.dirname(os.path.abspath(__file__))]
import numpy as np  # noqa: E402

import realeval  # noqa: E402
import realscenes as rs  # noqa: E402
from group_atten import apply, lvl  # noqa: E402
from lab import metrics2  # noqa: E402
from methods import x_combo_common as xc  # noqa: E402
from methods._core import beams, common as cm, tracker  # noqa: E402

FS = 16000
FLOORS = {"bus": None, "-3": -3.0, "-6": -6.0, "-10": -10.0, "-15": -15.0}
FRACS = (0.0, 0.5, 0.75, 1.0, 1.25)


def main(scenes):
    res = {}
    for sid in scenes:
        sc = rs.load_scene(sid)
        ctx = metrics2.SceneCtx(sc)
        x = np.asarray(sc["mix"], np.float64)
        info = xc.gn_fix(x, FS, realeval.make_info("bus", "measured", None))
        fe = tracker.front_end(x, FS, info, partners=False, talkers=False)
        cal = fe.calib
        g = beams.noise_coherence(cal, "measured")
        tids = [t for t, gg in sc["talker_group"].items() if sc["groups"][gg] == "partner"]
        act = ctx.active(tids)
        comps = {}
        for grp, role in sc["groups"].items():
            comps[grp] = (role, sc["comp2"][grp])
        for fl_name, fl in FLOORS.items():
            floor = beams.wng_floor("bus") if fl is None else np.full(cm.N_BINS, fl)
            for frac in FRACS:
                a = cm.steer(cm.FREQS, frac * float(cal.d_hat) / cm.C)
                w, _ = beams.mvdr_weights(a, g, floor)
                row = {}
                for grp, (role, x2) in comps.items():
                    ref, y = apply(x2, cal, w)
                    row[grp] = lvl(y, act) - lvl(ref, act)
                _, ymix = apply(x, cal, w)
                b1 = metrics2.partner_fidelity(ctx, ymix)
                key = f"floor{fl_name}_frac{frac}"
                row["estoi_A"], row["estoi_B"] = (b1 or {}).get("estoi_A"), (b1 or {}).get("estoi_B")
                row["estoi_in_A"], row["estoi_in_B"] = (b1 or {}).get("estoi_in_A"), (b1 or {}).get("estoi_in_B")
                res.setdefault(sid, {})[key] = {k: (None if v is None else float(v)) for k, v in row.items()}
                print(sid, key, " ".join(f"{k}:{v:+.2f}" for k, v in row.items() if v is not None), flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "endfire_sweep.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res)
    json.dump(old, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
