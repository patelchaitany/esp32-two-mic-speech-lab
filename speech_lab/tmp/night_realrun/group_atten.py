"""Per-group level change of LINEAR fixed beams on the real bus scenes (real:run diagnostic).
For a fixed calibration (computed once on the mix, as the method does) the beam is linear, so it is applied to each
group's 2-channel component separately: partners (REAL recordings), device noise (REAL fan + mic noise), passengers
and bus noise (simulated). Level = in-band (150-7000 Hz) mean square at Mic A in, beam output out, over the samples
where a partner is active (and, for the device noise, also over the turn gaps). Output: JSON + a table.
  .venv/bin/python -I tmp/night_realrun/group_atten.py bus/L1_snr+0_recorded ...
"""
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw")]
import numpy as np  # noqa: E402

import realeval  # noqa: E402
import realscenes as rs  # noqa: E402
from lab import metrics2  # noqa: E402
from methods import x_combo_common as xc  # noqa: E402
from methods._core import beams, calib, common as cm, tracker  # noqa: E402

FS = 16000


def lvl(y, mask):
    yb = metrics2.inband(np.asarray(y, np.float64))
    return float(10 * np.log10(np.mean(yb[mask] ** 2) + 1e-20))


def weights(x, info, kind):
    info = xc.gn_fix(x, FS, info)
    fe = tracker.front_end(x, FS, info, partners=False, talkers=False)
    cal = fe.calib
    if kind == "avg":
        w = np.tile(np.array([0.5, 0.5], complex), (cm.N_BINS, 1))
        return cal, w
    if kind.startswith("endfire"):
        frac = float(kind.split("@")[1]) if "@" in kind else 1.0
        a = cm.steer(cm.FREQS, frac * float(cal.d_hat) / cm.C)
        w, _ = beams.mvdr_weights(a, beams.noise_coherence(cal, "measured"), beams.wng_floor("bus"))
        return cal, w
    if kind == "broadside":
        a = cm.steer(cm.FREQS, 0.0)
        w, _ = beams.mvdr_weights(a, beams.noise_coherence(cal, "measured"), beams.wng_floor("bus"))
        return cal, w
    raise ValueError(kind)


def apply(x2, cal, w, scen="bus"):
    xh = calib.prefilter(np.asarray(x2, np.float64), scen)
    X = calib.apply(cm.stft(xh), cal)
    Y = np.conj(w[:, 0])[:, None] * X[0] + np.conj(w[:, 1])[:, None] * X[1]
    return xh[0], cm.istft(Y, x2.shape[-1])


def main(scenes, kinds=("avg", "broadside", "endfire@0.5", "endfire")):
    res = {}
    for sid in scenes:
        sc = rs.load_scene(sid)
        ctx = metrics2.SceneCtx(sc)
        x = np.asarray(sc["mix"], np.float64)
        info = realeval.make_info("bus", "measured", None)
        tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] == "partner"]
        act = ctx.active(tids)
        gap = ~act
        gap[: FS // 4] = False
        for kind in kinds:
            cal, w = weights(x, info, kind)
            row = {}
            for g, role in sc["groups"].items():
                ref, y = apply(sc["comp2"][g], cal, w)
                row[g] = {"role": role, "dL_act": lvl(y, act) - lvl(ref, act)}
                if role == "device_noise":
                    row[g]["dL_gap"] = lvl(y, gap) - lvl(ref, gap)
            res.setdefault(sid, {})[kind] = row
            print(sid, kind, " ".join(f"{g}:{v['dL_act']:+.1f}" for g, v in row.items()), flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "group_atten.json")
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update(res)
    json.dump(old, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
