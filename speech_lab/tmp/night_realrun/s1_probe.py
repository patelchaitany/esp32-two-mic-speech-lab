import os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw")]
import numpy as np
import realscenes as rs, realeval
from methods import x_combo_common as xc, x_s0v2
from methods._core import tracker, common as cm
for sid in sys.argv[1:]:
    sc = rs.load_scene(sid)
    x = np.asarray(sc["mix"], np.float64)
    for var in ("measured", "measured+gnfix", "x_s0v2"):
        info = realeval.make_info(sc["meta"]["scenario"], "measured", None)
        if var == "measured+gnfix":
            info = xc.gn_fix(x, 16000, info)
        if var == "x_s0v2":
            info.pop("calib_json"); info["calib_json"] = x_s0v2.estimate(x, 16000, info)
        fe = tracker.front_end(x, 16000, info, partners=True, talkers=False)
        trk, cal = fe.track, fe.calib
        tA = np.asarray(trk.tau_A, float) * 16000; tB = np.asarray(trk.tau_B, float) * 16000
        print(sid, var, "d_hat %.4f tau0 %.3f smp" % (cal.d_hat, cal.tau0 * 16000),
              "gn@2k/4k %.2f/%.2f" % (cal.gamma_n[64], cal.gamma_n[128]),
              "tau_A med %.3f tau_B med %.3f smp" % (np.nanmedian(tA), np.nanmedian(tB)),
              "d/c %.3f" % (cal.d_hat / cm.C * 16000), "insep", getattr(trk, "inseparable_why", None), flush=True)
