import os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw")]
import numpy as np
import realscenes as rs, realeval
from methods import x_combo_s1beam as sb
from methods._core import tracker
for sid in sys.argv[1:]:
    sc = rs.load_scene(sid)
    x = np.asarray(sc["mix"], np.float64)
    for var in ("measured", "wrong4cm"):
        info = sb.calibrated_info(x, 16000, realeval.make_info("bus", var, None))
        fe = tracker.front_end(x, 16000, info, partners=True, talkers=False)
        t = fe.track
        print(sid, var, "tauA %.2f tauB %.2f" % (t.tau_A * 16000, t.tau_B * 16000),
              "massA %.0f massB %.0f" % (np.nansum(t.p_A), np.nansum(t.p_B)), "why", t.inseparable_why, flush=True)
