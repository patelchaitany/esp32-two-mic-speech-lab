import os, sys, time
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw")]
import numpy as np, importlib
import realscenes as rs, realeval
names = sys.argv[1].split(",")
sid = sys.argv[2] if len(sys.argv) > 2 else "bus/L1_snr+0_recorded"
sc = rs.load_scene(sid)
x = np.asarray(sc["mix"], np.float64)[:, : 16000 * 10]
info = realeval.make_info(sc["meta"]["scenario"], "measured", "clean")
for n in names:
    m = importlib.import_module("methods." + n)
    t0 = time.perf_counter(); c0 = time.process_time()
    r = m.enhance(x, 16000, dict(info), return_stages=True)
    y = r["out"] if isinstance(r, dict) else r
    print(n, sid, "shape", np.shape(y), "finite", bool(np.all(np.isfinite(y))), "rms in/out %.4f/%.4f" % (np.std(x[0]), np.std(y)),
          "wall %.1fs cpu %.1fs" % (time.perf_counter() - t0, time.process_time() - c0), flush=True)
