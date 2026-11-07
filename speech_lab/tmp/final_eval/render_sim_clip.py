"""Render listening clips for one held-out simulated scene (same input and info as lab.bench2), cut a window, write 16-bit
mono WAVs peak-normalised to -1 dBFS. Usage: render_sim_clip.py SCENE_ID METHOD[,METHOD...] OUT_DIR [--window auto|full]"""
import json, os, sys
LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, LAB)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np, soundfile as sf
from lab import scene2, bench2
sid, methods, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
mode = sys.argv[5] if len(sys.argv) > 5 else "full"
os.makedirs(out, exist_ok=True)
sc = scene2.load_scene(sid, corpus="libri")
fs = sc["fs"]; T = sc["mix"].shape[1]; scen = sc["meta"]["scenario"]
a, b = 0, T
note = "whole scene"
if mode == "auto":  # 30 s window with the most distinct talkers, at least one <= 2 m and one >= 4 m; earliest on ties
    talk = sc["meta"]["talkers"]; best = None
    for s0 in range(0, max(1, T - 30 * fs), fs):
        ids = {t for (s, e, t) in sc["segs"] if s < s0 + 30 * fs and e > s0}
        d = [talk[t]["dist_m"] for t in ids if t in talk]
        if not d or min(d) > 2.0 or max(d) < 4.0:
            continue
        key = (len(ids), -s0)
        if best is None or key > best[0]:
            best = (key, s0, sorted(round(x, 1) for x in d))
    if best:
        a, b = best[1], best[1] + 30 * fs
        note = f"30 s from {best[1] / fs:.0f} s: {best[0][0]} talkers at {best[2]} m"
def write(name, y):
    y = np.asarray(y, np.float64)[a:b]
    pk = float(np.max(np.abs(y))) or 1.0
    sf.write(os.path.join(out, name), np.clip(y * 10 ** (-1 / 20) / pk, -1, 1), fs, subtype="PCM_16")
base = sid.split("/")[-1]
write(f"{base}__input_micA.wav", np.array(sc["mix"])[0])
for m in methods:
    mod, _, accepts = bench2.load_method(m)
    info = {"scenario": scen, "spacing_m": 0.04}
    if bench2.is_oracle(m):
        info["oracle"] = bench2.oracle_view(sc)
    r = mod.enhance(np.array(sc["mix"], np.float32), fs, info)
    y = r["out"] if isinstance(r, dict) else r
    write(f"{base}__{m}.wav", y)
print(json.dumps({"scene": sid, "window": note, "a_s": a / fs, "b_s": b / fs}))
