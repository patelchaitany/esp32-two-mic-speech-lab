#!/usr/bin/env python
"""Cross-check of the vectorised two-talker beam in physics_b1.py against the ORIGINAL design script
(scratchpad revise/physics_check_v2.py, imported read-only, its __main__ block is not run) at the new small spacings.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$LAB/.venv/bin/python" -I -B crosscheck_v2.py <path-to-revise-folder>
Prints AI-DI from the original loop implementation next to physics.json values.  ~20-60 CPU-s.
"""
import json
import os
import sys
import time

t0 = time.process_time()
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import physics_check_v2 as v2  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(here, "physics.json")))
out = {}
for d in (0.010, 0.012):
    for kind in ("3d", "2d"):
        orig = v2.two_talker(d, -10, kind, v2.a_c)[0]
        mine = R["two_talker"][f"{d*100:.1f}cm"]["wng-10"][kind]["aidi"]
        out[f"{d*100:.1f}cm_{kind}"] = dict(original=round(orig, 3), vectorised=round(mine, 3), diff=round(mine - orig, 3))
        print(f"d={d*100:.1f} cm {kind}: original v2 {orig:.3f} dB | physics_b1 {mine:.3f} dB | diff {mine-orig:+.3f}")
json.dump(out, open(os.path.join(here, "crosscheck_v2.json"), "w"), indent=1)
print(f"cpu {time.process_time() - t0:.1f} s")
