"""Hypothesis-constrained mixture fits with d restricted to the user's stated 1-2 cm (H2: d in [1.4, 2.0] cm),
plus the H1-type diffuse+skew fit with d in [1.0, 2.0] cm, on the loud ambient and the pooled reverb windows."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, fit_diffuse, r3  # noqa: E402
from s2_fits import LOUD, all_responses, block_sums, fit_mix, smoothed_window  # noqa: E402

f, blocks = block_sums(LOUD)
Pab = sum(b[1] for b in blocks)
Paa = sum(b[2] for b in blocks)
Pbb = sum(b[3] for b in blocks)
m = (f >= 300) & (f <= 6500)
sets = {"ambient": (f[m][::2], (Pab / np.sqrt(Paa * Pbb))[m][::2])}
H = all_responses()
for (t0, t1) in [(80, 250), (10, 80)]:
    acc = None
    for k, (h, p) in H.items():
        ff, a, b, c = smoothed_window(h, t0, t1, p)
        acc = (a, b, c) if acc is None else (acc[0] + a, acc[1] + b, acc[2] + c)
    idx = np.where((ff >= 300) & (ff <= 6500))[0][::16]
    sets[f"reverb_{t0}_{t1}"] = (ff[idx], (acc[0] / np.sqrt(acc[1] * acc[2]))[idx])
res = {}
for name, (fs_, g) in sets.items():
    o = {"H1type_d_1.0-2.0cm(3p)": {k: r3(v) for k, v in fit_diffuse(fs_, g, "sinc", d_grid=np.arange(0.010, 0.0201, 0.0001)).items()},
         "H2_tau0=-0.5_d_1.4-2.0cm(4p)": fit_mix(fs_, g, -0.5, "sinc", 0.014, dmax=0.020),
         "H2_tau0=-0.5_d_1.0-2.0cm(4p)": fit_mix(fs_, g, -0.5, "sinc", 0.010, dmax=0.020),
         "H3_tau0=0_d_1.0-2.0cm(4p)": fit_mix(fs_, g, 0.0, "sinc", 0.010, dmax=0.020)}
    res[name] = o
    print(name, json.dumps(o), flush=True)
with open(os.path.join(HERE, "s2c_bounded.json"), "w") as fh:
    json.dump(res, fh, indent=1)
