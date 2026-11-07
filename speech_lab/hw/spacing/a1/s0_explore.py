"""Exploration: onsets, ambient-only stretches of every recording, spectra, coherence and phase slope overview."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FS, HERE, RECS, coherence, fit_diffuse, load, onset_of, phase_slope, r3, stim)  # noqa: E402

out = {}
segs = {}
for name in RECS:
    x = load(name)
    if name == "ambient":
        segs[name] = [(int(0.3 * FS), int(11.9 * FS))]
        out[name] = {"onset_s": None}
        continue
    s = stim(name)
    on = onset_of(name)              # index where reference t=0 is
    n_ref = max(len(v) for v in s.values())
    start_sound = on + int(0.25 * FS)
    end_sound = on + n_ref
    # ambient-only: before the sound (minus 20 ms), and after the sound + 0.8 s of reverb tail
    pre = (int(0.05 * FS), start_sound - int(0.02 * FS))
    post = (end_sound + int(0.8 * FS), len(x) - int(0.1 * FS))
    segs[name] = [p for p in (pre, post) if p[1] - p[0] > int(0.25 * FS)]
    out[name] = {"ref_t0_s": r3(on / FS), "sound_start_s": r3(start_sound / FS), "sound_end_s": r3(end_sound / FS),
                 "ambient_segments_s": [[r3(a / FS), r3(b / FS)] for a, b in segs[name]]}

# levels in the ambient-only stretches vs the ambient recording
for name in RECS:
    x = load(name)
    xs = np.concatenate([x[a:b] for a, b in segs[name]])
    f, g, Pab, Paa, Pbb = coherence(xs, 512, 256)
    lv = {}
    for lo, hi in [(150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 7000), (7000, 7800)]:
        m = (f >= lo) & (f < hi)
        lv[f"{lo}-{hi}"] = [r3(10 * np.log10(np.sum(Paa[m]) / (FS * 512 * 0.375) * 2 + 1e-30)),
                            r3(np.mean(np.abs(g[m]) ** 2))]
    t1, t2, icpt = phase_slope(f, g, lo=300, hi=6500)
    fit = fit_diffuse(f[(f >= 150) & (f <= 7000)], g[(f >= 150) & (f <= 7000)], "sinc")
    out[name].update(ambient_s=r3(len(xs) / FS), band_level_dB_and_msc=lv, slope_300_6500=r3(t1),
                     slope_ls=r3(t2), intercept_deg=r3(icpt), sinc_fit_150_7000=fit)
    print(name, out[name]["ambient_s"], "slope", r3(t1), r3(t2), "icpt", r3(icpt), "fit", {k: r3(v) for k, v in fit.items()})

# fine spectrum of the ambient recording: PSD A, B (dB re full scale^2/Hz) and MSC/phase every 250 Hz
x = load("ambient")[int(0.3 * FS):int(11.9 * FS)]
f, g, Pab, Paa, Pbb = coherence(x, 1024, 256)
scale = 2 / (FS * np.sum(np.hanning(1026)[1:-1] ** 2))
rows = []
for fc in np.arange(100, 8000, 250):
    m = (f >= fc - 125) & (f < fc + 125)
    rows.append([int(fc), r3(10 * np.log10(np.mean(Paa[m]) * scale)), r3(10 * np.log10(np.mean(Pbb[m]) * scale)),
                 r3(np.abs(np.mean(Pab[m])) / np.sqrt(np.mean(Paa[m]) * np.mean(Pbb[m]))),
                 r3(np.degrees(np.angle(np.mean(g[m]))))])
out["ambient_fine"] = {"cols": ["f_center", "PSD_A_dB/Hz", "PSD_B_dB/Hz", "|coh|", "phase_deg"], "rows": rows}
for r in rows:
    print(r)
with open(os.path.join(HERE, "s0_explore.json"), "w") as fh:
    json.dump({"segments": {k: [[int(a), int(b)] for a, b in v] for k, v in segs.items()}, "info": out}, fh, indent=1)
