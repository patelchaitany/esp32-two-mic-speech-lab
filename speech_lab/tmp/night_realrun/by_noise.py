"""Bus L1 (measured): mean worst-partner dESTOI / dSNR / TA min split by device-noise condition (recorded vs quiet)."""
import json, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); LAB = os.path.dirname(os.path.dirname(HERE))
methods = sys.argv[1].split(",")
print("| method | recorded: dESTOI / dSNR / TA min (n) | quiet: dESTOI / dSNR / TA min (n) |")
print("|---|---|---|")
for m in methods:
    cells = []
    for cond in ("recorded", "quiet"):
        rows = []
        for p in glob.glob(os.path.join(LAB, "hw/realset/results/measured", m, f"bus__L1_*_{cond}.json")):
            r = json.load(open(p))
            if r.get("status") == "ok":
                rows.append(r["m"])
        if rows:
            cells.append(f"{np.mean([r['d_estoi_min'] for r in rows]):+.3f} / {np.mean([r['dsnr_active'] for r in rows]):+.1f} / "
                         f"{np.mean([r['ta_min'] for r in rows]):+.1f} ({len(rows)})")
        else:
            cells.append("–")
    print(f"| {m} | " + " | ".join(cells) + " |")
