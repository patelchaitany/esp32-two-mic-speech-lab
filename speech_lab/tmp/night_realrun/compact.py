"""Compact tables for RESULTS_REAL.md from tmp/night_realrun/summary.json (written by analyze.py)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "summary.json")))


def f(v, fm):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fm)


def bus(key, methods=None, title=""):
    d = S.get(key, {})
    print(f"\n{title}\n")
    print("| method | n | wrst ESTOI out | Δwrst ESTOI all / −5 / 0 / +5 | ΔSNR act all / −5 / 0 / +5 (dB) | NA act | ΔSIR pax | pax ESTOI drop | TA min / mean | closure | RTF cpu / wall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in (methods or d.keys()):
        if m not in d:
            continue
        a = d[m].get("all", {})
        s = [d[m].get(c, {}) for c in ("-5", "+0", "+5")]
        e = " / ".join([f(a.get("d_estoi_min"), "+.3f")] + [f(x.get("d_estoi_min"), "+.3f") for x in s])
        q = " / ".join([f(a.get("dsnr_active"), "+.1f")] + [f(x.get("dsnr_active"), "+.1f") for x in s])
        print(f"| {m} | {a.get('n')} | {f(a.get('estoi_wrst'), '.3f')} | {e} | {q} | {f(a.get('na_active'), '+.1f')} | "
              f"{f(a.get('dsir_near'), '+.1f')} | {f(a.get('pass_drop_med'), '+.3f')} | {f(a.get('ta_min'), '+.1f')} / "
              f"{f(a.get('ta_mean'), '+.1f')} | {f(a.get('closure_db'), '.0f')} | {f(a.get('rtf_cpu'), '.3f')} / {f(a.get('rtf'), '.3f')} |")


def meet(key, methods=None, title=""):
    d = S.get(key, {})
    print(f"\n{title}\n")
    print("| method | T1 capture | ESTOI mean (Δ) | worst talker ESTOI (Δ) | quiet M3+M6 (Δ) | far MF1+MF2 (Δ) | level std in→out (dB) | range in→out (dB) | NA act | RTF cpu / wall |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in (methods or d.keys()):
        if m not in d:
            continue
        a = d[m].get("mean", {})
        print(f"| {m} | {f(a.get('capture_rate'), '.2f')} | {f(a.get('estoi_mean'), '.3f')} ({f(a.get('d_estoi_mean'), '+.3f')}) | "
              f"{f(a.get('estoi_wrst'), '.3f')} ({f(a.get('d_estoi_wrst'), '+.3f')}) | {f(a.get('q_estoi'), '.3f')} ({f(a.get('q_d'), '+.3f')}) | "
              f"{f(a.get('far_estoi'), '.3f')} ({f(a.get('far_d'), '+.3f')}) | {f(a.get('lvl_in_std'), '.1f')} → {f(a.get('lvl_std'), '.1f')} | "
              f"{f(a.get('lvl_in_range'), '.1f')} → {f(a.get('lvl_range'), '.1f')} | {f(a.get('na_active'), '+.1f')} | "
              f"{f(a.get('rtf_cpu'), '.3f')} / {f(a.get('rtf'), '.3f')} |")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "bus"):
        bus("bus_measured_L1", title="Bus, measured calibration, L1 (6 scenes)")
        for v in ("blind", "wrong4cm"):
            bus(f"bus_{v}", title=f"Bus, variant {v} (L1 recorded, 3 scenes)")
        bus("bus_measured_rec3", title="Bus, measured, same 3 scenes")
        bus("bus_measured_L23", title="Bus, measured, L2/L3 scenes")
    if which in ("all", "meeting"):
        meet("meeting_measured", title="Meeting, measured (mean of 3 scenes)")
        for v in ("blind", "wrong4cm"):
            meet(f"meeting_{v}", title=f"Meeting, variant {v}")
