"""Tables for hw/realset/RESULTS_REAL.md from hw/realset/results/<variant>/<method>/*.json (real:run pass).

  .venv/bin/python -I tmp/night_realrun/analyze.py [--out tmp/night_realrun/tables.md] [--json tmp/night_realrun/summary.json]
Means are plain means over the listed scenes (the scenes share the same partner audio: no CIs). 'vs m0' columns are
paired per-scene differences against m0_mica_hpf (mean [min, max] over scenes, and wins = scenes where the method is
better by more than 0.005 ESTOI / 0.25 dB).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(LAB, "hw", "realset", "results")

QUIET_TALKERS = ("M3", "M6")         # -12 and -15 dB talkers
FAR_TALKERS = ("MF1", "MF2")          # 4 m emulated far talkers


def load(variant):
    rows = {}
    for d in sorted(glob.glob(os.path.join(RES, variant, "*"))):
        name = os.path.basename(d)
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(p))
            except Exception:
                continue
            rows.setdefault(name, {})[r["scene"]] = r
    return rows


def f(v, fmt):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return format(v, fmt)


def mean(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def bus_scene_ok(r):
    return r.get("status") == "ok" and r.get("scenario") == "bus"


BUS_KEYS = [("estoi_wrst", "wrst ESTOI", ".3f"), ("d_estoi_min", "Δwrst ESTOI", "+.3f"), ("d_sdr_min", "ΔSDR wrst", "+.1f"),
            ("dsnr_active", "ΔSNR act", "+.1f"), ("na_active", "NA act", "+.1f"), ("na_gap_pre", "NA gap", "+.1f"),
            ("dsir_near", "ΔSIR pax", "+.1f"), ("dsir_rest", "ΔSIR rest", "+.1f"), ("pass_drop_med", "pax ESTOI drop", "+.3f"),
            ("ta_min", "TA min", "+.1f"), ("ta_mean", "TA mean", "+.1f"), ("delay", "delay", "+.0f"),
            ("closure_db", "closure", ".0f"), ("rtf", "RTF wall", ".3f"), ("rtf_cpu", "RTF cpu", ".3f")]


def bus_table(rows, methods, scenes, title, by_snr=True):
    out = [f"\n### {title}\n", f"Scenes ({len(scenes)}): " + ", ".join(s.split('/')[1] for s in scenes) + "\n",
           "| method | SNR | n | " + " | ".join(k[1] for k in BUS_KEYS) + " |",
           "|---" * (len(BUS_KEYS) + 3) + "|"]
    summ = {}
    for m in methods:
        rs = [rows.get(m, {}).get(s) for s in scenes]
        rs = [r for r in rs if r is not None]
        err = [r for r in rs if r.get("status") != "ok"]
        ok = [r for r in rs if bus_scene_ok(r)]
        if err:
            out.append(f"| {m} | ERROR | {len(err)} | " + err[0].get("error", "")[:100] + " |")
        if not ok:
            continue
        groups = [("all", ok)]
        if by_snr:
            for snr in (-5, 0, 5):
                g = [r for r in ok if round(r["card"]["snr_db"]) == snr]
                if g:
                    groups.append((f"{snr:+d}", g))
        for cond, g in groups:
            vals = [mean([r["m"].get(k) for r in g]) for k, _, _ in BUS_KEYS]
            out.append(f"| {m} | {cond} | {len(g)} | " + " | ".join(f(v, fm) for v, (_, _, fm) in zip(vals, BUS_KEYS)) + " |")
            summ.setdefault(m, {})[cond] = {k: v for (k, _, _), v in zip(BUS_KEYS, vals)} | {"n": len(g)}
    return out, summ


def paired(rows, methods, scenes, ref, keys=(("d_estoi_min", 0.005, "+.3f"), ("dsnr_active", 0.25, "+.1f"))):
    out = [f"\nPaired per-scene differences against {ref} (mean [min, max]; wins/losses = better/worse by more than "
           f"{keys[0][1]} ESTOI or {keys[1][1]} dB):\n",
           "| method | n | " + " | ".join(f"Δ{k} vs {ref}" for k, _, _ in keys) + " | wins/losses (ESTOI) | wins/losses (ΔSNR) |",
           "|---" * (2 + 2 * len(keys)) + "|"]
    for m in methods:
        d = {k: [] for k, _, _ in keys}
        for s in scenes:
            a, b = rows.get(m, {}).get(s), rows.get(ref, {}).get(s)
            if a is None or b is None or a.get("status") != "ok" or b.get("status") != "ok":
                continue
            for k, _, _ in keys:
                va, vb = a["m"].get(k), b["m"].get(k)
                if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
                    d[k].append(va - vb)
        if not d[keys[0][0]]:
            continue
        cells, wl = [], []
        for k, thr, fm in keys:
            v = np.array(d[k])
            cells.append(f"{format(v.mean(), fm)} [{format(v.min(), fm)}, {format(v.max(), fm)}]")
            wl.append(f"{int((v > thr).sum())}/{int((v < -thr).sum())}")
        out.append(f"| {m} | {len(d[keys[0][0]])} | " + " | ".join(cells) + " | " + " | ".join(wl) + " |")
    return out


MEET_KEYS = [("capture_rate", "T1 capture", ".2f"), ("estoi_mean", "ESTOI mean", ".3f"), ("d_estoi_mean", "ΔESTOI mean", "+.3f"),
             ("estoi_wrst", "wrst ESTOI", ".3f"), ("d_estoi_wrst", "Δwrst", "+.3f"), ("q_estoi", "quiet M3/M6 ESTOI", ".3f"),
             ("q_d", "Δquiet", "+.3f"), ("far_estoi", "far MF1/2 ESTOI", ".3f"), ("far_d", "Δfar", "+.3f"),
             ("lvl_in_std", "lvl std in", ".1f"), ("lvl_std", "lvl std out", ".1f"), ("lvl_in_range", "range in", ".1f"),
             ("lvl_range", "range out", ".1f"), ("na_active", "NA act", "+.1f"), ("na_gap_pre", "NA gap", "+.1f"),
             ("ta_wrst", "TA wrst", "+.1f"), ("pump_std", "pump std", ".1f"), ("rtf", "RTF wall", ".3f"), ("rtf_cpu", "RTF cpu", ".3f")]


def meet_vals(r, rin):
    m = dict(r["m"])
    tk = r.get("talkers", {})
    q = [tk[t] for t in QUIET_TALKERS if t in tk]
    fa = [tk[t] for t in FAR_TALKERS if t in tk]
    m["q_estoi"] = mean([t.get("estoi") for t in q])
    m["q_d"] = mean([t.get("d_estoi") for t in q])
    m["far_estoi"] = mean([t.get("estoi") for t in fa])
    m["far_d"] = mean([t.get("d_estoi") for t in fa])
    if rin is not None:
        lv = [t.get("level_out_db") for t in rin.get("talkers", {}).values() if t.get("level_out_db") is not None]
        m["lvl_in_std"] = float(np.std(lv)) if lv else float("nan")
        m["lvl_in_range"] = float(np.ptp(lv)) if lv else float("nan")
    return m


def meet_table(rows, methods, scenes, title):
    out = [f"\n### {title}\n", "| method | cond | " + " | ".join(k[1] for k in MEET_KEYS) + " |",
           "|---" * (len(MEET_KEYS) + 2) + "|"]
    summ = {}
    for m in methods:
        rs = [(s, rows.get(m, {}).get(s)) for s in scenes]
        rs = [(s, r) for s, r in rs if r is not None]
        err = [r for _, r in rs if r.get("status") != "ok"]
        if err:
            out.append(f"| {m} | ERROR {len(err)} | " + err[0].get("error", "")[:100] + " |")
        ok = [(s, r) for s, r in rs if r.get("status") == "ok"]
        if not ok:
            continue
        per = []
        for s, r in ok:
            v = meet_vals(r, rows.get("mic_a", {}).get(s))
            per.append((s.split("/")[1], v))
        for cond, v in [("mean", None)] + per:
            if v is None:
                vals = [mean([p[1].get(k) for p in per]) for k, _, _ in MEET_KEYS]
            else:
                vals = [v.get(k) for k, _, _ in MEET_KEYS]
            out.append(f"| {m} | {cond} | " + " | ".join(f(x, fm) for x, (_, _, fm) in zip(vals, MEET_KEYS)) + " |")
            summ.setdefault(m, {})[cond] = {k: x for (k, _, _), x in zip(MEET_KEYS, vals)}
    return out, summ


def per_talker(rows, methods, scene):
    tids = ["M1", "M2", "M3", "M4", "M5", "M6", "MF1", "MF2"]
    out = [f"\nPer-talker ESTOI on {scene} (output; Mic A input in the first row; relative level in the take name order "
           "M1 0, M2 -6, M3 -12, M4 -3, M5 -9, M6 -15 dB, MF1/MF2 -6 dB at 4 m):\n",
           "| method | " + " | ".join(tids) + " |", "|---" * (len(tids) + 1) + "|"]
    r0 = rows.get("mic_a", {}).get(scene)
    if r0:
        out.append("| Mic A (input) | " + " | ".join(f(r0["talkers"].get(t, {}).get("estoi_in"), ".3f") for t in tids) + " |")
    for m in methods:
        r = rows.get(m, {}).get(scene)
        if r is None or r.get("status") != "ok":
            continue
        out.append(f"| {m} | " + " | ".join(f(r["talkers"].get(t, {}).get("estoi"), ".3f") for t in tids) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "tables.md"))
    ap.add_argument("--json", default=os.path.join(HERE, "summary.json"))
    a = ap.parse_args()
    meas = load("measured")
    lines, summary = ["# real:run tables (generated by tmp/night_realrun/analyze.py)"], {}
    L1 = [f"bus/L1_snr{s}_{n}" for s in ("-5", "+0", "+5") for n in ("recorded", "quiet")]
    bus_methods = ["mic_a", "m0_mica_hpf", "avg", "m1_das", "m2_bus_beam", "x_gnfix_m2_bus_beam", "m3_rtf_mvdr",
                   "x_gnfix_m3_rtf_mvdr", "m4_cdr", "x_gnfix_m4_cdr", "m5_fence", "m6_ns", "m7_cacgmm", "m8_gtcrn",
                   "m8_dpdfnet", "x_combo_cdr_gtcrn", "x_combo_endfire", "x_combo_endfire_cdr_gtcrn",
                   "m9_voxlock2", "m9_voxlock2+enroll_device", "m9_voxlock2_self", "x_combo_m9_cdr"]
    bus_methods += sorted(m for m in meas if m not in bus_methods and any(k.startswith("bus/") for k in meas[m]))
    t, s = bus_table(meas, bus_methods, L1, "Bus, variant measured, layout L1 (6 scenes: 3 SNR x recorded/quiet)")
    lines += t
    summary["bus_measured_L1"] = s
    lines += paired(meas, bus_methods, L1, "m0_mica_hpf")
    all_bus = sorted(k for m in meas.values() for k in m if k.startswith("bus/"))
    all_bus = sorted(set(all_bus))
    others = [sc for sc in all_bus if not sc.startswith("bus/L1_")]
    if others:
        t, s = bus_table(meas, bus_methods, others, "Bus, variant measured, layouts L2/L3 (whatever was run; extra scenes)")
        lines += t
        summary["bus_measured_L23"] = s
    # M9 scenes
    m9sc = ["bus/L1_snr+0_recorded", "bus/L1_snr+5_quiet"]
    m9m = ["m0_mica_hpf", "m8_gtcrn", "m8_dpdfnet", "x_combo_cdr_gtcrn", "m9_voxlock2", "m9_voxlock2+enroll_device",
           "m9_voxlock2_self", "x_combo_m9_cdr"]
    for sc in m9sc:
        t, _ = bus_table(meas, m9m, [sc], f"Bus M9 family on {sc} (measured)", by_snr=False)
        lines += t
    # variants
    for var in ("blind", "wrong4cm"):
        vr = load(var)
        if not vr:
            continue
        sc3 = [f"bus/L1_snr{s}_recorded" for s in ("-5", "+0", "+5")]
        vm = ["m1_das", "m2_bus_beam", "m3_rtf_mvdr", "m4_cdr", "m5_fence", "m7_cacgmm", "x_auto_m1_das", "x_auto_m2_bus_beam",
              "x_auto_m3_rtf_mvdr", "x_auto_m4_cdr", "x_auto_m5_fence", "x_auto_m7_cacgmm"]
        t, s = bus_table(vr, vm, sc3, f"Bus, variant {var}, L1 recorded at -5/0/+5 dB")
        lines += t
        summary[f"bus_{var}"] = s
        # same scenes, measured, for reference
    sc3 = [f"bus/L1_snr{s}_recorded" for s in ("-5", "+0", "+5")]
    t, s = bus_table(meas, ["m0_mica_hpf", "m1_das", "m2_bus_beam", "x_gnfix_m2_bus_beam", "m3_rtf_mvdr", "m4_cdr",
                            "x_gnfix_m4_cdr", "m5_fence", "m7_cacgmm"], sc3,
                     "Bus, variant measured, same 3 scenes (reference for the variant tables)")
    lines += t
    summary["bus_measured_rec3"] = s
    # meeting
    msc = ["meeting/recorded", "meeting/quiet", "meeting/quiet_omeeting"]
    mm = ["mic_a", "m0_mica_hpf", "avg", "m2_meeting_beam", "m4_cdr", "x_gnfix_m4_cdr", "m6_ns", "m8_gtcrn", "m8_dpdfnet",
          "m10_wpe", "m11_level", "x_combo_cdr_gtcrn", "x_combo_endfire", "x_combo_meet_chain", "x_combo_meet_ns_agc"]
    mm += sorted(m for m in meas if m not in mm and any(k.startswith("meeting/") for k in meas[m]))
    t, s = meet_table(meas, mm, msc, "Meeting, variant measured (3 scenes)")
    lines += t
    summary["meeting_measured"] = s
    for sc in msc:
        lines += per_talker(meas, mm, sc)
    for var in ("blind", "wrong4cm"):
        vr = load(var)
        if any(k.startswith("meeting/") for m in vr.values() for k in m):
            for k in list(vr):
                vr.setdefault("mic_a", meas.get("mic_a", {}))
            t, s = meet_table(vr, ["m2_meeting_beam", "m11_level", "m4_cdr", "x_auto_m2_meeting_beam", "x_auto_m4_cdr"], msc,
                              f"Meeting, variant {var}")
            lines += t
            summary[f"meeting_{var}"] = s
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(summary, open(a.json, "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
