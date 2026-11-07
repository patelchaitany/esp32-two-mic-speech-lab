"""Real-device evaluation: runs methods on the hw/realset scenes (all Hagerman runs) and scores them with lab.metrics2.

  .venv/bin/python -I hw/realeval.py run --methods a,b [--variant measured|blind|wrong4cm] [--set bus|meeting|all]
                                        [--enroll clean|device] [-j 2] [--limit N] [--scenes substr] [--skip-existing]
  .venv/bin/python -I hw/realeval.py table [--variant measured] [--set all] [--methods a,b] [--enroll clean]
  .venv/bin/python -I hw/realeval.py list

Variants (what the method is told in info; every variant gets 'scenario' and 'enroll'):
  measured  spacing_m = session spacing (1.19 cm) and calib_json = hw/realset/calib_session.json (S0 override)
  blind     spacing_m = session spacing, no calib_json (S0 estimates everything blindly)
  wrong4cm  spacing_m = 0.04 (the bench default the methods were tuned with), no calib_json
Results: hw/realset/results/<variant>/<method>[+enroll_device]/<scene>.json and the y1 output as 16-bit WAV
(clipped at +-1, never normalized). Scenes and their limits: hw/realset/README.md.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
for _p in (LAB, os.path.join(HERE, "realset", "build"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import realscenes as rs  # noqa: E402
from lab import bench2, metrics2, scene2  # noqa: E402

FS = 16000
RESULTS = os.path.join(HERE, "realset", "results")
VARIANTS = ("measured", "blind", "wrong4cm")
VERSION = "realeval/1"


def make_info(scenario, variant, enroll):
    cal = rs._calib()
    info = {"scenario": scenario, "spacing_m": 0.04 if variant == "wrong4cm" else float(cal["spacing_m"])}
    if variant == "measured":
        info["calib_json"] = dict(cal)
    if enroll is not None:
        info["enroll"] = rs.load_enroll(enroll)
    return info


def method_dir(variant, method, enroll):
    return os.path.join(RESULTS, variant, method + ("" if enroll == "clean" else f"+enroll_{enroll}"))


def own_metrics(sc, y1, ctx):
    """Small cross-check set computed here (independent of the bench composites): per target talker ESTOI and
    in-band SDR of y1 against the FULL image (comp2 Mic A of its group) over its speech."""
    out = {}
    tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] in ("partner", "talker")]
    for t in tids:
        iv = ctx.iv.get(t, [])
        full = np.asarray(sc["comp2"][sc["talker_group"][t]][0], np.float64)
        out[t] = {"estoi_full": metrics2.estoi(y1, full, iv), "sdr_full": metrics2.sdr_fir(y1, full, iv),
                  "estoi_full_in": ctx.memo(("fullin", t), lambda: metrics2.estoi(ctx.xa, full, iv))}
    return out


def gap_metrics(sc, ctx, outs, comps):
    """Noise attenuation where no target talker is active, with short guards (20 ms before, 150 ms after each
    target segment, first 0.25 s skipped): the bench's idle definition (0.1 s / 0.3 s guards, >= 1 s) finds no
    idle time in the bus scenes (0.5 s turn gaps). na_gap_* = -(level change of the decoded N) over those
    samples, nm_gap = na_gap_out - na_active_out (as the bench's NM, on the final output)."""
    tids = [t for t, g in sc["talker_group"].items() if sc["groups"][g] in ("partner", "talker")]
    act = ctx.active(tids)
    edge = np.zeros(ctx.T + 1, np.int32)
    for a, b in metrics2._from_mask(act):
        edge[max(a - 320, 0)] += 1
        edge[min(b + 2400, ctx.T)] -= 1
    gap = np.cumsum(edge[:-1]) == 0
    gap[:FS // 4] = False
    out = {"gap_s": float(gap.sum()) / FS}
    if gap.sum() < FS // 2:
        return out
    n_in = ctx.mic_a(comps["N"][1])
    for st in ("pre", "out"):
        nf = metrics2.inband(metrics2.decode(outs, comps["N"][0], st))
        out[f"na_gap_{st}"] = -metrics2._change(nf, n_in, gap)
        out[f"na_act_{st}"] = -metrics2._change(nf, n_in, act)
    out["nm_gap"] = out["na_gap_out"] - out["na_act_out"]
    return out


def run_one(method, sc, ctx, info):
    """bench2.run_method with a custom info dict (the bench fixes info to {scenario, spacing_m: 0.04})."""
    T = int(sc["mix"].shape[1])
    res = {"bench": VERSION, "scene": sc["id"], "method": method, "scenario": sc["meta"]["scenario"],
           "card": sc["meta"]["card"], "info_keys": sorted(info)}
    y1 = None
    try:
        mod, name, accepts = bench2.load_method(method)

        def call(x):
            return mod.enhance(x, FS, dict(info), return_stages=True) if accepts else mod.enhance(x, FS, dict(info))

        runs, comps = metrics2.plan(sc, extra=False)
        outs, timing, peaks, t_m = {}, [], {}, 0.0
        for run, flip in runs:
            x = scene2.signed_mix(sc, flip=flip)
            t0 = time.perf_counter()
            r = call(x)
            dt = time.perf_counter() - t0
            t_m += dt
            outs[run], peaks[run] = bench2._stages(r, T)
            timing.append({"run": run, "flip": list(flip), "rtf": dt / (T / FS)})
        y1 = outs["y1"]["out"]
        t0 = time.perf_counter()
        scored = metrics2.score(sc, outs, comps, ctx)
        rt = [t["rtf"] for t in timing]
        scored["m"].update(rtf=float(np.median(rt)), rtf_max=float(max(rt)), peak=peaks["y1"],
                           clipped=float(peaks["y1"] >= 1.0), n_runs=float(len(runs)))
        scored["own"] = own_metrics(sc, np.asarray(y1, np.float64), ctx)
        scored["m"].update(gap_metrics(sc, ctx, outs, comps))
        res.update(status="ok", runs=timing, stages=any(o["pre"] is not o["out"] for o in outs.values()),
                   algo_latency_ms=getattr(mod, "ALGO_LATENCY_MS", None),
                   time_s={"method": t_m, "score": time.perf_counter() - t0}, **scored)
    except Exception as e:  # report and go on
        res.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    return res, y1


def job(scene_id, methods, variant, enroll, skip):
    sc = rs.load_scene(scene_id)
    ctx = metrics2.SceneCtx(sc)
    info = make_info(sc["meta"]["scenario"], variant, enroll)
    lines = []
    name = scene_id.replace("/", "__")
    for m in methods:
        d = method_dir(variant, m, enroll)
        path = os.path.join(d, name + ".json")
        if skip and os.path.exists(path):
            try:
                if json.load(open(path)).get("status") == "ok":
                    lines.append(f"{scene_id:28s} {m:22s} skipped (exists)")
                    continue
            except Exception:
                pass
        t0 = time.time()
        res, y1 = run_one(m, sc, ctx, info)
        res.update(variant=variant, enroll=enroll, wall_s=time.time() - t0)
        os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(bench2._clean(res), fh, indent=1)
        if y1 is not None:
            sf.write(os.path.join(d, name + ".wav"), np.clip(np.asarray(y1, np.float64), -1, 1), FS, subtype="PCM_16")
        if res["status"] == "ok":
            m_ = res["m"]
            key = ("d_estoi_min", "dsnr_active", "nm_gap") if res["scenario"] == "bus" else ("d_estoi_wrst", "d_estoi_mean", "na_active")
            lines.append(f"{scene_id:28s} {m:22s} ok   " + "  ".join(f"{k}={m_.get(k, float('nan')):+.3f}" for k in key)
                         + f"  rtf={m_['rtf']:.3f}  {time.time() - t0:.0f}s")
        else:
            lines.append(f"{scene_id:28s} {m:22s} ERROR {res['error'][:200]}")
    return lines


def cmd_run(a):
    methods = [m for m in a.methods.split(",") if m]
    scenes = rs.list_scenes(a.set)
    if a.scenes:
        scenes = [s for s in scenes if any(p in s for p in a.scenes.split(","))]
    if a.limit:
        scenes = scenes[:a.limit]
    scenes.sort(key=lambda s: 0 if s.startswith("meeting") else 1)  # longest first
    print(f"{len(scenes)} scenes x {len(methods)} methods, variant={a.variant}, enroll={a.enroll}, j={a.j}", flush=True)
    t0 = time.time()
    if a.j <= 1:
        for s in scenes:
            for ln in job(s, methods, a.variant, a.enroll, a.skip_existing):
                print(ln, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.j) as ex:
            futs = {ex.submit(job, s, methods, a.variant, a.enroll, a.skip_existing): s for s in scenes}
            for f in as_completed(futs):
                try:
                    for ln in f.result():
                        print(ln, flush=True)
                except Exception as e:
                    print(f"{futs[f]:28s} WORKER ERROR {type(e).__name__}: {e}", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    a.methods = ",".join(methods)
    cmd_table(a)


BUS_COLS = [("d_estoi_min", "wrstΔESTOI", "+.3f"), ("estoi_wrst", "wrstESTOI", ".3f"), ("d_sdr_min", "wrstΔSDR", "+.1f"),
            ("ta_min", "TAmin", "+.1f"), ("balance", "bal", ".1f"), ("dsnr_active", "ΔSNRact", "+.1f"),
            ("na_active", "NAact", "+.1f"), ("nm", "NM", "+.1f"), ("dsir_near", "ΔSIRpax", "+.1f"),
            ("dsir_rest", "ΔSIRrest", "+.1f"), ("rda_p10", "RDAp10", "+.1f"), ("pass_estoi_med", "paxESTOI", ".3f"),
            ("pass_drop_med", "paxDrop", "+.3f"), ("rtf", "RTF", ".3f"), ("delay", "delay", "+.1f"),
            ("closure_db", "closure", ".0f"), ("na_gap_pre", "NAgap", "+.1f"), ("nm_gap", "NMgap", "+.1f")]
MEET_COLS = [("capture_rate", "T1cap", ".2f"), ("d_estoi_wrst", "wrstΔESTOI", "+.3f"), ("d_estoi_mean", "meanΔESTOI", "+.3f"),
             ("d_estoi_far", "farΔESTOI", "+.3f"), ("t2_margin", "T2far-near", "+.1f"), ("ta_wrst", "TAwrst", "+.1f"),
             ("lvl_std", "T3std", ".1f"), ("lvl_range", "T3range", ".1f"), ("na_active", "NAact", "+.1f"),
             ("nm", "NM", "+.1f"), ("pump_std", "pumpStd", ".1f"), ("rtf", "RTF", ".3f"), ("delay", "delay", "+.1f"),
             ("na_gap_pre", "NAgap", "+.1f"), ("nm_gap", "NMgap", "+.1f")]


def _fmt(v, f):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return format(v, f)


def load_results(variant, enroll="clean"):
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS, variant, "*"))):
        base = os.path.basename(d)
        if ("+enroll_" in base) != (enroll != "clean") or (enroll != "clean" and not base.endswith(enroll)):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            rows.append(json.load(open(p)))
    return rows


def cmd_table(a):
    rows = load_results(a.variant, a.enroll)
    methods = a.methods.split(",") if a.methods else sorted({r["method"] for r in rows})
    for scen, cols, keyf in (("bus", BUS_COLS, lambda r: f"snr{r['card']['snr_db']:+.0f}/{r['card']['device_noise']}"),
                             ("meeting", MEET_COLS, lambda r: r["card"]["id"].split("/")[1])):
        if a.set not in ("all", scen):
            continue
        rs_ = [r for r in rows if r.get("scenario") == scen and r["method"] in methods]
        if not rs_:
            continue
        print(f"\n## {scen} (variant {a.variant}, enroll {a.enroll}): mean over scenes; rows = method, then per condition")
        print("| method | cond | n | " + " | ".join(c[1] for c in cols) + " |")
        print("|---" * (len(cols) + 3) + "|")
        for m in methods:
            mr = [r for r in rs_ if r["method"] == m]
            err = [r for r in mr if r.get("status") != "ok"]
            ok = [r for r in mr if r.get("status") == "ok"]
            if err:
                print(f"| {m} | ERRORS | {len(err)} | {err[0].get('error', '')[:120]} |")
            groups = [("all", ok)] + sorted({keyf(r): [x for x in ok if keyf(x) == keyf(r)] for r in ok}.items())
            for cond, g in groups:
                if not g:
                    continue
                vals = []
                for k, _, f in cols:
                    v = [r["m"].get(k) for r in g]
                    v = [x for x in v if x is not None and np.isfinite(x)]
                    vals.append(_fmt(float(np.mean(v)) if v else float("nan"), f))
                print(f"| {m} | {cond} | {len(g)} | " + " | ".join(vals) + " |")
        # own cross-check metrics: ESTOI vs full image (mean over target talkers)
        print(f"\n{scen}: ESTOI of y1 vs the FULL image (mean over target talkers; Mic A input in brackets)")
        for m in methods:
            ok = [r for r in rs_ if r["method"] == m and r.get("status") == "ok"]
            if not ok:
                continue
            e = [v["estoi_full"] for r in ok for v in r["own"].values() if v["estoi_full"] is not None]
            ei = [v["estoi_full_in"] for r in ok for v in r["own"].values() if v["estoi_full_in"] is not None]
            print(f"  {m:24s} {np.mean(e):.3f} ({np.mean(ei):.3f})  Δ {np.mean(e) - np.mean(ei):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--methods", required=True)
    r.add_argument("--variant", default="measured", choices=VARIANTS)
    r.add_argument("--set", default="all", choices=("bus", "meeting", "all"))
    r.add_argument("--enroll", default="clean", choices=("clean", "device"))
    r.add_argument("-j", type=int, default=2)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--scenes", default="")
    r.add_argument("--skip-existing", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("--variant", default="measured", choices=VARIANTS)
    t.add_argument("--set", default="all", choices=("bus", "meeting", "all"))
    t.add_argument("--methods", default="")
    t.add_argument("--enroll", default="clean", choices=("clean", "device"))
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "run":
        a.j = min(a.j, 2)
        cmd_run(a)
    elif a.cmd == "table":
        cmd_table(a)
    else:
        for s in rs.list_scenes():
            print(s, json.dumps(rs.scene_card(s)))


if __name__ == "__main__":
    main()
