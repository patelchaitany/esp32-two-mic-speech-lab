"""Bench runner for the paired spacing set (scenes3/spacing, study/SCENES.md). It mirrors lab.bench2: the same
Hagerman runs (metrics2.plan), the same scoring (metrics2.score), the same method contract checks
(bench2.load_method: code policy) and result format; results go to results3/<set>/<label>/<name>.<corpus>.json.

Differences from lab.bench2 (which hard-codes info = {scenario, spacing_m: 0.04}):
- info['spacing_m'] is the scene's nominal spacing (0.012 or 0.04; --prior <m> forces a value);
- --calib factory adds info['calib_json'] = the scene's factory calibration (true spacing, skew, HPF corners,
  B gain); --calib blind leaves S0 to calibrate blind (it cannot represent this board: API.md, D_RANGE 1.5-8 cm,
  skew 0 / +0.5 sample); --calib both runs each method twice;
- --enroll adds info['enroll'] = {'A', 'B'}: the bus partners' reserved LibriSpeech enrollment samples (M9);
- the result label is '<method>@<calib>' (plus '-p<prior mm>' with a forced prior), so modes never mix;
- 'table' prints one table per spacing (gates evaluated within a spacing), 'spacing' the paired 40 - 12 mm
  differences per layout (bootstrap CI, stratified by E1 stratum on the bus; Wilcoxon; Holm across labels),
  also relative to a reference label ('what spacing buys' beyond the reference).
The extra Hagerman runs default to none here (bench2: 20% subset).

  OMP_NUM_THREADS=1 .venv/bin/python -I study/run.py run --methods m0_mica_hpf,m1_das [--scenario bus]
        [--spacing 12,40] [--snr -5,0] [--strata aligned] [--layouts L00,R01] [--calib factory|blind|both]
        [--prior <m>] [--enroll] [-j 2] [--extra none|subset|all] [--limit N] [--skip-existing] [--save-audio]
  .venv/bin/python -I study/run.py table [--labels a@factory,b@factory] [--scenario bus] [--by stratum,snr]
  .venv/bin/python -I study/run.py spacing [--labels ...] [--ref m0_mica_hpf@factory] [--metrics d_estoi_min,...]
  .venv/bin/python -I study/run.py compare --spacing 12 [--ref m0_mica_hpf@factory]
"""
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LAB not in sys.path:
    sys.path.insert(0, LAB)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from lab import bench2, metrics2, scene2, stats  # noqa: E402
from lab.common import FS  # noqa: E402

SCENES = os.path.join(LAB, "scenes3")
RESULTS = os.path.join(LAB, "results3")
SET = "spacing"
CORPUS = "libri"
VERSION = "study.run/1 (" + bench2.VERSION + " scoring)"
MODES = ("factory", "blind")
# decision metrics of the spacing comparison (bench2.COMPARE plus the input-side reference)
SPACING_METRICS = {
    "bus": ["d_estoi_min", "estoi_wrst", "estoi_wrst_in", "d_sdr_min", "dsnr_active", "na_active", "ta_min", "nm",
            "dsir_near", "rda_p10", "balance_p90", "pass_estoi_med"],
    "meeting": ["d_estoi_wrst", "d_estoi_mean", "d_estoi_far", "estoi_wrst", "estoi_wrst_in", "capture_rate",
                "na_active", "t2_margin", "lvl_std", "nm"],
}


# ----------------------------------------------------------------------------- info and one method
def label(method, mode, prior=None):
    return f"{method}@{mode}" + (f"-p{round(prior * 1000)}" if prior is not None else "")


def scene_info(sc, card, mode, prior=None, enroll=False):
    """What a method gets besides the signal: scenario, the spacing prior (the scene's nominal spacing unless
    `prior`), the factory calib_json in mode 'factory', partner enrollment with `enroll` (bus)."""
    m = sc["meta"]
    info = {"scenario": m["scenario"],
            "spacing_m": float(prior) if prior is not None else card["spacing_mm"] / 1000.0}
    if mode == "factory":
        info["calib_json"] = copy.deepcopy(m["recipe_meta"]["factory_calib"])
    elif mode != "blind":
        raise ValueError(f"unknown calib mode {mode}")
    if enroll and m["scenario"] == "bus":
        from lab import corpus_libri as cl  # reads data/ (audio only)
        info["enroll"] = {k: np.asarray(cl.enrollment_audio(m["talkers"][t]["speaker"])[0], np.float32)
                          for k, t in (("A", "pA"), ("B", "pB"))}
    return info


def run_method(method, sc, info, ctx=None, card=None, extra=False):
    """lab.bench2.run_method with a given `info` (never raises; errors are recorded in the result)."""
    T = int(sc["mix"].shape[1])
    res = {"bench": VERSION, "scene": sc["id"], "method": str(method), "scenario": sc["meta"]["scenario"],
           "corpus": sc["meta"].get("corpus"), "card": card or {}, "extra": bool(extra),
           "info_keys": sorted(info)}
    y1 = None
    try:
        mod, res["method"], accepts = bench2.load_method(method)
        base = dict(info)
        if bench2.is_oracle(res["method"]):
            base["oracle"] = bench2.oracle_view(sc)
            res["oracle"] = True

        def call(x):
            i = {k: (v if k == "oracle" else copy.deepcopy(v)) for k, v in base.items()}  # fresh per call
            return mod.enhance(x, FS, i, return_stages=True) if accepts else mod.enhance(x, FS, i)

        runs, comps = metrics2.plan(sc, extra)
        outs, timing, peaks, t_method = {}, [], {}, 0.0
        for run, flip in runs:
            x = scene2.signed_mix(sc, flip=flip)
            t0 = time.perf_counter()
            r = call(x)
            dt = time.perf_counter() - t0
            t_method += dt
            outs[run], peaks[run] = bench2._stages(r, T)
            timing.append({"run": run, "flip": list(flip), "rtf": dt / (T / FS)})
        y1 = outs["y1"]["out"]
        rerun = {}
        if extra:
            again = bench2._stages(call(scene2.signed_mix(sc)), T)[0]["out"].astype(np.float64)
            err = y1.astype(np.float64) - again
            rerun["rerun_db"] = float(min(10 * np.log10((y1.astype(np.float64) @ y1 + 1e-30) / (err @ err + 1e-30)),
                                          metrics2.RATIO_CAP_DB))
        t0 = time.perf_counter()
        scored = metrics2.score(sc, outs, comps, ctx)
        rtfs = [t["rtf"] for t in timing]
        scored["m"].update(rtf=float(np.median(rtfs)), rtf_y1=rtfs[0], rtf_max=float(max(rtfs)), peak=peaks["y1"],
                           clipped=float(peaks["y1"] >= 1.0), n_runs=float(len(runs)), **rerun)
        res.update(status="ok", runs=timing, stages=any(o["pre"] is not o["out"] for o in outs.values()),
                   algo_latency_ms=getattr(mod, "ALGO_LATENCY_MS", None),
                   offline_only=bool(getattr(mod, "OFFLINE_ONLY", False)),
                   time_s={"method": t_method, "score": time.perf_counter() - t0}, **scored)
    except Exception as e:
        res.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    return res, y1


def result_path(scene_id, lab_, corpus=CORPUS, root=None):
    set_name, name = scene_id.split("/")
    return os.path.join(root or RESULTS, set_name, lab_, f"{name}.{corpus}.json")


def _done(path):
    try:
        with open(path) as f:
            return json.load(f).get("status") == "ok"
    except (OSError, ValueError):
        return False


def _job(job):
    sid, todo, o = job  # todo: [(method, mode)]
    out = []
    try:
        sc = scene2.load_scene(sid, corpus=o["corpus"], root=o["scenes_root"])
        card = scene2.scene_card(sid, root=o["scenes_root"])
        ctx = metrics2.SceneCtx(sc)
    except Exception as e:
        for m, mode in todo:
            lb = label(m, mode, o["prior"])
            res = {"bench": VERSION, "scene": sid, "method": lb, "corpus": o["corpus"], "status": "error",
                   "error": f"scene: {type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            bench2.write_result(res, result_path(sid, lb, o["corpus"], o["results_root"]))
            out.append((sid, lb, "error", res["error"], None))
        return out
    for m, mode in todo:
        lb = label(m, mode, o["prior"])
        try:
            info = scene_info(sc, card, mode, o["prior"], o["enroll"])
            res, y1 = run_method(m, sc, info, ctx, card, extra=o["extra"])
        except Exception as e:  # info construction failed
            res, y1 = {"bench": VERSION, "scene": sid, "method": m, "scenario": sc["meta"]["scenario"],
                       "status": "error", "error": f"info: {type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-3000:]}, None
        res.update(set=sid.split("/")[0], corpus=o["corpus"], method_name=res.get("method", m), method=lb,
                   calib_mode=mode, prior_m=o["prior"], enroll=o["enroll"])
        path = result_path(sid, lb, o["corpus"], o["results_root"])
        bench2.write_result(res, path)
        if o["save_audio"] and y1 is not None:
            sf.write(path[:-len(".json")] + ".wav", y1, FS, subtype="FLOAT")
        mm = res.get("m") or {}
        key = "d_estoi_min" if res.get("scenario") == "bus" else "d_estoi_wrst"
        out.append((sid, lb, res["status"], res.get("error"), mm.get(key)))
    return out


def select_scenes(a):
    filters = {}
    if getattr(a, "spacing", None):
        filters["spacing_mm"] = tuple(int(v) for v in a.spacing.split(","))
    if getattr(a, "snr", None):
        filters["snr_db"] = tuple(float(v) for v in a.snr.split(","))
    if getattr(a, "strata", None):
        filters["stratum"] = tuple(a.strata.split(","))
    if getattr(a, "layouts", None):
        filters["layout_id"] = tuple(a.layouts.split(","))
    return scene2.list_scenes(a.set, scenario=a.scenario, root=a.scenes_root, **filters)


def cmd_run(a):
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        bench2.load_method(m)  # fail fast: missing method or code-policy violation
    modes = MODES if a.calib == "both" else (a.calib,)
    scenes = select_scenes(a)
    scenes = scenes[:a.limit] if a.limit else scenes
    if not scenes:
        print("no scenes match")
        return 1
    extra = bench2.extra_geometries(a.set, a.scenes_root) if a.extra == "subset" else set()
    jobs = []
    for sid in scenes:
        card = scene2.scene_card(sid, root=a.scenes_root)
        todo = [(m, md) for m in methods for md in modes
                if not (a.skip_existing and _done(result_path(sid, label(m, md, a.prior), a.corpus, a.results_root)))]
        if todo:
            o = {"corpus": a.corpus, "scenes_root": a.scenes_root, "results_root": a.results_root, "prior": a.prior,
                 "enroll": a.enroll, "save_audio": a.save_audio,
                 "extra": a.extra == "all" or card["geom_id"] in extra}
            jobs.append((card["duration_s"], sid, todo, o))
    jobs = [(sid, todo, o) for _, sid, todo, o in sorted(jobs, key=lambda j: -j[0])]
    print(f"{len(jobs)} scenes x {len(methods)} methods x {len(modes)} calib modes, {a.j} workers", flush=True)
    t0, done = time.time(), 0
    pool = get_context("spawn").Pool(a.j) if a.j > 1 else None
    try:
        for rs in (pool.imap_unordered(_job, jobs) if pool else map(_job, jobs)):
            done += 1
            for sid, lb, status, err, v in rs:
                tail = f"ERROR {err}" if status != "ok" else (f"{v:+.3f}" if v is not None else "nan")
                print(f"[{done}/{len(jobs)} {time.time() - t0:6.0f} s] {sid} {lb}: {tail}", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return 0


# ----------------------------------------------------------------------------- reading results
def load_rows(a, labels=None):
    rows = bench2.load_rows(a.set, a.corpus, labels, a.scenario, a.results_root)
    for r in rows:  # crossed breakdown keys for bench2.render_table
        c = r.setdefault("card", {})
        mm = c.get("spacing_mm")
        c["spacing_x_stratum"] = f"{mm} mm/{c.get('stratum')}"
        c["spacing_x_snr"] = f"{mm} mm/{c.get('snr_db')}"
    if getattr(a, "snr", None):
        snrs = {float(v) for v in a.snr.split(",")}
        rows = [r for r in rows if stats.card(r).get("snr_db") in snrs or r.get("scenario") != "bus"]
    if getattr(a, "strata", None):
        rows = [r for r in rows if stats.card(r).get("stratum") in set(a.strata.split(","))
                or r.get("scenario") != "bus"]
    return rows


def _labels(a):
    return [v.strip() for v in a.labels.split(",")] if getattr(a, "labels", None) else None


def cmd_table(a):
    rows = load_rows(a, _labels(a))
    by = a.by.split(",") if a.by else ()
    for mm in sorted({stats.card(r).get("spacing_mm") for r in rows}, key=str):
        print(f"\n## spacing {mm} mm (gates within this spacing; bus: 10 geometries, meeting: 4 rooms)")
        print(bench2.render_table([r for r in rows if stats.card(r).get("spacing_mm") == mm], by=by))
    return 0


def _relabel(rows):
    """Rows with the geometry key replaced by the layout id, so stats pairs the two spacings of a layout."""
    return [dict(r, card=dict(stats.card(r), geom_id=stats.card(r).get("layout_id"))) for r in rows]


def spacing_pairs(rows, lb, key, a_mm=40, b_mm=12):
    """stats.paired of `key` for label `lb`: layout means at a_mm minus at b_mm (both spacings needed)."""
    ra = _relabel([r for r in rows if r["method"] == lb and stats.card(r).get("spacing_mm") == a_mm])
    rb = _relabel([r for r in rows if r["method"] == lb and stats.card(r).get("spacing_mm") == b_mm])
    return stats.paired(ra, rb, key)


def spacing_vs_ref(rows, lb, ref, key, a_mm=40, b_mm=12):
    """Interaction: [lb - ref] at a_mm minus [lb - ref] at b_mm, per layout (bootstrap CI, Wilcoxon p)."""
    pg = {}
    for who in (lb, ref):
        for mm in (a_mm, b_mm):
            pg[who, mm] = stats.per_geometry(_relabel([r for r in rows if r["method"] == who
                                                       and stats.card(r).get("spacing_mm") == mm]), key)
    common = sorted(set.intersection(*(set(v) for v in pg.values())), key=str)
    if not common:
        return {"n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    d = np.array([(pg[lb, a_mm][g][0] - pg[ref, a_mm][g][0]) - (pg[lb, b_mm][g][0] - pg[ref, b_mm][g][0])
                  for g in common])
    strata = [pg[lb, a_mm][g][1] for g in common]
    use = any(s is not None for s in strata)
    delta, lo, hi = stats.bootstrap_ci(d, strata if use else None)
    return {"n": len(common), "delta": delta, "lo": lo, "hi": hi, "p": stats.wilcoxon_p(d)}


def render_spacing(rows, ref=None, metrics=None):
    f = bench2._fmt
    lines = []
    for scenario in ("bus", "meeting"):
        rs = [r for r in rows if r.get("scenario") == scenario and r.get("status") == "ok"]
        labels = sorted({r["method"] for r in rs})
        if not labels:
            continue
        lines += [f"\n### {scenario}: 40 mm minus 12 mm, paired per layout (mean over SNRs; 95% bootstrap CI"
                  + (", stratified by E1 stratum" if scenario == "bus" else "") + "; Wilcoxon p, Holm across labels)",
                  "", "| metric | label | n | 12 mm | 40 mm | Δ(40−12) | 95% CI | p | p (Holm) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key in metrics or SPACING_METRICS[scenario]:
            res = [spacing_pairs(rs, lb, key) for lb in labels]
            adj = stats.holm([q["p"] for q in res])
            for lb, q, ph in zip(labels, res, adj):
                if q["n"]:
                    lines.append(f"| {key} | {lb} | {q['n']} | {f(q['ref_mean'], '.3g')} | {f(q['mean'], '.3g')} | "
                                 f"{f(q['delta'], '+.3g')} | [{f(q['lo'], '+.3g')}, {f(q['hi'], '+.3g')}] | "
                                 f"{f(q['p'], '.3g')} | {f(ph, '.3g')} |")
        if ref and ref in labels and len(labels) > 1:
            lines += ["", f"{scenario}: what 40 mm buys beyond {ref}: [label − ref](40 mm) − [label − ref](12 mm), "
                      "paired per layout", "", "| metric | label | n | Δ | 95% CI | p | p (Holm) |",
                      "|---|---|---|---|---|---|---|"]
            for key in metrics or SPACING_METRICS[scenario]:
                others = [lb for lb in labels if lb != ref]
                res = [spacing_vs_ref(rs, lb, ref, key) for lb in others]
                adj = stats.holm([q["p"] for q in res])
                for lb, q, ph in zip(others, res, adj):
                    if q["n"]:
                        lines.append(f"| {key} | {lb} | {q['n']} | {f(q['delta'], '+.3g')} | [{f(q['lo'], '+.3g')}, "
                                     f"{f(q['hi'], '+.3g')}] | {f(q['p'], '.3g')} | {f(ph, '.3g')} |")
    return "\n".join(lines) if lines else "no paired results (need both spacings of a layout for a label)"


def cmd_spacing(a):
    rows = load_rows(a, _labels(a))
    print(render_spacing(rows, a.ref, a.metrics.split(",") if a.metrics else None))
    return 0


def cmd_compare(a):
    rows = load_rows(a, None)
    if a.spacing:
        keep = {int(v) for v in a.spacing.split(",")}
        rows = [r for r in rows if stats.card(r).get("spacing_mm") in keep]
    labels = _labels(a)
    if labels:
        rows = [r for r in rows if r["method"] in set(labels) | {a.ref}]
    print(bench2.render_compare(rows, a.ref, a.metrics.split(",") if a.metrics else None))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="study/run.py", description="paired spacing set runner (study/SCENES.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", default=SET)
    common.add_argument("--scenario", choices=("bus", "meeting"))
    common.add_argument("--corpus", default=CORPUS)
    common.add_argument("--snr", help="comma list of bus SNRs (dB)")
    common.add_argument("--strata", help="comma list of E1 strata (bus)")
    common.add_argument("--scenes-root", default=SCENES)
    common.add_argument("--results-root", default=RESULTS)
    r = sub.add_parser("run", parents=[common], help="run and score methods")
    r.add_argument("--methods", required=True, help="comma list of methods/<name>.py")
    r.add_argument("--spacing", help="comma list of spacings in mm (12, 40)")
    r.add_argument("--layouts", help="comma list of layout ids (L00..L09, R00..R03)")
    r.add_argument("--calib", choices=("factory", "blind", "both"), default="factory")
    r.add_argument("--prior", type=float, default=None, help="force info['spacing_m'] (m); default: nominal")
    r.add_argument("--enroll", action="store_true", help="give bus partners' enrollment samples (M9)")
    r.add_argument("--limit", type=int)
    r.add_argument("-j", type=int, default=2)
    r.add_argument("--extra", choices=("none", "subset", "all"), default="none")
    r.add_argument("--skip-existing", action="store_true")
    r.add_argument("--save-audio", action="store_true")
    t = sub.add_parser("table", parents=[common], help="bench2 table per spacing")
    t.add_argument("--labels", help="comma list of result labels (method@calib)")
    t.add_argument("--by", help="breakdowns: stratum, snr, spacing_x_stratum, spacing_x_snr, layout_id, ...")
    s = sub.add_parser("spacing", parents=[common], help="paired 40 - 12 mm differences per layout")
    s.add_argument("--labels")
    s.add_argument("--ref", help="reference label for the interaction table, e.g. m0_mica_hpf@factory")
    s.add_argument("--metrics")
    c = sub.add_parser("compare", parents=[common], help="bench2 compare within a spacing")
    c.add_argument("--spacing", help="comma list of spacings in mm")
    c.add_argument("--labels")
    c.add_argument("--ref", default="m0_mica_hpf@factory")
    c.add_argument("--metrics")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "table": cmd_table, "spacing": cmd_spacing, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
