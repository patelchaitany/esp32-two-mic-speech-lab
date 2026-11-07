"""v2 bench runner (DESIGN 2.3, 4.2-4.6): runs methods over scene sets, scores them with lab.metrics2 and
summarizes them with lab.stats. Usage: BENCH2.md "Bench usage".

  .venv/bin/python -m lab.bench2 run --methods m0_mica_hpf --set dev [--scenario bus] [--snr 0] [-j 3]
  .venv/bin/python -m lab.bench2 table --set dev [--by stratum]
  .venv/bin/python -m lab.bench2 compare --set dev --methods a,b [--ref m0_mica_hpf]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker, so RTF is measured single-threaded

import argparse
import ast
import copy
import functools
import glob
import importlib
import inspect
import json
import math
import sys
import time
import traceback
import zlib
from multiprocessing import get_context

import numpy as np
import soundfile as sf

from . import metrics2, scene2, stats
from .common import FS

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(LAB, "results2")
VERSION = "bench2/2"
NOMINAL_SPACING = 0.04
EXTRA_EVERY = 5  # extra Hagerman runs on 1 geometry in 5 per scenario and stratum (DESIGN 4.2: 20% subset)
ORACLE_PREFIX = "oracle"  # methods named oracle* get the scene in info['oracle'] (labelled cheating bounds)
POLICY_ROOTS = {"numpy", "scipy", "soundfile", "pyroomacoustics", "onnxruntime", "methods"}
POLICY_LAB = ("lab.dsp", "lab.common")
POLICY_PATHS = ("scenes2", "scenes/", "results2", "corpus/", "corpora/")

# (key, header, format); keys are looked up with stats.metric. '%' columns are fractions shown in percent.
COLUMNS = {
    "bus": [("d_estoi_min", "wrstΔESTOI", "+.3f"), ("estoi_wrst", "wrstESTOI", ".3f"), ("d_sdr_min", "wrstΔSDR", "+.1f"),
            ("ta_min", "TAmin", "+.1f"), ("balance_p90", "balP90", ".1f"), ("dsnr_active", "ΔSNRact", "+.1f"),
            ("na_active", "NAact", "+.1f"), ("nm", "NM", "+.1f"), ("dsir_near", "ΔSIRnear", "+.1f"),
            ("rda_p10", "RDAp10", "+.1f"), ("pass_estoi_med", "paxESTOI", ".3f"),
            ("pass_drop_med", "paxDrop", "+.3f"), ("focus_pass", "focus%", "%"), ("rtf", "RTF", ".3f"),
            ("delay", "delay", "+.1f"), ("algo_latency_ms", "alg_ms", ".0f"), ("closure_db", "closure", ".0f"),
            ("consistency_db", "consist", ".0f"), ("rerun_db", "rerun", ".0f")],
    "meeting": [("capture_rate", "T1cap%", "%"), ("d_estoi_wrst", "wrstΔESTOI", "+.3f"),
                ("d_estoi_mean", "meanΔESTOI", "+.3f"), ("d_estoi_far", "farΔESTOI", "+.3f"),
                ("t2_margin", "T2far-near", "+.1f"), ("ta_wrst", "TAwrst", "+.1f"), ("lvl_std", "T3std", ".1f"),
                ("lvl_range", "T3range", ".1f"), ("na_active", "NAact", "+.1f"), ("nm", "NM", "+.1f"),
                ("pump_std", "pumpStd", ".1f"), ("settle_med", "settle_s", ".2f"), ("even_pass", "even%", "%"),
                ("rtf", "RTF", ".3f"), ("delay", "delay", "+.1f"), ("algo_latency_ms", "alg_ms", ".0f"),
                ("consistency_db", "consist", ".0f"), ("rerun_db", "rerun", ".0f")],
}
LEGACY = {"bus": [("dsisdr_sum", "L:dSISDR", "+.1f"), ("br", "L:BR", "+.1f"), ("pesq", "L:PESQ", ".2f")],
          "meeting": [("loud_std", "L:loudσ", ".1f"), ("gap_snr", "L:gapSNR", ".1f"), ("dsisdr", "L:dSISDR", "+.1f")]}
COMPARE = {"bus": ["d_estoi_min", "estoi_wrst", "d_sdr_min", "ta_min", "balance_p90", "dsnr_active", "na_active", "nm",
                   "dsir_near", "rda_p10", "pass_estoi_med", "focus_pass"],
           "meeting": ["capture_rate", "d_estoi_wrst", "d_estoi_mean", "d_estoi_far", "t2_margin", "lvl_std",
                       "lvl_range", "na_active", "nm", "pump_std", "settle_med", "even_pass"]}


# ----------------------------------------------------------------------------- running one method
class PolicyError(RuntimeError):
    """A method's code breaks the method contract (imports or data paths it may not use)."""


@functools.lru_cache(maxsize=None)
def _file_violations(path):
    """Contract violations in one source file: imports outside POLICY_ROOTS, the standard library and
    POLICY_LAB, and non-docstring string literals naming scene, result or corpus folders."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = []
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            mods = [f"lab.{a.name}" for a in node.names] if node.module == "lab" else [node.module]
        for m in mods:
            root = m.split(".")[0]
            ok = (any(m == a or m.startswith(a + ".") for a in POLICY_LAB) if root == "lab"
                  else root in POLICY_ROOTS or root in sys.stdlib_module_names)
            if not ok:
                out.append(f"{os.path.relpath(path, LAB)}:{node.lineno} imports {m}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            out += [f"{os.path.relpath(path, LAB)}:{node.lineno} names {p!r}" for p in POLICY_PATHS if p in node.value]
    return tuple(out)


def policy_violations(mod):
    """Static method-contract check of a method module and every methods.* module loaded with it. Methods
    run in the bench process next to the rendered scene (and scene2's render cache), so imports of other lab
    modules or reads of scene, result or corpus files are what could leak references to a method. Dynamic
    imports are not detected."""
    files = {getattr(mod, "__file__", None)}
    files |= {getattr(m, "__file__", None) for n, m in list(sys.modules.items())
              if n.startswith("methods.") and not is_oracle(n.split(".")[-1])}
    return [v for f in sorted(p for p in files if p and p.endswith(".py")) for v in _file_violations(f)]


def is_oracle(name):
    return str(name).startswith(ORACLE_PREFIX)


def load_method(method):
    """(module-like object with enhance(), name, whether enhance accepts return_stages). `method` is a
    methods/<name>.py name or any object with an enhance function (tests). Named methods must pass
    policy_violations() unless they are oracles; PolicyError otherwise."""
    mod = importlib.import_module(f"methods.{method}") if isinstance(method, str) else method
    name = method if isinstance(method, str) else getattr(method, "__name__", "custom")
    if isinstance(method, str) and not is_oracle(name):
        bad = policy_violations(mod)
        if bad:
            raise PolicyError(f"method {name} breaks the method contract: " + "; ".join(bad))
    params = inspect.signature(mod.enhance).parameters
    stages = "return_stages" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())
    return mod, name, stages


def oracle_view(sc):
    """What an oracle method gets in info['oracle']: the unmodified scene (the y1 decomposition, whatever
    Hagerman run is being processed), read-only arrays, a private copy of meta."""
    return {"id": sc["id"], "fs": sc["fs"], "mix": sc["mix"], "comp2": dict(sc["comp2"]), "groups": dict(sc["groups"]),
            "refs": dict(sc["refs"]), "talker_group": dict(sc["talker_group"]), "segs": list(sc["segs"]),
            "meta": copy.deepcopy(sc["meta"])}


def _as_array(y, T):
    y = np.asarray(y, np.float64).reshape(-1)
    if y.shape[0] != T:
        raise ValueError(f"output has {y.shape[0]} samples, expected {T}")
    if not np.all(np.isfinite(y)):
        raise ValueError("non-finite output")
    return y


def _stages(r, T):
    """{'out', 'pre'} float32 outputs of one enhance() call; 'pre' is 'out' (same object) without stages."""
    out = r["out"] if isinstance(r, dict) else r
    pre = r.get("pre_agc", out) if isinstance(r, dict) else out
    o = _as_array(out, T)
    p = o if pre is out else _as_array(pre, T)
    o32 = o.astype(np.float32)
    return {"out": o32, "pre": o32 if p is o else p.astype(np.float32)}, float(np.abs(o).max())


def run_method(method, sc, ctx=None, card=None, extra=False):
    """Run a method on every Hagerman run of a scene and score it. Never raises: a failure is recorded in
    the result ('status': 'error'). Returns (result dict, final output of the normal run or None).
    With extra=True the normal run y1 is repeated after all others ('rerun_db': history dependence)."""
    T = int(sc["mix"].shape[1])
    res = {"bench": VERSION, "scene": sc["id"], "method": str(method), "scenario": sc["meta"]["scenario"],
           "corpus": sc["meta"].get("corpus"), "card": card or {}, "extra": bool(extra)}
    y1 = None
    try:
        mod, res["method"], accepts = load_method(method)
        info = {"scenario": res["scenario"], "spacing_m": NOMINAL_SPACING}
        if is_oracle(res["method"]):
            info["oracle"] = oracle_view(sc)
            res["oracle"] = True

        def call(x):
            return mod.enhance(x, FS, dict(info), return_stages=True) if accepts else mod.enhance(x, FS, dict(info))

        runs, comps = metrics2.plan(sc, extra)
        outs, timing, peaks, t_method = {}, [], {}, 0.0
        for run, flip in runs:
            x = scene2.signed_mix(sc, flip=flip)  # a fresh writable float32 copy
            t0 = time.perf_counter()
            r = call(x)
            dt = time.perf_counter() - t0
            t_method += dt
            outs[run], peaks[run] = _stages(r, T)
            timing.append({"run": run, "flip": list(flip), "rtf": dt / (T / FS)})
        y1 = outs["y1"]["out"]
        rerun = {}
        if extra:  # same input again after the other runs: any difference is state kept between calls
            again = _stages(call(scene2.signed_mix(sc)), T)[0]["out"].astype(np.float64)
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
    except Exception as e:  # one failing scene must not stop the run
        res.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    return res, y1


# ----------------------------------------------------------------------------- files
def result_path(scene_id, method, corpus, root=None):
    """results2/<set>/<method>/<name>.<corpus>.json"""
    set_name, name = scene_id.split("/")
    return os.path.join(root or RESULTS, set_name, method, f"{name}.{corpus}.json")


def _clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN and inf to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def write_result(res, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(res), f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_rows(set_name, corpus=None, methods=None, scenario=None, root=None):
    """Result rows of a set (all methods unless `methods`), optionally one corpus / scenario."""
    rows = []
    for p in sorted(glob.glob(os.path.join(root or RESULTS, set_name, "*", "*.json"))):
        with open(p) as f:
            r = json.load(f)
        if methods and r.get("method") not in methods:
            continue
        if (corpus and r.get("corpus") != corpus) or (scenario and r.get("scenario") != scenario):
            continue
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- 'run'
def extra_geometries(set_name, scenes_root=None):
    """Deterministic 1-in-EXTRA_EVERY subset of the set's geometries (at least one), per scenario and E1
    stratum, chosen by CRC32 of the geometry id so it does not depend on filters or on the order of ids."""
    by = {}
    for sid in scene2.list_scenes(set_name, root=scenes_root):
        c = scene2.scene_card(sid, root=scenes_root)
        by.setdefault((c["scenario"], str(c.get("stratum"))), set()).add(c["geom_id"])
    pick = set()
    for gs in by.values():
        gs = sorted(gs, key=lambda g: (zlib.crc32(g.encode()), g))
        pick.update(gs[:max(1, round(len(gs) / EXTRA_EVERY))])
    return pick


def _summary(res):
    m = res.get("m") or {}
    key = "d_estoi_min" if res.get("scenario") == "bus" else "d_estoi_wrst"
    return {"scene": res["scene"], "method": res["method"], "status": res["status"], "error": res.get("error"),
            key: m.get(key), "rtf": m.get("rtf")}


def _job(job):
    """Worker: load (render) one scene once, then run and score every requested method on it."""
    scene_id, methods, o = job
    out = []
    try:
        sc = scene2.load_scene(scene_id, corpus=o["corpus"], root=o["scenes_root"])
        card = scene2.scene_card(scene_id, root=o["scenes_root"])
        ctx = metrics2.SceneCtx(sc)
    except Exception as e:
        for m in methods:
            res = {"bench": VERSION, "scene": scene_id, "method": m, "corpus": o["corpus"], "status": "error",
                   "error": f"scene: {type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            write_result(res, result_path(scene_id, m, o["corpus"], o["results_root"]))
            out.append(_summary(res))
        return out
    for m in methods:
        res, y1 = run_method(m, sc, ctx, card, extra=o["extra"])
        res.update(set=scene_id.split("/")[0], corpus=o["corpus"])
        path = result_path(scene_id, res["method"], o["corpus"], o["results_root"])
        write_result(res, path)
        if o["save_audio"] and y1 is not None:
            sf.write(path[:-len(".json")] + ".wav", y1, FS, subtype="FLOAT")
        out.append(_summary(res))
    return out


def cmd_run(a):
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        load_method(m)  # fail fast on a missing method
    filters = {}
    if a.snr:
        filters["snr_db"] = tuple(float(v) for v in a.snr.split(","))
    if a.strata:
        filters["stratum"] = tuple(a.strata.split(","))
    scenes = scene2.list_scenes(a.set, scenario=a.scenario, root=a.scenes_root, **filters)
    scenes = scenes[:a.limit] if a.limit else scenes
    if not scenes:
        print(f"no scenes in set '{a.set}' match")
        return 1
    extra = extra_geometries(a.set, a.scenes_root) if a.extra == "subset" else set()
    jobs = []
    for sid in scenes:
        card = scene2.scene_card(sid, root=a.scenes_root)
        todo = [m for m in methods if not (a.skip_existing and _done(result_path(sid, m, a.corpus, a.results_root)))]
        if todo:
            opts = {"corpus": a.corpus, "scenes_root": a.scenes_root, "results_root": a.results_root,
                    "save_audio": a.save_audio, "extra": a.extra == "all" or card["geom_id"] in extra}
            jobs.append((card["duration_s"], sid, todo, opts))
    jobs = [(sid, todo, opts) for _, sid, todo, opts in sorted(jobs, key=lambda j: -j[0])]  # long scenes first
    print(f"{len(jobs)} scenes x {len(methods)} methods, {sum(j[2]['extra'] for j in jobs)} scenes with extra "
          f"runs, {a.j} workers", flush=True)
    t0, done = time.time(), 0
    if a.j <= 1:
        results = map(_job, jobs)
    else:
        pool = get_context("spawn").Pool(a.j)
        results = pool.imap_unordered(_job, jobs)
    try:
        for rs in results:
            done += 1
            for r in rs:
                tail = f"ERROR {r['error']}" if r["status"] != "ok" else \
                    " ".join(f"{k} {v:.3f}" for k, v in r.items() if k not in ("scene", "method", "status", "error")
                             and v is not None)
                print(f"[{done}/{len(jobs)} {time.time() - t0:6.0f} s] {r['scene']} {r['method']}: {tail}", flush=True)
    finally:
        if a.j > 1:
            pool.close()
            pool.join()
    rows = load_rows(a.set, a.corpus, methods, a.scenario, a.results_root)
    print(render_table(rows))
    return 0


def _done(path):
    try:
        with open(path) as f:
            return json.load(f).get("status") == "ok"
    except (OSError, ValueError):
        return False


# ----------------------------------------------------------------------------- 'table'
RT60_BANDS = (0.3, 0.45, 0.6, 0.8)  # used when the scene card has no 'rt60_bin'
E2_BINS = (0.45, 0.65)              # bus partner distance bins (DESIGN E2) of the farther partner


def _band(v, edges, lo=0.0, hi=math.inf, unit=""):
    e = [lo, *edges, hi]
    i = int(np.searchsorted(edges, v, side="right"))
    return f"{e[i]:g}-{e[i + 1]:g}{unit}"


def group_value(row, by):
    """Value of a breakdown key for a row: stratum, spacing (mm), snr, dist_bin (E2 bin of the farther bus
    partner), rt60 (the card's rt60_bin, else a band), tail, corpus, or any scene-card field."""
    c = stats.card(row)
    if by == "corpus":
        return row.get("corpus")
    if by == "spacing":
        return f"{round(c['spacing_m'] * 1000)} mm" if c.get("spacing_m") else None
    if by == "snr":
        return c.get("snr_db")
    if by == "dist_bin" and "dist_bin" not in c:
        return _band(c["partner_dist_max_m"], E2_BINS, 0.3, 1.0, " m") if c.get("partner_dist_max_m") else None
    if by == "rt60" and "rt60_bin" not in c:
        return _band(c["rt60"], RT60_BANDS, unit=" s") if c.get("rt60") else None
    return c.get("rt60_bin" if by == "rt60" else by)


def _fmt(v, f):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100 * v:.0f}" if f == "%" else format(v, f)


def _mean_over_geometries(rows, key):
    pg = stats.per_geometry(rows, key)
    return float(np.mean([v for v, _ in pg.values()])) if pg else float("nan")


def _summary_table(rs, scenario, by=None):
    cols = COLUMNS[scenario] + LEGACY[scenario]
    lines = [f"\n### {scenario}" + (f" by {by}" if by else ""), "",
             "| method | " + ("group | " if by else "") + "scenes | geoms | err | " + " | ".join(h for _, h, _ in cols)
             + " |", "|---|" + ("---|" if by else "") + "---|---|---|" + "---|" * len(cols)]
    for m in sorted({r["method"] for r in rs}):
        mr = [r for r in rs if r["method"] == m]
        groups = sorted({group_value(r, by) for r in mr}, key=lambda v: (v is None, str(v))) if by else [None]
        for gv in groups:
            gr = [r for r in mr if not by or group_value(r, by) == gv]
            ok = [r for r in gr if r.get("status") == "ok"]
            vals = [_fmt(_mean_over_geometries(ok, k), f) for k, _, f in cols]
            lines.append(f"| {m} | " + (f"{gv} | " if by else "") + f"{len(ok)} | {len({stats.geometry(r) for r in ok})}"
                         f" | {len(gr) - len(ok)} | " + " | ".join(vals) + " |")
    return lines


def _bubble_table(rs):
    """B7: mean probe level change relative to the partners per probe distance; bubble = 0.6 m minus 2 m."""
    probe = [r for r in rs if r.get("status") == "ok" and np.isfinite(stats.metric(r, "probe_rel_db"))]
    if not probe:
        return []
    dists = sorted({round(stats.metric(r, "probe_dist_m"), 1) for r in probe})
    lines = ["", "B7 probe level relative to the partners (dB) by probe distance (m):", "",
             "| method | " + " | ".join(f"{d:g}" for d in dists) + " | bubble (0.6 vs 2 m) |",
             "|---|" + "---|" * (len(dists) + 1)]
    for m in sorted({r["method"] for r in probe}):
        rel = {d: [stats.metric(r, "probe_rel_db") for r in probe
                   if r["method"] == m and round(stats.metric(r, "probe_dist_m"), 1) == d] for d in dists}
        mean = {d: float(np.mean(v)) if v else float("nan") for d, v in rel.items()}
        bubble = mean.get(0.6, float("nan")) - mean.get(2.0, float("nan"))
        lines.append(f"| {m} | " + " | ".join(_fmt(mean[d], "+.1f") for d in dists) + f" | {_fmt(bubble, '+.1f')} |")
    return lines


def _gate_tables(rs, scenario):
    lines = ["", f"Gates ({scenario}; G0 on all scenes, G1 without the bus stress SNR; decisive only on natural "
             "speech, DESIGN 4.6):", "", "| method | G0 | G1 | failing criteria |", "|---|---|---|---|"]
    detail = ["", "| method | gate | criterion | n | point | 95% CI | pass |", "|---|---|---|---|---|---|---|"]
    for m in sorted({r["method"] for r in rs}):
        mr = [r for r in rs if r["method"] == m]
        res = stats.evaluate_gates(mr, scenario, offline_only=any(r.get("offline_only") for r in mr))
        (g0, f0), (g1, f1) = stats.gate_summary(res, "G0"), stats.gate_summary(res, "G1")
        lines.append(f"| {m} | {g0} | {g1} | {'; '.join(f0 + f1) or '—'} |")
        for g in res:
            ci = f"[{_fmt(g['lo'], '.3g')}, {_fmt(g['hi'], '.3g')}]" if np.isfinite(g["lo"]) else "—"
            verdict = {True: "yes", False: "NO", None: g.get("note", "n/a")}[g["pass"]]
            detail.append(f"| {m} | {g['gate']} | {g['desc']} | {g['n']} | {_fmt(g['point'], '.3g')} | {ci} | {verdict} |")
    return lines + detail


def render_table(rows, by=()):
    """Markdown summary per scenario: decision columns with legacy columns last, optional breakdowns (`by`:
    keys for group_value), the B7 bubble, then the G0/G1 gates. Values are means over geometries of
    per-geometry means (DESIGN 4.5)."""
    lines = []
    for scenario in ("bus", "meeting"):
        rs = [r for r in rows if r.get("scenario") == scenario]
        if rs:
            lines += _summary_table(rs, scenario)
            for key in by:
                lines += _summary_table(rs, scenario, key)
            lines += _bubble_table(rs) + _gate_tables(rs, scenario)
    return "\n".join(lines) if lines else "no results"


def _selected_rows(a, methods=None):
    """Result rows for the table / compare commands, filtered by --snr and --strata like 'run'."""
    rows = load_rows(a.set, a.corpus, methods, a.scenario, a.results_root)
    if a.snr:
        snrs = {float(v) for v in a.snr.split(",")}
        rows = [r for r in rows if stats.card(r).get("snr_db") in snrs]
    if a.strata:
        rows = [r for r in rows if stats.card(r).get("stratum") in set(a.strata.split(","))]
    return rows


def cmd_table(a):
    methods = [m.strip() for m in a.methods.split(",")] if a.methods else None
    print(render_table(_selected_rows(a, methods), by=a.by.split(",") if a.by else ()))
    return 0


# ----------------------------------------------------------------------------- 'compare'
def render_compare(rows, ref, metrics=None):
    """Markdown table of paired per-geometry differences (method minus ref) with bootstrap 95% CIs,
    Wilcoxon p and Holm-corrected p across methods (per metric)."""
    lines = []
    for scenario in ("bus", "meeting"):
        rs = [r for r in rows if r.get("scenario") == scenario]
        ref_rows = [r for r in rs if r["method"] == ref]
        others = sorted({r["method"] for r in rs} - {ref})
        if not ref_rows or not others:
            continue
        lines += [f"\n### {scenario}: paired differences vs {ref} (per geometry)", "",
                  "| metric | method | n | method mean | ref mean | Δ | 95% CI | p | p (Holm) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key in metrics or COMPARE[scenario]:
            res = [stats.paired([r for r in rs if r["method"] == m], ref_rows, key) for m in others]
            adj = stats.holm([q["p"] for q in res])
            for m, q, ph in zip(others, res, adj):
                lines.append(f"| {key} | {m} | {q['n']} | {_fmt(q['mean'], '.3g')} | {_fmt(q['ref_mean'], '.3g')} | "
                             f"{_fmt(q['delta'], '+.3g')} | [{_fmt(q['lo'], '+.3g')}, {_fmt(q['hi'], '+.3g')}] | "
                             f"{_fmt(q['p'], '.3g')} | {_fmt(ph, '.3g')} |")
    return "\n".join(lines) if lines else f"nothing to compare (need results for {ref} and another method)"


def cmd_compare(a):
    methods = [m.strip() for m in a.methods.split(",")] if a.methods else None
    rows = _selected_rows(a, methods and list(set(methods) | {a.ref}))
    print(render_compare(rows, a.ref, a.metrics.split(",") if a.metrics else None))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lab.bench2", description="v2 bench: run methods over a scene set, summarize "
                                 "with gates, compare against a reference (BENCH2.md 'Bench usage')")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", required=True, help="scene set: dev, wide, test, smoke, demo")
    common.add_argument("--scenario", choices=("bus", "meeting"))
    common.add_argument("--corpus", default="tts", help="speech corpus of the scenes (default tts)")
    common.add_argument("--snr", help="comma list of bus SNRs (dB)")
    common.add_argument("--strata", help="comma list of E1 strata")
    common.add_argument("--results-root", default=None, help="default: results2/")
    common.add_argument("--scenes-root", default=None, help="default: scenes2/")
    r = sub.add_parser("run", parents=[common], help="run and score methods")
    r.add_argument("--methods", required=True, help="comma list of methods/<name>.py")
    r.add_argument("--limit", type=int, help="first N matching scenes")
    r.add_argument("-j", type=int, default=3, help="worker processes (default 3)")
    r.add_argument("--extra", choices=("subset", "all", "none"), default="subset",
                   help="extra Hagerman runs (finer bus split, consistency): 20%% subset (default), all, none")
    r.add_argument("--save-audio", action="store_true", help="also write the final output as float WAV")
    r.add_argument("--skip-existing", action="store_true", help="skip scene/method pairs with an ok result")
    t = sub.add_parser("table", parents=[common], help="markdown summary with gates")
    t.add_argument("--methods", help="comma list (default: all with results)")
    t.add_argument("--by", help="comma list of breakdowns: stratum, spacing, snr, dist_bin, rt60, tail, corpus "
                                "or any scene-card key")
    c = sub.add_parser("compare", parents=[common], help="paired differences against a reference method")
    c.add_argument("--methods", help="comma list (default: all with results)")
    c.add_argument("--ref", default="m0_mica_hpf", help="reference method (default m0_mica_hpf)")
    c.add_argument("--metrics", help="comma list of metric keys (default: the decision metrics)")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "table": cmd_table, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
