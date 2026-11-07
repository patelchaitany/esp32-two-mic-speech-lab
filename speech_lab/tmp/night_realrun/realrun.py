"""Thin wrapper around hw/realeval.py for the real:run pass (does not modify realeval.py).

Changes against `realeval.py run`:
  * the y1 WAV is written only when REALRUN_WAV=1 (the data volume had < 5 GB free; JSONs are ~5 kB each);
  * each result also stores CPU-time RTFs (time.process_time around every enhance call: 'rtf_cpu' median,
    'rtf_cpu_max'), because wall-time RTF is inflated by the machine load (~120-200 on 10 cores);
  * each result stores 'src_sha1': sha1 of methods/<method>.py + all methods/_core/*.py at run time (methods are
    being re-tuned by another run tonight), and the load average at the start of the call.
Usage: identical to realeval.py (run/table/list), e.g.
  OMP_NUM_THREADS=1 .venv/bin/python -I tmp/night_realrun/realrun.py run --methods m4_cdr --set bus -j 2 --skip-existing
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(LAB, "hw"), LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import soundfile as _sf  # noqa: E402

import realeval  # noqa: E402
from lab import bench2, metrics2, scene2  # noqa: E402

FS = realeval.FS


class _SF:
    """soundfile shim: write only when REALRUN_WAV=1."""

    def __getattr__(self, k):
        return getattr(_sf, k)

    @staticmethod
    def write(path, *a, **k):
        if os.environ.get("REALRUN_WAV", "0") == "1":
            _sf.write(path, *a, **k)


def src_sha1(method):
    h = hashlib.sha1()
    for p in [os.path.join(LAB, "methods", method + ".py")] + sorted(glob.glob(os.path.join(LAB, "methods", "_core", "*.py"))):
        try:
            h.update(open(p, "rb").read())
        except OSError:
            h.update(b"missing:" + p.encode())
    return h.hexdigest()[:12]


def run_one(method, sc, ctx, info):
    """realeval.run_one + CPU-time RTF + source hash."""
    T = int(sc["mix"].shape[1])
    res = {"bench": realeval.VERSION + "+realrun", "scene": sc["id"], "method": method,
           "scenario": sc["meta"]["scenario"], "card": sc["meta"]["card"], "info_keys": sorted(info),
           "src_sha1": src_sha1(method), "loadavg": os.getloadavg()[0]}
    y1 = None
    try:
        mod, name, accepts = bench2.load_method(method)

        def call(x):
            return mod.enhance(x, FS, dict(info), return_stages=True) if accepts else mod.enhance(x, FS, dict(info))

        runs, comps = metrics2.plan(sc, extra=False)
        outs, timing, peaks, t_m = {}, [], {}, 0.0
        for run, flip in runs:
            x = scene2.signed_mix(sc, flip=flip)
            t0, c0 = time.perf_counter(), time.process_time()
            r = call(x)
            dt, dc = time.perf_counter() - t0, time.process_time() - c0
            t_m += dt
            outs[run], peaks[run] = bench2._stages(r, T)
            timing.append({"run": run, "flip": list(flip), "rtf": dt / (T / FS), "rtf_cpu": dc / (T / FS)})
        y1 = outs["y1"]["out"]
        t0 = time.perf_counter()
        scored = metrics2.score(sc, outs, comps, ctx)
        rt = [t["rtf"] for t in timing]
        rc = [t["rtf_cpu"] for t in timing]
        scored["m"].update(rtf=float(np.median(rt)), rtf_max=float(max(rt)), rtf_cpu=float(np.median(rc)),
                           rtf_cpu_max=float(max(rc)), peak=peaks["y1"], clipped=float(peaks["y1"] >= 1.0),
                           n_runs=float(len(runs)))
        scored["own"] = realeval.own_metrics(sc, np.asarray(y1, np.float64), ctx)
        scored["m"].update(realeval.gap_metrics(sc, ctx, outs, comps))
        res.update(status="ok", runs=timing, stages=any(o["pre"] is not o["out"] for o in outs.values()),
                   algo_latency_ms=getattr(mod, "ALGO_LATENCY_MS", None),
                   time_s={"method": t_m, "score": time.perf_counter() - t0}, **scored)
    except Exception as e:  # report and go on
        res.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:])
    return res, y1


# patched at import time, so the spawned workers (which re-import this file as __mp_main__) get it too
realeval.sf = _SF()
realeval.run_one = run_one

if __name__ == "__main__":
    realeval.main()
