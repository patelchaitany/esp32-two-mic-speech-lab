"""Run enhancement methods over simulated scenes and score them.

A method is methods/<name>.py exposing  enhance(x, fs, info) -> y
  x: (2, T) float32 mic signals (row 0 = Mic A, row 1 = Mic B), fs = 16000
  info: {"scenario": "bus" | "meeting", "spacing_m": 0.04 (NOMINAL, may be wrong)}
  y: (T,) mono float output (any gain; scoring is gain-aware where it matters)
Methods must not use anything else from the scene (no positions, no references).
"""
import os, sys, json, glob, time, argparse, importlib, traceback
import numpy as np
from multiprocessing import Pool
from .common import FS, write_wav
from .scenes import load_scene, SETS
from .metrics import score_bus, score_meeting

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMINAL_SPACING = 0.04


def scene_names(setname):
    names = []
    for kind, seed, snr, d in SETS[setname]:
        mm = int(round(d * 1000))
        names.append(f"bus_s{seed}_snr{snr:+d}_d{mm}mm" if kind == "bus" else f"meeting_s{seed}_d{mm}mm")
    return names


def run_one(args):
    method, scene_name, setname, save_audio = args
    outdir = os.path.join(HERE, "results", setname, method)
    os.makedirs(outdir, exist_ok=True)
    try:
        mod = importlib.import_module(f"methods.{method}")
        sc = load_scene(os.path.join(HERE, "scenes", scene_name + ".npz"))
        info = {"scenario": sc["meta"]["scenario"], "spacing_m": NOMINAL_SPACING}
        t0 = time.time()
        y = np.asarray(mod.enhance(sc["mix"].copy(), FS, dict(info)), dtype=np.float64).reshape(-1)
        dt = time.time() - t0
        T = sc["mix"].shape[1]
        y = y[:T] if len(y) >= T else np.pad(y, (0, T - len(y)))
        if not np.all(np.isfinite(y)):
            raise ValueError("non-finite output")
        r = score_bus(y, sc) if info["scenario"] == "bus" else score_meeting(y, sc)
        r.update(scene=scene_name, method=method, rtf=dt / (T / FS), scenario=info["scenario"],
                 true_spacing=sc["meta"]["spacing_m"])
        if save_audio:
            write_wav(os.path.join(outdir, scene_name + ".wav"), y / (np.abs(y).max() + 1e-9) * 0.9)
    except Exception as e:
        r = dict(scene=scene_name, method=method, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    with open(os.path.join(outdir, scene_name + ".json"), "w") as f:
        json.dump(r, f)
    return r


BUS_KEYS = ["si_sdr_imp", "estoi", "estoi_worst_target", "pesq", "bg_reduction_db"]
MTG_KEYS = ["estoi_mean", "estoi_worst", "si_sdr_imp", "loud_std_db", "gap_snr_db", "pesq"]


def summarize(setname, methods=None):
    rows = {}
    for p in glob.glob(os.path.join(HERE, "results", setname, "*", "*.json")):
        r = json.load(open(p))
        if methods and r.get("method") not in methods:
            continue
        rows.setdefault(r.get("method"), []).append(r)
    out = {}
    for m, rs in sorted(rows.items()):
        errs = [r for r in rs if "error" in r]
        ok = [r for r in rs if "error" not in r]
        bus = [r for r in ok if r["scenario"] == "bus"]
        mtg = [r for r in ok if r["scenario"] == "meeting"]
        s = {"n_err": len(errs), "n_bus": len(bus), "n_mtg": len(mtg)}
        for k in BUS_KEYS:
            s["bus_" + k] = float(np.nanmean([r[k] for r in bus])) if bus else float("nan")
        for snr in (-5, 0, 5):
            sub = [r for r in bus if f"snr{snr:+d}_d40mm" in r["scene"]]
            if sub:
                s[f"bus_estoi_snr{snr:+d}"] = float(np.nanmean([r["estoi"] for r in sub]))
        for k in MTG_KEYS:
            s["mtg_" + k] = float(np.nanmean([r[k] for r in mtg])) if mtg else float("nan")
        s["rtf"] = float(np.mean([r["rtf"] for r in ok])) if ok else float("nan")
        if errs:
            s["first_error"] = errs[0]["error"]
        out[m] = s
    return out


def print_table(summary):
    hdr = (f"{'method':28s} | {'dSISDR':>6s} {'ESTOI':>5s} {'wrstT':>5s} {'PESQ':>4s} {'BR':>5s} |"
           f" {'ESTOI':>5s} {'worst':>5s} {'dSISDR':>6s} {'loudσ':>5s} {'gapSNR':>6s} {'PESQ':>4s} | {'RTF':>5s}")
    print(" " * 31 + "BUS (keep the 2 near talkers)" + " " * 8 + "MEETING (keep all 20, even level)")
    print(hdr)
    print("-" * len(hdr))
    for m, s in summary.items():
        print(f"{m:28s} | {s['bus_si_sdr_imp']:6.2f} {s['bus_estoi']:5.3f} {s['bus_estoi_worst_target']:5.3f} {s['bus_pesq']:4.2f} {s['bus_bg_reduction_db']:5.1f} |"
              f" {s['mtg_estoi_mean']:5.3f} {s['mtg_estoi_worst']:5.3f} {s['mtg_si_sdr_imp']:6.2f} {s['mtg_loud_std_db']:5.1f} {s['mtg_gap_snr_db']:6.1f} {s['mtg_pesq']:4.2f} | {s['rtf']:5.3f}"
              + (f"  ERR x{s['n_err']}: {s.get('first_error','')[:60]}" if s["n_err"] else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True, help="comma list of methods/<name>.py")
    ap.add_argument("--set", default="dev", choices=list(SETS))
    ap.add_argument("--only", default="", help="substring filter on scene names")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    jobs = [(m, s, a.set, a.save_audio) for m in methods for s in scene_names(a.set) if a.only in s]
    with Pool(a.j) as p:
        for r in p.imap_unordered(run_one, jobs):
            if "error" in r:
                print("ERROR", r["method"], r["scene"], r["error"], flush=True)
    print_table(summarize(a.set, methods))
