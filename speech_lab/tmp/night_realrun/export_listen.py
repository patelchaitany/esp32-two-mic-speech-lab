"""Listening WAVs for hw/realset/listen/ (real:run). For each (scene, method): one enhance() call on the scene mix
with the 'measured' info (calib_session.json, clean enrollment), output y1 and the raw Mic A input, each peak-normalised
to -1 dBFS (16-bit PCM); index.json lists every file with the applied gain (so the original relative levels are
recoverable), the scene card and the scene's scored metrics from hw/realset/results/measured/<method>/.
  .venv/bin/python -I tmp/night_realrun/export_listen.py --pairs bus/L1_snr+0_recorded:m8_dpdfnet,... --rank-note "..."
"""
import argparse
import json
import os
import sys

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LAB, os.path.join(LAB, "hw", "realset", "build"), os.path.join(LAB, "hw")]
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import realeval  # noqa: E402
import realscenes as rs  # noqa: E402
from lab import bench2  # noqa: E402

OUT = os.path.join(LAB, "hw", "realset", "listen")
FS = 16000
PEAK = 10 ** (-1 / 20)


def write(path, y):
    y = np.asarray(y, np.float64)
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    g = PEAK / pk if pk > 0 else 1.0
    sf.write(path, np.clip(y * g, -1, 1), FS, subtype="PCM_16")
    return {"gain_db": 20 * np.log10(g), "orig_peak_dbfs": 20 * np.log10(pk) if pk > 0 else None}


KEYS_BUS = ("d_estoi_min", "estoi_wrst", "estoi_wrst_in", "dsnr_active", "na_active", "dsir_near", "ta_min", "ta_mean")
KEYS_MEET = ("capture_rate", "d_estoi_mean", "d_estoi_wrst", "d_estoi_far", "lvl_std", "lvl_range", "na_active")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rank-note", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    idx_path = os.path.join(OUT, "index.json")
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {"files": [], "notes": []}
    have = {e["file"] for e in index["files"]}
    if a.rank_note and a.rank_note not in index["notes"]:
        index["notes"].append(a.rank_note)
    for pair in a.pairs.split(","):
        sid, method = pair.split(":")
        sc = rs.load_scene(sid)
        scen = sc["meta"]["scenario"]
        x = np.asarray(sc["mix"], np.float64)
        base = sid.replace("/", "__")
        fin = f"{base}__input_micA.wav"
        if fin not in have:
            e = write(os.path.join(OUT, fin), x[0])
            index["files"].append(dict(file=fin, scene=sid, role="input", method="mic_a (raw Mic A, no processing)",
                                       card=sc["meta"]["card"], **e))
            have.add(fin)
        mod, _, accepts = bench2.load_method(method)
        info = realeval.make_info(scen, "measured", "clean")
        r = mod.enhance(x, FS, dict(info), return_stages=True) if accepts else mod.enhance(x, FS, dict(info))
        y = r["out"] if isinstance(r, dict) else r
        fo = f"{base}__{method}.wav"
        e = write(os.path.join(OUT, fo), y)
        res_p = os.path.join(realeval.RESULTS, "measured", method, base + ".json")
        mets = {}
        if os.path.exists(res_p):
            m = json.load(open(res_p)).get("m", {})
            mets = {k: m.get(k) for k in (KEYS_BUS if scen == "bus" else KEYS_MEET)}
        doc = (mod.__doc__ or "").strip().split("\n")[0]
        index["files"] = [f for f in index["files"] if f["file"] != fo]
        index["files"].append(dict(file=fo, scene=sid, role="output", method=method, variant="measured",
                                   enroll="clean", description=doc, card=sc["meta"]["card"], metrics=mets, **e))
        print(fo, e, flush=True)
    json.dump(index, open(idx_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
