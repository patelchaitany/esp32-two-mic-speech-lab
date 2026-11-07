"""Enhance a two-mic ESP32 recording with the final pipelines.

  .venv/bin/python enhance.py INPUT.wav -m MODE [-o OUT.wav] [--calib calib.json] [--spacing-cm 1.25]

MODE
  meeting          x_meeting_v3: self-calibration -> WPE -> steered beam -> GTCRN -> per-talker leveling
  meeting-classic  x_meeting_v3_classic: same chain with classical noise suppression instead of GTCRN
  bus              x_bus_v3: self-calibration -> DPDFNet on Mic A -> remix -> partner-gated AGC
  bus-classic      x_bus_v3_classic: RTF-MVDR beam -> CDR and noise masks -> remix -> AGC (no neural model)
  bus-endfire      x_combo_endfire_dpdf: fixed beam along the mic axis toward Mic A's end, then DPDFNet.
                   Only for partners in line with the mics on Mic A's side; turned around it removes them.

Input: stereo WAV at 16 kHz from record.py (left = Mic A = ch0, right = Mic B = ch1), 16-bit or float.
Output: mono 16-bit WAV, peak at most -1 dBFS. Default name: INPUT_<mode>.wav.
Without --calib, the board is calibrated from the recording's own room noise (methods/x_s0v2.py); the estimates are
printed. Keep at least 0.5 s without speech in the recording (1-2 s is safer).
"""
import argparse
import json
import os
import sys
import time

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

MODES = {
    "meeting": ("x_meeting_v3", "meeting"),
    "meeting-classic": ("x_meeting_v3_classic", "meeting"),
    "bus": ("x_bus_v3", "bus"),
    "bus-classic": ("x_bus_v3_classic", "bus"),
    "bus-endfire": ("x_combo_endfire_dpdf", "bus"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-m", "--mode", required=True, choices=sorted(MODES))
    ap.add_argument("-o", "--output")
    ap.add_argument("--calib", help="calib.json (methods/_core/API.md schema); default: self-calibrate")
    ap.add_argument("--spacing-cm", type=float, default=1.25, help="mic spacing prior in cm (default 1.25)")
    a = ap.parse_args()

    data, fs = sf.read(a.input, always_2d=True, dtype="float32")
    if fs != 16000:
        sys.exit(f"{a.input}: sample rate {fs} Hz; the pipelines expect 16000 Hz")
    if data.shape[1] != 2:
        sys.exit(f"{a.input}: {data.shape[1]} channel(s); expected stereo (left = Mic A, right = Mic B)")
    x = np.ascontiguousarray(data.T)
    x -= x.mean(axis=1, keepdims=True)
    name, scenario = MODES[a.mode]
    info = {"scenario": scenario, "spacing_m": a.spacing_cm / 100.0}

    from methods import x_s0v2
    if a.calib:
        info["calib_json"] = json.load(open(a.calib))
        print(f"calibration: {a.calib}")
    else:
        cj = x_s0v2.estimate(x.astype(np.float64), fs, dict(info))
        info["calib_json"] = cj
        tau = cj["tau0_s"] * fs
        gains = np.asarray(cj.get("gain_db", [np.nan]), float)
        print(f"self-calibration: Mic B skew {tau:+.3f} samples ({cj['tau0_s'] * 1e6:+.1f} us, negative = B early), "
              f"Mic B gain correction {np.nanmedian(gains):+.2f} dB (median of bands), "
              f"confidence {cj.get('confidence', float('nan')):.2f}")
        if not -1.3 <= tau <= -0.8:
            print("  note: this board normally reads -0.8 to -1.3 samples. Check the wiring or the channel order, "
                  "or record 1-2 s of room noise without speech.")

    import importlib
    mod = importlib.import_module(f"methods.{name}")
    t0 = time.time()
    r = mod.enhance(x.copy(), fs, dict(info))
    y = np.asarray(r["out"] if isinstance(r, dict) else r, np.float64)
    dt = time.time() - t0
    if not np.all(np.isfinite(y)):
        sys.exit("output has NaN/Inf; not written")
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    lim = 10 ** (-1 / 20)
    if pk > lim:
        y *= lim / pk
    elif 0 < pk < 10 ** (-6 / 20):  # bus-endfire has no AGC: bring quiet output up for listening
        print(f"output peak {20 * np.log10(pk):.1f} dBFS; normalized to -1 dBFS")
        y *= lim / pk
    out = a.output or os.path.splitext(a.input)[0] + f"_{a.mode}.wav"
    sf.write(out, np.clip(y, -1, 1), fs, subtype="PCM_16")
    dur = len(y) / fs
    print(f"{name}: {dur:.1f} s processed in {dt:.1f} s (real-time factor {dt / max(dur, 1e-9):.2f}) -> {out}")


if __name__ == "__main__":
    main()
