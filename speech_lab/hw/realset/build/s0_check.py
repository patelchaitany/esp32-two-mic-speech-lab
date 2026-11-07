"""Check the session calibration through S0 itself (methods/_core/calib.py, read-only use).

For each scene in the list: run S0 offline on the scene mix (a) with info['calib_json'] = calib_session.json,
(b) blind with the 1.19 cm prior, (c) blind with the 4 cm prior; report S0's d_hat, tau0 (samples), mean B gain
correction, and the phase of the device-noise cross-spectrum E[A conj(B corr)] after S0's own correction
(should be ~0 when the skew is right). Writes hw/realset/s0_check.json.

  .venv/bin/python -I hw/realset/build/s0_check.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.dirname(os.path.dirname(ROOT))
for p in (HERE, LAB):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import realscenes as rs  # noqa: E402
from methods._core import calib  # noqa: E402
from methods._core import common as cm  # noqa: E402
from methods._core import tracker  # noqa: E402

SCENES = ["bus/L1_snr+0_recorded", "bus/L1_snr+0_quiet", "bus/L3_snr-5_quiet", "meeting/recorded", "meeting/quiet",
          "meeting/quiet_omeeting"]
BANDS = [(150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]


def noise_phase(sc, cal, scenario):
    """Median phase (deg) per band of E[A conj(B corr)] over the device-noise component, after S0's prefilter."""
    nz = np.asarray(sc["comp2"]["device"], np.float64)
    X = cm.stft(calib.prefilter(nz, scenario))
    Y = calib.apply(X, cal)
    S = np.mean(Y[0] * np.conj(Y[1]), -1)
    f = cm.FREQS
    return [round(float(np.degrees(np.angle(np.mean(S[(f >= lo) & (f < hi)])))), 1) for lo, hi in BANDS]


def summarize(cal):
    return {"d_hat_cm": round(cal.d_hat * 100, 3), "tau0_samples": round(cal.tau0 * 16000, 3),
            "gain_db_mean": round(float(np.mean(cal.gain_db)), 2), "source": cal.source,
            "confidence": round(float(getattr(cal, "confidence", float("nan"))), 3)}


def main():
    cfg = rs._calib()
    out = {}
    for sid in SCENES:
        sc = rs.load_scene(sid)
        scen = sc["meta"]["scenario"]
        X = cm.stft(calib.prefilter(np.asarray(sc["mix"], np.float64), scen))
        r = {}
        for name, info in (("measured", {"scenario": scen, "spacing_m": cfg["spacing_m"], "calib_json": cfg}),
                           ("blind", {"scenario": scen, "spacing_m": cfg["spacing_m"]}),
                           ("wrong4cm", {"scenario": scen, "spacing_m": 0.04})):
            cal = calib.calibrate(X, info)
            r[name] = dict(summarize(cal), noise_phase_deg=noise_phase(sc, cal, scen))
            if scen == "bus":  # S1's two-partner TDOAs (samples; + = B later) on the calibrated STFT
                trk = tracker.track(calib.apply(X, cal), cal, info)
                r[name]["s1_tau_A"] = None if trk.tau_A is None else round(float(trk.tau_A) * 16000, 3)
                r[name]["s1_tau_B"] = None if trk.tau_B is None else round(float(trk.tau_B) * 16000, 3)
                r[name]["d_over_c"] = round(cal.d_hat / 343.0 * 16000, 3)
        r["uncorrected_noise_phase_deg"] = noise_phase(sc, calib.identity(cfg["spacing_m"], scen), scen)
        out[sid] = r
        print(sid, json.dumps(r), flush=True)
    out["bands_hz"] = BANDS
    with open(os.path.join(ROOT, "s0_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
