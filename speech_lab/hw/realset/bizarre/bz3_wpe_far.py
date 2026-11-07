"""Exp 3: WPE dereverberation on the real takes, esp. the emulated far talkers MF1/MF2.

Target ('direct') = dry speech convolved with the first 5 ms after the direct path of the dry->Mic A response
(near takes: the fitted 4096-tap h; MF takes: g * h_far * h, h_far re-generated from hw/realset.py far_rir(seed=7)).
Also 'early50' = direct + 50 ms. Inputs: 'img' (fitted image, no noise) and 'rec' (the real recording, fan noise
at -2..+7 dB SNR). Processing: none (S0 80 Hz prefilter only), m10_wpe (2-ch, tuned 30 taps), 1-ch WPE (Mic A only),
2-ch WPE with 60 taps. Scores at Mic A over the speech (+0.3 s) interval: ESTOI, in-band SDR (lab.metrics2.sdr_fir,
512-tap distortion FIR) and SI-SDR (lab.metrics after alignment) against both targets.
Run: .venv/bin/python -I hw/realset/bizarre/bz3_wpe_far.py [-j 2] -> bz3_wpe_far.json (+ tmp/night_bizarre/wpe_out/*.npz)
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import FS, HERE, LAB, dump  # noqa: E402

OUTD = os.path.join(LAB, "tmp", "night_bizarre", "wpe_out")
TAKES = ["MF1", "MF2", "M1", "M2", "M3", "M4", "M5", "M6", "A1", "B1"]


def run_take(t):
    from lab import metrics as v1, metrics2 as m2
    from methods._core import calib, wpe
    import bz_takes as bt
    tk = bt.composite(t)
    iv = bt.speech_iv(tk)
    tgt = {"direct": bt.render(tk, tk["p"] + 80)[0], "early50": bt.render(tk, tk["p"] + 800)[0]}
    res = {"take": t, "far": tk["far"], "rebuild_err_db": tk["rebuild_err_db"],
           "drr_c50_A": bt.drr_c50(tk["comp"][:, 0], tk["p"]), "iv": iv}
    outs = {}
    for cond, x in (("img", tk["image"]), ("rec", tk["recording"])):
        xp = calib.prefilter(x, "meeting")
        t0 = time.time()
        y = {"none": xp[0],
             "wpe2": wpe.dereverb(xp, FS)[0],
             "wpe1": wpe.dereverb(xp[:1], FS)[0],
             "wpe2_60": wpe.dereverb(xp, FS, taps=60)[0]}
        res[f"{cond}_wpe_s"] = time.time() - t0
        for k, v in y.items():
            outs[f"{cond}_{k}"] = v.astype(np.float32)
            for tn, r in tgt.items():
                a, b = iv[0]
                ya, _ = v1.align(v[a:b], r[a:b], max_lag=160)
                res[f"{cond}_{k}_{tn}"] = {"estoi": m2.estoi(v, r, iv), "sdr": m2.sdr_fir(v, r, iv),
                                           "si_sdr": v1.si_sdr(ya, r[a:b])}
            # energy removed by the processing over the speech interval (dB): a blind reverberation feature
            res[f"{cond}_{k}_energy_db"] = float(10 * np.log10(np.sum(v[iv[0][0]:iv[0][1]] ** 2) /
                                                               np.sum(xp[0][iv[0][0]:iv[0][1]] ** 2)))
    np.savez(os.path.join(OUTD, t + ".npz"), **outs)
    print(t, {k: (round(v["estoi"], 3), round(v["sdr"], 1)) for k, v in res.items() if isinstance(v, dict)}, flush=True)
    return res


def main():
    os.makedirs(OUTD, exist_ok=True)
    j = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 1
    takes = [a for a in sys.argv[1:] if a in TAKES] or TAKES
    t0 = time.time()
    if j > 1:
        with ProcessPoolExecutor(j) as ex:
            rows = list(ex.map(run_take, takes))
    else:
        rows = [run_take(t) for t in takes]
    dump({"rows": rows, "wall_s": time.time() - t0}, os.path.join(HERE, "bz3_wpe_far.json"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
