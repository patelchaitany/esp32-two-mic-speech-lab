"""Exp 3b: WPE benefit vs SNR on the far takes (and two near controls): image + the take's own fan noise scaled to a
target SNR at Mic A (in-band, speech interval). Same targets/scores as bz3_wpe_far.py ('direct' = dry * first 5 ms
after the direct path of the dry->Mic A response). The image is an exact 4096-tap LTI convolution, so as the noise
vanishes WPE meets its own signal model (an upper bound; real rooms are not exactly LTI between takes).
Run: .venv/bin/python -I hw/realset/bizarre/bz3b_wpe_snr.py -> bz3b_wpe_snr.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from bz_common import FS, HERE, dump  # noqa: E402

BAND = butter(4, [150, 7000], "band", fs=FS, output="sos")


def main():
    from lab import metrics2 as m2
    from methods._core import calib, wpe
    import bz_takes as bt
    res = {}
    for t in ("MF1", "MF2", "M1", "B1"):
        tk = bt.composite(t)
        iv = bt.speech_iv(tk)
        (a, b), = iv
        tgt = bt.render(tk, tk["p"] + 80)[0]
        pi = np.mean(sosfiltfilt(BAND, tk["image"][0])[a:b] ** 2)
        pn = np.mean(sosfiltfilt(BAND, tk["noise"][0])[a:b] ** 2)
        res[t] = {"rec_snr_db": float(10 * np.log10(pi / pn))}
        for snr in (30, 20, 10, 5, 0):
            x = tk["image"] + tk["noise"] * np.sqrt(pi / pn / 10 ** (snr / 10))
            xp = calib.prefilter(x, "meeting")
            r = {}
            for k, y in (("none", xp[0]), ("wpe2", wpe.dereverb(xp, FS)[0]), ("wpe1", wpe.dereverb(xp[:1], FS)[0])):
                r[k] = {"estoi": m2.estoi(y, tgt, iv), "sdr": m2.sdr_fir(y, tgt, iv)}
            res[t][str(snr)] = r
            print(t, snr, {k: (round(v["estoi"], 3), round(v["sdr"], 1)) for k, v in r.items()}, flush=True)
    dump(res, os.path.join(HERE, "bz3b_wpe_snr.json"))


if __name__ == "__main__":
    main()
