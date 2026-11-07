"""Exp 6c: the most a time-invariant 2-mic spatial filter can buy against the REAL noise of this board (oracle).

Per take and STFT bin: R_n = covariance of the take's own noise (recording - image) over the speech interval,
h = principal eigenvector of the image covariance over the same interval (the target's RTF incl. reverberation's
average), w = R_n^-1 h / (h^H R_n^-1 h) (oracle MVDR, scaled to Mic A's component of h). SNR gain vs Mic A in
300-4000 Hz and per band. Also the same with R_n from the noise-only TAIL (what a calibration-from-pauses system
could know). Linear, so image and noise are scored separately. Both mics, board calibration applied first.
Run: .venv/bin/python -I hw/realset/bizarre/bz6c_oracle_mvdr.py -> bz6c_oracle_mvdr.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import CALIB, FS, HERE, dump  # noqa: E402

BANDS = [(150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]


def main():
    from methods._core import calib, common as cm
    import bz_takes as bt
    f = cm.FREQS
    res = {}
    for t in bt.TAKES:
        tk = bt.composite(t)
        (a, b), = bt.speech_iv(tk, extra_s=0.0)
        cal = calib.calibrate(cm.stft(calib.prefilter(tk["recording"], "meeting")),
                              {"scenario": "meeting", "spacing_m": CALIB["spacing_m"], "calib_json": CALIB})
        Xi = calib.apply(cm.stft(calib.prefilter(tk["image"], "meeting")), cal)
        Xn = calib.apply(cm.stft(calib.prefilter(tk["noise"], "meeting")), cal)
        cen = cm.frame_centers(Xi.shape[-1])
        sp = (cen >= a) & (cen < b)
        tail = (cen >= tk["onset"] + int(4.9 * FS)) & (cen < tk["N"] - FS // 4)
        Ri = np.einsum("cfn,dfn->fcd", Xi[:, :, sp], np.conj(Xi[:, :, sp])) / sp.sum()
        out = {}
        for rn_name, sel in (("speech_interval", sp), ("tail", tail)):
            Rn = np.einsum("cfn,dfn->fcd", Xn[:, :, sel], np.conj(Xn[:, :, sel])) / sel.sum()
            W = np.zeros((len(f), 2), complex)
            for k in range(len(f)):
                ev, V = np.linalg.eigh(Ri[k])
                h = V[:, -1] / (V[0, -1] + 1e-20)
                Rk = Rn[k] + 1e-9 * np.trace(Rn[k]).real * np.eye(2)
                v = np.linalg.solve(Rk, h)
                W[k] = v / (np.conj(h) @ v)
            yi = np.einsum("fc,cfn->fn", np.conj(W), Xi[:, :, sp])
            yn = np.einsum("fc,cfn->fn", np.conj(W), Xn[:, :, sp])
            row = {}
            for lo, hi in [(300, 4000)] + BANDS:
                m = (f >= lo) & (f < hi)
                e = lambda Z: np.sum(np.abs(Z[m]) ** 2)
                row[f"{lo}-{hi}"] = float(10 * np.log10(e(yi) / e(yn)) - 10 * np.log10(e(Xi[0][:, sp]) / e(Xn[0][:, sp])))
            row["target_change_300_4k"] = float(10 * np.log10(np.sum(np.abs(yi[(f >= 300) & (f < 4000)]) ** 2) /
                                                              np.sum(np.abs(Xi[0][(f >= 300) & (f < 4000)][:, sp]) ** 2)))
            out[rn_name] = row
        res[t] = out
        print(t, {k: round(v["300-4000"], 2) for k, v in out.items()}, flush=True)
    summ = {}
    for rn in ("speech_interval", "tail"):
        summ[rn] = {}
        for key in res[bt.TAKES[0]][rn]:
            v = np.array([res[t][rn][key] for t in bt.TAKES])
            summ[rn][key] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    dump({"rows": res, "summary": summ}, os.path.join(HERE, "bz6c_oracle_mvdr.json"))
    print(summ)


if __name__ == "__main__":
    main()
