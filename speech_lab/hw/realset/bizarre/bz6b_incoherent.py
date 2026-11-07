"""Exp 6b: why fixed beams gain ~0 dB on the real board: the incoherent part of the real noise.

(a) Measured: magnitude-squared coherence of each take's noise-only tail vs the ideal diffuse sinc^2(2fd/c) at
    1.19 cm (after the board calibration), per band -> the 'incoherent fraction' 1 - rho_eff(f).
(b) Theory (bz6 model) with an incoherent fraction mu = 0.01 / 0.02 / 0.05 / 0.1 (instead of the -47 dB self-noise):
    diffuse array gain of the cardioid / hypercardioid / superdirective MVDR at 1.19 / 2 / 4 cm, calibrated.
(c) Real takes: superdirective MVDR with diagonal loading mu_design in {1e-4, 0.01, 0.05, 0.2} (bz6 used 1e-4),
    SNR gain vs Mic A on the 18 takes (median, min, max), calibration 'full'.
Run: .venv/bin/python -I hw/realset/bizarre/bz6b_incoherent.py -> bz6b_incoherent.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import C, CALIB, FS, HERE, coh_stats, dump  # noqa: E402
from bz6_skew_sensitivity import BAND, F, design, evaluate  # noqa: E402

BANDS = [(150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]


def main():
    from methods._core import calib, common as cm
    import bz_takes as bt
    res = {"measured": {}, "theory": {}, "real_sdmvdr": {}}
    d = CALIB["spacing_m"]
    # (a) measured coherence of the noise-only tails (B corrected by the board calibration)
    fr = []
    for t in bt.TAKES:
        tk = bt.composite(t)
        on = tk["onset"]
        x = tk["recording"][:, on + int(4.9 * FS):tk["N"] - FS // 4]
        X = cm.stft(calib.prefilter(x, "meeting"))
        cal = calib.calibrate(cm.stft(calib.prefilter(tk["recording"], "meeting")),
                              {"scenario": "meeting", "spacing_m": d, "calib_json": CALIB})
        y = cm.istft(calib.apply(X, cal), x.shape[1])
        f, aa, bb, ab = coh_stats(y)
        msc = np.abs(ab) ** 2 / (aa * bb)
        model = np.sinc(2 * f * d / C) ** 2
        row = {}
        for lo, hi in BANDS:
            m = (f >= lo) & (f < hi)
            row[f"{lo}-{hi}"] = {"msc": float(np.mean(msc[m])), "model_msc": float(np.mean(model[m])),
                                 "incoherent_frac": float(np.mean(1 - np.sqrt(msc[m] / model[m])))}
        res["measured"][t] = row
        fr.append([row[f"{lo}-{hi}"]["incoherent_frac"] for lo, hi in BANDS])
    fr = np.array(fr)
    res["measured_summary"] = {f"{lo}-{hi}": {"median": float(np.median(fr[:, i])), "min": float(fr[:, i].min()),
                                              "max": float(fr[:, i].max())} for i, (lo, hi) in enumerate(BANDS)}
    print("incoherent fraction per band (median over takes):", res["measured_summary"], flush=True)
    # (b) theory with an incoherent fraction
    for mu in (0.001, 0.01, 0.02, 0.05, 0.1):
        for dd in (0.0119, 0.02, 0.04):
            for kind in ("cardioid", "hyper", "sdmvdr"):
                w = design(kind, F, dd, mu if kind == "sdmvdr" else 1e-4)
                ag, _ = evaluate(w, F, dd, 0.0, 0.0, mu)
                res["theory"][f"mu={mu}|{dd * 100:.2f}cm|{kind}"] = {
                    "ag_db_300_4k": float(np.mean(ag[BAND])), "ag_db_300": float(np.interp(300, F, ag)),
                    "ag_db_1k": float(np.interp(1000, F, ag)), "ag_db_3k": float(np.interp(3000, F, ag))}
    for mu in (0.01, 0.05):
        print(mu, {k.split("|", 1)[1]: round(v["ag_db_300_4k"], 1) for k, v in res["theory"].items() if k.startswith(f"mu={mu}|")},
              flush=True)
    # (c) real takes, loaded superdirective MVDR
    f = cm.FREQS
    band = (f >= 300) & (f <= 4000)
    for mu in (1e-4, 0.01, 0.05, 0.2):
        w = design("sdmvdr", f, d, mu)
        g = []
        for t in bt.TAKES:
            tk = bt.composite(t)
            (a, b), = bt.speech_iv(tk, extra_s=0.0)
            cal = calib.calibrate(cm.stft(calib.prefilter(tk["recording"], "meeting")),
                                  {"scenario": "meeting", "spacing_m": d, "calib_json": CALIB})
            Xi = calib.apply(cm.stft(calib.prefilter(tk["image"], "meeting")), cal)
            Xn = calib.apply(cm.stft(calib.prefilter(tk["noise"], "meeting")), cal)
            cen = cm.frame_centers(Xi.shape[-1])
            sel = (cen >= a) & (cen < b)
            yi = np.conj(w[:, 0])[:, None] * Xi[0] + np.conj(w[:, 1])[:, None] * Xi[1]
            yn = np.conj(w[:, 0])[:, None] * Xn[0] + np.conj(w[:, 1])[:, None] * Xn[1]
            e = lambda Z: np.sum(np.abs(Z[band][:, sel]) ** 2)
            g.append(10 * np.log10(e(yi) / e(yn)) - 10 * np.log10(e(Xi[0]) / e(Xn[0])))
        g = np.array(g)
        res["real_sdmvdr"][str(mu)] = {"median": float(np.median(g)), "min": float(g.min()), "max": float(g.max()),
                                       "per_take": dict(zip(bt.TAKES, g.round(2).tolist()))}
        print("real sdmvdr mu", mu, res["real_sdmvdr"][str(mu)]["median"], g.min(), g.max(), flush=True)
    dump(res, os.path.join(HERE, "bz6b_incoherent.json"))


if __name__ == "__main__":
    main()
