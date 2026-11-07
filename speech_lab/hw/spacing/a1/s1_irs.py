"""Room-response windows (hw/irs.npz): phase slope (direct vs reverberant), per-band slopes, SNR against the
deconvolved fan-noise floor, coherence fits (sinc / J0) per window, and the direct-minus-reverb TDOA per speaker.

Coherence inside a window is estimated by frequency smoothing: one zero-padded FFT of the (tapered) window,
then P_AB, P_AA, P_BB are each smoothed with a boxcar of SMOOTH_HZ before normalising.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FS, HERE, fit_diffuse, irs, phase_slope, r3  # noqa: E402

N = 16384
SMOOTH_HZ = 250.0
fN = np.fft.rfftfreq(N, 1 / FS)
P0 = 80  # direct-sound peak index in each IR

WINDOWS_MS = {"direct_-1_1.5": (-1.0, 1.5), "direct_-1_3": (-1.0, 3.0), "early_2_10": (2, 10), "10_40": (10, 40),
              "40_80": (40, 80), "10_80": (10, 80), "80_150": (80, 150), "80_200": (80, 200), "150_250": (150, 250),
              "250_400": (250, 400), "400_600": (400, 595)}
BANDS = [(150, 300), (300, 700), (700, 1200), (1200, 2000), (2000, 3000), (3000, 4500), (4500, 6500), (6500, 7500)]


def tukey(n, a=0.1):
    from scipy.signal.windows import tukey as tk
    return tk(n, a)


def win_spectra(h, t0_ms, t1_ms):
    i0, i1 = P0 + int(round(t0_ms * FS / 1000)), P0 + int(round(t1_ms * FS / 1000))
    seg = h[i0:i1] * tukey(i1 - i0, 0.2 if (t1_ms - t0_ms) < 5 else 0.1)[:, None]
    A = np.fft.rfft(seg[:, 0], N)
    B = np.fft.rfft(seg[:, 1], N)
    return A, B, i1 - i0


def smooth(v, hz=SMOOTH_HZ):
    k = max(1, int(round(hz / (FS / N))))
    ker = np.ones(k) / k
    if np.iscomplexobj(v):
        return np.convolve(v.real, ker, "same") + 1j * np.convolve(v.imag, ker, "same")
    return np.convolve(v, ker, "same")


def coh(A, B):
    Pab, Paa, Pbb = smooth(A * np.conj(B)), smooth(np.abs(A) ** 2), smooth(np.abs(B) ** 2)
    return Pab / np.sqrt(Paa * Pbb), Pab, Paa, Pbb


def band_slopes(f, g, w):
    out = {}
    for lo, hi in BANDS:
        t1, t2, ic = phase_slope(f, g, w=w, lo=lo, hi=hi)
        out[f"{lo}-{hi}"] = [r3(t1), r3(t2), r3(ic)]
    return out


def main():
    H = irs()
    res = {"windows_ms": WINDOWS_MS, "per_ir": {}, "pooled": {}}
    # noise floor of each IR (per band, energy per sample, mic A) from its last 100 ms (decay > 60 dB by then)
    floors = {}
    for name, h in H.items():
        tail = h[-1600:]
        A = np.fft.rfft(tail[:, 0] * np.hanning(1600), N)
        floors[name] = {f"{lo}-{hi}": np.mean(np.abs(A[(fN >= lo) & (fN < hi)]) ** 2) / np.sum(np.hanning(1600) ** 2)
                        for lo, hi in BANDS}
    sel = (fN >= 150) & (fN <= 7000)
    pooled = {}
    for name, h in H.items():
        rr = {}
        for wname, (t0, t1) in WINDOWS_MS.items():
            A, B, n = win_spectra(h, t0, t1)
            g, Pab, Paa, Pbb = coh(A, B)
            # SNR vs noise floor (per band, mic A, energy per sample)
            snr = {}
            for lo, hi in BANDS:
                m = (fN >= lo) & (fN < hi)
                e = np.mean(np.abs(A[m]) ** 2) / n
                snr[f"{lo}-{hi}"] = r3(10 * np.log10(e / floors[name][f"{lo}-{hi}"]))
            # weights: raw cross-spectrum magnitude, so phase-slope is dominated by energetic bins
            w = np.abs(Pab)
            t_all, t_ls, icpt = phase_slope(fN, g, w=w, lo=300, hi=6500)
            entry = {"n": n, "snr_db_by_band": snr, "slope_300_6500": r3(t_all), "slope_ls": r3(t_ls),
                     "intercept_deg": r3(icpt), "band_slopes[max,ls,icpt]": band_slopes(fN, g, w)}
            if t0 >= 10:
                # coherence model fits on a coarse frequency grid (every 62.5 Hz) to limit correlated points
                idx = np.where(sel)[0][::16]
                for kind in ("sinc", "j0"):
                    for lo, hi in [(150, 7000), (1000, 6500), (2000, 6500)]:
                        m = (fN[idx] >= lo) & (fN[idx] <= hi)
                        entry[f"fit_{kind}_{lo}_{hi}"] = {k: r3(v) for k, v in fit_diffuse(fN[idx][m], g[idx][m], kind).items()}
            rr[wname] = entry
            pooled.setdefault(wname, []).append((Pab, Paa, Pbb))
        res["per_ir"][name] = rr
        print(name)
        for wname, e in rr.items():
            line = f"  {wname:14s} n={e['n']:5d} slope {e['slope_300_6500']:7.3f} ls {e['slope_ls']:7.3f} icpt {e['intercept_deg']:6.1f}"
            if "fit_sinc_150_7000" in e:
                fs_ = e["fit_sinc_150_7000"]
                fj = e["fit_j0_150_7000"]
                line += f" | sinc d {fs_['d_cm']:.2f} t0 {fs_['tau0']:.3f} rho {fs_['rho']:.2f} rmse {fs_['rmse']:.3f}"
                line += f" | j0 d {fj['d_cm']:.2f} rmse {fj['rmse']:.3f}"
                line += f" | 1-6.5k sinc d {e['fit_sinc_1000_6500']['d_cm']:.2f} t0 {e['fit_sinc_1000_6500']['tau0']:.3f}"
            print(line)
            print("       snr", e["snr_db_by_band"])
            print("       bands", e["band_slopes[max,ls,icpt]"])
    # pooled over the three IRs (sum of smoothed spectra; each IR has similar level)
    for wname, lst in pooled.items():
        Pab = sum(p[0] for p in lst)
        Paa = sum(p[1] for p in lst)
        Pbb = sum(p[2] for p in lst)
        g = Pab / np.sqrt(Paa * Pbb)
        e = {"slope_300_6500": r3(phase_slope(fN, g, w=np.abs(Pab), lo=300, hi=6500)[0])}
        if WINDOWS_MS[wname][0] >= 10:
            idx = np.where(sel)[0][::16]
            for kind in ("sinc", "j0"):
                for lo, hi in [(150, 7000), (1000, 6500), (2000, 6500), (300, 4000)]:
                    m = (fN[idx] >= lo) & (fN[idx] <= hi)
                    e[f"fit_{kind}_{lo}_{hi}"] = {k: r3(v) for k, v in fit_diffuse(fN[idx][m], g[idx][m], kind).items()}
            # |coherence| table every 500 Hz
            e["abs_coh_every_500Hz"] = [[int(fc), r3(np.abs(np.mean(Pab[(fN >= fc - 125) & (fN < fc + 125)])) /
                                          np.sqrt(np.mean(Paa[(fN >= fc - 125) & (fN < fc + 125)]) *
                                                  np.mean(Pbb[(fN >= fc - 125) & (fN < fc + 125)])))]
                                        for fc in range(500, 7600, 500)]
        res["pooled"][wname] = e
        print("pooled", wname, e.get("slope_300_6500"), {k: v for k, v in e.items() if k.startswith("fit_sinc_1000") or k.startswith("fit_sinc_150") or k.startswith("fit_j0_150")})
    with open(os.path.join(HERE, "s1_irs.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
