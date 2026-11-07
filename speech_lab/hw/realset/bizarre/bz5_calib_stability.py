"""Exp 5: calibration stability of the ESP32 board (skew, spacing, gain) within a boot and across boots, and the
seconds of room noise needed for a +-0.05 sample skew estimate.

Data (all silent stretches; nothing is played or recorded here):
  boot 0 (first boot):   hw/rec/ambient.wav (01:19), hw/rec2/s2_ambient.wav (01:49), silent tails (>= 5.5 s) of the
                         session-1 sound recordings hw/rec/*.wav and of hw/rec2/s2_sweep_{L,R}.wav
  boots 1-8:             hw/rec2/reboot_00..07.wav (one esptool hard reset before each, 01:51-01:55)
  boot 8 (current boot): also hw/rec2/s3_ambient_after.wav, the noise-only tails of the 18 realset takes (01:56-02:00),
                         s3_sweep_L tail, hw/rec2/drift_00..02.wav (02:05, 02:15, 02:25)
Estimators (bz_common): skew = weighted phase slope of the noise cross-spectrum (150-7000 Hz);
spacing = |coherence| fit rho*sinc(2fd/c); gain = B re A level per band.
Run: .venv/bin/python -I hw/realset/bizarre/bz5_calib_stability.py  -> bz5_calib_stability.json
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import FS, HW, HERE, REALSET, band_gain_db, coh_stats, dump, load_take, load_wav, skew_phase, spacing_mag  # noqa: E402


def segments():
    """(boot, label, x (2, T)) for every silent stretch."""
    out = []
    # boot 0
    for p in ["rec/ambient.wav", "rec2/s2_ambient.wav"]:
        x = load_wav(os.path.join(HW, p))[:, FS:11 * FS]
        out.append((0, os.path.basename(p)[:-4], x))
    for p in sorted(glob.glob(os.path.join(HW, "rec", "*.wav"))) + [os.path.join(HW, "rec2", "s2_sweep_L.wav"),
                                                                 os.path.join(HW, "rec2", "s2_sweep_R.wav")]:
        nm = os.path.basename(p)[:-4]
        if nm == "ambient":
            continue
        x = load_wav(p)
        out.append((0, nm + "_tail", x[:, int(5.5 * FS):int(11.75 * FS)]))
    # boots 1..8
    for i in range(8):
        x = load_wav(os.path.join(HW, "rec2", f"reboot_{i:02d}.wav"))[:, FS:11 * FS]
        out.append((i + 1, f"reboot_{i:02d}", x))
    # current boot extras
    out.append((8, "s3_ambient_after", load_wav(os.path.join(HW, "rec2", "s3_ambient_after.wav"))[:, FS:11 * FS]))
    x = load_wav(os.path.join(HW, "rec2", "s3_sweep_L.wav"))
    out.append((8, "s3_sweep_L_tail", x[:, int(5.5 * FS):int(11.75 * FS)]))
    for t in ["A1", "A2", "A3", "B1", "B2", "B3", "AE1", "AE2", "BE1", "BE2", "M1", "M2", "M3", "M4", "M5", "M6",
              "MF1", "MF2"]:
        z = load_take(t)
        on = int(z["onset"])
        y = z["recording"].T.astype(np.float64)       # stored 80 Hz high-passed
        out.append((8, f"take_{t}_tail", y[:, on + int(4.9 * FS):len(y[0]) - FS // 4]))
    for i in range(3):
        out.append((8, f"drift_{i:02d}", load_wav(os.path.join(HW, "rec2", f"drift_{i:02d}.wav"))[:, FS:11 * FS]))
    return out


def est(x):
    f, aa, bb, ab = coh_stats(x)
    s = skew_phase(f, aa, bb, ab)
    d, rho, rms = spacing_mag(f, aa, bb, ab, skew=s)
    g = band_gain_db(f, aa, bb)
    m = (f > 200) & (f < 5000)
    gbb = float(10 * np.log10(np.sum(bb[m]) / np.sum(aa[m])))
    lvl = float(10 * np.log10(np.mean(x[0] ** 2)))
    return {"skew": s, "d_cm": d, "rho": rho, "fit_rms": rms, "gain_bands_db": g, "gain_200_5k_db": gbb,
            "level_dbfs": lvl, "dur_s": x.shape[1] / FS}


def main():
    segs = segments()
    rows = []
    for boot, lab, x in segs:
        r = est(x)
        r.update(boot=boot, label=lab)
        rows.append(r)
        print(f"boot {boot} {lab:22s} {r['dur_s']:5.1f}s skew {r['skew']:+.3f} d {r['d_cm']:.2f} rho {r['rho']:.2f} "
              f"gain {r['gain_200_5k_db']:+.2f} lvl {r['level_dbfs']:.1f}", flush=True)

    def summ(v):
        v = np.asarray(v, float)
        return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    res = {"rows": rows, "per_boot": {}}
    for b in range(9):
        rb = [r for r in rows if r["boot"] == b]
        res["per_boot"][b] = {k: summ([r[k] for r in rb]) for k in ("skew", "d_cm", "rho", "gain_200_5k_db")}
    # across boots: one value per boot (reboot_XX files only, same procedure; boot 0 = its two ambient files)
    boot_skews = [np.mean([r["skew"] for r in rows if r["boot"] == b and (r["label"].startswith("reboot") or
                                                                          r["label"] in ("ambient", "s2_ambient"))])
                  for b in range(9)]
    res["across_boots_skew"] = {"values": boot_skews, **summ(boot_skews)}
    frac = [s - np.floor(s) for s in boot_skews]
    res["across_boots_skew"]["integer_part"] = [int(np.floor(s)) for s in boot_skews]
    res["across_boots_skew"]["fraction"] = frac
    # within the current boot: quiet vs loud fan conditions
    cur = [r for r in rows if r["boot"] == 8]
    res["current_boot_level_vs_skew"] = [(r["label"], round(r["level_dbfs"], 1), round(r["skew"], 3), round(r["d_cm"], 2))
                                         for r in cur]

    # seconds of noise needed: windows of length L from the current-boot silent stretches (excluding the quiet drift
    # files, kept separately), std of the skew estimate vs L, and the same against the quiet drift_01/02
    def windows(sel, L):
        out = []
        for boot, lab, x in segs:
            if not sel(boot, lab):
                continue
            n = int(L * FS)
            for s in range(0, x.shape[1] - n + 1, n):
                out.append(x[:, s:s + n])
        return out

    res["window_study"] = {}
    for name, sel in (("current_boot_loud", lambda b, l: b == 8 and not l.startswith("drift_0") or l == "drift_00"),
                      ("current_boot_quiet", lambda b, l: l in ("drift_01", "drift_02")),
                      ("first_boot", lambda b, l: b == 0)):
        res["window_study"][name] = {}
        for L in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0):
            ws = windows(sel, L)
            if len(ws) < 3:
                continue
            sk = []
            for w in ws:
                f, aa, bb, ab = coh_stats(w, nperseg=512 if L < 0.5 else 1024)
                sk.append(skew_phase(f, aa, bb, ab))
            sk = np.array(sk)
            med = np.median(sk)
            res["window_study"][name][str(L)] = {"n": len(sk), "std": float(sk.std(ddof=1)), "median": float(med),
                                                 "p95_abs_dev": float(np.percentile(np.abs(sk - med), 95)),
                                                 "frac_within_0.05": float(np.mean(np.abs(sk - med) <= 0.05))}
            print(name, L, res["window_study"][name][str(L)], flush=True)
    dump(res, os.path.join(HERE, "bz5_calib_stability.json"))


if __name__ == "__main__":
    main()
