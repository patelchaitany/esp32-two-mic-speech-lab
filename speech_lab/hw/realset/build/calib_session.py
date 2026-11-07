"""Session calibration (current boot) of the ESP32 board from the realset takes and hw/rec2/s3_ambient_after.wav.

Writes hw/realset/calib_session.json (an info['calib_json'] dict for methods/_core/calib.py) and
hw/realset/calib_session_report.json (every estimate and the checks).

  .venv/bin/python -I hw/realset/build/calib_session.py

Conventions: tau > 0 = Mic B hears later (B lags A). calib.py multiplies X_B by
10^(gain_db/20) * exp(j 2 pi f tau0_s), so tau0_s = measured skew (s) and gain_db = -(B re A level, dB).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # hw/realset
HW = os.path.dirname(ROOT)
LAB = os.path.dirname(HW)
sys.path.insert(0, HW)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from analyze import gcc_phat_tdoa  # noqa: E402  (numpy/scipy only; no hardware access)

FS, C = 16000, 343.0
BANDS_HZ = 1000.0 * 2.0 ** (np.arange(-9, 9) / 3.0)   # the 18 S0 one-third-octave centres, 125 Hz - 6.35 kHz
TAKES = ["A1", "A2", "A3", "B1", "B2", "B3", "AE1", "AE2", "BE1", "BE2",
         "M1", "M2", "M3", "M4", "M5", "M6", "MF1", "MF2"]
SIDE = {t: s for t, s in zip(TAKES, "LLLRRRLLRRLRLRLRLR")}
_HP = butter(4, 80, "high", fs=FS, output="sos")
NFFT = 1024


def stft(x, n=NFFT, hop=NFFT // 2):
    w = np.hanning(n + 1)[:-1]
    idx = np.arange(0, x.shape[-1] - n, hop)
    fr = np.stack([x[..., i:i + n] * w for i in idx], -2)
    return np.fft.rfft(fr, axis=-1)                 # (..., frames, bins)


def cross_stats(x):
    """x: (2, T). Returns f, Saa, Sbb, Sab = E[A conj(B)] (phase = +2 pi f tau when B lags by tau)."""
    X = stft(x)
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    return f, np.mean(np.abs(X[0]) ** 2, 0), np.mean(np.abs(X[1]) ** 2, 0), np.mean(X[0] * np.conj(X[1]), 0)


def coh_fit(f, saa, sbb, sab, skew=None, lo=150.0, hi=7000.0):
    """Fit rho*sinc(2 f d / c)*exp(j 2 pi f tau / FS) to the complex coherence (same model as geomcheck)."""
    G = sab / np.sqrt(saa * sbb)
    m = (f > lo) & (f < hi)

    def res(p):
        s = skew if skew is not None else p[2]
        e = G[m] - p[0] * np.sinc(2 * f[m] * p[1] / C) * np.exp(2j * np.pi * f[m] * s / FS)
        return np.concatenate([e.real, e.imag])

    best = None
    for d0 in (0.008, 0.012, 0.02, 0.04):
        x0 = [0.95, d0] + ([] if skew is not None else [-1.0])
        lo_b = [0.2, 0.003] + ([] if skew is not None else [-4.0])
        hi_b = [1.0, 0.3] + ([] if skew is not None else [4.0])
        r = least_squares(res, x0, bounds=(lo_b, hi_b))
        if best is None or r.cost < best.cost:
            best = r
    return {"d_cm": float(best.x[1] * 100), "rho": float(best.x[0]),
            "tau_samples": float(skew if skew is not None else best.x[2]),
            "rmse": float(np.sqrt(np.mean(best.fun ** 2)))}


def band_db(f, ratio_pow):
    out = []
    for fc in BANDS_HZ:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10 * np.log10(np.mean(ratio_pow[m])))
    return np.array(out)


def band_db_ratio(f, pb, pa):
    out = []
    for fc in BANDS_HZ:
        m = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        out.append(10 * np.log10(np.sum(pb[m]) / np.sum(pa[m])))
    return np.array(out)


def load_take(t):
    z = np.load(os.path.join(ROOT, "img", t + ".npz"))
    return {k: z[k] for k in z.files}


def direct_tdoa(h):
    """Direct-path TDOA (samples, + = B late) from a fitted 2-ch response h (K, 2), as geomcheck.direct_tdoa."""
    p = int(np.argmax(np.abs(h).sum(1)))
    w = h[p - 8:p + 24] * np.hanning(48)[8:40, None]
    tau = gcc_phat_tdoa(w[:, 0], w[:, 1], maxlag=6.0)
    tau_long = gcc_phat_tdoa(h[p - 16:p + 48, 0], h[p - 16:p + 48, 1], maxlag=6.0)
    return tau, tau_long, p


def direct_gain(h, n_pre=16, n_post=80, nfft=4096):
    """B re A level (dB) per band of the first n_post samples (5 ms) of the direct sound."""
    p = int(np.argmax(np.abs(h).sum(1)))
    w = np.ones(n_pre + n_post)
    w[:8] = np.hanning(16)[:8]
    w[-16:] = np.hanning(32)[16:]
    seg = h[p - n_pre:p + n_post] * w[:, None]
    H = np.fft.rfft(seg, nfft, axis=0)
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return band_db_ratio(f, np.abs(H[:, 1]) ** 2, np.abs(H[:, 0]) ** 2)


def main():
    rep = {"takes": {}}
    # 1. s3_ambient_after (room noise after the resets, current boot)
    x, fs = sf.read(os.path.join(HW, "rec2", "s3_ambient_after.wav"), dtype="float64")
    x = sosfiltfilt(_HP, x - x.mean(0), axis=0)[FS:11 * FS].T
    f, saa, sbb, sab = cross_stats(x)
    rep["s3_ambient_after"] = {"free": coh_fit(f, saa, sbb, sab),
                               "psd_B_re_A_db": band_db_ratio(f, sbb, saa).round(2).tolist()}
    # 2. take noise parts (recording - fitted image): pooled and per take
    P = {"saa": 0, "sbb": 0, "sab": 0}
    gains_img, gains_noise, gains_direct, taus = {}, {}, {}, {}
    for t in TAKES:
        z = load_take(t)
        nz = z["noise"].T.astype(np.float64)[:, FS // 4:]
        f, a, b, ab = cross_stats(nz)
        P["saa"] = P["saa"] + a
        P["sbb"] = P["sbb"] + b
        P["sab"] = P["sab"] + ab
        fit = coh_fit(f, a, b, ab)
        on = int(z["onset"])
        im = z["image"].T.astype(np.float64)[:, on:on + int(4.2 * FS)]
        fi, ia, ib, _ = cross_stats(im)
        g_img = band_db_ratio(fi, ib, ia)
        g_noise = band_db_ratio(f, b, a)
        tau, tau_long, p = direct_tdoa(z["h"].astype(np.float64))
        g_dir = direct_gain(z["h"].astype(np.float64))
        gains_img[t], gains_noise[t], gains_direct[t], taus[t] = g_img, g_noise, g_dir, tau
        rep["takes"][t] = {"side": SIDE[t], "noise_fit": {k: round(v, 3) for k, v in fit.items()},
                           "direct_tdoa_samples": round(tau, 3), "direct_tdoa_4ms_samples": round(tau_long, 3),
                           "direct_idx": p, "img_B_re_A_db_mean": round(float(np.mean(g_img[2:16])), 2),
                           "noise_B_re_A_db_mean": round(float(np.mean(g_noise[2:16])), 2),
                           "direct_B_re_A_db_mean": round(float(np.mean(g_dir[6:16])), 2)}
    pooled = coh_fit(f, P["saa"], P["sbb"], P["sab"])
    rep["take_noise_pooled_fit"] = pooled
    fits = [rep["takes"][t]["noise_fit"] for t in TAKES]
    rep["take_noise_per_take"] = {
        "d_cm_median": float(np.median([q["d_cm"] for q in fits])),
        "d_cm_range": [float(min(q["d_cm"] for q in fits)), float(max(q["d_cm"] for q in fits))],
        "tau_median": float(np.median([q["tau_samples"] for q in fits])),
        "tau_range": [float(min(q["tau_samples"] for q in fits)), float(max(q["tau_samples"] for q in fits))]}

    # 3. decisions
    tau0 = float(np.median([pooled["tau_samples"], rep["s3_ambient_after"]["free"]["tau_samples"],
                            rep["take_noise_per_take"]["tau_median"]]))
    d_est = [pooled["d_cm"], rep["s3_ambient_after"]["free"]["d_cm"], rep["take_noise_per_take"]["d_cm_median"]]
    # spacing: the fit with the skew fixed to tau0 (pooled take noise + s3 ambient), median with the given 1.19 cm
    fix_pool = coh_fit(f, P["saa"], P["sbb"], P["sab"], skew=tau0)
    fix_s3 = coh_fit(*cross_stats(x), skew=tau0)
    rep["fits_skew_fixed"] = {"take_noise_pooled": fix_pool, "s3_ambient_after": fix_s3}
    # Decision: 1.19 cm (main agent's value from 10 diffuse windows, 1.09-1.36 cm). The current-boot diffuse fits
    # give 1.13-1.15 cm, but the partners' direct-path TDOAs after the skew correction (median 0.55 samples, both
    # speakers near endfire) need d >= ~1.17 cm; the 4 % difference is below what any method resolves.
    d_cm = 1.19
    G_img = np.median(np.stack(list(gains_img.values())), 0)
    G_noise = np.median(np.stack(list(gains_noise.values())), 0)
    G_dir = np.median(np.stack(list(gains_direct.values())), 0)
    rep["gain_B_re_A_db"] = {"bands_hz": BANDS_HZ.round(1).tolist(), "image_psd_median": G_img.round(2).tolist(),
                             "noise_psd_median": G_noise.round(2).tolist(), "direct_5ms_median": G_dir.round(2).tolist(),
                             "s3_ambient_psd": rep["s3_ambient_after"]["psd_B_re_A_db"]}
    # B re A used: per-band median of the four estimates (take images, take noise, 5 ms direct path, s3 ambient);
    # they agree within ~0.5 dB from 200 Hz up and scatter by 2 dB at 125-160 Hz (incoherent low-frequency noise)
    G_s3 = np.array(rep["s3_ambient_after"]["psd_B_re_A_db"])
    g_used = np.median(np.stack([G_img, G_noise, G_dir, G_s3]), 0)
    cal = {"version": 1, "spacing_m": round(d_cm / 100, 5), "tau0_s": tau0 / FS, "hpf_corner_hz": [60.0, 60.0],
           "gain_db": (-g_used).round(2).tolist(), "model": "sinc"}
    rep["decision"] = {"tau0_samples": tau0, "spacing_cm_candidates": d_est, "spacing_cm": d_cm,
                       "gain_rule": "gain_db = -(per-band median of image-PSD, noise-PSD, 5 ms direct-path and s3-ambient B re A)",
                       "hpf_corner": "unknown: [60, 60] adds no corner phase"}

    # 4. checks: corrected noise cross-spectrum phase and partner TDOAs inside +-d/c
    corr_tau = tau0
    ph_take = {}
    for t in TAKES:
        z = load_take(t)
        nz = z["noise"].T.astype(np.float64)[:, FS // 4:]
        f, a, b, ab = cross_stats(nz)
        ph = np.angle(ab * np.exp(-2j * np.pi * f * corr_tau / FS))   # A conj(B e^{+j w tau0})
        ph_take[t] = [float(np.degrees(np.angle(np.mean(np.exp(1j * ph[(f >= fc * 2 ** (-1 / 6)) &
                                                                       (f < fc * 2 ** (1 / 6))])))))
                      for fc in BANDS_HZ]
    ph_all = np.array(list(ph_take.values()))
    f, a, b, ab = cross_stats(x)
    ph_s3 = np.angle(ab * np.exp(-2j * np.pi * f * corr_tau / FS))
    raw_s3 = np.angle(ab)
    dc = d_cm / 100 / C * FS
    tdoa_corr = {t: round(taus[t] - tau0, 3) for t in TAKES}
    rep["checks"] = {
        "noise_phase_after_corr_deg": {"bands_hz": BANDS_HZ.round(0).tolist(),
                                       "median_over_takes": np.median(ph_all, 0).round(1).tolist(),
                                       "max_abs_over_takes": np.max(np.abs(ph_all), 0).round(1).tolist()},
        "s3_phase_deg_raw_vs_corrected": {
            str(int(fc)): [round(float(np.degrees(np.angle(np.mean(np.exp(1j * r[m]))))), 1)
                           for r in (raw_s3, ph_s3)]
            for fc in BANDS_HZ[::3] for m in [(f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))]},
        "d_over_c_samples": dc,
        "direct_tdoa_minus_tau0": tdoa_corr,
        "all_inside_d_over_c": bool(all(abs(v) <= dc for v in tdoa_corr.values())),
        "inside_d_over_c_plus_0p05": {t: bool(abs(v) <= dc + 0.05) for t, v in tdoa_corr.items()},
        "partner_takes_median_tdoa": float(np.median([tdoa_corr[t] for t in ("A1", "A2", "A3", "B1", "B2", "B3")])),
        "geometric_cos_theta": {t: round(v / dc, 2) for t, v in tdoa_corr.items()}}
    # S0 accepts the dict
    sys.path.insert(0, LAB)
    from methods._core import calib as s0
    s0.load_json(cal)
    rep["calib_json"] = cal
    with open(os.path.join(ROOT, "calib_session.json"), "w") as fh:
        json.dump(cal, fh, indent=1)
    with open(os.path.join(ROOT, "calib_session_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(json.dumps({"calib": cal, "decision": rep["decision"], "s3": rep["s3_ambient_after"]["free"],
                      "pooled": pooled, "per_take": rep["take_noise_per_take"], "fixed": rep["fits_skew_fixed"],
                      "gain": rep["gain_B_re_A_db"], "checks": rep["checks"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
