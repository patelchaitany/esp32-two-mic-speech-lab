"""Exp 6: how much do skew / gain errors hurt fixed 2-mic beams at 1.19 cm vs 4 cm (theory) and on the real board?

Part T (narrowband theory, free field, 3-D diffuse noise + incoherent mic self-noise):
  beams designed for an ideal array, target at endfire on Mic A's side (theta = 0, where both real speakers sit):
  'das' (delay-and-sum), 'cardioid' (null at 180 deg), 'hyper' (null at 109.5 deg), 'sdmvdr' (superdirective MVDR,
  diffuse coherence + loading = self-noise/ambient ratio), each with a flat on-axis EQ (max +30 dB).
  Real channel B = 10^(g/20) exp(-j 2 pi f s / fs) x ideal B (s = residual skew in samples, g = gain error in dB).
  Output: diffuse-noise array gain AG (dB, = SNR gain vs one mic in diffuse noise incl. self-noise), on-axis
  response error, averaged 300-4000 Hz, for d = 1.19 / 2 / 4 cm, s in {0, 0.01, 0.02, 0.05, 0.1, 0.25, 1.015},
  g in {0, 0.1, 0.3, 1.3} dB, ambient 'bus' (-40 dBFS: self-noise 47 dB below) and 'meeting' (-65 dBFS: 22 dB below).
  Also the apparent DOA of sources with the skew left uncorrected (S1's view).
Part R (real takes, linear so image and noise are scored separately): the same fixed beams at 1.19 cm on the 18
  realset takes with the board calibration 'full' (calib_session.json: skew + per-band gain), 'skew_only',
  'gain_only' and 'none'; SNR gain vs Mic A (300-4000 Hz, speech interval) = what a fixed beam buys against the real
  fan noise, and target (image) level change.
Run: .venv/bin/python -I hw/realset/bizarre/bz6_skew_sensitivity.py -> bz6_skew_sensitivity.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import C, CALIB, FS, HERE, dump  # noqa: E402

F = np.arange(100.0, 7001.0, 25.0)
BAND = (F >= 300) & (F <= 4000)
EQ_MAX_DB = 30.0


def steer(f, tau):
    """[A, B] response to a plane wave with TDOA tau (s, + = B later), referenced to Mic A."""
    return np.stack([np.ones_like(f, dtype=complex), np.exp(-2j * np.pi * f * tau)], -1)


def design(kind, f, d, mu):
    """Weights (F, 2): y = w^H x, flat on-axis (theta = 0) response, EQ limited to EQ_MAX_DB."""
    a0 = steer(f, d / C)
    if kind == "das":
        w = a0 / 2.0
    elif kind in ("cardioid", "hyper"):
        th = np.pi if kind == "cardioid" else np.deg2rad(109.47)
        an = steer(f, d * np.cos(th) / C)
        # null toward theta_null: w = [1, -an_B] gives w^H an = 1 - conj(an_B) an_B = 0 (y = w^H x)
        w = np.stack([np.ones_like(f, dtype=complex), -an[:, 1]], -1)
    elif kind == "sdmvdr":
        g = np.sinc(2 * f * d / C)
        w = np.zeros((len(f), 2), complex)
        for i in range(len(f)):
            G = np.array([[1.0, g[i]], [g[i], 1.0]]) + mu * np.eye(2)
            v = np.linalg.solve(G, a0[i])
            w[i] = v / (np.conj(a0[i]) @ v)
    else:
        raise ValueError(kind)
    r = np.einsum("fc,fc->f", np.conj(w), a0)                                   # on-axis response
    gmax = 10 ** (EQ_MAX_DB / 20)
    eq = 1.0 / np.where(np.abs(r) * gmax < 1, r / np.abs(r) / gmax, r)
    return w * np.conj(eq)[:, None]


def evaluate(w, f, d, s, gdb, mu):
    e = 10 ** (gdb / 20) * np.exp(-2j * np.pi * f * s / FS)                   # channel-B error
    a = steer(f, d / C) * np.stack([np.ones_like(e), e], -1)
    resp = np.einsum("fc,fc->f", np.conj(w), a)
    gam = np.sinc(2 * f * d / C)
    # true noise covariance at the outputs of the two channels: diffuse (with B's error) + self-noise
    Pn = (np.abs(w[:, 0]) ** 2 * (1 + mu) + np.abs(w[:, 1]) ** 2 * (np.abs(e) ** 2 + mu)
          + 2 * np.real(np.conj(w[:, 0]) * w[:, 1] * gam * np.conj(e)))
    ag = 10 * np.log10(np.abs(resp) ** 2 / Pn * (1 + mu))                      # vs one mic (diffuse + self-noise)
    return ag, 20 * np.log10(np.abs(resp) + 1e-12)


def part_t():
    out = {}
    for amb, mu in (("bus", 10 ** (-47 / 10)), ("meeting", 10 ** (-22 / 10))):
        for d in (0.0119, 0.02, 0.04):
            for kind in ("das", "cardioid", "hyper", "sdmvdr"):
                w = design(kind, F, d, mu)
                for s in (0.0, 0.01, 0.02, 0.05, 0.1, 0.25, -1.015):
                    for g in (0.0, 0.1, 0.3, 1.3):
                        ag, rdb = evaluate(w, F, d, s, g, mu)
                        key = f"{amb}|{d * 100:.2f}cm|{kind}|s={s}|g={g}"
                        out[key] = {"ag_db_300_4k": float(np.mean(ag[BAND])),
                                    "ag_db_min_300_4k": float(np.min(ag[BAND])),
                                    "ag_db_500": float(np.interp(500, F, ag)), "ag_db_2k": float(np.interp(2000, F, ag)),
                                    "resp_err_db_max_300_4k": float(np.max(np.abs(rdb[BAND]))),
                                    "resp_db_500": float(np.interp(500, F, rdb))}
    # apparent DOA with the skew left uncorrected
    doa = {}
    for d in (0.0119, 0.04):
        dc = d / C * FS
        rows = []
        for th in (0, 45, 90, 135, 180):
            tau = dc * np.cos(np.deg2rad(th)) - 1.015
            app = np.degrees(np.arccos(tau / dc)) if abs(tau) <= dc else float("nan")
            rows.append({"true_deg": th, "apparent_tdoa_samples": round(tau, 3), "apparent_deg": app,
                         "outside_physical_range": bool(abs(tau) > dc), "outside_S1_grid_1.2dc": bool(abs(tau) > 1.2 * dc)})
        doa[f"{d * 100:.2f}cm"] = {"d_over_c_samples": dc, "rows": rows}
    return out, doa


def part_r():
    from methods._core import calib, common as cm
    import bz_takes as bt
    f = cm.FREQS
    d = CALIB["spacing_m"]
    band = (f >= 300) & (f <= 4000)
    rows = {}
    for t in bt.TAKES:
        tk = bt.composite(t)
        (a, b), = bt.speech_iv(tk, extra_s=0.0)
        Xr = cm.stft(calib.prefilter(tk["recording"], "meeting"))
        cal = calib.calibrate(Xr, {"scenario": "meeting", "spacing_m": d, "calib_json": CALIB})
        corr = cal.corr
        variants = {"full": corr, "skew_only": corr / np.abs(corr), "gain_only": np.abs(corr) + 0j,
                    "none": np.ones_like(corr)}
        Xi = cm.stft(calib.prefilter(tk["image"], "meeting"))
        Xn = cm.stft(calib.prefilter(tk["noise"], "meeting"))
        cen = cm.frame_centers(Xi.shape[-1])
        fr = (cen >= a) & (cen < b)
        ref_i = np.sum(np.abs(Xi[0][band][:, fr]) ** 2)
        ref_n = np.sum(np.abs(Xn[0][band][:, fr]) ** 2)
        r = {"far": tk["far"]}
        for kind in ("das", "cardioid", "hyper", "sdmvdr"):
            w = design(kind, f, d, 10 ** (-40 / 10))                           # loading for real self-noise ~-40 dB re fans
            for vn, cv in variants.items():
                yi = np.conj(w[:, 0])[:, None] * Xi[0] + np.conj(w[:, 1])[:, None] * (cv[:, None] * Xi[1])
                yn = np.conj(w[:, 0])[:, None] * Xn[0] + np.conj(w[:, 1])[:, None] * (cv[:, None] * Xn[1])
                ei = np.sum(np.abs(yi[band][:, fr]) ** 2)
                en = np.sum(np.abs(yn[band][:, fr]) ** 2)
                r[f"{kind}|{vn}"] = {"snr_gain_db": float(10 * np.log10(ei / en) - 10 * np.log10(ref_i / ref_n)),
                                     "target_change_db": float(10 * np.log10(ei / ref_i)),
                                     "noise_change_db": float(10 * np.log10(en / ref_n))}
        rows[t] = r
        print(t, {k: round(v["snr_gain_db"], 2) for k, v in r.items() if isinstance(v, dict) and k.endswith(("|full", "|none"))},
              flush=True)
    summ = {}
    for key in [k for k in rows[bt.TAKES[0]] if "|" in k]:
        v = np.array([rows[t][key]["snr_gain_db"] for t in bt.TAKES])
        tc = np.array([rows[t][key]["target_change_db"] for t in bt.TAKES])
        summ[key] = {"snr_gain_median": float(np.median(v)), "snr_gain_min": float(v.min()), "snr_gain_max": float(v.max()),
                     "target_change_median": float(np.median(tc))}
    return {"rows": rows, "summary": summ}


def main():
    t, doa = part_t()
    print("theory done", flush=True)
    for amb in ("bus", "meeting"):
        for d in ("1.19", "4.00"):
            for kind in ("cardioid", "hyper", "sdmvdr", "das"):
                line = [f"{amb} {d}cm {kind:8s}"]
                for s in (0.0, 0.05, 0.1, -1.015):
                    for g in (0.0, 1.3):
                        line.append(f"s{s:+.2f}g{g}: {t[f'{amb}|{d}cm|{kind}|s={s}|g={g}']['ag_db_300_4k']:+6.1f}")
                print(" ".join(line))
    r = part_r()
    dump({"theory": t, "doa_uncorrected": doa, "real": r}, os.path.join(HERE, "bz6_skew_sensitivity.json"))


if __name__ == "__main__":
    main()
