"""V1 (adversarial verifier): direct-path TDOA of every sweep, measured in the SAME band as the skew (so a frequency-
dependent mic phase mismatch cancels in g = direct - skew), its change across the 01:51 esptool reboots, the implied
lower bound on d, and d as a function of the unknown board distance if the mic axis bisects the two speakers.

Direct TDOA estimators on the deconvolved response (17 ppm drift-stretched reference):
  phase-LS (magnitude weights, with intercept) and GCC-PHAT in windows anchored at the first arrival, bands 300-3000 and
  300-6000 Hz; leading-edge timing (x32 up-sampled, -6 dB crossing of each channel's rising edge).
Skew per boot: median of the 300-3000 Hz noise-field slopes (no intercept) and the 30-250 / 80-250 ms reverb slopes.
Output: v1_geom.json
"""
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert, resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import C, FS, HERE, deconv, gcc_phat, load, phase_delay_window, stretch  # noqa: E402
from v1_skew import stim_ref  # noqa: E402

SK = json.load(open(os.path.join(HERE, "v1_skew.json")))


def r(x, k=3):
    return float(np.round(x, k))


def boot_skew(boot):
    pre = ["rec/ambient", "rec/sweep_L", "rec/sweep_R", "rec/sweep_L2", "rec/talkA_L", "rec/talkB_R", "rec/talkAB_LR",
           "rec/noise_LR", "rec/talkA_noise", "rec/talkC_R_quiet", "rec2/s2_ambient", "rec2/s2_sweep_L", "rec2/s2_sweep_R"]
    post = [f"rec2/reboot_{i:02d}" for i in range(8)] + ["rec2/s3_sweep_L", "rec2/s3_ambient_after", "rec2/drift_00"] + \
        [k for k in SK["noise"] if k.startswith("realset/")]
    names = pre if boot == 1 else post
    noise = np.array([SK["noise"][k]["slope_300_3000_noicpt"][0] for k in names])
    noise_icpt = np.array([SK["noise"][k]["slope_300_3000"][0] for k in names])
    xc = np.array([SK["noise"][k]["xcorr_300_3000"] for k in names])
    sw = ["rec/sweep_L", "rec/sweep_R", "rec/sweep_L2", "rec2/s2_sweep_L", "rec2/s2_sweep_R"] if boot == 1 else ["rec2/s3_sweep_L"]
    rev = np.array([SK["sweeps"][k]["windows"][w][b][0] for k in sw for w in ("30_250ms", "80_250ms")
                    for b in ("slope_300_3000", "slope_300_6000")])
    allv = np.concatenate([noise, xc, rev])
    return {"n_noise": len(noise), "noise_noicpt_median": r(np.median(noise)), "noise_noicpt_range": [r(noise.min()), r(noise.max())],
            "noise_icpt_median": r(np.median(noise_icpt)), "xcorr_median": r(np.median(xc)),
            "reverb_median": r(np.median(rev)), "reverb_range": [r(rev.min()), r(rev.max())],
            "combined_median": r(np.median(allv)), "spread_iqr": [r(np.percentile(allv, 25)), r(np.percentile(allv, 75))]}


def leading_edge(h, fa, up=32):
    i0 = fa - 24
    seg = h[i0:fa + 24]
    u = resample(seg, len(seg) * up, axis=0)
    out = []
    for c in range(2):
        env = np.abs(hilbert(u[:, c]))
        k_pk = int(np.argmax(env[: (24 + 12) * up]))  # first peak within 0.75 ms after the first-arrival index
        k6 = int(np.argmax(env[:k_pk + 1] >= 0.5 * env[k_pk]))
        out.append((k_pk / up, k6 / up))
    return r(out[1][0] - out[0][0]), r(out[1][1] - out[0][1])


def main():
    sweeps = [("rec/sweep_L", ("stim/sweep_L.npz", "L"), 1, "L"), ("rec/sweep_R", ("stim/sweep_R.npz", "R"), 1, "R"),
              ("rec/sweep_L2", ("stim/sweep_L.npz", "L"), 1, "L"), ("rec2/s2_sweep_L", ("stim/sweep_L.npz", "L"), 1, "L"),
              ("rec2/s2_sweep_R", ("stim/sweep_R.npz", "R"), 1, "R"), ("rec2/s3_sweep_L", ("stim/sweep_L.npz", "L"), 2, "L")]
    skew = {1: boot_skew(1), 2: boot_skew(2)}
    out = {"skew_by_boot": skew, "sweeps": {}}
    print("skew boot1", skew[1])
    print("skew boot2", skew[2])
    for name, ref, boot, spk in sweeps:
        h = deconv(load(name), stretch(stim_ref(*ref), 17e-6))
        p = int(np.argmax(np.abs(h[:3 * FS]).sum(1)))
        e = np.abs(h[p - 48:p + 1]).sum(1)
        fa = p - 48 + int(np.argmax(e > e.max() * 10 ** (-12 / 20)))
        est = {}
        for t0, t1 in [(-0.5, 1.0), (-1.0, 1.5), (-0.5, 0.75)]:
            i0, i1 = fa + int(round(t0 * 16)), fa + int(round(t1 * 16))
            seg = h[i0:i1] * np.hanning(i1 - i0 + 2)[1:-1, None]
            for lo, hi in [(300, 3000), (300, 6000)]:
                est[f"phaseLS_{lo}_{hi}_{t0}_{t1}"] = r(phase_delay_window(seg, lo, hi)[0])
                est[f"gcc_{lo}_{hi}_{t0}_{t1}"] = r(gcc_phat(seg[:, 0], seg[:, 1], lo, hi))
        est["leading_edge_peak,-6dB"] = leading_edge(h, fa)
        vals = np.array([v for k, v in est.items() if k.startswith("phaseLS") or k.startswith("gcc")])
        band3k = np.array([v for k, v in est.items() if "300_3000" in k])
        sk = skew[boot]["combined_median"]
        g = np.median(vals) - sk
        g3 = np.median(band3k) - sk
        out["sweeps"][name] = {"boot": boot, "speaker": spk, "first_arrival_re_peak": int(fa - p), "estimates": est,
                               "direct_median_all": r(np.median(vals)), "direct_range_all": [r(vals.min()), r(vals.max())],
                               "direct_median_300_3000": r(np.median(band3k)),
                               "g_all": r(g), "g_300_3000": r(g3),
                               "d_min_cm_all": r(100 * g * C / FS, 2), "d_min_cm_300_3000": r(100 * g3 * C / FS, 2)}
        print(name, out["sweeps"][name])
    # d versus board distance if the mic axis bisects the speakers (equal g for L and R), speaker half-spacing 0.135 m,
    # board 15-25 deg off the laptop's centre line (r_R - r_L = 6-11 cm)
    rows = []
    for g in (0.57, 0.60, 0.63):
        for R in (0.15, 0.2, 0.3, 0.5, 1.0):
            half = np.degrees(np.arctan(0.135 * np.cos(np.radians(20)) / R))
            rows.append([g, R, r(half, 1), r(100 * g * C / FS / np.cos(np.radians(half)), 2)])
    out["d_vs_board_distance_bisector"] = {"cols": ["g_samples", "R_m", "half_angle_deg", "d_cm"], "rows": rows}
    for row in rows:
        print("d(R)", row)
    # flip-test predictions must be referenced to the skew of the boot in which the test is run
    out["flip_prediction_rule"] = "tau_after_180deg_flip = 2*skew_now - tau_now; with g = +0.60: boot1 (skew -1.12): -1.72; boot2 (skew -1.02): -1.62"
    with open(os.path.join(HERE, "v1_geom.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
