"""V1 (adversarial verifier): skew = phase slope of the noise field and of the reverberant tail, and direct-path TDOA,
for session 1 (hw/rec), the later takes in hw/rec2 (s2_* before the first esptool reboot at 01:51, reboot_*, s3_*,
drift_* after it) and the speech takes in hw/realset/rec (01:56-02:00, after the reboots).

Estimators are my own (not A1's):
 - noise slope: Welch 1024/512 periodic-Hann cross-spectrum, inverse-variance weighted LS fit of the phase (with an
   intercept) over 300-3000 Hz, where a symmetric diffuse field has zero phase; jackknife SE over 10 time blocks;
   cross-check with a band-limited time-domain cross-correlation peak.
 - sweeps: full deconvolution with the drift-stretched reference; direct TDOA from GCC-PHAT and from a magnitude-weighted
   phase fit in short windows; reverberant slopes from Welch frames (256/128) inside time windows after the direct peak.
Output: v1_skew.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import (FS, HERE, HW, deconv, delay_jack, delay_ls, frames_in, gcc_phat, load, nfft,  # noqa: E402
                   phase_delay_window, stretch, xcorr_delay, xspec)


def r(x, k=3):
    return None if x is None else float(np.round(x, k))


def stim_ref(path, key):
    d = np.load(os.path.join(HW, path))
    return d[key].astype(np.float64)


def onset(y, ref):
    """Integer lag of ref inside y (band-limited cross-correlation on ch0)."""
    n = nfft(len(y) + len(ref))
    f = np.fft.rfftfreq(n, 1 / FS)
    Y, R = np.fft.rfft(y, n), np.fft.rfft(ref, n)
    Gf = Y * np.conj(R)
    Gf[(f < 150) | (f > 7000)] = 0
    cc = np.fft.irfft(Gf, n)
    return int(np.argmax(cc[:len(y)]))


SESSION1 = {"ambient": None, "sweep_L": ("stim/sweep_L.npz", "L"), "sweep_R": ("stim/sweep_R.npz", "R"),
            "sweep_L2": ("stim/sweep_L.npz", "L"), "talkA_L": ("stim/talkA_L.npz", "L"),
            "talkB_R": ("stim/talkB_R.npz", "R"), "talkAB_LR": ("stim/talkAB_LR.npz", "L"),
            "noise_LR": ("stim/noise_LR.npz", "L"), "talkA_noise": ("stim/talkA_noise.npz", "L"),
            "talkC_R_quiet": ("stim/talkC_R_quiet.npz", "R")}


def noise_segments(name, x, ref_info):
    if ref_info is None:
        return [(int(1.0 * FS), int(11.9 * FS))], None
    ref = stim_ref(*ref_info)
    on = onset(x[:, 0], ref)
    end = on + len(ref)
    segs = []
    pre = (int(0.05 * FS), on + int(0.25 * FS) - int(0.02 * FS))
    post = (end + int(0.8 * FS), len(x) - int(0.1 * FS))
    for a, b in (pre, post):
        if b - a > int(0.5 * FS):
            segs.append((a, b))
    return segs, on


def noise_slope(name, ref_info):
    x = load(name)
    segs, on = noise_segments(name, x, ref_info)
    f, Pab, Paa, Pbb, n, fr = xspec(x, 1024, 512, segs)
    out = {"segments_s": [[r(a / FS, 2), r(b / FS, 2)] for a, b in segs], "n_frames": n}
    t, se, ic = delay_jack(fr, f, 300, 3000)
    out["slope_300_3000"] = [r(t), r(se), r(ic, 1)]
    t2, se2, _ = delay_jack(fr, f, 300, 3000, intercept=False)
    out["slope_300_3000_noicpt"] = [r(t2), r(se2)]
    bands = {}
    for lo, hi in [(300, 1500), (1500, 3000), (3000, 5000), (5000, 6500), (300, 6500)]:
        tb, seb, icb = delay_jack(fr, f, lo, hi)
        bands[f"{lo}-{hi}"] = [r(tb), r(seb), r(icb, 1)]
    out["band_slopes[tau,se,icpt_deg]"] = bands
    xs = np.concatenate([x[a:b] for a, b in segs])
    out["xcorr_300_3000"] = r(xcorr_delay(xs, 300, 3000))
    out["level_dbfs"] = [r(10 * np.log10(np.mean(xs[:, c] ** 2)), 1) for c in range(2)]
    g = Pab / np.sqrt(Paa * Pbb)
    out["msc_1-2k_4-6k"] = [r(np.mean(np.abs(g[(f >= 1000) & (f < 2000)]) ** 2)), r(np.mean(np.abs(g[(f >= 4000) & (f < 6000)]) ** 2))]
    return out


def sweep_analysis(name, ref_info, alpha=17e-6, eps=1e-4, detail=True):
    y = load(name)
    ref = stretch(stim_ref(*ref_info), alpha)
    h = deconv(y, ref, eps)
    p = int(np.argmax(np.abs(h[:3 * FS]).sum(1)))
    res = {"peak_idx": p}
    # first arrival: first sample (within 3 ms before the main peak) whose |h| (sum of channels) exceeds -12 dB of the peak
    e = np.abs(h[p - 48:p + 1]).sum(1)
    fa = p - 48 + int(np.argmax(e > e.max() * 10 ** (-12 / 20)))
    res["first_arrival_re_peak_samples"] = fa - p
    direct = {}
    for anchor_name, anc in (("peak", p), ("first", fa)):
        for t0, t1 in [(-0.5, 1.0), (-1.0, 1.5), (-1.0, 3.0), (-2.0, 4.0)]:
            i0, i1 = anc + int(round(t0 * 16)), anc + int(round(t1 * 16))
            seg = h[i0:i1] * np.hanning(i1 - i0 + 2)[1:-1, None]
            g = gcc_phat(seg[:, 0], seg[:, 1], 300, 6500)
            pdl, ic = phase_delay_window(seg, 300, 6000)
            direct[f"{anchor_name}_{t0}_{t1}ms"] = [r(g), r(pdl), r(ic, 1)]
    res["direct[gcc,phase_ls,icpt]"] = direct
    floor_i0, floor_i1 = p + int(1.5 * FS), p + int(2.5 * FS)
    ff, fab, faa, fbb, fn = frames_in(h, floor_i0, floor_i1)
    wins = {}
    for t0, t1 in [(2, 10), (10, 80), (30, 250), (80, 250), (250, 500), (1500, 2500)]:
        i0, i1 = p + int(t0 * 16), p + int(t1 * 16)
        f, ab, aa, bb, n = frames_in(h, i0, i1, 256 if t1 - t0 > 20 else 64, 128 if t1 - t0 > 20 else 32)
        entry = {"n_frames": n}
        for lo, hi in [(300, 3000), (300, 6000), (3000, 6000)]:
            tt, ic = delay_ls(f, ab, aa, bb, n, lo, hi)
            entry[f"slope_{lo}_{hi}"] = [r(tt), r(ic, 1)]
        if f.shape == ff.shape:
            snr = {}
            for lo, hi in [(300, 1000), (1000, 3000), (3000, 5000), (5000, 7000)]:
                m = (f >= lo) & (f < hi)
                snr[f"{lo}-{hi}"] = r(10 * np.log10((aa[m].sum() / n) / (faa[m].sum() / fn)), 1)
            entry["snr_vs_floor_db"] = snr
        wins[f"{t0}_{t1}ms"] = entry
    res["windows"] = wins
    return res, h, p


def main():
    out = {"noise": {}, "sweeps": {}}
    # ---------------- noise-field slopes
    names = [("rec/" + k, v) for k, v in SESSION1.items()]
    names += [("rec2/s2_ambient", None), ("rec2/s2_sweep_L", ("stim/sweep_L.npz", "L")),
              ("rec2/s2_sweep_R", ("stim/sweep_R.npz", "R"))]
    names += [(f"rec2/reboot_{i:02d}", None) for i in range(8)]
    names += [("rec2/s3_sweep_L", ("stim/sweep_L.npz", "L")), ("rec2/s3_ambient_after", None), ("rec2/drift_00", None)]
    takes = ["A1", "A2", "A3", "B1", "B2", "B3", "AE1", "AE2", "BE1", "BE2", "M1", "M2", "M3", "M4", "M5", "M6", "MF1", "MF2"]
    names += [(f"realset/rec/{t}", (f"realset/stim/{t}.npz", "played")) for t in takes]
    for name, ref in names:
        o = noise_slope(name, ref)
        out["noise"][name] = o
        print(f"{name:24s} lvl {o['level_dbfs']} slope300-3k {o['slope_300_3000']} noicpt {o['slope_300_3000_noicpt']} "
              f"xcorr {o['xcorr_300_3000']} bands {o['band_slopes[tau,se,icpt_deg]']}", flush=True)
    # ---------------- sweeps
    sweeps = [("rec/sweep_L", ("stim/sweep_L.npz", "L")), ("rec/sweep_R", ("stim/sweep_R.npz", "R")),
              ("rec/sweep_L2", ("stim/sweep_L.npz", "L")), ("rec2/s2_sweep_L", ("stim/sweep_L.npz", "L")),
              ("rec2/s2_sweep_R", ("stim/sweep_R.npz", "R")), ("rec2/s3_sweep_L", ("stim/sweep_L.npz", "L"))]
    for name, ref in sweeps:
        res, h, p = sweep_analysis(name, ref)
        out["sweeps"][name] = res
        print(name, "first-arrival re peak", res["first_arrival_re_peak_samples"])
        for k, v in res["direct[gcc,phase_ls,icpt]"].items():
            print("   direct", k, v)
        for k, v in res["windows"].items():
            print("   win", k, v)
    # sensitivity: no drift correction, weaker regularisation (sweep_L and s3_sweep_L)
    sens = {}
    for name, ref in [sweeps[0], sweeps[5]]:
        for tag, kw in [("alpha0", dict(alpha=0.0)), ("eps1e-6", dict(eps=1e-6))]:
            res, _, _ = sweep_analysis(name, ref, **kw)
            sens[f"{name}:{tag}"] = {"direct_peak_-1.0_3.0ms": res["direct[gcc,phase_ls,icpt]"]["peak_-1.0_3.0ms"],
                                     "slope_30_250_300_3000": res["windows"]["30_250ms"]["slope_300_3000"],
                                     "slope_80_250_300_3000": res["windows"]["80_250ms"]["slope_300_3000"]}
            print("sens", name, tag, sens[f"{name}:{tag}"], flush=True)
    out["sensitivity"] = sens
    with open(os.path.join(HERE, "v1_skew.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
