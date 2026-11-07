"""V1 (adversarial verifier): is the spacing d identifiable from the coherence magnitude, and how model-dependent is it?

Data sets (my own Welch / window estimates, magnitude only, after removing each set's own linear skew):
  FAN1  = noise-only stretches of the 10 session-1 recordings (boot 1, fan about -51 dBFS)
  FAN2  = reboot_00..07 + s3_ambient_after (after the 01:51 esptool reboots, fan about -44..-47 dBFS)
  REV1  = 30-250 ms and 80-250 ms of the session-1 sweeps + s2 sweeps (boot 1), Welch 256/128 frames inside the window
  REV2  = the same windows of s3_sweep_L (after the reboots)
Models for |gamma(f)| (axisymmetric field, energy density ~ (1-u^2)^m, u = cos angle to the mic axis):
  m = 0: 3-D isotropic (sinc); m = -0.5: J0 (horizontal 2-D); m > 0: broadside-heavy. Small-kd: 1-|g| ~ (kd)^2/(2(2m+3)),
  so only d^2/(2m+3) is identified to first order.  Also: mixture rho*(beta + (1-beta)*sinc) (coherent broadside part).
Output: v1_d.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v1lib import C, FS, HERE, HW, coh_axisym, deconv, delay_ls, frames_in, kd, load, stretch, xspec  # noqa: E402
from v1_skew import SESSION1, noise_segments, stim_ref  # noqa: E402


def r(x, k=3):
    return float(np.round(x, k))


def band_coh(f, ab, aa, bb, tau, centers, half=62.5):
    rot = np.exp(-2j * np.pi * f * tau / FS)
    g, ph = [], []
    for fc in centers:
        m = (f >= fc - half) & (f < fc + half)
        z = np.sum(ab[m] * rot[m]) / np.sqrt(np.sum(aa[m]) * np.sum(bb[m]))
        g.append(np.abs(z))
        ph.append(np.degrees(np.angle(z)))
    return np.array(g), np.array(ph)


def fit_family(fc, g, d_grid, m_list, rho_bounds=(0.85, 1.0)):
    """For each d: best (m, rho) -> SSE. Returns sse[d_index, m_index] and rho."""
    sse = np.empty((len(d_grid), len(m_list)))
    rho = np.empty_like(sse)
    for i, d in enumerate(d_grid):
        x = kd(fc, d)
        for j, m in enumerate(m_list):
            D = coh_axisym(x, m)
            rr = np.clip(np.sum(D * g) / np.sum(D * D), *rho_bounds)
            sse[i, j] = np.sum((g - rr * D) ** 2)
            rho[i, j] = rr
    return sse, rho


def fit_mixture(fc, g, d_grid, betas, rho_bounds=(0.85, 1.0)):
    sse = np.empty((len(d_grid), len(betas)))
    for i, d in enumerate(d_grid):
        S = np.sinc(2 * fc * d / C)
        for j, b in enumerate(betas):
            D = b + (1 - b) * S
            rr = np.clip(np.sum(D * g) / np.sum(D * D), *rho_bounds)
            sse[i, j] = np.sum((g - rr * D) ** 2)
    return sse


def collect_noise(names_refs):
    acc = None
    n_tot = 0
    for name, ref in names_refs:
        x = load(name)
        segs, _ = noise_segments(name, x, ref)
        f, ab, aa, bb, n, fr = xspec(x, 1024, 512, segs)
        s = (ab * n, aa * n, bb * n)
        acc = s if acc is None else tuple(a + b for a, b in zip(acc, s))
        n_tot += n
    return f, acc[0] / n_tot, acc[1] / n_tot, acc[2] / n_tot, n_tot


def collect_reverb(sweeps, t0, t1, nper=256, hop=128):
    acc = None
    n_tot = 0
    for name, ref in sweeps:
        y = load(name)
        h = deconv(y, stretch(stim_ref(*ref), 17e-6))
        p = int(np.argmax(np.abs(h[:3 * FS]).sum(1)))
        f, ab, aa, bb, n = frames_in(h, p + int(t0 * 16), p + int(t1 * 16), nper, hop)
        e = (aa.sum() + bb.sum())
        s = (ab / e, aa / e, bb / e)  # each sweep weighted equally
        acc = s if acc is None else tuple(a + b for a, b in zip(acc, s))
        n_tot += n
    return f, acc[0], acc[1], acc[2], n_tot


def main():
    out = {}
    S1 = [("rec/" + k, v) for k, v in SESSION1.items()]
    POST = [(f"rec2/reboot_{i:02d}", None) for i in range(8)] + [("rec2/s3_ambient_after", None)]
    SW1 = [("rec/sweep_L", ("stim/sweep_L.npz", "L")), ("rec/sweep_R", ("stim/sweep_R.npz", "R")),
           ("rec/sweep_L2", ("stim/sweep_L.npz", "L")), ("rec2/s2_sweep_L", ("stim/sweep_L.npz", "L")),
           ("rec2/s2_sweep_R", ("stim/sweep_R.npz", "R"))]
    SW2 = [("rec2/s3_sweep_L", ("stim/sweep_L.npz", "L"))]
    sets = {"FAN1": collect_noise(S1), "FAN2": collect_noise(POST)}
    for t0, t1 in [(30, 250), (80, 250)]:
        sets[f"REV1_{t0}_{t1}"] = collect_reverb(SW1, t0, t1)
        sets[f"REV2_{t0}_{t1}"] = collect_reverb(SW2, t0, t1)
    centers = np.arange(250, 7751, 125.0)
    d_grid = np.arange(0.0080, 0.02601, 0.0002)
    m_list = np.concatenate([np.arange(-0.5, 0.0, 0.1), np.arange(0.0, 8.01, 0.25)])
    betas = np.arange(0.0, 0.951, 0.025)
    for name, (f, ab, aa, bb, n) in sets.items():
        tau = delay_ls(f, ab, aa, bb, n, 300, 3000)[0]
        g, ph = band_coh(f, ab, aa, bb, tau, centers)
        o = {"tau_300_3000": r(tau), "n_frames": int(n),
             "abs_coh_every_500Hz": [[int(c), r(gg), r(pp, 1)] for c, gg, pp in zip(centers, g, ph) if c % 500 == 0]}
        for lo, hi in [(1000, 6500), (2000, 6500), (1000, 5000), (300, 6500), (1000, 7500)]:
            sel = (centers >= lo) & (centers <= hi)
            fc, gg = centers[sel], g[sel]
            sse, rho = fit_family(fc, gg, d_grid, m_list)
            npts = sel.sum()
            j0i = int(np.argmin(np.abs(m_list - 0.0)))
            jJ = int(np.argmin(np.abs(m_list + 0.5)))
            i_s = int(np.argmin(sse[:, j0i]))
            i_j = int(np.argmin(sse[:, jJ]))
            best = np.unravel_index(int(np.argmin(sse)), sse.shape)
            # rho fixed at 1, sinc
            sse1 = [np.sum((gg - np.sinc(2 * fc * d / C)) ** 2) for d in d_grid]
            i1 = int(np.argmin(sse1))
            prof = np.min(sse, 1)  # best m for each d
            rm = np.sqrt(prof / npts)
            rmin = rm.min()
            ok = d_grid[rm <= 1.1 * rmin]
            entry = {"sinc": [r(100 * d_grid[i_s]), r(rho[i_s, j0i]), r(np.sqrt(sse[i_s, j0i] / npts), 4)],
                     "sinc_rho1": [r(100 * d_grid[i1]), r(np.sqrt(sse1[i1] / npts), 4)],
                     "J0": [r(100 * d_grid[i_j]), r(rho[i_j, jJ]), r(np.sqrt(sse[i_j, jJ] / npts), 4)],
                     "best_m": [r(100 * d_grid[best[0]]), r(m_list[best[1]], 2), r(rho[best]), r(np.sqrt(sse[best] / npts), 4)],
                     "d_range_cm_with_rmse<=1.1*min_over_m": [r(100 * ok.min()), r(100 * ok.max())] if len(ok) else None,
                     "profile_rmse_at_d_cm": {f"{100 * d:.1f}": r(np.sqrt(prof[int(np.argmin(np.abs(d_grid - d)))] / npts), 4)
                                              for d in [0.010, 0.012, 0.013, 0.014, 0.016, 0.018, 0.020, 0.024]},
                     "best_m_at_d_cm": {f"{100 * d:.1f}": r(m_list[int(np.argmin(sse[int(np.argmin(np.abs(d_grid - d)))]))], 2)
                                        for d in [0.010, 0.012, 0.013, 0.014, 0.016, 0.018, 0.020, 0.024]}}
            msse = fit_mixture(fc, gg, d_grid, betas)
            prof_mix = np.sqrt(np.min(msse, 1) / npts)
            entry["mixture_profile_rmse_at_d_cm"] = {f"{100 * d:.1f}": r(prof_mix[int(np.argmin(np.abs(d_grid - d)))], 4)
                                                     for d in [0.012, 0.014, 0.016, 0.018, 0.020, 0.024]}
            o[f"fit_{lo}_{hi}"] = entry
        # statistical noise of |coh| estimates: rough, from the band coherence of independent halves of the frequency bins
        out[name] = o
        print(name, "tau", o["tau_300_3000"], "n", n)
        print("   |coh| every 500 Hz", o["abs_coh_every_500Hz"])
        for k, v in o.items():
            if k.startswith("fit_"):
                print("  ", k, "sinc", v["sinc"], "rho1", v["sinc_rho1"], "J0", v["J0"], "best", v["best_m"],
                      "range", v["d_range_cm_with_rmse<=1.1*min_over_m"])
                print("        profile", v["profile_rmse_at_d_cm"], "m*", v["best_m_at_d_cm"])
                print("        mixture", v["mixture_profile_rmse_at_d_cm"])
    with open(os.path.join(HERE, "v1_d.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
