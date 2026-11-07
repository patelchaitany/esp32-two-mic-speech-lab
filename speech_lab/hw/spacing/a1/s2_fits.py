"""Quantitative fits with uncertainty for d and tau0, plus constrained fits of the competing hypotheses.

A. Ambient (fan) field, loud-fan stretches: 1-s blocks, block bootstrap (B reps) of 3-D (sinc) and 2-D (J0) fits
   in several bands, free rho; also rho fixed to 1.
B. Noise-floor-robust coherence: across ALL ambient 1-s blocks of the 10 recordings (fan level varies ~20 dB at HF),
   regress Re{P_AB e^{-j w tau0}} on sqrt(P_AA P_BB) per 250 Hz band -> slope = coherence of the varying (acoustic)
   part, intercept -> stationary incoherent floor. Fit sinc/J0 to the slopes.
C. Reverberant windows of 10 room responses (3 sweeps + 7 estimated from speech/noise references), frequency-smoothed
   coherence, pooled; bootstrap over responses.
D. Residual phase after removing the linear skew, 100 Hz - 2 kHz (mic high-pass mismatch check).
E. Hypothesis-constrained fits: tau0 fixed (H2: -0.5, H3: 0), model rho*e^{jw tau0}[beta e^{jwg} + (1-beta) D(f;d)]
   with |g| <= d/c and d bounds; compare with the free diffuse+skew model (H1-type) and the free 5-parameter model.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, FS, HERE, dmodel, fit_diffuse, frames_spec, irs, load, r3  # noqa: E402

RNG = np.random.default_rng(1234)
NBOOT = 200
EXPL = json.load(open(os.path.join(HERE, "s0_explore.json")))
SEGS = {k: v for k, v in EXPL["segments"].items()}
LOUD = ["ambient", "sweep_L", "sweep_R", "sweep_L2", "talkA_L", "talkB_R"]


def refine(f, g, kind, x0, w=None, rho_max=1.0, fix_rho=None):
    w = np.ones_like(f) if w is None else w

    def cost(p):
        dd, tt = p[0], p[1]
        rr = fix_rho if fix_rho is not None else p[2]
        m = rr * dmodel(f, dd, kind)[0] * np.exp(2j * np.pi * f * tt / FS)
        return np.sum(w * np.abs(g - m) ** 2)
    x0 = list(x0[:2]) + ([] if fix_rho is not None else [x0[2]])
    bnds = [(0.002, 0.06), (-2.5, 1.5)] + ([] if fix_rho is not None else [(0, rho_max)])
    r = minimize(cost, x0, method="L-BFGS-B", bounds=bnds)
    d, t = r.x[0], r.x[1]
    rho = fix_rho if fix_rho is not None else r.x[2]
    rmse = float(np.sqrt(r.fun / np.sum(w)))
    return 100 * d, t, rho, rmse


def block_sums(names, blk_s=1.0, nper=512, hop=256):
    """Per-1s-block sums of A conj B, |A|^2, |B|^2 (nper=512 frames). Returns f, list of (name, Pab, Paa, Pbb, nfr)."""
    out = []
    for name in names:
        x = load(name)
        for a, b in SEGS[name]:
            n = int(blk_s * FS)
            for s in range(a, b - n + 1, n):
                f, A, B = frames_spec(x[s:s + n], nper, hop)
                out.append((name, np.sum(A * np.conj(B), 0), np.sum(np.abs(A) ** 2, 0), np.sum(np.abs(B) ** 2, 0), len(A)))
    return f, out


def part_a(res):
    f, blocks = block_sums(LOUD)
    Pab = np.array([b[1] for b in blocks])
    Paa = np.array([b[2] for b in blocks])
    Pbb = np.array([b[3] for b in blocks])
    nb = len(blocks)
    bands = [(150, 7000), (300, 6500), (1000, 6500), (1000, 6000), (2000, 6500), (500, 4000), (1000, 5000)]
    out = {"n_blocks": nb, "fits": {}}
    for kind in ("sinc", "j0"):
        for lo, hi in bands:
            m = (f >= lo) & (f <= hi)
            g = Pab.sum(0)[m] / np.sqrt(Paa.sum(0)[m] * Pbb.sum(0)[m])
            full = fit_diffuse(f[m], g, kind)
            x0 = [full["d_cm"] / 100, full["tau0"], full["rho"]]
            bd, bt, br = [], [], []
            for _ in range(NBOOT):
                idx = RNG.integers(0, nb, nb)
                gb = Pab[idx].sum(0)[m] / np.sqrt(Paa[idx].sum(0)[m] * Pbb[idx].sum(0)[m])
                d, t, rho, _ = refine(f[m], gb, kind, x0)
                bd.append(d)
                bt.append(t)
                br.append(rho)
            key = f"{kind}_{lo}_{hi}"
            out["fits"][key] = {"d_cm": r3(full["d_cm"]), "d_ci95": [r3(np.percentile(bd, 2.5)), r3(np.percentile(bd, 97.5))],
                                "tau0": r3(full["tau0"]), "tau0_ci95": [r3(np.percentile(bt, 2.5)), r3(np.percentile(bt, 97.5))],
                                "rho": r3(full["rho"]), "rmse": r3(full["rmse"]), "ph_rms_deg": r3(full["ph_rms_deg"])}
            # rho fixed to 1
            d1, t1, _, e1 = refine(f[m], g, kind, x0, fix_rho=1.0)
            out["fits"][key]["rho1_d_cm"] = r3(d1)
            out["fits"][key]["rho1_rmse"] = r3(e1)
            print("A", key, out["fits"][key], flush=True)
    # per-recording (block group) fits for stability, sinc 1000-6500
    m = (f >= 1000) & (f <= 6500)
    per = {}
    for name in LOUD:
        sel = np.array([b[0] == name for b in blocks])
        g = Pab[sel].sum(0)[m] / np.sqrt(Paa[sel].sum(0)[m] * Pbb[sel].sum(0)[m])
        r = fit_diffuse(f[m], g, "sinc")
        per[name] = {k: r3(v) for k, v in r.items()}
    out["per_recording_sinc_1000_6500"] = per
    print("A per recording", per, flush=True)
    # |coherence| and residual phase every 250 Hz (all loud blocks), for the record
    g = Pab.sum(0) / np.sqrt(Paa.sum(0) * Pbb.sum(0))
    out["table_250Hz"] = [[int(fc), r3(np.abs(np.mean(g[(f >= fc - 125) & (f < fc + 125)]))),
                           r3(np.degrees(np.angle(np.mean((g * np.exp(2j * np.pi * f * 1.10 / FS))[(f >= fc - 125) & (f < fc + 125)]))))]
                          for fc in range(250, 7900, 250)]
    res["A_ambient_loud"] = out


def part_b(res):
    f, blocks = block_sums(list(SEGS.keys()))
    Pab = np.array([b[1] for b in blocks]) / np.array([b[4] for b in blocks])[:, None]
    Paa = np.array([b[2] for b in blocks]) / np.array([b[4] for b in blocks])[:, None]
    Pbb = np.array([b[3] for b in blocks]) / np.array([b[4] for b in blocks])[:, None]
    rot = np.exp(-2j * np.pi * f * (-1.10) / FS)
    rows = []
    fc_list = np.arange(375, 7700, 250)
    slopes, ints, fcs, r2s = [], [], [], []
    for fc in fc_list:
        m = (f >= fc - 125) & (f < fc + 125)
        y = np.real((Pab[:, m] * rot[m]).sum(1))
        x = np.sqrt(Paa[:, m].sum(1) * Pbb[:, m].sum(1))
        X = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        floor_rel = -coef[1] / coef[0] / np.median(x)  # incoherent floor relative to the median block power
        # simple per-band coherence of the pooled loud data for comparison
        rows.append([int(fc), r3(coef[0]), r3(floor_rel), r3(r2), r3(10 * np.log10(x.max() / x.min()))])
        slopes.append(coef[0])
        ints.append(coef[1])
        fcs.append(fc)
        r2s.append(r2)
    fcs = np.array(fcs, float)
    sl = np.array(slopes)
    out = {"cols": ["f_center", "slope=coherence_of_varying_part", "floor_re_median_power", "R2", "block_power_range_dB"],
           "rows": rows}
    for kind in ("sinc", "j0"):
        for lo, hi in [(375, 6500), (1000, 6500), (1000, 7600), (2000, 6500)]:
            m = (fcs >= lo) & (fcs <= hi)
            # real-valued fit of rho*D(f;d) to the slopes
            best = None
            for d in np.arange(0.004, 0.04, 0.0001):
                D = dmodel(fcs[m], d, kind)[0]
                rho = min(1.0, max(0.0, np.sum(D * sl[m]) / np.sum(D * D)))
                e = np.sum((sl[m] - rho * D) ** 2)
                if best is None or e < best[0]:
                    best = (e, d, rho)
            e, d, rho = best
            # rho fixed 1
            best1 = min(((np.sum((sl[m] - dmodel(fcs[m], dd, kind)[0]) ** 2), dd) for dd in np.arange(0.004, 0.04, 0.0001)))
            out[f"fit_{kind}_{lo}_{hi}"] = {"d_cm": r3(100 * d), "rho": r3(rho), "rmse": r3(np.sqrt(e / m.sum())),
                                            "rho1_d_cm": r3(100 * best1[1]), "rho1_rmse": r3(np.sqrt(best1[0] / m.sum()))}
    # bootstrap over blocks for the sinc 1000-6500 fit of the slopes
    nb = len(blocks)
    bd = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, nb, nb)
        s_b = []
        for fc in fcs:
            m = (f >= fc - 125) & (f < fc + 125)
            y = np.real((Pab[idx][:, m] * rot[m]).sum(1))
            x = np.sqrt(Paa[idx][:, m].sum(1) * Pbb[idx][:, m].sum(1))
            X = np.stack([x, np.ones_like(x)], 1)
            s_b.append(np.linalg.lstsq(X, y, rcond=None)[0][0])
        s_b = np.array(s_b)
        mm = (fcs >= 1000) & (fcs <= 6500)
        best = None
        for d in np.arange(0.006, 0.025, 0.0002):
            D = dmodel(fcs[mm], d, "sinc")[0]
            rho = min(1.0, max(0.0, np.sum(D * s_b[mm]) / np.sum(D * D)))
            e = np.sum((s_b[mm] - rho * D) ** 2)
            if best is None or e < best[0]:
                best = (e, d)
        bd.append(100 * best[1])
    out["fit_sinc_1000_6500"]["d_ci95_boot"] = [r3(np.percentile(bd, 2.5)), r3(np.percentile(bd, 97.5))]
    out["n_blocks"] = nb
    res["B_regression_all_blocks"] = out
    for r in rows:
        print("B", r)
    print("B fits", {k: v for k, v in out.items() if k.startswith("fit")}, flush=True)


def smoothed_window(h, t0_ms, t1_ms, p0, N=16384, sm_hz=250.0):
    i0, i1 = p0 + int(t0_ms * 16), p0 + int(t1_ms * 16)
    from scipy.signal.windows import tukey
    seg = h[i0:i1] * tukey(i1 - i0, 0.1)[:, None]
    A, B = np.fft.rfft(seg[:, 0], N), np.fft.rfft(seg[:, 1], N)
    k = int(round(sm_hz / (FS / N)))
    ker = np.ones(k) / k
    cs = lambda v: np.convolve(v.real, ker, "same") + 1j * np.convolve(v.imag, ker, "same")
    Pab, Paa, Pbb = cs(A * np.conj(B)), np.convolve(np.abs(A) ** 2, ker, "same"), np.convolve(np.abs(B) ** 2, ker, "same")
    e = np.sum(Paa + Pbb)
    return np.fft.rfftfreq(N, 1 / FS), Pab / e, Paa / e, Pbb / e


def all_responses():
    import s3_speech as S3
    H = {k: (v, 80) for k, v in irs().items()}
    for name, keys in [("talkA_L", ["L"]), ("talkB_R", ["R"]), ("talkAB_LR", ["L", "R"]), ("talkA_noise", ["L"]),
                       ("talkC_R_quiet", ["R"])]:
        h, _ = S3.estimate_irs(name, keys if name != "talkA_noise" else ["L", "R"])
        for i, k in enumerate(keys):
            hk = h[:, :, i]
            p = int(np.argmax(np.abs(hk).sum(1)[:4000]))
            H[f"{name}:{k}"] = (hk, p)
    # talkAB_LR gives two (L and R): total 3 + 1 + 1 + 2 + 1 + 1 = 9 responses
    return H


def part_c(res, H):
    out = {}
    names = list(H.keys())
    for (t0, t1) in [(10, 80), (80, 250), (30, 250)]:
        specs = {}
        for n in names:
            h, p = H[n]
            if p + int(t1 * 16) > len(h):
                continue
            specs[n] = smoothed_window(h, t0, t1, p)
        f = next(iter(specs.values()))[0]
        idx = np.where((f >= 150) & (f <= 7000))[0][::16]
        ff = f[idx]
        keys = list(specs.keys())

        def pooled(sel):
            Pab = sum(specs[k][1][idx] for k in sel)
            Paa = sum(specs[k][2][idx] for k in sel)
            Pbb = sum(specs[k][3][idx] for k in sel)
            return Pab / np.sqrt(Paa * Pbb)
        g = pooled(keys)
        o = {"responses": keys}
        for kind in ("sinc", "j0"):
            for lo, hi in [(150, 7000), (1000, 6500), (2000, 6500)]:
                m = (ff >= lo) & (ff <= hi)
                full = fit_diffuse(ff[m], g[m], kind)
                x0 = [full["d_cm"] / 100, full["tau0"], full["rho"]]
                bd, bt = [], []
                for _ in range(NBOOT):
                    sel = [keys[i] for i in RNG.integers(0, len(keys), len(keys))]
                    d, t, rho, _ = refine(ff[m], pooled(sel)[m], kind, x0)
                    bd.append(d)
                    bt.append(t)
                o[f"{kind}_{lo}_{hi}"] = {"d_cm": r3(full["d_cm"]), "d_ci95": [r3(np.percentile(bd, 2.5)), r3(np.percentile(bd, 97.5))],
                                          "tau0": r3(full["tau0"]), "tau0_ci95": [r3(np.percentile(bt, 2.5)), r3(np.percentile(bt, 97.5))],
                                          "rho": r3(full["rho"]), "rmse": r3(full["rmse"]), "ph_rms_deg": r3(full["ph_rms_deg"])}
                d1, _, _, e1 = refine(ff[m], g[m], kind, x0, fix_rho=1.0)
                o[f"{kind}_{lo}_{hi}"]["rho1_d_cm"] = r3(d1)
                o[f"{kind}_{lo}_{hi}"]["rho1_rmse"] = r3(e1)
        # per response tau0/d (sinc 1000-6500)
        o["per_response_sinc_1000_6500"] = {}
        for k in keys:
            gk = pooled([k])
            m = (ff >= 1000) & (ff <= 6500)
            r = fit_diffuse(ff[m], gk[m], "sinc")
            o["per_response_sinc_1000_6500"][k] = [r3(r["d_cm"]), r3(r["tau0"]), r3(r["rho"])]
        out[f"{t0}_{t1}ms"] = o
        print("C", t0, t1, json.dumps(o), flush=True)
    res["C_reverb_windows"] = out


def part_d(res, H):
    """Residual phase (deg) after removing tau0=-1.10, in 1/3-octave bands 100-2000 Hz: ambient and reverb 30-250 ms."""
    centers = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000]
    f, blocks = block_sums(LOUD, nper=2048, hop=512)
    Pab = sum(b[1] for b in blocks)
    Paa = sum(b[2] for b in blocks)
    Pbb = sum(b[3] for b in blocks)
    rot = np.exp(2j * np.pi * f * 1.10 / FS)
    amb = []
    for fc in centers:
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        g = np.sum(Pab[m] * rot[m]) / np.sqrt(np.sum(Paa[m]) * np.sum(Pbb[m]))
        amb.append([fc, r3(np.degrees(np.angle(g))), r3(np.abs(g))])
    rev = []
    acc = None
    for k, (h, p) in H.items():
        ff, Pab_, Paa_, Pbb_ = smoothed_window(h, 30, 250, p, sm_hz=20.0)
        acc = (Pab_, Paa_, Pbb_) if acc is None else (acc[0] + Pab_, acc[1] + Paa_, acc[2] + Pbb_)
    rot2 = np.exp(2j * np.pi * ff * 1.10 / FS)
    for fc in centers:
        m = (ff >= fc / 2 ** (1 / 6)) & (ff < fc * 2 ** (1 / 6))
        g = np.sum(acc[0][m] * rot2[m]) / np.sqrt(np.sum(acc[1][m]) * np.sum(acc[2][m]))
        rev.append([fc, r3(np.degrees(np.angle(g))), r3(np.abs(g))])
    # direct sound residual phase (re its own fitted delay), sweep responses, 4 ms window
    res["D_lowfreq_phase"] = {"cols": ["f_center_Hz", "residual_phase_deg_after_-1.10", "|coh|"],
                              "ambient_loud": amb, "reverb_30_250ms_all_responses": rev}
    print("D ambient", amb)
    print("D reverb", rev, flush=True)


def mix_model(p, f, tau0, kind):
    rho, beta, g, d = p
    return rho * np.exp(2j * np.pi * f * tau0 / FS) * (beta * np.exp(2j * np.pi * f * g / FS) + (1 - beta) * dmodel(f, d, kind)[0])


def fit_mix(f, gam, tau0, kind, dmin, dmax=0.06, tau_free=False):
    best = None
    for d0 in np.linspace(dmin, min(dmax, max(dmin * 2, 0.03)), 4):
        gmax = d0 * FS / C
        for g0 in np.linspace(-gmax, gmax, 5):
            for b0 in (0.1, 0.5, 0.9):
                if tau_free:
                    for t0 in (-1.5, -1.1, -0.5, 0.0):
                        x0 = [0.95, b0, g0, d0, t0]

                        def cost(p):
                            if abs(p[2]) > p[3] * FS / C:
                                return 1e6
                            return np.sum(np.abs(gam - mix_model(p[:4], f, p[4], kind)) ** 2)
                        r = minimize(cost, x0, method="Powell",
                                     bounds=[(0, 1), (0, 1), (-2.8, 2.8), (dmin, dmax), (-2.5, 1.5)])
                        if best is None or r.fun < best[0]:
                            best = (r.fun, r.x)
                else:
                    x0 = [0.95, b0, g0, d0]

                    def cost(p):
                        if abs(p[2]) > p[3] * FS / C:
                            return 1e6
                        return np.sum(np.abs(gam - mix_model(p, f, tau0, kind)) ** 2)
                    r = minimize(cost, x0, method="Powell", bounds=[(0, 1), (0, 1), (-2.8, 2.8), (dmin, dmax)])
                    if best is None or r.fun < best[0]:
                        best = (r.fun, r.x)
    e, x = best
    out = {"rmse": r3(np.sqrt(e / len(f))), "rho": r3(x[0]), "beta_directional": r3(x[1]), "g_dir": r3(x[2]),
           "d_cm": r3(100 * x[3])}
    if tau_free:
        out["tau0"] = r3(x[4])
    return out


def part_e(res, H):
    out = {}
    # data sets: ambient loud (sinc band 300-6500, 62.5 Hz grid) and pooled reverb 80-250 ms
    f, blocks = block_sums(LOUD)
    Pab = sum(b[1] for b in blocks)
    Paa = sum(b[2] for b in blocks)
    Pbb = sum(b[3] for b in blocks)
    m = (f >= 300) & (f <= 6500)
    sets = {"ambient": (f[m][::2], (Pab / np.sqrt(Paa * Pbb))[m][::2])}
    acc = None
    for k, (h, p) in H.items():
        ff, a, b, c = smoothed_window(h, 80, 250, p)
        acc = (a, b, c) if acc is None else (acc[0] + a, acc[1] + b, acc[2] + c)
    idx = np.where((ff >= 300) & (ff <= 6500))[0][::16]
    sets["reverb_80_250"] = (ff[idx], (acc[0] / np.sqrt(acc[1] * acc[2]))[idx])
    acc = None
    for k, (h, p) in H.items():
        ff, a, b, c = smoothed_window(h, 10, 80, p)
        acc = (a, b, c) if acc is None else (acc[0] + a, acc[1] + b, acc[2] + c)
    sets["early_10_80"] = (ff[idx], (acc[0] / np.sqrt(acc[1] * acc[2]))[idx])
    for sname, (fs_, gam) in sets.items():
        o = {}
        r = fit_diffuse(fs_, gam, "sinc")
        o["H1type_diffuse+skew_free(3p)"] = {k: r3(v) for k, v in r.items()}
        o["free_mixture(5p)"] = fit_mix(fs_, gam, None, "sinc", 0.004, tau_free=True)
        o["H2_tau0=-0.5_d>=1.4(4p)"] = fit_mix(fs_, gam, -0.5, "sinc", 0.014)
        o["H2_tau0=-0.5_d>=0.4(4p)"] = fit_mix(fs_, gam, -0.5, "sinc", 0.004)
        o["H3_tau0=0_d>=2.5(4p)"] = fit_mix(fs_, gam, 0.0, "sinc", 0.025)
        o["tau0=-1.0_d>=0.4(4p)"] = fit_mix(fs_, gam, -1.0, "sinc", 0.004)
        o["tau0=-1.5_d>=0.4(4p)"] = fit_mix(fs_, gam, -1.5, "sinc", 0.004)
        out[sname] = o
        print("E", sname, json.dumps(o), flush=True)
    res["E_hypothesis_fits"] = out


def main():
    res = {}
    part_a(res)
    part_b(res)
    H = all_responses()
    part_c(res, H)
    part_d(res, H)
    part_e(res, H)
    with open(os.path.join(HERE, "s2_fits.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
