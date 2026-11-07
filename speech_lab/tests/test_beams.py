"""Tests for methods/_core/beams.py and the M1/M2 wrappers (DESIGN.md M1, M2, Appendix B.1, B.6-B.8).

Run from LAB: .venv/bin/python -m tests.test_beams [--quick]
--quick skips the end-to-end runs on synthetic scenes. Plain asserts; prints the measured numbers.
"""
import sys
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

from lab.common import C, FS
from lab.scenes import coherent_noise
from methods import m1_das, m2_bus_beam, m2_meeting_beam
from methods._core import beams as bm, calib, common as cm, tracker as tk

F = cm.FREQS
SPACINGS = (0.025, 0.04, 0.06)
BUS = {"scenario": "bus", "spacing_m": 0.04}
MEETING = {"scenario": "meeting", "spacing_m": 0.04}
GEOMS = {"endfire 0/180": (0, 180), "aligned 0/170": (0, 170), "asym 20/150": (20, 150), "45/135": (45, 135),
         "30/120": (30, 120), "60/100": (60, 100), "same side 10/50": (10, 50)}


def check(name, ok, msg):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {msg}")
    assert ok, f"{name}: {msg}"


def tau_of(deg, d):
    return d * np.cos(np.radians(deg)) / C


def frac_delay(x, tau):
    """Delay x by tau seconds (FFT phase shift, zero-padded to avoid wrap-around)."""
    n = len(x)
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.exp(-2j * np.pi * f * tau), nfft)[:n]


def deg(z):
    return np.degrees(np.angle(z))


# ------------------------------------------------------------------ floors and DAS
def test_floors_das():
    print("WNG floors and delay-and-sum")
    fb = bm.wng_floor("bus", np.array([250, 499, 500, 1000, 2000, 2001, 7000.0]))
    check("bus floors", np.array_equal(fb, [-10, -10, -10, -10, -10, -6, -6]), str(fb))
    check("meeting floor", np.all(bm.wng_floor("meeting") == -3.0), "-3 dB")
    err = max(np.max(np.abs(np.abs(bm.response(bm.das_weights(F, t), cm.steer(F, t))) - 1))
              for t in (-1.5e-4, 0.0, 7e-5))
    check("B.1 |w^H a| = 1 (DAS)", err < 1e-12, f"max dev {err:.1e}")


# ------------------------------------------------------------------ B.1 single target
def test_mvdr():
    print("B.1 single-target MVDR: |w^H a| = 1, WNG >= floor, smallest mu")
    worst_r, worst_m, worst_bind = 0.0, np.inf, 0.0
    for d in SPACINGS:
        g = cm.diffuse_model(F, d, "sinc")
        for floor in (bm.wng_floor("bus"), bm.wng_floor("meeting"), np.full(len(F), -10.0)):
            for t in (-d / C, 0.0, d / C):
                a = cm.steer(F, t)
                w, mu = bm.mvdr_weights(a, g, floor)
                worst_r = max(worst_r, np.max(np.abs(np.abs(bm.response(w, a)) - 1)))
                margin = bm.wng_db(w, a) - floor
                worst_m = min(worst_m, margin.min())
                bind = mu > 1e-8                                   # floor binds: WNG must sit on it
                if bind.any():
                    worst_bind = max(worst_bind, np.max(margin[bind]))
    check("|w^H a| = 1 +- 1e-6", worst_r < 1e-6, f"max dev {worst_r:.1e} (2.5/4/6 cm, tau = -d/c, 0, +d/c)")
    check("WNG >= floor in every bin", worst_m >= -1e-6, f"min margin {worst_m:+.2e} dB")
    check("bisection finds the smallest mu", worst_bind < 0.01, f"max margin where the floor binds {worst_bind:.4f} dB")
    w, _ = bm.mvdr_weights(cm.steer(F, 0.0), cm.diffuse_model(F, 0.04, "sinc"), bm.wng_floor("bus"))
    check("broadside MVDR with real Gamma = average", np.max(np.abs(w - 0.5)) < 1e-9, "w = [0.5, 0.5]")


# ------------------------------------------------------------------ B.1 two-talker beam
def test_two_talker_b1():
    print("B.1 two-talker beam, partners at +-d/c, -10 dB floor, center-referenced steering")
    for d, edge in zip(SPACINGS, (985.0, 616.0, 410.0)):
        g = cm.diffuse_model(F, d, "sinc")
        tt = bm.two_talker(F, d / C, -d / C, g, -10.0, d)
        first = F[F >= edge][0]
        check(f"d={d*1e3:.0f} mm dipole band", tt.band is not None and tt.band[0] == first,
              f"{tt.band} (DESIGN: floor-feasible from {edge:.0f} Hz, first bin {first} Hz)")
        k = (F >= first) & (F <= 0.8 * C / (2 * d))
        aA, aB = cm.steer(F, d / C), cm.steer(F, -d / C)
        rA, rB = bm.response(tt.w, aA), bm.response(tt.w, aB)
        mag = max(np.max(np.abs(20 * np.log10(np.abs(rA[k])))), np.max(np.abs(20 * np.log10(np.abs(rB[k])))))
        ph = max(np.max(np.abs(deg(rA[k]))), np.max(np.abs(deg(-rB[k]))))
        check(f"d={d*1e3:.0f} mm r_A = +1, r_B = -1 on {first:.0f}-{F[k][-1]:.0f} Hz", mag <= 1.0 and ph <= 10.0,
              f"max |gain| {mag:.3f} dB, max phase dev {ph:.2f} deg")
        mA, mB = bm.response(tt.w, aA * np.exp(-1j * np.pi * F * d / C)[:, None]), \
            bm.response(tt.w, aB * np.exp(1j * np.pi * F * d / C)[:, None])   # Mic-A-referenced [1, e^{-j2pi f tau}]
        dev = np.max(np.abs(deg(mB[k] / mA[k] / (-np.exp(2j * np.pi * F[k] * d / C)))))
        check(f"d={d*1e3:.0f} mm convention check r_B/r_A = -e^(+j2pi f d/c) (Mic-A ref)", dev <= 10.0,
              f"max dev {dev:.2f} deg")
        wng = 10 * np.log10(np.minimum(np.abs(rA) ** 2, np.abs(rB) ** 2) / bm.white_gain(tt.w))
        check(f"d={d*1e3:.0f} mm WNG >= -10 dB in every bin", wng.min() >= -10 - 1e-6, f"min {wng.min():.4f} dB")
        bal = np.max(np.abs(20 * np.log10(np.abs(rB))))
        step = np.max(np.abs(np.diff(tt.psi_deg)))
        check(f"d={d*1e3:.0f} mm balance and bin-to-bin phase steps", bal <= 1.0 + 1e-9 and step <= 30.0 + 1e-6,
              f"max |r_B| dev {bal:.3f} dB, max step {step:.2f} deg, Mic-A-alone bins {int(np.sum(tt.kind == 2))}")


def test_two_talker_geometries():
    print("Two-talker beam, other geometries and the bus floors: WNG, balance, steps, DF vs the per-bin optimum")
    grid = np.exp(1j * np.radians(np.arange(0, 360, bm.PSI_STEP)))
    for d in SPACINGS:
        g = cm.diffuse_model(F, d, "sinc")
        for floor_name in ("bus", "-10"):
            fl = bm.wng_floor("bus") if floor_name == "bus" else np.full(len(F), -10.0)
            out = []
            for name, (da, db) in GEOMS.items():
                ta, tb = tau_of(da, d), tau_of(db, d)
                tt = bm.two_talker(F, ta, tb, g, fl, d)
                aA, aB = cm.steer(F, ta), cm.steer(F, tb)
                wng = 10 * np.log10(np.minimum(1.0, np.abs(tt.h_B) ** 2) / bm.white_gain(tt.w))
                step = np.max(np.abs(np.diff(tt.psi_deg)))
                bal = np.max(np.abs(20 * np.log10(np.abs(tt.h_B))))
                assert np.all(wng >= fl - 1e-6), f"{name}: WNG below the floor"
                assert np.allclose(bm.response(tt.w, aA), 1.0), f"{name}: r_A != 1"
                assert bal <= 1.0 + 1e-9 and step <= 30.0 + 1e-6, f"{name}: balance {bal:.2f} dB / step {step:.1f}"
                with np.errstate(all="ignore"):     # per-bin optimum without the continuity constraint
                    W, ok = bm.lcmv2_weights(aA[:, None], aB[:, None], grid[None], g[:, None], fl[:, None])
                    rA, rB = bm.response(W, aA[:, None]), bm.response(W, aB[:, None])
                    s = 0.5 * (np.abs(rA) ** 2 + np.abs(rB) ** 2) / bm.noise_power(W, g[:, None])
                    s = np.where(ok & (np.abs(10 * np.log10(np.abs(rA) ** 2 / np.abs(rB) ** 2)) <= 1), s, 0)
                opt = bm.ai_index(10 * np.log10(np.maximum(s.max(axis=1), 1.0)))
                got = bm.ai_index(tt.df_db)
                assert got >= opt - 0.05, f"{name}: AI-DF {got:.2f} vs per-bin optimum {opt:.2f}"
                out.append(f"{name} {got:.2f}/{opt:.2f}")
            check(f"d={d*1e3:.0f} mm floors {floor_name}: WNG, r_A = 1, balance <= 1 dB, steps <= 30 deg, "
                  "AI-DF within 0.05 dB of the per-bin optimum", True, "; ".join(out))


def greedy_design_rule(ta, tb, g, fl, d):
    """DESIGN M2's literal phase rule, for comparison: psi = 180 deg on the dipole band (contiguous bins ending at
    c/2d where h = [1, -1] qualifies), then bin by bin away from it: best DF within +-30 deg of the neighbour's
    grid psi, else the full circle, else Mic A. Returns (AI-DF dB, max bin-to-bin step of arg(r_B/r_A) deg)."""
    grid = np.arange(0, 360, bm.PSI_STEP)
    aA, aB = cm.steer(F, ta), cm.steer(F, tb)
    with np.errstate(all="ignore"):
        W, ok = bm.lcmv2_weights(aA[:, None], aB[:, None], np.exp(1j * np.radians(grid))[None], g[:, None], fl[:, None])
        rA, rB = bm.response(W, aA[:, None]), bm.response(W, aB[:, None])
        df = 0.5 * (np.abs(rA) ** 2 + np.abs(rB) ** 2) / bm.noise_power(W, g[:, None])
        s = np.where(ok & (np.abs(10 * np.log10(np.abs(rA) ** 2 / np.abs(rB) ** 2)) <= 1), df, -np.inf)
    i180 = int(round(180 / bm.PSI_STEP))
    top = int(np.searchsorted(F, min(C / (2 * d), F[-1]), side="right")) - 1
    lo = top
    while lo > 0 and np.isfinite(s[lo - 1, i180]):
        lo -= 1
    pick = np.full(len(F), -1)
    pick[lo:top + 1] = i180
    for run in (range(lo - 1, -1, -1), range(top + 1, len(F))):
        prev = 180.0
        for k in run:
            win = (int(round(prev / bm.PSI_STEP)) + np.arange(-12, 13)) % len(grid)
            j = win[np.argmax(s[k, win])] if np.isfinite(s[k, win].max()) else \
                (int(np.argmax(s[k])) if np.isfinite(s[k].max()) else -1)
            pick[k] = j
            p = grid[j] if j >= 0 else np.degrees(np.pi * F[k] * (tb - ta)) % 360
            prev += (p - prev + 180) % 360 - 180
    rows = np.arange(len(F))
    dfs = np.where(pick >= 0, 10 * np.log10(np.maximum(df[rows, pick], 1e-12)), 0.0)
    ph = np.where(pick >= 0, np.angle(rB[rows, pick] / rA[rows, pick]), np.pi * F * (tb - ta))
    return bm.ai_index(dfs), float(np.max(np.abs(np.diff(np.degrees(np.unwrap(ph))))))


def test_design_rule():
    print("Phase rule as implemented (Viterbi) vs DESIGN's literal greedy rule (bus floors and -10 dB)")
    for d in SPACINGS:
        g = cm.diffuse_model(F, d, "sinc")
        out = []
        for floor_name, fl in (("bus", bm.wng_floor("bus")), ("-10", np.full(len(F), -10.0))):
            for name in ("endfire 0/180", "asym 20/150", "60/100"):
                ta, tb = (tau_of(x, d) for x in GEOMS[name])
                tt = bm.two_talker(F, ta, tb, g, fl, d)
                gd, gs = greedy_design_rule(ta, tb, g, fl, d)
                vd, vs = bm.ai_index(tt.df_db), float(np.max(np.abs(np.diff(tt.psi_deg))))
                assert vd >= gd - 0.01 and vs <= 30 + 1e-6, f"{name}: Viterbi {vd:.2f} dB / {vs:.1f} deg"
                out.append(f"{floor_name} {name}: AI-DF {vd:.2f} vs {gd:.2f} dB, max step {vs:.0f} vs {gs:.0f} deg")
        check(f"d={d*1e3:.0f} mm Viterbi >= greedy AI-DF, steps <= 30 deg (implemented vs literal)", True,
              "; ".join(out))


# ------------------------------------------------------------------ DESIGN 1.2 numbers
def test_design_numbers():
    print("AI-DI against DESIGN 1.2 (tolerance 0.1 dB; DESIGN rounds to 0.1)")
    rows = []

    def close(name, got, want):
        rows.append(abs(got - want) <= 0.1)
        print(f"     {name}: {got:.2f} (DESIGN {want})")

    want = {(-10, "sinc"): (5.2, 5.0, 4.6), (-10, "j0"): (4.1, 3.9, 3.7), (-3, "sinc"): (4.1, 4.3, 4.3),
            (-3, "j0"): (3.4, 3.5, 3.4)}
    for (fl, model), vals in want.items():
        for d, v in zip(SPACINGS, vals):
            g, a = cm.diffuse_model(F, d, model), cm.steer(F, d / C)
            close(f"single on axis, WNG >= {fl}, {model}, {d*1e3:.0f} mm",
                  bm.ai_index(bm.df_db(bm.mvdr_weights(a, g, fl)[0], a, g)), v)
    for d, v3, v2 in zip(SPACINGS, (3.4, 3.6, 3.5), (2.2, 2.3, 2.3)):
        g3, g2 = cm.diffuse_model(F, d, "sinc"), cm.diffuse_model(F, d, "j0")
        tt = bm.two_talker(F, d / C, -d / C, g3, -10.0, d)
        di = [bm.ai_index(10 * np.log10(0.5 * (1 + np.abs(tt.h_B) ** 2) / bm.noise_power(tt.w, g))) for g in (g3, g2)]
        close(f"two-talker, WNG >= -10, 3-D, {d*1e3:.0f} mm", di[0], v3)
        close(f"two-talker, WNG >= -10, 3-D design in 2-D noise, {d*1e3:.0f} mm", di[1], v2)
    g = cm.diffuse_model(F, 0.04, "sinc")
    for th, v10, v3 in ((0, 5.0, 4.3), (30, 4.3, 3.8), (45, 3.6, 3.1), (60, 2.6, 2.3), (90, 1.0, 1.0)):
        a = cm.steer(F, tau_of(th, 0.04))
        close(f"4 cm steered {th} deg, WNG >= -10", bm.ai_index(bm.df_db(bm.mvdr_weights(a, g, -10)[0], a, g)), v10)
        close(f"4 cm steered {th} deg, WNG >= -3", bm.ai_index(bm.df_db(bm.mvdr_weights(a, g, -3)[0], a, g)), v3)
    for d, vax, vbr in zip(SPACINGS, (1.3, 1.9, 2.3), (0.5, 1.0, 1.5)):
        g = cm.diffuse_model(F, d, "sinc")
        for t, v, nm in ((d / C, vax, "axis"), (0.0, vbr, "broadside")):
            close(f"DAS {nm}, {d*1e3:.0f} mm", bm.ai_index(bm.df_db(bm.das_weights(F, t), cm.steer(F, t), g)), v)
    check("all DESIGN 1.2 rows reproduced", all(rows), f"{sum(rows)}/{len(rows)} within 0.1 dB")


# ------------------------------------------------------------------ B.7 phase continuity
def crossfade_dip_db(z1, z2):
    """Max over bins of (min over c of |(1-c) z1 + c z2|) below min(|z1|, |z2|), in dB (>= 0)."""
    c = np.linspace(0, 1, 201)[:, None]
    m = np.abs((1 - c) * z1[None] + c * z2[None]).min(axis=0)
    return float(np.max(20 * np.log10(np.minimum(np.abs(z1), np.abs(z2)) / m)))


def test_phase_continuity():
    print("B.7 phase continuity across states A / AB / B (bins below 0.8 c/2d, bus floors)")
    for d in SPACINGS:
        g = cm.diffuse_model(F, d, "sinc")
        k = F <= 0.8 * C / (2 * d)
        res = []
        for name, (da, db) in GEOMS.items():
            ta, tb = tau_of(da, d), tau_of(db, d)
            bw = bm.bus_weights(ta, tb, g, d, bm.wng_floor("bus"))
            a = (cm.steer(F, ta), cm.steer(F, tb))
            r = np.array([[bm.response(bw.w[s], a[p])[k] for p in (0, 1)] for s in range(3)])   # (state, partner, F)
            worst_ph, worst_dip = 0.0, 0.0
            for s1, s2 in ((0, 1), (1, 2), (0, 2)):
                for p in (0, 1):
                    z1, z2 = r[s1, p], r[s2, p]
                    strong = (np.abs(z1) >= 0.5) & (np.abs(z2) >= 0.5)
                    dph = np.abs(deg(z2 / z1))
                    assert np.all(dph[strong] <= 30 + 1e-6) and np.all(dph <= 90 + 1e-6), \
                        f"{name}: states {s1}/{s2} partner {p}: phase step {dph.max():.1f} deg"
                    worst_ph = max(worst_ph, dph[strong].max() if strong.any() else 0.0)
                    worst_dip = max(worst_dip, crossfade_dip_db(z1, z2))
            assert worst_dip <= 1.0, f"{name}: crossfade dip {worst_dip:.2f} dB"
            for s, p in ((0, 0), (1, 0), (2, 1)):                    # the targeted partner keeps the AB response
                want = np.ones(k.sum()) if p == 0 else bw.ab.h_B[k]
                assert np.allclose(r[s, p], want, atol=1e-9), f"{name}: state {s} target response"
            res.append(f"{name}: dip {worst_dip:.2f} dB, step {worst_ph:.1f} deg, "
                       f"subst {bw.subst[:, k].sum(1).tolist()}")
        check(f"d={d*1e3:.0f} mm: steps <= 30 deg (both >= -6 dB) / 90 deg, dips <= 1 dB, targets keep the AB response",
              True, "; ".join(res))
    fl = bm.wng_floor("bus")
    bw = bm.bus_weights(tau_of(0, 0.04), tau_of(180, 0.04), cm.diffuse_model(F, 0.04, "sinc"), 0.04, fl)
    wng = [bm.wng_db(bw.w[s], cm.steer(F, t)) for s, t in ((0, tau_of(0, 0.04)), (2, tau_of(180, 0.04)))]
    check("state weights (after substitution) meet the floor", min((w - fl).min() for w in wng) >= -1e-6,
          "A and B states, 4 cm endfire")


# ------------------------------------------------------------------ plausibility gate, AB-only, release
def fake_track(N, ta, tb, p_a, p_b, d):
    """S1 Track with given constant partner TDOAs and posteriors (the other fields are zeros)."""
    z = np.zeros(N)
    trk = tk.Track(tau=z, q=z, psi=z, cdr_db=z, level_db=z, vad=z, noise_db=z, d_hat=d, online=False, srp_mode="fit")
    trk.tau_A, trk.tau_B, trk.tau_A_t, trk.tau_B_t = ta, tb, np.full(N, ta), np.full(N, tb)
    trk.p_A, trk.p_B = p_a, p_b
    return trk


def test_gate_release():
    print("TDOA plausibility gate, AB-only states, steering release to the neutral beam")
    d = 0.04
    r = [bm.implausible_reasons(1.2 * d / C, -0.5 * d / C, d), bm.implausible_reasons(d / C, -d / C, d),
         bm.implausible_reasons(1.2 * d / C, 0.0, d, max_tdoa=None), bm.implausible_reasons(np.nan, np.nan, d),
         bm.implausible_reasons(0.5 * d / C, 0.2 * d / C, d), bm.implausible_reasons(0.5 * d / C, 0.2 * d / C, d,
                                                                                      same_side=True)]
    check("TDOA beyond 1.1 d_hat/c or both on one side -> fallback", r == [("beyond_endfire",), (), (), (),
                                                                         ("same_side",), ()], str(r))
    N = 200
    X = np.zeros((2, len(F), N), complex)
    pa = np.r_[np.zeros(50), np.ones(150)]
    b = bm.bus_beam(X, calib.identity(d), fake_track(N, 1.2 * d / C, -0.5 * d / C, pa, np.zeros(N), d))
    ok = b.diag["fallback_frac"] == 1.0 and np.allclose(b.bank[:, :, 0], 1) and np.allclose(b.bank[:, :, 1], 0)
    check("grid-edge partner TDOA -> Mic A alone", ok, f"{b.diag['configs'][0]['why']} -> "
          f"{b.diag['configs'][0]['fallback']}")
    trk = fake_track(N, 0.9 * d / C, -0.9 * d / C, pa, np.zeros(N), d)
    b, b1 = bm.bus_beam(X, calib.identity(d), trk, states="ab"), bm.bus_beam(X, calib.identity(d), trk, states="s1")
    check("states='ab' keeps the AB beam throughout", np.all(b.diag["states"] == 1) and np.any(b1.diag["states"] == 0)
          and b.diag["fallback_frac"] == 0, f"state fractions {b.diag['state_frac']} (s1: {b1.diag['state_frac']})")
    trk.tau_A_t = 0.9 * d / C + 1e-6 * (np.arange(N) // 4)         # online-like: 50 designs, 150 bank entries
    b = bm.bus_beam(X, calib.identity(d), trk)
    check("more than 42 designs (bank index beyond int8)", len(b.diag["configs"]) == 50 and b.idx.max() >= 128,
          f"{len(b.diag['configs'])} designs, bank of {len(b.bank)}")
    t = np.r_[np.full(5, np.nan), np.full(10, 5e-5), np.full(20, np.nan), np.full(10, -5e-5)]
    h, h0 = bm.hold_tdoa(t, 50, n_release=12), bm.hold_tdoa(t, 50)
    want = np.r_[np.full(5, np.nan), np.full(21, 5e-5), np.full(9, np.nan), np.full(10, -5e-5)]
    check("hold_tdoa: neutral after 12 frames without evidence, next talker at once",
          np.array_equal(h, want, equal_nan=True) and np.all(h0[5:] == 5e-5),
          "without release the 50-frame dwell keeps the first talker")
    s = bm.hold_switch(np.r_[-np.ones(3), np.zeros(5), -np.ones(4), np.ones(3)].astype(int), 10, n_release=3)
    check("hold_switch: release and immediate re-target", s.tolist() == [-1] * 3 + [0] * 7 + [-1] * 2 + [1] * 3,
          str(s.tolist()))
    # a coherent noise source at +100 us (idle frames); a talker at -50 us whose frames 1500-1599 read +100 us
    rng = np.random.default_rng(0)
    N = 2000
    vad = np.r_[np.zeros(1000), np.ones(1000)]
    tau = np.where(vad > 0, -50e-6, 100e-6) + rng.normal(0, 3e-6, N)
    tau[1500:1600] = 100e-6
    z = np.zeros(N)
    trk = tk.Track(tau=tau, q=np.full(N, 0.5), psi=z, cdr_db=z, level_db=z, vad=vad, noise_db=z, d_hat=d,
                   online=False, srp_mode="fit")
    (tn, dom), (tc, dc) = bm.noise_tdoa(trk), bm.noise_tdoa(trk, causal=True)
    e, e0 = bm.active_talker_tdoa(trk), bm.active_talker_tdoa(trk, noise_excl=None)
    ok = (abs(tn[0] - 100e-6) < 3e-6 and dom.all() and not dc[0] and dc[200]
          and np.nanmax(np.abs(e[1000:] + 50e-6)) < 10e-6 and abs(np.nanmedian(e0[1530:1570]) - 100e-6) < 3e-6)
    check("coherent-noise TDOA is no steering evidence", ok, f"noise TDOA {tn[0] * 1e6:+.0f} us (causal: none in the "
          f"first second); talker frames on it: evidence {np.nanmedian(e0[1530:1570]) * 1e6:+.0f} us without the "
          f"exclusion, {np.nanmax(e[1000:]) * 1e6:+.0f} us at most with it")


# ------------------------------------------------------------------ state machine, crossfades, Beam
def test_states():
    print("Bus state machine (DESIGN M2)")
    n = 400
    pa, pb = np.zeros(n), np.zeros(n)
    pa[20:120] = 0.9                       # A alone
    pb[150:160] = 0.5                      # B shows up briefly (uncertainty) ...
    pb[200:300] = 0.9                      # ... then talks alone
    pa[330:360], pb[330:360] = 0.5, 0.5    # double talk
    st = bm.bus_states(pa, pb, hold_s=0.1, enter=0.75, other=0.2)          # DESIGN values
    check("starts in AB, enters A after 100 ms", st[0] == 1 and st[20 + 11] == 1 and st[20 + 12] == 0,
          f"A from frame {np.argmax(st == 0)} (condition from 20, 13 frames = 104 ms)")
    check("holds A while nobody talks", np.all(st[120:150] == 0), "frames 120-149")
    check("AB as soon as the other posterior reaches 0.2", st[150] == 1 and np.all(st[160:200] == 1), "frame 150")
    check("enters B after 100 ms, AB on double talk", st[200 + 12] == 2 and st[330] == 1 and st[399] == 1,
          f"B from frame {200 + np.argmax(st[200:] == 2)}")
    pa[20:120], pb[200:300] = 0.95, 0.95
    st2 = bm.bus_states(pa, pb)                                             # tuned: 0.9 for 300 ms
    check("tuned entry: p > 0.9 held for 300 ms", st2[20 + 36] == 1 and st2[20 + 37] == 0 and st2[200 + 37] == 2,
          f"A from frame {np.argmax(st2 == 0)}, B from frame {200 + np.argmax(st2[200:] == 2)} (38 frames)")


def test_crossfade_beam():
    print("Crossfades, hold, Beam.apply / weights / df")
    idx, val = bm.crossfade(np.r_[np.zeros(10), np.ones(20)].astype(int), 6)
    c1 = np.where(idx == 1, val, 0).sum(axis=1)
    ramp = [0, 1 / 6, 2 / 6, 0.5, 4 / 6, 5 / 6, 1, 1]
    check("linear 6-frame crossfade", np.allclose(val.sum(1), 1) and np.allclose(c1[9:17], ramp),
          str(np.round(c1[8:17], 3)))
    idx, val = bm.crossfade(np.r_[np.zeros(10), np.ones(3), 2 * np.ones(20)].astype(int), 6)
    c2 = np.where(idx == 2, val, 0).sum(axis=1)
    check("retarget mid-ramp", np.allclose(val.sum(1), 1) and idx.shape[1] == 3 and c2[18] == 1.0,
          f"max {idx.shape[1]} entries, target 2 complete at frame {np.argmax(c2 == 1)}")
    held = bm.hold_switch(np.tile(np.r_[np.zeros(10), np.ones(10)], 10).astype(int), 30, init=0)
    sw = np.flatnonzero(np.diff(held)) + 1
    check("hold: switches at least 30 frames apart", np.all(np.diff(sw) >= 30), f"switches at {sw.tolist()}")
    rng = np.random.default_rng(0)
    N = 300
    bank = rng.standard_normal((4, len(F), 2)) + 1j * rng.standard_normal((4, len(F), 2))
    tgt = np.repeat(rng.integers(0, 4, 15), 20)
    idx, val = bm.crossfade(tgt, 6)
    beam = bm.Beam(bank, idx, val)
    X = rng.standard_normal((2, len(F), N)) + 1j * rng.standard_normal((2, len(F), N))
    Y = beam.apply(X, chunk=64)
    Yref = np.stack([sum(v * (np.conj(bank[i, :, 0]) * X[0, :, l] + np.conj(bank[i, :, 1]) * X[1, :, l])
                         for i, v in zip(idx[l], val[l])) for l in range(N)], axis=1)
    check("Beam.apply == per-frame W^H X", np.allclose(Y, Yref), f"max err {np.max(np.abs(Y - Yref)):.1e}")
    cal = calib.identity(0.04)
    W = beam.weights()
    df_ref = 1 / bm.noise_power(W, cal.gamma_n[:, None])
    check("Beam.df == 1 / W^H Gamma W", np.allclose(beam.df(cal, chunk=64), df_ref), "identity calibration")


# ------------------------------------------------------------------ signals through the STFT
def test_signals():
    print("Signals: plane wave through the designed weights, diffuse-noise gain vs design")
    rng = np.random.default_rng(1)
    d = 0.04
    # band-limited: frac_delay cannot delay the real-valued Nyquist bin
    s = sosfiltfilt(butter(8, [100, 7000], "band", fs=FS, output="sos"), rng.standard_normal(4 * FS) * 0.01)
    for t in (d / C, -0.6 * d / C):
        x = np.stack([s, frac_delay(s, t)])
        X = cm.stft(x)
        w, _ = bm.mvdr_weights(cm.steer(F, t), cm.diffuse_model(F, d, "sinc"), bm.wng_floor("bus"))
        Y = np.conj(w[:, 0, None]) * X[0] + np.conj(w[:, 1, None]) * X[1]
        y = cm.istft(Y, x.shape[1])
        ref = frac_delay(s, t / 2)                          # the plane wave at the array centre
        sl = slice(FS // 2, -FS // 2)
        err = 10 * np.log10(np.sum((y[sl] - ref[sl]) ** 2) / np.sum(ref[sl] ** 2))
        check(f"plane wave tau={t*1e6:+.0f} us -> centre signal", err < -40, f"error {err:.1f} dB")
    T = 60 * FS
    noise = coherent_noise(T, d, rng)
    X = cm.stft(noise)
    g = cm.diffuse_model(F, d, "sinc")
    k = (F >= 200) & (F <= 6000)
    tt = bm.two_talker(F, d / C, -d / C, g, bm.wng_floor("bus"), d)
    for name, w in (("MVDR to A", bm.mvdr_weights(cm.steer(F, d / C), g, bm.wng_floor("bus"))[0]), ("AB beam", tt.w)):
        Y = np.conj(w[:, 0, None]) * X[0] + np.conj(w[:, 1, None]) * X[1]
        meas = 10 * np.log10(np.mean(np.abs(Y) ** 2, axis=1) / np.mean(np.abs(X[0]) ** 2, axis=1))
        pred = 10 * np.log10(bm.noise_power(w, g))
        err = np.abs(meas - pred)[k]
        check(f"diffuse-noise output power vs w^H Gamma w ({name})", np.median(err) < 0.15 and err.max() < 0.6,
              f"median |err| {np.median(err):.3f} dB, max {err.max():.2f} dB; "
              f"mean reduction {-np.mean(meas[k]):.2f} dB")


# ------------------------------------------------------------------ end to end (synthetic scenes)
def burst(n, rng, level):
    return rng.standard_normal(n) * np.hanning(n) ** 0.25 * level


def partners_scene(d, tau_a, tau_b, dur, rng, snr_db=6.0):
    """Alternating plane-wave bursts of partners A and B in sinc-diffuse noise. Returns the mixture, the
    components {'A', 'B', 'N'} (2, T) and the per-frame talker (-1 none, 0 A, 1 B)."""
    T = int(dur * FS)
    comp = {"A": np.zeros((2, T)), "B": np.zeros((2, T)), "N": coherent_noise(T, d, rng) * 0.01}
    who = np.full(T, -1)
    t, k = int(0.5 * FS), 0
    while t < T - 2 * FS:
        n = int(rng.uniform(0.8, 1.6) * FS)
        s = burst(n, rng, 0.01 * 10 ** (snr_db / 20))
        key, tau = ("A", tau_a) if k == 0 else ("B", tau_b)
        comp[key][0, t:t + n] += s
        comp[key][1, t:t + n] += frac_delay(s, tau)
        who[t:t + n] = k
        t += n + int(rng.uniform(0.3, 0.8) * FS)
        k = 1 - k
    cen = cm.frame_centers(cm.n_frames(T))
    return sum(comp.values()), comp, np.where((cen >= 0) & (cen < T), who[np.clip(cen, 0, T - 1)], -1)


def lag_samples(y, ref, max_lag=8):
    """Lag of y against ref (positive: y later) from the |cross-correlation| peak (polarity-tolerant: the AB
    beam returns partner B with r_B = -1), parabolic refinement."""
    n = len(y)
    cc = np.fft.irfft(np.fft.rfft(y, 2 * n) * np.conj(np.fft.rfft(ref, 2 * n)), 2 * n)
    cc = np.abs(np.concatenate([cc[-max_lag:], cc[:max_lag + 1]]))
    i = int(np.argmax(cc))
    den = cc[i - 1] - 2 * cc[i] + cc[i + 1]
    return i - max_lag + (0.5 * (cc[i - 1] - cc[i + 1]) / den if 0 < i < len(cc) - 1 and den < 0 else 0.0)


def component_outputs(fe, beam, comp, scenario):
    """Pass each component through the front end's calibration and the mixture's beam (both linear)."""
    out = {}
    for key, c in comp.items():
        Xc = calib.apply(cm.stft(calib.prefilter(c, scenario)), fe.calib)
        out[key] = cm.istft(beam.apply(Xc), c.shape[1])
    return out


def turns_scene(d, taus, dur, rng, fan_tau=None):
    """Seats at TDOAs taus taking 2.5 s turns in sinc-diffuse noise (about +10 dB), plus a steady coherent noise
    source as loud as the diffuse noise at fan_tau if given (a projector fan). Returns the mixture, the
    components {'S0', 'S1', ..., 'N'} (2, T) and the turns [(start, end, seat)] in samples."""
    T = int(dur * FS)
    comp = {f"S{i}": np.zeros((2, T)) for i in range(len(taus))}
    comp["N"] = coherent_noise(T, d, rng) * 0.003
    if fan_tau is not None:
        s = rng.standard_normal(T) * 0.003
        comp["N"] += np.stack([s, frac_delay(s, fan_tau)])
    turns = []
    t, k = FS // 2, 0
    while t < T - 3 * FS:
        n = int(2.5 * FS)
        s = burst(n, rng, 0.01)
        seat = k % len(taus)
        comp[f"S{seat}"][0, t:t + n] += s
        comp[f"S{seat}"][1, t:t + n] += frac_delay(s, taus[seat])
        turns.append((t, t + n, seat))
        t += n + FS // 2
        k += 1
    return sum(comp.values()), comp, turns


def check_components(name, fe, beam, comp, scenario, taus, min_nr=1.0):
    """Each talker component through the mixture's calibration and beam: |lag| <= 2 samples against its raw
    Mic A signal (B.6, polarity-tolerant), level change within 3 dB in 300-3000 Hz (the bursts are white; the
    full-band change is printed too), noise reduced by more than min_nr dB."""
    out = component_outputs(fe, beam, comp, scenario)
    sos = butter(4, [300, 3000], "band", fs=FS, output="sos")
    band = {k: 10 * np.log10(np.sum(sosfiltfilt(sos, out[k]) ** 2) / np.sum(sosfiltfilt(sos, comp[k][0]) ** 2))
            for k in comp}
    full = {k: 10 * np.log10(np.sum(out[k] ** 2) / np.sum(comp[k][0] ** 2)) for k in comp}
    talkers = [k for k in comp if k != "N"]
    lags = {k: lag_samples(out[k], comp[k][0]) for k in talkers}
    ok = all(abs(v) <= 2 for v in lags.values()) and band["N"] < -min_nr and all(abs(band[k]) < 3 for k in talkers)
    check(f"{name}: lag, gains", ok, ", ".join(f"{k} lag {lags[k]:+.2f} (centre ref {t * FS / 2:+.2f}) gain "
                                              f"{band[k]:+.2f} ({full[k]:+.2f}) dB" for k, t in zip(talkers, taus))
          + f", noise {band['N']:+.2f} ({full['N']:+.2f}) dB [300-3000 Hz (full band)]; lags include the HPF lead")


def test_end_to_end():
    print("End to end: beams and wrappers on synthetic scenes (alignment, gains, determinism, RTF)")
    rng = np.random.default_rng(7)
    d = 0.04
    taus = (0.8 * d / C, -0.6 * d / C)
    x, comp, who = partners_scene(d, *taus, 30.0, rng)
    fe = tk.front_end(x, FS, BUS, partners=True, talkers=False)
    check_components("M2 bus (AB beam)", fe, bm.bus_beam(fe.X, fe.calib, fe.track), comp, "bus", taus)
    beam = bm.bus_beam(fe.X, fe.calib, fe.track, states="s1")
    check_components("M2 bus (states)", fe, beam, comp, "bus", taus)
    check_components("M1 steered (bus)", fe, bm.das_beam(fe.X, fe.calib, fe.track, "bus"), comp, "bus", taus)
    st = beam.diag["states"]
    fr = [[float(np.mean(st[who == p] == s)) for s in range(3)] for p in (0, 1)]
    print(f"     M2 bus states (s1) during A / B bursts (A, AB, B): {np.round(fr, 2).tolist()}")
    seats = (0.9 * d / C, 0.0, -0.7 * d / C)
    xm, compm, _ = turns_scene(d, seats, 30.0, rng)
    fem = tk.front_end(xm, FS, MEETING, talkers=False)
    # M1: at most 1.9 dB AI-DI on axis (DESIGN 1.2), Mic A alone in the 0.5 s gaps between turns
    for name, b, nr in (("M2 meeting steered", bm.meeting_beam(fem.X, fem.calib, fem.track), 1.0),
                        ("M2 meeting fixed3", bm.meeting_beam(fem.X, fem.calib, fem.track, mode="fixed3"), 1.0),
                        ("M1 steered (meeting)", bm.das_beam(fem.X, fem.calib, fem.track, "meeting"), 0.3)):
        check_components(name, fem, b, compm, "meeting", seats, nr)
    for mod, sig, info, kw in ((m2_bus_beam, x, BUS, {}), (m1_das, x, BUS, {}), (m1_das, x, BUS, {"mode": "broadside"}),
                               (m2_bus_beam, x, BUS, {"online": True}), (m1_das, x, BUS, {"online": True}),
                               (m2_meeting_beam, xm, MEETING, {}), (m2_meeting_beam, xm, MEETING, {"mode": "fixed3"}),
                               (m2_meeting_beam, xm, MEETING, {"online": True}), (m1_das, xm, MEETING, {}),
                               (m2_bus_beam, xm, MEETING, {}), (m2_meeting_beam, x, BUS, {})):
        tag = f"{mod.__name__.split('.')[-1]} {kw or ''} on a {info['scenario']} scene"
        t0 = time.perf_counter()
        y = mod.enhance(sig.astype(np.float32), FS, dict(info), **kw)
        rtf = (time.perf_counter() - t0) / 30.0
        y2 = mod.enhance(sig.astype(np.float32), FS, dict(info), **kw)
        st = mod.enhance(sig.astype(np.float32), FS, dict(info), return_stages=True, **kw)
        ok = (y.shape == (sig.shape[1],) and np.all(np.isfinite(y)) and np.array_equal(y, y2) and rtf < 0.5
              and np.array_equal(st["out"], y) and np.array_equal(st["pre_agc"], y))
        check(tag, ok, f"RTF {rtf:.3f}, deterministic, return_stages ok")
    y = m1_das.enhance(x, FS, BUS, mode="broadside")
    Xh = calib.apply(cm.stft(calib.prefilter(x, "bus")), fe.calib)
    ref = cm.istft(0.5 * (Xh[0] + Xh[1]), x.shape[1])
    check("M1a = (X_A + X_B) / 2 after S0", np.allclose(y, ref, atol=1e-12), "")


def test_fallback():
    print("Inseparable partners: DESIGN 2.1 fallback")
    g = cm.diffuse_model(F, 0.04, "sinc")
    bw = bm.bus_weights(5e-6, -3e-6, g, 0.04, bm.wng_floor("bus"), why=("both_broadside",))
    check("both broadside -> broadside MVDR (= average)", bw.fallback == "broadside" and np.allclose(bw.w, 0.5),
          bw.fallback)
    bw = bm.bus_weights(9e-5, 9e-5, g, 0.04, bm.wng_floor("bus"), why=("one_peak",))
    check("one peak -> Mic A alone", bw.fallback == "omni" and np.allclose(bw.w[:, :, 0], 1) and
          np.allclose(bw.w[:, :, 1], 0), bw.fallback)
    rng = np.random.default_rng(3)
    d = 0.04
    x, _, _ = partners_scene(d, 0.2 * d / C, -0.15 * d / C, 20.0, rng)
    fe = tk.front_end(x, FS, BUS, partners=True, talkers=False)
    beam = bm.bus_beam(fe.X, fe.calib, fe.track)
    ok = fe.track.inseparable and beam.diag["fallback_frac"] == 1.0 and len(beam.bank) == 3
    check("broadside pair end to end", ok, f"S1 {fe.track.inseparable_why}, fallback "
          f"{beam.diag['configs'][0]['fallback']}, tau {fe.track.tau_A*1e6:+.1f}/{fe.track.tau_B*1e6:+.1f} us")


def test_meeting_steering():
    print("Meeting steering: three seats taking turns (steered: with a projector fan at -0.4 d/c)")
    d = 0.04
    band = (F >= bm.SELECT_BAND[0]) & (F <= bm.SELECT_BAND[1])
    fan = -0.4 * d / C
    for mode, taus, fan_tau in (("steered", (0.9 * d / C, 0.1 * d / C, -0.8 * d / C), fan),
                                ("fixed3", (0.9 * d / C, 0.0, -0.7 * d / C), None)):
        x, _, turns = turns_scene(d, taus, 30.0, np.random.default_rng(4), fan_tau=fan_tau)
        fe = tk.front_end(x, FS, MEETING, talkers=False)
        beam = bm.meeting_beam(fe.X, fe.calib, fe.track, mode=mode)
        held = beam.diag["steer_id" if mode == "steered" else "select"]
        if mode == "steered":
            tn, dom = bm.noise_tdoa(fe.track)
            check("the fan is the dominant coherent noise source", dom[0] and abs(tn[0] - fan) <= 6e-6,
                  f"noise TDOA {tn[0] * 1e6:+.0f} us (fan {fan * 1e6:+.0f} us)")
        hits = []
        for a, b, seat in turns:
            l0, l1 = cm.frame_of_sample(a + FS), cm.frame_of_sample(b)       # after the first second of a turn
            if mode == "steered":
                tau_l = np.array([np.nan if beam.labels[i][1] is None else beam.labels[i][1]
                                  for i in beam.idx[l0:l1, 0]])
                hits.append(np.mean(np.abs(tau_l - taus[seat]) <= 10e-6))
            else:       # the beam giving this talker the most 300-3000 Hz power (seat 2 at 134 deg: broadside)
                resp = [np.mean(np.abs(bm.response(w, cm.steer(F, taus[seat]))[band]) ** 2) for w in beam.bank]
                hits.append(np.mean(held[l0:l1] == int(np.argmax(resp))))
        hold = bm.STEER_HOLD_S if mode == "steered" else bm.SELECT_HOLD_S
        dwell = talker_dwells(held, neutral=-1 if mode == "steered" else None)
        ok = min(hits) > 0.9 and np.all(np.array(dwell) * cm.HOP_S >= hold - 1e-9)
        n_sw = int(np.sum(np.diff(held) != 0))
        check(f"{mode}: beam on the talker after 1 s of each turn, holds >= {hold} s between talkers", ok,
              f"per-turn hit rate min {min(hits):.2f}, {n_sw} switches (neutral beam included) for {len(turns)} "
              f"turns, shortest talker-to-talker dwell {min(dwell, default=np.inf) * cm.HOP_S:.2f} s")


def talker_dwells(held, neutral=None):
    """Frames spent on a steering before each direct switch to another one (a switch from or to the neutral
    beam, id `neutral`, has no minimum dwell)."""
    out, start = [], 0
    for l in range(1, len(held)):
        if held[l] != held[l - 1]:
            if neutral is None or (held[l] != neutral and held[l - 1] != neutral):
                out.append(l - start)
            start = l
    return out


def test_speed():
    print("Cost of the designs")
    g = cm.diffuse_model(F, 0.04, "sinc")
    t0 = time.perf_counter()
    for _ in range(5):
        bm.bus_weights(tau_of(20, 0.04), tau_of(150, 0.04), g, 0.04, bm.wng_floor("bus"))
    t1 = time.perf_counter()
    for _ in range(5):
        [bm.mvdr_weights(cm.steer(F, t), g, -3.0) for t in bm.tdoa_grid(0.04)]
    t2 = time.perf_counter()
    check("design times", (t1 - t0) / 5 < 0.5, f"bus state set {(t1 - t0) / 5 * 1e3:.0f} ms, "
          f"64 meeting MVDRs {(t2 - t1) / 5 * 1e3:.0f} ms")


def main():
    quick = "--quick" in sys.argv
    tests = [test_floors_das, test_mvdr, test_two_talker_b1, test_two_talker_geometries, test_design_rule,
             test_design_numbers, test_phase_continuity, test_gate_release, test_states, test_crossfade_beam,
             test_signals, test_speed]
    if not quick:
        tests += [test_fallback, test_meeting_steering, test_end_to_end]
    t0 = time.perf_counter()
    for t in tests:
        t()
    print(f"all {len(tests)} test groups passed in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
