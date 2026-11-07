"""V2 extra adversarial checks (own code): meeting/bus floors, band-averaging, mismatch (analytic
expected-power model), skew, scene-azimuth TDOA gaps, WNG-binding band, two-talker under the bus schedule.

Run: OMP_NUM_THREADS=1 .venv/bin/python -I -B hw/spacing/v2/v2_extra.py <speech_lab path>
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_physics import (C, FS, FC, FE, BIF, gamma, steer, G2, quad, sd_closed, di_db, wng_db, aidi,
                        single_target, tdoa_groups)
from scipy.special import j0
from scipy.optimize import brentq

LAB = sys.argv[1]
OUT = {}
BUSF = np.where(FC < 500, -15.0, np.where(FC <= 2000, -10.0, -6.0))


def st_floor(d, th, floor, design, evalm, f=FE):
    """single target with a per-band floor array (closed form)."""
    tau = d * np.cos(np.radians(th)) / C
    W = 10 ** (np.asarray(floor, float) / 10)
    w, a = sd_closed(f, d, tau, W, design)
    return di_db(w, a, G2(gamma(f, d, evalm))), wng_db(w, a), w, a


# ---------- (1) meeting floors and bus schedule, single target
rows = {}
for d in (0.010, 0.012, 0.015, 0.020, 0.025, 0.04, 0.06):
    r = {}
    for nm, fl in (("-3", -3.0), ("-6", -6.0), ("-10", -10.0), ("bus", BUSF)):
        flb = np.broadcast_to(np.asarray(fl, float), FC.shape)
        r[f"axis3d_{nm}"] = aidi(st_floor(d, 0, flb, "3d", "3d")[0])
        r[f"axis2d_{nm}"] = aidi(st_floor(d, 0, flb, "2d", "2d")[0])
        r[f"broad3d_{nm}"] = aidi(st_floor(d, 90, flb, "3d", "3d")[0])
        r[f"azavg2d_{nm}"] = float(np.mean([aidi(st_floor(d, t, flb, "3d", "2d")[0]) for t in range(0, 91, 5)]))
        r[f"azavg3d_{nm}"] = float(np.mean([aidi(st_floor(d, t, flb, "3d", "3d")[0]) for t in range(0, 91, 5)]))
    rows[f"{d*100:g}"] = r
OUT["single_target_floors"] = rows

# ---------- (2) highest frequency where the -10 dB floor binds (on axis, 3-D) and AI weight below it
fg = np.arange(50.0, 8000.0, 5.0)
bind = {}
for d in (0.012, 0.04):
    for fl in (-3.0, -10.0):
        w, a = sd_closed(fg, d, d / C, None, "3d")
        wn = wng_db(w, a)
        bound = fg[wn < fl]
        top = float(bound.max()) if len(bound) else 0.0
        bind[f"{d*100:g}cm_{fl:g}"] = dict(floor_binds_up_to_hz=top,
                                          ai_weight_of_bands_below=float(BIF[FC <= top].sum()))
OUT["floor_binding"] = bind

# ---------- (3) band-averaged AI-DI (DI averaged over each 1/3-octave band on a 10 Hz grid, in dB and in linear DF)
def band_avg_aidi(d, th, floor, design, evalm):
    lo = FC * 2 ** (-1 / 6); hi = np.minimum(FC * 2 ** (1 / 6), 7990.0)
    di_db_b, di_lin_b = [], []
    for l, h in zip(lo, hi):
        f = np.arange(l, h, 10.0)
        di, _, _, _ = st_floor(d, th, np.full(f.shape, floor), design, evalm, f=f)
        di_db_b.append(np.mean(di)); di_lin_b.append(10 * np.log10(np.mean(10 ** (di / 10))))
    return float(np.sum(BIF * np.array(di_db_b))), float(np.sum(BIF * np.array(di_lin_b)))

bavg = {}
for d in (0.012, 0.025, 0.04, 0.06):
    centre = aidi(st_floor(d, 0, np.full(FC.shape, -10.0), "3d", "3d")[0])
    bdb, blin = band_avg_aidi(d, 0, -10.0, "3d", "3d")
    bavg[f"{d*100:g}"] = dict(centre=centre, band_avg_db=bdb, band_avg_lin=blin)
OUT["band_averaging_axis3d_-10"] = bavg

# ---------- (4) mismatch, analytic expected-power model (per-mic gain U(+-g dB), phase U(+-p deg), independent)
def mm_moments(gdb, pdeg, n=200001):
    u = np.linspace(-1, 1, n)
    g = 10 ** (gdb * u / 20); ph = np.radians(pdeg * u)
    m = np.mean(g) * np.mean(np.exp(1j * ph))
    s2 = np.mean(g ** 2)
    return m, s2

def aidi_mismatch(d, th, floor, design, evalm, gdb, pdeg):
    m, s2 = mm_moments(gdb, pdeg)
    tau = d * np.cos(np.radians(th)) / C
    W = 10 ** (np.broadcast_to(np.asarray(floor, float), FE.shape) / 10)
    w, a = sd_closed(FE, d, tau, W, design)
    Ge = G2(gamma(FE, d, evalm))
    ww = np.real(np.einsum("fi,fi->f", w.conj(), w))
    r2 = np.abs(np.einsum("fi,fi->f", w.conj(), a)) ** 2
    num = abs(m) ** 2 * r2 + (s2 - abs(m) ** 2) * ww          # E|w^H (eps o a)|^2, |a_i| = 1
    den = abs(m) ** 2 * quad(w, Ge) + (s2 - abs(m) ** 2) * ww  # E[w^H (eps eps^H o G) w]
    return aidi(10 * np.log10(num / den)), aidi(10 * np.log10(r2 / quad(w, Ge)))

mm = {}
for d in (0.012, 0.04):
    e = {}
    for nm, fl in (("-10", -10.0), ("-20", -20.0), ("bus", BUSF), ("-3", -3.0)):
        a_u, a_i = aidi_mismatch(d, 0, fl, "3d", "3d", 1.0, 3.0)
        a_c, _ = aidi_mismatch(d, 0, fl, "3d", "3d", 0.25, 1.0)
        e[f"axis3d_{nm}"] = dict(ideal=a_i, uncal=a_u, cal=a_c, loss_uncal=a_i - a_u, loss_cal=a_i - a_c)
    az_i = np.mean([aidi_mismatch(d, t, -10.0, "3d", "2d", 1.0, 3.0)[1] for t in range(0, 91, 5)])
    az_u = np.mean([aidi_mismatch(d, t, -10.0, "3d", "2d", 1.0, 3.0)[0] for t in range(0, 91, 5)])
    e["azavg2d_-10"] = dict(ideal=float(az_i), uncal=float(az_u), loss=float(az_i - az_u))
    mm[f"{d*100:g}"] = e
OUT["mismatch_expected_power"] = mm

# ---------- (5) skew: beam designed for the geometric endfire (theta = 0, 3-D, -10 dB floor) applied to data whose
# channel B carries an extra delay delta (X_B,meas = X_B e^{-j 2 pi f delta}); hardware reading: delta = -68.75 us
def skew_aidi(d, delta, floor=-10.0):
    W = 10 ** (floor / 10)
    w, a = sd_closed(FE, d, d / C, np.full(FE.shape, W), "3d")
    D = np.exp(-2j * np.pi * FE * delta)
    am = a.copy(); am[:, 1] = am[:, 1] * D
    g = gamma(FE, d, "3d")
    Gm = np.stack([np.stack([np.ones_like(g), g * np.conj(D)], -1), np.stack([g * D, np.ones_like(g)], -1)], -2)
    return aidi(10 * np.log10(np.abs(np.einsum("fi,fi->f", w.conj(), am)) ** 2 / quad(w, Gm)))

sk = {}
for d in (0.012, 0.04):
    e = {"ideal": skew_aidi(d, 0.0)}
    for us in (2, 5, 10, 68.75):
        e[f"{us:g}us_pos"] = skew_aidi(d, us * 1e-6); e[f"{us:g}us_neg"] = skew_aidi(d, -us * 1e-6)
    sk[f"{d*100:g}"] = e
OUT["skew_axis3d_-10"] = sk

# ---------- (6) two-talker beam with a per-band floor (bus schedule) and uniform floors, own soft-LCMV
def two_talker_pb(d, floor_vec, sel_model, f=FE, lam=1e3):
    psis = np.radians(np.arange(0, 360, 2.5))
    aA = steer(f, d / C); aB = steer(f, -d / C)
    Gd = G2(gamma(f, d, "3d")); Gs = G2(gamma(f, d, sel_model))
    A = np.stack([aA, aB], -1); AAH = A @ np.conj(np.swapaxes(A, -1, -2))
    out = np.zeros(len(f)); I = np.eye(2)
    for i in range(len(f)):
        W = 10 ** (floor_vec[i] / 10)
        h = np.stack([np.ones_like(psis), np.exp(1j * psis)], -1).astype(complex)
        rhs = lam * (A[i] @ h.conj().T).T
        w_of = lambda mu: np.linalg.solve(Gd[i][None] + mu[:, None, None] * I + lam * AAH[i][None], rhs[..., None])[..., 0]
        def wngA(w):
            return np.abs(np.einsum("pi,i->p", w.conj(), aA[i])) ** 2 / np.real(np.einsum("pi,pi->p", w.conj(), w))
        l = np.full(len(psis), -9.0); hg = np.full(len(psis), 6.0)
        ok_lo = wngA(w_of(10 ** l)) >= W
        for _ in range(50):
            m = 0.5 * (l + hg); ok = wngA(w_of(10 ** m)) >= W
            hg = np.where(ok, m, hg); l = np.where(ok, l, m)
        w = w_of(np.where(ok_lo, 1e-9, 10 ** hg))
        rA = np.einsum("pi,i->p", w.conj(), aA[i]); rB = np.einsum("pi,i->p", w.conj(), aB[i])
        ok = (wngA(w) >= W * (1 - 1e-9)) & (np.abs(20 * np.log10(np.abs(rA) / np.abs(rB))) <= 1.0)
        df = 0.5 * (abs(rA) ** 2 + abs(rB) ** 2) / np.real(np.einsum("pi,ij,pj->p", w.conj(), Gs[i], w))
        out[i] = 10 * np.log10(df[np.argmax(np.where(ok, df, -np.inf))]) if ok.any() else 0.0
    return out

tt = {}
for d in (0.010, 0.012, 0.015, 0.020, 0.04):
    e = {}
    for nm, fl in (("bus", BUSF), ("-10", np.full(FC.shape, -10.0)), ("-6", np.full(FC.shape, -6.0))):
        e[f"{nm}_3d"] = aidi(two_talker_pb(d, fl, "3d")); e[f"{nm}_2d"] = aidi(two_talker_pb(d, fl, "2d"))
    tt[f"{d*100:g}"] = e
OUT["two_talker_floors"] = tt

# ---------- (7) TDOA: neighbouring-seat gaps < 10 us at the scenes' own azimuths, 4 and 1.2 cm
gaps = {}
for scene in ("meeting_s0", "meeting_s1", "meeting_s2", "meeting_s10", "meeting_s11"):
    z = np.load(os.path.join(LAB, "scenes", scene + "_d40mm.npz"), allow_pickle=False)
    m = json.loads(str(z["meta"]))
    seats = np.array(m["seats"]); dev = np.array(m["device"]); az = np.radians(m["array_azimuth_deg"])
    e = {}
    for d in (0.04, 0.012):
        u = np.array([np.cos(az), np.sin(az), 0.0]) * d / 2
        t = np.sort((np.linalg.norm(seats - (dev - u), axis=1) - np.linalg.norm(seats - (dev + u), axis=1)) / C)
        e[f"{d*100:g}cm_gaps_lt10us"] = int(np.sum(np.diff(t) < 10e-6))
        e[f"{d*100:g}cm_span_us"] = float((t[-1] - t[0]) * 1e6)
    # check the scene's stored mic positions agree with place_array convention
    mA = np.array(m["mics"][0]); mB = np.array(m["mics"][1])
    e["micA_minus_dev_dot_u"] = float(np.dot(mA - dev, [np.cos(az), np.sin(az), 0]))
    gaps[scene] = e
OUT["scene_azimuth_gaps"] = gaps

# ---------- (8) broadside DAS DI reaches 0.5 / 1 dB (3-D) at 1.2 cm
OUT["das_broadside_reach_1.2cm_hz"] = {str(t): brentq(lambda f: 10 * np.log10(2 / (1 + gamma(f, 0.012, "3d"))) - t, 100, 8000)
                                       for t in (0.5, 1.0)}

p = os.path.join(LAB, "hw", "spacing", "v2", "v2_extra.json")
with open(p, "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
print(json.dumps(OUT, indent=1, default=lambda x: round(float(x), 3)))
