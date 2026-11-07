"""V2 adversarial verifier: independent recomputation of the spacing physics table (B1).

Written from scratch (did not copy B1 or DESIGN scripts). Two independent solvers for the
WNG-constrained single-target beam:
  (1) closed form in the 2-D space orthogonal to the steering vector (exact KKT optimum),
  (2) diagonal-loading MVDR with bisection on mu (DESIGN's recipe),
and they must agree. Ideal free-field omni mics, c = 343 m/s, fs = 16 kHz.

Run: OMP_NUM_THREADS=1 .venv/bin/python -I -B hw/spacing/v2/v2_physics.py <speech_lab path>
"""
import sys, json, os
import numpy as np
from scipy.special import j0
from scipy.optimize import brentq

C = 343.0
FS = 16000.0
# ANSI S3.5-1997 Table 3, one-third-octave band procedure, band-importance function (average speech)
FC = np.array([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
               5000, 6300, 8000.0])
BIF = np.array([0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818, 0.0844,
                0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185])
assert abs(BIF.sum() - 1) < 1e-9
FE = FC.copy(); FE[-1] = 7900.0          # B1/DESIGN evaluate the 8 kHz band at 7.9 kHz (fs/2 = 8 kHz)


def gamma(f, d, model):
    kd = 2 * np.pi * np.asarray(f, float) * d / C
    return np.sinc(kd / np.pi) if model == "3d" else j0(kd)   # np.sinc(x) = sin(pi x)/(pi x)


def steer(f, tau):
    """centre-referenced a(f, tau) = [e^{+j pi f tau}, e^{-j pi f tau}], tau in seconds (shape f x 2)."""
    ph = np.pi * np.asarray(f, float) * tau
    return np.stack([np.exp(1j * ph), np.exp(-1j * ph)], -1)


def G2(g):
    g = np.asarray(g, float)
    return np.stack([np.stack([np.ones_like(g), g], -1), np.stack([g, np.ones_like(g)], -1)], -2)


def quad(w, G):   # w^H G w for stacks
    return np.real(np.einsum("...i,...ij,...j->...", w.conj(), G, w))


# ---------------- solver 1: exact closed form of  min w^H G w  s.t. w^H a = 1, w^H w <= 1/W
def sd_closed(f, d, tau, W_lin, model):
    a = steer(f, tau)
    g = gamma(f, d, model)
    G = G2(g)
    b = np.stack([a[..., 0], -a[..., 1]], -1) / np.sqrt(2)          # unit vector orthogonal to a
    e = np.einsum("...i,...ij,...j->...", a.conj(), G, b)           # a^H G b
    B0 = quad(b, G)
    R = np.sqrt(np.maximum(1.0 / W_lin - 0.5, 0.0)) if W_lin is not None else np.inf
    r = np.minimum(np.abs(e) / (2 * B0), R)
    z = r * np.exp(1j * (np.pi - np.angle(e)))
    w = a / 2 + z[..., None] * b
    return w, a


# ---------------- solver 2: diagonal-loading MVDR, smallest mu meeting the floor (bisection)
def sd_mvdr(f, d, tau, W_lin, model, lo=-12.0, hi=8.0, it=80):
    a = steer(f, tau)
    G = G2(gamma(f, d, model))
    I = np.eye(2)

    def w_of(mu):
        M = G + mu[..., None, None] * I
        x = np.linalg.solve(M, a[..., None])[..., 0]
        return x / np.einsum("...i,...i->...", a.conj(), x)[..., None]

    def wng(w):
        return 1.0 / np.real(np.einsum("...i,...i->...", w.conj(), w))

    n = a.shape[0]
    l = np.full(n, lo); h = np.full(n, hi)
    ok_lo = wng(w_of(10 ** l)) >= W_lin
    for _ in range(it):
        m = 0.5 * (l + h)
        ok = wng(w_of(10 ** m)) >= W_lin
        h = np.where(ok, m, h); l = np.where(ok, l, m)
    mu = np.where(ok_lo, 10 ** lo, 10 ** h)
    return w_of(mu), a


def di_db(w, a, G):
    return 10 * np.log10(np.abs(np.einsum("...i,...i->...", w.conj(), a)) ** 2 / quad(w, G))


def wng_db(w, a):
    return 10 * np.log10(np.abs(np.einsum("...i,...i->...", w.conj(), a)) ** 2
                         / np.real(np.einsum("...i,...i->...", w.conj(), w)))


def aidi(di):
    return float(np.sum(BIF * di))


def single_target(d, theta_deg, floor_db, design, evalm, solver=sd_closed, f=FE):
    tau = d * np.cos(np.radians(theta_deg)) / C
    W = None if floor_db is None else 10 ** (floor_db / 10)
    w, a = solver(f, d, tau, W, design)
    return di_db(w, a, G2(gamma(f, d, evalm))), wng_db(w, a)


# ---------------- two-talker soft LCMV (DESIGN M2), strict floor, +-1 dB balance, psi grid 2.5 deg
def two_talker(d, floor_db, sel_model, f=FE, lam=1e3, design="3d"):
    W = 10 ** (floor_db / 10)
    psis = np.radians(np.arange(0, 360, 2.5))
    aA = steer(f, d / C); aB = steer(f, -d / C)
    Gd = G2(gamma(f, d, design)); Gs = G2(gamma(f, d, sel_model))
    out = np.zeros(len(f)); best_psi = np.full(len(f), np.nan)
    A = np.stack([aA, aB], -1)                                    # f x 2 x 2 (columns a_A, a_B)
    AAH = A @ np.conj(np.swapaxes(A, -1, -2))
    I = np.eye(2)
    for i in range(len(f)):
        h = np.stack([np.ones_like(psis), np.exp(1j * psis)], -1).astype(complex)   # P x 2
        rhs = lam * (A[i] @ h.conj().T).T                                            # P x 2

        def w_of(mu):
            M = Gd[i][None] + mu[:, None, None] * I + lam * AAH[i][None]
            return np.linalg.solve(M, rhs[..., None])[..., 0]

        def wng_mean(w):
            rA = np.einsum("pi,i->p", w.conj(), aA[i]); rB = np.einsum("pi,i->p", w.conj(), aB[i])
            return 0.5 * (abs(rA) ** 2 + abs(rB) ** 2) / np.real(np.einsum("pi,pi->p", w.conj(), w))

        l = np.full(len(psis), -9.0); hgh = np.full(len(psis), 6.0)
        ok_lo = wng_mean(w_of(10 ** l)) >= W
        for _ in range(50):
            m = 0.5 * (l + hgh)
            ok = wng_mean(w_of(10 ** m)) >= W
            hgh = np.where(ok, m, hgh); l = np.where(ok, l, m)
        mu = np.where(ok_lo, 10 ** -9.0, 10 ** hgh)
        w = w_of(mu)
        rA = np.einsum("pi,i->p", w.conj(), aA[i]); rB = np.einsum("pi,i->p", w.conj(), aB[i])
        wn = wng_mean(w)
        bal = np.abs(20 * np.log10(np.abs(rA) / np.abs(rB)))
        df = 0.5 * (abs(rA) ** 2 + abs(rB) ** 2) / np.real(np.einsum("pi,ij,pj->p", w.conj(), Gs[i], w))
        ok = (wn >= W * (1 - 1e-9)) & (bal <= 1.0)
        if ok.any():
            k = np.argmax(np.where(ok, df, -np.inf))
            out[i] = 10 * np.log10(df[k]); best_psi[i] = np.degrees(psis[k])
        else:
            out[i] = 0.0
    return out, best_psi


def two_talker_eval(d, floor_db, f=FE):
    """3-D figure: psi chosen by 3-D DF, evaluated 3-D. 2-D figure: 3-D-designed weights, psi chosen by J0 DF."""
    d3, p3 = two_talker(d, floor_db, "3d", f)
    d2, p2 = two_talker(d, floor_db, "2d", f)
    return aidi(d3), aidi(d2), p3, p2


# ---------------- band edges
def dipole_edge(d, floor_db):
    # WNG of centre-referenced h=[1,-1]: 1 - cos(2 pi f d / c) = 10^(floor/10)
    return np.arccos(1 - 10 ** (floor_db / 10)) * C / (2 * np.pi * d)


def cdr_edge(d, tau, model="3d", thr=0.2):
    fgrid = np.arange(1.0, 8000.0, 0.25)
    dist = np.abs(np.exp(1j * 2 * np.pi * fgrid * tau) - gamma(fgrid, d, model))
    idx = np.nonzero(dist >= thr)[0]
    return float(fgrid[idx[0]]) if len(idx) else float("nan")


def coh_edge(d, level, model="3d"):
    return brentq(lambda f: gamma(f, d, model) - level, 1.0, 0.6 * C / d)


# ---------------- TDOA groups for meeting seats
def tdoa_groups(seats, dev, d, res_s, n_az=720):
    seats = np.asarray(seats); dev = np.asarray(dev)
    counts = []; close = []
    for az in np.arange(n_az) * 2 * np.pi / n_az:
        u = np.array([np.cos(az), np.sin(az), 0.0])
        A = dev + u * d / 2; B = dev - u * d / 2
        tau = (np.linalg.norm(seats - B, axis=1) - np.linalg.norm(seats - A, axis=1)) / C
        t = np.sort(tau)
        n = 1; start = t[0]
        for x in t[1:]:
            if x - start > res_s:
                n += 1; start = x
        counts.append(n)
        close.append(int(np.sum(np.diff(t) < 10e-6)))
    counts = np.array(counts)
    return int(np.median(counts)), int(counts.min()), int(counts.max()), float(np.mean(counts)), \
        (int(min(close)), int(max(close)))


def main(lab):
    res = {}
    ds = [0.010, 0.012, 0.015, 0.020, 0.025, 0.04, 0.06]

    # ---- cross-check the two single-target solvers
    maxdiff = 0.0
    for d in ds:
        for th in (0, 30, 45, 60, 90):
            for fl in (-3, -6, -10, -20):
                a1, w1 = single_target(d, th, fl, "3d", "3d", sd_closed)
                a2, w2 = single_target(d, th, fl, "3d", "3d", sd_mvdr)
                maxdiff = max(maxdiff, np.max(np.abs(a1 - a2)))
                assert np.all(w1 >= fl - 1e-6) and np.all(w2 >= fl - 1e-6), (d, th, fl)
    res["solver_maxdiff_db"] = maxdiff

    tab = {}
    for d in ds:
        r = {}
        r["alias_khz"] = C / (2 * d) / 1e3
        r["max_tdoa_us"] = d / C * 1e6
        r["max_tdoa_smp"] = d / C * FS
        r["coh09_3d_khz"] = coh_edge(d, 0.9, "3d") / 1e3
        r["coh09_2d_khz"] = coh_edge(d, 0.9, "2d") / 1e3
        # single target, on axis
        for fl in (-3, -6, -10, -20):
            di3, wn3 = single_target(d, 0, fl, "3d", "3d")
            di2, _ = single_target(d, 0, fl, "2d", "2d")
            r[f"sd_axis_3d_{fl}"] = aidi(di3)
            r[f"sd_axis_2d_{fl}"] = aidi(di2)
            r[f"sd_axis_minwng_{fl}"] = float(wn3.min())
        di_unc, wn_unc = single_target(d, 0, None, "3d", "3d")
        r["sd_axis_3d_unconstrained"] = aidi(di_unc)
        r["sd_axis_perband_-10"] = [round(x, 2) for x in single_target(d, 0, -10, "3d", "3d")[0]]
        # broadside, DAS
        r["sd_broad_3d_-10"] = aidi(single_target(d, 90, -10, "3d", "3d")[0])
        r["sd_broad_2d_-10"] = aidi(single_target(d, 90, -10, "3d", "2d")[0])
        das = lambda th, m: aidi(di_db(steer(FE, d * np.cos(np.radians(th)) / C) / 2,
                                       steer(FE, d * np.cos(np.radians(th)) / C), G2(gamma(FE, d, m))))
        r["das_axis_3d"] = das(0, "3d"); r["das_broad_3d"] = das(90, "3d")
        # azimuth average (19 angles 0..90 step 5, plain mean of dB) and a midpoint-rule variant
        angs = np.arange(0, 91, 5)
        r["azavg_3d_-10"] = float(np.mean([aidi(single_target(d, t, -10, "3d", "3d")[0]) for t in angs]))
        r["azavg_2d_-10"] = float(np.mean([aidi(single_target(d, t, -10, "3d", "2d")[0]) for t in angs]))
        mid = np.arange(0.5, 90, 1.0)
        r["azavg_2d_-10_midpoint1deg"] = float(np.mean([aidi(single_target(d, t, -10, "3d", "2d")[0]) for t in mid]))
        # dipole and CDR edges
        r["dipole_edge_-10"] = dipole_edge(d, -10)
        r["dipole_edge_-6"] = dipole_edge(d, -6)
        r["dipole_edge_-20"] = dipole_edge(d, -20)
        r["cdr_endfire_3d"] = cdr_edge(d, d / C, "3d")
        r["cdr_broad_3d"] = cdr_edge(d, 0.0, "3d")
        r["cdr_endfire_2d"] = cdr_edge(d, d / C, "2d")
        r["cdr_broad_2d"] = cdr_edge(d, 0.0, "2d")
        tab[f"{d*100:g}"] = r
    res["table"] = tab

    # ---- two-talker beam (slow-ish part; only the spacings that matter)
    tt = {}
    for d in (0.012, 0.025, 0.04, 0.06):
        for fl in (-10, -20):
            a3, a2, p3, p2 = two_talker_eval(d, fl)
            tt[f"{d*100:g}cm_{fl}"] = dict(aidi_3d=a3, aidi_2d=a2,
                                          psi3=[None if np.isnan(x) else float(x) for x in p3])
    res["two_talker"] = tt

    # ---- TDOA groups
    tg = {}
    for scene in ("meeting_s0", "meeting_s1", "meeting_s2", "meeting_s10", "meeting_s11", "meeting_s12"):
        cand = [x for x in os.listdir(os.path.join(lab, "scenes")) if x.startswith(scene + "_d") and x.endswith(".npz")]
        z = np.load(os.path.join(lab, "scenes", sorted(cand)[0]), allow_pickle=False)
        m = json.loads(str(z["meta"]))
        for d in (0.010, 0.012, 0.015, 0.020, 0.025, 0.04, 0.06):
            g20 = tdoa_groups(m["seats"], m["device"], d, 20e-6)
            g10 = tdoa_groups(m["seats"], m["device"], d, 10e-6)
            tg[f"{scene}_{d*100:g}cm"] = dict(g20_med=g20[0], g20_min=g20[1], g20_max=g20[2], g20_mean=g20[3],
                                              g10_med=g10[0], g10_min=g10[1], g10_max=g10[2],
                                              close10us_minmax=g20[4])
    res["tdoa_groups"] = tg
    return res


if __name__ == "__main__":
    lab = sys.argv[1]
    out = main(lab)
    p = os.path.join(lab, "hw", "spacing", "v2", "v2_physics.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1, default=float)[:6000])
    for k, v in out["table"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "sd_axis_perband_-10"})
