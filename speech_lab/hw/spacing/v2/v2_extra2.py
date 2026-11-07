"""V2: remaining validation rows at 2.5/4/6 cm (own code) + low-frequency per-band claims at 1.2 cm."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_physics import C, FE, FC, BIF, gamma, steer, G2, quad, sd_closed, di_db, wng_db, aidi
from v2_extra import st_floor, aidi_mismatch  # noqa: E402  (re-runs v2_extra main body; cheap)

LAB = sys.argv[1]
O = {}
d4 = 0.04
fb = np.array([150, 250, 500, 1000, 2000, 3000, 4000, 6000.0])
O["perband_4cm_-10"] = st_floor(d4, 0, np.full(fb.shape, -10.0), "3d", "3d", f=fb)[0].round(2).tolist()
w, a = sd_closed(np.array([250, 500, 1000.0]), d4, d4 / C, None, "3d")
O["unconstrained_wng_4cm"] = wng_db(w, a).round(2).tolist()
fbs = np.array([500, 1000, 2000, 3000, 4000.0])
O["broadside_4cm"] = st_floor(d4, 90, np.full(fbs.shape, -10.0), "3d", "3d", f=fbs)[0].round(3).tolist()
O["offaxis_4cm"] = {fl: [round(aidi(st_floor(d4, t, np.full(FE.shape, float(fl)), "3d", "3d")[0]), 2)
                         for t in (0, 30, 45, 60, 90)] for fl in (-10, -3)}


# Mic-A-referenced convention check: a_m(f, tau) = [1, e^{-j 2 pi f tau}], h = [1, -1], soft LCMV, strict floor
def micA_two_talker(d, floor_db, lam=1e3):
    W = 10 ** (floor_db / 10)
    am = lambda f, tau: np.stack([np.ones_like(f), np.exp(-2j * np.pi * f * tau)], -1)
    aA = am(FE, d / C); aB = am(FE, -d / C)
    G = G2(gamma(FE, d, "3d"))
    A = np.stack([aA, aB], -1); AAH = A @ np.conj(np.swapaxes(A, -1, -2))
    h = np.array([1.0, -1.0]).astype(complex)
    out = []
    for i in range(len(FE)):
        rhs = lam * (A[i] @ h.conj())
        w_of = lambda mu: np.linalg.solve(G[i] + mu * np.eye(2) + lam * AAH[i], rhs)
        wng = lambda w: abs(np.vdot(w, aA[i])) ** 2 / np.real(np.vdot(w, w))
        if wng(w_of(1e-9)) >= W:
            w = w_of(1e-9)
        else:
            lo, hi = -9.0, 6.0
            for _ in range(60):
                m = 0.5 * (lo + hi)
                lo, hi = (lo, m) if wng(w_of(10 ** m)) >= W else (m, hi)
            w = w_of(10 ** hi)
        rA, rB = np.vdot(w, aA[i]), np.vdot(w, aB[i])
        ok = wng(w) >= W * (1 - 1e-3) and abs(20 * np.log10(abs(rA) / abs(rB))) <= 1.0
        out.append(10 * np.log10(0.5 * (abs(rA) ** 2 + abs(rB) ** 2) / quad(w, G[i])) if ok else 0.0)
    return aidi(np.array(out))

O["micA_ref_two_talker_-10"] = [round(micA_two_talker(d, -10.0), 2) for d in (0.025, 0.04, 0.06)]
O["mismatch_azavg2d_-10_loss"] = {}
for d in (0.025, 0.04, 0.06, 0.012):
    i_ = np.mean([aidi_mismatch(d, t, -10.0, "3d", "2d", 1.0, 3.0)[1] for t in range(0, 91, 5)])
    u_ = np.mean([aidi_mismatch(d, t, -10.0, "3d", "2d", 1.0, 3.0)[0] for t in range(0, 91, 5)])
    O["mismatch_azavg2d_-10_loss"][f"{d*100:g}"] = round(float(i_ - u_), 3)

# 1.2 cm low-frequency claims (B1 notes)
fl4 = np.array([125, 250, 500, 1000.0]); d = 0.012
O["lf_1.2cm_DI_at_floor"] = {fl: st_floor(d, 0, np.full(fl4.shape, float(fl)), "3d", "3d", f=fl4)[0].round(2).tolist()
                            for fl in (-6, -10, -20)}
w, a = sd_closed(fl4, d, d / C, None, "3d")
O["lf_1.2cm_unconstrained_wng"] = wng_db(w, a).round(2).tolist()


def wng_for_di(d, f, target_db=3.0):
    """max WNG of an on-axis distortionless beam reaching DI = target (closed form along the Pareto front)."""
    a = steer(f, d / C); g = gamma(f, d, "3d"); G = G2(g)
    b = np.stack([a[..., 0], -a[..., 1]], -1) / np.sqrt(2)
    e = np.abs(np.einsum("...i,...ij,...j->...", a.conj(), G, b)); B0 = quad(b, G); A0 = quad(a, G) / 4
    q = 10 ** (-target_db / 10)                       # required w^H G w
    disc = e ** 2 - 4 * B0 * (A0 - q)
    r = (e - np.sqrt(np.maximum(disc, 0))) / (2 * B0)
    r = np.where(A0 <= q, 0.0, np.where(disc < 0, np.nan, r))
    return 10 * np.log10(1 / (0.5 + r ** 2))

O["lf_1.2cm_wng_for_3dB"] = wng_for_di(0.012, fl4).round(2).tolist()
O["lf_1.2cm_dipole_wng"] = (10 * np.log10(1 - np.cos(2 * np.pi * fl4 * 0.012 / C))).round(2).tolist()
O["lf_1.2cm_cardioid_wng"] = (10 * np.log10(1 - np.cos(4 * np.pi * fl4 * 0.012 / C))).round(2).tolist()
print(json.dumps(O, indent=1))
with open(os.path.join(LAB, "hw", "spacing", "v2", "v2_extra2.json"), "w") as fh:
    json.dump(O, fh, indent=1)
