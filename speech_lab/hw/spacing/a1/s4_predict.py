"""Geometry consistency, feasible board placements and flip-test predictions for H1/H2/H3.

Measured (this folder): direct-sound (first-arrival) TDOA tau_L = -0.51, tau_R = -0.50 (+-0.04, estimator spread);
diffuse/reverberant phase slope (skew under H1) -1.10 (+-0.03); first-arrival path difference r_R - r_L = 6-11 cm
(2.75-5 samples; +-1 sample margin -> 4-13 cm) assuming the Mac's L/R outputs are synchronous.

Placement model: desk plane, speakers at (-s/2, 0) and (+s/2, 0), s = 25-30 cm; board centre P = (x, y);
horizontal mic axis; Mic A at P + (d/2)u. g = (|S-B| - |S-A|) fs / c (exact spherical path difference).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C, FS, HERE, r3  # noqa: E402

TAU_L, TAU_R, TAU_ERR = -0.51, -0.50, 0.04
HYP = {
    "H1": {"tau0": (-1.13, -1.07), "d": (0.0110, 0.0140)},
    "H2": {"tau0": (-0.55, -0.45), "d": (0.0140, 0.0200)},
    "H3": {"tau0": (-0.05, 0.05), "d": (0.0250, 0.0600)},
}
DPATH = (0.04, 0.13)   # r_R - r_L (m)


def gfun(S, P, u, d):
    A = P + 0.5 * d * u
    B = P - 0.5 * d * u
    return (np.linalg.norm(S - B, axis=-1) - np.linalg.norm(S - A, axis=-1)) * FS / C


def feasible(h, s=0.27, use_path=True):
    t0lo, t0hi = HYP[h]["tau0"]
    dlo, dhi = HYP[h]["d"]
    SL, SR = np.array([-s / 2, 0.0]), np.array([s / 2, 0.0])
    xs = np.arange(-1.5, 1.5001, 0.02)
    ys = np.arange(-1.0, 2.0001, 0.02)
    phis = np.radians(np.arange(0, 360, 2))
    U = np.stack([np.cos(phis), np.sin(phis)], 1)            # (nphi, 2)
    ds = np.linspace(dlo, dhi, 7)[:, None, None]              # (nd, 1, 1)
    hits = []
    for x in xs:
        for y in ys:
            P = np.array([x, y])
            rl, rr = np.linalg.norm(SL - P), np.linalg.norm(SR - P)
            if min(rl, rr) < 0.08:
                continue
            if use_path and not (DPATH[0] <= rr - rl <= DPATH[1]):
                continue
            A = P + 0.5 * ds * U[None]                         # (nd, nphi, 2)
            B = P - 0.5 * ds * U[None]
            gl = (np.linalg.norm(SL - B, axis=-1) - np.linalg.norm(SL - A, axis=-1)) * FS / C
            gr = (np.linalg.norm(SR - B, axis=-1) - np.linalg.norm(SR - A, axis=-1)) * FS / C
            # exists tau0 in [t0lo, t0hi] with |tau0 + gl - TAU_L| <= err and |tau0 + gr - TAU_R| <= err
            lo = np.maximum.reduce([TAU_L - gl - TAU_ERR, TAU_R - gr - TAU_ERR, np.full_like(gl, t0lo)])
            hi = np.minimum.reduce([TAU_L - gl + TAU_ERR, TAU_R - gr + TAU_ERR, np.full_like(gl, t0hi)])
            okm = np.any(lo <= hi, axis=0)                     # (nphi,)
            ok_phi = list(np.degrees(phis[okm]))
            if ok_phi:
                # axis direction relative to the bearing from the board to the laptop centre (0 deg = A points at it)
                bearing = np.degrees(np.arctan2(-y, -x))
                rel = [((p - bearing + 180) % 360) - 180 for p in ok_phi]
                hits.append((x, y, rl, rr, rel))
    return hits


def summarise(hits):
    if not hits:
        return {"n_positions": 0}
    rl = np.array([h[2] for h in hits])
    rr = np.array([h[3] for h in hits])
    dist = np.array([np.hypot(h[0], h[1]) for h in hits])
    rel = np.concatenate([h[4] for h in hits])
    front = np.mean([h[1] > 0 for h in hits])
    xs = np.array([h[0] for h in hits])
    return {"n_positions": len(hits), "dist_to_laptop_centre_m": [r3(dist.min()), r3(np.median(dist)), r3(dist.max())],
            "r_L_m": [r3(rl.min()), r3(np.median(rl)), r3(rl.max())], "r_R_m": [r3(rr.min()), r3(rr.max())],
            "x_m(+=right)": [r3(xs.min()), r3(xs.max())], "fraction_in_front": r3(front),
            "axis_rel_bearing_deg(0=A points at laptop centre)": [r3(np.percentile(rel, 2)), r3(np.median(rel)),
                                                                   r3(np.percentile(rel, 98))]}


def predictions():
    """Flip (180 deg in place: g -> -g) and endfire-on-left-speaker predictions."""
    out = {}
    for h, (t0, d) in {"H1": (-1.10, 0.0125), "H2": (-0.50, 0.0160), "H3": (0.0, 0.030)}.items():
        gL, gR = TAU_L - t0, TAU_R - t0
        dc = d * FS / C
        out[h] = {"tau0": t0, "d_cm": 100 * d, "d_over_c_samples": r3(dc), "g_L_now": r3(gL), "g_R_now": r3(gR),
                  "consistent_|g|<=d/c": bool(abs(gL) <= dc + 0.02 and abs(gR) <= dc + 0.02),
                  "flip180_sweep_L": r3(t0 - gL), "flip180_sweep_R": r3(t0 - gR),
                  "axis_B_to_A_at_left_speaker_sweep_L": r3(t0 + dc),
                  "axis_A_to_B_at_left_speaker_sweep_L(reverse)": r3(t0 - dc)}
    # ranges over each hypothesis' parameter box
    for h in HYP:
        t0lo, t0hi = HYP[h]["tau0"]
        dlo, dhi = HYP[h]["d"]
        f1 = [t0lo - (TAU_L - t0lo), t0hi - (TAU_L - t0hi)]
        out[h]["flip180_sweep_L_range"] = [r3(min(f1)), r3(max(f1))]
        e1 = [t0lo + dlo * FS / C, t0hi + dhi * FS / C]
        out[h]["endfire_A_toward_left_range"] = [r3(min(e1)), r3(max(e1))]
        e2 = [t0lo - dhi * FS / C, t0hi - dlo * FS / C]
        out[h]["endfire_B_toward_left_range"] = [r3(min(e2)), r3(max(e2))]
    return out


def main():
    res = {"predictions": predictions(), "placements": {}}
    for h in HYP:
        for s in (0.25, 0.30):
            hits = feasible(h, s)
            res["placements"][f"{h}_s{int(s * 100)}cm_with_path_constraint"] = summarise(hits)
            print(h, s, res["placements"][f"{h}_s{int(s * 100)}cm_with_path_constraint"], flush=True)
        hits = feasible(h, 0.27, use_path=False)
        res["placements"][f"{h}_s27cm_no_path_constraint"] = summarise(hits)
        print(h, "nopath", res["placements"][f"{h}_s27cm_no_path_constraint"], flush=True)
    print(json.dumps(res["predictions"], indent=1))
    with open(os.path.join(HERE, "s4_predict.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
