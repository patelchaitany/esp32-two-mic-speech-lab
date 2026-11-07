"""Exp 2: 'nearness' from two mics 1.19 cm apart. Does the coherent-to-diffuse ratio (CDR) separate near from far?

Part A, real takes (18): per-take median CDR (dB) over oracle speech frames, DOA-free (Schwarz-Kellermann prop3)
and DOA-dependent (prop1, plane wave at the measured +0.55 sample acoustic TDOA), after the board calibration
(calib_session.json through methods._core.calib), smoothing 50 ms, Gamma_n = sinc(2fd/c) at 1.19 cm, in 2-4 kHz
and 4-7 kHz; conditions 'img' (fitted image, no noise), 'quiet' (image + noise at -12 dB), 'rec' (recording).
Truth: DRR / C50 of the dry->Mic A response (bz_takes; MF takes include the emulated 4 m room).
NB: the MF takes' far-room reverb is pre-convolved and played from the SAME near speaker, so at the device it is a
coherent plane wave: spatial coherence cannot see it. Part B therefore simulates real far talkers.
Part B, simulation (pyroomacoustics): bus-like (11.5 x 2.5 x 2.2 m) and meeting-like (8 x 6 x 3 m) rooms,
RT60 0.45 / 0.6 s, sources at 0.3-4 m (10 random placements each), 1.19 cm vs 4 cm pairs, noise-free and with diffuse speech-shaped noise
at 0 dB; per-utterance median CDR and the AUC of near (<= 1 m) vs far (>= 2 m).
Run: .venv/bin/python -I hw/realset/bizarre/bz2_nearness.py [A|B] -> bz2_nearness.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import C, CALIB, FS, HERE, REALSET, dump  # noqa: E402

TAKES = ["A1", "A2", "A3", "B1", "B2", "B3", "AE1", "AE2", "BE1", "BE2", "M1", "M2", "M3", "M4", "M5", "M6",
         "MF1", "MF2"]
BANDS = {"2-4k": (2000, 4000), "4-7k": (4000, 7000)}


def cdr_frames(X, d, tau_s=None, lam_s=0.05):
    """X (2, F, N) calibrated STFT -> {band: per-frame CDR dB (DOA-free), band+'_doa': (DOA-dependent)}."""
    from methods._core import common as cm
    lam = cm.lam(lam_s)
    paa, pbb, pab = cm.spectra(X)
    paa = cm.smooth(paa, lam)[0]
    pbb = cm.smooth(pbb, lam)[0]
    pab = cm.smooth(pab, lam)[0]
    G = cm.coherence(paa, pbb, pab)
    Gn = cm.diffuse_model(cm.FREQS, d)[:, None]
    out = {}
    r0 = cm.cdr_nodoa(G, Gn)
    r1 = cm.cdr_doa(G, Gn, cm.plane_coherence(cm.FREQS, tau_s)[:, None]) if tau_s is not None else None
    for b, (lo, hi) in BANDS.items():
        m = (cm.FREQS >= lo) & (cm.FREQS < hi)
        out[b] = 10 * np.log10(np.mean(r0[m], 0) + 1e-3)
        if r1 is not None:
            out[b + "_doa"] = 10 * np.log10(np.mean(r1[m], 0) + 1e-3)
    return out


def active_frames(img_a, iv):
    from methods._core import common as cm
    E = cm.stft(img_a)
    e = 10 * np.log10(np.mean(np.abs(E[10:128]) ** 2, 0) + 1e-20)
    cen = cm.frame_centers(E.shape[-1])
    inside = np.zeros_like(e, bool)
    for a, b in iv:
        inside |= (cen >= a) & (cen < b)
    return inside & (e > e[inside].max() - 25)


def part_a():
    from methods._core import calib, common as cm
    import bz_takes as bt
    rows = {}
    for t in TAKES:
        tk = bt.composite(t)
        iv = bt.speech_iv(tk, extra_s=0.0)
        X0 = cm.stft(calib.prefilter(tk["recording"], "meeting"))
        cal = calib.calibrate(X0, {"scenario": "meeting", "spacing_m": CALIB["spacing_m"], "calib_json": CALIB})
        act = active_frames(tk["image"][0], iv)
        drr, c50 = bt.drr_c50(tk["comp"][:, 0], tk["p"])
        r = {"far": tk["far"], "drr_db": drr, "c50_db": c50, "n_frames": int(act.sum())}
        for cond, x in (("img", tk["image"]), ("quiet", tk["image"] + 0.25 * tk["noise"]), ("rec", tk["recording"])):
            X = calib.apply(cm.stft(calib.prefilter(x, "meeting")), cal)
            cf = cdr_frames(X, CALIB["spacing_m"], tau_s=0.55 / FS)
            for k, v in cf.items():
                r[f"{cond}_{k}"] = float(np.median(v[act]))
                r[f"{cond}_{k}_frames"] = v[act].astype(np.float32)
        rows[t] = r
        print(t, {k: round(v, 2) for k, v in r.items() if isinstance(v, float)}, flush=True)
    return rows


def auc(pos, neg):
    """P(pos > neg) (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def speech_shaped_diffuse(T, d, spec, rng):
    """Two-channel diffuse noise with coherence sinc(2fd/c) and the magnitude spectrum `spec` (257,)."""
    from methods._core import common as cm
    N = cm.n_frames(T)
    n1 = rng.standard_normal((257, N)) + 1j * rng.standard_normal((257, N))
    n2 = rng.standard_normal((257, N)) + 1j * rng.standard_normal((257, N))
    g = cm.diffuse_model(cm.FREQS, d)[:, None]
    X = np.stack([n1, g * n1 + np.sqrt(np.maximum(1 - g ** 2, 0)) * n2]) * spec[None, :, None]
    return cm.istft(X, T)


def part_b():
    import pyroomacoustics as pra
    from methods._core import common as cm
    rng = np.random.default_rng(11)
    speech = []
    for t in ("A1", "B1", "M1", "M2", "M4", "M5"):
        st = np.load(os.path.join(REALSET, "stim", t + ".npz"))["dry"].astype(np.float64)
        speech.append(st[int(0.25 * FS):])
    spec = np.sqrt(np.mean(np.abs(cm.stft(np.concatenate(speech))) ** 2, 1))
    rooms = {"bus": ([11.5, 2.5, 2.2], 0.45, [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]),
             "meeting": ([8.0, 6.0, 3.0], 0.6, [0.5, 1.0, 2.0, 3.0, 4.0])}
    out = {}
    for rname, (dims, rt60, dists) in rooms.items():
        e_abs, max_order = pra.inverse_sabine(rt60, dims)
        for d in (0.0119, 0.04):
            for snr in (None, 0.0):
                key = f"{rname}_{d * 100:.2f}cm_{'clean' if snr is None else 'snr0'}"
                res = {}
                for dist in dists:
                    vals = {b: [] for b in BANDS}
                    frames = {b: [] for b in BANDS}
                    for rep in range(10):
                        # device placement: inside the room, 1.1 m high; source at the distance, random azimuth
                        for _ in range(100):
                            cx = rng.uniform(1.0, dims[0] - 1.0)
                            cy = rng.uniform(0.6, dims[1] - 0.6) if dims[1] > 3 else dims[1] / 2
                            az = rng.uniform(0, 2 * np.pi)
                            sx, sy = cx + dist * np.cos(az), cy + dist * np.sin(az)
                            if 0.3 < sx < dims[0] - 0.3 and 0.3 < sy < dims[1] - 0.3:
                                break
                        else:                                   # along the long axis
                            cx, cy, az = 1.0, dims[1] / 2, 0.0
                            sx, sy = cx + dist, cy
                        room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(e_abs), max_order=min(max_order, 25))
                        sp = speech[(rep + int(dist * 10)) % len(speech)]
                        room.add_source([sx, sy, 1.2], signal=sp)
                        ax = rng.uniform(0, 2 * np.pi)
                        mics = np.array([[cx - d / 2 * np.cos(ax), cx + d / 2 * np.cos(ax)],
                                         [cy - d / 2 * np.sin(ax), cy + d / 2 * np.sin(ax)], [1.1, 1.1]])
                        room.add_microphone_array(mics)
                        room.simulate()
                        x = room.mic_array.signals[:, :len(sp) + 8000]
                        X = cm.stft(x)
                        e = 10 * np.log10(np.mean(np.abs(X[0, 10:128]) ** 2, 0) + 1e-20)
                        act = e > e.max() - 25
                        if snr is not None:
                            n = speech_shaped_diffuse(x.shape[1], d, spec, rng)
                            ps = np.mean(np.abs(X[0, :, act]) ** 2)
                            pn = np.mean(np.abs(cm.stft(n)[0]) ** 2)
                            X = cm.stft(x + n * np.sqrt(ps / pn / 10 ** (snr / 10)))
                        cf = cdr_frames(X, d)
                        for b in BANDS:
                            vals[b].append(float(np.median(cf[b][act])))
                            frames[b].append(cf[b][act])
                    res[dist] = {"median_cdr_db": {b: vals[b] for b in BANDS},
                                 "frames": {b: np.concatenate(frames[b]) for b in BANDS}}
                near = [k for k in dists if k <= 1.0]
                farr = [k for k in dists if k >= 2.0]
                summ = {}
                for b in BANDS:
                    summ[b] = {"per_dist_median": {str(k): float(np.median(res[k]["median_cdr_db"][b])) for k in dists},
                               "auc_utt_near_vs_far": auc([v for k in near for v in res[k]["median_cdr_db"][b]],
                                                          [v for k in farr for v in res[k]["median_cdr_db"][b]]),
                               "auc_frame_near_vs_far": auc(np.concatenate([res[k]["frames"][b] for k in near])[::7],
                                                            np.concatenate([res[k]["frames"][b] for k in farr])[::7])}
                out[key] = summ
                print(key, summ, flush=True)
    return out


def main():
    t0 = time.time()
    parts = sys.argv[1:] or ["A", "B"]
    res = {}
    path = os.path.join(HERE, "bz2_nearness.json")
    if os.path.exists(path):
        import json
        res = json.load(open(path))
    if "A" in parts:
        rows = part_a()
        near = [t for t in TAKES if not rows[t]["far"]]
        far = [t for t in TAKES if rows[t]["far"]]
        summ = {}
        for key in [k for k in rows[TAKES[0]] if k.endswith("_frames") and k != "n_frames"]:
            base = key[:-7]
            summ[base] = {"near_median": float(np.median([rows[t][base] for t in near])),
                          "near_range": [float(min(rows[t][base] for t in near)), float(max(rows[t][base] for t in near))],
                          "far": {t: rows[t][base] for t in far},
                          "auc_frame_near_vs_far": auc(np.concatenate([rows[t][key] for t in near]),
                                                       np.concatenate([rows[t][key] for t in far])),
                          "spearman_vs_drr": float(_spearman([rows[t]["drr_db"] for t in TAKES],
                                                             [rows[t][base] for t in TAKES])),
                          "spearman_vs_drr_near_only": float(_spearman([rows[t]["drr_db"] for t in near],
                                                                       [rows[t][base] for t in near]))}
        for t in TAKES:
            for k in [k for k in rows[t] if k.endswith("_frames") and k != "n_frames"]:
                del rows[t][k]
        res["A"] = {"rows": rows, "summary": summ}
    if "B" in parts:
        res["B"] = part_b()
    res["wall_s"] = time.time() - t0
    dump(res, path)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


if __name__ == "__main__":
    main()
