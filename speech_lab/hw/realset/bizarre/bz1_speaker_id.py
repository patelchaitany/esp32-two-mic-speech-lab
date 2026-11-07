"""Exp 1: speaker identity on the real meeting sequence (8 talkers: M1-M6 at 0..-15 dB, MF1/MF2 emulated far).

Scenes: realscenes meeting/recorded, meeting/quiet, meeting/quiet_omeeting (38.2 s, one 4.2 s turn per talker).
Inputs (Mic A): 'clean' (sum of the talker images, no noise; upper bound), 'raw' (S0 80 Hz prefilter of the mix),
'gtcrn' (methods/m8_gtcrn, NN denoiser, meeting attenuation limit 12 dB), 'wpe' (methods/m10_wpe).
Embeddings: WeSpeaker ResNet34 (methods/_core/spk_onnx), 1.5 s chunks, hop 0.75 s, inside the TRUE turns (oracle
segmentation, so only identity is tested). Measures:
  - closed-set identification vs each talker's dry utterance (same utterance: channel/noise mismatch only), EER of
    chunk-vs-talker cosines (target/non-target);
  - unsupervised count/clustering: AHC (average linkage, cosine) cut at THETA_S = 0.375 (spk_onnx recommendation)
    and at 0.5; online CosineClusterer (spk.py) at 0.375 and 0.6; purity, talker coverage, ARI, clusters found;
  - S1 route as deployed (tracker.front_end with the embedder, cos_thr THETA_S, calib_session.json) and S1's
    DP-means fallback (no embedder), frame-level against the true turns.
Run: .venv/bin/python -I hw/realset/bizarre/bz1_speaker_id.py -> bz1_speaker_id.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import CALIB, FS, HERE, REALSET, dump  # noqa: E402

SCENES = ["meeting/recorded", "meeting/quiet", "meeting/quiet_omeeting"]
TALKERS = ["M1", "M2", "M3", "M4", "M5", "M6", "MF1", "MF2"]


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ct = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(ct, (ai, bi), 1)
    comb = lambda n: n * (n - 1) / 2.0
    sij = comb(ct).sum()
    sa, sb = comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    n = comb(len(a))
    exp = sa * sb / n if n else 0
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def cluster_report(labels, truth):
    labels, truth = np.asarray(labels), np.asarray(truth)
    if len(labels) == 0:
        return {"n_clusters": 0, "purity": float("nan"), "talkers_covered": 0, "absorbed_talkers": [], "ari": float("nan"),
                "empty": True}
    cl = np.unique(labels)
    maj = {}
    pure = 0
    for c in cl:
        t, n = np.unique(truth[labels == c], return_counts=True)
        maj[int(c)] = str(t[np.argmax(n)])
        pure += n.max()
    covered = sorted(set(maj.values()))
    # talkers whose chunks are mostly in a cluster whose majority is another talker = absorbed
    absorbed = []
    for t in np.unique(truth):
        c, n = np.unique(labels[truth == t], return_counts=True)
        if maj[int(c[np.argmax(n)])] != t:
            absorbed.append(str(t))
    return {"n_clusters": int(len(cl)), "purity": float(pure / len(labels)), "talkers_covered": len(covered),
            "absorbed_talkers": absorbed, "ari": ari(labels, truth)}


def eer(tgt, non):
    tgt, non = np.asarray(tgt), np.asarray(non)
    ths = np.sort(np.concatenate([tgt, non]))
    best = (1.0, 0.0)
    for th in ths:
        fr = np.mean(tgt < th)
        fa = np.mean(non >= th)
        if abs(fr - fa) < abs(best[0] - best[1]):
            best = (fr, fa)
    return float((best[0] + best[1]) / 2)


def main():
    import realscenes
    from scipy.cluster.hierarchy import fcluster, linkage
    from methods._core import calib, spk, spk_onnx, tracker
    import methods.m8_gtcrn as m8
    import methods.m10_wpe as m10
    emb = spk_onnx.get_embedder()
    refs = {}
    for t in TALKERS:
        dry = np.load(os.path.join(REALSET, "stim", t + ".npz"))["dry"].astype(np.float64)
        refs[t] = emb.embed(dry[int(0.25 * FS):])
    R = np.stack([refs[t] for t in TALKERS])
    res = {"ref_cross_cos": (R @ R.T).round(3).tolist(), "scenes": {}}
    t0 = time.time()
    for sid in SCENES:
        sc = realscenes.load_scene(sid)
        mix = np.asarray(sc["mix"], np.float64)
        segs = [(int(a), int(b), str(t)) for a, b, t in sc["segs"]]
        clean = sum(np.asarray(sc["comp2"][g], np.float64) for g, r in sc["groups"].items() if r == "talker")
        ins = {"clean": calib.prefilter(clean, "meeting")[0], "raw": calib.prefilter(mix, "meeting")[0],
               "gtcrn": m8.enhance(mix, FS, {"scenario": "meeting"}),
               "wpe": m10.enhance(mix, FS, {"scenario": "meeting"})}
        out = {}
        for name, y in ins.items():
            chunks, truth = [], []
            for a, b, t in segs:
                for s in range(a, b - int(1.5 * FS) + 1, int(0.75 * FS)):
                    chunks.append((s, s + int(1.5 * FS)))
                    truth.append(t)
            E = emb.embed_segments(y, chunks)
            truth = np.array(truth)
            S = E @ R.T                                      # (n, 8) cosine to each talker's dry utterance
            pred = np.array(TALKERS)[np.argmax(S, 1)]
            tgt = [S[i, TALKERS.index(truth[i])] for i in range(len(truth))]
            non = [S[i, j] for i in range(len(truth)) for j in range(len(TALKERS)) if TALKERS[j] != truth[i]]
            r = {"n_chunks": int(len(truth)), "id_acc": float(np.mean(pred == truth)),
                 "id_acc_per_talker": {t: float(np.mean(pred[truth == t] == t)) for t in TALKERS},
                 "cos_target_median": float(np.median(tgt)), "cos_nontarget_p95": float(np.percentile(non, 95)),
                 "eer": eer(tgt, non),
                 "cos_target_per_talker": {t: float(np.median(np.array(tgt)[truth == t])) for t in TALKERS}}
            Z = linkage(E, "average", metric="cosine")
            for thr in (0.375, 0.5):
                r[f"ahc_{thr}"] = cluster_report(fcluster(Z, t=1 - thr, criterion="distance"), truth)
            r["ahc_k8"] = cluster_report(fcluster(Z, t=8, criterion="maxclust"), truth)
            for thr in (0.375, 0.6):
                cc = spk.CosineClusterer(thr=thr)
                r[f"online_{thr}"] = cluster_report([cc.assign(e) for e in E], truth)
            out[name] = r
            print(sid, name, {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'id_acc_per_talker'})
                              for k, v in r.items() if k in ("id_acc", "eer", "ahc_0.375", "online_0.375", "ahc_k8")},
                  flush=True)
        # S1 as deployed (2-ch, offline, calibrated): frame-level cluster labels vs the true turns
        info = {"scenario": "meeting", "spacing_m": CALIB["spacing_m"], "calib_json": CALIB}
        from methods._core import common as cm
        for route, kw in (("s1_embed", {"embedder": spk_onnx.get_embedder(), "cos_thr": spk_onnx.THETA_S}),
                          ("s1_dpmeans", {})):
            fe = tracker.front_end(mix, FS, info, **kw)
            lab = fe.track.cluster
            cen = cm.frame_centers(len(lab))
            tr = np.array([""] * len(lab), dtype=object)
            for a, b, t in segs:
                tr[(cen >= a + int(0.3 * FS)) & (cen < b)] = t
            sel = (tr != "") & (lab >= 0)
            rr = cluster_report(lab[sel], tr[sel].astype(str))
            rr["frac_turn_frames_labelled"] = float(np.mean(lab[tr != ""] >= 0))
            rr["n_clusters_total"] = int(len(fe.track.clusters))
            rr["vad_gt_0.5_frac"] = float(np.mean(fe.track.vad > 0.5))
            rr["vad_gt_0.5_frac_in_turns"] = float(np.mean(fe.track.vad[tr != ""] > 0.5))
            rr["vad_gt_0.5_frac_per_talker"] = {t: float(np.mean(fe.track.vad[tr == t] > 0.5)) for t in TALKERS}
            rr["labelled_frac_per_talker"] = {t: float(np.mean(lab[tr == t] >= 0)) for t in TALKERS}
            out[route] = rr
            print(sid, route, rr, flush=True)
        res["scenes"][sid] = out
    res["wall_s"] = time.time() - t0
    dump(res, os.path.join(HERE, "bz1_speaker_id.json"))


if __name__ == "__main__":
    main()
