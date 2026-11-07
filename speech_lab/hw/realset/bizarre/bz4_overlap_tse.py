"""Exp 4: partner extraction under FULL overlap with M9's VoxLock pass (one pass per enrolled partner).

Mixtures:
  'acoustic' : hw/rec/talkAB_LR.wav (session 1, A = LibriSpeech 121 F on the left speaker and B = 1089 M on the right,
               played at the same time). References: both 4096-tap responses fitted JOINTLY on this take
               (two-input least squares, conjugate gradients on pre-whitened signals), so the images sum to the
               recording up to the take's own noise; refs = early images (direct + 50 ms) at Mic A.
  'digital_k': realset takes A_k + B_k (k = 1..3, same utterances, current boot), image_A + image_B + noise of A_k
               (exact image references; B is ~6 dB weaker at Mic A as recorded).
Processing: S0 bus prefilter (120 Hz) on Mic A, then methods._core.tse.extract (VoxLock 1.1.1 int8, reference
partition, no priming) enrolled on partner k with 'clean' (dry enrollment utterances AE1+AE2 / BE1+BE2), 'device'
(Mic A recordings of the same, fan noise ~0 dB SNR) or 'wrong' (a third speaker's dry speech: M1 3575 F for A,
M2 2300 M for B) as a control.
Scores over the speech interval: SI-SDR and ESTOI of the pass output vs the target's early image, and the
mask projection: the pass's effective lab-grid mask applied to each component separately -> SIR in/out,
target retention (TA, dB) and other-partner attenuation (dB).
Run: .venv/bin/python -I hw/realset/bizarre/bz4_overlap_tse.py -> bz4_overlap_tse.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, cg  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

from bz_common import FS, HERE, HW, REALSET, dump, load_wav  # noqa: E402

K = 4096
LEAD = int(0.25 * FS)


def _fft_conv(a, h, n, nfft):
    return np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def fit_joint(y, s1, s2, k=K, iters=400):
    """Least-squares h1, h2 (k taps) with y ~ s1*h1 + s2*h2 (all (N,), aligned). Pre-whitened by a common LPC
    filter of s1 + s2 (the LS solution of the convolutive model is unchanged), solved by CG on the normal eqs."""
    from scipy.linalg import solve_toeplitz
    N = len(y)
    ref = s1 + s2
    r = np.correlate(ref, ref, "full")[N - 1:N + 32]
    a = solve_toeplitz(r[:32], -r[1:33])
    w = np.concatenate([[1.0], a])
    yw, s1w, s2w = (lfilter(w, [1.0], z) for z in (y, s1, s2))
    nfft = 1 << int(np.ceil(np.log2(N + k)))
    S1, S2 = np.fft.rfft(s1w, nfft), np.fft.rfft(s2w, nfft)

    def fwd(v):
        H1, H2 = np.fft.rfft(v[:k], nfft), np.fft.rfft(v[k:], nfft)
        return np.fft.irfft(S1 * H1 + S2 * H2, nfft)[:N]

    def adj(e):
        E = np.fft.rfft(e, nfft)
        g1 = np.fft.irfft(E * np.conj(S1), nfft)[:k]
        g2 = np.fft.irfft(E * np.conj(S2), nfft)[:k]
        return np.concatenate([g1, g2])

    lam = 1e-6 * (np.sum(s1w ** 2) + np.sum(s2w ** 2)) / 2
    op = LinearOperator((2 * k, 2 * k), matvec=lambda v: adj(fwd(v)) + lam * v, dtype=np.float64)
    v, info = cg(op, adj(yw), maxiter=iters)
    return v[:k], v[k:], int(info)


def acoustic_mixture():
    from analyze import bandpass, stretch, xcorr_lag
    y = load_wav(os.path.join(HW, "rec", "talkAB_LR.wav"))           # (2, N), 80 Hz HP
    st = np.load(os.path.join(HW, "stim", "talkAB_LR.npz"))
    sA, sB = stretch(st["A"].astype(np.float64), 17e-6), stretch(st["B"].astype(np.float64), 17e-6)
    N = y.shape[1]
    n0 = xcorr_lag(bandpass(y[0], 150, 7000), bandpass(sA + sB, 150, 7000)) - 256
    pa, pb = np.zeros(N), np.zeros(N)
    pa[n0:n0 + len(sA)] = sA[:N - n0]
    pb[n0:n0 + len(sB)] = sB[:N - n0]
    hA, hB = np.zeros((K, 2)), np.zeros((K, 2))
    infos = []
    for c in range(2):
        hA[:, c], hB[:, c], inf = fit_joint(y[c], pa, pb)
        infos.append(inf)
    img = {"A": np.stack([_fft_conv(pa, hA[:, c], N, 1 << 19) for c in range(2)]),
           "B": np.stack([_fft_conv(pb, hB[:, c], N, 1 << 19) for c in range(2)])}
    noise = y - img["A"] - img["B"]
    on = n0 + 256 + LEAD
    act, quiet = slice(on, on + int(4.2 * FS)), slice(on + int(4.9 * FS), N - FS // 4)
    p = lambda z: float(10 * np.log10(np.mean(z ** 2)))
    diag = {"cg_info": infos, "n0": int(n0), "imgA_dbfs": p(img["A"][0, act]), "imgB_dbfs": p(img["B"][0, act]),
            "resid_speech_dbfs": p(noise[0, act]), "noise_only_dbfs": p(noise[0, quiet])}
    early = {}
    for s, h, src in (("A", hA, pa), ("B", hB, pb)):
        pk = int(np.argmax(np.abs(h).sum(1)))
        early[s] = _fft_conv(src, h[:pk + 800, 0], N, 1 << 19)
        diag[f"direct_idx_{s}"] = pk
    # cross-check against the separately recorded single-talker takes (direct 5 ms of the responses)
    from analyze import deconvolve  # noqa: F401
    for s, take, h in (("A", "talkA_L", hA), ("B", "talkB_R", hB)):
        ys = load_wav(os.path.join(HW, "rec", take + ".wav"))
        sts = np.load(os.path.join(HW, "stim", take + ".npz"))
        import realset as rs
        _, hs, _ = rs.fit_image(ys.T, sts[s].astype(np.float64))   # fit_image stretches itself
        pk, ps = int(np.argmax(np.abs(h).sum(1))), int(np.argmax(np.abs(hs).sum(1)))
        a_, b_ = h[pk - 16:pk + 80, 0], hs[ps - 16:ps + 80, 0]
        g = np.dot(a_, b_) / np.dot(b_, b_)
        diag[f"direct5ms_vs_single_take_{s}_db"] = float(10 * np.log10(np.sum((a_ - g * b_) ** 2) / np.sum(a_ ** 2)))
    iv = [(on, on + int(4.2 * FS))]
    return {"mix": y, "img": img, "noise": noise, "early": early, "iv": iv, "diag": diag}


def digital_mixture(k):
    import bz_takes as bt
    ta, tb = bt.composite(f"A{k}"), bt.composite(f"B{k}")
    # align B's take to A's by their onsets (both start their played signal at n0 + 0.25 s)
    sh = ta["n0"] - tb["n0"]
    N = ta["N"]

    def shift(z):
        out = np.zeros_like(z)
        if sh >= 0:
            out[..., sh:] = z[..., :N - sh]
        else:
            out[..., :N + sh] = z[..., -sh:]
        return out
    img = {"A": ta["image"], "B": shift(tb["image"])}
    early = {"A": bt.render(ta, ta["p"] + 800)[0], "B": shift(bt.render(tb, tb["p"] + 800)[0])}
    mix = img["A"] + img["B"] + ta["noise"]
    a = ta["n0"] + LEAD
    iv = [(a, a + int(4.2 * FS))]
    return {"mix": mix, "img": img, "noise": ta["noise"], "early": early, "iv": iv, "diag": {"shift": int(sh)}}


def score(m, cond_by_e):
    from lab import metrics as v1, metrics2 as m2
    from methods._core import calib, common as cm, tse
    xp = calib.prefilter(m["mix"], "bus")[0]
    comps = {s: calib.prefilter(m["img"][s], "bus")[0] for s in ("A", "B")}
    comps["N"] = calib.prefilter(m["noise"], "bus")[0]
    X = cm.stft(xp)
    CX = {s: cm.stft(c) for s, c in comps.items()}
    (a, b), = m["iv"]
    T = len(xp)
    e = lambda z: float(np.sum(z[a:b] ** 2))
    out = {}
    for tgt, oth in (("A", "B"), ("B", "A")):
        ref = m["early"][tgt]
        ya, _ = v1.align(xp[a:b], ref[a:b], max_lag=160)
        base = {"si_sdr": v1.si_sdr(ya, ref[a:b]), "estoi": m2.estoi(xp, ref, m["iv"]),
                "sir_db": 10 * np.log10(e(comps[tgt]) / e(comps[oth])),
                "snr_db": 10 * np.log10(e(comps[tgt]) / e(comps["N"]))}
        out[tgt] = {"input": base}
        for en, conds in cond_by_e.items():
            ex = tse.extract(xp, conds[tgt], X)
            ya, _ = v1.align(ex.y[a:b], ref[a:b], max_lag=160)
            proj = {s: cm.istft(ex.mask * CX[s], T) for s in CX}
            r = {"si_sdr": v1.si_sdr(ya, ref[a:b]), "estoi": m2.estoi(ex.y, ref, m["iv"]),
                 "sir_db": 10 * np.log10(e(proj[tgt]) / e(proj[oth])),
                 "snr_db": 10 * np.log10(e(proj[tgt]) / e(proj["N"])),
                 "ta_db": 10 * np.log10(e(proj[tgt]) / e(comps[tgt])),
                 "other_att_db": 10 * np.log10(e(proj[oth]) / e(comps[oth])),
                 "noise_att_db": 10 * np.log10(e(proj["N"]) / e(comps["N"])),
                 "out_energy_re_target_db": 10 * np.log10(e(ex.y) / e(comps[tgt])), "pass_s": ex.seconds}
            # time course: per 1 s, the kept share of the target (does the pass latch late or switch talkers?)
            r["sir_per_s"] = [float(10 * np.log10(np.sum(proj[tgt][s:s + FS] ** 2) / (np.sum(proj[oth][s:s + FS] ** 2) + 1e-20)))
                              for s in range(a, b - FS + 1, FS)]
            out[tgt][en] = r
    return out


def main():
    import realscenes
    from methods._core import tse
    t0 = time.time()
    model = tse.get_model()
    enr = {"clean": realscenes.load_enroll("clean"), "device": realscenes.load_enroll("device")}
    wrong = {"A": np.load(os.path.join(REALSET, "stim", "M1.npz"))["dry"].astype(np.float64),
             "B": np.load(os.path.join(REALSET, "stim", "M2.npz"))["dry"].astype(np.float64)}
    conds = {e: {s: tse.enroll(enr[e][s], model) for s in ("A", "B")} for e in enr}
    conds["wrong"] = {s: tse.enroll(wrong[s], model) for s in ("A", "B")}
    res = {}
    mixes = [("acoustic", acoustic_mixture)] + [(f"digital_{k}", (lambda k=k: digital_mixture(k))) for k in (1, 2, 3)]
    for name, fn in mixes:
        m = fn()
        r = score(m, conds)
        r["diag"] = m["diag"]
        res[name] = r
        print(name, m["diag"], flush=True)
        for tgt in ("A", "B"):
            print("  ", tgt, {k: {kk: round(vv, 2) for kk, vv in v.items() if kk in ("si_sdr", "estoi", "sir_db", "ta_db", "other_att_db")}
                              for k, v in r[tgt].items()}, flush=True)
    res["wall_s"] = time.time() - t0
    dump(res, os.path.join(HERE, "bz4_overlap_tse.json"))


if __name__ == "__main__":
    main()
