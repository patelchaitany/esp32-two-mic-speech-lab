"""Scoring for the two test cases.

bus:     SI-SDR / ESTOI / PESQ-WB vs the two target talkers' direct+50 ms image at mic A,
         worst-of-two-targets ESTOI (both must survive), and BR = background reduction
         (target gain minus gain in target-silent regions, gain-invariant, dB).
meeting: per-talker ESTOI and SI-SDR (direct+50 ms reference), worst-talker ESTOI,
         loudness spread across the 20 talkers (dB std), and gap SNR (median talker level
         minus noise floor in the silent lead-in/out).
"""
import numpy as np
from pystoi import stoi
from pesq import pesq
from .common import FS, db, rms, active_rms


def align(est, ref, max_lag=1600):
    n = len(ref)
    L = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(est, L) * np.conj(np.fft.rfft(ref, L)), L)
    cand = np.r_[xc[: max_lag + 1], xc[-max_lag:]]
    lags = np.r_[np.arange(max_lag + 1), np.arange(-max_lag, 0)]
    i = int(np.argmax(np.abs(cand)))
    k, sgn = lags[i], np.sign(cand[i]) or 1.0
    out = np.zeros_like(est)
    if k >= 0:
        out[: n - k] = est[k:n]
    else:
        out[-k:] = est[: n + k]
    return out * sgn, int(k)


def si_sdr(est, ref):
    est = est - est.mean(); ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    s = a * ref
    return float(10 * np.log10(np.sum(s ** 2) / (np.sum((est - s) ** 2) + 1e-20) + 1e-20))


def estoi(est, ref):
    try:
        return float(stoi(ref, est, FS, extended=True))
    except Exception:
        return float("nan")


def pesq_wb(est, ref):
    try:
        e = est / (np.abs(est).max() + 1e-9) * 0.5
        r = ref / (np.abs(ref).max() + 1e-9) * 0.5
        return float(pesq(FS, r, e, "wb"))
    except Exception:
        return float("nan")


def _mask(segs, T, pad=0):
    m = np.zeros(T, bool)
    for s0, s1, *_ in segs:
        m[max(s0 - pad, 0): min(s1 + pad, T)] = True
    return m


def score_bus(out, scene):
    ref = scene["ref_early"].astype(np.float64)
    x0 = scene["mix"][0].astype(np.float64)
    out, lag = align(out.astype(np.float64), ref)
    T = len(ref)
    segs = scene["segs"]
    act = _mask(segs, T)
    sil = ~_mask(segs, T, pad=int(0.4 * FS))
    sil[: int(0.3 * FS)] = False
    # target gain via projection onto reference in active regions; background gain in silences
    a = np.dot(out[act], ref[act]) / np.dot(ref[act], ref[act])
    a_in = np.dot(x0[act], ref[act]) / np.dot(ref[act], ref[act])
    g_t = 20 * np.log10(abs(a) / abs(a_in) + 1e-12)
    g_n = max(10 * np.log10(np.mean(out[sil] ** 2) / np.mean(x0[sil] ** 2) + 1e-20), g_t - 40.0)  # cap BR at 40 dB
    per_t = []
    for tid in (0, 1):
        m = np.zeros(T, bool)
        for s0, s1, t in segs:
            if t == tid:
                m[s0:s1] = True
        per_t.append(estoi(out[m], ref[m]))
    r = dict(si_sdr=si_sdr(out, ref), si_sdr_in=si_sdr(x0, ref), estoi=estoi(out, ref), estoi_in=estoi(x0, ref),
             estoi_worst_target=float(np.nanmin(per_t)), pesq=pesq_wb(out, ref), pesq_in=pesq_wb(x0, ref),
             bg_reduction_db=float(g_t - g_n), target_gain_db=float(g_t), lag=lag)
    r["si_sdr_imp"] = r["si_sdr"] - r["si_sdr_in"]
    return r


def score_meeting(out, scene):
    ref = scene["ref_early"].astype(np.float64)
    x0 = scene["mix"][0].astype(np.float64)
    out, lag = align(out.astype(np.float64), ref)
    segs = scene["segs"]
    res = {"talker": []}
    for name, sig in (("out", out), ("in", x0)):
        es, sd, ld = [], [], []
        for s0, s1, tid in segs:
            es.append(estoi(sig[s0:s1], ref[s0:s1]))
            sd.append(si_sdr(sig[s0:s1], ref[s0:s1]))
            ld.append(db(active_rms(sig[s0:s1])))
        ld = np.array(ld)
        quiet = np.r_[sig[int(0.2 * FS): segs[0][0] - int(0.3 * FS)], sig[segs[-1][1] + int(0.6 * FS):]]
        res[name] = dict(estoi_mean=float(np.nanmean(es)), estoi_worst=float(np.nanmin(es)),
                         estoi_p10=float(np.nanpercentile(es, 10)), si_sdr_mean=float(np.mean(sd)),
                         loud_std_db=float(np.std(ld)), loud_range_db=float(ld.max() - ld.min()),
                         gap_snr_db=float(np.median(ld) - db(rms(quiet))))
        if name == "out":
            res["talker"] = [dict(tid=int(t), estoi=float(e), si_sdr=float(s), loud_db=float(l))
                             for (_, _, t), e, s, l in zip(segs, es, sd, ld)]
    r = {f"{k}": v for k, v in res["out"].items()}
    r.update({f"{k}_in": v for k, v in res["in"].items()})
    r["si_sdr_imp"] = r["si_sdr_mean"] - r["si_sdr_mean_in"]
    r["pesq"] = pesq_wb(out, ref)
    r["pesq_in"] = pesq_wb(x0, ref)
    r["talker"] = res["talker"]
    r["lag"] = lag
    return r
