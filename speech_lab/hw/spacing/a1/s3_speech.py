"""Speech/noise recordings: per-speaker room responses estimated from the played references (2-input LS per
frequency, Welch-averaged, references stretched by the measured +17 ppm), then
  - direct-sound TDOA per talker (GCC-PHAT, phase slope) and reverberant-window phase slope;
  - arrival-time difference right speaker minus left speaker (same recording, same playback start) -> r_R - r_L;
  - blind frame-level TDOA (PHAT steered response on the raw recording, high-MSC speech frames).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FS, HERE, bandpass, frames_spec, gcc_phat, load, phase_slope, r3, stim, stretch  # noqa: E402

ALPHA = 17e-6
NF = 8192
HOP = 2048
CASES = {"talkA_L": ["L"], "talkB_R": ["R"], "talkAB_LR": ["L", "R"], "talkA_noise": ["L", "R"],
         "noise_LR": ["L", "R"], "talkC_R_quiet": ["R"], "sweep_L": ["L"], "sweep_R": ["R"], "sweep_L2": ["L"]}


def align(y, ref):
    n = 1 << int(np.ceil(np.log2(len(y) + len(ref))))
    cc = np.fft.irfft(np.fft.rfft(bandpass(y[:, 0], 200, 6000), n) * np.conj(np.fft.rfft(bandpass(ref, 200, 6000), n)), n)
    return int(np.argmax(cc[:len(y)]))


def estimate_irs(name, keys):
    y = load(name)
    s = stim(name)
    refs = [stretch(s[k], ALPHA) for k in keys]
    nref = max(len(r) for r in refs)
    # common integer lag: strongest reference
    lag = align(y, refs[int(np.argmax([np.sum(r ** 2) for r in refs]))])
    lag -= 200  # leave room for acausal part / pre-arrival
    X = np.zeros((len(y), len(keys)))
    for i, r in enumerate(refs):
        seg = r[:max(0, min(len(r), len(y) - lag))]
        X[lag:lag + len(seg), i] = seg
    # active stretch: reference active region + 0.7 s tail
    a0 = lag + int(0.2 * FS)
    a1 = min(len(y), lag + nref + int(0.7 * FS))
    w = np.hanning(NF + 2)[1:-1]
    Sxx = np.zeros((NF // 2 + 1, len(keys), len(keys)), complex)
    Syx = np.zeros((NF // 2 + 1, 2, len(keys)), complex)
    for st in range(a0, a1 - NF, HOP):
        Xf = np.fft.rfft(X[st:st + NF] * w[:, None], axis=0)      # (nf, k)
        Yf = np.fft.rfft(y[st:st + NF] * w[:, None], axis=0)      # (nf, 2)
        Sxx += Xf[:, :, None] * np.conj(Xf[:, None, :])
        Syx += Yf[:, :, None] * np.conj(Xf[:, None, :])
    reg = 1e-6 * np.trace(Sxx, axis1=1, axis2=2).real.max()
    Hf = np.einsum("fck,fkj->fcj", Syx, np.linalg.inv(Sxx + reg * np.eye(len(keys))[None]))  # (nf, 2, k)
    h = np.fft.irfft(Hf, NF, axis=0)  # (NF, 2, k); time 0 = lag (i.e. 200 samples before the aligned onset)
    return h, lag


def analyse_ir(hk):
    """hk: (NF, 2). Find direct peak, compute TDOAs."""
    e = np.abs(hk).sum(1)
    p = int(np.argmax(e[:4000]))
    out = {"peak_idx": p}
    for t0, t1 in [(-1, 1.5), (-2, 4)]:
        i0, i1 = p + int(t0 * 16), p + int(t1 * 16)
        seg = hk[i0:i1] * np.hanning(i1 - i0 + 2)[1:-1, None]
        out[f"gcc_300_6500_{t0}_{t1}ms"] = r3(gcc_phat(seg[:, 0], seg[:, 1], lo=300, hi=6500))
        A, B = np.fft.rfft(seg[:, 0], 4096), np.fft.rfft(seg[:, 1], 4096)
        f = np.fft.rfftfreq(4096, 1 / FS)
        P = A * np.conj(B)
        out[f"slope_{t0}_{t1}ms"] = r3(phase_slope(f, P / (np.abs(P) + 1e-30), w=np.abs(P), lo=300, hi=6500)[0])
    # reverberant window 20-200 ms: frequency-smoothed coherence phase slope
    seg = hk[p + 320:p + 3200]
    A, B = np.fft.rfft(seg[:, 0], 16384), np.fft.rfft(seg[:, 1], 16384)
    f = np.fft.rfftfreq(16384, 1 / FS)
    k = 256
    sm = lambda v: np.convolve(v, np.ones(k) / k, "same")
    Pab = sm((A * np.conj(B)).real) + 1j * sm((A * np.conj(B)).imag)
    out["reverb_20_200ms_slope"] = r3(phase_slope(f, Pab / np.abs(Pab), w=np.abs(Pab), lo=300, hi=6500)[0])
    # direct energy and DRR (2.5 ms split), mic A
    d = hk[p - 16:p + 24, 0]
    rv = hk[p + 24:p + 9600, 0]
    out["direct_energy_dB"] = r3(10 * np.log10(np.sum(d ** 2)))
    out["drr_dB"] = r3(10 * np.log10(np.sum(d ** 2) / np.sum(rv ** 2)))
    # sub-sample arrival time of the direct sound at mic A: x8-upsampled envelope peak
    from scipy.signal import resample
    up = resample(hk[p - 32:p + 32, 0], 64 * 8)
    out["peak_frac_A"] = r3(p - 32 + np.argmax(np.abs(up)) / 8)
    up = resample(hk[p - 32:p + 32, 1], 64 * 8)
    out["peak_frac_B"] = r3(p - 32 + np.argmax(np.abs(up)) / 8)
    # onset: first time the envelope (A) exceeds -12 dB re its peak
    env = np.abs(hk[:p + 1, 0])
    out["onset_-12dB_A"] = int(np.argmax(env > env[p] * 10 ** (-12 / 20)))
    return out


def blind_tdoa(name, seg_s):
    """S1-style blind TDOA: per frame PHAT steered response (200-3500 Hz) on frames with high broadband MSC."""
    y = load(name)
    x = y[int(seg_s[0] * FS):int(seg_s[1] * FS)]
    f, A, B = frames_spec(x, 512, 128)
    m = (f >= 200) & (f <= 3500)
    lam = 0.85
    Pab = np.zeros(m.sum(), complex)
    Paa = np.zeros(m.sum())
    Pbb = np.zeros(m.sum())
    taus = np.arange(-2.5, 1.0, 0.01)
    E = np.exp(-2j * np.pi * np.outer(f[m], taus) / FS)
    est, lev, msc_l = [], [], []
    for i in range(len(A)):
        a, b = A[i, m], B[i, m]
        Pab = lam * Pab + (1 - lam) * a * np.conj(b)
        Paa = lam * Paa + (1 - lam) * np.abs(a) ** 2
        Pbb = lam * Pbb + (1 - lam) * np.abs(b) ** 2
        g = Pab / np.sqrt(Paa * Pbb)
        msc = np.abs(g) ** 2
        W = np.clip(msc / (1 - msc + 0.05), 0, 20)
        P = np.real((W * g) @ E) / W.sum()
        est.append(taus[int(np.argmax(P))])
        lev.append(10 * np.log10(np.mean(np.abs(a) ** 2) + 1e-30))
        msc_l.append(msc.mean())
    est, lev = np.array(est), np.array(lev)
    keep = lev > np.percentile(lev, 50)
    h, edges = np.histogram(est[keep], bins=np.arange(-2.5, 1.0, 0.05))
    order = np.argsort(h)[::-1][:3]
    return {"median_loud_half": r3(np.median(est[keep])),
            "hist_top3_bin_centres": [r3(edges[i] + 0.025) for i in order],
            "p10_p90_loud_half": [r3(np.percentile(est[keep], 10)), r3(np.percentile(est[keep], 90))]}


def main():
    res = {}
    for name, keys in CASES.items():
        h, lag = estimate_irs(name, keys)
        r = {"keys": keys, "lag": lag}
        for i, k in enumerate(keys):
            r[k] = analyse_ir(h[:, :, i])
        if len(keys) == 2:
            # right-speaker arrival minus left-speaker arrival (samples) at mic A and B, and r_R - r_L in cm
            dA = r["R"]["peak_frac_A"] - r["L"]["peak_frac_A"]
            dB = r["R"]["peak_frac_B"] - r["L"]["peak_frac_B"]
            don = r["R"]["onset_-12dB_A"] - r["L"]["onset_-12dB_A"]
            r["arrival_R_minus_L_samples_[A,B,onsetA]"] = [r3(dA), r3(dB), int(don)]
            r["rR_minus_rL_cm_from_peakA"] = r3(dA / FS * 343 * 100)
        if name.startswith("talk") or name.startswith("noise"):
            on = lag + 200
            r["blind_active"] = blind_tdoa(name, ((on + int(0.25 * FS)) / FS, (on + int(4.45 * FS)) / FS))
        res[name] = r
        print(name, json.dumps(r))
    with open(os.path.join(HERE, "s3_speech.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
