"""Analyse the speaker -> ESP32 recordings (hw/rec) against the sounds that were played (hw/stim).

Writes hw/analysis.json and hw/irs.npz (room responses, ch0 = Mic A, ch1 = Mic B).
Run: .venv/bin/python -I hw/analyze.py

Conventions: TDOA tau = arrival at Mic B minus arrival at Mic A (tau < 0: B hears first), as in DESIGN.md.
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, csd, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 16000
BANDS = [(150, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]
_HP = butter(4, 80, "high", fs=FS, output="sos")


def load(name):
    x, fs = sf.read(os.path.join(HERE, "rec", name + ".wav"), dtype="float64")
    assert fs == FS
    return sosfiltfilt(_HP, x - x.mean(0), axis=0)


def stim(name):
    return {k: v.astype(np.float64) for k, v in np.load(os.path.join(HERE, "stim", name + ".npz")).items()}


def bandpass(x, lo, hi):
    return sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), x, axis=0)


def db(p):
    return round(float(10 * np.log10(max(p, 1e-20))), 1)


def nfft(n):
    return 1 << int(np.ceil(np.log2(n)))


def xcorr_lag(y, r):
    """Integer lag of reference r inside recording y (plain cross-correlation)."""
    n = nfft(len(y) + len(r))
    cc = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(r, n)), n)
    return int(np.argmax(cc[:len(y)]))


def frac_shift(x, tau):
    """Delay x (n,) or (n, c) by tau samples (fractional, circular on a padded FFT)."""
    n = nfft(len(x) + 256)
    f = np.fft.rfftfreq(n)
    ph = np.exp(-2j * np.pi * f * tau)
    X = np.fft.rfft(x, n, axis=0)
    return np.fft.irfft(X * (ph[:, None] if x.ndim == 2 else ph), n, axis=0)[:len(x)]


def phase_delay(a, b, lo=200, hi=4000):
    """Delay of b relative to a in samples (|tau| < 4): A conj(B) = |.| exp(+j 2 pi f tau) when b lags a by tau."""
    n = nfft(len(a))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1 / FS)
    m = (f > lo) & (f < hi)
    G = A[m] * np.conj(B[m])
    taus = np.arange(-4, 4, 0.01)
    score = [np.real(np.sum(G * np.exp(-2j * np.pi * f[m] * t / FS))) for t in taus]
    return float(taus[int(np.argmax(score))])


def gcc_phat_tdoa(a, b, maxlag=8.0, up=64):
    """tau = delay of b relative to a, in samples, by GCC-PHAT with 1/up resolution."""
    n = nfft(len(a) + len(b)) * 2
    G = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    G /= np.abs(G) + 1e-12
    cc = np.fft.irfft(G, n * up)
    L = int(maxlag * up)
    cc = np.concatenate([cc[-L:], cc[:L + 1]])
    return float((np.argmax(cc) - L) / up)


def deconvolve(y, x, eps=1e-4):
    n = nfft(len(y) + len(x))
    X = np.fft.rfft(x, n)
    Y = np.fft.rfft(y, n, axis=0)
    H = Y * np.conj(X)[:, None] / (np.abs(X)[:, None] ** 2 + eps * np.max(np.abs(X)) ** 2)
    return np.fft.irfft(H, n, axis=0)[:len(y)]


def stretch(x, alpha):
    """Resample x as if played on a clock (1 + alpha) relative to the recorder (band-limited, 8x linear)."""
    from scipy.signal import resample_poly
    up = resample_poly(x, 8, 1)
    t = np.arange(int(len(x) * (1 + alpha))) / (1 + alpha) * 8
    return np.interp(t, np.arange(len(up)), up)


def lag_track(y, r, win=0.25, hop=0.125, up=64):
    """Fractional lag of successive windows of r inside y (GCC-PHAT). Returns (times, lags)."""
    L0 = xcorr_lag(y, r)
    w, h = int(win * FS), int(hop * FS)
    out = []
    for s in range(0, len(r) - w, h):
        seg = r[s:s + w]
        if np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        ys = y[L0 + s - 64:L0 + s + w + 64]
        G = np.fft.rfft(ys, 4096) * np.conj(np.fft.rfft(seg, 4096))
        G /= np.abs(G) + 1e-9
        c = np.fft.irfft(G, 4096 * up)
        out.append((s / FS, np.argmax(c[:128 * up]) / up - 64))
    t, lag = np.array(out).T
    return t, lag


def schroeder_t20(h, lo, hi):
    """T20-based RT60 (s) of one IR channel in a band; Schroeder integral truncated at the noise floor."""
    e = bandpass(h, lo, hi) ** 2
    nf = np.mean(e[-int(0.1 * FS):])
    e = np.clip(e - nf, 0, None)
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-20)
    i5, i25 = np.argmax(edc_db < -5), np.argmax(edc_db < -25)
    if i25 <= i5:
        return None
    slope = np.polyfit(np.arange(i5, i25) / FS, edc_db[i5:i25], 1)[0]
    return round(float(-60 / slope), 3)


def main():
    res = {}

    # A. ambient: level per band, low-frequency noise, coherence and direction of the room/fan noise
    amb = load("ambient")[FS:11 * FS]
    raw, _ = sf.read(os.path.join(HERE, "rec", "ambient.wav"), dtype="float64")
    raw = raw - raw.mean(0)
    lf = bandpass(raw, 2, 60)
    f, Pab = csd(amb[:, 1], amb[:, 0], FS, nperseg=2048)  # scipy: conj(B) A = A conj(B)
    _, Paa = csd(amb[:, 0], amb[:, 0], FS, nperseg=2048)
    _, Pbb = csd(amb[:, 1], amb[:, 1], FS, nperseg=2048)
    msc = np.abs(Pab) ** 2 / (Paa.real * Pbb.real)
    m = (f > 200) & (f < 4000)
    taus = np.arange(-4, 4, 0.01)
    score = [np.real(np.sum(Pab[m] * np.exp(-2j * np.pi * f[m] * t / FS))) for t in taus]
    res["ambient"] = {
        "rms_dbfs_after_80Hz_hpf": [db(np.mean(amb[:, c] ** 2)) for c in range(2)],
        "band_dbfs_micA": {f"{lo}-{hi}": db(np.mean(bandpass(amb[:, 0], lo, hi) ** 2)) for lo, hi in BANDS},
        "below_60Hz_dbfs": [db(np.mean(lf[:, c] ** 2)) for c in range(2)],
        "msc_by_band": {f"{lo}-{hi}": round(float(msc[(f >= lo) & (f < hi)].mean()), 2) for lo, hi in [(20, 60)] + BANDS},
        # phase of A conj(B) = +2 pi f tau, tau = delay of B re A
        "noise_tdoa_samples": round(float(taus[int(np.argmax(score))]), 2),
    }

    # B. clock drift between the Mac's output and the ESP32 (talker A on the left speaker, Mic A;
    #    the played bus noise is below the fan noise, so it cannot be tracked)
    y = load("talkA_L")[:, 0]
    r = bandpass(stim("talkA_L")["A"], 150, 7000)
    t, lag = lag_track(bandpass(y, 150, 7000), r)
    keep = np.abs(lag - np.median(lag)) < 2
    slope = np.polyfit(t[keep], lag[keep], 1)[0]
    res["clock"] = {"drift_ppm": round(float(slope / FS * 1e6), 1),
                    "max_window_step_samples": round(float(np.abs(np.diff(lag[keep])).max()), 2),
                    "windows_used": int(keep.sum()), "windows": int(len(lag))}
    alpha = slope / FS

    # C. room responses from the sweeps (reference stretched by the measured drift)
    sweep = stretch(stim("sweep_L")["L"], alpha)
    irs, ir_info = {}, {}
    for name in ["sweep_L", "sweep_R", "sweep_L2"]:
        h = deconvolve(load(name), sweep)
        p = int(np.argmax(np.abs(h).sum(1)))
        ir = h[p - 80:p + int(0.6 * FS)]
        irs[name] = ir
        d = ir[80 - 16:80 + 48]
        n = 4096
        A, B = np.fft.rfft(d[:, 0], n), np.fft.rfft(d[:, 1], n)
        fr = np.fft.rfftfreq(n, 1 / FS)
        direct = slice(80 - 40, 80 + 40)
        ir_info[name] = {
            "onset_s_after_rec_start": round((p - 80) / FS - 0.25, 3),
            "direct_tdoa_samples": round(gcc_phat_tdoa(d[:, 0], d[:, 1]), 2),
            "direct_level_B_re_A_db": {f"{lo}-{hi}": db(np.sum(np.abs(B[(fr >= lo) & (fr < hi)]) ** 2) / np.sum(np.abs(A[(fr >= lo) & (fr < hi)]) ** 2)) for lo, hi in BANDS},
            "snr_db_by_band": {f"{lo}-{hi}": db(np.mean(bandpass(ir[:, 0], lo, hi)[:1600] ** 2) / np.mean(bandpass(ir[:, 0], lo, hi)[int(0.45 * FS):] ** 2)) for lo, hi in BANDS},
            "rt60_s_micA": {f"{lo}-{hi}": schroeder_t20(ir[:, 0], lo, hi) for lo, hi in [(250, 500), (500, 1000), (1000, 2000), (2000, 4000)]},
            "drr_db_micA": db(np.sum(ir[direct, 0] ** 2) / np.sum(ir[80 + 40:, 0] ** 2)),
        }
    a, b = irs["sweep_L"], irs["sweep_L2"]
    shift = phase_delay(a[:4880, 0], b[:4880, 0])
    bs = frac_shift(b, -shift)
    g = np.sum(a[:4880] * bs[:4880]) / np.sum(bs[:4880] ** 2)
    rep = {}
    for seg, (s0, s1) in {"0-5ms": (0, 160), "5-50ms": (160, 880), "50-300ms": (880, 4880)}.items():
        rep[seg] = db(np.sum((a[s0:s1] - g * bs[s0:s1]) ** 2) / np.sum(a[s0:s1] ** 2))
    res["room_responses"] = ir_info
    res["repeatability_sweepL_vs_L2_residual_db"] = rep
    np.savez(os.path.join(HERE, "irs.npz"), fs=FS, **irs)

    # D. superposition: does a mixture recording equal the sum of its separately recorded parts?
    def onset(rec, key):
        return xcorr_lag(bandpass(load(rec)[:, 0], 150, 7000), bandpass(stim(rec)[key], 150, 7000))

    sup = {}
    for mix, parts in {"talkAB_LR": [("talkA_L", "A"), ("talkB_R", "B")],
                       "talkA_noise": [("talkA_L", "A"), ("noise_LR", "N_L")]}.items():
        ym = load(mix)
        comps = []
        for rec, key in parts:
            c = np.roll(load(rec), onset(mix, key) - onset(rec, key), axis=0)
            act = slice(int(0.5 * FS), int(4.5 * FS))
            sh = phase_delay(ym[act, 0], c[act, 0])
            comps.append(frac_shift(c, -sh))
        act = slice(int(0.45 * FS), int(4.6 * FS))
        M = np.stack([c[act].ravel() for c in comps], 1)
        gains, *_ = np.linalg.lstsq(M, ym[act].ravel(), rcond=None)
        e = ym[act] - sum(gi * c[act] for gi, c in zip(gains, comps))
        sup[mix] = {"gains": [round(float(v), 3) for v in gains],
                    "residual_db_by_band": {f"{lo}-{hi}": db(np.sum(bandpass(e, lo, hi) ** 2) / np.sum(bandpass(ym[act], lo, hi) ** 2)) for lo, hi in BANDS + [(150, 7000)]}}
    res["superposition"] = sup

    # E. how far each played sound rises above the room/fan noise at the mics
    amb_p = {(lo, hi): np.mean(bandpass(amb, lo, hi) ** 2) for lo, hi in BANDS + [(150, 7000)]}
    sar = {}
    for rec, key in [("talkA_L", "A"), ("talkB_R", "B"), ("talkC_R_quiet", "C"), ("noise_LR", "N_L")]:
        on = onset(rec, key)
        s = load(rec)[on + int(0.25 * FS):on + int(4.45 * FS)]
        sar[rec] = {f"{lo}-{hi}": db(max(np.mean(bandpass(s, lo, hi) ** 2) - p, 1e-15) / p) for (lo, hi), p in amb_p.items()}
    res["sound_to_ambient_db"] = sar

    with open(os.path.join(HERE, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
