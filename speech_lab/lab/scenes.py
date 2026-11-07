"""Simulate 2-mic INMP441 recordings of the two test cases.

Room acoustics: image-source method (pyroomacoustics, deterministic images -> physically correct
inter-mic coherence) up to a capped order, plus a stochastic late tail with diffuse-field coherence
(Lehmann & Johansson 2010 hybrid idea). Diffuse noise with sinc coherence (Habets & Gannot 2007).
Levels are in Pa, converted to digital with the INMP441 sensitivity; sensor self-noise and +/-1 dB
mic gain mismatch are added.
"""
import os, sys, json, argparse
import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve, butter, sosfiltfilt, sosfilt
from .common import FS, C, PA_TO_DIG, spl_to_pa, rms, active_rms, read_wav, save_json, write_wav

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(HERE, "corpus")
SCENES = os.path.join(HERE, "scenes")
FRAC = pra.constants.get("frac_delay_length") // 2  # pyroomacoustics RIR onset offset (samples)


# ---------------------------------------------------------------- helpers
def load_utt(name):
    x, _ = read_wav(os.path.join(CORPUS, name))
    return x.astype(np.float64)


def coherent_noise(n, spacing, rng, shape=None, fs=FS):
    """2-ch noise with spherically-diffuse coherence sinc(2 f d / c). shape(f) = amplitude spectrum."""
    nfft = int(2 ** np.ceil(np.log2(n)))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    g = np.sinc(2 * f * spacing / C)
    a = rng.standard_normal((2, len(f))) + 1j * rng.standard_normal((2, len(f)))
    X1 = a[0]
    X2 = g * a[0] + np.sqrt(np.maximum(1 - g ** 2, 0)) * a[1]
    amp = shape(f) if shape is not None else np.ones_like(f)
    x = np.fft.irfft(np.stack([X1, X2]) * amp, nfft)[:, :n]
    return x / rms(x[0])


def hybrid_rirs(dims, rt60, srcs, mics, rng, order=18, fs=FS):
    """Return list (per source) of (2, L) RIRs, direct-path sample index per source (mic 0)."""
    e_abs, _ = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=fs, materials=pra.Material(e_abs), max_order=order,
                       use_rand_ism=True, max_rand_disp=0.05)
    for s in srcs:
        room.add_source(s)
    room.add_microphone_array(np.array(mics).T)
    room.compute_rir()
    # time up to which the ISM is (roughly) complete
    t_x = order * min(dims) / C
    n_x = int(t_x * fs)
    L = int((rt60 * 1.1 + t_x) * fs)
    decay = 6.908 / rt60  # amplitude decay rate (60 dB over rt60)
    t = np.arange(L) / fs
    spacing = np.linalg.norm(np.array(mics[0]) - np.array(mics[1]))
    out, direct = [], []
    for si, s in enumerate(srcs):
        h = np.zeros((2, L))
        for m in range(2):
            r = room.rir[m][si][:L]
            h[m, : len(r)] = r
        # stochastic diffuse tail from n_x on, energy-matched to ISM around n_x
        w0, w1 = max(n_x - int(0.03 * fs), 0), n_x
        e_ism = np.mean(h[:, w0:w1] ** 2) + 1e-20
        tail = coherent_noise(L, spacing, rng)  # unit rms
        env = np.exp(-decay * (t - t_x))
        amp = np.sqrt(e_ism) / np.sqrt(np.mean(env[w0:w1] ** 2) + 1e-20)
        xf = np.clip((np.arange(L) - (n_x - int(0.01 * fs))) / (0.01 * fs), 0, 1)  # 10 ms crossfade
        h = h * (1 - xf) + tail * env * amp * xf
        dist = np.linalg.norm(np.array(s) - np.array(mics[0]))
        out.append(h)
        direct.append(int(round(dist / C * fs)) + FRAC)
    return out, direct


def speech_track(names, total_len, rng, start_pad, gaps=(0.3, 2.5), overlap_p=0.0):
    """Concatenate utterances with random gaps; returns signal and list of (start, end)."""
    x = np.zeros(total_len)
    t = int(start_pad * FS)
    segs = []
    for nm in names:
        u = load_utt(nm)
        if t + len(u) > total_len:
            break
        x[t: t + len(u)] += u
        segs.append((t, t + len(u)))
        t += len(u) + int(rng.uniform(*gaps) * FS)
    return x, segs


def place_array(center, spacing, azimuth):
    d = np.array([np.cos(azimuth), np.sin(azimuth), 0.0]) * spacing / 2
    c = np.array(center)
    return [list(c + d), list(c - d)]  # mic A, mic B


def finalize(name, comps, refs, segs, meta, rng, spacing, sensor_spl=33.0):
    """comps: dict of 2-ch Pa signals; refs: dict of 1-ch Pa signals at mic A. Writes .npz."""
    T = next(iter(comps.values())).shape[1]
    gain = 10 ** (rng.uniform(-1, 1, 2) / 20)  # INMP441 +/-1 dB sensitivity tolerance
    sensor = rng.standard_normal((2, T)) * spl_to_pa(sensor_spl)
    mix = sum(comps.values()) + sensor
    dig = lambda p, m=0: (p * PA_TO_DIG * gain[m]).astype(np.float32)
    mixd = np.stack([dig(mix[0], 0), dig(mix[1], 1)])
    peak = np.abs(mixd).max()
    if peak > 0.99:
        print(f"WARNING {name}: clipping (peak {peak:.2f})")
    meta["mic_gain_db"] = list(20 * np.log10(gain))
    np.savez_compressed(
        os.path.join(SCENES, name + ".npz"),
        mix=mixd,
        **{f"ref_{k}": dig(v) for k, v in refs.items()},
        **{f"comp_{k}": dig(v[0]) for k, v in comps.items()},
        segs=np.array(segs, dtype=np.int64).reshape(-1, 3),
        meta=json.dumps(meta),
    )
    write_wav(os.path.join(SCENES, name + "_mix.wav"), mixd / max(peak, 1e-9) * 0.9)
    return name


def render_sources(rirs, direct, signals, early_ms=50):
    """Convolve each source; return full 2-ch image and early (direct+early_ms) image at mic A."""
    T = len(signals[0])
    full = np.zeros((2, T))
    early = np.zeros(T)
    ne = int(early_ms * 1e-3 * FS)
    for h, d0, s in zip(rirs, direct, signals):
        for m in range(2):
            full[m] += fftconvolve(s, h[m])[:T]
        he = h[0].copy()
        he[d0 + ne:] = 0
        early += fftconvolve(s, he)[:T]
    return full, early


def bandpass_power(x):
    sos = butter(4, [150, 7000], btype="band", fs=FS, output="sos")
    return np.mean(sosfiltfilt(sos, x) ** 2)


# ---------------------------------------------------------------- bus
def make_bus(seed, snr_db, spacing=0.04, duration=32.0, orientation="random"):
    rng = np.random.default_rng(1000 + seed)
    man = json.load(open(os.path.join(CORPUS, "manifest.json")))
    from .corpus import TARGET_VOICES, BABBLE_VOICES, slug
    dims = [11.5 * rng.uniform(0.95, 1.05), 2.5, 2.3]
    rt60 = rng.uniform(0.35, 0.6)
    T = int(duration * FS)
    dev = [rng.uniform(3.0, 8.5), rng.uniform(0.7, 1.8), rng.uniform(0.9, 1.1)]
    # two talkers facing each other across the device
    phi = rng.uniform(0, 2 * np.pi)
    tal = []
    for k, ang in enumerate([phi, phi + np.pi + rng.uniform(-0.7, 0.7)]):
        r = rng.uniform(0.3, 0.8)
        p = [dev[0] + r * np.cos(ang), dev[1] + r * np.sin(ang), rng.uniform(1.1, 1.35)]
        p[1] = float(np.clip(p[1], 0.25, dims[1] - 0.25))
        tal.append(p)
    az = rng.uniform(0, 2 * np.pi) if orientation == "random" else np.arctan2(tal[0][1] - dev[1], tal[0][0] - dev[0])
    mics = place_array(dev, spacing, az)
    # passengers (babble), >= 1.0 m from device
    nb = int(rng.integers(16, 26))
    pas = []
    while len(pas) < nb:
        p = [rng.uniform(0.4, dims[0] - 0.4), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(1.1, 1.7)]
        if np.hypot(p[0] - dev[0], p[1] - dev[1]) >= 1.0:
            pas.append(p)
    engine = [dims[0] - 0.4, dims[1] / 2, 0.5]
    rirs, direct = hybrid_rirs(dims, rt60, tal + pas + [engine], mics, rng)

    # target conversation: alternate voices, Lombard level 66-70 dB SPL @1 m
    vA, vB = rng.choice(TARGET_VOICES[:12], 2, replace=False)
    lines = list(range(len([m for m in man if m["file"].startswith(slug(vA) + "__bus")])))
    sigA = np.zeros(T); sigB = np.zeros(T); segs = []
    t = int(1.0 * FS)
    for i in lines:
        v = vA if i % 2 == 0 else vB
        u = load_utt(f"{slug(v)}__bus{i:02d}.wav")
        if t + len(u) > T - int(1.0 * FS):
            break
        (sigA if v == vA else sigB)[t: t + len(u)] += u
        segs.append((t, t + len(u), 0 if v == vA else 1))
        gap = rng.uniform(0.6, 2.0) if rng.random() < 0.6 else rng.uniform(-0.3, 0.2)  # long pauses + quick turns
        t += len(u) + int(gap * FS)
    for s, lvl in ((sigA, rng.uniform(66, 70)), (sigB, rng.uniform(66, 70))):
        s *= spl_to_pa(lvl) * 4 * np.pi / (active_rms(s[s != 0]) + 1e-12)
    tgt_full, tgt_early = render_sources(rirs[:2], direct[:2], [sigA, sigB])

    # babble: each passenger talks intermittently, 60-68 dB SPL @1 m
    bab_names = [m["file"] for m in man if "__bab" in m["file"]]
    bab_sigs = []
    for k in range(nb):
        v = rng.choice(BABBLE_VOICES + TARGET_VOICES[12:])
        names = [n for n in bab_names if n.startswith(slug(v) + "__")]
        seq = list(rng.permutation(names)) * 3
        s, _ = speech_track(seq, T, rng, start_pad=rng.uniform(0, 3), gaps=(0.2, 4.0))
        if np.any(s):
            s *= spl_to_pa(rng.uniform(60, 68)) * 4 * np.pi / (active_rms(s[s != 0]) + 1e-12)
        bab_sigs.append(s)
    bab_full, _ = render_sources(rirs[2:2 + nb], direct[2:2 + nb], bab_sigs)
    # engine: slowly varying harmonic series + low-passed noise
    tt = np.arange(T) / FS
    f0 = 38 + 6 * np.sin(2 * np.pi * rng.uniform(0.02, 0.06) * tt + rng.uniform(0, 6))
    ph = 2 * np.pi * np.cumsum(f0) / FS
    eng = sum(np.sin(k * ph + rng.uniform(0, 6)) / k for k in range(1, 25))
    sos = butter(2, 300, btype="low", fs=FS, output="sos")
    eng = eng / rms(eng) + 0.7 * sosfilt(sos, rng.standard_normal(T)) / 0.05
    eng *= spl_to_pa(80) * 4 * np.pi / rms(eng)
    eng_full, _ = render_sources(rirs[-1:], direct[-1:], [eng])
    # road / tyre noise: diffuse, brownish with tyre bump near 800 Hz
    shape = lambda f: 1 / np.maximum(f, 40) ** 0.9 + 0.004 * np.exp(-((f - 800) / 400) ** 2)
    road = coherent_noise(T, spacing, rng, shape) * spl_to_pa(70)

    # set in-band SNR (targets vs everything else at mic A)
    # composition: babble 0 dB, road -4 dB, engine -8 dB (in-band 150-7000 Hz)
    pb = bandpass_power(bab_full[0])
    road *= np.sqrt(pb * 10 ** (-4 / 10) / bandpass_power(road[0]))
    eng_full *= np.sqrt(pb * 10 ** (-8 / 10) / bandpass_power(eng_full[0]))
    noise = bab_full + eng_full + road
    act = np.zeros(T, bool)
    for s0, s1, _ in segs:
        act[s0:s1] = True
    ps = bandpass_power(tgt_full[0][act]) if False else bandpass_power(tgt_full[0]) / max(act.mean(), 1e-3)
    pn = bandpass_power(noise[0])
    g = np.sqrt(ps / pn / 10 ** (snr_db / 10))
    meta = dict(scenario="bus", seed=seed, snr_db=snr_db, spacing_m=spacing, rt60=rt60, dims=dims,
                device=dev, mics=mics, array_azimuth_deg=float(np.degrees(az) % 360), talkers=tal,
                talker_dist=[float(np.linalg.norm(np.array(p) - np.array(dev))) for p in tal],
                voices=[str(vA), str(vB)], n_passengers=nb, orientation=orientation)
    name = f"bus_s{seed}_snr{snr_db:+d}_d{int(round(spacing * 1000))}mm" + ("" if orientation == "random" else "_endfire")
    return finalize(name, {"target": tgt_full, "babble": bab_full * g, "engine": eng_full * g, "road": road * g},
                    {"early": tgt_early, "rev": tgt_full[0]}, segs, meta, rng, spacing)


# ---------------------------------------------------------------- meeting
def make_meeting(seed, spacing=0.04, rt60=None, hvac_spl=None):
    rng = np.random.default_rng(2000 + seed)
    from .corpus import TARGET_VOICES, slug
    dims = [12.0 * rng.uniform(0.92, 1.08), 7.0 * rng.uniform(0.92, 1.08), 3.0]
    rt60 = rt60 if rt60 is not None else rng.uniform(0.5, 0.9)
    tc = [dims[0] / 2, dims[1] / 2]
    dev = [tc[0] + rng.uniform(-0.3, 0.3), tc[1] + rng.uniform(-0.2, 0.2), 0.78]
    az = rng.uniform(0, 2 * np.pi)
    mics = place_array(dev, spacing, az)
    # 8 m x 2 m table; 8 seats per long side, 2 per end; mouths 0.45 m from edge at 1.2 m
    seats = []
    for k in range(8):
        x = tc[0] - 3.5 + k
        seats += [[x, tc[1] - 1.45, 1.2], [x, tc[1] + 1.45, 1.2]]
    for y in (-0.5, 0.5):
        seats += [[tc[0] - 4.45, tc[1] + y, 1.2], [tc[0] + 4.45, tc[1] + y, 1.2]]
    seats = [[p[0] + rng.uniform(-0.15, 0.15), p[1] + rng.uniform(-0.1, 0.1), p[2] + rng.uniform(-0.05, 0.1)] for p in seats]
    fan = [tc[0] + rng.uniform(-1, 1), tc[1] + rng.uniform(-1, 1), 2.7]
    rirs, direct = hybrid_rirs(dims, rt60, seats + [fan], mics, rng)
    voices = list(rng.permutation(TARGET_VOICES))[:20]
    order = rng.permutation(20)  # speaking order of seats
    lines = rng.permutation(20)
    T = int((20 * 4.6 + 4) * FS)
    sigs = [np.zeros(T) for _ in range(20)]
    segs = []
    t = int(2.0 * FS)
    levels = rng.uniform(57, 66, 20)  # soft to loud talkers, dB SPL @ 1 m
    for k, si in enumerate(order):
        u = load_utt(f"{slug(voices[si])}__mtg{lines[k]:02d}.wav")
        u = u * spl_to_pa(levels[si]) * 4 * np.pi / active_rms(u)
        sigs[si][t: t + len(u)] += u
        segs.append((t, t + len(u), int(si)))
        gap = rng.uniform(-0.8, -0.3) if rng.random() < 0.15 else rng.uniform(0.2, 0.8)  # 15% overlaps
        t += len(u) + int(gap * FS)
    T = t + int(2.0 * FS)
    sigs = [s[:T] for s in sigs]
    full, early = render_sources(rirs[:20], direct[:20], sigs)
    # HVAC (diffuse, pink with low-frequency emphasis) and projector fan (point, hum + broadband)
    hvac_spl = hvac_spl if hvac_spl is not None else rng.uniform(40, 45)
    hv = coherent_noise(T, spacing, rng, lambda f: 1 / np.maximum(f, 30) ** 0.6) * spl_to_pa(hvac_spl)
    tt = np.arange(T) / FS
    fan_sig = sum(np.sin(2 * np.pi * k * 120 * tt) / k for k in (1, 2, 3)) * 0.3 + rng.standard_normal(T)
    fan_sig *= spl_to_pa(45) * 4 * np.pi / rms(fan_sig)
    fan_full, _ = render_sources(rirs[20:], direct[20:], [fan_sig])
    dist = [float(np.linalg.norm(np.array(p) - np.array(dev))) for p in seats]
    meta = dict(scenario="meeting", seed=seed, spacing_m=spacing, rt60=rt60, dims=dims, device=dev, mics=mics,
                array_azimuth_deg=float(np.degrees(az) % 360), seats=seats, talker_dist=dist,
                talker_level_db=[float(v) for v in levels], voices=[str(v) for v in voices], hvac_spl=hvac_spl)
    name = f"meeting_s{seed}_d{int(round(spacing * 1000))}mm"
    return finalize(name, {"target": full, "hvac": hv, "fan": fan_full}, {"early": early, "rev": full[0]},
                    segs, meta, rng, spacing)


def load_scene(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(z["meta"]))
    return d


SETS = {
    "dev": [("bus", s, snr, 0.04) for s in (0, 1, 2) for snr in (-5, 0, 5)]
           + [("bus", s, 0, d) for s in (0, 1) for d in (0.025, 0.06)]
           + [("meeting", s, None, 0.04) for s in (0, 1, 2)]
           + [("meeting", 0, None, d) for d in (0.025, 0.06)],
    "test": [("bus", s, snr, 0.04) for s in (10, 11) for snr in (-5, 0, 5)]
            + [("bus", 12, 0, d) for d in (0.025, 0.06)]
            + [("meeting", s, None, 0.04) for s in (10, 11)]
            + [("meeting", 12, None, 0.06)],
}


def _build(job):
    kind, seed, snr, d = job
    if kind == "bus":
        return make_bus(seed, snr, spacing=d)
    return make_meeting(seed, spacing=d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=list(SETS))
    ap.add_argument("-j", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(SCENES, exist_ok=True)
    from multiprocessing import Pool
    with Pool(a.j) as p:
        for n in p.imap_unordered(_build, SETS[a.set]):
            print("built", n, flush=True)
