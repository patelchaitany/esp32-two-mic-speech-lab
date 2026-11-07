"""Spectrogram of a WAV as an alpha-mask PNG (gray+alpha), mel rows, for CSS mask-image. Usage: spec_png.py OUT_DIR WAV..."""
import json, os, struct, sys, zlib
import numpy as np
import soundfile as sf
def png_gray_alpha(alpha):
    h, w = alpha.shape
    px = np.zeros((h, w, 2), np.uint8); px[..., 1] = alpha
    raw = b"".join(b"\x00" + px[r].tobytes() for r in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 4, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
def spec_alpha(x, fs, width=900, rows=150, fmin=60.0, dyn=55.0):
    fmax = fs / 2; n, hop = 512, 128
    if x.ndim > 1: x = x[:, 0]
    x = np.asarray(x, np.float64)
    nfr = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / fs)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700); imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(mel(fmin), mel(fmax), rows + 1))
    M = np.zeros((nfr, rows))
    for r in range(rows):
        sel = (f >= edges[r]) & (f < edges[r + 1])
        M[:, r] = S[:, sel].mean(axis=1) if sel.any() else S[:, np.argmin(np.abs(f - 0.5 * (edges[r] + edges[r + 1])))]
    cols = np.array_split(np.arange(nfr), min(width, nfr))
    P = np.stack([M[c].max(axis=0) for c in cols], axis=1)
    db = 10 * np.log10(P + 1e-12); top = np.percentile(db, 99.7)
    a = np.clip((db - (top - dyn)) / dyn, 0, 1) ** 1.35
    return (255 * a[::-1]).astype(np.uint8)
if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); info = {}
    for p in sys.argv[2:]:
        x, fs = sf.read(p, always_2d=False)
        a = spec_alpha(x, fs); stem = os.path.splitext(os.path.basename(p))[0]
        open(os.path.join(out, stem + ".png"), "wb").write(png_gray_alpha(a))
        info[stem] = {"dur": round(len(x) / fs, 2)}
    print(json.dumps(info))
