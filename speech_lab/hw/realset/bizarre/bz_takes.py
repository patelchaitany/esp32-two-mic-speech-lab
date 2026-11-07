"""Per-take composite responses (dry speech -> each mic) for the realset takes, incl. the emulated far talkers.

Near takes: played = dry, so the dry->mic response is the fitted 4096-tap h.
Far takes (MF1/MF2): played = g * (dry * h_far) (hw/realset.py far_rir(seed=7), 4 m, RT60 0.8 s, re-generated here
and its gain g fitted against the stored 'played'), so the dry->mic response is g * (h_far * h).
Everything is aligned like realscenes.take_parts: the stretched (17 ppm) signal starts at n0 = onset - 256 - 0.25 s.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from bz_common import FS, REALSET, load_take  # noqa: E402

LEAD = int(0.25 * FS)
TAKES = ["A1", "A2", "A3", "B1", "B2", "B3", "AE1", "AE2", "BE1", "BE2", "M1", "M2", "M3", "M4", "M5", "M6",
         "MF1", "MF2"]
_HFAR = None


def hfar():
    global _HFAR
    if _HFAR is None:
        import realset as rs        # hw/realset.py (numpy/pyroomacoustics only; no hardware access in far_rir)
        _HFAR = rs.far_rir(seed=7)
    return _HFAR


def composite(t):
    """dict: dry_pad (stretched dry incl. 0.25 s lead), comp (K, 2) dry->mic response, p (direct index in comp),
    n0, image (2, N), noise (2, N), recording (2, N), far (bool), fit_err_db (played rebuild), rebuild_err_db."""
    from analyze import stretch
    z = load_take(t)
    st = np.load(os.path.join(REALSET, "stim", t + ".npz"))
    played, dry = st["played"].astype(np.float64), st["dry"].astype(np.float64)
    h = z["h"].astype(np.float64)
    far = t.startswith("MF")
    out = {"far": far}
    if far:
        hf = hfar()
        c = np.convolve(dry[LEAD:], hf)[:len(played) - LEAD]
        m = slice(0, int(3.0 * FS))                     # before the fade-out
        g = float(np.dot(played[LEAD:][m], c[m]) / np.dot(c[m], c[m]))
        out["fit_err_db"] = float(10 * np.log10(np.sum((played[LEAD:][m] - g * c[m]) ** 2) / np.sum(played[LEAD:][m] ** 2)))
        comp = np.stack([np.convolve(g * hf, h[:, k]) for k in range(2)], 1)
        out["p_far"] = int(np.argmax(np.abs(hf[:400])))
    else:
        comp = h
        dry = played
    p_h = int(np.argmax(np.abs(h).sum(1)))
    out["p"] = p_h + (out["p_far"] if far else 0)
    onset = int(z["onset"])
    n0 = onset - 256 - LEAD
    N = z["image"].shape[0]
    dry2 = stretch(dry, 17e-6)
    out.update(dry_pad=dry2, comp=comp, h=h, p_h=p_h, n0=n0, onset=onset,
               image=z["image"].T.astype(np.float64), noise=z["noise"].T.astype(np.float64),
               recording=z["recording"].T.astype(np.float64), N=N)
    full = render(out, comp.shape[0])[0]
    img = out["image"][0]
    sl = slice(onset, min(N, onset + int(4.0 * FS)))
    out["rebuild_err_db"] = float(10 * np.log10(np.sum((full[sl] - img[sl]) ** 2) / np.sum(img[sl] ** 2)))
    return out


def render(tk, upto, start=0):
    """dry_pad convolved with comp[start:upto] (both mics), placed at n0 -> (2, N)."""
    y = np.zeros((2, tk["N"]))
    for k in range(2):
        seg = np.convolve(tk["dry_pad"], tk["comp"][start:upto, k])[:max(0, tk["N"] - tk["n0"])]
        y[k, tk["n0"]:tk["n0"] + len(seg)] = seg
    return y


def drr_c50(resp, p, half=32):
    """DRR (direct = [p-half, p+half)) and C50 (early = [.., p+800)) of a 1-D response, dB."""
    e = resp ** 2
    d = e[max(0, p - half):p + half].sum()
    early = e[:p + 800].sum()
    return float(10 * np.log10(d / max(e.sum() - d, 1e-30))), float(10 * np.log10(early / max(e[p + 800:].sum(), 1e-30)))


def speech_iv(tk, extra_s=0.3):
    """Interval of the take's speech (+ tail) in recording samples."""
    n = len(np.trim_zeros(tk["dry_pad"][LEAD:], "b"))
    a = tk["n0"] + LEAD + tk["p"]
    return [(a, min(tk["N"], a + n + int(extra_s * FS)))]
