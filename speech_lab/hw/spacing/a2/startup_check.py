"""A2 helper: does each recording start with an I2S/INMP441 start-up transient?

If every recording begins with the INMP441 start-up signature (zeros for 2^18 SCK cycles after power-up,
or a decaying burst after SCK restarts from standby), then every take is a fresh I2S start (or reboot),
and the consistency of the inter-channel skew across takes is evidence that the skew is deterministic.

Also prints, per recording, a crude broadband inter-channel lag from the coherent fan noise
(GCC-PHAT, 150-7000 Hz, on 'quiet' stretches) to see whether the skew ever jumps by about one sample.
Read-only on the recordings; writes nothing except stdout.
Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -I hw/spacing/a2/startup_check.py
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "..", "..", "rec")
NAMES = ["ambient", "sweep_L", "sweep_R", "sweep_L2", "talkA_L", "talkB_R", "talkAB_LR", "noise_LR",
         "talkA_noise", "talkC_R_quiet"]
FS = 16000


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def gcc_lag(a, b, lo=150, hi=7000, interp=64):
    """Lag of b relative to a in samples (positive = b later), GCC-PHAT with band limit and FFT interpolation."""
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    X = B * np.conj(A)
    f = np.fft.rfftfreq(nfft, 1 / FS)
    W = ((f >= lo) & (f <= hi)).astype(float)
    X = X / (np.abs(X) + 1e-20) * W
    cc = np.fft.irfft(X, nfft * interp)
    m = 8 * interp  # search +-8 samples
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    k = np.argmax(cc)
    return (k - m) / interp


def main():
    sos = butter(4, 80, "highpass", fs=FS, output="sos")
    out = {}
    for name in NAMES:
        p = os.path.join(REC, name + ".wav")
        if not os.path.exists(p):
            continue
        x, fs = sf.read(p, dtype="float64")
        assert fs == FS
        # raw start-up view (no filtering): block RMS in 5 ms blocks over the first 400 ms
        blk = 80
        nb = 80
        raw = x[: blk * nb]
        rms = np.sqrt(np.mean(raw.reshape(nb, blk, 2) ** 2, axis=1))
        zeros_lead = [int(np.argmax(np.abs(x[:, c]) > 0)) for c in range(2)]
        exact_zero_frac = [float(np.mean(x[:4000, c] == 0.0)) for c in range(2)]
        late_rms = np.sqrt(np.mean((x[FS * 6: FS * 11] - x[FS * 6: FS * 11].mean(0)) ** 2, axis=0))
        ipk = [int(np.argmax(np.abs(x[:, c] - np.median(x[:, c])))) for c in range(2)]
        # inter-channel lag on the fan noise: last 1.5 s of each take (sounds end by ~4.5 s; tails decay by ~6 s)
        y = sosfiltfilt(sos, x, axis=0)
        seg = y[int(10.0 * FS): int(11.9 * FS)]
        lag_tail = gcc_lag(seg[:, 0], seg[:, 1])
        out[name] = dict(first_nonzero_sample=zeros_lead, exact_zero_frac_first_250ms=exact_zero_frac,
                         rms_db_first_blocks_A=[round(db(v), 1) for v in rms[:12, 0]],
                         rms_db_first_blocks_B=[round(db(v), 1) for v in rms[:12, 1]],
                         rms_db_late_AB=[round(db(v), 1) for v in late_rms],
                         peak_index_AB=ipk, peak_time_s_AB=[round(i / FS, 4) for i in ipk],
                         lag_B_minus_A_samples_tail_10to12s=round(lag_tail, 3))
        print(name, json.dumps(out[name]))


if __name__ == "__main__":
    main()
