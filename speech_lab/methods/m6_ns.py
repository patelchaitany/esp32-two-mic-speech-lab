"""M6 standalone: Mic A -> S0 high-pass -> MMSE-SPP noise tracking + LSA gain (DESIGN.md M6).

Single channel, tuned ns.PARAMS (xi_H1 5 dB, alpha 0.97, alpha_dd 0.7), plain tracker in both
scenarios: no spatial cue standalone on the bus (the DESIGN cue, band mean of G_cdr * G_fence > 0.6,
fires in 0-3 % of partner frames on v2 dev scenes, and even an oracle hold above 500 Hz changed
nothing), no pause gate in the meeting (it measured worse than the plain tracker). Gain floor -15 dB
(bus) / -12 dB (meeting). The high-pass is S0 step 1 (calib.prefilter: 2nd-order Butterworth, 120 Hz
bus / 80 Hz meeting). The initial noise PSD looks 2 s ahead (offline); stacks call
methods._core.ns.suppress() on their linear output.
"""
from methods._core import calib, common, ns

ALGO_LATENCY_MS = 32.0   # 512-point STFT; the NS stage adds no delay


def enhance(x, fs, info, return_stages=False):
    assert fs == common.FS, "M6 expects 16 kHz input"
    scenario = info.get("scenario", "bus")
    x_a = calib.prefilter(x[0], scenario)
    y = common.istft(ns.suppress(common.stft(x_a), scenario, fs=fs)["S"], x_a.shape[-1])
    return {"out": y, "pre_agc": y} if return_stages else y
