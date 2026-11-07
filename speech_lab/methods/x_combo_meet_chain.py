"""x_combo_meet_chain (experimental stack, real:run 2026-11-03), DESIGN 2.2 order: S0 prefilter + calibration on both
mics -> two-channel WPE (methods._core.wpe offline defaults) -> S1 on the WPE output -> M2 meeting beam (steered,
WNG >= -3 dB) -> M8 GTCRN post-filter (12 dB attenuation limit) -> M11 leveling (agc.level with the S1 track).
Returns {'out': leveled, 'pre_agc': GTCRN output}. A bus scene runs the same chain with the bus parameters.
"""
import numpy as np

from methods import x_combo_common as xc
from methods._core import agc, beams, calib, common as cm, nn, tracker, wpe

ATTEN_DB = {"bus": 20.0, "meeting": 12.0}


def front(x, fs, info):
    scen = info.get("scenario", "meeting")
    x = np.asarray(x, np.float64)
    info = xc.gn_fix(x, fs, info)
    xh = calib.prefilter(x, scen)
    X0 = cm.stft(xh)
    cal = calib.calibrate(X0, info)
    X = calib.apply(X0, cal)
    Xw = wpe.wpe_offline(X, fs=fs)
    trk = tracker.track(Xw, cal, info, x=xh[0])
    return scen, xh, cal, Xw, trk


def enhance(x, fs, info, return_stages=False):
    T = np.shape(x)[-1]
    scen, xh, cal, Xw, trk = front(x, fs, info)
    beam = beams.meeting_beam(Xw, cal, trk, scenario=scen, mode="steered")
    yb = cm.istft(beam.apply(Xw), T)
    y_ns = nn.post_filter(yb, "gtcrn", atten_db=ATTEN_DB.get(scen, 12.0))["y"]
    res = agc.level(y_ns, trk, scen, Y=cm.stft(y_ns))
    return xc.stages(res.out, res.pre, return_stages)
