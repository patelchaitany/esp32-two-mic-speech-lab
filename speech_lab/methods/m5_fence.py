"""M5 standalone: S0 + S1 front end -> two-hypothesis phase fence on Mic A (DESIGN.md M5, 2.1).

Bus: hypotheses = S1's two partner TDOAs; in 1 kHz - min(4 kHz, 0.8 c/2 d_hat) the gain is
max(0.5, max_k 1/(1 + 16 e_k^2)), outside that band 1 (tuned masks.FENCE_PARAMS; DESIGN start values
gamma 4, floor 0.3, band median outside: methods/_core/masks.py); in frames where S1 finds the partners
spatially inseparable the fence is off (gain 1; DESIGN 2.1 fallback).
Meeting (ablation only; DESIGN gives no rule): one hypothesis, the TDOA of the active S1 talker cluster
(held through pauses), same parameters.
Then the DESIGN 2.1 time rule (instant rise, fall <= 4.8 dB per hop), applied to the S0-prefiltered Mic A.
info['online'] = True runs the causal front end.
The fence is only as good as S1's partner TDOAs: on v2 dev bus (master/slave scoring, 90 scenes) it lowers
worst-partner dESTOI by 0.004 against M0 with S1's TDOAs and raises it by 0.002 with the true TDOAs
(diagnostic), while rejecting off-cone transients (dSNR_active +0.5 dB / +1.0 dB).
"""
import numpy as np

from methods._core import common as cm, masks, ns, tracker


def enhance(x, fs, info, return_stages=False):
    bus = info.get("scenario", "bus") == "bus"
    fe = tracker.front_end(np.asarray(x, dtype=np.float64), fs, info, online=bool(info.get("online", False)))
    cal = fe.calib if fe.schedule is None else fe.schedule
    taus = masks.partner_taus(fe.track, cal) if bus else masks.talker_taus(fe.track)
    G = ns.limit_gain(masks.fence_gain(fe.X, cal, taus), floor_db=None, fall_db=masks.FALL_DB)
    y = cm.istft(G * fe.X[0], fe.x.shape[1])
    return {"out": y, "pre_agc": y} if return_stages else y
