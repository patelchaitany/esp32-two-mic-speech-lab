"""M4 standalone: S0 front end -> CDR coherence post-filter on Mic A (DESIGN.md M4, 2.1).

Bus and meeting (meeting: ablation only, DESIGN 2.2): DOA-free CDR (prop3) in every bin with the tuned
masks.CDR_PARAMS (lam 0.8, mu 0.5, no conditioning; methods/_core/masks.py), then the DESIGN 2.1 time
rule (instant rise, fall <= 4.8 dB per hop), applied to the S0-prefiltered Mic A.
Deviation from DESIGN 2.1 (DOA-dependent prop1 under S1's two partner TDOAs): on v2 dev that mode lowered
worst-partner dESTOI by 0.013 against DOA-free (95% CI [0.006, 0.022]; 90 scenes, master/slave scoring)
because S1's partner TDOAs are often wrong, and prop1 then rejects the partner. It stays available as
BUS_HYPOTHESES = True; with the true TDOAs (diagnostic only) it would beat DOA-free by about 0.014.
info['online'] = True runs the causal front end (S0 refreshed every 5 s).
"""
import numpy as np

from methods._core import common as cm, masks, ns, tracker

BUS_HYPOTHESES = False        # True: DESIGN 2.1 mode, prop1 under S1's partner TDOAs (DOA-free where inseparable)


def enhance(x, fs, info, return_stages=False):
    doa = BUS_HYPOTHESES and info.get("scenario", "bus") == "bus"
    fe = tracker.front_end(np.asarray(x, dtype=np.float64), fs, info, online=bool(info.get("online", False)),
                           partners=doa, talkers=False)
    cal = fe.calib if fe.schedule is None else fe.schedule
    G = masks.cdr_gain(fe.X, cal, masks.partner_taus(fe.track, cal) if doa else None)
    G = ns.limit_gain(G, floor_db=None, fall_db=masks.FALL_DB)
    y = cm.istft(G * fe.X[0], fe.x.shape[1])
    return {"out": y, "pre_agc": y} if return_stages else y
