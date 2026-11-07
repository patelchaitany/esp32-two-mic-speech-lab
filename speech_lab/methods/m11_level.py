"""M11 standalone (DESIGN.md M11): Mic A after S0's high-pass, leveled by methods._core.agc.

The S0 + S1 front end (tracker.front_end) supplies the talker cues. Meeting: S1 VAD, speech segments and
talker clusters; offline every S1 speech segment gets the static gain of its own (noise-subtracted)
level, online the S1 clusters key the gain memory; at most 6 dB above the room-average gain, no
off-attenuation. Bus: one gated AGC whose level updates only when p_A + p_B > 0.8 under the DESIGN
noise cap (output noise <= -50 dBFS), which on raw bus input holds the gain at about 0 dB. Both end in
the -1 dBTP limiter. No noise suppression runs here; stacks call agc.level() on their enhanced output
instead. Optional info keys: 'online' (causal front end and AGC), 'axis_along_table' (|tau|-bin
starting gains for new keys, DESIGN 5).
"""
import numpy as np

from methods._core import agc, tracker

ALGO_LATENCY_MS = 21.0   # online: frame gains act at frame centres (16 ms ahead) + 5 ms limiter look-ahead


def enhance(x, fs, info, return_stages=False):
    online = bool(info.get("online", False))
    scenario = info.get("scenario", "bus")
    fe = tracker.front_end(np.asarray(x, dtype=np.float64), fs, info, online=online)
    res = agc.level(fe.x[0], fe.track, scenario, Y=fe.X[0], online=online,
                    axis_along_table=bool(info.get("axis_along_table", False)))
    return {"out": res.out, "pre_agc": res.pre} if return_stages else res.out
