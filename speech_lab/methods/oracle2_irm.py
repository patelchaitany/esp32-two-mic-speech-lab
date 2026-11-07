"""ORACLE upper bound, NOT a method: it cheats by reading the true scene components.

Ideal ratio mask on Mic A (DESIGN §0 headroom; v2 of lab/oracle.py). lab.bench2 hands the unmodified scene
to methods named oracle* only, in info['oracle'] (bench2.oracle_view). Target T = the comp2 of every
'partner' group (bus) or every 'talker' group (meeting) at Mic A, reverberation included; N = everything else
(interferers, noise, sensor) = mix − T. On the lab.dsp STFT (sqrt-Hann 512/128),
    m = (|T|² / (|T|² + |N|²)) ** 0.5,    y = ISTFT(m · X_A).
The mask comes from the unmodified scene, so every Hagerman run gets the same mask: the oracle is a fixed
linear time-varying filter, and its separated components decode exactly. Zero latency (offline, non-causal).
"""
import numpy as np

from lab.dsp import istft, stft

BETA = 0.5
TARGET_ROLES = {"bus": ("partner",), "meeting": ("talker",)}
ALGO_LATENCY_MS = 0.0


def irm(sc, roles, beta=BETA):
    """Ideal ratio mask (F, N) of the groups with `roles` against the rest of the scene, at Mic A."""
    mix = np.asarray(sc["mix"][0], np.float64)
    tgt = np.zeros_like(mix)
    for g, role in sc["groups"].items():
        if role in roles:
            tgt += np.asarray(sc["comp2"][g][0], np.float64)
    ps, pn = np.abs(stft(tgt)) ** 2, np.abs(stft(mix - tgt)) ** 2
    return (ps / (ps + pn + 1e-30)) ** beta


def enhance(x, fs, info, return_stages=False):
    if "oracle" not in info:
        raise RuntimeError("oracle2_irm needs the true scene in info['oracle'] (lab.bench2 oracle side channel)")
    m = irm(info["oracle"], TARGET_ROLES[info["scenario"]])
    y = istft(m * stft(np.asarray(x[0], np.float64)), x.shape[-1])
    return {"out": y, "pre_agc": y} if return_stages else y
