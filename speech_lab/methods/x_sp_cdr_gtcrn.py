"""Spacing-study stack (x_, study/SPACING_STUDY.md): M4 CDR post-filter on Mic A, then M8 GTCRN on its output.

Spatial stage = methods.m4_cdr unchanged (S0 front end, DOA-free CDR gain from the two mics' coherence, DESIGN
time rule); neural stage = methods._core.nn.denoise(..., 'gtcrn') with the DESIGN M8 attenuation limit
(20 dB bus, 12 dB meeting) and per-file level normalisation, exactly as m8_gtcrn applies it to Mic A.
Purpose: does a neural single-channel post-filter make the array spacing irrelevant? Compare with m8_gtcrn
(the same model on Mic A alone) at 12 and 40 mm. No AGC, so 'pre_agc' equals 'out'.
"""
import numpy as np

from methods import m4_cdr
from methods._core import nn


def enhance(x, fs, info, return_stages=False):
    scenario = info.get("scenario", "bus")
    r = m4_cdr.enhance(x, fs, info, return_stages=True)
    y0 = r["pre_agc"] if isinstance(r, dict) else r
    y = nn.denoise(np.asarray(y0, np.float64), "gtcrn", atten_db=nn.ATTEN_DB[scenario], level="file")
    return {"out": y, "pre_agc": y} if return_stages else y
