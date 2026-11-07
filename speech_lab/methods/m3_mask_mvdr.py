"""M3 mask-driven variant, offline: S0 -> S1 -> mask-weighted covariances -> Souden MVDR -> ISTFT (DESIGN.md M3).

Bus. Masks from S1: m_k = p_k * CDR_k / (1 + CDR_k) per bin (DOA-dependent CDR toward partner k, M4
conditioning). Noise covariance: the hybrid model Gamma_n + sigma_s I, as in m3_rtf_mvdr (deviation: DESIGN's
mask-weighted measured Phi_n leaked partner speech with S1's masks and lost worst-partner ESTOI on dev).
Per-partner beams in the Souden form h = (Phi_n + mu I)^-1 Phi_s e_A / tr(.) under the WNG floors,
re-referenced to the array centre; the AB beam, states, crossfades and the fallback ('auto': Mic A unless both
partners are near broadside) as in m3_rtf_mvdr. Meeting: Mic A (M3's meeting mode is off by default).
M7/M8/M9 masks can be passed to methods._core.rtf.run(masks=...).
"""
from methods._core import rtf

MASK = dict(labels="mask", mask_cue="post*cdr", noise="hybrid", beam="souden")


def enhance(x, fs, info, return_stages=False):
    return rtf.enhance(x, fs, info, online=False, return_stages=return_stages, **MASK)
