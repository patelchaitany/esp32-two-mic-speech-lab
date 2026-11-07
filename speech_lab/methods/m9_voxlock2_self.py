"""M9 with SELF-enrollment: two-partner VoxLock extraction enrolled from the scene itself (DESIGN.md M9, bench).

OFFLINE-ONLY (module attribute OFFLINE_ONLY = True, read by lab/bench2: RTF reported, not gated). It needs the
whole file for S1 and the enrollment search; when enrollment succeeds it runs two VoxLock passes (RTF about 0.95
on one M1 Pro core at low load, as m9_voxlock2), when it fails TSE is skipped (RTF 0.006 measured on a 30-s
scene at load average 7). ALGO_LATENCY_MS = 856 ms is the VoxLock + mask
chain (24 ms look-ahead + 800 ms run + 32 ms lab STFT, tse.latency_ms()); the enrollment search itself needs
the whole file, and there is no online variant. Bus only; meeting: S0-prefiltered Mic A unchanged.
info['enroll'] is ignored here.

Self-enrollment (methods/_core/tse.self_enroll, uses the S1 tracker of methods/_core/tracker.py read-only):
1. S1 partner model (offline, whole file). If S1 flags the partners inseparable (one TDOA peak, TDOAs < 25 us
   apart, both near broadside), self-enrollment fails.
2. Per partner k: frames with S1 posterior p_k > 0.9 and no double talk; runs (gaps <= 48 ms bridged) of
   >= 0.3 s whose median frame CDR is > 0 dB (near-field gate) and >= 25 % of whose frames lie within 10 us of
   tau_k. Runs ranked by SNR (S1 level over its noise floor), best first, up to 10 s.
3. Audio = delay-and-sum beam steered to tau_k (center-referenced), the best runs concatenated and cut to the
   exactly-3-s voice sample. At least 3 s of runs are required (DESIGN); info['tse']['self_min_total_s'] may
   lower this (>= 1.5 s), the material is then tiled up to 3 s.
4. Checks with the S1 speaker model (WeSpeaker ResNet34): cosine(first 1.5 s, second 1.5 s) >= 0.4 (not when
   tiled) and cosine(sample A, sample B) < 0.5 (two different voices). Thresholds and why they differ from the
   DESIGN's 0.7 are in methods/_core/tse.py. The voice samples must not be digital silence, and both passes
   must give finite output.
If any step fails, TSE is skipped for the file (prefiltered Mic A returned; the reason is in diag).
Then as methods/m9_voxlock2.py: VoxLock passes, union mask, guard, adaptive strength. info['tse'] takes the
same keys as there; with 'online': True the method skips TSE (self-enrollment is offline only).

Diagnostics: return_stages=True gives diag['self_enrollment'] = {'ok', 'segments': {'A': [(start, end,
snr_db)], 'B': [...]} (scene samples chosen, best first), 'sims', 'reasons', 'diag'} plus 'out', 'pre_agc',
'tse', 'conds', 'front_end' as in m9_voxlock2. T = 0 returns an empty array.
"""
from methods._core import tse

OFFLINE_ONLY = True                                   # Tier 3 and a whole-file enrollment search (DESIGN 2.3)
ALGO_LATENCY_MS = tse.latency_ms()                    # 856.0: VoxLock + mask chain; the search needs the file


def enhance(x, fs, info, return_stages=False):
    return tse.enhance(x, fs, info, mode="self", return_stages=return_stages)
