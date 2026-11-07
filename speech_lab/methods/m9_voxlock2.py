"""M9 with EXPLICIT enrollment: two-partner target-speaker extraction, VoxLock x2, union mask (DESIGN.md M9).

OFFLINE-ONLY (DESIGN 2.3 rule for Tier 3; module attribute OFFLINE_ONLY = True, which lab/bench2 reads, so the
RTF is reported, not gated). Cost on the M1 Pro, one onnxruntime thread, measured on a 30-s bus scene at load
average 7: one VoxLock pass RTF 0.41-0.47, the method (two passes + S0/S1 + masks) wall RTF 0.94-0.95 (CPU
0.93-0.94). info['tse']['parallel'] runs the two passes in two threads (same output bit for bit): wall RTF 0.55
on two cores (CPU 1.04). Under the load of other agents (load average 100-200) the method measured CPU RTF 1.10
(median of 32 scenes; wall 2.6). Bus only; in the meeting the method returns the S0-prefiltered Mic A unchanged
(TSE would remove the non-enrolled talkers, DESIGN 2.2).

Algorithmic latency (ALGO_LATENCY_MS, tse.latency_ms()): 24 ms VoxLock look-ahead + one 100-frame run (800 ms,
the reference partition) + 32 ms lab STFT = 856 ms. The default call also needs the whole file: the S1 partner
model and the guard's S1 segments are offline. The online preset (below) has 96 ms (ALGO_LATENCY_ONLINE_MS).

Pipeline (bus): S0 prefilter + S1 tracker (tracker.front_end, offline) -> VoxLock pass per partner on the
prefiltered Mic A -> effective masks m_A, m_B on the lab STFT grid (512/128) -> union max(m_A, m_B) ->
DESIGN guard (S1 posteriors) -> VoiceFilter-Lite adaptive strength (mix-back of the input) ->
y = istft(m_eff X_A), time-aligned with x[0].

info keys
---------
scenario    'bus' | 'meeting' (meeting: TSE disabled, prefiltered Mic A returned)
spacing_m   prior for S0/S1 only
enroll      REQUIRED for explicit enrollment: {'A': (n,) float array, 'B': (n,) float array} (or a sequence of
            two clips, A first: list, tuple or (2, n) array), 16 kHz mono, digital full scale 1.0, any level,
            n >= 48000 (3 s; 5-10 s recommended). A few seconds of each partner speaking alone, recorded
            beforehand on the device in quiet (the user records each partner once); a 2-D clip uses its Mic A
            channel. It must be a different recording from the scene (not oracle data). A clip longer than 3 s
            gives its loudest 3 s within its first 30 s (the graphs take exactly 3 s). A clip is unusable if it is
            missing or shorter than 3 s, contains NaN/inf in its first 30 s, or is digital silence (voice sample
            below -140 dBFS); a pass whose output is not finite also makes it unusable. Then the method falls
            back to self-enrollment (methods/m9_voxlock2_self.py) unless info['tse']['fallback'] = 'none'; if
            that fails too, TSE is skipped (prefiltered Mic A returned). diag['reasons'] and
            diag['enroll_source'] show which enrollment was used.
            lab/bench2 passes only scenario and spacing_m, so on bench2 scenes this wrapper runs its
            self-enrollment fallback and scores like m9_voxlock2_self, not like explicit enrollment.
tse         optional overrides of methods._core.tse.DEFAULTS: threads (1), parallel (False: run the two passes
            in two threads, wall time ~ one pass), combine ('max' | 'sum'), strength ('adaptive' | 'full' |
            float), guard (True), prime (False | 'sample' | 'mix': run a primer through each pass first,
            see tse.PRIME), frames_per_run (None = 100, the reference partition), self_min_total_s (3.0),
            fallback ('self' | 'none'), online (False; True applies tse.ONLINE, see below)

return_stages=True returns {'out', 'pre_agc' (= out, no AGC), 'diag', 'tse' (UnionResult: masks, per-pass
estimates, strength w), 'conds' (per-partner Conditioning), 'front_end' (S0/S1, None when not computed)}.
T = 0 returns an empty array.

Online variant (DESIGN 2.3): info['tse'] = {'online': True} (= {'frames_per_run': 5, 'guard': False,
'fallback': 'none'}) with explicit enrollment runs VoxLock in 40 ms runs (the streaming partition; each pass
differs from the offline one only by the int8 per-run activation quantization, 46 dB SNR measured on the
release test call), drops the offline guard and never self-enrolls, so no S1 statistics are used: S0 high-pass,
VoxLock, union mask and the strength recursion (with its causal VAD) are all causal (tests.test_tse: the output
for a prefix of the input equals the output for the whole input, bit for bit, up to 0.25 s before the cut).
Priming, if used, happens before the call. Algorithmic latency: 24 ms VoxLock look-ahead + 32 ms lab STFT (mask
stage) + the 40 ms run buffer = 96 ms. Quality (32-scene dev validation, 16 scenes with the preset): the same as
the offline passes without the guard (paired dESTOI 0.0000 [-0.0001, +0.0001]); against the default it keeps
partners 0.35 dB lower (min TA) and competitors 1.1 dB lower. Cost per pass (CPU RTF): 1.10 with 40 ms runs,
0.71 with 80 ms, 0.55 with 160 ms, 0.41-0.43 with 800 ms (2.7x / 1.7x / 1.3x / 1x); the preset measured RTF
2.5 on one core, so it is causal but not real-time on this Mac. {'online': True, 'frames_per_run': 10,
'parallel': True} (80 ms runs, latency 136 ms) measured wall RTF 0.94 on two cores. Self-enrollment has no
online variant (it searches the whole file).
"""
from methods._core import tse

OFFLINE_ONLY = True                                   # Tier 3: RTF reported by the bench, not gated (DESIGN 2.3)
ALGO_LATENCY_MS = tse.latency_ms()                    # 856.0: default call (100-frame runs); plus offline S1
ALGO_LATENCY_ONLINE_MS = tse.latency_ms(tse.ONLINE_FRAMES_PER_RUN)   # 96.0: info['tse'] = {'online': True}


def enhance(x, fs, info, return_stages=False):
    return tse.enhance(x, fs, info, mode="explicit", return_stages=return_stages)
