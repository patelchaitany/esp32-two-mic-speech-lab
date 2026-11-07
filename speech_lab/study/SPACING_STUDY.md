# SPACING_STUDY: paired 12 mm vs 40 mm natural-speech study (run spacing:run, 2026-11-03)

Saved by the main session from the agent's returned report (the subagent could not write files).

The paired spacing study is complete, but study/SPACING_STUDY.md was not written. The harness first refused a .md report file from a subagent ("Subagents should return findings as text"), and the second Write attempt timed out in its PreToolUse hook. I did not work around the hook, so the report's content is here.

## Answer
- **Bus: moving from 12 mm to 40 mm buys almost nothing.**
  - Classical spatial methods (factory calibration): ΔSNR_active +0.2 to +0.5 dB and NA_active +0.3 to +1.2 dB. Worst-partner ΔESTOI moves by at most +0.004.
  - Best spatial gains: m3_rtf_mvdr +0.48 [+0.13, +0.83] dB ΔSNR; m2_bus_beam +0.42 [+0.08, +0.76] dB; m4_cdr NA +1.17 [+0.99, +1.34] dB (Holm p 0.027).
  - All of this is 10–100× below the G1 bars (worst-partner ΔESTOI +0.03, ΔSNR +1.5 dB).
- **Neural single-channel processing makes spacing practically irrelevant on the bus.**
  - GTCRN on Mic A alone: worst ΔESTOI +0.024, ΔSNR +5.2 dB at both spacings.
  - Best system, CDR → GTCRN (x_sp_cdr_gtcrn): +0.025 at 12 mm and +0.029 at 40 mm; ΔSNR +5.5 and +5.7 dB.
  - 40 mm minus 12 mm for that system: ESTOI +0.004 [−0.000, +0.008], ΔSNR +0.19 [−0.18, +0.56] dB.
  - Caveat: the GTCRN rows fail the G0 TA floor (TA_min −9.6 to −11 dB). That needs level restoration and does not depend on spacing.
- **Meeting: 40 mm buys a real but modest gain.**
  - At 12 mm the steered beam falls back to Mic A in 4/4 rooms. At 40 mm it adds mean ΔESTOI +0.022 [+0.019, +0.026], far-talker +0.023 and NA +0.9 dB.
  - The gain persists on top of GTCRN + M11. Beam stack vs the Mic A stack: mean ΔESTOI +0.070 → +0.093; worst talker +0.008 [−0.005, +0.031], not established with n = 4.
  - Level spread is M11's job and is nearly spacing-free: 2.07 → 1.94 dB, against a G1 cap of 1.5.
  - Best mean ΔESTOI at 40 mm minus best at 12 mm: +0.013 [+0.009, +0.017]. Best worst-talker ESTOI is CDR → GTCRN at both spacings (0.414 / 0.409), but its NM is above 7 dB, which fails G0.
- **Decision.**
  - Do not move the mics for the bus.
  - For the meeting, 40 mm is worth roughly the G1 mean-ESTOI bar (+0.02). Do it only if a board revision is cheap, and only with a calibration.
  - With the default blind S0 the 40 mm meeting gain disappears (−0.022 vs factory). S0 v2 (x_auto) keeps it, and even gives +0.006 mean and +0.019 worst-talker over factory at 40 mm.

## Why 12 mm fails
- **S1 cannot localize the partners.** It flags them inseparable in 9/10 layouts at 12 mm vs 6/10 at 40 mm (truth by the 25 µs rule: 6/10 vs 3/10).
- **Per-frame TDOAs collapse toward 0.** Coherent diffuse sound biases them to about 45 % of the true value at 12 mm, against 85 % at 40 mm. Example: L00 reads −5 / +3 µs against a true ∓33 µs.
- **So the S1-steered methods fall back to Mic A.** m1_das 18/20 bus scenes at 12 mm, m5_fence 18/20, the meeting beam 4/4 rooms.
- **It is mostly physics, not thresholds.** Scaling S1's µs constants by d / 4 cm (x_s1sc_*) recovers only +0.01 to +0.03 dB on the bus and +0.005 of the +0.022 meeting gap.
- **M9 self-enrollment fails everywhere: 0/20 scenes.** At 12 mm because S1 says inseparable; at 40 mm because there is less than 3 s of confident solo speech per partner in a 24 s scene. m9_voxlock2_self was therefore not run, since it would return Mic A.

## Report content for study/SPACING_STUDY.md
- **Set-up.**
  - 48 paired scenes (scenes3/spacing). All methods were run from a code snapshot frozen at 02:55 (tmp/night_spacing-run/snap_0255), so both spacings of a method use the same code. The 02:16 baseline in results3/spacing is untouched.
  - Calibration: factory (scene truth), S0 v2 blind (x_auto_*) and default-S0 blind.
  - Statistics: paired 40 − 12 mm per layout (mean over SNRs), stratified 95 % bootstrap, Wilcoxon p with Holm. n = 10 bus layouts; n = 4 meeting rooms, so meeting p ≥ 0.125.
- **Bus, worst-partner ΔESTOI 12 → 40 mm (Δ [95% CI]).**
  - m0 +0.001 → +0.000; avg −0.003 → −0.012 (−0.009 [−0.012, −0.006]).
  - m1 0 → 0; m2 −0.001 → −0.001; m3 −0.002 → −0.006 (−0.004 [−0.009, +0.001]).
  - m4 −0.001 → +0.002 (+0.003 [+0.001, +0.004]); m5 −0.002 → +0.000; m6 +0.002 → +0.001; m7 0 → −0.001.
  - m8 +0.024 → +0.026; CDR → GTCRN +0.025 → +0.029.
  - x_auto_m4_cdr (blind) −0.001 → +0.003 (+0.004 [+0.003, +0.005]).
- **Bus, ΔSNR_active 12 → 40 mm (dB).**
  - m1 +0.01 → +0.10; m2 +0.15 → +0.56; m3 +0.14 → +0.62; m4 +0.15 → +0.36; m5 −0.01 → +0.09.
  - m6 +0.81 → +0.91; m7 +0.11 → +0.41; m8 +5.24 → +5.30; CDR → GTCRN +5.49 → +5.68.
  - x_auto_m2_bus_beam +0.12 → +0.54; m2_bus_beam with default-S0 blind +0.00 → −0.03 (Mic A fallback 20/20 at 12 mm, 18/20 at 40 mm).
- **Bus per SNR (Δ 40 − 12 mm).** At 0 dB: m4 +0.004 ESTOI (Holm p 0.027), CDR → GTCRN +0.005 [+0.001, +0.009]. At −5 dB: m3 −0.007 [−0.010, −0.004] ESTOI.
- **Bus per stratum.** 2 layouts each, so anecdotal. The beamformers gain about +0.9 dB ΔSNR in broadside and lap-flat (broadside-MVDR fallback with some superdirectivity at 40 mm) and about 0 in the aligned stratum, because S1 also fails in one aligned layout at 40 mm.
- **Meeting, mean ΔESTOI 12 → 40 mm.**
  - m2_meeting_beam +0.001 → +0.023; m1 +0.001 → +0.006; m4 +0.010 → +0.020; m10_wpe +0.031 → +0.036.
  - m8 +0.072 → +0.072; CDR → GTCRN +0.080 → +0.087.
  - Mic A stack +0.070 → +0.070; beam stack +0.070 → +0.093.
  - x_auto_m2_meeting_beam (blind) +0.003 → +0.029.
- **Meeting, worst-talker ESTOI 12 → 40 mm.** m0 0.344 → 0.345; m8 0.401 → 0.396; CDR → GTCRN 0.414 → 0.409; beam stack 0.396 → 0.404.
- **Meeting, other.** Level std for m11 2.07 → 1.94 dB. NM: m6 6.3 dB, m8 6.0, CDR → GTCRN 7.4 (G0 cap 6); the M11 stacks are at 4.8–5.3.
- **Limitations.**
  - Simulated scenes with exact factory calibration, synthetic tails, no device housing.
  - Small n.
  - Method thresholds were tuned at 4 cm; no 12 mm re-tune was done, because this set uses the test split.
  - The x_sp stacks stand in for the absent bus_final / meeting_final.
  - Not run: m9 with enrollment, m8_dpdfnet, m3_mask_mvdr, online variants, x_auto for m3 and m7, extra Hagerman runs.
  - Wall time was about 2.5 h, well over the ~75 min budget, mainly because tool latency grew late in the run.

## Key numbers
- bus best system, CDR→GTCRN, worst-partner dESTOI: 12 mm +0.025, 40 mm +0.029; Δ +0.004 [-0.000,+0.008]. At SNR 0 dB: +0.040 vs +0.045, Δ +0.005 [+0.001,+0.009]
- bus best system, dSNR_active: 12 mm +5.49, 40 mm +5.68 dB; Δ +0.19 [-0.18,+0.56]. G0 TA_min fails (-10.6 / -11.2 dB)
- bus GTCRN alone (Mic A): worst dESTOI +0.024 / +0.026; dSNR +5.24 / +5.30 dB (12 / 40 mm); Δ ESTOI +0.002 [-0.000,+0.004]
- bus spatial methods, 40-12 mm dSNR_active: m3_rtf_mvdr +0.48 [+0.13,+0.83]; m2_bus_beam +0.42 [+0.08,+0.76]; m7 +0.30 [-0.04,+0.64]; m4_cdr +0.21 [-0.04,+0.46]; m1 +0.09 dB
- bus NA_active 40-12 mm: m4_cdr +1.17 [+0.99,+1.34] (Holm p 0.027); m3 +0.74 [+0.35,+1.12] (Holm p 0.051); m7 +0.71; CDR→GTCRN +0.69 [+0.41,+0.96]
- bus worst-partner dESTOI 40-12 mm, best spatial: m4_cdr +0.003 [+0.001,+0.004]; x_auto_m4_cdr +0.004 [+0.003,+0.005]; m3_rtf_mvdr -0.004; avg -0.009
- neural interaction, bus (CDR→GTCRN minus GTCRN, 40-12 mm): ESTOI +0.002 [-0.002,+0.006]; dSNR +0.13 [-0.10,+0.35] dB; NA +0.70 [+0.58,+0.83] dB
- meeting beam 40-12 mm: m2_meeting_beam mean dESTOI +0.022 [+0.019,+0.026]; far +0.023; NA +0.89; Mic A fallback 4/4 rooms at 12 mm, 0/4 at 40 mm
- meeting full stack (beam→GTCRN→M11) vs Mic A stack: 12 mm identical (beam off). 40 mm: mean dESTOI +0.093 vs +0.070 (Δ +0.022 [+0.017,+0.027]); worst-talker 0.404 vs 0.391; level std 2.07 vs 2.08 dB
- meeting best per spacing: mean dESTOI 12 mm +0.080 (CDR→GTCRN), 40 mm +0.093 (beam stack), best40-best12 +0.013 [+0.009,+0.017]; worst-talker ESTOI 0.414 / 0.409 (CDR→GTCRN, NM >7 dB); level std 2.07 / 1.94 dB (m11)
- calibration, meeting at 40 mm: default blind S0: mean dESTOI -0.022 vs factory (gain lost). S0 v2 x_auto: +0.006 [+0.003,+0.013] mean and +0.019 worst-talker vs factory
- calibration, bus: x_auto (S0 v2) within ±0.03 dB dSNR of factory. Default S0 m2_bus_beam: Mic A in 20/20 (12 mm) and 18/20 (40 mm) scenes, -0.59 [-1.07,-0.11] dB at 40 mm
- S1 partner model (bus, SNR 0): inseparable 9/10 at 12 mm vs 6/10 at 40 mm (truth 6/10, 3/10); resolved TDOA ratio τ̂/τ 0.45 vs 0.85; M9 self-enrollment 0/20
- S1 µs constants scaled by d/4 cm, at 12 mm: dSNR gain +0.01 (m1, m2), +0.03 (m3); gap to 40 mm left +0.41 (m2), +0.52 (m3) dB; meeting recovers +0.005 of +0.022 mean dESTOI
- Mic A fallback counts, bus 12 / 40 mm: m1_das 18/14, m5_fence 18/10, m2_bus_beam 1/7, x_auto_m2_bus_beam 10/10, m3/m4/m7 0/0 of 20

## Deliverables
- NOT WRITTEN (harness hook blocked/timed out the subagent .md write): /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/study/SPACING_STUDY.md. Its content is in the summary above.
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/results3/run_0255/spacing/ (764 results, 0 errors: 14 bus labels x 40 scenes, 4 x_s1sc bus labels x 20 (12 mm), 13 meeting factory labels x 8, plus x_auto_m2_meeting_beam@blind, m2_meeting_beam@blind and x_s1sc_m2_meeting_beam)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/results3/run_0255/summary.json (all table numbers: means, paired deltas, CIs, Holm p, strata, best per spacing, fallback counts, blind vs factory)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/tmp/night_spacing-run/tables_final.txt (full markdown tables from analyze.py)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/tmp/night_spacing-run/ (runw.py snapshot runner, snap_0255/ frozen methods + sha/mtime lists, analyze.py, s1sc_compare.py/.txt, s1_probe.py/.jsonl + s1_probe_summary.txt + s1_probe_tdoa_err.txt, tdoa_diag.py/.jsonl, timing.log, chain scripts and all pass logs)
- New methods (x_ prefix): /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/methods/x_sp_cdr_gtcrn.py, x_sp_meet_stack.py, x_sp_mica_stack.py, x_s1scale.py, x_s1sc_m1_das.py, x_s1sc_m2_bus_beam.py, x_s1sc_m3_rtf_mvdr.py, x_s1sc_m5_fence.py, x_s1sc_m2_meeting_beam.py

## Issues
- study/SPACING_STUDY.md was not written. The harness refused subagent report files, and the retry's PreToolUse hook timed out. The parent should save the summary content to that path.
- Wall time was about 2.5 h, against the ~75 min budget. Machine load fell from ~130 to ~3 near 03:40, but tool-call latency grew to minutes per call afterwards.
- Methods ran from a snapshot frozen at 02:55. The other run kept editing methods/ (agc.py, cacgmm.py and others), so the numbers may not match the latest code. m1_das differs from the 02:16 baseline: +0.09 vs +0.37 dB dSNR at 40 mm.
- The meeting has n = 4 rooms (Wilcoxon p ≥ 0.125, indicative CIs). Bus per-stratum results rest on 2 layouts each.
- Method thresholds were tuned at 4 cm, which favours 40 mm. Only S1's µs constants were rescaled (x_s1sc); other baked-in defaults (beams.STEER_HYST, rtf tolerance, bus WNG floors) were not, and no 12 mm re-tune was done because the set uses the test split.
- The x_sp stacks stand in for the absent bus_final / meeting_final. Bus GTCRN rows break G0 TA (-9.6 to -11 dB). Meeting m6, m8 and CDR→GTCRN break or touch the G0 NM cap of 6 dB.
- Not run: m9_voxlock2 with enrollment (costly; its spatial dependence is only the S1 guard), m9_voxlock2_self (= Mic A here, since self-enrollment fails 0/20), m8_dpdfnet, m3_mask_mvdr, online variants, x_auto for m3 and m7, extra Hagerman runs.
- Simulated scenes only: exact factory calibration, broadband B gain, synthetic diffuse tails, no housing diffraction.
- S0 v2 beats the factory calibration on the 40 mm meeting beam (+0.006 mean, +0.019 worst-talker). This is unexplained; possibly its estimated d or gamma_n differs from the true geometry in a way the beam likes. Check before trusting it.

## Next
- Save the summary as study/SPACING_STUDY.md; the parent agent can write it.
- Re-run once bus_final / meeting_final exist, with the same snapshot runner and a new snapshot. In particular the meeting stack with WPE: WPE → beam → M6/M8 → M11.
- Bus: add level restoration after GTCRN (G0 TA). The spacing question is settled for the bus; put the effort into neural / TSE (m9 with enrollment) and longer enrollment material.
- Meeting: if a board revision is possible, 4 cm buys about +0.02 mean ESTOI, +0.5 to +0.9 dB NA and better far-talker capture. Keep S0 v2 or a factory calibration; the default S0 loses the gain.
- Try a coherence-compensated TDOA estimator for S1 (subtract the diffuse cross-PSD before the phase) to fix the 12 mm TDOA collapse. Scaling the thresholds alone is not enough.
- Rebuild the meeting rooms with min_turns = 3 and more rooms for worst-talker power, and investigate why S0 v2 beats the factory calibration at 40 mm in meetings.
