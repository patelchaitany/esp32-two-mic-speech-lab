# RESULTS_REAL: methods on the real-board sets (run real:run, 2026-11-03 02:47-04:28)

Saved by the main session from the agent's returned report (the subagent could not write files).

I ran the methods on the real-board sets (21 scenes; finished at 04:28, about 100 min of wall time instead of the planned 75). One clear result: on this 1.19 cm board, spatial filtering only helps when a fixed beam points along the mic axis at the talkers. Stacked with DPDFNet it is the best result by a wide margin. Every S1-steered method gives the same output as Mic A.

**hw/realset/RESULTS_REAL.md was not written.** The harness refused report files from this subagent ("Subagents should return findings as text"), and I removed my early stub. Please save this summary as that file. The generated tables are in tmp/night_realrun/tables.md and summary.json.

## Bus: layout L1, 6 scenes (−5/0/+5 dB × recorded/quiet fan noise), measured calibration
Worst-partner input ESTOI: 0.126 / 0.187 / 0.250 at −5 / 0 / +5 dB.

Columns: ΔESTOI = worst-partner ΔESTOI (all scenes, then −5/0/+5 dB); ΔSNR = ΔSNR during partner speech; NA = noise attenuation; pax ESTOI drop = how much less intelligible the passengers become (positive is good); TA = partner level change (min / mean); RTF cpu = process CPU time ÷ audio duration.

| method | ΔESTOI all (−5/0/+5) | ΔSNR dB (−5/0/+5) | NA | ΔSIR pax | pax ESTOI drop | TA min / mean | RTF cpu |
|---|---|---|---|---|---|---|---|
| m0_mica_hpf | +0.001 | +0.0 | +0.1 | +0.1 | 0.000 | −0.1 / −0.1 | – |
| avg | +0.002 | +0.1 | +0.7 | 0.0 | +0.003 | −0.7 / −0.6 | – |
| m1_das, m5_fence | +0.001 (same as M0) | 0.0 | +0.1 | 0.0 | 0.000 | −0.1 | 0.009–0.011 |
| m2_bus_beam / m3_rtf_mvdr | −0.007 | −0.1 | +0.1 | −0.1 | +0.001 | −0.3 / −0.2 | 0.010 / 0.012 |
| x_gnfix_m2_bus_beam | +0.001 | +0.2 | +0.2 | +0.1 | −0.002 | −0.1 | 0.011 |
| m4_cdr / x_gnfix_m4_cdr | −0.005 / −0.005 | 0.0 / +0.1 | +1.7 / +2.0 | 0.0 | +0.001 | −1.8 / −2.0 | 0.012 |
| m6_ns | −0.010 | +0.7 (−0.4/+0.8/+1.8) | +3.6 | +1.6 | +0.015 | −3.9 / −2.9 | 0.009 |
| m7_cacgmm | −0.002 | 0.0 | +0.1 | 0.0 | −0.003 | −0.2 | 0.078 |
| m8_gtcrn | +0.009 (+0.004/+0.005/+0.018) | +3.4 (+2.5/+3.3/+4.4) | +9.5 | +2.8 | −0.010 | −7.8 / −6.1 | 0.053 |
| m8_dpdfnet | +0.010 (−0.000/−0.005/+0.036) | +5.3 (+4.4/+5.0/+6.4) | +14.1 | +3.8 | 0.000 | **−14.1** / −8.8 | 0.240 |
| x_combo_cdr_gtcrn | −0.003 | +3.0 | +11.2 | +2.6 | 0.000 | −10.1 / −8.2 | 0.072 |
| x_combo_s1beam (blind) | +0.031 (+0.001/+0.047/+0.045) | +1.2 | +1.7 | +1.3 | +0.010 | −0.7 / −0.5 | 0.011 |
| x_combo_s1beam_cdr_gtcrn (blind) | +0.028 (−0.003/+0.042/+0.045) | +4.5 | +12.4 | +4.2 | +0.004 | −9.7 / −8.0 | 0.069 |
| **x_combo_endfire** | **+0.045 (+0.043/+0.047/+0.045)** | +1.8 (+1.7/+1.8/+2.0) | +2.7 | +1.7 | +0.021 | −1.1 / −0.9 | 0.010 |
| x_combo_endfire_cdr_gtcrn | +0.042 (+0.040/+0.042/+0.045) | +5.8 (+5.2/+5.8/+6.6) | +13.2 | +5.4 | +0.011 | −9.0 / −7.3 | 0.072 |
| **x_combo_endfire_dpdf** | **+0.117 (+0.060/+0.123/+0.169)** | **+9.6 (+9.6/+9.8/+9.4)** | +14.6 | +8.6 | +0.027 | −8.2 / −5.0 | 0.247 |
| m9_voxlock2, clean enrollment (2 scenes) | −0.081 (0 dB rec −0.046; +5 dB quiet −0.115) | −0.3 | +6.2 | −1.3 | – | −8.1 / −6.5 | 1.10 |
| m9_voxlock2, device enrollment (1 scene) | −0.006 | +1.1 | +3.6 | +0.1 | – | −2.6 / −2.5 | 1.10 |
| m9_voxlock2_self (1 scene) | +0.002 (passed audio through) | 0.0 | – | – | – | −0.1 | 0.008 |

- **Endfire stacks:** x_combo_endfire and x_combo_endfire_dpdf beat M0 on 6 of 6 scenes. On the L3 extension (6 scenes), x_combo_endfire gives +0.030 to +0.061 and the endfire→CDR×GTCRN stack +0.045 to +0.059. L2 was not run.
- **M9 self-enrollment:** it passed the audio through because S1 found the partners "inseparable: one_peak, both_broadside".
- **Hagerman closure** (150 dB = the method behaves linearly): 44 dB for the endfire beam, because S0 and the gamma_n repair are re-estimated on every run; 33–35 dB for the NN stacks.

**What works**
1. A fixed one-sided endfire beam (x_combo_endfire): a superdirective MVDR beam, limited by white-noise gain (bus floors −15/−10/−6 dB), steered toward the Mic A end of the axis (+d/c, the partners' side).
   - It is the only processing that raises worst-partner ESTOI on its own, at every SNR.
   - On the **real** recorded parts alone, measured by applying the beam to each component separately: fan noise drops 3.0–3.4 dB (2.9–3.6 dB in the gaps between turns), while the real partners lose only 0.6 dB (A) and 1.1 dB (B). That is a real SNR gain of about 2.3–2.6 dB.
   - Per octave on the real fan noise: −3.0 dB at 0.5–1 kHz, −5.6 at 1–2 kHz, −4.8 at 2–4 kHz, −4.8 at 4–7 kHz, and +0.9 at 250–500 Hz (the cost of the white-noise-gain limit).
   - Simulated passengers drop by −4.4 dB (147°), −2.3 dB (95°) and −1.3 dB (77°).
2. Endfire → DPDFNet (x_combo_endfire_dpdf) is the best bus result by far: +0.117 ESTOI and +9.6 dB SNR.
   - The beam lifts partner B's SNR enough for DPDFNet to help B too. B's ESTOI goes 0.158 → 0.269 at 0 dB recorded; DPDFNet alone gives B nothing (0.158 → 0.153).
   - Cost: B's level still drops 3.5–12.8 dB, so an AGC is needed afterwards.
3. The single-channel NNs remove noise but do not raise intelligibility, and they push the weaker partner down: DPDFNet takes B down 7.5–20 dB, which fails the bench's TA_min ≥ −3 dB criterion.

**What does not work, and why**
1. **S1 steers to the wrong place.** calib_session.json has no gamma_n, so calib._override mixes in a noise coherence measured with the blind skew. It reads 0.12 at 4 kHz instead of 0.83. S1 then places both partners at +0.02 samples.
   - With gamma_n repaired, S1 reads +0.61 samples (true direct path +0.55), but as a single peak.
   - At −5 dB it adds the nearest passenger (−0.39…−0.41 samples) as "partner B", with the same posterior mass as a real partner.
2. **M2/M3 have no beam for this geometry.** Their design falls back to Mic A or a broadside average when both partners are on one side. At 1.19 cm a broadside sum does nothing against diffuse noise: on the real parts it changes the fan noise by only −0.08 dB.
3. **M4 and M7** stay within ±0.005 ESTOI.
4. **M9 fails on real recordings.** With dry enrollment it suppresses partner A by about 8 dB. Device-path enrollment removes the harm but still gives no gain.

**Calibration variants** (L1 recorded, 3 scenes)
- Under blind and wrong4cm, M1, M2, M3 and M5 equal Mic A; M4 gives −0.003 and M7 up to +0.005 / +0.5 dB.
- The x_auto_* wrappers stay within −0.008…+0.001 ESTOI and gain at most 1.1 dB of SNR (x_auto_m2_bus_beam, Hagerman closure only 16 dB).
- The endfire beam needs the skew correction:
  - With the blind x_s0v2 calibration (x_combo_endfire_auto) it keeps +0.044 / +0.046 / +0.043, the same as the measured calibration.
  - With default S0 and the 4 cm prior it gives −0.063 / −0.114 / −0.130.
  - Applied to the real parts: blind S0 (skew 0) costs the partners 3.4–5.5 dB; wrong4cm gives −0.06…−0.18 ESTOI.
- Rotating the device 180° (channels and calibration swapped): the beam then points away from the partners and gives −0.08 ESTOI, with partners −4 dB. Repaired S1 does read the correct side (−0.60 samples), and steering there gives +0.04.

## Meeting: 3 scenes (recorded / quiet / quiet + diffuse OMEETING), mean
Input: talker ESTOI mean 0.396, worst 0.182; level spread std 4.2 dB, range 15.4 dB. T1 capture = fraction of talkers with ESTOI not worse by more than 0.03 and level within ±3 dB of the median.

| method | T1 capture | Δmean ESTOI | Δworst | Δquiet (M3, M6) | Δfar (MF1, MF2) | level std / range out (dB) | NA | RTF cpu |
|---|---|---|---|---|---|---|---|---|
| m2_meeting_beam | 0.62 | +0.001 (same as Mic A) | 0.000 | 0.000 | 0.000 | 4.2 / 15.4 | 0.0 | 0.010 |
| m10_wpe | 0.62 | +0.007 | +0.007 | +0.005 | +0.003 | 4.3 / 15.8 | +0.5 | 0.045 |
| m11_level | 0.75 | 0.000 | −0.003 | −0.001 | 0.000 | **2.5 / 7.8** | 0.0 | 0.017 |
| m6_ns | 0.58 | +0.005 | −0.005 | +0.003 | +0.005 | 4.9 / 18.2 | +2.8 | 0.012 |
| m4_cdr | 0.54 | −0.007 | −0.010 | −0.017 | −0.007 | 4.6 / 16.6 | +2.1 | 0.014 |
| m8_gtcrn | 0.75 | +0.036 | +0.025 | +0.038 | +0.006 | 5.5 / 21.2 | +5.8 | 0.056 |
| m8_dpdfnet | 0.29 | +0.048 | −0.004 | +0.045 | +0.001 | 7.6 / 26.9 | +8.2 | 0.248 |
| x_combo_endfire | 0.62 | +0.021 | +0.010 | +0.012 | **+0.022** | 4.5 / 16.5 | +2.0 | 0.010 |
| x_combo_meet_chain (WPE→beam→GTCRN→AGC) | **0.75** | +0.038 | **+0.023** | +0.034 | +0.004 | 3.5 / 12.1 | +6.0 | 0.116 |
| x_combo_meet_ns_agc | 0.75 | +0.016 | +0.001 | +0.009 | +0.013 | 3.4 / 11.5 | +3.2 | 0.068 |
| x_combo_endfire_dpdf (+AGC) | 0.71 | **+0.076** | +0.006 | **+0.067** | +0.022 | 5.4 / 18.5 | +10.0 | 0.251 |

- **Best balance:** x_combo_meet_chain (T1 0.75, both the mean and the worst talker improve, level spread reduced).
- **Best intelligibility:** x_combo_endfire_dpdf.
- **Do not use DPDFNet alone:** it spreads the talker levels apart (T1 0.29).
- The M2 meeting beam falls back to Mic A, and leveling (M11) only helps the level spread.
- Blind and wrong4cm give the same picture: x_auto_m2_meeting_beam +0.002, x_auto_m4_cdr −0.004.

## Listening files (hw/realset/listen, 20 WAVs + index.json)
Each file is 16-bit and peak-normalised to −1 dBFS; the applied gains are in index.json.
- **Bus** (L1_snr+0_recorded and L1_snr+5_quiet): the Mic A input, x_combo_endfire_dpdf, x_combo_endfire and x_combo_endfire_cdr_gtcrn, plus m8_gtcrn as the best blind single-channel reference.
- **Meeting** (recorded and quiet): the Mic A input, x_combo_meet_chain, m8_gtcrn and x_combo_endfire_dpdf, plus m8_dpdfnet as a reference.
- Ranking: bus by mean worst-partner ΔESTOI; meeting by T1 capture, then mean ΔESTOI. Strictly by that meeting rule the third method would be x_combo_meet_ns_agc (T1 0.75, +0.016). I exported x_combo_endfire_dpdf (T1 0.71, +0.076) instead and say so in index.json.

## Key numbers
- Bus best: x_combo_endfire_dpdf (L1, 6 scenes): worst-partner ΔESTOI +0.117 (−5/0/+5 dB: +0.060/+0.123/+0.169), ΔSNR +9.6 dB, ΔSIR passengers +8.6 dB, partner B level −3.5…−12.8 dB, CPU RTF 0.247
- Bus: endfire beam alone (x_combo_endfire): ΔESTOI +0.045 (+0.043/+0.047/+0.045), ΔSNR +1.8 dB, partner level −1.1/−0.9 dB, 6/6 scenes, CPU RTF 0.010; L3 extension +0.030…+0.061
- Endfire beam on REAL components (bus L1): fan noise −3.0…−3.4 dB, real partners −0.6 (A) / −1.1 (B) dB; per octave on the fan noise −3.0/−5.6/−4.8/−4.8 dB at 0.5–1/1–2/2–4/4–7 kHz, +0.9 dB at 250–500 Hz
- S1 partner TDOA: calib_session.json (corrupted gamma_n 0.12 at 4 kHz): +0.02 samples; gamma_n repaired: +0.61 (true +0.55); at −5 dB a passenger appears at −0.39…−0.41
- S1-steered M1/M2/M3/M5/M7 (bus L1, measured): ΔESTOI −0.007…+0.002, ΔSNR −0.1…+0.2 dB (same as Mic A or the calibrated average)
- Single-channel NNs (bus L1): GTCRN ΔESTOI +0.009 / ΔSNR +3.4 dB / weaker-partner level −7.8 dB; DPDFNet +0.010 / +5.3 dB / −14.1 dB
- M9 VoxLock: clean enrollment −0.046 (0 dB rec), −0.115 (+5 dB quiet), partner A −7.8/−8.4 dB; device enrollment −0.006; self-enrollment passes audio through; CPU RTF 1.10 (wall 2.4–3.1)
- Calibration dependence of the endfire beam (bus, L1 recorded): measured +0.045; x_s0v2 blind +0.044/+0.046/+0.043; default S0 with 4 cm prior −0.063/−0.114/−0.130
- Device rotated 180° (real data, channels swapped): beam pointing away: ΔESTOI −0.08, partners −4 dB; S1 reads the correct side (−0.60 samples)
- Endfire white-noise-gain floor sweep (bus L1 0 dB rec): −3 dB: +0.026; −6 dB: +0.035; −10 dB / bus floors: +0.045; −15 dB: +0.046; steering 0.75–1.25 d/c all ≈ +0.042…+0.046
- Meeting best balance: x_combo_meet_chain: T1 0.75, mean ΔESTOI +0.038, worst +0.023, level std 4.2→3.5 dB, NA +6.0 dB, CPU RTF 0.116
- Meeting best intelligibility: x_combo_endfire_dpdf (+AGC): mean ΔESTOI +0.076, quiet talkers +0.067, far talkers +0.022, worst +0.006, T1 0.71
- Meeting leveling m11_level: level std 4.2→2.5 dB, range 15.4→7.8 dB, ESTOI unchanged; DPDFNet alone widens the spread to 7.6 dB std (T1 0.29)
- Run size: about 330 scene–method results, 0 errors; machine load 115–175; 02:47–04:28

## Deliverables
- NOT WRITTEN: /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/hw/realset/RESULTS_REAL.md (harness refused report files from this subagent; the summary above is its content)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/tmp/night_realrun/tables.md (all generated tables) and summary.json (numbers)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/hw/realset/results/{measured,blind,wrong4cm}/<method>/*.json (per-scene results; each has src_sha1, rtf_cpu and loadavg; no WAVs)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/hw/realset/listen/ (20 WAVs + index.json)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/methods/x_combo_endfire.py, x_combo_endfire_auto.py, x_combo_endfire_dpdf.py, x_combo_endfire_cdr_gtcrn.py, x_combo_endfire_cdr_gtcrn_auto.py, x_combo_s1beam.py, x_combo_s1beam_cdr_gtcrn.py, x_combo_s1beam_dpdf.py (not scored), x_combo_cdr_gtcrn.py, x_combo_m9_cdr.py (not scored), x_combo_meet_chain.py, x_combo_meet_ns_agc.py, x_combo_common.py (helpers)
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/tmp/night_realrun/realrun.py (realeval wrapper: CPU-time RTF, source hash, WAVs optional), analyze.py, compact.py, by_noise.py, export_listen.py, chainA-D.sh, logs/
- /Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab/tmp/night_realrun/{group_atten,group_atten_meet,group_atten_var,endfire_sweep,endfire_bands,flip_test,s1_probe,s1_mass}.py with their .json outputs (diagnostics on the real components)

## Issues
- hw/realset/RESULTS_REAL.md was not written: the harness refused report files from this subagent, and I removed my own early draft stub. Its content is the summary of this result.
- Geometry: every real talker sits at one endfire position on Mic A's side (two MacBook speakers). The endfire stacks use this as a prior: the device is oriented with Mic A's end toward the talkers. Turned the other way they are harmful (flip test: −0.08 ESTOI, partners −4 dB). Partners on opposite sides, at broadside or on the B side were never tested.
- The blind form x_combo_s1beam gets only +0.031 (+0.001 at −5 dB), because S1 takes the nearest passenger as 'partner B'. The blind DPDFNet stack x_combo_s1beam_dpdf was written but killed before any result.
- NN stacks pull the weaker partner B down 3.5–20 dB (TA), even with 15–20 dB attenuation limits; no bus AGC was tested after them.
- Passengers and bus noise are simulated (with the board's skew and gain), and 'quiet' is the fan recording scaled by −12 dB. The mics' own self-noise in a quiet room was never isolated, so white-noise-gain risk is under-tested. A −6 dB floor gives +0.035 instead of +0.045.
- Scenes share the same partner audio, so there are no confidence intervals; treat differences below ~0.01 ESTOI or 0.3 dB as noise.
- Coverage: M9 ran on 1–2 scenes; blind and wrong4cm only on 3 bus (L1 recorded) and 3 meeting scenes; L2 not run; L3 only for 4 methods. x_combo_m9_cdr and x_combo_s1beam_dpdf were not scored.
- Some results reused through --skip-existing came from earlier runs with possibly older method code (no src_sha1): mic_a, m0 and avg; m1_das and s0_calib_only on L1_snr+0; x_gnfix_m2_bus_beam and x_gnfix_m4_cdr on L1_snr+0_quiet; x_auto_m2_bus_beam and x_auto_m4_cdr wrong4cm on L1_snr+0_recorded. Methods were also being re-tuned by other runs during this pass; every new result stores src_sha1.
- Wall-time RTF is inflated by the load (115–175 on 10 cores); CPU RTF is the better cost proxy. The NN and M9 stacks will not fit the ESP32-C3 (no FPU, ~400 KB SRAM). The fixed endfire beam is two short FIR filters and plausibly fits (not measured).
- I finished at 04:28, over the ~75 min budget: tool calls became very slow at the end. All my processes were stopped. The data volume had only ~5 GB free, so result WAVs were not written.

## Next
- Core owner: give calib_session.json a correct gamma_n, or fix calib._override to rotate gamma_n with the json's own skew. With the repair, S1 places the partners at +0.61 instead of +0.02 samples.
- M2: add a one-sided state, so that when S1 finds both partners on one side it steers a WNG-limited MVDR at their mean TDOA instead of falling back to Mic A. Add a partner-vs-passenger check, for example from directness, distance or turn-taking.
- Score x_combo_s1beam_dpdf (written), and add a bus AGC after the DPDFNet stacks so partner B's level comes back.
- Record new real takes: talkers on the B side, at broadside and on opposite sides; a truly quiet room for self-noise and white-noise-gain risk; real passengers. Then re-rank the endfire versus the S1-steered beams.
- On the device: the endfire beam as two fixed FIR filters, plus a per-boot skew from 2–3 s of room noise (x_s0v2). Keep the NN post-filter off-device unless a much smaller model fits.
- Drop M9 for this use: dry enrollment hurts, and device-path enrollment gives no gain at 0 dB.
- When bus_final / meeting_final exist, run them in all three variants against x_combo_endfire_dpdf and x_combo_meet_chain.
