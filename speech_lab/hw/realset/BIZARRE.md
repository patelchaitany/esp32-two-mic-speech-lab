# Quick experiments on the real ESP32 recordings (run `real:bizarre`)

2026-11-03, 02:30–03:05 IST. I worked only on audio already recorded: nothing was played, recorded or sent to the board.
I wrote only to `hw/realset/bizarre/` (11 experiment scripts, the helpers `bz_common.py` and `bz_takes.py`, one JSON result per script),
to this file and to `tmp/night_bizarre/` (logs, plus the WPE outputs in `wpe_out/`, 59 MB).
Each script ran in under 8 minutes at −j ≤ 2 while the machine load was 115–165.
Every number below comes from the named JSON file.

Data used:
- the realset takes in `hw/realset/img` and `stim` (current boot);
- the meeting scenes, through `realscenes.load_scene`;
- session 1 in `hw/rec` (first boot);
- the reboot and drift files in `hw/rec2`;
- the board calibration `calib_session.json` (1.19 cm, τ0 −1.015 samples, per-band gain for B).

## Main findings for the decision

1. **Skew: estimate it at every boot, from ambient noise.**
   - The skew is −1 sample plus a fraction that changes per boot. The fraction was 0.11 at the cold first boot and 0.013–0.035 after each of 8 resets.
   - Within one boot the spread is ±0.007 sample (std) over 31 minutes.
   - 0.5 s of room noise is enough for ±0.05 sample: 100 % of windows in this room, and ≥ 99.6 % in emulated ambient down to −75 dBFS. At −80 dBFS it takes 1–2 s.
   - Speech from near-endfire talkers is biased by +0.10 (reverberant tails) to +0.22 sample (all frames), so it cannot replace the noise estimate.
   - The B gain is a fixed property: its std across boots is ≤ 0.03 dB at 0.5–2 kHz, so a factory constant will do.
   - The spacing fitted from noise is unreliable (1.06–1.99 cm). Use the mechanical value.
2. **Uncorrected skew destroys any beam at 1.19 cm.**
   - The target falls into the beam's null. Theoretical array gain is −15 dB for a cardioid and −10 dB for a superdirective beam, instead of +4.6 / +5.9 dB.
   - Talkers more than about 35° from endfire get TDOAs outside the physical range.
   - At 4 cm the same skew degrades the beam rather than destroying it: superdirective +1.4 dB instead of +5.0, with on-axis errors of 4.5–12 dB.
3. **The real noise caps what two mics can do at 1.19 cm.**
   - The noise is 15 % incoherent at 150–500 Hz (median over takes, up to 24 %). It is nearly fully coherent at 0.5–2 kHz.
   - Unloaded fixed superdirective or differential beams therefore gain about 0 dB median on the 18 takes, ranging −5 to +5 dB.
   - With diagonal loading of 0.01–0.05 the gain is +2.0 to +2.2 dB median, and positive on every take (minimum +0.3 dB).
   - The oracle ceiling for a time-invariant 2-mic filter is +2.5 dB (range 1.1–4.2).
   - In theory, the same incoherence costs a 4 cm pair less: +4.3 vs +3.2 dB for a loaded superdirective beam at 5 % incoherence.
4. **The CDR "nearness" cue does not hold up in bus noise.**
   - Noise-free simulation: CDR separates talkers at 0.3–1 m from talkers at ≥ 2 m (utterance AUC 0.92–1.00), as well at 1.19 cm as at 4 cm.
   - With diffuse noise at 0 dB it collapses: AUC 0.56–0.80, and 0.53–0.62 per frame.
   - The emulated far takes MF1/MF2 cannot test this at all. Their reverb was baked in and played from a near speaker, so it reaches the board as a coherent plane wave.
5. **Meeting speaker identity is limited by SNR, not by the embedder.**
   - Clean images: 8 of 8 talkers found (ARI 1.0 at k = 8).
   - With the real fan noise (talkers at −8 to +6 dB SNR): 9 clusters, purity 0.72, and the −12 dB and −15 dB talkers are absorbed into others.
   - At −12 dB fan noise ('quiet'): 8 of 8 talkers found, purity 0.94–1.0.
   - S1 as deployed labels 0 % of turn frames in 'recorded'.
   - TDOA/CDR clustering cannot separate these talkers here: only two source positions exist, so it gives 1 cluster.
6. **WPE helps only when the room is quiet.**
   - On the noise-free LTI images, ESTOI against the direct target rises by +0.21 / +0.23 (MF1/MF2) and +0.29 (median, near takes).
   - On the real recordings it rises by +0.01 to +0.03.
   - The benefit on MF1 shrinks with noise: +0.13 at 30 dB SNR, +0.07 at 10 dB, +0.03 at 0 dB.
   - 2-channel WPE beats 1-channel; 60 taps are no better than 30.
7. **M9/VoxLock under full overlap extracts whichever partner is louder.**
   - The louder partner A (+4 to +9 dB SIR) comes out at +9 to +16 dB SIR. With device enrollment, A loses 0.6–2.1 dB and ESTOI rises by +0.05 to +0.20.
   - A wrong same-gender enrollment extracts A almost as well, so the pass follows the dominant voice, not the enrolled identity.
   - The weaker partner B (−4 to −9 dB SIR) is not recovered with any enrollment: B loses 6–29 dB, and ESTOI changes by −0.17 to +0.01.
   - Device enrollment keeps more of the target than clean dry enrollment in all 8 cases.

## Exp 1: Speaker identity on the real meeting sequence (`bz1_speaker_id.py` → `bz1_speaker_id.json`)

**Setup.**
- Scenes and talkers: 8 talkers, one 4.2 s turn each: M1–M6 at 0 … −15 dB, and MF1/MF2, the emulated 4 m talkers.
- Embeddings: WeSpeaker ResNet34 (`spk_onnx`) on 1.5 s chunks (hop 0.75 s) inside the true turns, so only identity is tested. That gives 4 chunks per talker and 32 per scene.
- Inputs: 'clean' (sum of the images), 'raw' (Mic A after the 80 Hz prefilter), 'gtcrn' (`m8_gtcrn`) and 'wpe' (`m10_wpe`).
- ID / EER: closed-set identification and equal error rate against each talker's dry utterance. It is the same utterance, so these measure channel and noise mismatch only.
- Clustering: AHC (average linkage, cosine) cut at θ = 0.375 (`spk_onnx.THETA_S`) or at k = 8, and the online `CosineClusterer`.
- Cosine between the 8 dry references: max 0.33, mean 0.10.

| Scene / input | ID acc | EER | AHC θ0.375: clusters / purity / talkers covered / ARI | AHC k=8 ARI | online θ0.375: clusters / purity / ARI |
|---|---|---|---|---|---|
| any scene, clean images | 0.94 | 0.09 | 10 / 1.00 / 8 / 0.90 | 1.00 | 8 / 1.00 / 1.00 |
| recorded, raw | 0.50 | 0.28 | 9 / 0.72 / 7 (M3, M6 absorbed) / 0.40 | 0.40 | 5 / 0.44 / 0.15 |
| recorded, gtcrn | 0.44 | 0.34 | 11 / 0.84 / 8 / 0.63 | 0.59 | 6 / 0.50 / 0.19 |
| recorded, wpe | 0.56 | 0.28 | 9 / 0.69 / 7 / 0.36 | 0.36 | 5 / 0.44 / 0.15 |
| quiet, raw | 0.81 | 0.19 | 14 / 0.94 / 8 / 0.67 | 0.87 | 11 / 1.00 / 0.89 |
| quiet, gtcrn | 0.78 | 0.16 | 12 / 0.94 / 8 / 0.71 | 0.87 | 9 / 0.94 / 0.85 |
| quiet_omeeting, raw | 0.81 | 0.16 | 15 / 1.00 / 8 / 0.72 | 0.70 | 11 / 0.94 / 0.78 |

**What the table shows.**
- Hardest talkers: median target cosine on 'recorded' raw is M6 (−15 dB) 0.08, MF1 0.13, MF2 0.18 and M4 0.18. The 95th percentile of non-target cosines is 0.22.
- Even on clean images, MF1/MF2 reach only 0.24 / 0.33 and are identified in 3 of 4 chunks.
- The NN denoiser makes verification against clean references worse (ID 0.50 → 0.44), but makes within-scene clustering better at low SNR (ARI 0.40 → 0.63).
- WPE changes nothing.

**S1 as deployed** (`tracker.front_end` with the embedder, `cos_thr` 0.375 and the session calibration):
- 'recorded': VAD > 0.5 on only 11 % of turn frames (M2 1 %, M6 3 %). No chunk reaches 1.5 s of confident speech, so 0 frames are labelled.
- 'quiet': 4 clusters, purity 0.55, ARI 0.51, 76 % of turn frames labelled.
- 'quiet_omeeting': 1 cluster; only MF1/MF2 frames are labelled (19 %).
- The DP-means fallback (TDOA + CDR) gives 1 cluster in all three scenes, as it must with 2 identical positions.

**Answer.** An embedding back-end can count and keep the 8 talkers once each talker is about 5 dB or more above the noise floor. Its main failure is the quiet talkers (−12 and −15 dB), and in the deployed chain the VAD/segmentation fails before the embeddings do.

## Exp 2: Nearness from two mics 1.19 cm apart (`bz2_nearness.py` → `bz2_nearness.json`)

**Real takes (part A).**
- Estimators: DOA-free CDR (prop3) and DOA-dependent CDR (prop1, plane wave at +0.55 sample), calibrated, 50 ms smoothing, Γn = sinc at 1.19 cm.
- Each take's value is the median over oracle speech frames.
- Truth (DRR / C50 of the dry→Mic A response):
  - L speaker: −1.7 to +2.9 dB DRR (median +1.2) and +2.1 to +6.7 dB C50 on the 0 dB takes.
  - R speaker: −2.6 to −4.2 / +0.6 to +2.8 dB.
  - MF1/MF2: −16.7 / −21.2 and −5.7 / −7.9 dB.
- The DRR of the low-level takes (M2–M6, −5 to −14 dB) is pulled down by fit noise in the 4096-tap response.

| CDR (dB), condition | L takes (8) median | R takes (8) median | MF1 / MF2 | AUC near vs MF (frames) | Spearman vs DRR (near takes) |
|---|---|---|---|---|---|
| image only, 2–4 kHz, DOA-free | +4.9 | +0.4 | +0.5 / +0.2 | 0.66 | 0.62 |
| image only, 2–4 kHz, DOA | +3.6 | −2.0 | −2.2 / −5.0 | 0.71 | 0.80 |
| image only, 4–7 kHz, DOA-free | +4.5 | +0.9 | +3.0 / +1.8 | 0.45 | 0.65 |
| recording, 2–4 kHz, DOA-free | +1.3 | −1.1 | −0.6 / −1.7 | 0.62 | 0.66 |
| recording, 2–4 kHz, DOA | −0.9 | −6.5 | −5.6 / −6.8 | 0.61 | 0.88 |

CDR does rank the two real speaker positions by their DRR: on the 0 dB takes, recording, 2–4 kHz, L +2.5 vs R −0.9 dB. It does not see the emulated far talkers as far (AUC 0.45–0.71).

**Simulation (part B).**
- Room and array: pyroomacoustics, 10 random placements per distance, the real takes' dry speech.
- Bus room: 11.5 × 2.5 × 2.2 m, RT60 0.45 s.
- Meeting room: 8 × 6 × 3 m, RT60 0.6 s.
- Noise: none, or diffuse speech-shaped noise at 0 dB.
- Values are the median CDR per distance (4–7 kHz) and the AUC for ≤ 1 m vs ≥ 2 m.

| Room, spacing, noise | 0.3 / 0.5 / 0.8 / 1.2 / 2 / 3 m (bus) or 0.5 / 1 / 2 / 3 / 4 m (meeting) | AUC per utterance | AUC per frame |
|---|---|---|---|
| bus, 1.19 cm, none | +8.8 / +5.5 / +3.6 / +2.6 / +3.0 / +0.4 | 0.97 | 0.76 |
| bus, 4 cm, none | +8.4 / +5.2 / +3.3 / +2.2 / +2.3 / +1.8 | 0.96 | 0.78 |
| bus, 1.19 cm, 0 dB | −1.2 / −1.3 / −1.3 / −1.4 / −1.4 / −1.4 | 0.56 (2–4 kHz: 0.68) | 0.53 |
| meeting, 1.19 cm, none | +7.2 / +3.0 / +0.9 / +0.7 / +0.2 | 0.99 | 0.78 |
| meeting, 1.19 cm, 0 dB | −1.2 / −1.4 / −1.5 / −1.6 / −1.4 | 0.70 | 0.58 |
| meeting, 4 cm, 0 dB | −1.6 / −1.5 / −1.7 / −1.7 / −1.7 | 0.69 (2–4 kHz: 0.80) | 0.59 |

**Answer.** In quiet conditions, two mics 1.19 cm apart can tell a talker at 0.3–0.5 m from one at ≥ 2 m, and 4 cm is not better. At bus SNRs (−5 to +5 dB) CDR mostly measures SNR, so it cannot decide which talkers to keep. A level/distance cue or speaker identity has to do that job. MF1/MF2 are not valid far-talker tests for any spatial cue.

## Exp 3: WPE on the far takes (`bz3_wpe_far.py`, `bz3b_wpe_snr.py` → JSON)

**Setup.**
- Target: dry speech ∗ (first 5 ms after the direct path of the dry→Mic A response).
- Responses: for MF takes, g·h_far∗h, with h_far re-generated from `far_rir(seed=7)`; it rebuilds the stored image to −37.7 / −34.0 dB.
- WPE settings: `m10_wpe` (2-ch, 30 taps); 1-ch WPE is the same on Mic A only.
- Metric: ESTOI on the speech interval.
- 'image' is an exact LTI convolution, so it is an upper bound.

| Take (DRR) | image: none → 2-ch / 1-ch | recording: none → 2-ch / 1-ch | in-band SDR, image | SDR, recording |
|---|---|---|---|---|
| MF1 (−16.7 dB) | 0.152 → 0.360 / 0.240 | 0.132 → 0.165 / 0.154 | −4.4 → −0.8 | −7.7 → −7.1 |
| MF2 (−21.2 dB) | 0.036 → 0.265 / 0.100 | 0.030 → 0.044 / 0.033 | −8.2 → −1.5 | −15.0 → −14.8 |
| 8 near takes, median Δ | +0.286 (+0.09…+0.39) / +0.058 | +0.012 (0.00…+0.02) / +0.011 | +9.2 dB | +0.4 dB |

**ESTOI vs SNR** (2-ch / 1-ch WPE; noise = the take's own fan noise rescaled; the original recordings are at −0.6 / −4.4 dB):

| SNR | 30 dB | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| MF1: none → 2-ch / 1-ch | 0.153 → 0.287 / 0.218 | 0.156 → 0.264 / 0.212 | 0.159 → 0.226 / 0.197 | 0.152 → 0.201 / 0.183 | 0.136 → 0.170 / 0.158 |
| MF2 | 0.036 → 0.133 / 0.078 | 0.033 → 0.117 / 0.073 | 0.020 → 0.083 / 0.056 | 0.021 → 0.067 / 0.048 | 0.030 → 0.056 / 0.044 |

- Against the early-50 ms target, MF1/MF2 images go from 0.357 / 0.232 to 0.614 / 0.519.
- 60 taps are no better than 30 (image MF1 0.367, MF2 0.269).
- On near takes at 0 dB SNR WPE gives about +0.02.

**Answer.** WPE is worth running for far meeting talkers only when the room is quiet (≥ 10 dB SNR). Use the 2-channel version: even at 1.19 cm it gives roughly 2–4 times the gain of 1-channel WPE. At the fan-noise SNRs in this room it is close to useless. The image numbers are an upper bound, because the fit is exactly LTI and the real room is not.

## Exp 4: Partner extraction under full overlap with M9/VoxLock (`bz4_overlap_tse.py`, `bz4b_refcheck.py` → JSON)

**Mixtures.**
- 'acoustic': `hw/rec/talkAB_LR.wav`, where A and B were played at the same time. Both responses were fitted jointly on that take by 2-input least squares; the residual during speech is −53.3 dBFS against −55.2 dBFS noise-only.
  - The joint references agree with independent single-talker fits (`talkA_L`, `talkB_R`) to −8.3 / −10.7 dB (A, 0–5 ms / 0–50 ms) and −16.8 / −16.2 dB (B).
  - That is the expected inter-take room variance. They are unchanged at 1500 iterations; `bz4b_refcheck.json`.
  - Ignore `direct5ms_vs_single_take_*` in `bz4_overlap_tse.json`: it has no lag search.
- 'digital_k': images A_k + B_k with the A_k take noise, so the references are exact.

**Pass.** One VoxLock pass per target (S0 bus prefilter, 100-frame reference partition).

**Enrollment.**
- 'clean': dry AE/BE utterances.
- 'device': Mic A recordings of the same utterances.
- 'wrong': a same-gender third talker (M1 3575 F for A, M2 2300 M for B).

**Metrics.** SIR and TA come from projecting the pass's mask onto each component; "oth" is how much the other partner is attenuated.

| Mixture, target | input SIR / ESTOI | clean: SIR / TA / ESTOI | device: SIR / TA / ESTOI | wrong: SIR / TA / ESTOI |
|---|---|---|---|---|
| acoustic, A | +4.7 / 0.59 | +15.1 / −2.3 / 0.69 | +13.8 / −0.6 / 0.79 | +14.2 / −2.0 / 0.69 |
| digital_1, A | +5.2 / 0.47 | +14.8 / −3.9 / 0.45 | +12.2 / −0.7 / 0.63 | +14.6 / −3.0 / 0.47 |
| digital_2, A | +4.3 / 0.39 | +9.1 / −22.2 / 0.27 | +10.1 / −2.1 / 0.44 | +11.1 / −7.3 / 0.36 |
| digital_3, A | +8.6 / 0.59 | +15.3 / −13.4 / 0.42 | +13.3 / −0.9 / 0.71 | +15.9 / −4.1 / 0.60 |
| acoustic, B | −4.7 / 0.29 | −8.0 / −15.9 / 0.22 | −9.9 / −6.4 / 0.12 | −2.8 / −20.6 / 0.22 |
| digital_1, B | −5.2 / 0.21 | −2.9 / −25.6 / 0.16 | −5.4 / −7.4 / 0.20 | −1.9 / −29.1 / 0.16 |
| digital_2, B | −4.3 / 0.15 | +4.3 / −11.2 / 0.11 | 0.0 / −7.6 / 0.16 | +5.2 / −12.7 / 0.11 |
| digital_3, B | −8.6 / 0.12 | −1.7 / −16.0 / 0.11 | −8.0 / −7.6 / 0.13 | −3.2 / −19.1 / 0.09 |

**Answer.**
- Enrolled on A, the pass extracts A from A+B: +5 to +10 dB SIR. With device enrollment, ESTOI rises by +0.05 to +0.20 and SI-SDR by +1 to +4 dB.
- A wrong enrollment does about as well, so the pass follows the dominant voice rather than the enrolled identity.
- B cannot be extracted from A+B: B is suppressed by 6–29 dB with every enrollment, and ESTOI changes by −0.17 to +0.01.
- Clean enrollment is fragile on these takes (TA −13 and −22 dB on digital_2/3). Enroll on the device.
- For the bus this means M9's union mask keeps a dominant talker during overlaps, not specifically the two partners. Overlapped speech from the quieter partner is at risk.

## Exp 5: Calibration stability (`bz5_calib_stability.py`, `bz5b_quiet_skew.py` → JSON)

**Estimators.**
- Skew: weighted phase slope of the noise cross-spectrum, 150–7000 Hz. The diffuse coherence sinc(2fd/c) > 0 below 14.4 kHz, so the phase is only the skew.
- Spacing: fit of |coherence| to ρ·sinc.
- Gain: B re A per band.
- Each estimate uses a silent stretch of 5–10 s.

| Boot | Silent stretches | Skew (samples) | Spacing fit (cm) | B re A, 200–5000 Hz (dB) |
|---|---|---|---|---|
| 0: cold first boot, 01:19–01:49 | 13 | −1.112 ± 0.014 (−1.128…−1.071) | 1.14–1.99 (median 1.25) | −0.47…−0.76 |
| 1–7: one esptool reset each, 01:51–01:54 | 1 each | −1.035, −1.022, −1.015, −1.013, −1.017, −1.013, −1.013 | 1.12–1.20 | −0.47…−0.71 |
| 8: current boot, 01:54–02:25 | 24 | −1.013 ± 0.007 (−1.029…−1.000) | 1.06–1.49 (median 1.15) | −0.44…−0.74 |

**Skew.**
- s2_ambient (01:49, still boot 0) reads −1.120; reboot_00 two minutes later reads −1.035. The jump coincides with the reset.
- Halving the room level later (drift_01/02, −56/−58 dBFS) moved the estimate by only −0.005 to −0.016 sample. So the per-boot fraction is real and not an artefact of the noise field.
- If SCK runs at 64·fs (0.0156 sample per period), the fraction is about 7 periods at the cold boot and 1–2 periods after resets. That is a hypothesis about where the fraction comes from, not a measured cause.

**Gain.**
- Per band (150–500 / 0.5–1k / 1–2k / 2–4k / 4–7k Hz) across the 8 reset boots: −0.01 / −0.74 / −0.68 / −0.46 / −0.91 dB, with std 0.35 / 0.02 / 0.03 / 0.17 / 0.14.
- These are noise-based values. The direct-sound value (−1.3 dB) differs, but both are fixed.

**Spacing.** Quieter stretches give ρ 0.99–1.00 and d 1.39–1.99 cm, because the fan field is then not diffuse. Do not calibrate spacing from ambient noise here.

**Seconds of noise needed** (std over windows; percentage of windows within ±0.05 sample of the median or the truth):

| Ambient level | 0.25 s | 0.5 s | 1 s | 2 s | 5 s |
|---|---|---|---|---|---|
| recorded, −40 to −47 dBFS (current boot) | 0.011 (99.6 %) | 0.009 (100 %) | 0.008 | 0.007 | 0.006 (4 s) |
| recorded, −56/−58 dBFS (drift_01/02) | 0.017 (100 %) | 0.014 | 0.013 | 0.012 | – |
| emulated −70 dBFS (+ −87 dBFS self-noise) | – | 0.012 (100 %) | 0.010 | 0.008 | 0.007 |
| emulated −75 dBFS | – | 0.015 (99.6 %) | 0.012 (100 %) | 0.010 | 0.008 |
| emulated −80 dBFS | – | 0.023 (95.9 %) | 0.016 (99.2 %) | 0.012 (100 %) | 0.009 |

- The emulated levels are the fan noise scaled down plus independent white noise at −87 dBFS (the datasheet self-noise). They assume the field stays near-diffuse.
- **Speech only, no noise** (reverberant images of the 18 takes): all speech frames give a median of −0.79 (error +0.22, toward the talkers' +0.55 TDOA). Decay frames (≥ 8 dB below the previous 100 ms) give −0.91 (error +0.10, range −1.05…−0.64).

## Exp 6: How much skew, gain and spacing matter (`bz6_skew_sensitivity.py`, `bz6b_incoherent.py`, `bz6c_oracle_mvdr.py` → JSON)

**Theory.**
- Model: free field, target at endfire (as both real speakers are), 3-D diffuse noise, flat EQ on the target axis.
- Figure of merit: mean array gain over 300–4000 Hz, i.e. the SNR gain over one mic.
- Noise: bus-like (self-noise 47 dB down) and, separately, with an incoherent fraction μ.

| Beam (calibrated, μ ≈ 0) | 1.19 cm | 2 cm | 4 cm | 1.19 cm, residual skew 0.05 / 0.1 / 0.25 sample | 1.19 cm, gain error 1.3 dB | skew −1.015 left uncorrected: 1.19 / 2 / 4 cm |
|---|---|---|---|---|---|---|
| cardioid | +4.6 | +4.3 | +1.5 | +4.4 / +4.2 / +3.7 | +4.1 | −15.2 / +3.8 / +4.6 |
| superdirective MVDR | +5.9 | +5.8 | +5.0 | +5.9 / +5.7 / +5.1 | +4.7 | −9.8 / −12.0 / +1.4 |
| delay-and-sum | +0.4 | +1.0 | +2.5 | +0.3 / +0.3 / +0.1 | +0.4 | +1.7 / +2.7 / +2.6 |

- On-axis response error with uncorrected skew: up to 21 dB (cardioid, 1.19 cm) and 4.5–12 dB at 4 cm.
- Talker positions seen with the skew uncorrected (`doa_uncorrected` in the JSON):
  - At 1.19 cm, a talker at 0° appears at 146°, and talkers at 45–180° fall outside ±d/c.
  - At 4 cm, 0° → 63°, 45° → 81° and 90° → 123°.

**Incoherent noise** (`bz6b_incoherent.json`):
- Measured incoherent fraction of the take noise after calibration, median over the 18 takes: 0.155 at 150–500 Hz (0.02–0.24), ≈ 0 at 0.5–2 kHz, 0.024 at 2–4 kHz.
- With μ = 0.05 the theory gives:
  - 1.19 cm: hypercardioid +1.5 (−11.1 dB at 300 Hz), loaded superdirective MVDR +3.2.
  - 2 cm: +3.4 / +4.1.
  - 4 cm: +4.0 / +4.3.

**Real takes** (fixed beams at 1.19 cm, target at endfire; image and noise scored separately; median and range over 18 takes):

| Beam | full calibration | skew corrected only | gain corrected only | none |
|---|---|---|---|---|
| delay-and-sum | +0.07 (−0.03…+0.14) | +0.08 | +0.61 | +0.61 |
| cardioid | +0.74 (−3.3…+4.4) | +1.01 | −3.50 (target −4.6 dB) | −1.92 (target −3.6 dB) |
| superdirective MVDR, unloaded | −0.07 (−5.1…+5.0) | +0.67 | −3.62 | −2.30 |
| superdirective MVDR, loading 0.01 | **+2.18 (+0.32…+3.78)** | – | – | – |
| superdirective MVDR, loading 0.05 | +1.96 (+0.76…+2.82) | – | – | – |
| oracle time-invariant MVDR (true noise covariance and RTF) | **+2.53 (+1.13…+4.23)** | – | – | – |

- Oracle MVDR per band (150–500 / 0.5–1k / 1–2k / 2–4k / 4–7k Hz): +0.9 / +2.0 / +3.3 / +3.2 / +2.2 dB.
- With the noise covariance taken from the noise-only tail instead: +2.3 dB (−0.4…+4.0).

**Answer.**
- At 1.19 cm the skew must be corrected; a 0.05 sample residual is harmless. The 1.3 dB gain error costs 0.5–1.2 dB.
- The real limit is low-frequency incoherent noise: two mics buy about +2 dB of time-invariant SNR on this board.
- Moving to 2–4 cm would make fixed beams more robust to that noise (theory: +1 to +2.5 dB more at 5 % incoherence) and to skew. 4 cm loses directivity above about 3 kHz in a first-order cardioid.

## Caveats and what I could not do

- All real talkers come from 2 positions near endfire on Mic A's side, so no spatial result here covers broadside or B-side talkers.
- MF1/MF2 are spatially near: the reverb was played through the near speaker, so their emulated reverb arrives as a coherent plane wave. They test reverberation, not distance.
- The image references are 4096-tap LTI fits. Results labelled "image" are noise-free upper bounds, and WPE in particular matches that model exactly. Low-level takes (M2–M6) carry fit noise in their response tails, which biases their DRR low.
- Exp 1 uses the same dry utterance as the identity reference (channel mismatch only) and only 4 chunks per talker. Its accuracies come in steps of 1/32 and have no confidence intervals.
- Exp 4: VoxLock runs one pass per partner without priming. 'wrong' enrollments are same-gender only; cross-gender wrong enrollment and partial overlaps were not tested.
- Exp 5: the "quiet room" levels are emulated, not recorded. The cause of the per-boot skew fraction (decimator phase vs SCK) is a hypothesis; checking it needs a cold power cycle on the hardware.
- Exp 6: the theory is narrowband free field. The real fixed-beam numbers include the image-fit residual in the "noise", and the per-take spread is large (±5 dB unloaded). The loaded and oracle results are consistent across all takes.
- Not run: no composed stacks (`bus_final`, `meeting_final`) were used, and nothing from the bus or meeting evaluation sets was re-scored.
- Housekeeping: running the scripts may have written Python bytecode caches (`__pycache__`) next to the modules they import. No source file outside `hw/realset/bizarre/` was changed.

## Reproduce

From `speech_lab`, with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

```
.venv/bin/python -I hw/realset/bizarre/bz5_calib_stability.py      # ~1.5 min
.venv/bin/python -I hw/realset/bizarre/bz5b_quiet_skew.py [speech]  # ~4 min (speech: 30 s)
.venv/bin/python -I hw/realset/bizarre/bz2_nearness.py [A|B]        # A ~2 min, B ~1 min
.venv/bin/python -I hw/realset/bizarre/bz3_wpe_far.py -j 2          # ~2 min
.venv/bin/python -I hw/realset/bizarre/bz3b_wpe_snr.py              # ~4 min
.venv/bin/python -I hw/realset/bizarre/bz1_speaker_id.py            # ~2.5 min
.venv/bin/python -I hw/realset/bizarre/bz4_overlap_tse.py           # ~8 min
.venv/bin/python -I hw/realset/bizarre/bz4b_refcheck.py             # ~3 min
.venv/bin/python -I hw/realset/bizarre/bz6_skew_sensitivity.py      # ~1 min
.venv/bin/python -I hw/realset/bizarre/bz6b_incoherent.py           # ~1.5 min
.venv/bin/python -I hw/realset/bizarre/bz6c_oracle_mvdr.py          # ~1 min
```
