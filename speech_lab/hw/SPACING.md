# Mic spacing and channel skew

2026-11-03. The single reference for the measured mic spacing and inter-channel skew of the ESP32-C3 + 2 × INMP441 board, and for what they change.
Data: session 1 (`hw/rec`, `hw/irs.npz`) and the later takes in `hw/rec2` and `hw/realset`, which include 8 esptool resets at 01:51–01:54.
Inputs: A1 fit (`hw/spacing/a1/`), V1 check of A1 (`v1/`), A2 mechanism review (`a2/notes.md`), B1 physics by spacing (`b1/`), V2 check of B1 (`v2/`), B2 design impact with the full change list (`hw/spacing/IMPACT.md`, `b2/`).
Convention: τ = arrival at Mic B (ch1) − arrival at Mic A (ch0), in samples at 16 kHz (1 sample = 62.5 µs = 64 SCK periods); τ < 0 means B hears first. Measured τ = τ0 (electronic skew) + g (geometric TDOA, at most d·fs/c in size).
Supersedes `HW_TESTS.md` findings 3 and 8, and the "one direction" reading of finding 4.

## Summary
- **Spacing:** the two mics are about **1.3 cm** apart (plausible 1.2–1.5 cm, 1.6 not excluded), in line with your "one to two centimetres". The design, the bench and every tuned threshold assumed 4 cm.
- **Skew:** Mic B samples about **one I2S frame later** than Mic A, so in the recordings B seems to hear everything about one sample early: −1.12 samples (−70 µs) in the first boot, −1.01 to −1.03 (−63 to −64 µs) after the resets. The whole frame was present in all 9 boots seen; the fraction is set at each start-up.
- **How sure:** the skew is known to about ±0.03 sample within a boot, and it is electronic: fan noise and room reverb share it, and a reset moved the direct sound, the reverb and the fan-noise phase of an untouched setup by the same ≈ 0.1 sample. The spacing is known only to about ±0.15 cm, because at this size the coherence cannot pin it down; calipers or two endfire sweeps will.
- **What it changes:** the skew is about twice the largest possible acoustic delay, so S0 must remove it at every start (residual ≤ 5 µs) before any spatial stage; S0's {0, 31.25 µs} skew lattice and its 4 cm fallback are wrong for this board. At 1.3 cm, spatial processing helps only above about 1 kHz and for talkers near the mic axis, gives 0.7–1.1 dB less than at 4 cm, and every µs threshold must scale with d.
- **Recommendation:** re-mount the mics 4 cm apart (2.5 cm if size matters) on one rigid strip, then tap-test, measure with calipers and calibrate with two endfire sweeps in one boot (§5). Keep the present board as a small-spacing development rig.

## 1. Measurement

### 1.1 Values
| Quantity | Value (95 % CI or range) | Basis (source) |
|---|---|---|
| Skew τ0, boot 1 (01:18–01:49, all session-1 takes) | −1.10 [−1.14, −1.07] (A1); −1.12 ± 0.03, IQR −1.133…−1.107 (V1) | Fan-noise phase slope, 13 stretches (median −1.124); reverb 30–250 and 80–250 ms of 5 sweeps (−1.111); cross-correlation (−1.135) |
| Skew τ0 after the resets (to 02:35) | −1.013, IQR −1.017…−1.007; reboot takes cluster at −1.031 (2) and −1.015 (6); s3 reverb −1.02 (V1) | 29 stretches: reboot, realset, s3 and drift takes |
| Change across the resets, board untouched | +0.095 = 6.08 SCK periods, a pure delay (residual 0.5° rms, intercept 0.0°); same-boot control −0.02 (V1) | Double difference s2_sweep_L → s3_sweep_L over −1…+5 ms, 300–6500 Hz |
| Frequency dependence of τ0 | None: band slopes flat 0.3–7 kHz within ±0.1, intercept 0 ± 3°, residual ±2° from 160 Hz to 2 kHz (A1) | No measurable high-pass-corner phase mismatch above 160 Hz; undeterminable below ~150 Hz (coherence 0.27–0.36 at 100–125 Hz) |
| Direct-sound TDOA (first arrival) | Boot 1: −0.50…−0.53 for L, R, L2 and talkers A, B, C; boot 2: −0.41 (s3_sweep_L) (A1, V1) | Windowed GCC-PHAT and phase LS, 300–6000 Hz; the session-1 4 ms-window values (−0.53…−0.56) lean toward the skew as early reflections mix in |
| Geometric TDOA g = direct − τ0 | +0.59 ± 0.04 (A1); +0.60 ± 0.03 in both boots (V1); leading-edge timing +0.53…+0.56 | The same for both speakers, and for talker B vs A (≤ 0.05 apart) |
| Lower bound d ≥ g·c/fs | 1.18–1.35 cm (A1), 1.14–1.35 (V1), 1.16–1.27 (V2), 1.09–1.26 (B1) | Spread = estimator and skew choice; together d ≳ 1.2 cm |
| d from diffuse coherence, 3-D isotropic (sinc) | 1.25 [1.10, 1.40] cm (A1 bootstrap). V1, 1–6.5 kHz: fan 1.24 / 1.16 (boot 1 / 2), reverb 1.18–1.20, s3 1.06–1.18; 1–5 kHz 1.26–1.44; same-boot drift takes 1.18–1.54 | An isotropic-equivalent d only (§1.3) |
| d from geometry (g equal for both speakers, so the mic axis bisects them) | 1.23–1.36 cm at 1 m, 1.26–1.39 at 0.5 m, 1.33–1.47 at 0.3 m, 1.45–1.60 at 0.2 m (V1) | Board distance was not observed |
| **Adopted** | **d ≈ 1.3 cm, plausible 1.2–1.5, up to 1.6 not excluded** (1.25 cm is a fine working value). **τ0 = −1 frame + a per-boot fraction of 0 to −0.12** | V1, B2. Above 2 cm is excluded only by your 1–2 cm. B2: two 14 mm breakouts side by side with centred ports would be ≥ 1.4 cm |

- **Inferred geometry (H1; not observed).** Both speakers sit within about 0–25° of ch0's (Mic A's) endfire if d ≲ 1.35 cm, about 35° at 1.5 cm. The first-arrival path difference r_R − r_L ≈ 6–11 cm (if the Mac's L/R outputs are synchronous) puts the board ≥ 0.2 m away, mostly front-left of the laptop, with Mic A toward it; you can confirm which mic was nearer. The right speaker's direct path looks partly obstructed (about 6 dB weaker than the left's; DRR −1 vs +5 dB).
- **Coherence caveats.** A stationary incoherent floor of −110 to −120 dB re FS²/Hz (13–23 dB above the INMP441 datasheet self-noise, source unknown) lowers the loud-fan coherence by 5–16 % above 4 kHz (plain fit 1.28 cm, floor regression 1.19 cm). Above 6.5 kHz the coherence flattens at 0.7–0.83 against a model 0.58–0.67 (band edge or aliasing). Fit 1–6.5 kHz only.
- **V1 checks that found no problem:** sign convention (synthetic b(t) = a(t + 1.1) gives −1.100), fractional delay inside Welch frames (a synthetic diffuse pair returns −1.100 ± 0.002 and 1.250 ± 0.005 cm), the 17 ppm clock drift, deconvolution regularisation.

### 1.2 Hypotheses
| | Claim | Verdict | Deciding evidence |
|---|---|---|---|
| H1 | τ0 ≈ −1.1, d ≈ 1.2–1.3 cm, fan and late reverb near-diffuse, both speakers near ch0's endfire (g ≈ +0.6) | **Supported** | Fan noise and both speakers' reverb share one linear phase slope; a diffuse cross-spectrum is real, so a slope common to all is a channel delay. The reset moved the direct sound of an untouched path by the same pure delay as the fields, which needs no coherence model (V1). Best 3-parameter fit (rmse 0.062 on the reverb). |
| H2 | τ0 = −0.5 (half frame), speakers broadside, directional fan, d ≥ 1.4 cm | Refuted | Needs 76–79 % of the energy as a plane wave exactly at Mic B's endfire, in both speakers' late reverb and in the fan alike, and still fits worse with an extra parameter (rmse 0.071 vs 0.062 reverb, 0.167 vs 0.151 fan); with d free the fan fit wants 5.8 cm. −1.10 is 2.2 half-frames. The geometry needs the board ≥ 1.5–2 m away, which the left speaker's DRR of +5 dB argues against. |
| H3 | τ0 = 0, d ≥ 2.5 cm | Refuted | Diffuse coherence 0.77 / 0.74 at 5 / 6 kHz against 0.33 / 0.14 predicted at 2.5 cm; rmse 0.21–0.27 vs 0.06–0.15; reverb slope −1.11, not 0; only placements inside the laptop fit; your 1–2 cm. |
| H4 | Fan = directional point source at −1.15 (HW_TESTS 4, 8) | Refuted | Floor-corrected fan coherence 0.92 / 0.88 / 0.77 / 0.74 at 3 / 4 / 5 / 6 kHz follows sinc at 1.25 cm (0.92 / 0.87 / 0.80 / 0.71); its slope is the skew, so its net g ≈ 0 ± 0.05. Only 2–22° of residual phase at 2–7 kHz is directional. |
| H5 | 2-D (J0) field, d ≈ 0.9–1.0 cm | Refuted (rests on H1) | J0 fits as well as sinc (d 0.86–1.09 cm) but violates d ≳ 1.2 cm from the direct sound. |
| H6 | Skew = whole frame (slot pairing, maybe swapped labels) + ≈ 0.1 from ports or mic | Partly decided | The ≈ 0.1 part changed across a software reset with the board untouched, so it is electronic and boot-dependent, not port, gasket or air path (V1). The −1 frame held in all 9 boots. Channel identity is untested (tap test). |

### 1.3 Where estimators and verifiers disagree
- **Is the skew constant?** A1 gave −1.10 as a fixed correction, and A2 expected a deterministic skew with the ≈ −0.1 coming from analog and fit systematics. V1's reset takes show the fraction moves at start-up (now ≈ −1.015): applying −1.10 today leaves ≈ 0.1 sample (6 µs), 17 % of the ±0.6-sample TDOA range, and A1's flip predictions hold only for boot 1. **V1 governs.**
- **Spacing interval.** A1's [1.10, 1.40] is a bootstrap inside one model (isotropic sinc) and one field set. V1: the low end violates the direct-path bound; with the field's anisotropy free, the fit rmse is flat within ±1 % from 1.0 to 2.4 cm in all six data sets, so the high end reflects the isotropy assumption, not the data. A1's own 4-parameter mixture with τ0 = −1.0 and d = 3.6 cm fits the reverb slightly better (rmse 0.057 vs 0.062). B2 adds that self-noise makes blind fits read 1.6–2.0 cm in quiet rooms. **Use 1.2–1.5 cm until measured.**
- **J0 vs sinc.** Indistinguishable at this size. A1 rejects J0 (smaller d) through the direct-path bound, which rests on H1; V1 agrees, but notes that nothing excludes broadside-heavy fields, which give a larger d.
- **IR-tail SNR.** A1: 15–35 dB above the deconvolved fan floor at 80–250 ms. V1: 5.5–24 dB (sweep_R at 1–5 kHz only 5.5–6.3 dB), and the 250–500 ms windows, like the main agent's 400–700 ms windows, are deconvolved fan noise. So the "three unrelated fields" are partly one field; the slope is still unbiased, because the fan carries the same skew.
- **Early reflections (2–10 ms).** A1: −1.11…−1.17. V1 got −0.88…−1.50 across sweeps and bands (only a few discrete reflections). Not evidence for the skew.
- **Precision of a blind estimate.** A1: ±0.01 per 10 s of fan noise. V1: a field-dependent bias of ±0.02–0.03 (fan sub-bands −1.06…−1.23, drift_00 −0.985; the fan adds −2 to −9° at 2.2–4.8 kHz and up to +22° at 5–7 kHz). τ0 from diffuse fields also assumes no net energy flow along the axis (the 10–80 ms reverb gives −1.07). The 30–250 ms reverb is the cleaner reference. B2: `x_s0v2` is within 0.06 sample on all 56 board takes (P90 0.02 from 1 s of near-diffuse noise).
- **Speaker angle.** A1's "within 0–25° of endfire" holds only for d ≲ 1.35 cm (V1); B1 gives 11–24° at 1.2 cm and 22–36° at 1.36 cm.
- **Blind tracking with the skew left in** put talker A at −0.69 and talker B at −0.97…−1.03 (A1): two directions where the geometry has one, because B's weak direct sound is pulled toward the skew.

### 1.4 Flip-test predictions (valid only within one boot)
Rotating the board 180° about its own centre maps g → −g, so t_flip = 2·τ0 − t; the fan and reverb slope stays at τ0 under every hypothesis. Use the τ0 of the boot in which the test runs.

| Hypothesis | sweep_L / sweep_R after a 180° flip, boot 1 (range) | Endfire sweep_L, boot 1: Mic A nearer / Mic B nearer |
|---|---|---|
| H1 (τ0 −1.10, d 1.25 cm, g +0.59 / +0.60) | −1.69 / −1.70 (−1.63…−1.75) | −0.52 (−0.42…−0.62) / −1.68 |
| H2 (τ0 −0.5, d ≈ 1.6 cm, g ≈ 0) | −0.49 / −0.50 (−0.39…−0.59) | +0.25 (+0.10…+0.48) / −1.25 |
| H3 (τ0 0, d ≈ 3 cm, g ≈ −0.5) | +0.51 / +0.50 (+0.41…+0.61) | +1.40 (≥ +1.12) / −1.40 |

In the current boot (τ0 ≈ −1.015, sweep_L now −0.41), H1 predicts −1.62 after the flip (V1), not −1.69. Two endfire takes in one boot give τ0 = (t₊ + t₋)/2 and d = c·(t₊ − t₋)/(2·fs). A flip off the axis gives only (t₁ − t₂)/2 = (d·fs/c)·cos θ, a lower bound on d.

## 2. Why the channels are skewed
- **Vendor intent is zero skew.** TDK states that mics sharing a bus sample simultaneously: explicitly for the ICS-43434 (at the WS falling edge that starts the frame) and the ICS-52000 TDM mic. The INMP441 sheet says only that the mics are "in sync" and never says which frame each slot's word carries. No primary source supports a half-frame (31.25 µs) skew. The INMP441 group delay (17.2/fs = 1.08 ms) and the Philips one-SCK MSB delay are the same in both slots and cancel (A2).
- **The whole frame (−1, 62.5 µs) has two candidates, both rated medium (A2):**
  1. *Host slot-pairing offset:* the buffer pairs (R_n, L_n+1) instead of (L_n, R_n), for example through inverted WS polarity, the RX starting on the right slot (as an internal slave in full duplex) or one 4-byte word dropped. This also swaps the labels: ch0 would be the L/R = 3V3 mic.
  2. *Inside the INMP441:* L-mode and R-mode output differ by one frame of latency. Undocumented but not ruled out; it would recur identically at every start, with no label swap.

  The tap test (§5, step 1) separates them.
- **The fraction (0 to −0.12 sample, 0 to −7.5 µs) is set at start-up.** It moved by +0.095 sample, 6.08 SCK periods, across resets with nothing touched, so it is electronic. V1 reads the whole-SCK step as a hint, not proof, of mic timing that locks to the bit clock at start-up.
- **Determinism.** The integer frame was the same in every boot seen (boot 1 plus 8 resets, through 02:35); the fraction is not (−1.12 in boot 1, −1.01…−1.03 after). Start-up failures are documented on this I2S IP: the ESP32-C3 on IDF v5.0.1 came up bit-misaligned (a gain error) in about 1 of 10 restarts (esp-idf #11135, gone in later IDF), and the ESP32-S3 flipped its first slot across reboots in slave full-duplex TDM (#9513). A slot slip would show as a ±1-frame jump plus a label swap, so S0 must check at every start.

## 3. Physics by spacing
| d (cm) | | d/c µs (smp) | 68 µs ÷ d/c | c/2d kHz | 3-D Γ > 0.9 below, kHz | CDR usable from, endfire / broadside kHz | Dipole from, Hz | AI-DI at −10 dB: axis 3-D / az-avg 2-D / two-talker 2-D | Two-talker, bus schedule: 2-D (3-D) | Meeting −3 dB: az-avg 2-D / axis 3-D / broadside | Seat groups 20 / 10 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | n | 29.2 (0.47) | 2.33 | 17.2 | 4.29 | 1.10 / 6.18 | 2462 | 4.45 / 2.12 / 1.39 | – | – | 3 / 5 |
| **1.2** | n ◆ | 35.0 (0.56) | 1.94 | 14.3 | 3.58 | 0.91 / 5.15 | 2052 | 4.69 / 2.22 / 1.58 | 1.41 (2.14) | 1.65 / 3.18 / 0.13 | 4 / 6 |
| **1.5** | n ◆ | 43.7 (0.70) | 1.55 | 11.4 | 2.86 | 0.73 / 4.12 | 1641 | 4.93 / 2.34 / 1.80 | – | – | 4 / 7 |
| 2.0 | n | 58.3 (0.93) | 1.17 | 8.57 | 2.15 | 0.55 / 3.09 | 1231 | 5.12 / 2.47 / 2.04 | – | – | 5 / 8 |
| 2.5 | ✓ | 72.9 (1.17) | 0.93 | 6.86 | 1.72 | 0.44 / 2.47 | 985 | 5.15 / 2.54 / 2.18 | 2.25 (3.52) | 2.20 / 4.13 / 0.52 | 6 / 10 |
| **4** | ✓ | 116.6 (1.87) | 0.58 | 4.29 | 1.07 | 0.27 / 1.54 | 616 | 4.96 / 2.66 / 2.31 | 2.48 (3.86) | 2.47 / 4.32 / 1.04 | 8 / 12 |
| 6 | ✓ | 174.9 (2.80) | 0.39 | 2.86 | 0.72 | 0.18 / 1.03 | 410 | 4.61 / 2.72 / 2.25 | 2.36 (3.69) | 2.64 / 4.26 / 1.53 | 11 / 14 |

✓ reproduces DESIGN v1.1 §1 (B1: 42 of 43 checks; the miss is DESIGN printing 1924.6 Hz as 1.93 kHz). n = new spacing. ◆ = brackets the board. V2 recomputed every B1 cell independently (within 0.01 dB, 1 Hz, exact counts). The bus-schedule and −3 dB columns are V2's at 1.2 and 4 cm and B2's (with V2's solvers) at 2.5 and 6 cm; at 1.25 / 1.4 cm B2 gets 1.51 / 1.66 (bus schedule, 2-D) and 1.68 / 1.78 (meeting az-avg). Model: ideal free-field omnis in ideal diffuse noise, c = 343 m/s; AI-DI = DI weighted by ANSI S3.5 band importance, 160 Hz–8 kHz; subtract 0.5–1.5 dB for field losses. CDR usable where |Γs − Γn| ≥ 0.2 (M4 rule); dipole edge at WNG ≥ −10 dB; seat groups for the 20 seats of `meeting_s0`, median over 720 azimuths.

- **Skew vs geometry.** At 1.2 cm a 68 µs skew is 1.94× the largest TDOA: uncorrected, every source looks beyond endfire, and a geometry-steered axis beam gives −6.22 dB AI-DI (−0.84 dB at 4 cm). A residual skew of 2 / 5 / 10 µs costs 0.05 / 0.15 / 0.39 dB at 1.2 cm (0.01 / 0.02 / 0.08 dB at 4 cm), hence the ≤ 5 µs (0.08 sample) target. RTF-based M3 absorbs a fixed skew; model-steered M1, M2 and M4 do not.
- **Low frequencies.** WNG scales with d²: 3 dB of on-axis DI at 125 / 250 / 500 / 1000 Hz needs WNG −24.2 / −18.2 / −12.3 / −6.5 dB at 1.2 cm, 10.5 dB worse than at 4 cm. At the −10 dB floor the attainable on-axis DI there is 0.5 / 1.1 / 2.3 / 4.6 dB. Broadside talkers cannot be told from diffuse noise below 3.58 kHz (3-D) or 2.91 kHz (2-D babble).
- **The −10 dB columns understate the meeting penalty (V2).** At DESIGN's meeting floor (−3 dB), 1.2 cm is 0.82 dB (az-avg 2-D) and 1.14 dB (axis 3-D) below 4 cm, against 0.27–0.44 dB at −10 dB. With the bus WNG schedule unchanged, the two-talker beam falls to 1.41 dB (2-D) at 1.2 cm against 2.48 dB at 4 cm; holding −10 dB up to about 3.5 kHz restores 1.58 dB.
- **Calibration becomes mandatory below a −10 dB floor.** Uncalibrated ±1 dB / ±3° costs, on axis at −20 dB and 1.2 cm, 0.80 dB (B1, Monte Carlo) or 0.99 dB (V2, expected power), against 0.21–0.24 dB at 4 cm; calibrated to ±0.25 dB / ±1°, 0.10–0.11 dB.
- **Meeting identity.** The 20 seats collapse to 4 TDOA groups at 20 µs (6 at 10 µs); 16–19 of the 19 neighbouring-seat gaps are under 10 µs depending on orientation (V2; B1 said all 19). Identity must come from speaker embeddings. Near-field ILD is useless at 1.2 cm (0.34 / 0.21 / 0.10 dB at 0.3 / 0.5 / 1 m).
- **Scaling (V2 corrects B1's wording):** the TDOA span scales with d; band edges and the skew ratio scale with 1/d. The on-axis superdirective AI-DI peaks near 2–2.5 cm; the two-talker and azimuth-averaged values fall steadily below 2.5 cm.

## 4. What changes in the design
**S0** (DESIGN §2.0, `calib.py`; needed at any spacing)
- Replace the skew lattice (`SKEW_STEP`, {0, +31.25 µs}; it returned τ0 = 0 on 16/16 board-like scenes) with integer frame + continuous fraction. `calib.json` holds the integer (−1, B early) and the verified channel map; the fraction is re-estimated at every start within −1.3…−0.8 sample (A2 proposed −1 ± 0.3) and never stored (A2 proposed storing it; the reset data overrule that). `methods/x_s0v2.py` already estimates continuously.
- Correct the skew before S1, the CDR, the fence and d_lb. Prefer late-reverb frames after talker offsets with a phase-linearity gate: directional noise biased blind slopes by a median 0.07–0.17 sample (up to 0.53) on simulated scenes. Report SE and confidence. Target residual ≤ 5 µs at 1.3 cm, ≤ 10 µs at 4 cm.
- Slip and swap check: after removing the integer, the diffuse slope must lie within ±0.3 sample and talker TDOAs within ±1.2·d̂/c. A population about ±1 sample off is a slot slip: shift one sample and re-check the channel identity.
- Phase fit (step 3): 150–6500 Hz instead of 100–2000 Hz, |Γ|-weighted Huber; default high-pass corners [60, 60] Hz, since they are not identifiable on the board.
- Spacing (step 5): fall back to the acoustic d in `calib.json`, never to a hard-coded 4 cm (a ≈ 3× TDOA mis-scaling at 1.3 cm). Compute d_lb only after the skew correction, and flag a calibration fault if d_lb > 1.25 × the stored d. While d < 2 cm: `D_RANGE` 0.8–8 cm (was 1.5–8) and fit band 400–6500 Hz. Declare d̂ unidentified if the fit RMS > 0.15, there is < 1.5 s of noise, the field is < 15 dB above self-noise, the fit hits a grid edge, SE > 25 %, or d < 2 cm with sinc and J0 residuals within ±20 %; then keep the measured Γ̂n and take the steering d from `calib.json`.
- Fix `calib._override`, which rotates Γ̂n by the blind skew (0.36 instead of 0.84 at 4 kHz on the board). For Γ̂n use the diffuse-field gain (B re A −0.67 dB); the −1.3 dB near-endfire direct value belongs in the RTF (M3). Discard ≥ 0.5 s after any I2S start.

**Methods** (needed while d̂ < ~2.5 cm; harmless at 4 cm)
- Scale every absolute-µs constant by d̂/4 cm, floored at about twice the per-frame TDOA scatter (3–4 µs at 12 mm): `tracker.py` PEAK_SEP 25, POST_WIN 20, HIST_SIGMA 6, GRID_STEP 2; `beams.py` NOISE_EXCL 40 and its ±20 µs window (at 12 mm it covers the whole ±36 µs range, which is why m2_meeting_beam never leaves Mic A), STEER_HYST 10; `spk.py` TAU_THR 15; `agc.py` tau_bin_s 10; `cacgmm.py` GUARD_SEP 30; `tse.py` SELF_TAU_TOL 10 (use explicit enrollment instead).
- S1: SRP band top min(6.5 kHz, 0.8·c/2d̂) instead of 3.5 kHz; flag a calibration fault when > 20 % of direct frames pile up at the ±1.2·d̂/c edge; side-by-side partners (11 µs apart) are inseparable.
- M2: the bus WNG schedule must follow d (hold −10 dB to about 3.5 kHz, or use a noise-aware floor); a −20 dB floor restores the 4 cm value (2.65 dB) only with ±0.25 dB / ±1° calibration; drop the meeting `fixed3` broadside beam (it is delay-and-sum, 0.14 dB). At 1.25 cm the dipole band starts at 1.97 kHz (−10 dB) or 3.16 kHz (−6 dB).
- M3 (geometry-free) is preferred over M2 at small d. M4: re-enable the |Γs − Γn| ≥ 0.2 conditioning and retune µ, G_min and λ at 12.5 mm; it helps only partners within about 45–60° of the axis. M5 (fence normalized by d̂/c) and M7 (EM only above 1.5–2 kHz) drop to P3.
- The load moves to M6 (noise update gated by VAD or M8/M9 presence), M8, M9 (the only stage that removes bystanders), M10 (with diagonal loading) and M11 (embedding-keyed).

**Bench and DESIGN text**
- E8 skew: draw −62.5 − U(0, 8) µs (B early) in about 2/3 of scenes, 0 or +31.25 µs in the rest, plus a rare slot slip (an extra ±62.5 µs with swapped channels). `scene2` delays Mic B by +skew_us, so board values are negative (`scenes3` already uses −63.125 µs).
- Spacing: add 12.5 mm to `SPACINGS` in `lab/scene_bus.py` and `lab/scene_meeting.py` (main spacing if the board stays small, robustness stratum after a move to 4 cm). Board impairments: incoherent per-mic noise below 60 Hz at about −35 dBFS; B gain −0.7 dB (diffuse) / −1.3 dB (near-endfire direct); a +10 dB sensor-noise variant.
- `lab.bench2` hard-codes `spacing_m: 0.04`; add `factory` (spacing, integer skew, gains; not the fraction) and `blind` modes.
- Real noise: DEMAND pairs (≥ 5 cm) are much less coherent than what the board hears (sinc at 2 kHz: 0.53 vs 0.97 at 1.25 cm). Synthesize matching pairs by coherence transplant (`IMPACT.md` §6) or record noise with the final board.
- Gates: report per spacing; recompute Appendix B for 12.5 mm; the G1 bus targets assume 4 cm physics (simulated Tier-1 ΔSNR_active at 12 mm: +0.15 dB). Ablate lattice vs continuous skew, 12.5 mm truth with a 40 mm prior, and a residual skew of 2 / 5 / 10 / 33 µs.
- DESIGN.md (owned by the build runs, not edited here) still assumes a 0 / 31.25 µs skew (E8, §2.0 step 4, §5 item 1), a 4 cm fallback (§2.0 step 5) and 8 TDOA groups (§1.5, S1).

## 5. Recommendation and hardware steps
**Move the mics to 4 cm, port centre to port centre** (3.5–4.5 cm is fine; measure to ±0.5 mm), on one rigid strip with both ports facing the same way, and recalibrate (B2). From 1.3 to 4 cm: +0.8–1.0 dB for the bus two-talker beam, +0.7–1.1 dB in the meeting at its −3 dB floor, 3× finer direction resolution, about 6× less damage from the per-boot skew change, and no spacing-specific retuning, because DESIGN, the bench and the tuned constants already assume 4 cm. If size matters, 2.5 cm is within 0.1–0.3 dB of 4 cm; 6 cm only for a dedicated meeting node. Keep the present board as a small-spacing development rig: its spatial stage adds 0.15–0.3 dB of ΔSNR (simulation and real takes; +0.55 / +0.36 dB for m2 / m4 at 40 mm in the paired simulation) and needs a ≤ 5 µs skew calibration at every boot. Spacing is a ≈ 1 dB lever, not an intelligibility decider (worst-partner ESTOI moves ≤ 0.003 in simulation). Whatever the spacing, the skew lattice must go.

Hardware steps (for you; Claude does not play, record or open the serial port):
1. **Tap test.** While recording, tap next to (or cover) the port of the mic whose L/R pin goes to GND: ch0 must react (> 10 dB). If ch1 does, the labels are swapped and the −1 frame is probably a slot-pairing offset.
2. **Calipers.** Measure the present port-centre spacing (expected 1.2–1.5 cm).
3. **Optional, before moving anything, in the same boot:** play `hw/stim/sweep_L.wav` from 0.5–1 m roughly along the mic axis, rotate the board 180° about its centre, play it again. τ0 = (t₁ + t₂)/2 should be −1.0…−1.13 (≈ −0.5 means H2, ≈ 0 means H3). If nothing moved or reset since 01:55: t₁ ≈ −0.4, t₂ ≈ −1.6.
4. **Re-mount at 40 ± 0.5 mm (or 25).** One rigid strip (perfboard, ruler or 3D print) with foam tape; ports coplanar and facing the same way, nothing over or between them; ESP32 and wires behind; soft base; L/R wiring unchanged (A to GND, B to 3V3).
5. **Calibrate in one boot** (no reset, power cycle or reflash between takes). Endfire sweeps at 0.5–1 m, Mic A nearer (t₊) and Mic B nearer (t₋); GCC-PHAT on 4 ms at the direct sound, 300–6000 Hz (`hw/analyze.py` `gcc_phat_tdoa`); τ0 = (t₊ + t₋)/2, d = 343·(t₊ − t₋)/(2·16000) m. Expected at 4 cm: t₊ ≈ +0.73…+0.87, t₋ ≈ −2.87…−3.0. Pass if d is within 2 mm of the calipers, τ0 within −1.15…−0.95, and a repeat agrees within 0.05 sample; optionally a broadside sweep reads τ0 ± 0.1.
6. **Ambient:** 30 s with the fans running and 10 s of near-silence. Save `calib.json` with spacing, integer skew, channel map, gains and `gamma_n`, but not the fractional skew.
7. **Firmware (A2):** an RX-only master channel (`i2s_new_channel(&cfg, NULL, &rx)`) with the Philips default slot config unchanged (32-bit stereo, ws_pol false, no GPIO inversion), on an ESP-IDF newer than v5.0.1. Keep BCLK/WS running between takes, drop 0.5 s after each start, log resets, and report the integer lag of 1–2 s of ambient at boot. Optionally emit (ch0[n], ch1[n−1]) to remove the integer frame, paired with that check. If the tap test shows swapped labels, fixing the pairing should remove the whole frame; the per-boot fraction may remain. Do 3–5 power cycles with 10 s of ambient each to map the fraction.
8. **Placement:** bus, axis through both partners' mouths with the device between them, within 0.5–0.6 m; meeting, axis along the table, ports within about 2 cm of the tabletop.

Mic options with documented simultaneous sampling (A2; the host pairing still needs checking): ICS-43434 (2/fs group delay), ICS-52000 TDM (3–4 mics on one data line), or a stereo ADC with analog mics. PDM is not recommended on the C3, which also has only one I2S controller.

**Still open:** the exact spacing; channel identity and which one-frame mechanism applies; how the fraction varies over power cycles, reflashes and I2S restarts; the source of the incoherent floor (preliminary: mostly 150–500 Hz, where a small array needs its lowest WNG); whether B's −0.7 / −1.3 dB deficit is board shadowing that vanishes at 4 cm; field anisotropy at the device; whether the device must be ≤ 3 cm; spacing-specific G1/G2 targets; the coherence transplant for near sources; housing scattering (unmodelled at any spacing).
