# Mic spacing: design impact and change list (B2)

2026-11-03. Inputs: A1 (spacing/skew fit), V1 (its check), A2 (skew mechanism), B1 (physics vs spacing), V2 (its check),
`methods/x_s0v2.md` (experimental blind S0 on 62 board recordings) and `study/SCENES.md` (paired 12 mm vs 40 mm scenes).
New numbers: `hw/spacing/b2/b2_numbers.py` → `b2_numbers.json` (analytic, reuses V2's verified solvers; 4 cm values match
V2 exactly). All AI-DI values are ideal (free-field omnis, ideal diffuse noise); subtract 0.5–1.5 dB for field losses.

## 1. What the hardware is today

- **Spacing ≈ 1.3 cm (plausible 1.2–1.5 cm)**, which fits your "1 to 2 cm".
  - The direct sound requires d ≥ 1.2 cm.
  - Coherence alone cannot pin d down at this size: it gives an isotropic-equivalent 1.1–1.3 cm, rising to 1.5–2.0 cm in quiet takes.
  - Two 14 mm boards side by side with centred ports would be ≥ 1.4 cm.
  - Calipers settle it.
- **Inter-channel skew = one I2S frame plus a fraction that is set at start-up.**
  - Measured: −1.12 samples in boot 1 and −1.01 to −1.02 after the esptool resets. That is −63 to −70 µs, with Mic B early.
  - The current S0 allows only {0, +31.25 µs}, so it leaves the whole skew in place.
  - At 1.3 cm the skew is 1.5–1.9× the largest physical TDOA (d/c = 36–41 µs), so uncorrected, every source appears beyond endfire.
- **Not yet verified:** ch0 = the L/R-to-GND mic (needs a tap test). Both MacBook speakers sat near ch0's endfire.

## 2. Recommendation

**Move the two mic boards apart to 4 cm, measured between the centres of the sound holes.** Anything from 3.5 to 4.5 cm is
fine; measure the result to ±0.5 mm. Mount both boards on one rigid strip with both holes facing the same way, then redo the
two-sweep calibration (§6).

Why 4 cm, compared with the present ~1.3 cm:
- About 1 dB more spatial gain in both use cases, and 3× finer direction resolution.
- It matches what DESIGN, the bench and every tuned threshold already assume.
- The boot-to-boot skew change hurts about 6× less.

If the device must be small, 2.5 cm is acceptable: 0.1–0.3 dB behind 4 cm in both cases. Keep the present side-by-side
mounting only as a development rig. Its spatial stage adds only 0.15–0.3 dB of ΔSNR on average (simulation and the real
board takes), and it needs a fresh skew calibration accurate to ≤ 5 µs at every boot.

## 3. Options

| | ~1.3 cm (keep; 1.25–1.4) | 2.5 cm | **4 cm** | 6 cm |
|---|---|---|---|---|
| **Bus** two-talker beam, partners on the axis, DESIGN bus WNG schedule, 2-D babble (3-D) | 1.5–1.7 (2.3–2.5) | 2.25 (3.5) | **2.5 (3.9)** | 2.4 (3.7) |
| Bus single target on the axis / azimuth-averaged, 2-D | 4.0–4.1 / 2.3–2.4 | 4.3 / 2.6 | 4.0 / 2.7 | 3.7 / 2.7 |
| **Meeting** (WNG ≥ −3 dB): azimuth-averaged 2-D / on the axis 3-D / broadside | 1.7–1.8 / 3.3–3.4 / 0.15 | 2.2 / 4.1 / 0.5 | **2.5 / 4.3 / 1.0** | 2.6 / 4.3 / 1.5 |
| M4 CDR usable from, partner at endfire / 45° / broadside (kHz) | 0.8–0.9 / 1.1–1.2 / 4.4–4.9 | 0.44 / 0.62 / 2.5 | 0.27 / 0.39 / 1.5 | 0.18 / 0.26 / 1.0 |
| M5 phase fence, ideal rejection of a passenger at broadside / 45° (γ 4, floor −10.5 dB) | −6 / −1 dB | −9 / −3 dB | −10 / −5 dB | −10 / −6 dB (band only 1–2.3 kHz) |
| TDOA range; gap between side-by-side partners at 20° and 50° | ±36–41 µs; 11–12 µs | ±73; 22 µs | ±117; 35 µs | ±175; 52 µs |
| 20 meeting seats: resolvable TDOA groups at 20 / 10 µs | 4 / 6–7 | 6 / 10 | 8 / 12 | 11 / 14 |
| WNG needed for 3 dB on-axis DI at 500 Hz / 1 kHz (self-noise and wind gain) | −11 to −12 / −5 to −6 dB | −6.2 / −1.1 | −2.6 / +1.4 | 0.0 / +2.7 |
| Cost of a residual skew error of 6 µs (the per-boot change) / 33 µs (worst blind S0 on simulated scenes) | 0.16–0.18 / 5.2–6.3 dB | 0.07 / 1.8 | 0.03 / 0.9 | 0.02 / 0.6 |
| Spatial aliasing c/2d | 12–14 kHz | 6.9 kHz | 4.3 kHz | 2.9 kHz |
| Paired simulation, current methods with factory calibration: bus ΔSNR_active (m2 / m4); meeting mean ΔESTOI (m2) | +0.15 / +0.15 dB; 0 (stays on Mic A) | not run | +0.55 / +0.36 dB; +0.022 | not run |
| **Practical cost** | None. But calibration-critical: ±1 mm = 8 % of d, and blind d is unidentifiable. Every µs threshold needs rescaling, plus a new bench spacing and matching noise | 10–15 min re-mount and recalibration. Still pocket-sized. Bench has 25 mm scenes | Same effort; strip ≥ 5.5 cm long. Matches DESIGN, the bench prior and the tuned constants. Blind S0 can check d (sinc zero at 4.3 kHz) | Strip ≥ 7.5 cm. Aliasing inside the speech band limits S1 and M5 to ≤ 2.3 kHz |
| **Verdict** | Development rig only | Good compact choice | **Recommended** | Dedicated meeting node only |

Reading:
- From 1.3 to 4 cm, the bus two-talker beam gains 0.8–1.0 dB, and the meeting gains 0.7–0.8 dB (azimuth-averaged) or 0.9–1.1 dB (on the axis).
- 4 cm is within 0.3 dB of the best spacing in every beam row, except broadside DI, where 6 cm is 0.5 dB better. The
  on-axis single target peaks at 2.5 cm, the meeting average at 6 cm.
- In simulation, at either spacing, worst-partner ESTOI moves by ≤ 0.003. Spacing is a ~1 dB lever; it does not decide intelligibility.

## 4. S0 changes

*Needed at any spacing*

1. **Drop the skew lattice.** Replace `calib.py` SKEW_STEP (allowed values {0, +31.25 µs}) with a continuous skew, written as integer frame + fraction.
   - The lattice gave τ0 = 0 on 16/16 board-like scenes.
   - `x_s0v2` is within 0.06 samples on all 56 board takes (P90 0.02 from 1 s of near-diffuse noise).
2. **Hardware prior.** In `calib.json`, store the integer frame (−1, B early) and the verified channel mapping.
   - Re-estimate the fraction at every start, searching −1.3 to −0.8 samples. The window rules out gross misfits such as
     the 0.53-sample (33 µs) errors seen on simulated scenes, which cost 5–6 dB at 1.3 cm. Inside it, the board
     estimator's ±0.03 samples (2 µs) cost ≤ 0.05 dB.
   - Never store a fractional value as truth: the skew went from −1.12 to −1.02 across resets.
3. **Slip / swap detector.** After removing the integer frame, the diffuse slope must lie within ±0.3 samples and talker TDOAs within ±1.2·d̂/c.
   - A population about ±1 sample off means a slot slip.
   - Fix it with a one-sample shift, then re-check the channel identity.
4. **Replace "fall back to the nominal 4 cm"** (DESIGN §2.0 step 5; `calib.py` uses `info['spacing_m']`, default 0.04).
   - Hardware: fall back to the acoustic spacing in `calib.json` (endfire or flip sweeps, cross-checked with calipers).
   - Bench: fall back to the device family's nominal.
   - Never fall back to a hard-coded 4 cm: at 1.3 cm it mis-scales every TDOA prior by about 3×.
5. **Fix `calib._override`.** It rebuilds Γ̂n rotated by the *blind* skew.
   - On the board this gives 0.36 instead of 0.84 at 4 kHz.
   - Recompute Γ̂n from `st.Sn` with the json correction, or always ship `gamma_n` in `calib.json`.
6. **Phase fit band.** Change 100–2000 Hz to 150–6500 Hz, using |Γ|-weighted Huber least squares (the board's coherence at 100–125 Hz is only 0.3).
   - Set the default HPF corners to [60, 60] Hz; they are not identifiable on the board.
7. **TDOA lower bound.** Compute d_lb = c·P99|τ̂| only after the continuous skew correction. With τ0 snapped to 0, a
   talker on Mic B's side reads |τ̂| ≈ d/c + 66 µs, so d_lb ≈ 3.5 cm.
   - If d_lb > 1.25 × the stored d, flag a calibration fault instead of raising d̂.
8. **Accuracy targets.** Residual skew ≤ 5 µs at 1.3 cm (≤ 0.15 dB) and ≤ 10 µs at 4 cm.
   - Directional noise biases the blind slope by about q·τ_d: a median of 0.07–0.17 samples and up to 0.53 on simulated scenes.
   - Prefer late-reverb frames after talker offsets, plus the phase-linearity gate. Report the SE and a confidence value.
9. **Data hygiene.**
   - Discard ≥ 0.5 s after any I2S start.
   - Use the diffuse-field gain (B re A −0.67 dB) for Γ̂n. The −1.3 dB near-endfire direct value belongs in the RTF (M3).

*Needed while the array is below about 2 cm*

10. **Spacing search range.** Change `D_RANGE` from 1.5–8 cm to 0.8–8 cm, and the fit band from 200–7500 Hz to 400–6500 Hz.
    - Below 300 Hz the board has incoherent junk; the INMP441 pass band ends at 6.77 kHz.
11. **Unidentifiable fit.** Below 2 cm the coherence fixes only an isotropic-equivalent d:
    - sinc and J0 differ by < 0.006;
    - with the anisotropy free, the fit residual is flat over 1.0–2.4 cm;
    - self-noise turns 1.2 cm into 1.6–2.0 cm in a quiet room.

    Declare d̂ *unidentified* if any of these holds: fit RMS > 0.15; < 1.5 s of noise; the field is < 15 dB above self-noise; the fit hits a grid edge; SE > 25 %; or d < 2 cm with sinc and J0 residuals within ±20 %.

    When unidentified, keep the measured Γ̂n (its shape is right) and take the steering d from `calib.json`.

## 5. Method changes for small spacing (any d̂ < 2.5 cm; harmless at 4 cm)

- **Scale every absolute-µs constant by d̂/(4 cm)**, with a floor of about twice the measured per-frame TDOA scatter (3–4 µs at 12 mm in simulation):
  - `tracker.py`: PEAK_SEP 25 µs (= 0.69·d/c at 1.25 cm), POST_WIN 20, HIST_SIGMA 6, GRID_STEP 2 (step ≤ d̂/c/20);
  - `beams.py`: NOISE_EXCL 40 µs and its ±20 µs window. At 12 mm this window covers the whole ±36 µs range, which is why m2_meeting_beam never leaves Mic A. Also STEER_HYST 10 µs;
  - `spk.py`: TAU_THR 15;
  - `agc.py`: tau_bin_s 10;
  - `cacgmm.py`: GUARD_SEP 30;
  - `tse.py`: SELF_TAU_TOL 10. At 1.3 cm this falls below the TDOA noise, so use explicit enrollment.
- **S1**
  - Raise the SRP band's top from 3.5 kHz to min(6.5 kHz, 0.8·c/2d̂): no aliasing below 12 kHz, and resolution grows with f.
  - Flag "calibration fault" when > 20 % of direct frames pile up at the ±1.2·d̂/c edge.
  - Expect side-by-side partners to be inseparable: they are 11 µs apart.
- **M2**
  - The bus WNG schedule must follow d. At d̂ < 2 cm, hold −10 dB up to about 3.5 kHz: the two-talker beam goes from 1.5 to 1.6 dB (2-D).
  - Alternatively, use a noise-aware floor, max(−20 dB, −(noise − self-noise − 6 dB)) per band. A −20 dB floor restores the 4 cm value (2.65 dB), but needs ±0.25 dB / ±1° calibration: uncalibrated ±1 dB / ±3° costs 1.0 dB at 1.2 cm against 0.24 dB at 4 cm.
  - Dipole band at 1.25 cm: from 1.97 kHz (−10 dB) or 3.16 kHz (−6 dB).
  - Meeting: drop the broadside beam of `fixed3` (it is delay-and-sum: 0.14 dB). MAX_TDOA 1.1·d̂/c is valid only after a correct skew.
- **M3:** geometry-free, so it absorbs gain and residual-skew errors; prefer it over M2 at small d. Its plausibility check needs skew-corrected τ_k, and Φ_w needs the measured Γ̂n.
- **M4**
  - The tuned `cond=0` uses raw CDR everywhere. Below 2 cm, re-enable conditioning (|Γs − Γn| ≥ 0.2) and fill the remaining bins from the 1–6.5 kHz median.
  - Retune µ, G_min and λ at 12.5 mm.
  - It helps only partners within about 45–60° of the axis (usable from 1.1–1.2 kHz at 45°, from 4.4–4.9 kHz at broadside).
  - 1° of phase error is 8 % of the 1 kHz phase range at 1.25 cm (2.4 % at 4 cm).
- **M5:** define the fence on the TDOA error normalized by d̂/c (γ ∝ (4 cm/d̂)²) and extend the band to min(6.5 kHz, 0.8·c/2d̂). Calibration phase error and self-noise then dominate, so it is P3 at small d.
- **M7:** run EM only at ≥ 1.5–2 kHz, where partner and diffuse classes differ (Γn ≥ 0.9 below 3.6 kHz), and carry frame posteriors down. P3 at small d.
- **What carries the load at ~1.3 cm:**
  - M6, with its noise-update gate driven by VAD or by M8/M9 target presence, because the CDR and fence cue is weak;
  - M8, the main babble reducer;
  - M9, the only stage that removes bystanders, with product enrollment;
  - M10, with diagonal loading (two channels add little diversity below 3.6 kHz);
  - M11, embedding-keyed (the |τ̂| prior has about 4 bins).

## 6. Bench changes

1. **Spacing family.**
   - If the board stays small: add 12.5 mm to `SPACINGS` (25/40/60 mm) in `lab/scene_bus.py` and `lab/scene_meeting.py` for dev, wide and test.
   - If the user moves to 4 cm: keep 40 mm as the main spacing and keep 12.5 mm as a robustness stratum.
   - `scenes3/spacing` (paired 12 vs 40 mm, test-split speakers) stays a study set.
2. **Skew defaults to the board.** `draw_impairments` should draw skew_us = −62.5 − U(0, 8) µs (B early) in about 2/3 of scenes, 0 or +31.25 µs in the rest, plus a rare slot-slip case.
   - `scene2` delays Mic B by +skew_us, so the board needs negative values. They work with the FFT phase and PAD_PRE = 512; `scenes3` already uses −63.125 µs.
3. **Board impairments:**
   - incoherent per-mic noise below 60 Hz at about −35 dBFS;
   - B gain −0.7 dB in diffuse noise, −1.3 dB for near-endfire direct sound;
   - a variant with sensor noise +10 dB (the unexplained board floor).
4. **What methods are told.** `lab.bench2` hard-codes `info = {scenario, spacing_m: 0.04}`. Add the two modes `study/run.py` already has:
   - `factory`: `calib_json` with spacing, the integer skew and gains, but not the per-boot fraction;
   - `blind`: the device's own prior.
5. **Matching real noise at ~1.3 cm.** DEMAND pairs are ≥ 5 cm apart. At 2 kHz a 5 cm pair has sinc coherence 0.53 against 0.97 at 1.25 cm, so `noise_real.field` hands methods far easier noise than the board hears.
   - Synthesize matching pairs by *coherence transplant*:
     1. Take the complex coherence of a 5 cm pair, γ5(f), measured in diffuse blocks.
     2. Set the target γT(f) = γ5(f·dT/5 cm). This needs γ5 only up to 2–2.6 kHz, where it is well measured, and it keeps the field's anisotropy.
     3. Per STFT bin, X_A = N1 and X_B = γT·N1 + √(1−|γT|²)·N2. N1 is one channel; N2 is a channel ≥ 15 cm away, made orthogonal to N1 and equalized to its PSD.
   - Validate that the output's Welch coherence is within ±0.02 of γT and that the B spectrum is within 0.5 dB of A.
   - The same scaling (0.8) fixes the 5 vs 4 cm mismatch.
   - Best of all: 2–5 min of bus, café and meeting noise recorded with the board at its final spacing.
6. **Gates and tests.**
   - Report everything per spacing and recompute the Appendix B numbers for 12.5 mm.
   - The G1 bus targets assume 4 cm physics: at 12 mm the simulated Tier-1 ΔSNR_active is +0.15 dB.
   - Add ablations: lattice vs continuous skew, 12.5 mm true with a 40 mm prior, and a residual-skew sweep of 2 / 5 / 10 / 33 µs.

## 7. Hardware steps for you (no Claude-side recording)

1. **Tap test.**
   - While recording, tap lightly next to the sound hole of the mic whose L/R pin goes to GND, or cover that hole while noise plays. Channel 0 must react (> 10 dB).
   - If channel 1 reacts instead, the labels are swapped. The −1 frame is then probably a slot-pairing offset.
2. **Calipers.** Measure the present hole-centre to hole-centre distance and write it down.
3. **Optional: confirm the skew reading before moving anything.**
   - Without a reset or reflash, play one 3 s sweep (`hw/stim/sweep_L.wav`) from a speaker 0.5–1 m away, roughly along the mic axis. Then rotate the board 180° about its own centre and play it again.
   - The skew is τ0 = (t1 + t2)/2.
   - Expected: τ0 ≈ −1.0 to −1.13 samples. ≈ −0.5 would mean half-frame skew with broadside speakers. ≈ 0 would mean no skew and d ≈ 3 cm.
   - If the board and laptop have not moved since session 1 and nothing was reset since 01:55, the left speaker should
     give t1 ≈ −0.4 and t2 ≈ −1.6.
4. **Re-mount at 4 cm (or 2.5 cm).**
   - Use one rigid strip (perfboard, plastic ruler or 3D print) with foam tape.
   - Both sound holes face the same way, in the same plane, 40 ± 0.5 mm apart on a straight line.
   - Keep nothing over or between the holes. Put the ESP32 and the wires behind the strip, and use a soft base.
   - Keep the L/R wiring unchanged: Mic A to GND, Mic B to 3V3.
5. **Calibrate with two 3 s sweeps in the same boot** (no reset, power cycle or reflash between them).
   - Point the strip's axis at the speaker, 0.5–1 m away, with Mic A nearer: this gives t+. Rotate 180° so Mic B is nearer: this gives t−.
   - Read t± by GCC-PHAT on a 4 ms window at the direct sound, 300–6000 Hz (`hw/analyze.py` `gcc_phat_tdoa`).
   - Compute τ0 = (t+ + t−)/2 and d = 343·(t+ − t−)/(2·16000) m.
   - Expected at 4 cm (d/c = 1.87 samples): t+ ≈ +0.73…+0.87 and t− ≈ −2.87…−3.0 samples.
   - Pass if d is within 2 mm of the calipers, τ0 is within −1.15…−0.95, and a repeat sweep agrees within 0.05 samples.
   - Optional: a broadside sweep should read τ0 ± 0.1.
6. **Record ambient noise:** 30 s with the fans running (for gain and Γn) and 10 s of near-silence.
   - Save `calib.json`: spacing, integer skew, channel map, gains and `gamma_n`, but not the fractional skew.
7. **Firmware hygiene.**
   - Keep I2S running between takes and drop 0.5 s after each start.
   - Log reset and boot events.
   - Do 3–5 power cycles with 10 s of ambient each, to map how much the fractional skew moves.
8. **Placement.**
   - Bus: axis through both partners' mouths, device between them, within 0.5–0.6 m.
   - Meeting: axis along the table's long axis, holes within about 2 cm of the tabletop.

## 8. Open questions

1. The exact present spacing: the data allow 1.2–1.5 cm and cannot pin it down without a geometry-controlled sweep or calipers.
2. Channel identity, and whether the −1 frame is a host slot-pairing offset (labels swapped) or a mic-internal L/R latency.
3. How the fractional skew varies across power cycles, reflashes and I2S restarts. Only two values have been seen: −1.12 and −1.02.
4. The board's incoherent noise floor, 13–23 dB above the INMP441 datasheet self-noise, has no known source.
   - A concurrent run's preliminary numbers (`hw/realset/bizarre/bz6b_incoherent.json`) put the excess mainly at 150–500 Hz, at 10–24 % of the noise power.
   - That is exactly the band where a small array needs its lowest WNG, so it penalises 1.3 cm more than 4 cm.
5. Does the direction-dependent B gain deficit (−0.7 vs −1.3 dB) come from one board shadowing the other, so that it disappears at 4 cm?
6. How anisotropic real bus and meeting fields are at the device. That decides blind spacing and Γn; at 4 cm the sinc zero near 4.3 kHz can test it.
7. Is a device ≤ 3 cm long required? That decides between 2.5 and 4 cm.
8. Should the G1/G2 targets be spacing-specific?
9. Is the coherence transplant valid for near sources (passengers at 0.6 m)? Check it against a real bus recording made with the board.
10. Housing and PCB scattering are not modelled at any spacing.
