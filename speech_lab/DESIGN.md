# Two-Mic Speech Capture: Algorithm Design and Benchmark Plan

**Hardware:** ESP32-C3 with 2 × INMP441 on one I2S bus (16 kHz, 24-bit), streamed to an M1 Pro Mac and processed in Python
**Version:** 1.1, 2026-11-02 (revised after design review; responses at the end)
**Audience:** engineers who will implement and benchmark in `speech_lab/` (the bench contract is in `speech_lab/BENCH.md`)
**Scope:** capture only. No far-end audio is played into the room; if it is, AEC comes first (§5).

**How to read this document**
- **§1 Physics.** What the hardware can and cannot do. Read it first; most bad ideas fail here.
- **§2 Pipelines.** The two target pipelines, plus the shared front end that every method uses (S0 calibration, S1 talker tracker).
- **§3 Methods.** Twelve candidate methods, M0–M11.
- **§4 Evaluation.** Benchmark protocol, metrics that cannot be gamed by gating, and pass/fail gates.
- **§5 Hardware.** Advice to pass on to the user.
- **§6 References.** Verified sources only; `[n]` in the text points here.
- **Appendices A–D** give the plan, unit tests, scripts and tooling. **Review responses** close the document.

---

## 0. Executive summary

**Scope.** Capture only: no loudspeaker plays far-end audio in the room. If one does, acoustic echo cancellation (AEC) and double-talk handling become mandatory and come before everything in this document (§5).

**What the hardware can do**
- **Directivity is low.** Two omni mics 2.5–6 cm apart form a first-order array. The maximum directivity is 6 dB (hypercardioid), and only toward the two ends of the mic axis. Toward the broadside plane it is about 1 dB.
- **Speech-weighted gain (AI-DI) at 4 cm**, ideal diffuse noise, white-noise-gain floor of −10 dB strictly enforced:
  - beam aimed at one talker on the axis: 5.0 dB in 3-D noise, 3.9 dB in horizontal (2-D) babble;
  - two-talker beam keeping two partners who face each other across the device: 3.6 dB (3-D), 2.3 dB (2-D). It works only when the partners are near the mic axis; when both sit near broadside it gives about 0 dB;
  - single-target beam averaged over device orientation: 2.7 dB (2-D). The critical distance therefore grows only ×1.36, not ×2;
  - subtract 0.5–1.5 dB for real-world losses.
- **One spatial degree of freedom per frequency.** A two-mic filter can null one source, or it can keep two partners. It cannot do both.
- **Distance is invisible to level.** The level difference between the two mics is at most 1.1 dB for a talker 0.3 m away at 4 cm spacing (1.6 dB at 6 cm). That is below calibration and reverberation error, so it is not a distance cue.
- **The only physical near/far cue is directness** (coherence, or direct-to-reverberant ratio). It works only inside the critical distance:
  - bus cabin: 0.6–1.2 m, for T60 between 0.6 s and 0.15 s;
  - meeting room: 0.9–1.4 m.
- **The two-mic ceiling is a few dB.** Most of the gain must come from single-channel noise reduction, WPE dereverberation, leveling and speaker-identity cues (§1.5).

**Test case 1: bus**
- "Only the two partners audible" cannot be achieved with classical two-mic processing in loud babble.
- The best classical stack is calibrated two-talker beam + coherence/phase masks + stationary noise suppression. With the mic axis through both partners and partners at 0.6 m or closer, I estimate 3–4 dB effective SNR gain: about 2.3 dB from the beam in 2-D babble plus 1.2–2.0 dB from the CDR mask, which partly overlap. This estimate is not measured yet, and orientation decides most of it.
- Removing an intelligible nearby passenger needs a speaker-identity cue: enrollment-based target-speaker extraction for both partners (M9). Changing the hardware also helps (§5).

**Test case 2: 20-person meeting**
- Far talkers are reverberant, not quiet. At 6 m the DRR is −13 to −17 dB, yet the total level is only 2–6 dB below a talker at 1 m.
- Two mics cannot tell 20 talkers apart by direction: the 20 seats collapse to a median of 8 resolvable TDOA groups. Per-talker memory (gain, dereverberation state) is keyed by speaker embeddings, with TDOA as a secondary cue.
- Planned stack: two-channel WPE dereverberation, a gentle beam, stationary noise suppression and speaker-keyed leveling.
- That stack evens out loudness and gives a modest gain in clarity. It cannot make a 6 m talker sound close.
- Industry products solve this by placing a microphone within about 3 m of every talker [32].

**Approach and priorities**
- **Measure correctly before choosing methods.** The v1.0 bench had five defects:
  - it scored both bus partners against one summed reference with a single global sign;
  - its background metrics (BR, gapSNR) rewarded gating;
  - its scale-invariant metrics and talker medians hid far-talker attenuation;
  - its speech was TTS;
  - it had only 3 geometries per scenario.
  Fix the harness (§4.2–4.5), switch to natural speech and at least 30 stratified geometries per scenario, and record real ESP32 data in Week 1.
- **Self-calibration is mandatory.** It estimates a per-band complex (gain and phase) correction, the inter-channel skew, the spacing (calipers plus endfire chirps on hardware) and the measured noise coherence. Every method learns its steering and noise coherence from the data; the bench's 4 cm is only a prior.
- **Build order:**
  1. Harness, S0 calibration and S1 tracker.
  2. Bus: M3 or M2 beam, M4 coherence mask, M6 noise suppressor. Meeting: M10 WPE, M11 leveling, M6.
  3. M7 spatial clustering and M8 ONNX neural post-filter.
  4. M9 target-speaker extraction for both partners.
- **Where the headroom is.** On the TTS dev set, an oracle mask applied to Mic A raises bus ESTOI from 0.398 to 0.877, and meeting ESTOI from 0.650 to 0.923 (worst talker from 0.476 to 0.887). Re-measure on natural speech. The bottleneck is estimating the mask. In diffuse babble the two-mic spatial cues give weak masks; speaker identity is the strongest extra cue available.

**Hardware advice in one line:** point the mic axis through both partners (bus) or along the long axis of the table (meeting); mount meeting mics within about 2 cm of the tabletop; calibrate once with calipers and chirps; keep about 4 cm spacing; add mics or nodes for meetings.

---

## 1. Physics reality check

Constants: c = 343 m/s, fs = 16 kHz. Every number in this section was computed for this document (scripts in Appendix C). The v1.0 figures agree with the research-phase verification to within ±0.1 dB, except the two-talker beam. That beam, the azimuth averages and the CDR figures were recomputed in v1.1 with `physics_check_v2.py` and the review probes.

### 1.1 Geometry, delays and aliasing

| d | c/2d (aliasing; the two ends of the axis become indistinguishable) | c/4d | max TDOA | Spherical diffuse coherence sinc(kd) > 0.9 / 0.7 / 0.5 below | Horizontal diffuse coherence J0(kd) > 0.9 / 0.7 / 0.5 below |
|---|---|---|---|---|---|
| 2.5 cm | 6.86 kHz | 3.43 kHz | 72.9 µs = 1.17 samples | 1.72 / 3.08 / 4.14 kHz | 1.40 / 2.49 / 3.32 kHz |
| 4 cm | 4.29 kHz | 2.14 kHz | 116.6 µs = 1.87 samples | 1.07 / 1.93 / 2.59 kHz | 0.87 / 1.56 / 2.08 kHz |
| 5 cm | 3.43 kHz | 1.72 kHz | 145.8 µs = 2.33 samples | 0.86 / 1.54 / 2.07 kHz | 0.70 / 1.25 / 1.66 kHz |
| 6 cm | 2.86 kHz | 1.43 kHz | 174.9 µs = 2.80 samples | 0.72 / 1.28 / 1.73 kHz | 0.58 / 1.04 / 1.38 kHz |
| 8 cm | 2.14 kHz | 1.07 kHz | 233.2 µs = 3.73 samples | 0.54 / 0.96 / 1.29 kHz | 0.44 / 0.78 / 1.04 kHz |

**Consequences**
- **Delays are fractional (1.2–2.8 samples).** Steer by per-bin phase rotation in the STFT domain. Never use integer-sample delay-and-subtract: its cardioid has an on-axis null at c/2d, which falls inside the speech band.
- **TDOA and DOA estimation must stay below 0.8·c/2d.** That is 3.4 kHz at 4 cm and 2.3 kHz at 6 cm. Per-bin probabilistic models that use a predicted phase (cACGMM, likelihood scoring) remain valid above c/2d.
- **Below the "coherence > 0.9" frequency** (about 1.1 kHz at 4 cm), diffuse babble and engine noise look spatially like a broadside point source. Coherence and clustering methods cannot discriminate there. That band must be handled by:
  - superdirective beams;
  - spectral (single-channel) suppression;
  - frame-level decisions carried down from higher bands.
- **Cone of confusion.** A two-mic pair measures only the angle to its own axis. All sources on the same cone are spatially identical. Examples: everything in the broadside plane, or a passenger standing on the axis behind a partner. Front and back along the axis are distinguishable (opposite TDOA sign).

### 1.2 Directivity and diffuse-noise gain (linear processing)

- **AI-DI** is the directivity index weighted by the ANSI S3.5 one-third-octave band importance, 160 Hz–8 kHz. It assumes ideal matched mics in free field.
- **3-D** means spherically diffuse noise. **2-D** means horizontally diffuse noise (J0 coherence), which is closer to cabin babble.

| Beam (per-bin MVDR against a diffuse model, with a white-noise-gain floor) | 2.5 cm | 4 cm | 6 cm |
|---|---|---|---|
| Single target on the axis, WNG ≥ −10 dB, 3-D / 2-D | 5.2 / 4.1 | 5.0 / 3.9 | 4.6 / 3.7 |
| Single target on the axis, WNG ≥ −3 dB (quiet room), 3-D / 2-D | 4.1 / 3.4 | 4.3 / 3.5 | 4.3 / 3.4 |
| Single target on the axis, WNG ≥ −20 dB, 3-D | 5.5 | 5.1 | 4.7 |
| Single target, steered, averaged over azimuth, WNG ≥ −10 dB, 3-D / 2-D | 3.2 / 2.5 | 3.3 / 2.7 | 3.3 / 2.7 |
| Single target at broadside, WNG ≥ −10 dB, 3-D / 2-D | 0.5 / 0.8 | 1.0 / 1.5 | 1.5 / 2.0 |
| Two-talker beam (A at 0°, B at 180°), WNG ≥ −10 dB strictly enforced, 3-D / 2-D | 3.4 / 2.2 | 3.6 / 2.3 | 3.5 / 2.3 |
| Two-talker beam, WNG ≥ −20 dB strictly enforced, 3-D / 2-D | 4.4 / 2.8 | 4.1 / 2.6 | 3.7 / 2.4 |
| Delay-and-sum steered along the axis / to broadside, 3-D | 1.3 / 0.5 | 1.9 / 1.0 | 2.3 / 1.5 |

On the "on the axis" rows, 2-D figures use weights designed for J0 coherence. On the azimuth-averaged, broadside and two-talker rows, 2-D figures use weights designed for the 3-D model and evaluated against J0. v1.0 listed the two-talker beam at 3.8 / 3.8 / 3.7 dB (−10 dB floor); the old script had silently kept weights below the floor (§1.6, item 17).

**Further numbers at 4 cm**
- **Off-axis steering** (WNG ≥ −10 / −3 dB): 0°: 5.0 / 4.3; 30°: 4.3 / 3.8; 45°: 3.6 / 3.1; 60°: 2.6 / 2.3; 90°: 1.0 / 1.0 dB.
- **Per-band DI on the axis** (WNG ≥ −10 dB): 2.3 dB at 150 Hz, 3.9 at 250 Hz, 6.0 at 500 Hz, 5.9 at 1 kHz, 5.4 at 2 kHz, 4.5 at 3 kHz, 3.3 at 4 kHz, about 2.9 at 6 kHz.
- **Broadside MVDR:** 0.05 / 0.19 / 0.76 / 1.65 / 2.7 dB at 0.5 / 1 / 2 / 3 / 4 kHz.

**What reduces these numbers in practice**

1. **Self-noise, through the white-noise gain (WNG).**
   - Unconstrained superdirectivity at 4 cm needs WNG of −15.3 / −9.3 / −3.6 dB at 250 / 500 / 1000 Hz. Output self-noise rises by |WNG| in those bins.
   - INMP441 equivalent input noise is about 33 dBA (94 − 61), which is −87 dBFS. Reference: 0 dBFS (AES17) corresponds to 120 dB SPL.
   - **Bus:** noise at 70–80 dBA (−50 to −40 dBFS) sits about 40 dB above self-noise, so WNG floors of −15 to −20 dB are affordable below 500 Hz.
   - **Meeting:** HVAC at 35–45 dBA is only 2–12 dB above self-noise, so keep WNG ≥ −3 to −6 dB.
   - A low WNG floor also amplifies anything spatially uncorrelated: wind, vent turbulence, handling and vibration. This is why hearing aids fall back to omni in wind.
2. **Mismatch between the two mics.**
   - Each INMP441 is within ±1 dB, so a pair can differ by up to 2 dB.
   - Uncalibrated, per-unit errors cost about 0.5 dB of DI at 500 Hz (WNG ≥ −10 dB). At 250 Hz with WNG at −15 to −20 dB they cost about 1.4 dB on average and about 3 dB at the 10th percentile.
   - A fixed −1 dB / −10° error at 250 Hz turns +6 dB into −3.1 dB; with the opposite sign it gives +3.7 dB [4].
   - At the −10 dB floor the speech-weighted loss is small: uncalibrated ±1 dB / ±3° costs ≤ 0.1 dB of azimuth-averaged AI-DI in 2-D noise. The losses above matter at the −15 to −20 dB bus floors below 500 Hz, and phase errors also bias the CDR estimator and the TDOA tracker.
   - Calibrate to ±0.25 dB / ±1° before any superdirective stage (§2.0).
3. **Degrees of freedom.**
   - A two-mic linear filter has one degree of freedom per bin. It can null one point source.
   - Or it can keep both partners with responses (+1, −1) under center-referenced steering (§2.0): the figure-8 (dipole) two-talker beam in the table. Referenced to Mic A, the same beam has r_B/r_A = −e^(+j2πfd/c).
   - The dipole's WNG is 1 − cos(2πfd/c), so it meets a −10 dB floor only above 985 / 616 / 410 Hz (2.5 / 4 / 6 cm). Below that, the balanced two-talker solution must rotate away from the dipole (M2). Never blend dipole and omni by amplitude: the sum is a cardioid, which nulls one partner.
   - It cannot keep both partners undistorted and also null a third source; that needs at least three mics.
   - Constraining both talkers to (+1, +1) collapses the filter to a single omni.
4. **Adaptive cancellation adds little in diffuse or reverberant noise.**
   - For N = 2, the sidelobe-cancelling path of a generalized sidelobe canceller (GSC) gives 0 dB against diffuse noise.
   - GSC gains appear only against one dominant point interferer, and shrink when T60 is well above 100 ms [7].
   - Vendor figures for two-mic adaptive nulling are measured with a single point noise source, which is the best case.
5. **Field losses.**
   - Non-ideal noise fields, scattering from the device body and residual mismatch: expect 0.5–1.5 dB below the table.
   - Two-mic hearing-aid directional benefit is about 3–4 dB in the lab. It is higher with discrete noise sources and vendor-run tests, and lower in daily life [18][19][20].

### 1.3 Distance cues: telling a near partner from a far passenger

- **Level difference between the mics (near-field ILD) for a talker on the axis:** 20·log10((r+d)/r).
  - At 4 cm: 1.1 / 0.67 / 0.34 dB at 0.3 / 0.5 / 1.0 m.
  - At 6 cm: 1.6 / 1.0 / 0.5 dB.
  - This is below the inter-unit tolerance, and below the calibration and reverberation error of a level estimate, so ILD is not a distance cue on this device. Phone methods that rely on a level difference (Jeub's power-level-difference method [24], Audience's ILD [25]) therefore do not carry over to a tabletop device. They need one mic within a few cm of the mouth.
- **Directness** (coherence, DRR) is the only usable physical distance cue, and only inside the critical distance (§1.4).
- **Lombard effect and bus SNR.**
  - Above 45 dBA of noise, talkers raise their voice by about 0.6 dB per dB of noise [97].
  - At 75 dBA bus noise, a partner produces about 73 dBA at 1 m.
  - Direct-path SNR at the device: +8.5 / +2.4 / −2.0 dB at 0.3 / 0.6 / 1.0 m. At 80 dBA noise: +6.5 / +0.4 / −4.0 dB.
  - Measured in-bus Leq is 75.3 ± 2.6 dBA; the highest in-bus Lmax was 96.8 dBA [96].
  - A partner at 1 m is therefore at or below 0 dB SNR before any processing.

### 1.4 Rooms: critical distance, DRR and level versus distance

- Critical distance: r_c ≈ 0.057·√(γV/T60) m [15]. γ = 1 for an omni. A perfect hypercardioid (γ = 4) doubles r_c, but only on the axis. A steered two-mic MVDR averaged over orientation in 2-D noise (2.5–2.7 dB) stretches r_c by only ×1.34–1.37.
- DRR(r) ≈ 20·log10(r_c/r).

**Bus cabin** (11.5 × 2.5 × 2.3 m ≈ 66 m³). No published bus T60 was found, so the benchmark must sweep it.

| T60 | r_c | DRR at 0.3 / 0.6 / 1.0 / 2.0 m |
|---|---|---|
| 0.15 s | 1.20 m | +12.0 / +6.0 / +1.6 / −4.5 dB |
| 0.30 s | 0.85 m | +9.0 / +3.0 / −1.5 / −7.5 dB |
| 0.45 s | 0.69 m | +7.2 / +1.2 / −3.2 / −9.2 dB |
| 0.60 s | 0.60 m | +6.0 / 0.0 / −4.5 / −10.5 dB |

**Meeting room**

| Room | T60 | r_c | DRR at 1 / 2 / 3 / 4.6 / 6 m | Total level drop, 1 → 6 m (Sabine / Barron [16]) |
|---|---|---|---|---|
| 10×7×3 m (210 m³) | 0.5 s | 1.17 m | +1.3 / −4.7 / −8.2 / −11.9 / −14.2 | 3.5 / 5.4 dB |
| | 0.7 s | 0.99 m | −0.1 / −6.1 / −9.7 / −13.4 / −15.7 | 2.8 / 4.1 dB |
| | 0.9 s | 0.87 m | −1.2 / −7.2 / −10.7 / −14.5 / −16.8 | 2.3 / 3.4 dB |
| 12×8×3 m (288 m³) | 0.5 s | 1.37 m | +2.7 / −3.3 / −6.8 / −10.5 / −12.8 | 4.3 / 6.2 dB |
| | 0.7 s | 1.16 m | +1.3 / −4.8 / −8.3 / −12.0 / −14.3 | 3.5 / 4.8 dB |
| | 0.9 s | 1.02 m | +0.2 / −5.9 / −9.4 / −13.1 / −15.4 | 2.9 / 4.0 dB |

Talker directivity and the spread in vocal effort add to the level differences. A 2–4 dB directivity gain stretches r_c by a factor of only 1.3–1.6.

### 1.5 What this means for the two test cases

**The two-mic ceiling.** All figures are ideal, before 0.5–1.5 dB of field losses (scripts in Appendix C).

| Cue or stage | Gain | Condition |
|---|---|---|
| Two-talker beam against horizontal (2-D) babble | 2.2–2.3 dB AI-DI | Partners near the mic axis. About 0 dB when both sit near broadside, where the beam is ill-posed |
| Steered single-target MVDR, averaged over azimuth, 2-D | 2.5–2.7 dB, so r_c × 1.34–1.37 | 0.8–2.0 dB at broadside. Uncalibrated ±1 dB / ±3° mismatch costs ≤ 0.1 dB at the −10 dB floor |
| CDR post-filter, near partner (DRR +6 dB, 0 dB SNR) | +1.2–2.0 dB broadband ΔSNR at λ = 0.68; up to +3.6 dB at λ = 0.9 | About 0 dB at DRR ≈ 0 dB. For a meeting talker at DRR −8 dB: −0.8 dB in 1–4 kHz, with the talker cut by 5.6–9 dB |
| Near-field level difference (ILD) | ≤ 1.6 dB at 0.3 m (6 cm, endfire) | Below calibration and reverberation error, so not usable as a distance cue |
| TDOA as talker identity, 20 meeting seats | Median of 8 resolvable groups (20 µs resolution, 4 cm) | Cannot key per-talker memory |

Most of the achievable gain must therefore come from four sources:
- single-channel noise reduction (M6, M8);
- WPE dereverberation (M10);
- leveling (M11);
- speaker-identity cues (M9, and speaker embeddings in S1 and M11).

Spatial processing adds a few dB, and only with a correct orientation.

**Bus: keep the two near talkers, suppress the crowd**

*Can do:*
- 2.2–2.3 dB of speech-weighted babble reduction in 2-D (3.6 dB in 3-D noise) with the mic axis through both partners; about 0 dB if both sit near broadside.
- Null one dominant point interferer when the cabin is not too reverberant, but not while also keeping both partners undistorted.
- Gate diffuse babble, road noise and reverberation with coherence (CDR) masks, for partners well inside r_c (0.5–0.6 m or closer), and only in bins where the CDR estimator is well conditioned (M4).
- Suppress stationary engine and road noise spectrally.

*Cannot do with classical processing:*
- Separate a partner from a passenger on the same cone.
- Reject a passenger 1–2 m away with coherence; that passenger is also coherent.
- Tell a partner from a passenger by level or distance.
- Use spatial cues on engine or babble energy below about 1 kHz (coherence > 0.9).
- Keep a partner who is at or beyond r_c out of a coherence gate's floor.

Reaching "only the partners audible" therefore needs speaker-identity cues (M9), aggressive nonlinear masks with audible artifacts, or hardware changes (§5).

**Meeting: capture all 20 talkers evenly**

*Can do:*
- Even out loudness. AGC is just a gain, and the total level spread between talkers is small.
- Remove stationary HVAC and projector noise.
- Linearly remove part of the late reverberation (WPE).
- Add 2–4 dB of DRR for talkers within ±45° of the mic axis.

*Cannot do:*
- Make a 6 m talker (DRR −13 to −17 dB) sound like a 1 m talker with one two-mic node.
- Null individual interferers without risking cancellation of the target.
- Separate 20 talkers spatially or identify them by TDOA:
  - at 4 cm the seats collapse to a median of 8 resolvable TDOA groups at 20 µs resolution (12 at 10 µs; 6 at 2.5 cm and 20 µs);
  - 8–13 of 19 neighbouring seats differ by less than 10 µs;
  - mirror-image seats share a TDOA but not a room response.

### 1.6 Corrections to research-phase claims (carry these; do not re-litigate)

1. **Mismatch losses** were understated by 1.6–2× (the error was modelled on one mic only). Use the per-unit figures in §1.2.
2. **Bitzer's "≤ 1 dB GSC" limit** is for the adaptive path on top of a broadside delay-and-sum beam with N = 3–7; for N = 2 the paper gives 0 dB [7]. Bitzer's IWAENC'99 warning about signal cancellation from "adaptive schemes" refers to the adaptive post-filters, not to beamformers [8]. The practical lesson is to keep post-filter floors high.
3. **Covariance-whitening RTF.** The estimate is Φ_w·v_max normalized by its first entry, not v_max itself [10].
4. **Tashev's gain calibration** is degenerate for two mics. Use an equal-level assumption on diffuse frames, with an amplitude (square-root) correction [12].
5. **Coherence post-filters do not suppress nearby passengers or low-frequency engine noise.**
   - In the verification simulation, CDR attenuated noise by 4–8 dB but also attenuated the target by 3–7 dB.
   - Net gain: +1.2 to +2.0 dB with smoothing λ = 0.68 and +2 to +3.6 dB with λ = 0.9.
   - Net gain is about 0 dB for a partner at DRR ≈ 0 dB, which is about 1 m in the bus.
   - For a meeting talker at DRR −8 dB it is negative in 1–4 kHz (−0.8 dB), while the talker loses 5.6–9 dB.
6. **Spacing changes the beams by ≤ 0.5 dB across 2.5–6 cm.** At 6 cm:
   - it wins with a WNG floor of −3 dB, for CDR conditioning, for low-frequency WNG (+3.5 dB) and for DOA resolution;
   - but aliasing starts at 2.86 kHz.
7. **Industry figures are not transferable to this device.** Fabry's 7.7 dB came from discrete noise sources and a vendor-run test with 7 subjects. Signia's +3.9 dB SRT80 is a vendor study. Phonak's "up to 10 dB" is a manufacturer claim. VoxLock's leakage and WER figures are self-reported. None of these transfer to a tabletop two-MEMS device.
8. **NSNet** degradation figures on reverberant speech (MOS 2.44 → 2.16) belong to NSNet v1, not NSNet2.
9. **DeepFilterNet and DPDFNet results** come from synthetic, additive, non-reverberant mixtures. Uninformed (not speaker-conditioned) enhancement failed to improve intelligibility in multi-talker restaurant noise [82].
10. **H-GTCRN's two-mic gain** comes mostly from AuxIVA separating one point interferer. IVA alone scores STOI 70.2 against 61.1 for single-channel GTCRN at −7.5 dB [75]. Diffuse babble removes most of that advantage.
11. **Online WPE with α = 0.9999** has a memory of about 80 s at an 8 ms hop, far too slow for turn-taking.
12. **TDOA groups for 20 talkers:** at 4 cm, a median of 8 resolvable groups at 20 µs resolution (6–10 over orientations) and 12 at 10 µs; 6 at 2.5 cm. Not 10–15. TDOA cannot key per-talker memory; speaker embeddings must (S1).
13. **DNSMOS** correlates with human ratings at only 0.71 (SIG), 0.83 (BAK) and 0.82 (OVRL) per clip. It is out of domain for far-field reverberant audio, and it is invalid for judging whether the right talker was kept.
14. **pyroomacoustics** cannot combine source directivity with ray tracing, and its cardioid source patterns do not vary with frequency.
15. **anf-generator:** the processing option is `'balance+smooth'`. The package imports matplotlib at module level.
16. **Not used:** the InvenSense AN-1140 and ST UM2214 vendor numbers could not be re-fetched during verification.

Added in the v1.1 review:

17. **Two-talker beam figures were too high.** v1.0 gave 3.8 / 3.8 / 3.7 dB at the −10 dB floor because `physics_check.two_talker()` kept floor-violating weights whenever its bisection failed. With the floor strictly enforced: 3.4 / 3.6 / 3.5 dB (3-D) and 2.2 / 2.3 / 2.3 dB (2-D).
18. **"Optimal phase π at every frequency" needs two qualifiers.**
    - It holds only for center-referenced steering, and only where the dipole meets the WNG floor.
    - `physics_check.py` used Mic-A-referenced steering, where the same optimum appears as 180° + 360°·f·d/c (190 / 200 / 222 / 265 / 312 / 348° at 0.25 / 0.5 / 1 / 2 / 3.15 / 4 kHz, 4 cm).
    - A literal r_B = −1 in Mic-A referencing is a different and poor beam: 0.9 / −0.4 / −1.6 dB AI-DI.
19. **Orientation.** Averaged over azimuth in 2-D noise, the single-target gain is 2.5–2.7 dB (r_c × 1.34–1.37). The on-axis hypercardioid figure is not representative.
20. **CDR estimator floor.** On pure diffuse noise at λ = 0.68, the short-time CDR has a median of −0.4 to −0.8 dB with 45–47% of bins above 0 dB; only the power-averaged gain (−8.6 to −9.7 dB) is reliable. The µ / G_min / λ in [34] were tuned at d = 8 cm. At 1 kHz a broadside partner has |Γs − Γn| = 0.03 (2.5 cm) or 0.09 (4 cm), so the estimator is ill-conditioned there.
21. **Scoring artefacts in the v1.0 bench.**
    - An ideal noise-free output with partner B inverted scored dSISDR −2.6 dB (+235 dB with the same polarity), ESTOI 0.80 and target gain −9.4 dB.
    - An oracle-VAD gate on raw Mic A scored BR +40 dB with ESTOI unchanged.
    - Neither metric can be used for decisions (§4.3).

---

## 2. Pipeline recommendations

### 2.0 Shared conventions and infrastructure (build these first)

**Signal conventions**
- `x[0]` is Mic A (left I2S slot) and `x[1]` is Mic B. In the lab, Mic A sits at center + (d/2)·u, where u points along the array azimuth.
- θ is the angle between the source direction and u. τ = d·cosθ/c; τ > 0 means Mic B hears the source later. A plane wave gives X_B = X_A·e^(−j2πfτ).
- Center-referenced steering vector: `a(f,τ) = [e^(+jπfτ), e^(−jπfτ)]ᵀ`.
  - Every response constraint in this document (for example h = [1, −1] for the two-talker beam) is written for this convention.
  - RTFs normalized to Mic A, ĝ = [1, G], must be symmetrized before such constraints are applied: g̃ = ĝ/√(ĝ₁·ĝ₂), with the phase of ĝ₂ unwrapped before halving. For a plane wave g̃ equals a(f,τ).
- Cross-spectrum Φ_AB = E{X_A·X_B*}. For a plane wave the coherence is Γ_s = e^(+j2πfτ).
- Beam output y = wᴴX.
  - White-noise gain: WNG = |wᴴa|²/(wᴴw).
  - Directivity factor: DF = |wᴴa|²/(wᴴΓw).
- **Sign trap:** `lab.dsp.gcc_phat(x0, x1)` returns a positive value when x0 lags. Our τ is therefore `gcc_phat(x_B, x_A)`.
- **STFT:** `lab.dsp.stft` / `istft` (sqrt-Hann, 512-point window, hop 128, 257 bins, 31.25 Hz bin spacing).
- **Smoothing constants:** recursive smoothing λ = exp(−8 ms/τ_avg):

  | τ_avg | 20 ms | 50 ms | 76 ms | 160 ms | 2 s |
  |---|---|---|---|---|---|
  | λ | 0.67 | 0.85 | 0.90 | 0.95 | 0.996 |
- **Magnitude-squared coherence:** MSC(l,k) = |Φ_AB|²/(Φ_AA·Φ_BB).

**S0: Self-calibration** (both scenarios, mandatory)

- **Hardware:** the one-time session in §5 is the primary source and is saved to `calib.json`. At run time S0 only re-estimates the gain and tracks slow drift.
- **Bench:** S0 runs blindly per file (methods may use only x, fs, info). Its errors are measured against the E8 impairments (§4.1).

1. **Pre-filter.** Apply an identical 2nd-order Butterworth high-pass to both channels: 120 Hz (bus) or 80 Hz (meeting).
2. **Diffuse-dominant frames D.**
   - Compute MSC_hf(l) = mean of MSC over 2.5–6 kHz, with λ = 0.9.
   - D = frames with MSC_hf below its 30th percentile and broadband level above its 20th percentile (this excludes near-silence dominated by self-noise).
   - MSC is immune to aliasing: a single plane wave always has coherence 1.
3. **Per-band complex correction c_b = ĝ_b·e^(+jφ̂_b), applied to X_B.** Use one-third-octave bands b from 125 Hz to 6.3 kHz. A scalar gain and delay is not enough: unit-to-unit spread of the INMP441's low-frequency roll-off (corner near 60 Hz) gives a frequency-dependent phase mismatch.
   - **Gain:** ĝ_b = √(median over D of Φ_AA,b / median over D of Φ_BB,b). Smooth over 3 bands, clip to ±3 dB, interpolate to bins.
   - **Phase:** a diffuse field has a real cross-spectrum, so the residual phase of Σ_D Φ_AB,b is the mismatch.
     - Fit φ(f) = 2πf·τ0 + atan(f_A/f) − atan(f_B/f) jointly over 100–2000 Hz: a skew τ0 (step 4) plus two first-order high-pass corners f_A, f_B in 30–150 Hz (a first-order high-pass has phase atan(f_c/f)).
     - Weight each band by |Σ_D Φ_AB,b|.
     - φ̂_b is the corner term of the fit. Flag a hardware fault if |φ̂_b| > 10° in any band.
   - **Bus:** the engine at the rear is a point source and biases the low-band phase. Take φ̂_b from `calib.json` on hardware. In the bench, estimate it but also report the error against the E8 truth.
   - On the bus, D excludes near partners automatically (they have high MSC). This matters because a partner at 0.3 m along the axis produces a real 1.1 dB level difference.
4. **Fixed inter-channel delay τ0** (I2S skew; a half-sample skew is 31.25 µs, 27% of the 117 µs endfire TDOA at 4 cm).
   - Hardware (primary): endfire chirps (§5). A chirp on the axis beyond Mic A gives τ₊ = +d/c + δ; one beyond Mic B gives τ₋ = −d/c + δ. Then τ0 = δ = (τ₊ + τ₋)/2.
   - Blind: the slope term of the step-3 fit. In the bench this is a diagnostic only, because scenes have no skew unless E8 adds it.
5. **Spacing d̂.**
   - Hardware (primary): d = c·(τ₊ − τ₋)/2 from the same chirps, cross-checked against calipers (port centre to port centre). If they differ by more than 2 mm, use the acoustic value: it is the spacing the beams see.
   - Blind fallback (bench, or no session):
     - Fit Re Γ̂_D(f) ≈ ρ·model(f; d) over 200–4000 Hz, for d in 1.5–8 cm and a free scale ρ in [0.5, 1].
     - ρ absorbs self-noise and residual uncorrelated noise. In a 40–45 dBA room, sensor self-noise alone caps |γ| at about 0.83–0.94; a fit with ρ fixed at 1 then overestimates d.
     - Fit both models, sinc(2fd/c) and J0(2πfd/c), and keep the one with the lower residual. Choosing the wrong model alone moves d̂ by about 25%.
     - Lower bound: d_lb = c · (99th percentile of |τ̂| over high-directness frames).
     - d̂ = max(d̂_fit, d_lb). If the fit residual is large, fall back to the nominal 4 cm.
   - Use `info['spacing_m']` only as a prior.
6. **Measured noise coherence Γ̂n(f).**
   - Re{Σ Φ_AB}/√(Σ Φ_AA · Σ Φ_BB), taken over D (bus) or over VAD pauses (meeting).
   - Smooth over 5 bins and clip to [−0.3, 0.98].
   - Blend with ρ̂·model at d̂: weight N/(N + 200) on the measurement, where N is the number of frames used.

Output: `calib = {ĝ_b, φ̂_b, τ0, d̂, model, ρ̂, Γ̂n(f), confidence}`. Apply c_b and the τ0 correction to X_B before any other stage.

**S1: Talker and DOA tracker** (both scenarios) [13][14]

- Smooth Φ_AB with λ = 0.85; take Γ̂x = Φ_AB/√(Φ_AA·Φ_BB).
- Steered response, PHAT-weighted:
  `P(l,τ) = Σ_k W(l,k)·Re{Γ̂x(l,k)·e^(−j2πf_kτ)} / Σ_k W(l,k)`.
  - Bins: 200 Hz ≤ f ≤ min(3.5 kHz, 0.8·c/2d̂).
  - Weights: W = clip(MSC/(1 − MSC + 0.05), 0, 20).
  - Grid: τ in ±1.2·d̂/c, step 2 µs.
- τ̂(l) = argmax P. Directness ψ(l) = MSC_hf(l).
- **Bus partner model.**
  - Build a histogram of τ̂ over frames with ψ above its 70th percentile and energy above the median, energy-weighted, smoothed with σ = 6 µs.
  - τ_A and τ_B are the two highest peaks at least 25 µs apart.
  - Offline, use the whole file; online, use the first 5 s and then forgetting with a 5 s time constant.
  - Partner posteriors p_A, p_B: an indicator (|τ̂ − τ_k| < 20 µs and ψ high), smoothed with τ = 160 ms (λ = 0.95).
  - Double talk: both posteriors > 0.25.
  - Log each partner's angle from the axis, θ_k = arccos(c·τ_k/d̂), and the orientation stratum (aligned / 45° / broadside, §4.1).
  - Flag "partners spatially inseparable" if any of these holds:
    - only one peak exists;
    - |τ_A − τ_B| < 25 µs;
    - both |τ_A| and |τ_B| are below 0.3·d̂/c (22 / 35 / 52 µs at 2.5 / 4 / 6 cm). This covers both partners near broadside, or a device lying flat with the partners above it.
- **Meeting talker identity.**
  - **Primary key: speaker embeddings.**
    - For every 1.5 s of confident speech (p_s ≥ 0.95), compute an embedding with a small ONNX speaker-embedding model (≤ 30 MB, run with onnxruntime). An example is a WeSpeaker or 3D-Speaker export as packaged for sherpa-onnx; check its licence first.
    - Cluster online by cosine similarity to cluster centroids. A segment joins the best cluster if similarity ≥ θ_s (tune on dev; start at 0.6); otherwise it opens a new cluster (at most 30).
  - **Secondary cues:** the |τ̂| bin (10 µs) and the mean CDR. They break ties and give a new cluster its starting gain (M11). They cannot identify talkers: 20 seats give a median of 8 resolvable TDOA groups (§1.5).
  - **Fallback without the model:** online DP-means on (τ̂, mean CDR in dB), thresholds 15 µs and 4 dB, at most 12 clusters. Expect merged talkers.
  - **Talker change:** a change in TDOA or CDR proposes a change after 250 ms; the embedding confirms or rejects it within 1.5 s.
  - Clusters key the gain memory in M11 and the WPE state in M10.

### 2.1 Bus pipeline (goal: only near talkers)

```
x_A, x_B ─► S0  HPF 120 Hz ─► per-band complex EQ c_b ─► τ0 comp ─► d̂, Γ̂n     (hardware: calib.json)
            │ STFT 512/128 (32 ms algorithmic latency)
            ├─► S1 tracker: τ_A, τ_B, p_A, p_B, ψ, double talk, orientation stratum, "inseparable" flag
            ▼
   SPATIAL (linear, per bin, center-referenced steering)    states A / AB / B, 50 ms weight crossfade
     M3  RTF-MVDR {w_A, w_AB, h_B*·w_B}   w_AB: h = [1, −1] where the dipole meets the WNG floor,
                                          continuation phase search elsewhere (M2); per-bin phase-continuity check
     WNG ≥ −15 dB <500 Hz, −10 dB to 2 kHz, −6 dB above
     (A/B: M2 model-based beams; M7 cACGMM+MVDR offline; inseparable → fallback below)
            │ Y_lin(l,k)
            ▼
   MASKS (each estimated from LINEAR signals; combined ONCE)
     G_cdr  M4  DOA-dependent CDR (2 hypotheses), only where |Γs,k − Γn| ≥ 0.2; µ, G_min, λ retuned for 2–6 cm
     G_fen  M5  phase fence 1–4 kHz, γ 4, floor 0.3
     G_ns   M6  MMSE-SPP + LSA, floor −15 dB, noise update frozen on target cue
    [G_nn   M8  GTCRN or DPDFNet-4 mask on y_lin, attenuation limit 20 dB]
    [G_tse  M9  VoxLock(enrollment A) ∪ VoxLock(enrollment B), adaptive strength]
     m = max(m_floor, G_cdr·G_fen·G_ns·G_nn·G_tse); rise instantly, fall ≤ 4.8 dB per 8 ms hop
            ▼
   SPLIT REMIX (hearing-aid style [21]): target T = m·Y_lin (linear, AGC);
     residual R = (1−m)·Y_lin → extra NS + slow compression, mixed at ρ = −15 dB (user slider −6…−25 dB)
            │ ISTFT
   M11-bus: AGC level updates only when p_A + p_B > 0.8 (never on babble); −1 dBTP limiter ─► y
```

**Tiers**
- **Tier 1** (classical, real-time, no models): S0, S1, M3 (or M2), M4, M5, M6, split remix, M11-bus.
- **Tier 2:** add the M8 mask (or M7 offline).
- **Tier 3:** add M9 (needs enrollment).

**Fallback when the partners are spatially inseparable** (S1 flag: both partners near broadside, the device lying flat, or |τ_A − τ_B| < 25 µs):
- use a single broadside MVDR (0.8–2.0 dB in 2-D noise) or omni;
- rely on M4 (DOA-free variant, conditioned bins only), M6, M8 and M9;
- report this condition in the method's logs; on hardware, tell the user to rotate the device so its axis points through both partners (§5).

**Optional:** WPE (M10) is low priority on the bus. The problem there is additive noise, and WPE ignores additive noise.

### 2.2 Meeting pipeline (goal: everyone, even loudness)

```
x_A, x_B ─► S0  HPF 80 Hz ─► per-band complex EQ ─► τ0 comp ─► d̂, Γ̂n (from VAD pauses)
            │ STFT 512/128
   M10 2-ch WPE: taps 10, delay 3; segment-wise offline  |  online RLS α 0.998 + per-speaker state
            │ (two dereverberated channels)
   S1  tracker on WPE output: τ̂, speaker cluster (embeddings; |τ̂|, CDR secondary), VAD, change points
            ▼
   M2-meeting: superdirective MVDR steered to τ̂ of the active talker, WNG ≥ −3 dB, 300–500 ms hold, 50 ms crossfade
     [alt: 3 fixed beams (+axis, −axis, broadside) + best-beam selection]
            ▼
   M6  MMSE-SPP + LSA for HVAC/projector, floor −12 dB, noise update in VAD pauses only
     [alt M8: GTCRN or DPDFNet, attenuation limit 12 dB]
            │ ISTFT
   M11 leveling: speech-gated, noise-capped AGC with per-speaker gain memory (embedding-keyed);
                 |τ̂|-indexed starting gain when the mic axis lies along the table
                 → off-attenuation −3 dB in pauses (NM ≤ 6 dB) → −1 dBTP limiter
     [offline: diarization → per-speaker loudness normalization → programme −23 LUFS]  ─► y

Multi-node: N × (S0 … M6) per node → coarse alignment (GCC-PHAT on speech) + sampling-rate-offset
            compensation → Dugan gain-sharing [28] or best-node selection → M11
```

Do not use the following in the meeting pipeline:
- adaptive nulls;
- generalized sidelobe cancellers;
- target-speaker extraction (it removes 19 of the 20 talkers);
- the CDR post-filter, even a mild one. In the verification simulation it cut a DRR −8 dB talker by 5.6 dB broadband (9 dB at λ = 0.9). In 1–4 kHz the talker lost 6.2 dB and the noise 5.5 dB. Dereverberate with WPE instead; M4 runs in the meeting only as an ablation.

### 2.3 Integration with the bench (`speech_lab/`)

**Layout**
- Shared code in `methods/_core/`: `calib.py`, `tracker.py`, `beams.py`, `masks.py`, `ns.py`, `agc.py`, `wpe.py`, `nn.py`, `spk.py` (speaker embeddings).
- `methods/<id>_<name>.py` are thin wrappers exposing `enhance(x, fs, info)`.
- Name stacks explicitly. Example: `bus_t1_m3m4m5m6` = S0 + S1 + M3 + M4 + M5 + M6 + remix + M11-bus.
- Methods with an AGC stage also accept `return_stages=True` and then return `{'pre_agc': y0, 'out': y}`. The meeting metric T2 needs the level before AGC.

**Prerequisite harness changes** (§4.1–4.2; build them before benchmarking any method)
- `finalize()` stores 2-channel components `comp2_*`, including the sensor noise and the per-mic gains, so that Σ comp2 = mix.
- Separate early images per partner, per near passenger and per meeting talker.
- Per-talker scoring that tolerates polarity and phase: separated components, a 512-tap allowed-distortion SDR and per-talker ESTOI. Remove the global sign flip in `align()`.
- Hagerman multi-signal runs for black-box methods.
- Natural-speech scenes, at least 30 stratified geometries per scenario, and seeded pyroomacoustics.

**Dependencies**
- Use only numpy, scipy, onnxruntime, soundfile, pyroomacoustics and `lab.dsp`.
- Do not pip-install into the shared `.venv`. It currently has Python 3.12.13, numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.1, pyroomacoustics 0.10.1, pystoi and pesq.
- Tools needed only for scoring or reference outputs (sherpa-onnx, an ONNX ASR model for WER, pyloudnorm) go in a separate `scoring_venv`. Never import them from `methods/`.
- Vendor permissively licensed code into `methods/_vendor/`, keeping license headers:
  - from NARA-WPE `wpe.py`: `wpe_v8` and `OnlineWPE`, with the `import click` and the CLI removed (MIT);
  - the core of the pb_bss cACGMM (MIT);
  - GTCRN, DPDFNet and VoxLock inference loops (Apache-2.0 for DPDFNet and VoxLock; check GTCRN's licence).
- Models go in `speech_lab/models/`, about 150 MB in total including the speaker-embedding and ASR models. Free disk is currently about 8 GB; budget for less than 4.8 GB, natural-speech corpora included.

**Rules**
- Never trust `info['spacing_m']` beyond using it as a prior.
- Real-time factor (RTF) must stay below 0.5. Tier 3 (M9) may exceed it only if it is flagged "offline-only" and its RTF is reported.
- Output must be time-aligned with the input; compensate model latency yourself.
- Two-pass offline processing is allowed (statistics pass, then processing pass). Each method must also have an online variant.

---

## 3. Candidate methods to implement and benchmark

| ID | Method | Target | Needs geometry? | Expected gain vs Mic A (pre-benchmark estimate) | Route | Priority |
|---|---|---|---|---|---|---|
| M0 | Mic A + HPF | both | no | reference | exists (`mic_a`) | baseline |
| M1 | Delay-and-sum (broadside average / steered) | both | learned τ | AI-DI ≤ 1.9 dB on the axis (3-D); lab `avg` ≈ 0 dB | numpy | baseline |
| M2 | WNG-constrained superdirective and two-talker beams | bus; meeting (steered) | learned τ_k + Γ̂n or d̂ | aligned: 3.9 dB single / 2.3 dB two-talker (2-D, 4 cm); 2.5–2.7 dB single averaged over azimuth; ≈ 0 dB two-talker at broadside | numpy | P1 |
| M3 | Geometry-free RTF-MVDR (covariance whitening) | bus (meeting optional) | none | same ceiling as M2, more robust | numpy | P1 |
| M4 | CDR coherence post-filter | bus | Γ̂n (+ τ_k) | near partner: +1.2–2.0 dB ΔSNR (λ 0.68), up to +3.6 dB (λ 0.9); ≈ 0 at DRR ≈ 0 dB | numpy | P1 |
| M5 | Two-hypothesis phase fence | bus | τ_k, d̂ | a few dB against off-cone coherent sources | numpy | P2 |
| M6 | Spatially gated MMSE-SPP + LSA | both | no | stationary noise −6 to −12 dB; ESTOI ≈ 0 | numpy | P1 |
| M7 | Anchored cACGMM + per-talker mask-MVDR | bus | TDOA anchors only | a few dB SIR, mostly above 1 kHz | numpy (vendored) | P2 |
| M8 | Neural post-filter or mask (GTCRN, DPDFNet; NSNet2 only from a pinned source) | both (meeting limited) | no | large BAK; ESTOI ≈ 0 to +0.05 on babble | onnxruntime + port | P2 |
| M9 | Two-partner target-speaker extraction (VoxLock ×2, explicit enrollment) | bus | no (enrollment) | removes intelligible bystanders (unverified here) | onnxruntime + port, 74 MB | P3 |
| M10 | Two-channel WPE | meeting | no | partial late-reverb removal | vendored NARA-WPE | P1 (meeting) |
| M11 | Speaker-keyed AGC + automix + off-attenuation | meeting; bus gated | no | post-AGC level std ≤ 1.5 dB (T3 target) | numpy | P1 (meeting) |

Calibrating expectations: in the lab bus scenes (TTS), input ESTOI rises about 0.034 per dB of in-band SNR (mean 0.23 / 0.39 / 0.57 at −5 / 0 / +5 dB). A 3 dB effective SNR gain is therefore about +0.10 ESTOI. Re-measure this slope on natural speech (E0).

---

### M0: Single-mic reference (baseline)

- **Idea:** Mic A after S0's high-pass filter. Every other method is scored relative to this.
- **Algorithm:** y = HPF(x_A). The existing `methods/mic_a.py` omits the high-pass; add a variant `m0_mica_hpf` that includes it.
- **Target:** both. **Geometry:** none.
- **Measured on lab dev** (v1.0 TTS scenes and v1.0 metrics; re-baseline on E0 scenes with the §4 metrics before comparing methods):
  - bus: ESTOI 0.398, worst-target ESTOI 0.388, PESQ 1.04, background reduction (BR) 0;
  - meeting: ESTOI 0.650, worst talker 0.476, loudσ 2.6 dB, gapSNR 14.8 dB.
- **Implementation:** exists.

### M1: Delay-and-sum (baseline)

- **Idea:**
  - M1a, broadside average: y = (x_A + x_B)/2 (the existing `avg`).
  - M1b, steered: Y = a(f,τ̂)ᴴX/2, with τ̂ from S1 (bus: the active partner; meeting: the active talker).
- **Sources:** directivity definitions [1]; delay-and-sum with post-filtering [8].
- **Algorithm (M1b):** compute center-referenced a(f,τ) per bin; hold time 300 ms; 50 ms crossfade when τ̂ changes.
- **Target:** both. **Geometry:** τ̂ (learned).
- **Expected gain:**
  - AI-DI against 3-D diffuse noise: steered along the axis 1.3 / 1.9 / 2.3 dB, broadside 0.5 / 1.0 / 1.5 dB (2.5 / 4 / 6 cm).
  - +3 dB against uncorrelated self-noise.
  - Measured `avg`: −0.08 dB dSISDR (bus), −0.24 dB (meeting).
- **Failure modes:** essentially no gain at low frequency; grating lobes above c/2d (harmless).
- **Implementation:** numpy, about 15 lines.

### M2: WNG-constrained superdirective and two-talker beams (fixed weight sets)

- **Idea:** per-bin MVDR against a diffuse-noise coherence model, with a white-noise-gain floor (robust superdirectivity). This is the per-bin generalization of first-order differential arrays:
  - a hypercardioid aimed at the talking partner;
  - a figure-8 (dipole) beam that keeps both partners during double talk.
  This is the hearing-aid directional approach.
- **Sources:** [1][2] WNG-constrained superdirectivity; [3] differential-array design; [4] mismatch robustness; [18][19] hearing-aid evidence; [5] adaptive variant.
- **Algorithm**
  - **Noise model:** Γ(f) = Γ̂n(f) from S0, or else ρ̂·sinc/J0 at d̂.
  - **Steering:** center-referenced a_k(f) = a(f, τ_k) (§2.0), with τ_k learned by S1. Never use the nominal spacing. Every constraint vector h below assumes center referencing.
  - **Single-target beam:** w_k = (Γ+μI)⁻¹a_k / (a_kᴴ(Γ+μI)⁻¹a_k).
    - μ(f) is the smallest value satisfying 1/(wᴴw) ≥ WNG_min(f): bisect log10 μ over [−9, 6], 40 steps, vectorized over bins. The bisection always succeeds: μ → ∞ gives delay-and-sum (WNG +3 dB).
    - Use the closed-form 2×2 inverse.
  - **Two-talker beam (soft LCMV):** w_AB = (Γ + μI + λ_c·A·Aᴴ)⁻¹·λ_c·A·h*.
    - A = [a_A, a_B], h = [1, e^(jψ_AB(f))], λ_c = 10³, μ found by WNG bisection.
    - The WNG floor is a hard constraint. If no μ meets it, that (bin, ψ_AB) candidate is rejected; weights that violate the floor are never kept.
    - **Phase rule, partners on opposite sides of the device:**
      - **Dipole band:** ψ_AB = 180°, i.e. h = [1, −1], the figure-8. The band is the contiguous range of bins ending at c/2d̂ in which the dipole meets the WNG floor; its lower edge is the first bin, going down, where the dipole fails. The dipole has WNG = 1 − cos(2πf·d̂/c). At a constant −10 dB floor the band starts at 985 / 616 / 410 Hz for d = 2.5 / 4 / 6 cm.
      - **Continuation search outside that band:** move bin by bin away from the band, downward from its lower edge (even where the dipole would meet a lower floor again, as with the −15 dB bus floor below 500 Hz) and upward above c/2d̂.
        - In each bin, search ψ_AB on a 2.5° grid within ±30° of the neighbouring bin's ψ_AB.
        - Keep the candidate with the largest DF that meets the WNG floor and keeps |r_A|/|r_B| within ±1 dB.
        - If no candidate in the window qualifies, search the full circle. If still none, use Mic A alone in that bin.
      - Verified on the STFT grid (Appendix C): this rule reaches the per-bin optimum (3.4 / 3.6 / 3.5 dB AI-DI in 3-D noise) with bin-to-bin phase steps ≤ 30°. An unconstrained per-bin search jumps by up to 180° between bins, which lengthens the filter's impulse response.
      - Below the dipole band, ψ_AB rotates smoothly away from 180° toward the single-mic solution. Never amplitude-blend the dipole with an omni: omni plus figure-8 is a cardioid, which nulls partner B in the crossover band.
    - **Convention check.** With responses referenced to Mic A (as in `arch/physics_check.py`, or RTFs normalized to Mic A), the same figure-8 has r_B/r_A = −e^(+j2πf·d̂/c).
      - Writing h = [1, −1] with Mic-A-referenced vectors produces a different beam. Its WNG falls to zero at c/2d̂, and its AI-DI is 0.9 / −0.4 / −1.6 dB (3-D, 2.5 / 4 / 6 cm), worse than one omni at 4–6 cm.
      - Unit test B.1 guards against this.
    - **Partners not on opposite sides:** there is no 180° shortcut. Do a full-circle search at 1 kHz, then continue up and down with the ±30° window.
    - If S1 flags "inseparable" (|τ_A − τ_B| < 25 µs, or both |τ_k| < 0.3·d̂/c), the two-talker beam is ill-posed. Use the §2.1 fallback.
  - **WNG floors:**

    | Scenario | Floor |
    |---|---|
    | Bus | −15 dB below 500 Hz, −10 dB from 500 Hz to 2 kHz, −6 dB above 2 kHz |
    | Meeting | −3 dB (option −6 dB) |

  - **Bus state machine:** states A / AB / B, from S1 posteriors.
    - Enter A when p_A > 0.75 and p_B < 0.2 for 100 ms (B symmetric). Use AB during double talk or uncertainty. Hold the last state when nobody talks.
    - Let h_A(f) = 1 and h_B(f) = e^(jψ_AB(f)) be the AB beam's responses. Scale each state's weights so that the targeted partner keeps the AB response: state A uses w_A (response 1 to A), and state B uses h_B*(f)·w_B (response h_B to B; −1 in the dipole band).
    - Weights: w(l) = c_A·w_A + c_AB·w_AB + c_B·h_B*·w_B, where the c's are 50 ms linear ramps summing to 1.
    - **Phase continuity**, checked per bin for the partner who is not targeted. Consider r = w_Aᴴa_B, partner B's response in state A.
      - If |∠(r/h_B)| > 30° while |r| ≥ 0.5 (−6 dB), or > 90° while |r| < 0.5, use w_AB instead of w_A in that bin.
      - Apply the same check to partner A in state B.
      - This keeps every crossfade dip within 1 dB of the smaller endpoint magnitude, and within 0.3 dB in the first case (test B.7).
    - Never use w_A + w_B (it collapses to omni at low frequency).
  - **Meeting:**
    - Cache weights for a 64-point τ grid and steer to the active talker's τ̂, with a 300–500 ms hold.
    - Or use 3 fixed beams (τ = +d̂/c, −d̂/c, 0) with best-beam selection: maximum smoothed 300–3000 Hz output power, 50 ms crossfade, 500 ms hold.
  - **M2-adapt (experimental, bus only) — Elko adaptive first-order beam [5]:**
    - C_F = X_A − X_B·e^(−j2πf d̂/c) and C_B = X_B − X_A·e^(−j2πf d̂/c) (back-to-back cardioids).
    - y = C_F − β·C_B.
    - Update β ← clip(β + μ·Re{y·C_B*}/(|C_B|² + ε), 0.5, 1), with μ = 0.1. Freeze while either partner talks. β ≥ 0.5 keeps the rear partner at −6 dB or better.
    - EQ: 1/|1 − e^(−j4πf d̂/c)|, capped at +20 dB.
- **Target:** bus; meeting (steered).
- **Geometry:** τ_k learned; Γ measured or from d̂. With learned τ_k and measured Γ̂n it is robust to an error in the nominal spacing.
- **Expected gain (§1.2, WNG floor strictly enforced; 2.5 / 4 / 6 cm):**
  - single target on the axis: 5.2 / 5.0 / 4.6 dB (3-D) and 4.1 / 3.9 / 3.7 dB (2-D) at WNG ≥ −10 dB;
  - two-talker beam: 3.4 / 3.6 / 3.5 dB (3-D) and 2.2 / 2.3 / 2.3 dB (2-D) at WNG ≥ −10 dB; 4.4 / 4.1 / 3.7 dB (3-D) at WNG ≥ −20 dB;
  - single target averaged over azimuth: 2.5–2.7 dB (2-D); at broadside 0.8–2.0 dB; the two-talker beam at broadside is ill-posed (≈ 0 dB);
  - minus 0.5–1.5 dB for field losses;
  - meeting: 4.3 dB on the axis and 1.0 dB at broadside (3-D, WNG ≥ −3 dB), about 2.5 dB of DRR on average;
  - M2-adapt: ≤ 1 dB in babble.
- **Failure modes:**
  - mismatch at the −15 to −20 dB bus floors below 500 Hz (at the −10 dB floor, ±1 dB / ±3° costs ≤ 0.1 dB);
  - wind, vibration and handling amplified by |WNG|;
  - near c/2d the front and back steering vectors coincide;
  - wrong state means the beam points away from the active partner (the hypercardioid rear lobe is −6 dB), so prefer AB when uncertain;
  - partners at broadside get about 1 dB (single target) or nothing (two-talker);
  - mixing steering conventions (see the convention check).
- **Implementation:** numpy, about 180 lines; weights precomputed per τ pair; real-time.

### M3: Geometry-free RTF-MVDR (covariance whitening) and mask-driven MVDR

- **Idea:** learn each partner's relative transfer function (RTF) from frames where that partner dominates. Steer an MVDR with it against a measured or hybrid noise covariance. No array geometry is needed; mismatch, near-field effects and device-body scattering are absorbed into the RTF.
- **Sources:** [9] RTF-based beamforming; [10] covariance whitening; [11] Souden MVDR; [50][51] mask-based MVDR / GEV; [52] spatial priors and online operation.
- **Algorithm**
  - **Frame labels from S1** (or from M7 posteriors):
    - partner frames L_k = {p_k > 0.85, ψ high, no double talk};
    - noise frames L_0 = {p_A + p_B < 0.15}.
  - **Covariances**, recursive over labeled frames only, effective memory 2–4 s of labeled frames: Φ_x,k ← λ·Φ_x,k + (1−λ)·XXᴴ.
  - **Noise covariance (default, hybrid):** Φ_w = Γ̂n + σ_s·I.
    - σ_s is the self-noise ratio, estimated from the quietest 5% of frames.
    - The scale of Φ_w does not matter for whitening or MVDR.
    - Option "measured": use the average of XXᴴ over L_0.
  - **Covariance-whitening RTF estimate:**
    1. L = chol(Φ_w).
    2. M = L⁻¹·Φ_x,k·L⁻ᴴ.
    3. v = principal eigenvector of M (`np.linalg.eigh`, vectorized over bins).
    4. ĝ_k = L·v / (L·v)₁.
    This equals Φ_w·v_max with v_max the principal generalized eigenvector; it is **not** v_max itself.
  - **Plausibility check per bin:** over 300–3000 Hz require |∠ĝ_k,2 + 2πf·τ_k| < 60° and |ĝ_k,2| within ±3 dB. Otherwise use M2's model steering for that bin.
  - **MVDR:** w_k = (Φ_w+μI)⁻¹ĝ_k / (ĝ_kᴴ(Φ_w+μI)⁻¹ĝ_k).
    - Same WNG floors as M2.
    - Compute WNG with ĝ normalized to ‖ĝ‖² = 2.
  - **Two-talker beam:** soft LCMV with A = [g̃_A, g̃_B], where g̃ are the symmetrized (center-referenced) RTFs of §2.0, and the M2 phase rule:
    - h = [1, −1] in bins where that solution meets the WNG floor;
    - the continuation search elsewhere.
    - Never apply h = [1, −1] to RTFs normalized to Mic A: that is the poor Mic-A-referenced beam of M2's convention check (−0.4 dB AI-DI at 4 cm).
  - **State machine and crossfade:** identical to M2.
  - **Updates:** recompute RTFs every 0.5 s. Freeze when fewer than 50 labeled frames are available.
  - **Mask-driven variant:** Φ_s and Φ_n weighted by an M7 or M8 mask (one mask shared across channels), Souden form h = Φ_n⁻¹·Φ_s·e_ref / tr(Φ_n⁻¹·Φ_s).
- **Target:** bus (meeting optional, steered per cluster).
- **Geometry:** none. TDOA is used only to label frames.
- **Expected gain:**
  - Same 6 dB first-order ceiling as M2, and about equal to M2 when the geometry is known.
  - Hypothesis to test: +0.5 to +1.5 dB over M2 under spacing error, mismatch, near-field or scattering.
  - Against one dominant point interferer in a reverberant cabin: roughly 3–8 dB, while giving up the two-talker constraint.
- **Failure modes:**
  - babble frames mislabeled as partner frames steer the beam to a passenger;
  - partner leakage into a measured Φ_w cancels the target (the reason the hybrid model is the default);
  - a moving device or partner;
  - only one null.
- **Implementation:** numpy, about 120 lines.

### M4: Coherence (CDR) post-filter — hearing-aid/phone-style near-talker gate

- **Idea:** estimate the coherent-to-diffuse power ratio (CDR) in each time-frequency bin and turn it into a Wiener-like gain. Partners inside r_c are coherent; babble, road noise and late reverberation are diffuse.
  - The DOA-free estimator needs only Γn.
  - The DOA-dependent estimator, with two partner hypotheses, also rejects coherent sources from other directions.
- **Sources:**
  - [34] CDR estimators;
  - [35] real-bus evidence and the beamformer correction;
  - [36] coherence-based post-filter;
  - [37] signal-to-diffuse estimation;
  - [8] APES W1 post-filter;
  - [38] Allen gain (gentle baseline).
- **Algorithm**
  - **Smoothing:** smooth Φ with λ = 0.9 (sweep 0.68 / 0.8 / 0.9). Γ̂x = Φ_AB/√(Φ_AA·Φ_BB), clipped to |Γ̂x| ≤ 0.999. Use Γn = Γ̂n (or ρ̂·model at d̂), clipped to ≤ 0.95.
  - **DOA-independent (Schwarz prop3):**
    `CDR = max(0, [Γn·ReΓx − |Γx|² − √(Γn²·(ReΓx)² − Γn²·|Γx|² + Γn² − 2Γn·ReΓx + |Γx|²)] / (|Γx|² − 1))`
  - **DOA-dependent (prop1), per partner k:**
    `CDR_k = max(0, Re{Γs,k*·(Γn − Γx)} / (Re{Γs,k*·Γx} − 1))`, with Γs,k = e^(+j2πf·τ_k).
    Bus: CDR = max over k ∈ {A, B} of CDR_k.
  - **Conditioning mask:** the estimators divide by quantities that vanish when the target's coherence is close to the noise's.
    - Use a bin only where |Γs,k(f) − Γn(f)| ≥ 0.2 for the partner k being tested.
    - For a broadside partner (3-D model) that means above 2.47 / 1.54 / 1.03 kHz at d = 2.5 / 4 / 6 cm. For an endfire partner it means above 0.44 / 0.27 / 0.18 kHz.
    - At 1 kHz a broadside partner has |Γs − Γn| = 0.03 (2.5 cm) or 0.09 (4 cm), so the estimate there is noise.
    - Ill-conditioned bins get the frame's median CDR over the well-conditioned bins in 1–4 kHz. This replaces the v1.0 rule "below f_c, where Γn > 0.9".
  - **Correction for the beam that precedes the post-filter** [35]: CDR_out = CDR · DF_beam(f), with DF = 1/(wᴴ·Γn·w) for the active weights.
  - **Gain:** G = max(G_min, 1 − √(µ/(CDR_out + 1))).
  - **Smoothing of G:** 3-bin frequency median. Over time, rises are instant; falls are limited to 4.8 dB per 8 ms hop (about 60 dB per 100 ms, the minimum-decay rule from [70]).
  - **Parameters:** µ = 1.3, G_min = 0.15 and λ are starting points only. The values in [34] were tuned at d = 8 cm. Retune on the dev set for 2.5 / 4 / 6 cm by sweeping µ 0.5 / 1.0 / 1.3 / 1.6 / 2.0, G_min 0.1 / 0.15 / 0.3 and λ 0.68 / 0.8 / 0.9. Select by worst-partner ESTOI and ΔSNR_active (B1, B3), never by BAK.
  - Note that G(CDR = 0) = 1 − √µ: µ = 0.3 caps attenuation of fully diffuse bins at −6.9 dB, µ = 0.5 at −10.7 dB.
  - **Estimator floor:** on pure diffuse noise at λ = 0.68, the short-time CDR has a median of −0.4 to −0.8 dB and 45–47% of bins above 0 dB. Only the power-averaged gain (−8.6 to −9.7 dB) is dependable. Do not read individual bins with CDR ≈ 0 as "noise".
  - **Meeting:** not used except as an ablation. Use WPE (M10) for dereverberation (§2.2).
  - **Variants:**
    - M4b: McCowan–Bourlard gain [36].
    - M4c: APES W1 = P_ZZ/P_YbYb, superdirective output over delay-and-sum output, with the product of gains clipped to [0.05, 1] [8].
    - M4d: Allen gain |Γ̂x| [38].
- **Target:** bus.
- **Geometry:** the DOA-free version needs only Γ̂n (geometry-free if measured). The DOA-dependent version needs learned τ_k.
- **Expected gain** (verification simulation, 0 dB SNR, partner DRR +6 dB, T60 0.45 s, µ 1.3, G_min 0.1):
  - λ = 0.68: noise attenuation 5–6 dB, target attenuation about 4 dB, net ΔSNR +1.8–2.0 dB at endfire and +1.2 dB at broadside;
  - λ = 0.9: net +2.8–3.6 dB, with the target attenuated by 6–7 dB, so the gain trades speech level for SNR;
  - endfire beats broadside by 0.6–1 dB; 6 cm beats 2.5 cm by up to 0.5 dB;
  - about 0 dB for a partner at DRR ≈ 0 dB (about 1 m in the bus);
  - real bus recordings (CHiME-3, after a 5-mic beamformer): WER 18.53% → 17.42–17.71% [35];
  - mostly a comfort (BAK) gain; classical post-filters rarely raise intelligibility [45].
- **Failure modes:**
  - coherent passengers pass;
  - low-frequency engine noise passes;
  - partners at or beyond r_c are attenuated;
  - musical noise when λ is small;
  - phase mismatch biases Γ̂x (calibrate first);
  - meeting: a talker at DRR −8 dB loses 5.6 dB (9 dB at λ = 0.9), with −0.8 dB net ΔSNR in 1–4 kHz.
- **Implementation:** numpy, about 90 lines. Port the estimators from the authors' MATLAB code (`estimate_cdr_nodoa.m`, `estimate_cdr_robust_unbiased.m`, BSD-2). scipy's `ShortTimeFFT` cannot build the paper's 1024-window / 512-FFT filterbank; use `lab.dsp` 512/512/128.

### M5: Two-hypothesis phase mask ("acoustic fence")

- **Idea:** attenuate bins whose inter-mic phase matches neither partner's TDOA. This rejects coherent passengers off the partners' cones, which M4 cannot do. It is the 2-mic version of industry pickup zones: Poly Acoustic Fence [31] and Qualcomm's directional-coherency detection [26].
- **Sources:** [40] phase-error filtering; [39] coherence-based dual-mic filter; [26][31] industry.
- **Algorithm**
  - Phase error: e_k(l,f) = wrap(∠Φ_AB(l,f) − 2πf·τ_k), with Φ_AB smoothed at λ = 0.7.
  - Mask: η = max over k of 1/(1 + γ·e_k²), with γ = 4 rad⁻² (sweep 1 / 4 / 16).
    - Example at 4 cm and 2 kHz: a broadside passenger has e = 1.47 rad, so η ≈ 0.10 (−20 dB).
  - Apply only for f in [1 kHz, min(4 kHz, 0.8·c/2d̂)]. Outside that band, use the band median of η (bus) or 1.
  - G_fence = max(0.3, η), smoothed as in M4.
  - Do not port the Yousefian–Loizou low-band gain G1 = 1 − |ReΓ̂|^16: it attenuates on-axis targets by about −3.5 dB at 500 Hz (4 cm).
- **Target:** bus.
- **Geometry:** learned τ_k; d̂ only for the band edge.
- **Expected gain:**
  - A few dB of extra rejection of side passengers (TDOA ≈ 0) in 1–4 kHz.
  - 0 dB against sources on the same cone.
  - The large published intelligibility gains [39] were for point maskers at least 60° away, in a mild room (T60 ≈ 220 ms), at −5 dB SNR. Diffuse babble is that method's weak case.
- **Failure modes:**
  - interferers on the same cone;
  - phase noise at low SNR causing musical noise;
  - partner head movement (keep τ_k tracking on).
- **Implementation:** numpy, about 30 lines.

### M6: Spatially gated single-channel noise suppression (phone/hearing-aid style)

- **Idea:** remove stationary noise (engine, road, HVAC, projector) with an MMSE speech-presence noise tracker and a log-spectral-amplitude gain. Freeze the noise update while a spatial cue says a partner is talking (the Audience ILD-gated tracker [25], with our CDR or fence cue in place of ILD). Optionally, trust a two-channel noise reference only in bands where it separates well (Apple [27]; Oticon's back-cardioid noise estimate [22]).
- **Sources:** [43] noise tracking; [44] LSA gain; [41][42] minimum statistics and IMCRA alternatives; [25][27][22] industry; [45] intelligibility caveat.
- **Algorithm** (on the beam output Y)
  - **Speech presence probability:**
    `P = [1 + (1 + ξ_H1)·exp(−(|Y|²/σ̂N²)·ξ_H1/(1+ξ_H1))]⁻¹`, with ξ_H1 = 15 dB.
    - Stagnation guard: P̄ ← 0.9·P̄ + 0.1·P; if P̄ > 0.99 then P ← min(P, 0.99).
  - **Noise tracking:** E|N|² = (1 − P)·|Y|² + P·σ̂N²; σ̂N² ← α·σ̂N² + (1 − α)·E|N|², with α = 0.9 per 8 ms hop.
  - **Spatial gating (bus):** if the frame's target cue (mean of G_cdr·G_fence over 1–4 kHz) exceeds 0.6, hold σ̂N² above 500 Hz. Below 500 Hz always update; engine noise is stationary.
  - **Meeting:** update σ̂N² only in VAD pauses longer than 300 ms. This stops the noise floor pumping between talkers.
  - **Gain:**
    - Decision-directed a priori SNR: ξ = α_dd·G²(l−1)·γ(l−1) + (1 − α_dd)·max(γ − 1, 0), with α_dd = 0.98 and ξ_min = −25 dB.
    - G = ξ/(1+ξ)·exp(½·E1(ξγ/(1+ξ))), where E1 is `scipy.special.exp1`.
    - Floor: −15 dB (bus), −12 dB (meeting).
  - **Optional two-channel reference:**
    - The blocking signal U = X_B − ĝ_k·X_A (RTF of the active partner, from M3) contains no target.
    - Estimate the noise PSD by inverting the blocking path's known diffuse-field response.
    - Per band, use the larger of the one-channel and two-channel estimates, but only where the measured target suppression in U is at least 10 dB.
- **Target:** both. **Geometry:** none.
- **Expected gain:** −6 to −12 dB of stationary noise in pauses; better comfort and BAK; little intelligibility gain [45]; non-stationary babble is mostly untouched.
- **Failure modes:**
  - babble peaks treated as speech;
  - musical noise with low floors;
  - speech attenuated at low SNR;
  - noise-floor pumping if the tracker is too fast.
- **Implementation:** numpy, about 70 lines.

### M7: Anchored spatial clustering (cACGMM with classes A, B, noise) + per-talker mask-MVDR + floored post-mask

- **Idea:** cluster the normalized 2-channel STFT vectors in each bin into partner A, partner B and a diffuse/garbage class, using a complex angular central Gaussian mixture (cACGMM). The posteriors give covariances; each partner gets a Souden MVDR; the outputs are selected by posterior; a post-mask is applied with a floor.
- **Sources:**
  - [49] cACGMM;
  - [47] bin-wise clustering and permutation alignment;
  - [48] MESSL (garbage class, ITD grid);
  - [50][11] mask-based MVDR;
  - [53][54] guided source separation and its practical lessons;
  - [52] spatial prior and block-online operation;
  - [46] DUET (TDOA anchors only).
- **Algorithm**
  - **Features:** z_tf = x_tf/‖x_tf‖ (two-dimensional).
  - **Initialization:** B_A = a_A·a_Aᴴ + 0.1·I and B_B likewise (center-referenced steering from τ_A, τ_B); B_0 = Γ̂n + 0.05·I; normalize to tr(B) = 2; class weights α_k = 1/3.
  - **EM per bin (vectorized):**
    - E-step: γ_k ∝ α_k·det(B_k)⁻¹·(zᴴB_k⁻¹z)⁻².
    - M-step: B_k = 2·Σ_t γ_k·zzᴴ/(zᴴB_k⁻¹z) / Σ_t γ_k; floor eigenvalues at 10⁻³·λ_max; renormalize the trace.
    - α_k = mean of γ_k over time, or, GSS-style, time-varying α_tk ∝ α_k·a_tk with activity a_tk from S1. The noise class always has activity 1.
  - **Iterations:** offline, 10–20 over the whole file (up to 60 s). Block-online, 2.4 s blocks with 2.4 s of past context and 1–2 iterations per block, warm-started.
  - **Permutation check:** after EM, compare the phase of the principal eigenvector of B_A and B_B with a_A and a_B, and swap where reversed.
  - **Covariances from the unnormalized X:**
    - Φ_k = Σ γ_k·XXᴴ / Σ γ_k;
    - Φ_N = Σ γ_0·XXᴴ / Σ γ_0, plus loading 10⁻³·tr/2.
  - **Per-partner MVDR:** h_k = Φ_N⁻¹·Φ_k·e_ref / tr(Φ_N⁻¹·Φ_k), with e_ref the mic of higher estimated SNR.
  - **Output:** Ŝ = Σ_k (γ_k/(γ_A + γ_B + ε))·h_kᴴX, then post-mask M = max(G_min, γ_A + γ_B) with G_min = 0.3 (sweep 0.1 / 0.2 / 0.3) [50].
- **Target:** bus.
- **Geometry:** none beyond the TDOA anchors.
- **Expected gain:**
  - On noise-free mixtures with 2 mics at 4 cm: about 15.7 → 8.7 dB SDR for 2 sources as T60 goes 0.13 → 0.44 s [49].
  - With 0–5 dB babble, expect a few dB, mostly above 1–1.5 kHz. The low-frequency classes overlap.
  - Treat 8 dB as optimistic.
- **Failure modes:**
  - diffuse low-frequency babble is indistinguishable from broadside partners;
  - a class drifts when its partner is silent for long (use activity guidance or priors);
  - permutation errors;
  - musical noise with low floors.
- **Implementation:** numpy, about 150 lines. Adapt the pb_bss cACGMM (MIT) without its heavy dependencies; do not install pb_bss.

### M8: Neural single-channel post-filter / mask estimator (ONNX, no PyTorch)

- **Idea:** "beam, then DNN mask", as in current premium hearing aids [23]. Run a causal model trained on the DNS noise-suppression data on the linear beam output. Use its magnitude mask:
  - combined once with the spatial masks;
  - optionally as the speech mask for mask-driven MVDR [50][51].
- **Sources:** [74] DPDFNet; [71] DeepFilterNet; [73] GTCRN; [72] NSNet2; [69] RNNoise; [70] PercepNet (ideas); [23] Phonak; [82][81][87] evidence and caveats.
- **Models, in porting order.** Each needs a reference implementation to validate against.
  1. **GTCRN:** 0.54 MB; 33 MMAC/s; a cheap mask. Reference: sherpa-onnx's speech-enhancement release and its `offline-speech-enhancement-gtcrn.py` example.
  2. **DPDFNet-4:** 16 kHz native; 2.84M parameters; 2.36 GMAC/s; `dpdfnet4.onnx`, 11.2–11.7 MB depending on the source. Reference: sherpa-onnx's DPDFNet example. Use **DPDFNet-2** (1.35 GMAC/s) if RTF is tight.
  3. **NSNet2:** only from a pinned source. It is not in sherpa-onnx's speech-enhancement release (that has GTCRN and DPDFNet; VoxLock has its own release), and it is not on the DNS-Challenge master branch.
     - Port it only if a copy of the original NSNet2 baseline (the ONNX model plus its reference enhancement script) can be located in the repository history or an archive.
     - Record the commit hash and the model's SHA-256, and validate against that script. Otherwise drop it.
  4. **DeepFilterNet3:** reference only (48 kHz internally).
- **Algorithm**
  - Run the model on the time-domain linear output y_lin (before masks). Measure its delay once on speech (§4.2) and compensate it. Algorithmic latency: DPDFNet uses a 20 ms window and a 10 ms hop.
  - Mask in our STFT: m_nn = clip(|Ŝ|/(|Y| + ε), 0, 1). For NSNet2, use the model's own gain.
  - Attenuation limit: m_nn ← max(m_nn, 10^(−A/20)), with A = 20 dB (bus) or 12 dB (meeting). This is equivalent to mixing some input back in [71][87].
  - Combine with the spatial masks: m = max(G_floor, m_spatial·m_nn). Never cascade one model's output into another.
  - Meeting: use only as an HVAC/projector suppressor. Reject the configuration if any distance bin's pre-AGC attenuation (T2), WER (T6) or ESTOI (T7) gets worse.
- **Target:** both (meeting limited). **Geometry:** none.
- **Expected gain:**
  - Published, on a synthetic low-SNR additive set: DPDFNet-4 PESQ 1.32 → 3.18, STOI 83.2 → 93.0%, SI-SNR 0.38 → 14.1 dB.
  - On real DNS4 recordings, DNSMOS SIG goes 3.23 → 3.40 for DPDFNet-8, whereas NSNet2, GTCRN and DTLN lower SIG [74].
  - In 0 dB bus babble, expect mainly a comfort/BAK gain and ΔESTOI ≈ 0 to +0.05.
  - The model is not speaker-conditioned, so intelligible passengers survive.
- **Failure modes:**
  - competing speech is preserved;
  - speech distorted at low SNR;
  - soft or far talkers attenuated (use the attenuation limit);
  - far-field reverberant input is out of domain.
- **Implementation:** onnxruntime (already installed) plus numpy pre/post-processing.
  1. In the `scoring_venv` (never the shared one), install sherpa-onnx 2.0.1 (2.1 MB wheel plus about 9.5 MB core). Generate reference outputs with its GTCRN and DPDFNet configurations, for example `OfflineSpeechDenoiserDpdfNetModelConfig(model=..., attenuation_limit_db=...)`.
  2. Inspect the ONNX I/O (`session.get_inputs()`). Port the pre/post-processing from the Apache-2.0 DPDFNet code or from sherpa-onnx.
  3. Validate the port against the reference: SI-SDR > 40 dB.
  4. The bench method then imports only onnxruntime.
  - Models come from the sherpa-onnx `speech-enhancement-models` release or Hugging Face `Ceva-IP/DPDFNet`.
- **Research track, not for the bench:** H-GTCRN [75] (MIT, trained for exactly 2 mics at 4 cm) needs WPE and IVA moved out of torch before ONNX export, and fine-tuning needs PyTorch on another machine.

### M9: Enrollment-based target-speaker extraction for both partners (VoxLock ×2, union mask)

- **Idea:** use speaker identity instead of space. Run a causal target-speaker extraction (TSE) model twice, once per partner, and keep the union of the two outputs. Intelligible nearby passengers are removed; no 2-mic spatial or coherence method can do that. Apply it strongly only when bystander speech is present (VoiceFilter-Lite idea [77]).
- **Sources:**
  - [79] VoxLock;
  - [80] REAL-TSE;
  - [82] human-listener evidence;
  - [77][78] personalized-filter ideas;
  - [81][83] risks;
  - [84] target-conversation extraction and [88] noisy enrollment (research alternatives);
  - [76] evidence that 2-mic phase features add little.
- **VoxLock facts that shape the design:** it extracts a single target. In automatic mode it picks the dominant voice in the first 15 s, which on the bus would keep one partner and remove the other. Always pass an explicit enrollment sample (≥ 3 s) for each partner, and never run automatic mode on the bus.
- **Algorithm**
  - **Enrollment, product:** each partner records 5–10 s in quiet once. This is preferred.
  - **Enrollment, bench (self-enrollment), gated on near-field confidence:**
    - From S1/M3, collect frames where partner k dominates: p_k > 0.9, |τ̂ − τ_k| < 10 µs, frame CDR > +3 dB in conditioned bins, no double talk.
    - Take the linear beam output steered to k, keep contiguous runs of at least 0.5 s, highest SNR first, up to 3–10 s total.
    - Require at least 3 s. Also require the cosine similarity between embeddings of the first and second halves of the sample to be ≥ 0.7 (same speaker model as S1). Otherwise skip TSE for that file.
  - **Extraction:** run VoxLock on y_lin for k = A and B with their enrollment samples: `enroll.onnx` produces the conditioning; `chunk_int8.onnx` runs on 40–80 ms chunks carrying LSTM state. Compensate the latency (about 24 ms plus the chunk length).
  - **Masks:** m_k = clip(|Ŝ_k|/(|Y| + ε), 0, 1); union m_tse = max(m_A, m_B).
  - **Adaptive strength (VoiceFilter-Lite-style):**
    - f(l) = clip(1 − Σ m_tse²|Y|² / Σ |Y|², over 300–4000 Hz, 0, 1) × VAD(l).
    - w(l) = 0.8·w(l−1) + 0.2·(0.8·f(l) + 0.2), per 10 ms frame.
    - m_eff = (1 − w) + w·m_tse.
  - **Guard:** if partner k's output level in its own segments drops more than 6 dB relative to the input, set m_k = 1 when p_k is high.
- **Wrong-speaker-lock rate (bench metric):** for each file and each k, compare the extracted stream's separated components (§4.2) with partner k, the other partner and the nearest passengers. Count a lock error when partner k's share of the output energy is not the largest. Report the rate for oracle enrollment versus self-enrollment.
- **Target:** bus only. Never use in the meeting.
- **Geometry:** none (enrollment only).
- **Expected gain:**
  - VoxLock self-reported, on near-mic calls (not peer reviewed): leaked bystander words 66.5 → 0.11 per minute; WER 23.2 → 4.95%.
  - Peer-reviewed analogue: +17% sentence intelligibility for normal-hearing listeners in restaurant noise [82].
  - For this device at 0.3–1 m in babble: unknown. Real-world conversational TSE baselines score token error rates of 0.65–0.77 [80].
- **Failure modes:**
  - babble-contaminated enrollment (the main failure);
  - enrollment locking onto the wrong talker (measured by the lock rate);
  - partners with similar voices;
  - overlap between the partners;
  - far or reverberant partners;
  - cost: VoxLock runs at RTF 0.6–1.1 per pass on one Xeon thread, so two passes take about 1.2–2.2× real time single-threaded. Run the two passes in parallel processes with 2 threads each and measure. If RTF ≥ 0.5, Tier 3 is "offline-only" (§2.3).
- **Implementation:**
  - Models: `sherpa-onnx-voxlock-int8.tar.bz2`, 72.5 MB, Apache-2.0.
  - Run with onnxruntime and the inference loop vendored from the pure-Python `voxlock` 1.3.0 package (Apache-2.0); drop `soxr` and `huggingface_hub`.
  - Validate against sherpa-onnx in the `scoring_venv`.

### M10: Two-channel WPE dereverberation

- **Idea:** blind multichannel linear prediction removes late reverberation for every talker, without any position knowledge. The filter is linear, so it produces no musical noise, and it preserves the cues between channels for the later stages.
- **Sources:** [59][60] WPE and its MIMO form; [61] NARA-WPE; [62] Google 2-mic online WPE; [63] REVERB challenge; [64] WPD variant.
- **Algorithm**
  - STFT 512/128, D = 2 channels.
  - Taps K = 10 (sweep 10 / 15 / 20); delay Δ = 3 (sweep 2 / 3); 3 iterations.
  - Per bin, iterate:
    - λ_t = mean over d of |X_d,t|², averaged over ±1 frame and floored;
    - R = Σ ỹỹᴴ/λ_t;
    - P = Σ ỹyᴴ/λ_t;
    - G = R⁻¹P;
    - X_t = Y_t − Gᴴỹ_t.
  - **Variants:**

    | Variant | Description |
    |---|---|
    | (a) Offline global | One filter for the whole file |
    | (b) Offline segment-wise | Segments from S1 change points; filter estimated on [segment − 3 s, segment + 1 s], applied to the segment, 50 ms crossfades |
    | (c) Online RLS | `OnlineWPE`: taps 10, delay 2, α = 0.998, with per-speaker storage and restore of (G, P) on each confirmed talker change. The key is the S1 speaker-embedding cluster, with \|τ̂\| as a secondary cue: 20 seats give only a median of 8 resolvable TDOA groups, and mirror-image seats share a TDOA but not a room response |

    Google's α = 0.9999 is too slow for 20 alternating talkers. Variant (c) is exercised only by the E7 returning-talker scenes; the v1.0 scenes, where each talker speaks once, never test it.
- **Target:** meeting. **Geometry:** none.
- **Expected gain:**
  - ASR evidence: Google's 2-mic (71 mm) online WPE cut WER by 5–8% relative [62].
  - In the REVERB challenge listening tests, two-channel linear processing improved overall quality [63].
  - ESTOI gain for talkers at 3–6 m with two mics is unknown; measure it. Expect partial tail reduction.
- **Failure modes:** filter mismatch for 1–2 s after a talker change; speech distortion with too little context; moving talkers; additive noise is not modelled.
- **Implementation:** vendored NARA-WPE `wpe_v8` / `OnlineWPE` without click (MIT), or about 100 lines written from scratch.

### M11: Leveling — speaker-keyed speech-gated AGC + gain-sharing automix + off-attenuation (conferencing style)

- **Idea:** give all 20 talkers even loudness without pumping room noise.
  - Measure speech level only on confident speech, with a cap on output noise (WebRTC AGC2 [29]).
  - Remember a gain per talker (Nureva remembers one per position [30]; here the key is a speaker-embedding cluster, because positions are not resolvable, §1.5).
  - Share gain across beams or nodes (Dugan [28]).
  - Attenuate in pauses (gating, as in Poly NoiseBlock [31]), within the noise-modulation limit of §4.3–4.4.
  - Measure loudness per EBU R128 / ITU BS.1770 [33].
- **Sources:** [29][30][28][31][33][32]; [68] diarization option; [13] TDOA.
- **Algorithm**
  - **VAD:** p_s per 10 ms, from energy plus CDR. Optionally, sherpa-onnx's bundled ONNX VAD (scoring venv for reference, vendored for the method).
  - **Speech level L_c per speaker cluster c** (S1 embedding clusters; DP-means clusters only as a fallback):
    - update only on frames with p_s ≥ 0.95 for at least 12 consecutive frames (the AGC2 rule);
    - leaky RMS over 300–4000 Hz with a time constant of about 1 s of speech.
  - **Target:** T = −26 dBFS active speech level (configurable).
  - **Gain:** g_c = clip(T − L_c, −10, +45 dB). Raw speech sits at about −60 to −65 dBFS, so up to about 40 dB of gain is needed.
  - **Noise cap:** g ≤ −50 dBFS − L_noise, with L_noise from M6 after noise suppression.
  - **Dynamics:**
    - within one talker, at most 6 dB/s;
    - on a confirmed change to a known cluster, crossfade to the stored g_c over 50–100 ms;
    - for a new cluster, start from a prior gain, adapt at up to 20 dB/s for 0.5 s, then 6 dB/s.
  - **Prior gain for a new cluster:**
    - If the mic axis lies along the table (§5), |τ̂| tracks talker distance (Spearman 0.98 in the lab meeting geometry, against 0.06 for a random azimuth). Use the median stored gain of earlier clusters in the same 10 µs |τ̂| bin.
    - Otherwise, use the room-average gain.
  - **Pauses:** hold the gain. Apply off-attenuation of −3 dB (sweep 0 / −3 / −6) after 300 ms without speech; restore within 20 ms (20 ms look-ahead offline). Keep the noise modulation (idle minus active noise attenuation, T4) ≤ 6 dB, including M6's own share. Spectral noise suppression already attenuates noise more in pauses than under speech.
  - **Limiter:** −1 dBTP, 4× oversampled peak detection, 5 ms look-ahead, 50 ms release.
  - **Automix across nodes (Dugan) [28]:**
    - amplitude gain g_k = √(P_k / Σ_j P_j), where P_k is the 300–4000 Hz power smoothed over 20 ms;
    - gain smoothing: attack 10 ms, release 200 ms.
    - This keeps the background at the level of one open mic.
    - Within one node, beams are correlated, so prefer best-beam selection.
  - **Offline option:**
    - diarize with pyannote segmentation-3.0 (ONNX, 7 MB) plus speaker embeddings [68];
    - apply a static per-speaker gain from BS.1770 loudness (capped at +20 dB);
    - normalize the programme to −23 LUFS.
  - **Bus mode:** update levels only when p_A + p_B > 0.8. A plain VAD also fires on bystanders.
- **Target:** meeting (primary); bus (gated).
- **Geometry:** none (clusters are relative).
- **Expected gain:** these are targets, not verified figures.
  - post-AGC level std across talkers, measured on the separated speech (T3): 2.6 dB (lab, noisy-output loudσ) → ≤ 1.5 dB;
  - far-bin pre-AGC attenuation no worse than near-bin + 3 dB (T2), so that the AGC does not re-boost a suppressed far talker together with its noise floor;
  - noise attenuation during speech ≥ 3 dB with M6 (T4);
  - ESTOI unchanged. AGC scales reverberation along with speech.
- **Failure modes:**
  - two talkers merged into one cluster, which embeddings reduce but do not remove;
  - the noise cap limits far-talker gain in noisy rooms;
  - pumping;
  - the first syllables of a new talker come out at the wrong gain.
- **Implementation:** numpy, about 220 lines, plus `spk.py` for the embeddings. pyloudnorm is not installed, so implement K-weighting in scipy and validate it against pyloudnorm in the `scoring_venv`.

### Considered and not recommended (with reasons)

- **GSC / TF-GSC adaptive cancellers** [6][9]: 0 dB against diffuse noise for N = 2 [7]; only one null; cancel the target in reverberation. A short experiment is acceptable only against one loud neighbour.
- **AuxIVA, ILRMA, online IVA for hearing aids** [56][57]: two outputs that both still contain babble. No reduction of diffuse noise; useful only against one point interferer.
- **FastMNMF** [58]: no evidence with two mics; slow.
- **DUET as a separator** [46]: binary masks need d ≲ 2.15 cm at 16 kHz. Use it only to find TDOA anchors.
- **Guided source separation in the meeting** [53][54]: needs diarization; going from 7 to 2 channels loses most of the gain (WER reduction 50.4% / 21.8% relative with 7 versus 2 channels). `desh2608/gss` requires CuPy.
- **Neural continuous speech separation (CSS)** [66]: separation degraded non-overlapped speech, and 3 mics gave no meaningful gain.
- **SpeakerBeam / FaSNet** [76]: no weights; two-mic phase features add about 0.3 dB.
- **Phone level-difference methods** [24][25]: the level difference is ≤ 1.1 dB at 4 cm (1.6 dB at 6 cm), below the mismatch and the reverberation error (§1.3).
- **Any personalized model or TSE in the meeting:** removes the non-enrolled talkers by design [81][83].
- **CDR post-filter in the meeting:** it cuts far talkers (DRR −8 dB) by 5.6–9 dB for no net SNR gain in 1–4 kHz. Use WPE (§2.2).
- **TDOA as talker identity in the meeting:** 20 seats give a median of 8 resolvable groups (§1.5). Use it only as a secondary key behind speaker embeddings.
- **Amplitude blending of the dipole with an omni:** the sum is a cardioid that nulls one partner (§1.2).

---

## 4. Evaluation protocol

### 4.1 Scenes and data (extends the existing lab)

**Current lab (v1.0) and its known defects**
- **Too few geometries.** dev: 13 bus + 5 meeting scenes; test: 8 + 3. Scenes that share a seed share a geometry, so dev holds only 3 bus geometries and 3 meeting rooms, and test holds 3 + 3.
- **Speech is macOS `say` TTS.**
  - 4 of the 12 bus target voices (Fred, Kathy, Ralph, Junior) are formant synthesizers.
  - The babble is non-English voices reading English.
  - The NN models, DNSMOS and ESTOI were trained or calibrated on natural speech.
- **Bus scenes are too easy.**
  - Partners are 0.3–0.8 m away, but passengers are always ≥ 1.0 m away: an artificial guard band.
  - Partners speak at Lombard 66–70 dB SPL at 1 m, while passengers in the same noise speak at only 60–68.
  - Partners always sit on opposite sides of the device (±40°), the one geometry a figure-8 handles well.
  - All sources are omnidirectional.
  - The late reverberation tail and the road noise have exactly the sinc coherence that the CDR and MVDR models assume. Real tails deviate from it ([34], Fig. 5).
  - There is no handling, vibration or wind noise.
- **Orientation is uncontrolled.** Partner angles from the mic axis (A/B, folded to 0–90°):
  - dev s0 21°/38°, s1 78°/88°, s2 74°/49°;
  - test s10 46°/56°, s11 10°/4°, s12 15°/9°.
  In s1 both partners are near broadside and the figure-8 nulls both. Dev and test differ systematically, so G1/G2 decisions could flip on orientation luck.
- **Meeting scenes.**
  - The device floats at 0.78 m with no table. The dominant tabletop reflection is missing, while a floor reflection that the table would block is present.
  - The farthest talker is 4.6–4.85 m away, against the 6 m requirement.
  - Each talker speaks exactly once, so per-talker memory is never exercised.
  - There is no standing presenter, and all talkers are omnidirectional.
- **Not reproducible.** pyroomacoustics' global RNG is unseeded, so the randomized image-source jitter changes every time a scene is regenerated.
- **Components incomplete.** `finalize()` adds sensor noise and per-mic gains to the mix but stores only Mic-A components (`comp_*`).
- Methods are told 40 mm; the true spacing is 25 / 40 / 60 mm (keep this).

**Required changes**

| Code | Factor | Change |
|---|---|---|
| E0 | Natural speech (before any method selection) | Partners and meeting talkers: LibriSpeech test-clean (346 MB, CC BY 4.0). Conversational dynamics (back-channels, interruptions, returning talkers): a conversational corpus such as AMI individual-headset audio [107]; check the licence and download size first. Babble: natural speech from speakers never used as targets (for example LibriSpeech test-other). TTS only for smoke tests. G1 is decided on E0 scenes |
| E1 | Bus geometry strata | Angles folded to 0–90° from the axis. Aligned (both partners ≤ 20° from the axis), 45° (30–60°), broadside (both ≥ 70°); a side-by-side seated pair (same side, 20–40° apart); a device lying flat in a lap (partners above it, \|τ\| ≈ 0 for both). Equal counts per stratum in dev, wide and test |
| E2 | Bus partner distance | 0.3–1.0 m; report bins 0.3–0.45 / 0.45–0.65 / 0.65–1.0 m |
| E3 | Bus RT60 | 0.15–0.6 s (vehicle cabins are heavily damped; §1.4) |
| E4 | Bus interferers | Nearest passengers from 0.6 m (no guard band). Passenger levels from the partners' Lombard distribution (62–72 dB SPL at 1 m). 1–2 conversing pairs at 0.6–1.2 m at partner level. A passenger on the axis behind partner A at 1.5–2 m (same cone). Door and brake transients at 85–97 dB SPL [96]. Store early images for the 3 nearest passengers and for each conversing pair |
| E5 | Source directivity | Talker directivity in pyroomacoustics; version 0.10.1 supports it only with the image-source method in shoebox rooms. Partners face each other; passengers face random directions; the meeting presenter faces away from the device |
| E6 | Noise-field realism | (a) Real 2-channel noise: DEMAND [94] TBUS, PCAFETER and OMEETING (adjacent mics 5 cm apart; download only these zips), and ETSI TS 103 224 "Inside Bus" [108] (8 channels; check the pair spacing and licence). (b) In half the scenes, replace the sinc-coherent synthetic tail with a full-order image-source tail, so CDR and MVDR are not tested on their own model. (c) Handling, vibration and wind: spatially uncorrelated bursts below 500 Hz |
| E7 | Meeting realism | Explicit table: add the first-order table image (specular point inside the 8 × 2 m table, reflection factor 0.9) and zero the first-order floor image whose specular point lies under the table (edit `source.damping` between `room.image_source_model()` and `room.compute_rir()`; unit test B.10). Mic height above the table 1 / 2 / 10 / 20 cm. Talkers out to 6 m (a second row, and a standing presenter facing away). Returning talkers (each ≥ 3 turns), back-channels and interruptions, in scenes of 5–10 min. Soft talkers at 52–55 dB SPL at 1 m. HVAC at 35 / 40 / 45 dBA. Mic axis along the table versus random |
| E8 | Hardware impairments | ±1 dB per mic (present); high-pass corners drawn per mic from 50–80 Hz (frequency-dependent phase mismatch); fixed inter-channel skew of 0 or 31.25 µs |
| E9 | Determinism | Seed every scene with `pra.random.seed(numpy=s, libroom=s)` inside the worker process, in addition to the numpy generator |

**Set sizes**
- At least 30 independent geometries per scenario in each of dev, wide and test:
  - bus: stratified by E1 orientation, E2 distance bin and E3 RT60;
  - meeting: rooms and table layouts with the E7 options.
- Bus SNR sweep (in-band at Mic A): −10 (stress), −5, 0, +5, +10 dB. Meeting HVAC: 35 / 40 / 45 dBA.
- dev: tune here. wide: decide here, at least 30 geometries per reported cell. test: the same design with new seeds, run once per milestone, never tuned on.
- **Disk:** a 32 s bus scene costs about 18 MB, and a 10 min meeting much more. Generate, score and discard on the fly; cache only RIRs and seeds.

### 4.2 Measurement method

- **References.** Store a separate early image (direct path plus 50 ms at Mic A) for partner A, partner B, the 3 nearest passengers, each conversing interferer and each meeting talker.
- **Components.** Store `comp2_*`, the 2-channel digital signal of every source group after the per-mic gains, plus `comp2_sensor`, so that Σ comp2 = mix to float32 precision. Without the sensor noise and gains, an s ± n re-synthesis changes the scene.
- **Linear, time-varying pipelines:** log the per-bin weights and gains and apply them to each component exactly ("master/slave" evaluation [8]).
- **Black boxes** (M8, M9, any adaptive or mask-based stack): Hagerman phase inversion [102] with the multi-signal extension.
  - Bus, 3 groups (partner A, partner B, everything else), 4 runs: y1 = F(A + B + N), y2 = F(A − B − N), y3 = F(−A + B − N), y4 = F(−A − B + N).
  - Decode: A_out = (y1 + y2 − y3 − y4)/4, B_out = (y1 − y2 + y3 − y4)/4, N_out = (y1 − y2 − y3 + y4)/4.
  - Use more groups (nearest passengers, conversing pair) on a 20% subset.
  - Meeting: 2 runs, all speech versus everything else. Take per-talker values from the speech component inside each talker's turns, excluding overlapped regions. Linear Tier-1 meeting stacks (M10, M2, logged M6 and M11 gains) use master/slave instead.
  - Consistency: flip the target instead of the noise and report the dB difference between the two s_out estimates as a nonlinearity indicator.
- **Polarity and phase.** There is no global sign flip. Each separated component is compared with its own reference:
  - SDR through an allowed-distortion FIR of 512 taps (BSS-Eval style [98]);
  - ESTOI on the separated component against the talker's own early image.
  A partner who comes out inverted (as from a figure-8), or with the beam's spectral tilt, is not penalized. A reference-domain check in v1.0 showed why this matters: an ideal output with partner B inverted scored dSISDR −2.6 dB and ESTOI 0.80.
- **Levels.** Talker k's level change before AGC is TA_k = L(output component k, last pre-AGC stage) − L(component k at Mic A), in dB; negative values are attenuation. Post-AGC level L_k is measured on the separated output component, never on the noisy output.
- **Delay and latency.**
  - Measure each method's delay by cross-correlating its separated target component with the reference over speech.
  - Report algorithmic latency separately: 32 ms for the 512-point STFT, 20 ms window and 10 ms hop for DPDFNet, about 24 ms plus the chunk length for VoxLock.
  - The scorer's ±100 ms alignment search is a safety net only.

### 4.3 Bus metrics ("only the two partners audible")

| ID | Metric | Definition | Desired direction |
|---|---|---|---|
| B1 | Per-partner fidelity | SDR_k (512-tap allowed distortion) and ESTOI_k of partner k's separated output against its own early image, over k's active segments. Report the worse partner (wrstT) | higher |
| B2 | Partner level and balance | TA_k before AGC; balance \|TA_A − TA_B\| (median and 90th percentile over 1 s windows) | TA_k ≥ −3 dB; balance small |
| B3 | Noise reduction during speech (replaces BR) | NA_active = attenuation of the separated noise component in partner-active frames; ΔSNR_active = NA_active + mean TA_k (an SNR improvement during speech, like G.160 App. II's SNRI). Report NA_idle in partner pauses separately (Teams "idle and active channel SpNR" [103]) and the noise modulation NM = NA_idle − NA_active | ΔSNR_active higher; NM ≤ 6 dB |
| B4 | ΔSIR per interferer group | Nearest passengers (< 1.5 m), conversing pairs, far babble, engine/road; separated components, partner-active frames. Report near-interferer leakage separately from distant babble | higher |
| B5 | Relative distractor attenuation (RDA) | Mean partner TA minus the nearest-interferer group's level change, per 1 s window; report mean and 10th percentile (the Teams test's "average and minimum" structure [103]) | higher |
| B6 | Passenger intelligibility | ESTOI of passenger j's separated output against j's early image, in regions where j talks alone, for the 3 nearest passengers and the conversing pairs | should fall strongly |
| B7 | Pickup versus distance ("bubble") | Probe scenes: partners plus one probe talker at r ∈ {0.3, 0.6, 1.0, 1.5, 2, 3} m, random azimuth, babble at +10 dB SNR; probe level change relative to the partners' (Hagerman probe flip) | large attenuation beyond 2 m |
| B8 | Per-partner WER | Small ONNX ASR in the `scoring_venv` (≤ 150 MB English model; log its name and SHA-256) on partner segments, against the corpus transcripts. Also WER on passenger-only segments | partners lower; passengers higher |
| B9 | Quality (report only) | DNSMOS P.835 SIG/BAK/OVRL [101] plus pDNSMOS; ≥ 9 s windows; aggregate at least 20 clips (per-clip correlation is only 0.71–0.83). Never tune on BAK or OVRL: BAK rewards silence | — |
| B10 | Cost | RTF, measured delay, algorithmic latency | within limits |

Legacy columns (reported, never used for decisions): summed-reference dSISDR, BR, PESQ-WB.

Why BR was dropped: an oracle-VAD gate on raw Mic A, which leaves the noise under speech untouched, scored BR +40 dB (the cap) with ESTOI unchanged at 0.381. With partner B inverted and noise at −10 dB, BR fell from +10.1 to +0.7 dB although nothing audible changed.

**Composite: two-talker focus rate.** A scene passes if all four hold:
1. worst-partner ESTOI_out ≥ worst-partner ESTOI_in;
2. the median ESTOI of the 3 nearest passengers in their solo regions is ≤ 0.25 (a provisional proxy for "not followable", to be calibrated by listening);
3. the 10th percentile of RDA is ≥ 10 dB;
4. both TA_k ≥ −3 dB.

Report the percentage of scenes that pass, per E1 stratum.

### 4.4 Meeting metrics ("all 20 captured evenly")

| ID | Metric | Definition |
|---|---|---|
| T1 | Capture rate | Fraction of talkers with ΔESTOI_k ≥ −0.03, WER_k no more than 5 points (absolute) worse than at Mic A, and \|L_k − median\| ≤ 3 dB (post-AGC, separated speech) |
| T2 | Pre-AGC attenuation by distance | TA_k per talker; median per distance bin (< 2 / 2–4 / > 4 m) and the worst talker. Requirement: far-bin median ≥ near-bin median − 3 dB |
| T3 | Level evenness | Std and range of L_k across talkers (post-AGC, separated speech). Quiet talkers' level relative to the median, and level stability for alternating talkers (Teams "signal level with quiet speech" and "speech level for alternating near-end talkers" [103]) |
| T4 | Noise during speech and idle | NA_active, NA_idle and NM as in B3. Noise-floor pumping: std and maximum step of the separated output noise level across turn changes |
| T5 | AGC settling | Time from turn onset until the 400 ms level is within ±3 dB of the talker's final level; with and without per-talker memory, on returning-talker scenes |
| T6 | Per-talker WER | As B8; by distance bin and worst talker |
| T7 | Per-talker ESTOI | Separated component against the talker's early image; by distance bin and worst talker |
| T8 | Output ELR50 (optional) | Pass the per-talker RIRs through frozen linear stages |
| T9 | Quality (report only) | DNSMOS SIG per talker (concatenate the talker's turns to ≥ 9 s); never tune on BAK or OVRL |

Legacy columns (reported, never used for decisions):
- gapSNR rewards gating.
- loudσ on the noisy output lets an AGC that re-boosts a suppressed far talker, together with its noise floor, look even.
- The talker median hides the far talkers.
- dSISDR is scale-invariant, so it cannot see far-talker attenuation before AGC.

**Composite: even-capture rate.** A scene passes if T1 = 100%, the worst-talker ESTOI_out ≥ the worst-talker ESTOI_in, and the T2 requirement holds.

### 4.5 Statistics and decision rules

- The unit of analysis is the geometry (scene): at least 30 per scenario per set.
- Compute metrics per scene, then paired differences against `m0` and against the current best stack.
- 95% confidence intervals by paired bootstrap (10,000 resamples over scenes, stratified by E1 stratum). Wilcoxon signed-rank tests, Holm-corrected across methods.
- An improvement gate "Δ ≥ X" passes when the point estimate is ≥ X **and** the CI excludes zero (lower bound > 0).
- A no-regression gate "Δ ≥ −tol" passes when the CI's lower bound is ≥ −tol.
- Always report breakdowns by orientation stratum, spacing, SNR, distance bin, RT60, stress case and speech type (TTS versus natural).

### 4.6 Pass/fail gates

These are provisional engineering targets, to be recalibrated with the listening test in §4.7. All are evaluated on E0 natural-speech scenes with the §4.5 rules.

| Gate | Bus (both partners kept, crowd suppressed) | Meeting (all 20, even loudness) |
|---|---|---|
| **G0: hard no-regression, always** | worst-partner ESTOI ≥ input − 0.01; both TA_k ≥ −3 dB; balance P90 ≤ 3 dB; NM ≤ 6 dB; no clipping or NaN; RTF < 0.5 (Tier 3: reported, "offline-only" allowed) | worst-talker ESTOI ≥ input − 0.01; ΔESTOI ≥ −0.03 for at least 19/20 talkers; T2 far-bin median TA ≥ near-bin median − 3 dB; NM ≤ 6 dB; RTF < 0.5 |
| **G1: classical stack (Tier 1)** | worst-partner ΔESTOI ≥ +0.03 overall and ≥ +0.05 in the aligned stratum; ΔSNR_active ≥ +3 dB (aligned) / +1.5 dB (all strata) | T3 std ≤ 1.5 dB and range ≤ 6 dB; mean ΔESTOI ≥ +0.02; ΔESTOI ≥ +0.03 in the > 4 m bin; WER not worse in any distance bin; NA_active ≥ 3 dB |
| **G2: final tool** | Aligned stratum, 0 dB SNR, partners ≤ 0.6 m: worst-partner ΔESTOI ≥ +0.08; RDA P10 ≥ 10 dB; passenger ESTOI drop ≥ 0.15 and passenger ESTOI_out ≤ 0.25; bubble ≥ 10 dB (2 m versus 0.6 m); partner WER not worse; DNSMOS SIG drop ≤ 0.2; ≥ 70% of clips judged "no followable bystander speech" in the P.835 test | T3 std ≤ 1.0 dB; capture rate 100%; noise pumping ≤ 2 dB; AGC settling ≤ 0.5 s (≤ 0.3 s for returning talkers); far-talker DNSMOS SIG drop ≤ 0.1; listeners judge loudness as even |

Expect G2 on the bus to be reachable only with Tier 3 (TSE) and correct device placement. Partners at 1 m, and the 45° and broadside strata, are reported but not gated for G2 (§1.3–1.5).

### 4.7 Validation against real recordings, and listening tests

1. **ESP32 field protocol (Week 1, not Week 4):** sim-to-real mismatch is the main risk [104].
   - (a) Calibration session (§5), saved to `calib.json`.
   - (b) Component recordings: play each source alone through a loudspeaker at each partner, passenger or seat position; record the noise alone; mix digitally. This gives ground-truth components with real acoustics, and the §4.2 metrics apply unchanged.
   - (c) Real bus rides and real meetings with no reference: score them with DNSMOS / pDNSMOS (report only) and by listening.
2. **ACE corpus** [95] (optional, about 1.1 GB): measured RIRs and noise for real meeting and lecture rooms. The Chromebook pair is 62 mm. Lecture Room 1 (T60 0.64 s) is the room closest to our range.
3. **Simulation can mislead for MVDR:** most methods rank the same on simulated and real data, but MVDR-type beamformers can look better in simulation [104]. Inject model errors (E6, E8) and confirm on real data.
4. **Listening test per ITU-T P.835** (the procedure behind the SIG / BAK / OVRL scales):
   - At least 6 listeners and 20 clips per condition. This is a lab-scale P.835 that follows its rating procedure.
   - Bus: also ask "How many voices can you understand?" (target: 2) and "Could you follow any bystander?"
   - Meeting: also rate loudness evenness and how understandable each talker is.
   - Use the results to recalibrate the 0.25 ESTOI, 10 dB RDA and NM thresholds.

### 4.8 Required ablations

- Each stage on and off.
- Wrong spacing assumption (nominal 40 mm, true 25 / 60 mm), with and without S0; S0 with and without the per-band phase correction and the free-ρ spacing fit.
- Mismatch, high-pass corner spread and skew on and off.
- Orientation strata (E1).
- Two-talker beam: continuation phase rule versus fixed ψ_AB = 180° (omni where infeasible) versus Mic-A-referenced h = [1, −1] (convention error).
- WNG floor sweep.
- CDR λ, µ and G_min sweeps at each spacing; conditioning mask on and off.
- Neural attenuation limit.
- TSE with oracle enrollment versus self-enrollment, including the wrong-speaker-lock rate.
- WPE variants (a), (b) and (c).
- AGC with speaker-embedding memory, TDOA-keyed memory, and no memory.
- Table mic height (E7) and table reflection on and off.
- Natural versus TTS speech, and synthetic sinc tail versus full image-source tail.

---

## 5. Hardware advice for the user

0. **Scope check.** This design assumes capture only. If the device will also play the far end of a call through a loudspeaker in the same room, add acoustic echo cancellation with the playback signal as reference (for example AEC3 in the WebRTC APM [29]) and test double talk before anything else. Teams certification tests echo, double-talk attenuation and AEC convergence [103].
1. **Calibrate once** (biggest return per hour of effort).
   - Tap Mic A to confirm it is row 0.
   - Measure the port spacing with calipers, port centre to port centre.
   - Play a chirp (for example 100 Hz–7 kHz, 1 s, from a phone speaker) 1 m away on the axis beyond Mic A, then beyond Mic B, 10 times each. Take the median GCC-PHAT TDOA with sub-sample interpolation, giving τ₊ and τ₋:
     - d = c·(τ₊ − τ₋)/2 (about ±117 µs each at 4 cm);
     - I2S skew δ = τ0 = (τ₊ + τ₋)/2.
     - A half-frame skew (31.25 µs) would be obvious, and would explain the conflicting 2.3 cm and 4 cm spacing estimates.
     - Accept if the acoustic d and the caliper d agree within 2 mm; otherwise use the acoustic d.
   - Check with a chirp at broadside 1 m away: the measured TDOA should equal δ.
   - Record 30 s of a diffuse field (HVAC, or noise played in a reverberant room) and 30 s of silence (self-noise) to fit the per-band gain and phase correction. Target ±0.25 dB / ±1°.
   - Why calibrate when mismatch costs ≤ 0.1 dB at a −10 dB WNG floor? Because the bus beams use −15 to −20 dB floors below 500 Hz, and because the CDR estimator and the TDOA tracker read inter-mic phase directly.
   - Save everything to `calib.json`; the tool loads it at start-up.
2. **Placement for the bus.**
   - Point the mic axis through both partners' mouths, with the device between them so both are at endfire. Keep the axis horizontal.
   - The orientation matters more than any algorithm choice. The two-talker beam gives 2.3 dB in horizontal babble when aligned and about 0 dB when both partners are near broadside.
   - Do not lay the device flat between two people who lean over it: both TDOAs are then about 0. The tool detects this ("inseparable"), falls back to non-spatial processing and asks you to rotate the device.
   - Keep the device within 0.5–0.6 m of both mouths. That keeps them inside r_c, so the coherence cue works.
   - Use a soft, hands-free mount (low WNG amplifies handling noise) and keep away from windows and vents (wind).
3. **Placement for meetings.**
   - Table centre, with the mic axis along the long axis of the table:
     - the farthest talkers, at the table ends, are then at endfire;
     - |τ| then tracks talker distance (Spearman 0.98, against 0.06 for a random orientation), which M11 uses for a distance-compensating starting gain.
   - Mount the mic ports within about 2 cm of the tabletop. The table reflection causes comb notches: at 10 cm or 20 cm height, a talker 3 m away gets a notch near 6.2 kHz or 3.1 kHz (1.5 m away: 3.2 or 1.6 kHz). Within 2 cm the first notch is above 15 kHz.
   - Keep away from laptop fans and the projector.
4. **Spacing.**
   - Keep about 4 cm; spacing alone changes beam gain by ≤ 0.5 dB. Measure it precisely rather than rebuilding.
   - For a meeting-oriented second board, 6 cm gives: better CDR conditioning (coherence 0.9 reached at 0.72 vs 1.07 kHz), +3.5 dB of low-frequency WNG and finer DOA. The cost is aliasing at 2.86 kHz. 8 cm is best for CDR but aliases at 2.14 kHz.
   - Mount both mics identically: same port length and diameter, same gasket. Espressif guidance [17]: port > 1 mm, depth-to-diameter < 2:1, resonance above 9 kHz, sensitivity difference ≤ 3 dB, phase difference ≤ 10°. Software calibration must then bring this down to ±0.25 dB / ±1°.
5. **More mics (largest upgrade).**
   - **A third mic** (triangle with 4 cm sides) gives two degrees of freedom per bin: keep both partners and null one interferer. It also gives 2-D DOA, which removes the broadside cone ambiguity.
     - The INMP441 offers only L/R slots, so at most 2 mics per I2S data line.
     - Options:
       - a second data line or I2S controller (check the ESP32 variant);
       - TDM: if the ESP32-C3's I2S peripheral supports TDM in ESP-IDF (confirm in the ESP-IDF I2S documentation), TDM-capable MEMS mics give a 3–4-mic array on one data line. For the meeting this removes the front–back ambiguity of a pair;
       - a second synchronized node.
   - **Meetings:** use 2–4 two-mic nodes on the table centreline so every seat is within about 2.4–3 m of a node. Microsoft's guidance for certified rooms is ≤ 3 m per mic even in 12–16-person rooms [32].
     - Mix the nodes with Dugan gain-sharing [28] after coarse alignment and sampling-rate-offset compensation, or select the best node.
     - Evidence: 3 / 5 / 7 distributed devices gave 14.8 / 20.3 / 22.4% relative improvement in speaker-attributed WER [65]; 7 versus 2 channels gave 50.4% / 21.8% relative WER reduction [54].
     - Moving a talker from 6 m to 2.5 m from a node gains about 7.6 dB of DRR, more than all two-mic DSP combined.
   - **Bus:** if a wearable is acceptable, put a mic within a few cm of each partner (clip-on, short leads). That gives a level difference of 10 dB or more, enabling phone-style level gating [24]. Otherwise keep the device within 0.5 m of both partners.
6. **Mic grade.** The INMP441's 61 dBA SNR is below Espressif's ≥ 62 dB minimum (≥ 64 dB recommended) [17]. That is fine on a bus. In a quiet meeting room, a higher-SNR MEMS (≥ 65 dBA) would allow lower WNG floors and therefore more low-frequency directivity.
7. **Data path.**
   - Keep 24-bit samples and convert to float32.
   - Process both channels identically on the device: no per-channel AGC or HPF.
   - Add a frame counter to detect dropped frames.
   - Verify the L/R slot mapping and the bit alignment of 24-in-32-bit samples.

---

## 6. References (verified sources only)

**Tags**
- **[R]** re-verified independently in the research phase (corrected citations used where errors were found).
- **[S]** fetched and checked by the scout; not independently re-checked.
- **[D]** added during verification and checked through the DOI registry, arXiv or the publisher.
- **[Rv]** added or corrected in the v1.1 review; checked against the publisher or registry record fetched by the reviewer, or against a copy saved in the research scratchpad.

Sources the verification phase flagged as unverified, or cited from memory, are excluded. Standards and products named in the text but not fetched are listed at the end of this section.

**Arrays, beamforming and room acoustics**
1. H. Cox, R. M. Zeskind, T. Kooij, "Practical supergain," IEEE Trans. ASSP 34(3):393–398, 1986. https://doi.org/10.1109/TASSP.1986.1164847 [R]
2. H. Cox, R. M. Zeskind, M. M. Owen, "Robust adaptive beamforming," IEEE Trans. ASSP 35(10):1365–1376, 1987. https://doi.org/10.1109/TASSP.1987.1165054 [R]
3. J. Benesty, J. Chen, *Study and Design of Differential Microphone Arrays*, Springer Topics in Signal Processing vol. 6, 2013. https://doi.org/10.1007/978-3-642-33753-6 [R]
4. S. Doclo, M. Moonen, "Superdirective beamforming robust against microphone mismatch," IEEE Trans. ASLP 15(2):617–631, 2007. https://doi.org/10.1109/TASL.2006.881676 [R]
5. J. Cezanne, G. W. Elko, "Adaptive microphone array," US Patent 5,473,701, 1995. https://patents.google.com/patent/US5473701A/en [S] (A companion WASPAA 1995 paper by Elko and Nguyen Pong is named by title only; its DOI is not verified. DOI 10.1109/ASPAA.1995.482964 belongs to a different paper, Kulkarni et al. on HRTFs, and is not used here.)
6. L. J. Griffiths, C. W. Jim, "An alternative approach to linearly constrained adaptive beamforming," IEEE Trans. Antennas Propag. AP-30(1):27–34, 1982. https://doi.org/10.1109/TAP.1982.1142739 [S]
7. J. Bitzer, K. U. Simmer, K.-D. Kammeyer, "Theoretical noise reduction limits of the generalized sidelobe canceller (GSC) for speech enhancement," Proc. ICASSP 1999, vol. 5, pp. 2965–2968. https://www.ant.uni-bremen.de/en/publications/2095/ [R]
8. J. Bitzer, K. U. Simmer, K.-D. Kammeyer, "Multi-microphone noise reduction by post-filter and superdirective beamformer," Proc. IWAENC 1999, pp. 100–103. https://www.ant.uni-bremen.de/sixcms/media.php/102/4874/IWAENC_1999_bitzer.pdf [R]
9. S. Gannot, D. Burshtein, E. Weinstein, "Signal enhancement using beamforming and nonstationarity with applications to speech," IEEE Trans. SP 49(8):1614–1626, 2001. https://doi.org/10.1109/78.934132 [R]
10. S. Markovich-Golan, S. Gannot, W. Kellermann, "Performance analysis of the covariance-whitening and the covariance-subtraction methods for estimating the relative transfer function," Proc. EUSIPCO 2018, pp. 2513–2517. https://new.eurasip.org/Proceedings/Eusipco/Eusipco2018/papers/1570437654.pdf [R]
11. M. Souden, J. Benesty, S. Affes, "On optimal frequency-domain multichannel linear filtering for noise reduction," IEEE Trans. ASLP 18(2):260–276, 2010. https://doi.org/10.1109/TASL.2009.2025790 [R]
12. I. Tashev, "Gain self-calibration procedure for microphone arrays," Proc. IEEE ICME 2004, pp. 983–986. https://www.microsoft.com/en-us/research/wp-content/uploads/2004/06/2004-ivantash-icme.pdf [R]
13. C. H. Knapp, G. C. Carter, "The generalized correlation method for estimation of time delay," IEEE Trans. ASSP 24(4):320–327, 1976. https://doi.org/10.1109/TASSP.1976.1162830 [S]
14. C. Blandin, A. Ozerov, E. Vincent, "Multi-source TDOA estimation in reverberant audio using angular spectra and clustering," Signal Processing 92(8):1950–1960, 2012. https://doi.org/10.1016/j.sigpro.2011.09.032 [D]
15. H. Kuttruff, *Room Acoustics*, 6th ed., CRC Press, 2016 (ISBN 978-1-4822-6043-4). https://www.taylorfrancis.com/books/9781315372150 [S]
16. M. Barron, L.-J. Lee, "Energy relations in concert auditoriums. I," JASA 84(2):618–628, 1988. https://doi.org/10.1121/1.396840 [D]

**Industry: hardware, hearing aids, phones, conferencing**

17. Espressif Systems, "Espressif Microphone Design Guidelines" (ESP-SR audio front end, ESP32-S3), accessed Nov 2026. https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/audio_front_end/Espressif_Microphone_Design_Guidelines.html [R]
18. D. A. Fabry, "Adaptive directional microphone technology and hearing aids: theoretical and clinical implications," The Hearing Review, Apr. 2005. https://hearingreview.com/2005/04/adaptive-directional-microphone-technology-and-hearing-aids-theoretical-and-clinical-implications [R]
19. T. A. Ricketts, "Directional hearing aids," Trends in Amplification 5(4):139–176, 2001. https://doi.org/10.1177/108471380100500401 [D]
20. M. T. Cord, R. K. Surr, B. E. Walden, O. Dyrlund, "Relationship between laboratory measures of directional advantage and everyday success with directional microphone hearing aids," J. Am. Acad. Audiol. 15(5):353–364, 2004. https://doi.org/10.3766/jaaa.15.5.3 [D]
21. N. S. Jensen, E. H. Høydal, E. Branda, J. Weber, "Augmenting speech recognition with a new split-processing paradigm," Hearing Review 28(6):24–27, 2021. https://hearingreview.com/hearing-products/hearing-aids/speech-in-noise/speech-recognition [R]
22. Oticon Medical, "Ponto 4" course slides (OpenSound Navigator: omni plus back-cardioid noise estimate, 16 bands), AudiologyOnline, 2019. https://www.audiologyonline.com/files/event/33200/33217/cont-om-ponto4-060619.pdf [S]
23. H. Hasemann, A. Krylova, "Revolutionary speech understanding with Spheric Speech Clarity," Phonak Insight, Sonova, June 2024. https://phonak.com/content/dam/phonak/fr/evidence-library/technical-paper/PH_Insight_SphericSpeechClarity_210x297_EN_028-2684-02_V1.00.pdf [R]
24. M. Jeub, C. Herglotz, C. M. Nelke, C. Beaugeant, P. Vary, "Noise reduction for dual-microphone mobile phones exploiting power level differences," Proc. ICASSP 2012, pp. 1693–1696. https://doi.org/10.1109/ICASSP.2012.6288223 [R]
25. C. Avendano, L. Watts, P. Santos (Audience Inc.), "System and method for utilizing inter-microphone level differences for speech enhancement," US 8,867,759 B2, 2014; related US 8,194,880 B2 and US 8,718,290 B2. https://patents.google.com/patent/US8867759 [S]
26. E. Visser, E. Liu (Qualcomm), "Systems, methods, apparatus, and computer-readable media for phase-based processing of multichannel signal," US 8,620,672 B2, 2013. https://patents.google.com/patent/US8620672 [S]
27. V. Iyengar, S. V. Dusan (Apple), "Audio noise estimation and audio noise reduction using multiple microphones," US 9,966,067 B2, 2018; related US 10,176,823 B2. https://patents.google.com/patent/US9966067 [S]
28. D. W. Dugan, "Automatic microphone mixer," US Patent 3,992,584, 1976. https://patents.google.com/patent/US3992584A/en [S]
29. WebRTC project, Audio Processing Module API (AGC2 adaptive digital gain, noise suppression, HPF), main branch, accessed 2026. https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/audio_processing.h [S]
30. Nureva Inc., "A groundbreaking approach to gain control that cuts conference call annoyance," technical blog (c. 2020). https://www.nureva.com/learn/posts/a-groundbreaking-approach-to-gain-control-that-cuts-conference-call-annoyance [S]
31. HP Inc. / Poly, "Audio innovations: NoiseBlockAI and Acoustic Fence," brochure c09038815, July 2026. https://h20195.www2.hp.com/v2/GetPDF.aspx/c09038815 [S]
32. Microsoft, "Teams Rooms certified systems and peripherals" (microphone-to-talker distance guidance), Microsoft Learn, 2026. https://learn.microsoft.com/en-us/microsoftteams/rooms/certified-hardware [S]
33. EBU R 128 v5.0, "Loudness normalisation and permitted maximum level of audio signals" (measured per ITU-R BS.1770), 2023. https://tech.ebu.ch/publications/r128 [S]

**Coherence and phase post-filters**

34. A. Schwarz, W. Kellermann, "Coherent-to-diffuse power ratio estimation for dereverberation," IEEE/ACM TASLP 23(6):1006–1018, 2015. https://arxiv.org/abs/1502.03784 (code: https://github.com/andreas12345/cdr-dereverb) [R]
35. H. Barfuss, C. Huemmer, A. Schwarz, W. Kellermann, "Robust coherence-based spectral enhancement for speech recognition in adverse real-world environments," Computer Speech & Language 46:388–400, 2017. https://arxiv.org/abs/1604.03393 [R]
36. I. A. McCowan, H. Bourlard, "Microphone array post-filter based on noise field coherence," IEEE Trans. SAP 11(6):709–716, 2003. https://doi.org/10.1109/TSA.2003.818212 [R]
37. O. Thiergart, G. Del Galdo, E. A. P. Habets, "On the spatial coherence in mixed sound fields and its application to signal-to-diffuse ratio estimation," JASA 132(4):2337–2346, 2012. https://doi.org/10.1121/1.4750493 [D]
38. J. B. Allen, D. A. Berkley, J. Blauert, "Multimicrophone signal-processing technique to remove room reverberation from speech signals," JASA 62(4):912–915, 1977. https://doi.org/10.1121/1.381621 [R]
39. N. Yousefian, P. C. Loizou, "A dual-microphone speech enhancement algorithm based on the coherence function," IEEE Trans. ASLP 20(2):599–609, 2012; volume and pages from the PMC record, while Crossref holds only the 2011 early-access record, doi:10.1109/TASL.2011.2162406. https://pmc.ncbi.nlm.nih.gov/articles/PMC3246289 [R]
40. P. Aarabi, G. Shi, "Phase-based dual-microphone robust speech enhancement," IEEE Trans. SMC-B 34(4):1763–1773, 2004. https://doi.org/10.1109/TSMCB.2004.830345 [R]

**Single-channel noise suppression**

41. R. Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics," IEEE Trans. SAP 9(5):504–512, 2001. https://doi.org/10.1109/89.928915 [D]
42. I. Cohen, "Noise spectrum estimation in adverse environments: Improved minima controlled recursive averaging," IEEE Trans. SAP 11(5):466–475, 2003. https://doi.org/10.1109/TSA.2003.811544 [D]
43. T. Gerkmann, R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," IEEE Trans. ASLP 20(4):1383–1393, 2012. https://doi.org/10.1109/TASL.2011.2180896 [D]
44. Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE Trans. ASSP 33(2):443–445, 1985. https://doi.org/10.1109/TASSP.1985.1164550 [D]
45. Y. Hu, P. C. Loizou, "A comparative intelligibility study of single-microphone noise reduction algorithms," JASA 122(3):1777–1786, 2007. https://doi.org/10.1121/1.2766778 [D]

**Spatial clustering and blind source separation**

46. Ö. Yılmaz, S. Rickard, "Blind separation of speech mixtures via time-frequency masking," IEEE Trans. SP 52(7):1830–1847, 2004. https://personal.math.ubc.ca/~oyilmaz/preprints/wdo_final.pdf [S]
47. H. Sawada, S. Araki, S. Makino, "Underdetermined convolutive blind source separation via frequency bin-wise clustering and permutation alignment," IEEE Trans. ASLP 19(3):516–527, 2011. https://www.kecl.ntt.co.jp/icl/signal/sawada/mypaper/IEEEtaslp2011sawada.pdf [R]
48. M. I. Mandel, R. J. Weiss, D. P. W. Ellis, "Model-based expectation-maximization source separation and localization," IEEE Trans. ASLP 18(2):382–394, 2010. https://www.ee.columbia.edu/~dpwe/pubs/MandWE09-messl.pdf [R]
49. N. Ito, S. Araki, T. Nakatani, "Complex angular central Gaussian mixture model for directional statistics in mask-based microphone array signal processing," Proc. EUSIPCO 2016, pp. 1153–1157. https://new.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256519.pdf [R]
50. H. Erdogan, J. R. Hershey, S. Watanabe, M. I. Mandel, J. Le Roux, "Improved MVDR beamforming using single-channel mask prediction networks," Proc. Interspeech 2016, pp. 1981–1985. https://www.isca-archive.org/interspeech_2016/erdogan16_interspeech.pdf [R]
51. J. Heymann, L. Drude, R. Haeb-Umbach, "Neural network based spectral mask estimation for acoustic beamforming," Proc. ICASSP 2016, pp. 196–200. https://doi.org/10.1109/ICASSP.2016.7471664 [R]
52. T. Higuchi, N. Ito, S. Araki, T. Yoshioka, M. Delcroix, T. Nakatani, "Online MVDR beamformer based on complex Gaussian mixture model with spatial prior for noise robust ASR," IEEE/ACM TASLP 25(4):780–793, 2017. https://doi.org/10.1109/TASLP.2017.2665341 [R]
53. C. Boeddeker, J. Heitkaemper, J. Schmalenstroeer, L. Drude, J. Heymann, R. Haeb-Umbach, "Front-end processing for the CHiME-5 dinner party scenario," Proc. CHiME-5 Workshop 2018, pp. 35–40. https://www.isca-archive.org/chime_2018/boeddecker18_chime.html [S]
54. D. Raj, D. Povey, S. Khudanpur, "GPU-accelerated guided source separation for meeting transcription," Proc. Interspeech 2023. https://arxiv.org/abs/2212.05271 [R]
55. N. Ito, S. Araki, T. Nakatani, "Data-driven and physical model-based designs of probabilistic spatial dictionary for online meeting diarization and adaptive beamforming," Proc. EUSIPCO 2017. https://new.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570346869.pdf [S]
56. N. Ono, "Stable and fast update rules for independent vector analysis based on auxiliary function technique," Proc. IEEE WASPAA 2011, pp. 189–192. https://doi.org/10.1109/ASPAA.2011.6082320 [R]
57. M. Sunohara, C. Haruta, N. Ono, "Low-latency real-time blind source separation for hearing aids based on time-domain implementation of online independent vector analysis with truncation of non-causal components," Proc. ICASSP 2017, pp. 216–220. https://doi.org/10.1109/ICASSP.2017.7952149 [R]
58. K. Sekiguchi, A. A. Nugraha, Y. Bando, K. Yoshii, "Fast multichannel source separation based on jointly diagonalizable spatial covariance matrices," Proc. EUSIPCO 2019. https://arxiv.org/abs/1903.03237 [S]

**Dereverberation and meetings**

59. T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi, B.-H. Juang, "Speech dereverberation based on variance-normalized delayed linear prediction," IEEE Trans. ASLP 18(7):1717–1731, 2010. https://doi.org/10.1109/TASL.2010.2052251 [R]
60. T. Yoshioka, T. Nakatani, "Generalization of multi-channel linear prediction methods for blind MIMO impulse response shortening," IEEE Trans. ASLP 20(10):2707–2720, 2012. https://doi.org/10.1109/TASL.2012.2210879 [R]
61. L. Drude, J. Heymann, C. Boeddeker, R. Haeb-Umbach, "NARA-WPE: A Python package for weighted prediction error dereverberation in Numpy and Tensorflow for online and offline processing," ITG Speech Communication 2018. https://github.com/fgnt/nara_wpe [R]
62. J. Caroselli, I. Shafran, A. Narayanan, R. Rose, "Adaptive multichannel dereverberation for automatic speech recognition," Proc. Interspeech 2017. https://www.isca-archive.org/interspeech_2017/caroselli17_interspeech.html [S]
63. K. Kinoshita et al., "A summary of the REVERB challenge: state-of-the-art and remaining challenges in reverberant speech processing research," EURASIP J. Adv. Signal Process. 2016:7. https://link.springer.com/article/10.1186/s13634-016-0306-6 [S]
64. T. Nakatani, K. Kinoshita, "A unified convolutional beamformer for simultaneous denoising and dereverberation," IEEE Signal Process. Lett. 26(6):903–907, 2019. https://doi.org/10.1109/LSP.2019.2911179 [D]
65. T. Yoshioka, Z. Chen, D. Dimitriadis, W. Hinthorn, X. Huang, A. Stolcke, M. Zeng, "Meeting transcription using virtual microphone arrays," arXiv:1905.02545, 2019. https://arxiv.org/abs/1905.02545 [S]
66. Z. Chen, T. Yoshioka, L. Lu, T. Zhou, Z. Meng, Y. Luo, J. Wu, X. Xiao, J. Li, "Continuous speech separation: dataset and analysis," Proc. ICASSP 2020, pp. 7284–7288. https://doi.org/10.1109/ICASSP40776.2020.9053426 [R]
67. S. Watanabe et al., "CHiME-6 Challenge: Tackling multispeaker speech recognition for unsegmented recordings," Proc. CHiME 2020 Workshop, pp. 1–7. https://arxiv.org/abs/2004.09249 [S]
68. A. Plaquet, H. Bredin, "Powerset multi-class cross entropy loss for neural speaker diarization," Proc. Interspeech 2023. https://arxiv.org/abs/2310.13025 [D]

**Neural enhancement and target-speaker extraction**

69. J.-M. Valin, "A hybrid DSP/deep learning approach to real-time full-band speech enhancement," Proc. IEEE MMSP 2018. https://arxiv.org/abs/1709.08243 [R]
70. J.-M. Valin, U. Isik, N. Phansalkar, R. Giri, K. Helwani, A. Krishnaswamy, "A perceptually-motivated approach for low-complexity, real-time enhancement of fullband speech," Proc. Interspeech 2020. https://arxiv.org/abs/2008.04259 [R]
71. H. Schröter, A. N. Escalante-B., T. Rosenkranz, A. Maier, "DeepFilterNet2: Towards real-time speech enhancement on embedded devices for full-band audio," Proc. IWAENC 2022 (DeepFilterNet3: Interspeech 2023, arXiv:2305.08227). https://arxiv.org/abs/2205.05474 [R]
72. S. Braun, I. Tashev, "Data augmentation and loss normalization for deep noise suppression" (NSNet2), Proc. SPECOM 2020. https://arxiv.org/abs/2008.06412 [R]
73. X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu, J. Lu, "GTCRN: A speech enhancement model requiring ultralow computational resources," Proc. ICASSP 2024, pp. 971–975. https://doi.org/10.1109/ICASSP48485.2024.10448310 [R]
74. D. Rika, N. Sapir, I. Gus, "DPDFNet: Boosting DeepFilterNet2 via dual-path RNN," arXiv:2512.16420 (accepted, Speech Communication). https://arxiv.org/abs/2512.16420 [R]
75. Z. Wang, X. Rong, Y. Sun, T. Sun, Z. Lin, J. Lu, "A lightweight hybrid dual channel speech enhancement system under low-SNR conditions" (H-GTCRN), Proc. Interspeech 2025. https://arxiv.org/abs/2505.19597 [R]
76. M. Delcroix, T. Ochiai, K. Zmolikova, K. Kinoshita, N. Tawara, T. Nakatani, S. Araki, "Improving speaker discrimination of target speech extraction with time-domain SpeakerBeam," Proc. ICASSP 2020, pp. 691–695. https://arxiv.org/abs/2001.08378 [R]
77. Q. Wang et al., "VoiceFilter-Lite: Streaming targeted voice separation for on-device speech recognition," Proc. Interspeech 2020, pp. 2677–2681. https://arxiv.org/abs/2009.04323 [R]
78. R. Giri, S. Venkataramani, J.-M. Valin, U. Isik, A. Krishnaswamy, "Personalized PercepNet: Real-time, low-complexity target voice separation and enhancement," Proc. Interspeech 2021. https://arxiv.org/abs/2106.04129 [R]
79. A. Siddiqui, VoxLock: causal target-speaker voice isolation (ONNX, Apache-2.0), 2026; integrated in sherpa-onnx v2.0.1. https://huggingface.co/arbazsiddiqui/voxlock [R]
80. S. Wang et al., "SLT 2026 REAL-TSE Challenge: Real-world target speaker extraction from conversational recordings," arXiv:2607.15198, 2026. https://arxiv.org/abs/2607.15198 [R]
81. H. Dubey et al., "ICASSP 2023 Deep Noise Suppression Challenge," arXiv:2303.11510, 2023. https://arxiv.org/abs/2303.11510 [R]
82. I. Thoidis, T. Goehring, "Using deep learning to improve the intelligibility of a target speaker in noisy multi-talker environments for people with normal hearing and hearing loss," JASA 156(1):706–724, 2024. https://pubmed.ncbi.nlm.nih.gov/39082692/ [R]
83. S. E. Eskimez, T. Yoshioka, H. Wang, X. Wang, Z. Chen, X. Huang, "Personalized speech enhancement: New models and comprehensive evaluation," Proc. ICASSP 2022. https://arxiv.org/abs/2110.09625 [D]
84. T. Chen, Q. Wang, B. Wu, M. Itani, S. E. Eskimez, T. Yoshioka, S. Gollakota, "Target conversation extraction: Source separation using turn-taking dynamics," Proc. Interspeech 2024, pp. 3550–3554. https://arxiv.org/abs/2407.11277 [D]
85. T. Chen, M. Itani, S. E. Eskimez, T. Yoshioka, S. Gollakota, "Hearable devices with sound bubbles," Nature Electronics 7:1047–1058, 2024. https://doi.org/10.1038/s41928-024-01276-z [D]
86. K. Patterson, K. Wilson, S. Wisdom, J. R. Hershey, "Distance-based sound separation," Proc. Interspeech 2022. https://arxiv.org/abs/2207.00562 [D]
87. K. Iwamoto, T. Ochiai, M. Delcroix, R. Ikeshita, H. Sato, S. Araki, S. Katagiri, "How bad are artifacts?: Analyzing the impact of speech enhancement errors on ASR," Proc. Interspeech 2022. https://arxiv.org/abs/2201.06685 [D]
88. B. Veluri, M. Itani, T. Chen, T. Yoshioka, S. Gollakota, "Look once to hear: Target speech hearing with noisy examples," Proc. ACM CHI 2024. https://arxiv.org/abs/2405.06289 [D]

**Evaluation**

89. J. B. Allen, D. A. Berkley, "Image method for efficiently simulating small-room acoustics," JASA 65(4):943–950, 1979. https://doi.org/10.1121/1.382599 [R]
90. R. Scheibler, E. Bezzam, I. Dokmanić, "Pyroomacoustics: A Python package for audio room simulation and array processing algorithms," Proc. ICASSP 2018, pp. 351–355. https://arxiv.org/abs/1710.04196 [R]
91. E. De Sena, N. Antonello, M. Moonen, T. van Waterschoot, "On the modeling of rectangular geometries in room acoustic simulations," IEEE/ACM TASLP 23(4):774–786, 2015. https://doi.org/10.1109/TASLP.2015.2405476 [D]
92. E. A. P. Habets, S. Gannot, "Generating sensor signals in isotropic noise fields," JASA 122(6):3464–3470, 2007. https://doi.org/10.1121/1.2799929 [R]
93. AudioLabs Erlangen, anf-generator (PyPI 0.1.3, MIT). Methods: E. A. P. Habets, I. Cohen, S. Gannot, JASA 124(5):2911–2917, 2008; D. Mirabilii, S. J. Schlecht, E. A. P. Habets, JASA 149(3):1425–1433, 2021. https://github.com/audiolabs/anf-generator [R]
94. J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)," Proc. Meetings on Acoustics 19:035081, 2013; dataset on Zenodo. https://zenodo.org/records/1227121 [S]
95. J. Eaton, N. D. Gaubitch, A. H. Moore, P. A. Naylor, "Estimation of room acoustic parameters: The ACE Challenge," IEEE/ACM TASLP 24(10):1681–1693, 2016; ACE corpus. https://zenodo.org/records/6257551 [S]
96. R. Neitzel, R. R. M. Gershon, M. Zeltser, A. Canton, M. Akram, "Noise levels associated with New York City's mass transit systems," Am. J. Public Health 99(8):1393–1399, 2009. https://pmc.ncbi.nlm.nih.gov/articles/PMC2707461 [R]
97. Y.-H. Wu, E. Stangl, O. Chipara, S. S. Hasan, A. Welhaven, J. Oleson, "Characteristics of real-world signal to noise ratios and speech listening situations of older adults with mild to moderate hearing loss," Ear and Hearing 39(2):293–304, 2018. https://doi.org/10.1097/AUD.0000000000000486 [R]
98. J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR – half-baked or well done?," Proc. ICASSP 2019, pp. 626–630. https://arxiv.org/abs/1811.02508 [S]
99. C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, "An algorithm for intelligibility prediction of time–frequency weighted noisy speech," IEEE Trans. ASLP 19(7):2125–2136, 2011; and J. Jensen, C. H. Taal, "An algorithm for predicting the intelligibility of speech masked by modulated noise maskers," IEEE/ACM TASLP 24(11):2009–2022, 2016 (ESTOI), https://doi.org/10.1109/TASLP.2016.2585878 [Rv]. The v1.0 link (Taal2011_1.pdf) points to the 2011 STOI paper and could not be fetched in review.
100. ITU-T Rec. P.862.2, "Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs" (PESQ-WB), 2005/2007. The P.862 family was deleted in Jan 2024. https://www.itu.int/rec/T-REC-P.862.2 [S]
101. C. K. A. Reddy, V. Gopal, R. Cutler, "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors," Proc. ICASSP 2022, pp. 886–890. https://arxiv.org/abs/2110.01763 [R]
102. B. Hagerman, Å. Olofsson, "A method to measure the effect of noise reduction algorithms using simultaneous speech and noise," Acta Acustica united with Acustica 90(2):356–361, 2004. Multi-signal extension: H. Husstedt, A. Mertins, M. Frenz, "Evaluation of noise reduction algorithms in hearing aids for multiple signals from equal or different directions," Trends in Hearing 22, 2018, doi:10.1177/2331216518803198 [Rv]. https://www.ingentaconnect.com/contentone/dav/aaua/2004/00000090/00000002/art00016 [S]
103. Microsoft Teams Audio Test Specification V5.0 incl. Addendum A, as implemented in the HEAD acoustics ACQUA "MS Teams" test suite, data sheet 60026 rev. 10, 2026. https://cdn.head-acoustics.com/fileadmin/data/global/Datasheets/Analysis_Software/ACQUA_Standards/HEAD-acoustics-Data-Sheet-ACQUA-Standards-MS-Teams-Measurements-and-Analyses-According-to-Microsoft-Teams-Audio-Test-Specification-60026-EN.pdf [S]
104. E. Vincent, S. Watanabe, A. A. Nugraha, J. Barker, R. Marxer, "An analysis of environment, microphone and data simulation mismatches in robust speech recognition," Computer Speech & Language 46:535–557, 2017. https://eprints.whiterose.ac.uk/id/eprint/111196 [R]
105. J. Barker, R. Marxer, E. Vincent, S. Watanabe, "The third 'CHiME' speech separation and recognition challenge: Dataset, task and baselines," Proc. IEEE ASRU 2015, pp. 504–511. https://doi.org/10.1109/ASRU.2015.7404837 [D]
106. A. Vinnikov et al., "NOTSOFAR-1 Challenge: New datasets, baseline, and tasks for distant meeting transcription," Proc. Interspeech 2024, pp. 5003–5007, doi:10.21437/Interspeech.2024-1788. https://arxiv.org/abs/2401.08887 [D][Rv]
107. J. Carletta et al., "The AMI Meeting Corpus: A pre-announcement," Machine Learning for Multimodal Interaction (MLMI 2005), LNCS 3869, pp. 28–39, Springer, 2006. https://doi.org/10.1007/11677482_3 [Rv]. Check the corpus licence and download size before use.
108. ETSI TS 103 224 V1.7.2 (2026-07), "Speech and multimedia Transmission Quality (STQ); A sound field reproduction method for terminal testing including a background noise database" ("Inside Bus" and other 8-channel recordings, 30 s each). Find it through the ETSI standards search (https://www.etsi.org/standards, "103 224"). The title and version were checked against the copy saved in the research scratchpad (`pdfs/ts103224.txt`) [Rv].

**Named in the text but not fetched in this revision** (check before relying on them):
- ITU-T Rec. P.835, the subjective test method for noise-suppression systems (SIG / BAK / OVRL), used for the listening test.
- ITU-T Rec. G.160, Appendix II (TNLR, NPLR, SNRI, DSN), which B3 and T4 follow in spirit.
- The ESP-IDF I2S documentation, for TDM support on the ESP32-C3 (§5).
- LibriSpeech (test-clean 346 MB, CC BY 4.0, as noted in the research phase).
- The sherpa-onnx speaker-embedding and ASR model exports (S1, B8).

---

## Appendix A: Implementation order (4 weeks)

**Week 1: measurement first**
- **Harness** (§4.2):
  - `comp2_*` including sensor noise and per-mic gains;
  - separate early images per talker;
  - per-talker scoring with a 512-tap allowed-distortion SDR and no global sign flip;
  - 4-run Hagerman decoding and the consistency figure;
  - metrics B1–B10 and T1–T9, with the ONNX ASR scorer in the `scoring_venv`.
- **Scenes** (§4.1):
  - E0 natural speech (LibriSpeech first; conversational audio once its licence and size are checked);
  - E1–E4 bus strata and interferers; E8 impairments; E9 seeding;
  - at least 30 stratified geometries per scenario for dev, wide and test.
- **Hardware:** ESP32 calibration session (calipers, endfire chirps, diffuse and silence recordings) and the first component recordings (§4.7).
- S0 calibration and S1 tracker (including speaker embeddings), plus the unit tests in Appendix B.
- Methods M0, M1 and M2. Re-baseline M0 with the new metrics.

**Week 2: Tier-1 stacks**
- Methods M3, M4 (with the conditioning mask and the 2–6 cm retune), M5, M6 and M11.
- Bus Tier-1 stack.
- Meeting stack: M10 variants (a) and (b), M2-meeting, M6 and M11 with speaker-keyed memory.
- Scene factors E5–E7 (directivity, real noise, table and 6 m meetings).
- First G1 evaluation on the wide set, natural speech only.

**Week 3: advanced methods**
- M10 variant (c): online WPE with per-speaker memory keyed by embeddings.
- M7.
- M8 port: GTCRN first, then DPDFNet-4, each validated against the sherpa-onnx reference in the `scoring_venv`. NSNet2 only if a pinned source and reference script exist.
- Second round of real component recordings, now using the chosen placements.

**Week 4: TSE, listening and final selection**
- M9 with explicit and gated self-enrollment; wrong-speaker-lock rate; RTF with parallel passes.
- Real bus rides and meetings without references.
- ITU-T P.835 listening test.
- Choose the final stacks against gate G2.
- Make one run on the held-out test set and write up the results.

---

## Appendix B: Unit tests and sanity checks

1. **Plane wave.** Synthesize X_B = X_A·e^(−j2πfτ) for τ ∈ {−d/c, 0, +d/c}.
   - S1 returns τ̂ within ±2 µs.
   - |w_kᴴa_k| = 1 ± 1e-6.
   - With center-referenced steering, the two-talker beam gives r_A = +1 and r_B = −1 within ±1 dB and ±10° in every bin from the dipole's floor-feasible edge up to 0.8·c/2d̂. At a −10 dB floor that is 616–3430 Hz at 4 cm, 985–5488 Hz at 2.5 cm and 410–2287 Hz at 6 cm.
   - Convention check: the same weights, with responses referenced to Mic A, give r_B/r_A = −e^(+j2πf·d̂/c) within ±10°.
   - WNG is at or above the floor in every bin. A violation is a failure, never a silent fallback.
   - Bin-to-bin phase steps of the AB response are ≤ 30°.
2. **Diffuse field.** Feed `lab.scenes.coherent_noise` at d = 2.5, 4 and 6 cm.
   - Long-term CDR, from the coherence averaged over the whole file: median below −10 dB above 300 Hz.
   - Short-time CDR at λ = 0.68: power-averaged gain ≤ −8 dB. Do not test the short-time median, which sits near 0 dB by design (§M4).
   - Γ̂n matches sinc within 0.05 above 300 Hz.
   - d̂ is within ±2 mm, with the sinc model selected. With sensor noise added at a known level, ρ̂ is within 0.02 of the true coherence cap.
3. **Calibration.** Apply +1.5 dB, a 31.25 µs delay and high-pass corners of 50 Hz (Mic A) and 80 Hz (Mic B). S0 recovers the gain within 0.2 dB, the delay within 3 µs and the per-band phase within 1° from 125 Hz up.
4. **Sign conventions.**
   - `lab.dsp.gcc_phat(x_B, x_A)` has the same sign as S1's τ.
   - CDR prop1 with the true τ returns more than +20 dB for a single plane wave.
   - An M3 two-talker beam built from symmetrized RTFs matches the model-based center-referenced M2 beam within 0.1 dB in every bin.
5. **Harness.**
   - Σ comp2 equals the mix to within 1e-6.
   - For a linear method (M1), the target-flip and noise-flip estimates of s_out agree to within 60 dB, and the 4-run decoding recovers A, B and N to within 60 dB.
   - Report the consistency figure for every black-box method.
6. **Latency.**
   - Linear time-invariant methods: an impulse lands at lag 0 ± 2 samples.
   - All other methods (NN, mask-based, adaptive): cross-correlating the separated target component with its reference on speech gives |lag| ≤ 2 samples.
   - Algorithmic latency is reported separately (§4.2).
7. **Phase continuity across states.** Across states A, AB and B, for each partner and every bin below 0.8·c/2d̂:
   - the response phase changes by ≤ 30° where both state responses are ≥ −6 dB, and by ≤ 90° elsewhere;
   - no crossfade dips more than 1 dB below the smaller endpoint magnitude.
8. **Determinism and cost.**
   - The same input always gives the same output (no unseeded randomness).
   - Regenerating a scene with the same seed is bit-identical (E9).
   - Log the real-time factor (RTF) for every run.
9. **Scoring invariance.**
   - Inverting partner B in an ideal output changes per-partner SDR and ESTOI by less than 0.1 dB and 0.005.
   - An oracle-VAD gate on Mic A scores ΔSNR_active within ±0.5 dB of 0 and NM ≥ 20 dB, which fails G0.
10. **Table reflection** (E7). A talker 3 m away, mouth 0.42 m above the table, mics 10 cm above it: the first comb notch is at 6.2 kHz ± 5%. Zeroing the floor image under the table changes the RIR as predicted.

---

## Appendix C: Reproduction scripts

The scratchpad scripts live in a session temp folder. Copy them into the repo before relying on them. Run Python scripts with `speech_lab/.venv/bin/python -I`.

- **§1 physics numbers (v1.0).** Runs in about 2 s:
  `/private/tmp/claude-501/-Users-chpatel-Library-Application-Support-Claude-scratch-workspaces-7223bc40-0755-4c47-a940-84a904b8ead4-809a743c-126a-440a-9a06-f59818e63020-scratch-2026-11-02-5e756e/c3614d02-9a38-4a14-8dcb-766e8c2e70cf/scratchpad/arch/physics_check.py`
  Its `two_talker()` keeps weights that violate the WNG floor when the bisection fails, and it uses Mic-A-referenced steering. The two-talker numbers come from the v1.1 scripts below.
- **v1.1 revision scripts.** `physics_check_v2.py` (about 6 s) covers the two-talker beam with a strict floor, the center- versus Mic-A-referenced comparison and the dipole feasibility band. `continuation_check2.py` (about 12 s; pass this folder as the argument) checks the M2 phase rule on the STFT grid.
  `/private/tmp/claude-501/-Users-chpatel-Library-Application-Support-Claude-scratch-workspaces-7223bc40-0755-4c47-a940-84a904b8ead4-809a743c-126a-440a-9a06-f59818e63020-scratch-2026-11-02-5e756e/c3614d02-9a38-4a14-8dcb-766e8c2e70cf/scratchpad/revise/`
- **Review probes.** Pass the speech_lab path or the review folder as the argument.
  - `probe.py`: polarity and gating scoring, bus orientation, meeting TDOA groups.
  - `di_real.py`: azimuth-averaged MVDR, mismatch, dipole WNG, CDR conditioning.
  - `fig8c.py`: two-talker beam with a strict floor.
  - `axis.py`: |τ| versus distance.
  `/private/tmp/claude-501/-Users-chpatel-Library-Application-Support-Claude-scratch-workspaces-7223bc40-0755-4c47-a940-84a904b8ead4-809a743c-126a-440a-9a06-f59818e63020-scratch-2026-11-02-5e756e/c3614d02-9a38-4a14-8dcb-766e8c2e70cf/scratchpad/review/`
- **Research-phase beam and directivity scripts** (`di_wng.py`, `aidi.py`, `mismatch.py`, `room.py`, `alpha_beam.py`, `cyl.py`, `cyl2.py`):
  `/private/tmp/claude-501/-Users-chpatel-Library-Application-Support-Claude-scratch-workspaces-7223bc40-0755-4c47-a940-84a904b8ead4-809a743c-126a-440a-9a06-f59818e63020-scratch-2026-11-02-5e756e/c3614d02-9a38-4a14-8dcb-766e8c2e70cf/scratchpad/calc/`
- **Verification scripts** (`check_di.py`, `check_2talk.py`, `check_mc1.py`):
  `/private/tmp/claude-501/-Users-chpatel-Library-Application-Support-Claude-scratch-workspaces-7223bc40-0755-4c47-a940-84a904b8ead4-809a743c-126a-440a-9a06-f59818e63020-scratch-2026-11-02-5e756e/c3614d02-9a38-4a14-8dcb-766e8c2e70cf/scratchpad/verify_bf/`
- **CDR verification simulation** (about 15 s; pass the sibling `audio/` folder as the argument). Its noise-only diagnostics are in `diag.py` in the same folder:
  `/private/tmp/claude-501/-Users-chpatel-Library-Application-Support-Claude-scratch-workspaces-7223bc40-0755-4c47-a940-84a904b8ead4-809a743c-126a-440a-9a06-f59818e63020-scratch-2026-11-02-5e756e/c3614d02-9a38-4a14-8dcb-766e8c2e70cf/scratchpad/sim/cdr_check.py`
- **Bench and oracle.** The bench is `BENCH.md`, `lab/bench.py`, `lab/metrics.py`, `lab/scenes.py` and `lab/oracle.py` in the speech_lab folder below. The oracle ideal-ratio-mask upper bound in §0 comes from `.venv/bin/python -m lab.oracle dev` (about 30 s):
  `/Users/chpatel/Library/Application Support/Claude/scratch-workspaces/7223bc40-0755-4c47-a940-84a904b8ead4/809a743c-126a-440a-9a06-f59818e63020/scratch-2026-11-02-5e756e/speech_lab`

---

## Appendix D: Simulation tooling, metrics and research tracks

**Simulation**
- Room impulse responses use the image-source method [89] through pyroomacoustics 0.10.1 [90].
- Turn on `use_rand_ism=True` [91] to avoid sweeping echoes in shoebox rooms, and seed it (E9).
- Source directivity works with the image-source method in shoebox rooms only, not with ray tracing.
- Diffuse noise must have the correct inter-mic coherence [92][93]. Also test on noise whose coherence is not the model's own (E6).

**Metrics**
- SI-SDR [98]: legacy summed-reference column only. Decisions use per-talker SDR with a 512-tap allowed distortion [98].
- ESTOI [99]: pystoi with `extended=True`, on separated per-talker components.
- PESQ-WB [100]: legacy, so use it for relative comparisons only.
- DNSMOS P.835 [101]: only on clips of 9 s or longer, aggregated over at least 20 clips; report only, never a tuning target.

**Optional real data** (check download sizes against the disk budget first)
- DEMAND [94] TBUS, PCAFETER and OMEETING: 16-mic array with 5 cm between neighbours, so real 2-channel noise at our spacing (preferred).
- ETSI TS 103 224 background-noise database [108]: "Inside Bus" and others, 8 channels.
- NOTSOFAR-1 real meetings recorded on tabletop devices [106].
- CHiME-3 real bus recordings [105]: a 6-mic tablet with mic 2 facing backward. No pair is confirmed to fall in 2.5–6 cm, and the WSJ0-based content is LDC-licensed. Use it only if both are acceptable.
- CHiME-6 front-end recipes, for reference [67].

**Research tracks (outside the 4-week plan)**
- **Bus:** both of these need training on a PyTorch machine elsewhere.
  - Hearables that keep only talkers inside a set radius ("sound bubbles") [85].
  - Single-mic separation by talker distance [86].
- **Meeting:** NTT's spatial-dictionary approach to online diarization and beamforming [55].
- **Two-mic neural:** fine-tuning H-GTCRN [75].
- **Partner extraction from one enrollment:** target-conversation extraction, which recovers both partners from a single enrolled voice [84].

---

## Review responses

| # | Finding | Response |
|---|---|---|
| 1 | Two-talker beam spec | **Partly accepted.** Accepted and applied: the corrected AI-DI (3.4 / 3.6 / 3.5 dB in 3-D and 2.2 / 2.3 / 2.3 dB in 2-D, floor strictly enforced); the physics_check.py floor-violation bug; no amplitude blending of dipole and omni; B.1 limited to floor-feasible bins below 0.8·c/2d̂; B.7 rewritten as phase continuity. Not accepted: that the r_B = −1 spec is wrong. The document specifies center-referenced steering (§2.0), and there h = [1, −1] is exactly the equalized figure-8 the review derives. Re-running the search with center-referenced vectors gives ψ_AB = 180° in every floor-feasible bin up to c/2d (physics_check_v2.py). The 190–348° phases and the 0.9 / −0.4 / −1.6 dB beam arise only with Mic-A-referenced vectors, which physics_check.py used. M2, M3 and B.1 now state the convention and test it, and the continuation rule replaces the "rotate toward omni" wording with a procedure checked to reach the optimum |
| 2 | Per-partner, polarity-tolerant bus scoring | Accepted (§4.2–4.3, B1, B.9) |
| 3 | Metrics that reward gating or hide far talkers | Accepted. BR and gapSNR are demoted to legacy; separated noise is measured during speech and idle; pre-AGC TA_k by distance bin and worst talker; per-talker WER; the far-versus-near G0 gate; no tuning on DNSMOS BAK/OVRL (§4.3–4.6) |
| 4 | Bus simulation too easy | Accepted (E1, E2, E4–E6) |
| 5 | State the 2-mic ceiling | Accepted (§0, §1.2, §1.5) |
| 6 | Orientation as a factor and an instruction | Accepted (E1 strata, the S1 inseparable flag, §5 placement, the TDOA-indexed gain prior in M11) |
| 7 | 20 talkers are not TDOA-separable | Accepted. S1, M10(c) and M11 are keyed by speaker embeddings; returning-talker scenes added (E7) |
| 8 | Meeting geometry | Accepted (E7, §5 item 3) |
| 9 | Replace TTS speech | Accepted (E0, Week 1; G1 decided on natural speech) |
| 10 | Too few scenes | Accepted. At least 30 geometries per scenario per set; CI-based gates (§4.5); seeding (E9) |
| 11 | CDR test and conditioning | Accepted (M4, B.2, §2.2) |
| 12 | Calibration model | Accepted. Per-band complex correction; calipers plus endfire chirps as the primary procedure; free-ρ, two-model blind fit (S0) |
| 13 | Hagerman and latency harness | Accepted (§4.2, B.5, B.6) |
| 14 | M8/M9 as written | Accepted. GTCRN and DPDFNet first, NSNet2 only from a pinned source; VoxLock always with explicit enrollment, gated self-enrollment, lock rate, RTF policy. The v1.0 M9 already ran VoxLock twice with per-partner enrollment; automatic mode is now explicitly forbidden |
| 15 | Scope and industry practice | Accepted. Capture-only scope with AEC as a prerequisite otherwise; ESP32 recordings moved to Week 1; optional TDM path marked "confirm"; listening test per ITU-T P.835 |
| 16 | Citations and text | Accepted ([39], [99], [102], [106], Appendix D). Checked: refs 1–106 do not use DOI 10.1109/ASPAA.1995.482964. Ref [5] names the Elko–Nguyen Pong WASPAA 1995 paper by title only, and is now marked as such |