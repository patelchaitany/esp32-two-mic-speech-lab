# B1: two-mic physics numbers versus spacing (1.0–6.0 cm)

Hardware follow-up to DESIGN.md v1.1 §1, 2026-11-03. Spacings 1.0 / 1.2 / 1.5 / 2.0 / 2.5 / 4.0 / 6.0 cm, fs 16 kHz, c 343 m/s. Ideal free-field omni mics, ideal diffuse fields, no device body, no field losses (subtract 0.5–1.5 dB as DESIGN §1.2 says). Analytic and vectorised; no audio was played, recorded or opened.

**Status.** Every DESIGN §1 / M2 / M4 / §1.5 number at 2.5 / 4 / 6 cm was reproduced first: 42 of 43 checks pass. The one miss is a last-digit slip in the DESIGN §1.1 table (sinc > 0.7 at 4 cm is 1924.6 Hz, printed as 1.93 kHz). The vectorised two-talker beam matches the original `physics_check_v2.py` loop exactly at 1.0 and 1.2 cm as well (§2).

Files: `physics_b1.py` (all computations → `physics.json`, `tables.md`), `crosscheck_v2.py` (→ `crosscheck_v2.json`), `write_md.py` (→ this file). Run with `OMP_NUM_THREADS=1 .venv/bin/python -I -B physics_b1.py "$LAB"` (about 2 CPU-s).

## Bottom line at 1.0–2.0 cm (most likely about 1.2 cm)

- **Axis-aligned single talker: almost no loss.** The superdirective beam at WNG ≥ −10 dB gives 4.69 dB AI-DI (3-D) / 3.89 dB (2-D) at 1.2 cm, against 4.96 / 3.91 dB at 4 cm. The small array loses below 1 kHz but has no spatial aliasing (c/2d = 14.29 kHz), so it keeps 5.1–6.0 dB per band from 1.6 to 8 kHz, where the 4 cm array drops to 2.5–4.3 dB above 3 kHz.
- **Orientation-averaged gain drops a little; broadside drops to zero.** Averaged over azimuth in 2-D noise: 2.22 dB at 1.2 cm vs 2.66 dB at 4 cm (r_c × 1.29 vs × 1.36). At broadside: 0.13 dB (3-D) vs 1.04 dB.
- **Two-talker figure-8 loses about a third.** 1.58 dB (2-D) / 2.45 dB (3-D) at 1.2 cm vs 2.31 / 3.60 dB at 4 cm. The dipole band starts at 2052 Hz (−10 dB floor), or at 3296 Hz under the M2 bus schedule (−6 dB above 2 kHz). With a −20 dB floor the 4 cm figures come back (2.63 / 4.13 dB), but such low floors need calibration (§4.5 shows the cost for the single-target beam).
- **Below 1 kHz every useful beam has a large white-noise cost** (§5.2). Reaching just 3 dB of DI on the axis costs 6.5–24.2 dB of WNG over 1000–125 Hz, and the figure-8 costs 16.2–34.2 dB. That is 10.5 dB more than at 4 cm (WNG scales with d²).
- **Broadside talkers look like diffuse noise up to about 3.6 kHz** (coherence > 0.9), and the CDR rule (|Γs − Γn| ≥ 0.2) is met only above 5.15 kHz. Endfire partners stay usable for CDR from 914 Hz.
- **TDOA is nearly useless as talker identity:** the 20 meeting seats collapse to 4 groups at 20 µs resolution (6 at 10 µs).
- **The measured 68 µs skew is 1.94 × the largest physical TDOA at 1.2 cm.** Uncompensated, it turns the axis beam into -6.17 dB AI-DI. S0 must remove it to within about 5 µs (0.08 samples), which costs ≤ 0.15 dB.

## 1. Definitions used (as in the scripts behind DESIGN v1.1)

Sources read: DESIGN.md §1, M1, M2, M4, §1.5/§1.6 and Appendix C, plus the scratchpad scripts it names (`arch/physics_check.py`, `revise/physics_check_v2.py`, `review/di_real.py`, `review/probe.py`). Each reading below reproduces the published numbers.

- **AI-DI**: Σ I_i · DI(f_i) in dB, ANSI S3.5 one-third-octave band-importance weights (sum 1.000) at 18 band centres 160 Hz–8 kHz, with the 8 kHz centre evaluated at 7.9 kHz. DI is evaluated at the band centre only (no averaging inside a band).
- **Noise models**: 3-D (spherically diffuse) Γ = sinc(2fd/c) (numpy sinc); 2-D (horizontally diffuse) Γ = J0(2πfd/c).
- **WNG** = |wᴴa|²/(wᴴw), relative to one omni mic (delay-and-sum = +3 dB). Floors are hard constraints.
- **Single-target beam (M2)**: per-band MVDR against the diffuse model, w = (Γ+μI)⁻¹a / (aᴴ(Γ+μI)⁻¹a), μ = the smallest value meeting the floor (bisection on log10 μ in [−9, 6]). 'On the axis' = θ = 0. In the on-axis rows the 2-D figures use J0-designed weights. In the azimuth-averaged and broadside rows they use 3-D-designed weights evaluated against J0.
- **Azimuth average**: plain mean of the AI-DI (dB) over 19 angles θ = 0, 5, …, 90° (folded azimuth, talkers in the horizontal plane, uniform). Broadside = θ = 90°; there the MVDR is identical to delay-and-sum.
- **Delay-and-sum (M1)**: w = a/2 steered to θ = 0 (axis) or 90° (broadside), evaluated in 3-D noise.
- **Two-talker beam (M2)**: soft LCMV w = (Γ + μI + λ_c AAᴴ)⁻¹ λ_c A h*, centre-referenced steering a(f,τ) = [e^(+jπfτ), e^(−jπfτ)], A at 0°, B at 180°, h = [1, e^(jψ)], λ_c = 10³. ψ is searched on a 2.5° grid. A candidate is rejected if it violates the floor (strict) or if |r_A|/|r_B| is outside ±1 dB. The kept candidate maximizes DF = ½(|r_A|²+|r_B|²)/(wᴴΓw). A band with no candidate gets a single omni (0 dB). Weights are designed for the 3-D model; the 2-D figure picks ψ by the J0-evaluated DF (as `physics_check_v2.py` does).
- **Dipole band**: centre-referenced h = [1, −1] has WNG = 1 − cos(2πfd/c). The band runs from f = arccos(1 − 10^(floor/10))·c/(2πd) up to min(c/2d, fs/2). Under the M2 bus schedule (−15 dB below 500 Hz, −10 dB to 2 kHz, −6 dB above), the edge is found on a 0.1 Hz grid.
- **CDR usability (M4)**: a bin is used where |Γs − Γn| ≥ 0.2, with Γs = e^(+j2πfτ) and Γn the 3-D sinc model. Broadside: τ = 0. Endfire: τ = d/c. The reported value is the first crossing. 2-D (J0) variants are in the JSON.
- **Coherence > 0.9** (main-table column): the spherical (sinc) value, as in DESIGN §1.1 ('about 1.1 kHz at 4 cm'). J0 values are in §4.2.
- **Meeting TDOA groups (§1.5)**: the 20 seats of lab scene `meeting_s0` (8 m × 2 m table, mouths at 1.2 m, device at 0.78 m). The mic axis is horizontal at a uniform azimuth, τ = (|s−B| − |s−A|)/c, and sorted τ values are grouped greedily (a new group when τ − group start > resolution). Headline = median over 720 uniform azimuths. At 2.5 / 4 / 6 cm this equals the exact 500-draw replica of `review/probe.py` (same RNG stream).
- **Skew ratio** = 68 µs / (d/c), the measured inter-channel skew magnitude over the largest geometric TDOA.

## 2. Validation against DESIGN v1.1 (2.5 / 4 / 6 cm unless stated)

Pass rule: within 0.2 dB for AI-DI rows; otherwise within the stated rounding of each design cell; counts exact.

| quantity | design | reproduced | ok |
|---|---|---|---|
| 1.1 c/2d aliasing (kHz) 2.5/4/6 cm | 6.86 / 4.29 / 2.86 kHz | 6.86 / 4.29 / 2.86 kHz | yes |
| 1.1 max TDOA (us) 2.5/4/6 cm | 72.9 / 116.6 / 174.9 us | 72.9 / 116.6 / 174.9 us | yes |
| 1.1 max TDOA (samples) 2.5/4/6 cm | 1.17 / 1.87 / 2.80 smp | 1.17 / 1.87 / 2.80 smp | yes |
| 1.1 sinc coherence > 0.9 below (kHz) 2.5/4/6 cm | 1.72 / 1.07 / 0.72 kHz | 1.72 / 1.07 / 0.72 kHz | yes |
| 1.1 sinc coherence > 0.7 / 0.5 below at 4 cm (Hz; design 1.93 / 2.59 kHz, stated rounding +-5 Hz) | 1930.0 / 2590.0 Hz | 1924.6 / 2586.9 Hz (1924.6 Hz is 1.92 kHz: last-digit slip in the DESIGN 1.1 table; the same formula reproduces every other cell, e.g. 3.08 / 1.54 / 1.28 kHz for > 0.7 at 2.5 / 5 / 6 cm) | **no** |
| 1.1 sinc coherence > 0.7 below (kHz) at 2.5 / 5 / 6 cm | 3.08 / 1.54 / 1.28 kHz | 3.08 / 1.54 / 1.28 kHz | yes |
| 1.1 J0 coherence > 0.9 below (kHz) 2.5/4/6 cm | 1.40 / 0.87 / 0.58 kHz | 1.40 / 0.87 / 0.58 kHz | yes |
| 1.2 single target on axis, WNG>=-10, 3-D | 5.20 / 5.00 / 4.60 dB | 5.15 / 4.96 / 4.61 dB | yes |
| 1.2 single target on axis, WNG>=-10, 2-D (J0 weights) | 4.10 / 3.90 / 3.70 dB | 4.12 / 3.91 / 3.65 dB | yes |
| 1.2 single target on axis, WNG>=-3, 3-D | 4.10 / 4.30 / 4.30 dB | 4.13 / 4.32 / 4.26 dB | yes |
| 1.2 single target on axis, WNG>=-3, 2-D (J0 weights) | 3.40 / 3.50 / 3.40 dB | 3.41 / 3.50 / 3.44 dB | yes |
| 1.2 single target on axis, WNG>=-20, 3-D | 5.50 / 5.10 / 4.70 dB | 5.54 / 5.10 / 4.65 dB | yes |
| 1.2 single target azimuth-averaged, WNG>=-10, 3-D | 3.20 / 3.30 / 3.30 dB | 3.21 / 3.29 / 3.27 dB | yes |
| 1.2 single target azimuth-averaged, WNG>=-10, 2-D (3-D weights) | 2.50 / 2.70 / 2.70 dB | 2.54 / 2.66 / 2.72 dB | yes |
| 1.2 single target at broadside, WNG>=-10, 3-D | 0.50 / 1.00 / 1.50 dB | 0.52 / 1.04 / 1.53 dB | yes |
| 1.2 single target at broadside, WNG>=-10, 2-D (3-D weights) | 0.80 / 1.50 / 2.00 dB | 0.79 / 1.46 / 1.99 dB | yes |
| 1.2 delay-and-sum along the axis, 3-D | 1.30 / 1.90 / 2.30 dB | 1.31 / 1.86 / 2.27 dB | yes |
| 1.2 delay-and-sum to broadside, 3-D | 0.50 / 1.00 / 1.50 dB | 0.52 / 1.04 / 1.53 dB | yes |
| 1.4 r_c factor of azimuth-averaged 2-D MVDR, min-max over 2.5-6 cm (x) | 1.34 / 1.37 x | 1.34 / 1.37 x | yes |
| M2 meeting: on the axis / broadside at 4 cm, WNG>=-3, 3-D | 4.30 / 1.00 dB | 4.32 / 1.04 dB | yes |
| 1.2 per-band DI on axis 4 cm, WNG>=-10 (150/250/500/1k/2k/3k/4k/6k Hz) | 2.3 / 3.9 / 6.0 / 5.9 / 5.4 / 4.5 / 3.3 / 2.9 dB | 2.3 / 3.9 / 6.0 / 5.9 / 5.4 / 4.5 / 3.3 / 2.9 dB | yes |
| 1.2 WNG of unconstrained endfire SD 4 cm (250/500/1000 Hz) | -15.3 / -9.3 / -3.6 dB | -15.3 / -9.3 / -3.6 dB | yes |
| 1.2 broadside MVDR DI 4 cm (0.5/1/2/3/4 kHz) | 0.05 / 0.19 / 0.76 / 1.65 / 2.70 dB | 0.05 / 0.19 / 0.76 / 1.65 / 2.71 dB | yes |
| 1.2 off-axis steering 4 cm, WNG>=-10 (0/30/45/60/90 deg) | 5.0 / 4.3 / 3.6 / 2.6 / 1.0 dB | 5.0 / 4.3 / 3.6 / 2.6 / 1.0 dB | yes |
| 1.2 off-axis steering 4 cm, WNG>=-3 (0/30/45/60/90 deg) | 4.3 / 3.8 / 3.1 / 2.3 / 1.0 dB | 4.3 / 3.8 / 3.1 / 2.3 / 1.0 dB | yes |
| 1.2 two-talker beam, WNG>=-10 strict, 3-D | 3.40 / 3.60 / 3.50 dB | 3.42 / 3.60 / 3.51 dB | yes |
| 1.2 two-talker beam, WNG>=-10 strict, 2-D (3-D weights) | 2.20 / 2.30 / 2.30 dB | 2.18 / 2.31 / 2.25 dB | yes |
| 1.2 two-talker beam, WNG>=-20 strict, 3-D | 4.40 / 4.10 / 3.70 dB | 4.42 / 4.13 / 3.73 dB | yes |
| 1.2 two-talker beam, WNG>=-20 strict, 2-D (3-D weights) | 2.80 / 2.60 / 2.40 dB | 2.79 / 2.63 / 2.39 dB | yes |
| M2/1.6-18: optimal psi_AB = 180 deg in every floor-feasible band up to c/2d (max |psi-180|, deg) | 0.0 / 0.0 / 0.0 deg | 0.0 / 0.0 / 0.0 deg | yes |
| M2 convention check: Mic-A-referenced h=[1,-1], WNG>=-10, 3-D | 0.90 / -0.40 / -1.60 dB | 0.92 / -0.39 / -1.59 dB | yes |
| M2 dipole band lower edge, WNG>=-10 (Hz) | 985 / 616 / 410 Hz | 985 / 616 / 410 Hz | yes |
| Appendix B.1 dipole test band upper edge 0.8 c/2d (Hz) | 5488 / 3430 / 2287 Hz | 5488 / 3430 / 2287 Hz | yes |
| M4 CDR usable, broadside partner, 3-D (kHz) | 2.47 / 1.54 / 1.03 kHz | 2.47 / 1.54 / 1.03 kHz | yes |
| M4 CDR usable, endfire partner, 3-D (kHz) | 0.44 / 0.27 / 0.18 kHz | 0.44 / 0.27 / 0.18 kHz | yes |
| M4 |Gs-Gn| at 1 kHz, broadside, 2.5/4 cm | 0.03 / 0.09 | 0.03 / 0.09 | yes |
| 1.3 near-field ILD 0.3/0.5/1.0 m, 4 cm | 1.10 / 0.67 / 0.34 dB | 1.09 / 0.67 / 0.34 dB | yes |
| 1.3 near-field ILD 0.3/0.5/1.0 m, 6 cm | 1.60 / 1.00 / 0.50 dB | 1.58 / 0.98 / 0.51 dB | yes |
| 1.2 mismatch cost +-1 dB/+-3 deg, az-avg 2-D, WNG>=-10 (dB, <=0.1) | <= 0.1 dB at 2.5 / 4 / 6 cm | 0.07 / 0.06 / 0.04 dB | yes |
| 1.5 TDOA groups, 20 seats, 20 us, 4 cm: median (min-max) | 8 / 6 / 10 | 8 / 6 / 10 | yes |
| 1.5 TDOA groups, 20 seats, 10 us, 4 cm: median | 12 | 12 | yes |
| 1.5 TDOA groups, 20 seats, 20 us, 2.5 cm: median | 6 | 6 | yes |
| 1.5 neighbouring seats < 10 us apart (of 19), 4 cm scenes, min-max | 8 / 13 | 8 / 13 | yes |

Cross-check of the extension: the original `physics_check_v2.two_talker()` loop, run unchanged at the new spacings (WNG ≥ −10 dB), against this vectorised version:

| spacing, model | original v2 | physics_b1 | diff |
|---|---|---|---|
| 1.0cm, 3d | 2.147 dB | 2.147 dB | +0.000 |
| 1.0cm, 2d | 1.394 dB | 1.394 dB | +0.000 |
| 1.2cm, 3d | 2.445 dB | 2.445 dB | +0.000 |
| 1.2cm, 2d | 1.583 dB | 1.583 dB | +0.000 |

## 3. Main table

AI-DI in dB, WNG ≥ −10 dB. SD = single-target superdirective (MVDR) beam. 'bs' = broadside. Two-talker = partners on the axis on opposite sides, 2-D babble. Band edges in Hz. TDOA groups are medians for 20 seats.

| d (cm) | c/2d (kHz) | max TDOA µs (samples) | sinc > 0.9 until (kHz) | DAS axis / bs, 3-D | SD axis, 3-D | SD bs, 3-D | SD azimuth-avg, 2-D | two-talker, 2-D | dipole band from | CDR usable from: endfire / bs | TDOA groups at 20 / 10 µs | 68 µs skew ÷ max TDOA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 17.15 | 29.2 (0.47) | 4.29 | 0.34 / 0.09 | 4.45 | 0.09 | 2.12 | 1.39 | 2462 | 1097 / 6175 | 3 / 5 | 2.33 |
| 1.2 | 14.29 | 35.0 (0.56) | 3.58 | 0.48 / 0.13 | 4.69 | 0.13 | 2.22 | 1.58 | 2052 | 914 / 5146 | 4 / 6 | 1.94 |
| 1.5 | 11.43 | 43.7 (0.70) | 2.86 | 0.70 / 0.20 | 4.93 | 0.20 | 2.34 | 1.80 | 1641 | 731 / 4116 | 4 / 7 | 1.55 |
| 2.0 | 8.57 | 58.3 (0.93) | 2.15 | 1.04 / 0.34 | 5.12 | 0.34 | 2.47 | 2.04 | 1231 | 548 / 3087 | 5 / 8 | 1.17 |
| 2.5 | 6.86 | 72.9 (1.17) | 1.72 | 1.31 / 0.52 | 5.15 | 0.52 | 2.54 | 2.18 | 985 | 439 / 2470 | 6 / 10 | 0.93 |
| 4.0 | 4.29 | 116.6 (1.87) | 1.07 | 1.86 / 1.04 | 4.96 | 1.04 | 2.66 | 2.31 | 616 | 274 / 1544 | 8 / 12 | 0.58 |
| 6.0 | 2.86 | 174.9 (2.80) | 0.72 | 2.27 / 1.53 | 4.61 | 1.53 | 2.72 | 2.25 | 410 | 183 / 1029 | 11 / 14 | 0.39 |

c/2d above 8 kHz (1.0–2.0 cm) means no spatial aliasing anywhere in the 16 kHz band.

## 4. More numbers by spacing

### 4.1 AI-DI (dB) by beam, WNG floor and noise model (3-D / 2-D)

| d (cm) | SD axis −3 | SD axis −10 | SD axis −20 | SD axis, bus schedule | SD az-avg −10 | SD az-avg −3 | SD broadside −10 | 2-talker −10 | 2-talker −20 | 2-talker, bus schedule | 2-talker, fixed dipole −10 | DAS axis / bs (3-D) | r_c factor (az-avg 2-D) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 2.85 / 2.53 | 4.45 / 3.72 | 5.71 / 4.58 | 4.53 / 3.82 | 2.62 / 2.12 | 1.70 / 1.50 | 0.09 / 0.14 | 2.15 / 1.39 | 3.95 / 2.52 | 1.62 / 1.09 | 1.67 / 1.05 | 0.34 / 0.09 | × 1.28 |
| 1.2 | 3.18 / 2.78 | 4.69 / 3.89 | 5.76 / 4.60 | 4.82 / 4.01 | 2.78 / 2.22 | 1.90 / 1.65 | 0.13 / 0.19 | 2.45 / 1.58 | 4.13 / 2.63 | 2.14 / 1.41 | 1.65 / 1.04 | 0.48 / 0.13 | × 1.29 |
| 1.5 | 3.54 / 3.04 | 4.93 / 4.05 | 5.77 / 4.58 | 5.09 / 4.19 | 2.95 / 2.34 | 2.15 / 1.83 | 0.20 / 0.30 | 2.79 / 1.80 | 4.31 / 2.73 | 2.74 / 1.78 | 2.05 / 1.29 | 0.70 / 0.20 | × 1.31 |
| 2.0 | 3.93 / 3.30 | 5.12 / 4.14 | 5.68 / 4.48 | 5.32 / 4.30 | 3.12 / 2.47 | 2.43 / 2.05 | 0.34 / 0.53 | 3.19 / 2.04 | 4.43 / 2.80 | 3.24 / 2.08 | 2.80 / 1.76 | 1.04 / 0.34 | × 1.33 |
| 2.5 | 4.13 / 3.41 | 5.15 / 4.12 | 5.54 / 4.34 | 5.37 / 4.27 | 3.21 / 2.54 | 2.62 / 2.20 | 0.52 / 0.79 | 3.42 / 2.18 | 4.42 / 2.79 | 3.52 / 2.25 | 3.09 / 1.95 | 1.31 / 0.52 | × 1.34 |
| 4.0 | 4.32 / 3.50 | 4.96 / 3.91 | 5.10 / 3.98 | 5.08 / 3.98 | 3.29 / 2.66 | 2.91 / 2.47 | 1.04 / 1.46 | 3.60 / 2.31 | 4.13 / 2.63 | 3.86 / 2.48 | 3.10 / 1.88 | 1.86 / 1.04 | × 1.36 |
| 6.0 | 4.26 / 3.44 | 4.61 / 3.65 | 4.65 / 3.66 | 4.65 / 3.66 | 3.27 / 2.72 | 3.06 / 2.64 | 1.53 / 1.99 | 3.51 / 2.25 | 3.73 / 2.39 | 3.69 / 2.36 | 2.45 / 1.39 | 2.27 / 1.53 | × 1.37 |

Bus schedule = −15 dB below 500 Hz, −10 dB to 2 kHz, −6 dB above (M2). 'Fixed dipole' = ψ = 180° in every band, omni where the dipole is infeasible (the v2 script's comparison beam). No band fell back to omni in the phase-optimized two-talker beam at any spacing or floor.

### 4.2 Band edges (Hz)

| d (cm) | dipole band from: −10 / −15 / −20 dB / bus schedule | CDR endfire 3-D (2-D) | CDR broadside 3-D (2-D) | coherence > 0.9 until, 3-D / 2-D | broadside beam reaches 0.5 / 1 dB | AI weight above: dipole −10 / CDR endfire / CDR bs |
|---|---|---|---|---|---|---|
| 1.0 | 2462 / 1377 / 773 / 3955 | 1097 (1098) | 6175 (5013) | 4295 / 3497 | 6459 / 9211 (> fs/2) | 0.36 / 0.62 / 0.05 |
| 1.2 | 2052 / 1147 / 644 / 3296 | 914 (915) | 5146 (4178) | 3579 / 2914 | 5382 / 7676 | 0.36 / 0.70 / 0.05 |
| 1.5 | 1641 / 918 / 515 / 2637 | 731 (732) | 4116 (3342) | 2863 / 2331 | 4306 / 6141 | 0.45 / 0.77 / 0.11 |
| 2.0 | 1231 / 688 / 386 / 1231 | 548 (549) | 3087 (2507) | 2147 / 1749 | 3229 / 4605 | 0.62 / 0.84 / 0.27 |
| 2.5 | 985 / 551 / 309 / 985 | 439 (439) | 2470 (2005) | 1718 / 1399 | 2583 / 3684 | 0.70 / 0.89 / 0.36 |
| 4.0 | 616 / 344 / 193 / 616 | 274 (274) | 1544 (1253) | 1074 / 874 | 1615 / 2303 | 0.84 / 0.97 / 0.53 |
| 6.0 | 410 / 229 / 129 / 230 | 183 (183) | 1029 (836) | 716 / 583 | 1076 / 1535 | 0.89 / 0.99 / 0.62 |

The dipole band ends at min(c/2d, 8 kHz); for 1.0–2.0 cm that is 8 kHz. 'AI weight above' = the share of the speech band-importance whose band centres lie above the edge. The broadside beam is delay-and-sum, whatever the floor.

### 4.3 White-noise gain that a useful beam needs below 2 kHz

WNG (dB) of the on-axis MVDR at the point where its 3-D DI reaches 3 dB. Values near +3 dB mean a beam close to delay-and-sum already gets there.

| d (cm) | 125 Hz | 160 Hz | 250 Hz | 315 Hz | 500 Hz | 630 Hz | 800 Hz | 1000 Hz | 1250 Hz | 1600 Hz | 2000 Hz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | -25.8 | -23.7 | -19.8 | -17.8 | -13.8 | -11.9 | -9.8 | -8.0 | -6.2 | -4.3 | -2.6 |
| 1.2 | -24.2 | -22.1 | -18.2 | -16.2 | -12.3 | -10.3 | -8.3 | -6.5 | -4.7 | -2.9 | -1.4 |
| 1.5 | -22.3 | -20.2 | -16.3 | -14.3 | -10.4 | -8.5 | -6.5 | -4.7 | -3.1 | -1.4 | 0.0 |
| 2.0 | -19.8 | -17.7 | -13.8 | -11.9 | -8.0 | -6.1 | -4.3 | -2.6 | -1.1 | 0.3 | 1.4 |
| 2.5 | -17.9 | -15.7 | -11.9 | -10.0 | -6.2 | -4.4 | -2.6 | -1.1 | 0.2 | 1.4 | 2.2 |
| 4.0 | -13.8 | -11.7 | -8.0 | -6.1 | -2.6 | -1.1 | 0.3 | 1.4 | 2.2 | 2.8 | 3.0 |
| 6.0 | -10.4 | -8.3 | -4.7 | -3.0 | 0.0 | 1.1 | 2.1 | 2.7 | 3.0 | 3.0 | 3.0 |

At 1.2 cm, by beam type (WNG in dB, then the DI that a given floor allows):

| quantity at 1.2 cm | 125 Hz | 250 Hz | 500 Hz | 1000 Hz | 2000 Hz |
|---|---|---|---|---|---|
| WNG, unconstrained superdirective (hypercardioid, ≈ 6 dB DI) | -31.7 | -25.7 | -19.7 | -13.7 | -7.8 |
| WNG, rear-null cardioid (best WNG for any null) | -28.2 | -22.2 | -16.2 | -10.2 | -4.4 |
| WNG, two-talker figure-8 (dipole) | -34.2 | -28.2 | -22.2 | -16.2 | -10.2 |
| DI (3-D) allowed at WNG ≥ −6 dB | 0.3 | 0.6 | 1.3 | 2.8 | 5.5 |
| DI (3-D) allowed at WNG ≥ −10 dB | 0.5 | 1.1 | 2.3 | 4.6 | 6.0 |
| DI (3-D) allowed at WNG ≥ −15 dB | 1.0 | 2.0 | 4.2 | 6.0 | 6.0 |
| DI (3-D) allowed at WNG ≥ −20 dB | 1.8 | 3.7 | 6.0 | 6.0 | 6.0 |

For comparison at 4 cm: unconstrained WNG -21.3, -15.3, -9.3, -3.6, 1.2 dB and dipole WNG -23.8, -17.8, -11.8, -5.9, -0.5 dB at the same frequencies.

### 4.4 Per-band DI (dB), 1.2 cm against 4 cm

| band (Hz) | SD axis −10, 3-D, 1.2 | SD axis −10, 3-D, 4 | SD axis −20, 3-D, 1.2 | 2-talker −10, 3-D, 1.2 | 2-talker −10, 2-D, 1.2 | 2-talker −10, 2-D, 4 | ψ_AB (°) −10, 1.2 | broadside, 3-D, 1.2 | broadside, 3-D, 4 |
|---|---|---|---|---|---|---|---|---|---|
| 160 | 0.7 | 2.4 | 2.3 | 0.0 | 0.0 | 0.1 | 9 | 0.0 | 0.0 |
| 200 | 0.9 | 3.1 | 2.9 | 0.0 | 0.0 | 0.2 | 349 | 0.0 | 0.0 |
| 250 | 1.1 | 3.9 | 3.7 | 0.0 | 0.0 | 0.3 | 346 | 0.0 | 0.0 |
| 315 | 1.4 | 4.8 | 4.7 | 0.1 | 0.0 | 0.6 | 343 | 0.0 | 0.0 |
| 400 | 1.8 | 5.8 | 5.6 | 0.1 | 0.1 | 1.0 | 338 | 0.0 | 0.0 |
| 500 | 2.3 | 6.0 | 6.0 | 0.2 | 0.1 | 1.7 | 28 | 0.0 | 0.0 |
| 630 | 2.9 | 6.0 | 6.0 | 0.3 | 0.2 | 3.0 | 35 | 0.0 | 0.1 |
| 800 | 3.7 | 5.9 | 6.0 | 0.4 | 0.3 | 3.0 | 45 | 0.0 | 0.1 |
| 1000 | 4.6 | 5.9 | 6.0 | 0.7 | 0.5 | 3.0 | 57 | 0.0 | 0.2 |
| 1250 | 5.6 | 5.8 | 6.0 | 1.2 | 0.8 | 2.9 | 74 | 0.0 | 0.3 |
| 1600 | 6.0 | 5.6 | 6.0 | 2.1 | 1.5 | 2.9 | 101 | 0.0 | 0.5 |
| 2000 | 6.0 | 5.4 | 6.0 | 4.3 | 2.8 | 2.8 | 154 | 0.1 | 0.8 |
| 2500 | 5.9 | 5.0 | 5.9 | 4.7 | 3.0 | 2.7 | 180 | 0.1 | 1.2 |
| 3150 | 5.9 | 4.3 | 5.9 | 4.7 | 3.0 | 2.5 | 180 | 0.2 | 1.8 |
| 4000 | 5.8 | 3.3 | 5.8 | 4.7 | 2.9 | 2.0 | 180 | 0.3 | 2.7 |
| 5000 | 5.7 | 2.5 | 5.7 | 4.6 | 2.9 | 1.3 | 180 | 0.4 | 3.6 |
| 6300 | 5.4 | 3.1 | 5.4 | 4.5 | 2.8 | 1.1 | 180 | 0.7 | 4.1 |
| 7900 | 5.1 | 3.3 | 5.1 | 4.3 | 2.7 | 2.4 | 180 | 1.1 | 3.4 |

### 4.5 Calibration sensitivity

| d (cm) | az-avg 2-D −10: ideal / ±1 dB ±3° / ±0.25 dB ±1° | az-avg 2-D −10 with residual skew 2 / 5 / 10 µs | axis 3-D −10: ideal, then skew 2 / 5 / 10 µs | uncompensated 68 µs skew: axis 3-D / az-avg 2-D | axis 3-D −20: ideal / ±1 dB ±3° / ±0.25 dB ±1° |
|---|---|---|---|---|---|
| 1.0 | 2.12 / 2.02 / 2.11 | 2.08 / 1.98 / 1.64 | 4.45, then 4.39 / 4.26 / 3.98 | -4.70 / -4.10 | 5.71 / 4.81 / 5.60 |
| 1.2 | 2.22 / 2.13 / 2.22 | 2.19 / 2.09 / 1.81 | 4.69, then 4.64 / 4.54 / 4.30 | -6.17 / -5.54 | 5.76 / 4.97 / 5.66 |
| 1.5 | 2.34 / 2.25 / 2.34 | 2.31 / 2.22 / 2.00 | 4.93, then 4.90 / 4.82 / 4.63 | -9.87 / -8.00 | 5.77 / 5.09 / 5.69 |
| 2.0 | 2.47 / 2.38 / 2.46 | 2.43 / 2.36 / 2.19 | 5.12, then 5.10 / 5.04 / 4.90 | -11.47 / -14.47 | 5.68 / 5.17 / 5.63 |
| 2.5 | 2.54 / 2.47 / 2.54 | 2.51 / 2.45 / 2.31 | 5.15, then 5.14 / 5.10 / 4.99 | -4.63 / -5.89 | 5.54 / 5.14 / 5.50 |
| 4.0 | 2.66 / 2.60 / 2.66 | 2.64 / 2.59 / 2.50 | 4.96, then 4.96 / 4.94 / 4.89 | -0.61 / -1.57 | 5.10 / 4.88 / 5.08 |
| 6.0 | 2.72 / 2.68 / 2.72 | 2.71 / 2.68 / 2.61 | 4.61, then 4.61 / 4.60 / 4.56 | 1.54 / -0.06 | 4.65 / 4.54 / 4.64 |

Mismatch: independent uniform per-mic gain and phase errors, the DF averaged over 200–400 draws (as in `review/di_real.py`). Skew: a fixed inter-channel delay that the beam does not know about; the worse sign is reported. AI-DI in dB.

### 4.6 Meeting: 20 seats as TDOA groups

| d (cm) | TDOA span (µs) | 20 µs: median (min–max, p10–p90) | 10 µs: median (min–max, p10–p90) | median over 6 seat layouts, 20 / 10 µs | neighbouring seats < 10 µs apart (of 19) |
|---|---|---|---|---|---|
| 1.0 | 58 | 3 (3–3, 3–3) | 5 (4–6, 5–6) | 3 / 5 | 19 |
| 1.2 | 70 | 4 (3–4, 3–4) | 6 (5–7, 5–6) | 4 / 6 | 19 |
| 1.5 | 87 | 4 (4–5, 4–4) | 7 (5–8, 6–8) | 4 / 7 | 18 |
| 2.0 | 117 | 5 (4–6, 5–6) | 8 (6–10, 7–9) | 5 / 8 | 16 |
| 2.5 | 146 | 6 (5–7, 5–6) | 10 (7–12, 8–11) | 6 / 9.5 | 13 |
| 4.0 | 233 | 8 (6–10, 7–9) | 12 (8–16, 10–14) | 8 / 12 | 10 |
| 6.0 | 350 | 11 (8–13, 9–12) | 14 (10–18, 12–16) | 10.5 / 14 | 7 |

Greedy grouping cannot exceed ⌊span/resolution⌋ + 1 groups (4 at 20 µs and 7 at 10 µs for 1.2 cm), so below 2 cm the count is set by the spacing, not by the seating.

## 5. Notes

### 5.1 When a broadside talker and diffuse noise are spatially indistinguishable

- A broadside talker has Γs = 1; diffuse noise has Γn = sinc(2fd/c) (3-D) or J0 (2-D). The two differ by about (kd)²/6, which at 1 kHz is only 0.008 at 1.2 cm (0.087 at 4 cm, 0.035 at 2.5 cm).
- At 1.2 cm they are indistinguishable (coherence > 0.9) below 3.58 kHz in 3-D noise and 2.91 kHz in 2-D babble. The M4 conditioning rule |Γs − Γn| ≥ 0.2 is met only above 5.15 kHz (3-D) or 4.18 kHz (2-D). That leaves 5.5% of the speech band-importance.
- The broadside beam (= delay-and-sum) reaches 0.5 dB of DI only at 5.38 kHz and 1 dB at 7.68 kHz.
- Over 1.0–2.0 cm the coherence-0.9 edge is 4.29–2.15 kHz (3-D) and the CDR-rule edge 6.17–3.09 kHz. Both scale as 1/d.
- So for a talker near broadside, coherence, CDR, clustering and beams give essentially nothing across the speech band. Only spectral and speaker-identity cues remain.

### 5.2 White-noise amplification below 1 kHz at 1.2 cm

- **Smallest cost for a beam to be 'useful' (DI ≥ 3 dB, on the axis)**: WNG -24.2, -22.1, -18.2, -16.2, -12.3, -10.3, -8.3, -6.5 dB at 125, 160, 250, 315, 500, 630, 800, 1000 Hz. Self-noise and any uncorrelated noise (wind, vent turbulence, handling, vibration) are amplified by that magnitude, roughly 6–24 dB.
- **Specific beams at 125 / 250 / 500 / 1000 Hz**:
  - full hypercardioid: -31.7, -25.7, -19.7, -13.7 dB;
  - rear-null cardioid (the most WNG-friendly beam with a null): -28.2, -22.2, -16.2, -10.2 dB;
  - two-talker figure-8: -34.2, -28.2, -22.2, -16.2 dB.
  - Nulling a point source whose TDOA differs from the target's by Δτ costs WNG = 1 − cos(2πfΔτ). For the largest possible Δτ (70 µs, opposite ends of the axis) that is −10.2 dB at 1 kHz and −22.2 dB at 250 Hz.
- **What the floors allow** (on-axis DI at 125 / 250 / 500 / 1000 Hz): −10 dB floor 0.5, 1.1, 2.3, 4.6 dB; −20 dB floor 1.8, 3.7, 6.0, 6.0 dB; meeting −6 dB floor 0.3, 0.6, 1.3, 2.8 dB.
- Everything is 10.5 dB worse than at 4 cm (WNG ∝ d² at low frequency: 20·log10(4/1.2) = 10.46 dB).
- **In absolute terms** (DESIGN §1.2: INMP441 EIN ≈ −87 dBFS):
  - A −20 dB floor puts the amplified self-noise near −67 dBFS in those bins. That is still well below bus noise (−50 to −40 dBFS) and below this lab's fan noise (−51 dBFS), so the bus can use it.
  - A meeting room's HVAC (−85 to −75 dBFS) cannot. There the −3 to −6 dB floors leave the beam essentially omni below 1 kHz.
- **Calibration is the price of the −20 dB floor.** Uncalibrated ±1 dB / ±3° errors cost 0.80 dB of on-axis AI-DI at 1.2 cm (0.21 dB at 4 cm). Calibrated to ±0.25 dB / ±1°, the cost is 0.10 dB.

### 5.3 What spatial processing still contributes at 1.2 cm, and where

- **Single-target superdirective beam, talker within about ±45° of the axis.** At −10 dB, per-band DI ≥ 3 dB from about 700 Hz, about 6 dB from 1.6 to 4 kHz, and still 5.1 dB at 7.9 kHz. AI-DI: 4.69 dB (3-D), 3.89 dB (2-D). The bus schedule adds 0.13 dB. Main band: 0.8–8 kHz. Steered over azimuth (2-D): 2.22 dB.
- **Two-talker figure-8, partners on the axis on opposite sides.** The dipole works from 2052 Hz to 8 kHz at −10 dB, from 3296 Hz under the bus schedule, and from 644 Hz at −20 dB. Below the dipole band the best balanced beam gives under 1 dB below 1 kHz. AI-DI: 1.58 dB (2-D) / 2.45 dB (3-D); bus schedule 1.41 / 2.14 dB.
  - The −6 dB part of the bus schedule above 2 kHz pushes the dipole band up to 3.3 kHz. At small spacing a −10 dB floor up to about 3.3 kHz is the better trade when noise is far above self-noise.
- **CDR / coherence mask, M4.** Endfire partner: from 914 Hz (70% of the AI weight). Broadside partner: only above 5.15 kHz, which is effectively unusable.
- **TDOA, DOA and clustering (S1, M7).**
  - Gain: there is no spatial aliasing below 14.3 kHz, so all 0.1–8 kHz bins carry unwrapped phase. At 4 cm, TDOA estimation must stay below 3.4 kHz.
  - Cost: the whole TDOA range is ±35 µs (±0.56 samples). The 20 meeting seats give 4 groups at 20 µs (6 at 10 µs).
  - Bus partners at opposite ends of the axis are 70 µs apart and remain separable.
  - S1's fixed 25 µs 'inseparable' threshold is 0.71·d/c at 1.2 cm and should scale with d̂.
- **Point-interferer null (e.g. the fan).** Possible in any band, at WNG = 1 − cos(2πfΔτ). Affordable above about 1 kHz, or lower only in loud noise.
- **Not usable at all:** near-field ILD (0.34 / 0.21 / 0.10 dB at 0.3 / 0.5 / 1.0 m), and any spatial cue for broadside talkers below about 5 kHz.
- **Net.** At 1.2 cm, spatial processing is an above-1-kHz tool for talkers near the axis. Its broadband value is a few dB AI-DI (2.2–3.9 dB in 2-D), and it is almost unchanged from 4 cm for an aligned single talker. Below 1 kHz, and for broadside talkers, the gain must come from single-channel noise reduction, WPE, leveling and speaker identity (DESIGN §1.5), as before, but more so.

### 5.4 Skew versus spacing

- The measured skew of about 68 µs (1.1 samples) is 1.94 × the largest physical TDOA at 1.2 cm (2.33 × at 1.0 cm, 1.17 × at 2.0 cm).
- Uncompensated, every source appears outside the physical TDOA range, and a geometry-steered beam attenuates its own target: -6.17 dB AI-DI on the axis.
- Residual skew errors of 2 / 5 / 10 µs cost 0.05 / 0.15 / 0.39 dB on the axis at 1.2 cm, against 0.01 / 0.02 / 0.08 dB at 4 cm.
- Target: |residual skew| ≤ 5 µs (0.08 samples), i.e. ≤ 14% of d/c at 1.2 cm. RTF-based M3 absorbs a fixed skew; model-steered M1, M2 and M4 do not.

### 5.5 Consistency with the hardware reading (to be tested, not assumed)

- If the skew is −1.07 to −1.12 samples, the measured direct-path TDOAs (−0.53 to −0.56 samples) imply geometric TDOAs of +0.51 to +0.59 samples (32–37 µs).
- That needs d ≥ 1.09–1.26 cm (d/c ≥ τ). At d = 1.0 cm (maximum 0.47 samples) the reading is impossible. At 1.2 cm the speakers would be at 24° / 11° / beyond endfire (impossible) from the Mic-A end of the axis, for τ = 0.51 / 0.55 / 0.59 samples. At 1.36 cm they would be at 36° / 30° / 22°.
- This agrees with the coherence fits (1.09–1.36 cm), but the speaker geometry was not measured.
- The rival reading (skew −0.5 samples, speakers near broadside, geometric τ ≈ −0.05 samples) has its own geometric limit. The fans' measured −1.15 samples would then be a geometric −0.65 samples (40.6 µs). That is physical only if d ≥ 1.39 cm, above all but one of the fits (1.09–1.36 cm; one gave 1.74 cm) but inside the user's 1–2 cm. Under the −1.1 reading the fans sit near broadside (−0.05 samples) at any d.
- These are consistency limits from the maximum TDOA alone. A biased GCC-PHAT peak on reverberant fan noise would weaken them, so they point toward the −1.1 reading but do not settle it.

### 5.6 Uncertainty and caveats

- **Model.** Ideal point mics in free field, ideal diffuse fields, band-centre evaluation, no device-body scattering or field losses (−0.5 to −1.5 dB in practice).
- **Spacing uncertainty.** Between 1.0 and 1.5 cm the beam AI-DIs in the main table change by 0.22–0.65 dB. Band edges and the skew ratio scale as 1/d (−17% to +25% around 1.2 cm). The 1.2 cm row is a point estimate; the 1.0 and 1.5 cm rows bracket it.
- **TDOA groups** depend on the assumed resolution and the seat layout. Six seat layouts give identical medians at 1.0–2.0 cm and agree within one group at 2.5–6 cm.
- **Not determined here:** the actual spacing, skew and speaker directions. Those are data questions for the other spacing tasks.

