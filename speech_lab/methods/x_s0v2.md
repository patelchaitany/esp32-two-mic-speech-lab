# x_s0v2: blind self-calibration that works on the real board

Run `s0v2`, 2026-11-03 02:01–02:50. Experimental (`x_` prefix). Nothing under `methods/_core` was changed.
Code: `methods/x_s0v2.py`. Wrappers: `methods/x_auto_<m>.py`. Validation scripts: `tmp/night_s0v2/*.py`.
Results: `results3/s0v2/*.json`.

## 1. What it is and how to use it

```python
from methods import x_s0v2
cj = x_s0v2.estimate(x, 16000, info)        # x (2, T) raw (DC kept is fine); info: spacing_m prior, scenario
fe = tracker.front_end(x, 16000, dict(info, calib_json=cj))   # S0 override: everything below is used
```

`estimate()` returns a `calib.json` dict (API.md schema), measured on the input itself:
`version`, `spacing_m` (clipped to [1, 10] cm because `calib.load_json` rejects other values), `tau0_s` (continuous,
+ = B late), `hpf_corner_hz` (`[60, 60]` = no corner phase unless identified), `model`, `rho`,
`gamma_n` (`{f_hz, gamma}`, see §4.1 for why it matters), `band_hz` + `gain_db` (18 bands, correction applied to X_B
= −(B re A)), `confidence`, and the diagnostics under `x_s0v2` (raw skew, SEs, band slopes, both model fits, raw
per-band B re A, per-parameter confidences, fallbacks taken). CLI: `.venv/bin/python -m methods.x_s0v2 file.wav`.

Wrappers `x_auto_{m1_das, m2_bus_beam, m2_meeting_beam, m3_rtf_mvdr, m4_cdr, m5_fence, m7_cacgmm, m10_wpe}`: if
`info` has no `calib_json`, they call `x_s0v2.estimate(x, fs, info)` on this call's own `x` and pass it on; otherwise
they defer. Deterministic and stateless (each Hagerman run re-estimates from its own input). `m10_wpe` never reads
S0, so its wrapper changes nothing (kept for a uniform list). `bus_final` / `meeting_final` did not exist when this run
finished, so no wrappers were made for them. Cost: 0.3–1.2 s per 12 s file (RTF ≈ 0.03–0.1 under load 140–190, with
jackknife).

Also added:
- `x_s0v2.gamma_n_for(x, fs, cfg)`: the noise coherence measured after an *existing* json's own correction.
- Wrappers `x_gnfix_{m4_cdr, m2_bus_beam, m3_rtf_mvdr}`: they keep a given `calib_json` (for example the hand
  `calib_session.json`) and only add that `gamma_n` when it is missing (finding §4.1). Without a json they are the
  plain method.

## 2. Design

1. **Prefilter:** DC removal and a 4th-order 90 Hz Butterworth high-pass, identical on both rows. It removes the
   < 60 Hz per-mic junk at −35 dBFS and leaves the inter-channel transfer function unchanged. STFT = `common.stft`.
2. **Noise frames:** valid frames whose 76 ms-smoothed level is within 3 dB of the 4 s sliding minimum in both
   200–1000 Hz and 1–6.5 kHz. A 100 ms guard is placed around every rejected frame (onsets, reverberant tails). The
   threshold is relaxed to 4.5 and then 6 dB if fewer than 3 s qualify. The frames are then split into ≤ 24 blocks of
   ≥ 1 s (with less than 3 s of noise: 3 blocks of ≥ 0.2 s, so the jackknife still works), and pooled cross- and
   auto-spectra are kept per block.
3. **Tonal bins:** bins > 4 dB above the 9-bin running median spectrum (fan and engine tones, usually directional),
   plus their neighbours, are excluded from the fits.
4. **Skew (continuous, ±2.5 samples):**
   - Coarse: argmax over τ of Re Σ Γ(f) e^{−j2πfτ} over 150–1200 Hz. There is no wrapping, and no sign change of the
     diffuse coherence for d ≤ 10 cm in either model.
   - Fine: weighted Huber least squares on the phase of Γ·sign(M)·e^{−j model} over 150–6500 Hz, with weights = inverse
     phase variance from min(|M|, smoothed |Γ|), capped at |Γ| = 0.98.
   - Jointly with an HPF-corner term (∂/∂Δ of atan(f_A/f) − atan(f_B/f) ≈ f/(f² + 60²)). The corner is kept only if
     |Δ| ≥ 3 SE and the phase at 125 Hz is ≥ 1°.
5. **Spacing (0.6–10 cm):** free-ρ (0.5–1) grid fit of ρ·sinc and ρ·J0 to Re Γ after the skew and corner are
   removed, over 400–6500 Hz.
   - Weights 1/max(1 − Γ², 0.1)², two Huber re-weightings, parabolic refinement on a 0.1 mm grid.
   - J0 is chosen only if its weighted residual is below 0.8 × sinc's (prior toward a 3-D field).
   - Why 400 Hz: below ~300 Hz the board's per-mic LF junk lowers the coherence to 0.4–0.9.
   - Why 6.5 kHz: the INMP441 decimation pass band ends at 0.423 fs = 6.77 kHz.
6. **Gain:** per band, 10 log10(ΣP_A/ΣP_B) over the noise frames. Bands with jackknife SE > 0.3 dB (in practice
   125–200 Hz) are shrunk toward the median of the reliable bands.
7. **Robustness:**
   - Blocks whose skew or 4–6.5 kHz coherence is beyond 5 MAD (and beyond 0.1 sample / 0.05) are dropped.
   - Jackknife standard errors are computed over blocks.
   - **Phase-linearity gate:** per-band skews over 500–6500 Hz (bands with |M| ≥ 0.3). A spread above 0.5 samples
     means a directional noise component, and the skew is declared unidentifiable.
8. **Fallbacks:**
   - Skew → 0 if: < 1 s of noise, SE > 0.15 samples, a bound was hit, phase RMS > 30°, or the slope spread is
     > 0.5 samples.
   - Spacing → `info['spacing_m']` if: < 1.5 s of noise, fit RMS > 0.15 (calib.py's threshold), a grid edge was hit,
     or SE > 25 %.
   - Gain: omitted (calib.py's blind gain is then used) with < 1 s of noise.
   - Fewer than 60 noise frames (~0.5 s): the prior, skew 0, no gain.
9. **gamma_n:** as calib.py step 6 (measured over the noise frames after **this** correction, 5-bin smoothing, clipped
   to [−0.3, 0.98], blended with ρ·model with weight N/(N+200)). It is shipped in the json; §4.1 explains why.
10. **Confidence:** a geometric mean of three per-parameter terms:
    - skew: noise-time factor × exp(−(SE/0.05)²) × exp(−(phase RMS/15°)²) × exp(−(spread/0.3)²);
    - spacing: noise-time factor × exp(−(SE/0.1d)²) × exp(−(fit RMS/0.1)²);
    - gain: from the median band SE.

    The noise-time factor is min(1, noise s / 3).

Options:
- `info['x_s0v2_skew_candidates'] = [0, 0.5]` snaps the skew to the nearest allowed value (calib.py's one-bus I2S
  rule). Off by default, because the board's skew is not on that lattice.
- `info['x_s0v2_head_s'] = s` estimates from the first s seconds only (boot-time or causal calibration). On reboot_04,
  with the full 12 s estimate at −1.021 / 1.26 cm:

  | Input | Skew (samples) | Spacing |
  |---|---|---|
  | 1 s | −1.021 | not used (0.78 s of noise < 1 s) |
  | 1.5 s | −1.020, used | prior (1.28 s of noise < 1.5 s) |
  | 2 s | −1.024 | 1.26 cm, conf 0.55 |
  | 3 s | −1.026 | 1.27 cm, conf 0.88 |

## 3. Validation

### 3.1 Real board (`results3/s0v2/real_board.json`, `tmp/night_s0v2/val_real.py`; prior 4 cm, nothing else given)

Reference values (main agent): first boot −1.09 to −1.11 samples; current boot −1.01 (±0.01); d = 1.19 cm
(1.09–1.36); B re A ≈ −1.3 dB (direct sound).

| Files | skew (samples): median [min, max] | max \|skew − ref\| | d (cm): median [min, max] | B re A, median over bands (dB) | conf (median) | noise used (s) |
|---|---|---|---|---|---|---|
| First boot: hw/rec/* + rec2/s2_* (13; ambient, sweeps, talk, played bus noise) | −1.118 [−1.123, −1.059] | 0.041 (re −1.10) | 1.30 [1.26, 1.66] | −0.67 [−0.75, −0.57] | 0.74 | 4.8 |
| Current boot, ambient: reboot_00–07, s3_ambient_after (9) | −1.020 [−1.039, −1.017] | 0.029 | 1.25 [1.22, 1.26] | −0.69 [−0.80, −0.63] | 0.96 | 9.6 |
| Current boot, realset takes, recording = speech + noise (18) | −1.007 [−1.025, −0.950] | 0.060 | 1.28 [1.09, 1.41] | −0.67 [−0.77, −0.59] | 0.76 | 3.7 |
| Current boot, realset takes, noise only (18) | −1.016 [−1.024, −1.007] | 0.014 | 1.23 [1.09, 1.29] | −0.66 [−0.77, −0.56] | 0.82 | 4.5 |
| Current boot, s3_sweep_L + drift_00–02 (4; drift = the main agent's 10-min ambient watch, made during this run) | −1.009 [−1.021, −1.005] | 0.011 | 1.60 [1.21, 2.01] | −0.66 [−0.69, −0.62] | 0.54 | 4.9 |

All 62 recordings (`real_board.json`) give a skew within 0.06 samples of the reference for their boot.

**Quiet fans:** drift_01–03 (02:15–02:35, after the machine load dropped) are 15 dB quieter: −56 to −59 dBFS
against −41. There the skew stays right (−1.005 to −1.017), but the spacing reads **1.61–2.01 cm**. Two causes:
- **Mic self-noise:** in a quiet room it lowers the coherence at high frequencies, which the fit reads as a larger d.
- **A less diffuse field:** fit RMS 0.11–0.14 against 0.027 on the loud reboot files.

A self-noise-aware fit, Re Γ ≈ ρ·M(f; d)·(1 − σ²/P(f)), was tried (`tmp/night_s0v2/selfnoise.py`,
`results3/s0v2/selfnoise.txt`):

| File | plain d (cm) | σ² fitted blind (cm) | σ² from the datasheet, −61 dB/bin (cm) |
|---|---|---|---|
| reboot_03, reboot_05 (loud) | 1.25, 1.25 | 1.25, 1.25 (picks σ² = 0) | 1.23, 1.24 |
| drift_01 | 2.01 | 1.64 | 1.86 |
| drift_02 | 1.59 | 1.41 | 1.51 |
| drift_03 | 1.61 | 1.44 | 1.54 |

The blind fit helps without hurting the loud files, but its valley is flat: on drift_02 the RMS only moves
0.100 → 0.096. It is therefore not enabled.

**Use:** in quiet rooms, trust the S0v2 skew, but take the spacing from a stored device value (1.19 cm, or this
estimator's loud-noise value 1.25 cm).

- **Skew:** right on all 62 recordings to ±0.06 samples (±0.03 on noise-only recordings). Every current-boot file is
  ≥ 0.9 samples away from both values calib.py allows (0 and +0.5).
  - The largest errors are the takes with speech leaking into the noise frames: A1 −0.968 ± 0.064, M5 −0.950 ± 0.036
    and M2 −0.975. Their jackknife SEs are 5–30× the clean ones, and their skew confidence is 0.16–0.74.
  - Pooled current-boot calibration (`results3/s0v2/board_calib_current_boot.json`, 74 s of noise): −1.021 ± 0.001
    samples, d = 1.26 cm sinc, ρ = 1.00. The main agent's `hw/realset/calib_session.json` has −1.015.
  - Pooled first-boot calibration (`board_calib_first_boot.json`): −1.125 ± 0.002.
- **The skew is defined only to about ±0.02 samples on this board.** The per-band phase slopes are not constant, and
  the pattern is identical after all 8 resets:

  | Band (Hz) | 150–500 | 500–1k | 1–2k | 2–3k | 3–4k | 4–5k | 5–6.5k |
  |---|---|---|---|---|---|---|---|
  | Current boot, pooled (samples) | −0.937 | −0.993 | −1.019 | −1.029 | −1.032 | −1.037 | −1.000 |
  | First boot, pooled | −1.054 | −1.102 | −1.116 | −1.133 | −1.155 | −1.162 | −1.082 |

  Either the two mics differ slightly in phase (frequency-dependent), or the fan field has a small fixed directional
  part. The data cannot tell which. The residual is ≤ 2° anywhere below 6.5 kHz, so a single skew is adequate. Which
  band weighting you choose explains the 0.005–0.02 difference between estimators (the main agent's unweighted
  150–7000 Hz complex fit gives −1.002 to −1.029 per file; this estimator gives −1.017 to −1.039).
- **Spacing:** 1.22–1.26 cm on the current-boot ambient files and 1.20–1.29 cm on noise only, against the reference
  1.19 cm (1.09–1.36), with sinc chosen everywhere.
  - Directional sound in the "noise" biases d upward: 1.53–1.66 cm on talkAB_LR, talkA_noise, noise_LR and
    talkC_R_quiet, where played speech or bus noise from the MacBook speakers is part of the noise frames.
  - Mechanism: a source at TDOA ≈ +0.54 samples rotates the coherence phase after skew removal, Re Γ falls, and the
    fit reads that as more decorrelation.
- **Gain:** the diffuse-field B re A is −0.67 dB (all 62 files within −0.56 to −0.80). Per band (reboot median,
  150–500 / 500–1k / 1–2k / 2–4k / 4–7k): −0.49 / −0.75 / −0.71 / −0.45 / −0.88 dB.
  - This is 0.3–1 dB smaller in magnitude than the direct-sound value (−0.8/−1.3/−0.7/−1.2/−1.9).
  - The broadband B re A of the take *images* is −0.20 to −1.11 (median −0.78).
  - So the B deficit depends on direction: the sources sit near endfire on Mic A's side, and B is possibly shadowed by
    the package or board. A diffuse-noise S0 measures the direction-averaged sensitivity. That is the right value for
    Γ_n, but it differs from the partners' RTF gain (RTF methods estimate that themselves).
- **HPF corners:** not identifiable on the board (no file passed the |t| ≥ 3 and ≥ 1° test), so `[60, 60]` is used. On
  synthetic data, 50/80 Hz corners are recovered as Δ = −29.7 Hz (true −30).

### 3.2 Seconds of noise needed (`duration_real.json`, `synth_duration.json`)

| Noise window (s) | Real board, 9 ambient files × 2–4 windows: P90 \|skew − own median\| | P90 \|skew + 1.01\| | P90 \|d/median − 1\| | P90 \|d/1.19 cm − 1\| | Synthetic 1.2 cm diffuse field: P90 \|skew err\| | P90 \|d err\| |
|---|---|---|---|---|---|---|
| 0.5 | no estimate (< 60 noise frames after the gate) | | | | no estimate | |
| 1 | 0.015 | 0.021 | 4.4 % | 6.4 % | 0.004 | 1.4 % |
| 2 | 0.014 | 0.018 | 4.1 % | 6.3 % | 0.004 | 2.1 % |
| 4 | 0.012 | 0.017 | 2.9 % | 5.9 % | 0.002 | 2.5 % |
| 8 | 0.008 | 0.018 | 2.4 % | 6.1 % | 0.002 | 2.1 % |
| 11–16 | 0.010 | 0.021 | 1.4 % | 6.1 % | 0.001 | 2.0 % |

**Answer:**
- **Skew to ±0.05 samples:** 1 s of near-diffuse noise suffices (P90 0.02, which includes the ±0.01 definition
  ambiguity above).
- **Spacing to ±10 %:** 1 s of noise (P90 4–6 %).
- **Practical minimum:** about 1–1.5 s of window, because 30–80 % of the board's fan-noise frames pass the
  level gate. The skew is used from 1.0 s of gated noise on, the spacing from 1.5 s.
- **Above 1 s, more data does not help much.** The spread is limited by the field (fan level changes and non-diffuse
  parts), not by estimation variance. The jackknife SE (0.001–0.005) therefore under-reports the true run-to-run
  spread (0.01–0.02).

### 3.3 Synthetic diffuse noise (`synth_grid.json`; 20 s, fan-like spectrum, white self-noise −30 dB, 60 files)

Grid: d 0.8/1.2/2/4/6 cm × skew −1.1/−0.5/0/+0.5 samples × B gain ±1 dB; sinc in all cells, J0 at skews −1.1 and +0.5.

| Field | d (cm) | n | max \|skew err\| (samples) | d error | model right | max \|gain err\|: median / any band ≥ 250 Hz (dB) | min conf |
|---|---|---|---|---|---|---|---|
| sinc | 0.8 | 8 | 0.001 | +2.5 … +4.1 % | 8/8 | 0.00 / 0.04 | 1.00 |
| sinc | 1.2 | 8 | 0.001 | +0.9 … +2.0 % | 8/8 | 0.00 / 0.07 | 1.00 |
| sinc | 2 | 8 | 0.002 | +0.1 … +0.8 % | 8/8 | 0.01 / 0.07 | 0.99 |
| sinc | 4 | 8 | 0.007 | −0.5 … +0.3 % | 8/8 | 0.02 / 0.13 | 0.98 |
| sinc | 6 | 8 | 0.011 | −0.2 … +0.2 % | 8/8 | 0.02 / 0.15 | 0.97 |
| J0 | 0.8 | 4 | 0.001 | +25.5 … +26.6 % | 0/4 | 0.00 / 0.05 | 1.00 |
| J0 | 1.2 | 4 | 0.001 | +24.0 … +24.8 % | 0/4 | 0.01 / 0.05 | 1.00 |
| J0 | 2 | 4 | 0.004 | +0.0 … +25.4 % | 3/4 | 0.00 / 0.09 | 0.99 |
| J0 | 4 | 4 | 0.004 | −0.1 … +0.6 % | 4/4 | 0.01 / 0.10 | 0.97 |
| J0 | 6 | 4 | 0.012 | −0.1 … +0.3 % | 4/4 | 0.02 / 0.12 | 0.97 |

Extra cases at the board's geometry (1.2 cm, skew −1.01, `synth_extra.json`):

| Case | skew error | d | Comment |
|---|---|---|---|
| Corners 50/80 Hz and 80/50 Hz | −0.001 | 1.22 cm | corner Δ recovered (−29.7 / +29.8 Hz) |
| White self-noise −15 dB re field | −0.002 | **1.67 cm (+39 %)** | coloured field + white self-noise cannot be told from a larger d |
| Directional component q = 0.1, τ_d = +0.5 | +0.052 | 1.29 cm | bias ≈ q·τ_d, confidence stays 1.0 (not detectable) |
| Directional component q = 0.3, τ_d = +0.5 | +0.160 | 1.33 cm | same |
| Directional component q = 0.1, τ_d = −0.5 | −0.053 | 1.28 cm | same |

### 3.4 Simulated dev scenes (`scenes_dev.json`; impairments on; truth from `meta.impair`, `meta.spacing_m`)

| Scene | d true | skew true | x_s0v2 skew | ok | used | calib.py τ0 (slope) | d fit / used (cm) | calib.py d | B re A true | x_s0v2 | calib.py |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bus_g00_snr+0 | 2.5 | 0 | −0.221 | 0 (spread) | 0.000 | 0 (+0.13) | 2.59 / 2.59 | 2.63 | +1.50 | +1.65 | +1.60 |
| bus_g03_snr+0 | 2.5 | 0.5 | +0.584 | 1 | +0.584 | 0.5 (+0.64) | 2.78 / 2.78 | 2.23 | +0.11 | −0.01 | +0.04 |
| bus_g06_snr+0 | 4 | 0 | +0.076 | 1 | +0.076 | 0 (+0.05) | 3.29 / 3.29 | 4.18 | −0.10 | −0.11 | −0.12 |
| bus_g09_snr+0 | 4 | 0.5 | +0.565 | 1 | +0.565 | 0.5 (+0.43) | 2.78 / 4.00 | 4.00 | −0.27 | −0.28 | −0.28 |
| bus_g12_snr+0 | 6 | 0 | +0.053 | 1 | +0.053 | 0 (+0.15) | 6.38 / 4.00 | 4.00 | −1.49 | −1.51 | −1.58 |
| bus_g15_snr+0 | 2.5 | 0.5 | +0.428 | 1 | +0.428 | 0.5 (+0.73) | 2.53 / 2.53 | 2.48 | −0.86 | −0.80 | −0.85 |
| bus_g21_snr+0 | 4 | 0 | **−0.527** | 1 | −0.527 | 0 (−0.46) | 4.50 / 4.50 | 3.77 | −0.25 | −0.25 | −0.28 |
| bus_g27_snr+0 | 6 | 0.5 | +0.539 | 1 | +0.539 | 0.5 (+0.55) | 6.15 / 4.00 | 4.00 | +0.51 | +0.51 | +0.55 |
| meeting_r00 | 2.5 | 0 | +0.181 | 1 | +0.181 | 0 (+0.17) | 2.35 / 2.35 | 2.98 | +0.44 | +0.46 | +0.43 |
| meeting_r05 | 2.5 | 0 | +0.154 | 1 | +0.154 | 0 (+0.14) | 2.66 / 2.66 | 2.75 | +0.91 | +0.96 | +0.95 |
| meeting_r10 | 6 | 0 | −0.013 | 1 | −0.013 | 0 (−0.07) | 3.44 / 4.00 | 6.59 | −1.15 | −1.03 | −1.06 |
| meeting_r15 | 6 | 0.5 | **+0.851** | 1 | +0.851 | 0.5 (+0.78) | 3.26 / 4.00 | 6.26 | +0.34 | +0.19 | +0.27 |

**Larger run: 50 dev scenes** (all 30 bus geometries at 0 dB and all 20 meetings; `results3/s0v2/scenes_dev50.json`):

| Quantity | Bus (30) | Meeting (20) |
|---|---|---|
| x_s0v2 skew used, \|err\| median / P90 / max (samples) | 0.082 / 0.476 / 0.527 | 0.161 / 0.366 / 0.506 |
| … within 0.05 / 0.1 / 0.25 samples | 7 / 16 / 20 of 30 | 2 / 5 / 13 of 20 |
| Phase-linearity gate fired (skew → 0) | 5 (right in 3) | 1 (wrong: truth 0.5) |
| calib.py two-value rule right | 27/30 | 20/20 |
| x_s0v2 + `skew_candidates=[0, 0.5]` right | 26/30 | 18/20 |
| d fit \|err\| median / P90 (cm) | 0.35 / 0.93 | 0.27 / 2.56 |
| d used (with 4 cm fallback) \|err\| median / P90; fallbacks | 0.42 / 2.00; 9 | 0.27 / 2.00; 4 |
| calib.py d \|err\| median / P90 | 0.44 / 2.00 | 0.37 / 0.58 |
| d fit median at true 2.5 / 4 / 6 cm | 2.48 / 3.37 / 6.26 | 2.66 / 3.63 / 4.38 |
| B re A \|err\| median / max (dB): x_s0v2 vs calib.py | 0.064 / 0.164 vs 0.040 / 0.150 | 0.046 / 0.154 vs 0.025 / 0.098 |

**Summary of the dev scenes:**
- **Skew:** on simulated scenes the continuous skew is clearly worse than calib.py's two-value rule. calib.py's
  premise (skew ∈ {0, 0.5}) is exactly the truth of these scenes, and their noise frames carry directional energy that
  adds its own phase slope:
  - bus: engine, passengers;
  - meetings: the HVAC point source and the talkers' reverberant tails.

  calib.py's raw slope is off by similar amounts (−0.46 … +0.78 in the 12-scene table). Rounding to the lattice
  hides that error.
- **Gate:** the phase-linearity gate catches only a few of these cases (6 of 50 fired, 3 rightly).
- **Spacing:** fits are good at 2.5 cm and better than calib.py's in the meetings at 2.5 and 4 cm. They are biased
  low in the 6 cm meetings (4.38 against calib.py's 6.29).
- **Spacing fallback threshold** (decision on stored fits, |err| median / P90 in cm):

  | FIT_RMS_MAX | Bus | Meeting | Board | Prior used |
  |---|---|---|---|---|
  | 0.10 | 1.50 / 2.00 | 0.22 / 2.00 | 0.08 / 0.33 | 25 bus, 7 meetings, 4 board |
  | **0.15 (kept)** | 0.42 / 2.00 | 0.27 / 2.00 | 0.08 / 0.26 | 9 bus, 4 meetings, 0 board |
  | 0.20 | 0.33 / 1.30 | 0.27 / 2.00 | 0.08 / 0.26 | 5 bus, 2 meetings, 0 board |
  | 0.30 | 0.35 / 0.93 | 0.27 / 2.56 | 0.08 / 0.26 | none |

  calib.py's 0.15 is kept: 0.2 looks slightly better, but it would be tuned on the evaluation scenes themselves.
- **Gain:** within 0.17 dB of the truth everywhere. Slightly noisier than calib.py, which uses the median over frames
  where this uses pooled sums.

### 3.5 Effect inside the front end and on enhancement

- **Front end on real takes** (`wrappers_smoke.json`):
  - Blind S0 gives d_hat 4.0 cm (prior) and τ0 = 0, leaving an inter-channel noise phase of −20° at 1 kHz and −126° at
    5–6 kHz.
  - With x_s0v2 the residual is +1.0° / −0.4° (M1) and −1.5° / −8.4° (A1, the speech-leaking take).
  - All 8 wrappers run on a bus take and a meeting take with finite outputs of the right length (0.5–3 s each).
- **Real-scene A/B** (`hw/realeval.py`, 6 bus scenes at 0 dB, Hagerman-scored): see §3.6.

### 3.6 Real-scene A/B (hw/realset bus scenes, SNR 0 dB, L1–L3 × recorded/quiet device noise)

Scored by `lab.metrics2` through `hw/realeval.py` (all Hagerman runs). The x_* results are in
`hw/realset/results/<variant>/<method>/`. The plain-method baselines from this run were moved to
`hw/realset/results_s0v2_baselines/`, so the runner's `--skip-existing` cache never reuses them after re-tuning.
Variants:
- `measured`: hand `calib_session.json`;
- `wrong4cm`: 4 cm prior, no json, blind calib.py S0;
- `x_auto_* @ wrong4cm`: 4 cm prior, x_s0v2 estimates everything from the scene (fully blind);
- `x_gnfix_* @ measured`: the hand json plus a correctly rotated `gamma_n`.

Mean over the 6 scenes:

| Method @ variant | worst ΔESTOI | worst ΔSDR (dB) | ΔSNR active (dB) | NA active | ΔSIR pax (dB) | RDA p10 | Hagerman closure (dB) | NM gap |
|---|---|---|---|---|---|---|---|---|
| m2_bus_beam @ measured | −0.007 | −0.2 | −0.1 | +0.1 | −0.1 | −0.2 | 55 | +0.0 |
| m2_bus_beam @ wrong4cm (blind calib.py) | +0.000 | −0.0 | +0.0 | +0.1 | +0.0 | −0.0 | 150 | +0.0 |
| **x_auto_m2_bus_beam @ wrong4cm** | −0.010 | −0.4 | **+0.3** | +0.4 | −0.5 | −1.9 | 20 | +0.3 |

Per scene, x_auto ΔSNR active: +0.68 / +0.41 / −0.02 / −0.08 / +0.79 / −0.06 (L1 rec / L1 quiet / L2 rec / L2 quiet
/ L3 rec / L3 quiet). measured: −0.11 … −0.08. wrong4cm: −0.01 … +0.04.

Reading:
- At 1.19 cm, m2_bus_beam is nearly transparent in every variant.
- With x_s0v2 it removes a little more noise: +0.3 dB on average, and +0.7 / +0.8 dB in two of the three scenes with
  the recorded device noise (L1, L3). It also distorts the partners slightly more (worst ΔESTOI −0.010, worst ΔSDR
  −0.4 dB, ΔSIR toward passengers −0.5 dB).
- **Hagerman closure drops to 20 dB**, because each signed run re-estimates its own calibration. The estimate is
  deterministic per input, but the + and − mixes are different inputs. If the bench requires high closure, estimate
  once on the unsigned mix and pass the json to every run.

CDR (Γ_n-heavy) and the `gamma_n` repair alone, same 6 scenes:

| Method @ variant | worst ΔESTOI | worst ΔSDR (dB) | TA min | ΔSNR active (dB) | NA active (dB) | ΔSIR pax | RDA p10 | closure (dB) | NA gap | NM gap |
|---|---|---|---|---|---|---|---|---|---|---|
| m4_cdr @ measured | −0.007 | +0.1 | −1.8 | +0.0 | +1.8 | −0.1 | −0.6 | 54 | +1.8 | +0.0 |
| **x_gnfix_m4_cdr @ measured** (hand json + correct `gamma_n`) | **−0.005** | +0.2 | −2.0 | **+0.2** | **+2.1** | −0.1 | −0.7 | 55 | +2.3 | +0.2 |
| **x_auto_m4_cdr @ wrong4cm** (fully blind x_s0v2) | −0.008 | +0.2 | −2.0 | +0.1 | +2.1 | −0.2 | −0.7 | 54 | +2.2 | +0.2 |
| m2_bus_beam @ measured | −0.007 | −0.2 | −0.3 | −0.1 | +0.1 | −0.1 | −0.2 | 55 | +0.1 | +0.0 |
| **x_gnfix_m2_bus_beam @ measured** | **+0.000** | −0.0 | −0.1 | +0.1 | +0.1 | +0.0 | −0.1 | 107 | +0.2 | +0.1 |

Per-scene ΔSNR(active) for m4_cdr, measured → gnfix:

| Scene | measured | gnfix |
|---|---|---|
| L1 recorded | +0.07 | +0.31 |
| L1 quiet | −0.07 | −0.05 |
| L2 recorded | +0.11 | +0.36 |
| L2 quiet | +0.02 | +0.05 |
| L3 recorded | +0.13 | +0.36 |
| L3 quiet | −0.01 | −0.00 |

All 6 scenes improve. The largest gains (+0.23 to +0.25) are in the three scenes with the recorded device noise, which
is the near-diffuse field whose coherence was wrong.

Reading:
- **The repaired `gamma_n` is a small, consistent gain for CDR.** Worst ΔESTOI and ΔSDR are not worse, but the
  target is attenuated 0.2 dB more (TA min −1.8 → −2.0 dB), as expected from slightly stronger suppression. It also
  removes m2_bus_beam's slight distortion under the hand json.
- **CDR, fully blind:** the x_s0v2 path matches the hand calibration plus the repair within 0.1 dB on every metric.
- **m2_bus_beam, fully blind:** it removes more noise than hand plus repair (ΔSNR active +0.3 against +0.1 dB), but
  distorts more (worst ΔESTOI −0.010 against 0.000, worst ΔSDR −0.4 against −0.0 dB) and has the lower closure. Its
  spacing (1.2–1.4 cm) and per-run skew differ slightly from the hand json; this is not a clear win.
- **The absolute effects are small** (≤ 0.4 dB ΔSNR). At 1.19 cm, a 2-mic spatial stage has little to work with
  whatever the calibration.
- **Not run:** m3_rtf_mvdr, m5_fence, m7_cacgmm, and the meeting scenes in this A/B (time).

## 4. Findings worth acting on

1. **`gamma_n` in a json without `gamma_n` is corrupted** (core bug or gotcha; not fixed here because `_core` is off
   limits). `calib._override` rebuilds Γ_n from `diag['gamma_n_meas']`, which is the blind measurement rotated by the
   *blind* correction (skew 0 or 0.5).

   On this board the true skew is −1.02, so the measured coherence is rotated by up to 2π·f·1.02/16000. Measured on
   take A1 (`results3/s0v2/gamma_n_check.txt`), Γ_n at 0.5 / 1 / 2 / 3 / 4 / 5 / 6 kHz:

   | Calibration | 0.5 kHz | 1 kHz | 2 kHz | 3 kHz | 4 kHz | 5 kHz | 6 kHz |
   |---|---|---|---|---|---|---|---|
   | Blind, or hand `calib_session.json` (the same numbers) | 0.99 | 0.96 | 0.80 | 0.62 | 0.36 | 0.06 | 0.41 |
   | x_s0v2 json (with `gamma_n`) | 0.99 | 0.98 | 0.96 | 0.91 | 0.84 | 0.79 | 0.57 |
   | sinc at 1.19 cm | 1.00 | 0.99 | 0.97 | 0.93 | 0.88 | 0.81 | 0.74 |

   Every MVDR, CDR or Γ_n-based method in the `measured` variant therefore uses a wrong noise model above ~1.5 kHz.
   Repairing only that (`x_gnfix_m4_cdr`) improves m4_cdr's ΔSNR(active) on 6/6 real bus scenes (mean +0.0 → +0.2
   dB, §3.6).

   Fixes, any of:
   - put `gamma_n` into `calib_session.json` (copy it from `results3/s0v2/board_calib_current_boot.json`);
   - pass the x_s0v2 json;
   - or have `_override` recompute `gm` from `st.Sn` with the json correction (core owner).
2. **The board skew is not on calib.py's {0, +0.5} lattice** (−1.02 current boot, −1.12 first boot), so blind S0
   always applies a wrong skew there: −1 sample of phase ramp, −126° at 6 kHz. x_s0v2 measures it from ~1 s of room noise:
   ±0.03 on noise-only recordings, and ±0.06 in all 62 recordings including those with speech.
3. **The diffuse-field B gain is −0.67 dB, not −1.3 dB.** The −1.3 dB direct-sound value applies to sources near
   endfire on Mic A's side.

## 5. Tried and rejected

- **Mixture model** Γ = e^{j2πfτ0}[(1 − q)ρM + q e^{j2πfτ_d}] (grid over q and τ_d, `tmp/night_s0v2/mixture.py`).
  - It fixed some directional cases: synthetic d = 4 cm 0.175 → 0.047; g21 −0.527 → +0.009; g06; g03.
  - But it overfits the board's non-ideal field: current boot −1.02 → −1.15, first boot −1.12 → −1.58.
  - And it worsened 5/8 bus scenes (g00 → +0.36, g09 → −0.51 error).
  - Residual ratios do not separate good from bad fits. Not used.
- **Level gate with only a 2 dB threshold:** too few frames on the board's fluctuating fan noise; 3 dB is used.
- **A 200 Hz lower edge for the spacing fit:** the board's LF junk then dominated the unweighted residual. 400 Hz is
  used.

## 6. Limits

- **Directional noise biases the skew by about q·τ_d** (power fraction × its TDOA), and the frequency-flat case is not
  detectable from one field.
  - On the board (near-diffuse fans): < 0.03 samples.
  - 50 simulated dev scenes: bus median 0.08 / P90 0.48 / max 0.53 samples; meeting 0.16 / 0.37 / 0.51.
  - Use the lattice option when the platform guarantees a quantized skew. Otherwise, calibrate on the device in
    diffuse noise and check that the slope spread and SE are small.
- **Speech leaking into the noise frames:** the skew is biased toward the talker's TDOA (takes A1, M2, M5: +0.04 to
  +0.07). The jackknife SE grows and flags it (confidence 0.16–0.74).
- **Small spacing: sinc and J0 cannot be told apart below about 2 cm** (they differ by < 0.006 in coherence up to
  6.5 kHz after rescaling). The default sinc then reads a 2-D field 24–27 % too large. The fitted Γ_n shape is still
  right, so Γ_n-based methods are unaffected; only `spacing_m` is.
- **Self-noise and quiet rooms:** white self-noise under a sloped field is confounded with a larger d.
  - Synthetic: +39 % at −15 dB.
  - Real board, after the fans slowed by 15 dB: 1.61–2.01 cm instead of 1.19–1.25.
  - With the fans loud this is negligible.
  - The skew is not affected (−1.005 to −1.017 in the quiet takes).
  - A blind σ² term recovers only part of the error (§3.1). The meeting room, the quiet use case, needs a stored
    device spacing.
- **Large spacing (4–6 cm) in reverberant simulated scenes:** the fit RMS often exceeds 0.15 and the 4 cm prior is
  used (calib.py does better on the 6 cm meetings).
- **HPF corners:** only their difference is identifiable, and only with strong, coherent LF noise. Not on the board.
- **Not done:**
  - online / causal version, apart from the `x_s0v2_head_s` option; the wrappers re-estimate from the whole input;
  - dev bus scenes at ±5 dB (50 of 110 dev scenes were run);
  - test scenes;
  - m3, m5, m7 and the meeting scenes in the real-scene A/B;
  - a test on `bus_final` / `meeting_final` (they did not exist).

## 7. Reproduce

```
LAB=...; cd "$LAB"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python tmp/night_s0v2/val_real.py             # results3/s0v2/real_board.json
.venv/bin/python tmp/night_s0v2/val_duration_real.py    # results3/s0v2/duration_real.json
.venv/bin/python tmp/night_s0v2/val_synth.py grid|extra|duration
.venv/bin/python tmp/night_s0v2/val_scenes.py dev/bus_g00_snr+0 ...   # results3/s0v2/scenes_dev.json
S0V2_OUT=scenes_dev50.json .venv/bin/python tmp/night_s0v2/val_scenes.py $(cat tmp/night_s0v2/scene_ids_50.txt)
.venv/bin/python tmp/night_s0v2/board_calib.py          # results3/s0v2/board_calib_{current,first}_boot.json
.venv/bin/python tmp/night_s0v2/test_wrappers.py        # results3/s0v2/wrappers_smoke.json
.venv/bin/python tmp/night_s0v2/mixture.py <scene ids>  # rejected mixture-model experiment (prints only)
.venv/bin/python tmp/night_s0v2/selfnoise.py            # results3/s0v2/selfnoise.txt (via tee)
.venv/bin/python -I hw/realeval.py run --methods x_auto_m2_bus_beam,x_auto_m4_cdr --variant wrong4cm --set bus --scenes snr+0 -j 1
.venv/bin/python -I hw/realeval.py run --methods m2_bus_beam,m4_cdr,x_gnfix_m2_bus_beam,x_gnfix_m4_cdr --variant measured --set bus --scenes snr+0 -j 1
.venv/bin/python -I hw/realeval.py table --variant measured --set bus --methods m4_cdr,x_gnfix_m4_cdr
```
Logs of this run: `tmp/night_s0v2/*.log`.
