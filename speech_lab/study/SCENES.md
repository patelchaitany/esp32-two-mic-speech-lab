# Paired spacing scene set: 12 mm (current board) vs 40 mm

Owner: night task `spacing:scenes`, 2026-11-03. Code: `study/scene_spacing.py` (generator, build, validation),
`study/run.py` (bench runner and paired helpers). Scenes: `scenes3/spacing/` (15 MB). Results: `results3/spacing/`.
Scratch and evidence are in `tmp/night_spacing-scenes/`:
- logs: `build.log`, `validate.log`, `baseline.log`, `blind.log`;
- `s0_check.py` → `s0_check.json`;
- `summarize_validation.py` → `validation_summary.md`;
- `baseline_tables.py` → `baseline_tables.md`;
- `fallback_check.py`, `theory.py`, `rebuild_check/`.

Nothing under `lab/`, `methods/`, `scenes2/`, `results2/` or the docs was changed.

## Key findings (details and caveats below)

1. **The set is built and paired.** It has 48 scenes:
   - bus: 10 layouts × SNR −5 / 0 dB × 12 / 40 mm, 24 s each;
   - meeting: 4 rooms × 12 / 40 mm, 99–111 s each.

   The 24 recipe pairs are identical across spacings, and the image sources were checked identical. The paired
   input differs only by Mic A moving 14 mm: worst-partner input ESTOI changes by +0.003.
2. **The board's impairments are in every scene and validated.**
   - Mic B gain −1.300 dB and skew −63.12 µs, measured from the impaired over ideal renders.
   - Realized SNR is within −0.09 to +0.18 dB of target.
   - Noise coherence matches sinc(2fd/c) within ±0.01: at 4 kHz it is 0.87 at 12 mm and 0.07 at 40 mm.
3. **Physics of 12 mm.**
   - |τ| ≤ 35 µs, so the −63 µs skew is larger than any geometric TDOA.
   - The bench's 25 µs separability rule flags 6 / 10 layouts at 12 mm and 3 / 10 at 40 mm.
   - Theory: a 2-mic DAS gains ≤ 0.3 dB below 4 kHz at 12 mm. Superdirective gain costs about 10 dB more white
     noise than at 40 mm.
4. **Blind S0 cannot calibrate this board, on either spacing.** τ0 is 0 in 16 / 16 scenes, leaving −59 to
   −63 µs of skew, and d_hat is 12–40 mm on the 12 mm scenes. The factory calib_json leaves ≤ 0.06 µs.
5. **Baseline with the current methods and the factory calibration** (a snapshot; they are still being tuned).
   - **Gains.** 40 mm buys this much more noise attenuation (NA_active): +0.5 to +1.2 dB on the bus (Holm
     p ≤ 0.04) and +0.2 to +1.5 dB on the meeting. On the bus it adds +0.2 to +0.4 dB of ΔSNR_active, and on
     the meeting up to +0.02 mean ΔESTOI.
   - **No ESTOI gain on the bus.** Worst-partner ESTOI does not move: |Δ| ≤ 0.003.
   - **Fallbacks.** At 12 mm `m1_das` falls back to Mic A in 18 / 20 bus scenes and 4 / 4 rooms, and
     `m2_meeting_beam` in 4 / 4 rooms (0 / 4 at 40 mm). Much of the measured "spacing gain" of the beamformers is
     their own µs-based thresholds switching them off.

## What the set is for

Each layout is rendered twice, once with the board's 12 mm array and once with a 40 mm array. Everything else is
the same: room, every source position and directivity, RIR seed, speech, levels, noise seeds, impairments and
scene length. Only the two mic positions differ. Because the two renders form a pair, a per-layout
difference (40 mm − 12 mm) measures what moving the mics would buy, with the layout itself cancelled out.

| Scenario | Layouts | Variants per layout | Scenes | Length | Speech |
|---|---|---|---|---|---|
| bus | 10: 2 per E1 stratum (aligned, 45, broadside, side_by_side, lap_flat) | SNR −5 / 0 dB × 12 / 40 mm | 40 | 24 s | LibriSpeech, test split: partners from the 20 test-clean speakers, all passengers and babble from the 17 test-other speakers |
| meeting | 4 rooms | 12 / 40 mm | 8 | 99–111 s | LibriSpeech: the 20 test-split talkers (all of them, one per seat) |

- **Scene ids.** `spacing/bus_L<NN>_d<12|40>_snr<±S>` and `spacing/meeting_R<NN>_d<12|40>`. Geometries are
  `bus_spacing_L<NN>_d<mm>` and `meeting_spacing_R<NN>_d<mm>`. The corpus is `libri`; no TTS recipe exists.
- **Scene-card fields** for `scene2.list_scenes(..., root="scenes3")`: `layout_id` (the pairing key), `spacing_mm`,
  `stratum`, `snr_db`, `speaker_split`, and the usual bus or meeting geometry keys.
- **Seeds.** Bus layouts use seeds 61000–61009 and meeting rooms 481000–481003; no `scenes2` set uses these.
- **Do not tune on this set.** It uses the same test-split speakers that the bench reserves for final evaluation.
  It is a study set for measuring an effect, not for choosing parameters.

### Bus layouts (40 mm geometry; τ > 0: Mic B hears later; `inseparable` is the bench rule)

| layout | stratum | RT60 (s) | partners dist A / B (m) | fold A / B (°) | τ A / B at 12 mm (µs) | τ A / B at 40 mm (µs) | inseparable 12 / 40 mm | passengers | handling | road | DRR A / B (dB) | partner speakers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L00 | aligned | 0.442 | 0.46 / 0.55 | 20 / 18 | −32.9 / +33.3 | −109.7 / +111.0 | no / no | 21 | no | j0 | +4.0 / +3.6 | 1995, 1320 |
| L01 | 45 | 0.248 | 0.54 / 0.48 | 54 / 52 | −20.6 / +21.7 | −68.5 / +72.2 | no / no | 24 | no | sinc | +5.8 / +7.2 | 2300, 908 |
| L02 | broadside | 0.219 | 0.38 / 0.35 | 80 / 74 | +5.8 / −9.4 | +19.4 / −31.4 | close+broad / broad | 18 | yes | sinc | +8.9 / +9.3 | 1580, 672 |
| L03 | side_by_side | 0.168 | 0.75 / 0.62 | 23 / 53 | +32.3 / +21.0 | +107.6 / +69.9 | close / no | 19 | no | sinc | +5.4 / +6.0 | 6930, 908 |
| L04 | lap_flat | 0.253 | 0.79 / 0.58 | 87 / 83 | −1.8 / −4.2 | −6.1 / −13.8 | close+broad / close+broad | 18 | no | j0 | +3.5 / +3.5 | 3570, 4077 |
| L05 | aligned | 0.360 | 0.41 / 0.65 | 11 / 14 | −34.3 / +33.9 | −114.3 / +112.9 | no / no | 24 | yes | sinc | +7.5 / +4.1 | 5142, 1188 |
| L06 | 45 | 0.598 | 0.39 / 0.57 | 34 / 60 | +28.9 / −17.6 | +96.2 / −58.6 | no / no | 18 | yes | j0 | +6.0 / +2.0 | 3575, 6930 |
| L07 | broadside | 0.422 | 0.86 / 0.70 | 86 / 70 | −2.4 / +11.9 | −8.1 / +39.5 | close / no | 25 | no | j0 | −0.3 / +1.4 | 5142, 6930 |
| L08 | side_by_side | 0.544 | 0.94 / 0.92 | 79 / 44 | +6.7 / +25.0 | +22.2 / +83.5 | close / no | 24 | yes | j0 | −3.2 / −3.0 | 2300, 3570 |
| L09 | lap_flat | 0.471 | 0.53 / 0.90 | 82 / 88 | +4.7 / +1.1 | +15.5 / +3.7 | close+broad / close+broad | 25 | yes | sinc | −0.4 / −2.0 | 1188, 4970 |

"close" means |τ_A − τ_B| < 25 µs and "broad" means both |τ| < 0.3·d/c (the `inseparable_why` flags in the
geometry meta).

The bench's absolute 25 µs rule already shows the main cost of a small array:
- 6 of 10 layouts are spatially inseparable at 12 mm (L02, L03, L04, L07, L08, L09);
- 3 of 10 are at 40 mm (L02, L04, L09).

Design within each stratum:
- one replicate in each RT60 half (0.15–0.375 s and 0.375–0.6 s);
- handling in one replicate, diffuse road model `sinc` in one and `j0` in the other;
- opposed strata use 2 of the bench's distance-bin pairs.

All bus layouts use the synthetic tail (order-18 image sources, then a stochastic tail with sinc coherence at the
true spacing). The `ism` tail was not used, to keep the build fast.

### Meeting rooms

| room | RT60 (s) | tail | mic height | axis | presenter | HVAC (dBA) | talkers < 2 / 2–4 / > 4 m | nearest / farthest (m) | T30 (s) |
|---|---|---|---|---|---|---|---|---|---|
| R00 | 0.578 | synthetic | 1 cm | table | yes | 40 | 3 / 9 / 8 | 1.56 / 5.56 | 0.59 |
| R01 | 0.626 | ism | 2 cm | random | no | 35 | 4 / 11 / 5 | 1.41 / 5.98 | 0.92 |
| R02 | 0.702 | synthetic | 10 cm | table | no | 45 | 3 / 12 / 5 | 1.55 / 5.65 | 0.71 |
| R03 | 0.875 | synthetic | 2 cm | random | yes | 40 | 4 / 9 / 7 | 1.45 / 6.07 | 0.88 |

Rooms come from `lab.scene_meeting.room_layout`: a 12 × 7 × 3 m room (±8 %), an 8 × 2 m table, the E7 table
reflection, 20 talkers including a second row and a far corner, a fan and HVAC. The conversation uses
`target_s` = 100 and `min_turns` = 1, which gives 22 main turns per room. In R00, 6 of those 22 are
interruptions, and there are 4 back-channels besides. This is what makes a room about 100 s long. As a
consequence, at least 18 of the 20 talkers have a single main turn (see Limitations).

## How the pairing is guaranteed

**Bus.** `lab.scene_bus.sample_layout` draws each layout once, at 40 mm. Its lap-flat acceptance rule
(|τ| ≤ 0.28·d/c) depends on the spacing. For the 12 mm geometry, the mics are re-placed at ±6 mm about the same
centre, azimuth and elevation (`_replace_mics`). The spacing-dependent truth is then recomputed for 12 mm: τ,
fold, angle class and the inseparable flags.

**Meeting.** `lab.scene_meeting.room_layout` makes no spacing-dependent draws. The build asserts that the room,
table, talkers and seats are identical for both spacings.

**Both scenarios:**
- **Image sources.** Both spacings use the same RIR seed, so the randomized image sources are the same.
  - Checked on L05, sources 0–2: the image positions match exactly (max difference 0.0). RIR length and
    image-source length (`n_rir`, `n_x`) are also equal.
  - Mic A's stochastic tail is the same noise realization: `coherent_noise` channel 0 does not depend on d.
    Only its energy match to the image-source part can differ slightly.
- **Recipes.** They are made by `bus_recipe` / `meeting_recipe`, thin wrappers around the bench generators. The
  bus wrapper sets the transient source levels from the 40 mm geometry's band gains, kept in meta `acoustics`;
  the 12 mm geometry's own measurement is in `acoustics_own`. The meeting wrapper recomputes the scene length
  from the array centre. **All 24 recipe pairs are identical** except for the ids and the spacing in the calib
  truth (`scenes3/spacing/build_log.json`, `recipe_identity`).
- **SNR rule.** The rule is evaluated per scene on that scene's own Mic A, so the rule gain differs slightly
  between the two spacings: by −0.15 to +0.17 dB (median −0.02).

## Impairments on every scene (the real board, hw/HW_TESTS.md)

| Item | Truth in every recipe (`scene["meta"]["impair"]`) |
|---|---|
| Skew | Mic B −63.125 µs = −1.01 samples (B early; the current boot after 8 resets) |
| Gain | A 0 dB, B −1.3 dB, broadband. The board's per-band spread (−0.7 to −1.9 dB) is **not** modelled |
| High-pass | First-order, corners drawn per mic in 50–80 Hz exactly as the bench draws them (same seeds for both spacings) |
| Sensor | 33 dBA self-noise per mic (core). The board's incoherent noise below 60 Hz (~−35 dBFS) is **not** modelled |

**Factory calibration.** Each scene carries a calib_json holding its truth, in
`scene["meta"]["recipe_meta"]["factory_calib"]` and in `scenes3/spacing/calib/<scene name>.json`:

```json
{"version": 1, "spacing_m": 0.012001, "tau0_s": -6.3125e-05, "hpf_corner_hz": [73.01, 63.531],
 "gain_db": [1.3, ... 18 values: correction applied to X_B], "note": "factory calibration = scene truth ..."}
```

There is no `model`: the noise field belongs to the room, not to the device. `spacing_m` is the µm-rounded
geometry value (0.012001 or 0.04).

## Validation

Every scene was rendered with impairments and with ideal mics, and the facts below were measured from the signals:
- all 48 scenes, 197 s with 2 workers, no error;
- records: `study/scene_spacing_validation.json`, written by `study/scene_spacing.py validate`;
- tables: `tmp/night_spacing-scenes/validation_summary.md`.

**Levels and SNR (bus, 20 scenes per spacing)**

| spacing | realized − target SNR (dB) | ideal mics | all-inclusive SNR − target | rule gain (dB) | partners (dBA) | rule noise (dBA) | peak \|mix\| |
|---|---|---|---|---|---|---|---|
| 12 mm | −0.09 … +0.18 (median +0.08) | 0.000 | −10.0 … 0.0 (median −1.2) | −15.1 … +4.9 (median −7.2) | 60.5 … 77.2 (median 70.5) | 61.4 … 79.6 (median 72.5) | 0.047 … 0.180 |
| 40 mm | −0.09 … +0.18 (median +0.08) | 0.000 | −11.1 … 0.0 (median −1.1) | −15.1 … +4.7 (median −7.1) | 60.5 … 77.1 (median 70.4) | 61.4 … 79.5 (median 72.6) | 0.046 … 0.188 |

Paired 40 − 12 mm differences per scene:
- realized SNR: −0.005 to 0.000 dB;
- partner level: −0.16 to +0.13 dB;
- rule-noise level: −0.13 to +0.13 dB.

The all-inclusive SNR (which adds transients, handling and sensor) differs by −1.9 to +2.6 dB between the
spacings. Transients stay at their physical level, and a tonal brake squeal at Mic A changes when Mic A moves by
14 mm (room-mode interference). The source levels are identical.

**Impairment chain** (transfer function of the impaired mix over the ideal mix, Welch; HPF magnitude and phase
removed): Mic A gain 0.000 dB and delay 0.00 µs; Mic B gain −1.300 dB and delay −63.12 µs. This holds in every
one of the 48 scenes.

**Noise coherence.** The table gives Re γ after the known chain is removed (median over scenes):

| field | spacing | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 3 kHz | 4 kHz | 6 kHz |
|---|---|---|---|---|---|---|---|---|
| bus rule noise (passengers, babble, engine/road) | 12 mm | 1.00 | 1.00 | 0.99 | 0.97 | 0.92 | 0.87 | 0.72 |
| meeting HVAC (diffuse) | 12 mm | 1.00 | 1.00 | 0.99 | 0.97 | 0.93 | 0.87 | 0.73 |
| model sinc(2fd/c) | 12 mm | 1.00 | 1.00 | 0.99 | 0.97 | 0.93 | 0.88 | 0.73 |
| bus rule noise | 40 mm | 0.99 | 0.98 | 0.91 | 0.66 | 0.32 | 0.03 | −0.23 |
| meeting HVAC (diffuse) | 40 mm | 0.99 | 0.98 | 0.91 | 0.67 | 0.36 | 0.07 | −0.21 |
| model sinc(2fd/c) | 40 mm | 0.99 | 0.98 | 0.91 | 0.68 | 0.37 | 0.07 | −0.22 |

- **Meeting HVAC.** It lies within −0.009 to +0.010 of the sinc model at both spacings.
- **Bus rule noise.** It follows sinc at both spacings, although half the layouts use the `j0` road model:
  reverberant passengers and engine dominate the rule noise.
- **12 mm.** The noise is almost fully coherent up to 3 kHz (≥ 0.92), so there is little for coherence-based
  noise reduction to work with.
- **For comparison.** The board's measured diffuse-field fit gave 1.09–1.36 cm (hw/HW_TESTS.md).

**Partner TDOA.** The table compares the truth (τ + skew) with the phase slope of each partner's own
component (300–3000 Hz, known HPF phase removed), at 0 dB:

| spacing | stratum | τ truth (µs) | τ + skew (µs) | measured − truth (µs) |
|---|---|---|---|---|
| 12 mm | aligned | −34.3 … +33.9 | −97.4 … −29.2 | −8.6 … +13.0 |
| 12 mm | 45 | −20.6 … +28.9 | −83.7 … −34.3 | −8.1 … +5.1 |
| 12 mm | broadside | −9.4 … +11.9 | −72.6 … −51.3 | −2.3 … +4.4 |
| 12 mm | side_by_side | +6.7 … +32.3 | −56.5 … −30.8 | −12.4 … −0.9 |
| 12 mm | lap_flat | −4.2 … +4.7 | −67.3 … −58.5 | −1.4 … −0.1 |
| 40 mm | aligned | −114.3 … +112.9 | −177.5 … +49.8 | −24.3 … +39.6 |
| 40 mm | 45 | −68.5 … +96.2 | −131.6 … +33.0 | −22.9 … +14.5 |
| 40 mm | broadside | −31.4 … +39.5 | −94.5 … −23.6 | −1.4 … +12.6 |
| 40 mm | side_by_side | +22.2 … +107.6 | −41.0 … +44.5 | −59.4 … −1.7 |
| 40 mm | lap_flat | −13.8 … +15.5 | −77.0 … −47.6 | −4.6 … +0.1 |

The |error| is a median 3.7 µs at 12 mm (max 13) and 8.1 µs at 40 mm (max 59). The estimator is a crude
weighted phase slope, and reverberation and phase wrapping at 40 mm limit it. The exact check is the chain fit
above. At 12 mm the skew (−63 µs) is larger than any geometric TDOA (|τ| ≤ d/c = 35 µs). Uncorrected, every
partner therefore appears beyond endfire on Mic B's side.

**S0 blind vs factory** (`tmp/night_spacing-scenes/s0_check.py` → `s0_check.json`; 12 bus scenes at 0 dB, 4
meeting scenes). The "residual" columns are the correction times the true chain over 200–4000 Hz; 0 means
perfect.

| | blind S0 | factory calib_json |
|---|---|---|
| τ0 | 0 in 16 / 16 (allowed values {0, +31.25 µs}; fitted slope −22 to −93 µs) | −63.125 µs |
| residual delay | −59 to −63 µs (the whole skew stays) | ≤ 0.06 µs |
| residual gain | −0.11 to +0.22 dB | −0.12 to +0.24 dB (the json gain plus S0's blind drift, ≤ ±1 dB by design) |
| d_hat on 12 mm scenes (8) | 31.2, 18.0, 12.0, 18.0, 29.8, 12.0 mm (bus; "12.0" = the prior, kept on a poor fit); 33.6, 40.3 mm (meeting) | 12.0 mm |
| d_hat on 40 mm scenes (8) | 40.0 mm (prior) on all 6 bus scenes; 64.6 and 54.1 mm on the meetings | 40.0 mm |

Blind S0 cannot represent this board: its D_RANGE is 1.5–8 cm and the skew is restricted to {0, +0.5 sample}.
Use `--calib factory` for spacing comparisons. `--calib blind` measures the deployment risk instead.

**Meeting scenes** (identical for both spacings, except the TDOA):

| room | length (s) | talkers with < 1 s / < 3 s alone | alone time min / median (s) | overlap | speech / noise in band (dBFS) | nearest-talker TDOA error at 12 / 40 mm (µs) |
|---|---|---|---|---|---|---|
| R00 | 106.5 | 1 / 8 | 0.8 / 3.6 | 0.151 | −64.7 / −78.0 | −2.4 / −6.5 |
| R01 | 104.0 | 1 / 5 | 0.9 / 3.4 | 0.132 | −63.0 / −77.3 | +6.2 / +13.3 |
| R02 | 99.2 | 1 / 4 | 0.8 / 3.7 | 0.118 | −64.1 / −74.2 | +3.0 / +9.6 |
| R03 | 110.7 | 0 / 4 | 1.1 / 4.0 | 0.126 | −63.5 / −78.0 | +5.8 / +15.6 |

Render time is 2–5 s per bus scene and 3–4.3 s per meeting under load (about 190).

## Running methods

`lab.bench2` can address the set (`--set spacing --scenes-root scenes3 --corpus libri`). This was checked: one
scene with M0 went into `tmp/night_spacing-scenes/r_bench2/` and matched `study/run.py`. But bench2 hard-codes
`info = {scenario, spacing_m: 0.04}`, so it would tell 12 mm methods the wrong spacing and give no calibration.
Use `study/run.py` instead, from the lab folder. `-I` is needed because enrollment reads `data/`.

```
OMP_NUM_THREADS=1 .venv/bin/python -I study/run.py run --methods m0_mica_hpf,m2_bus_beam --scenario bus -j 2
        [--spacing 12,40] [--snr -5,0] [--strata aligned] [--layouts L00,L05] [--calib factory|blind|both]
        [--prior 0.04] [--enroll] [--extra none|subset|all] [--skip-existing] [--save-audio] [--limit N]
.venv/bin/python -I study/run.py table   [--scenario bus] [--labels a@factory,b@factory] [--by stratum,snr]
.venv/bin/python -I study/run.py spacing [--scenario bus] [--ref m0_mica_hpf@factory] [--metrics d_estoi_min,dsnr_active]
.venv/bin/python -I study/run.py compare --spacing 12 [--ref m0_mica_hpf@factory]
```

Typical use: what does 40 mm buy for method `<m>`?

```
OMP_NUM_THREADS=1 .venv/bin/python -I study/run.py run --methods m0_mica_hpf,<m> -j 2      # both scenarios, both spacings
.venv/bin/python -I study/run.py spacing --labels m0_mica_hpf@factory,<m>@factory --ref m0_mica_hpf@factory
```

**Same as `lab.bench2`:**
- the Hagerman runs (`metrics2.plan`) and scoring (`metrics2.score`);
- the method code-policy check (`bench2.load_method`);
- the oracle side channel;
- the result format and the `rerun_db` check.

**What differs:**
- **Prior.** `info["spacing_m"]` is the scene's nominal spacing, 0.012 or 0.04. `--prior <m>` forces a value.
- **Calibration.** `--calib factory` (the default) adds `info["calib_json"]` (the scene's factory calibration).
  `blind` lets S0 run blind. `both` runs each method twice.
- **Enrollment.** `--enroll` adds `info["enroll"] = {"A", "B"}`: the bus partners' reserved LibriSpeech enrollment
  samples (`lab.corpus_libri.enrollment_audio`, first sample), for `m9_voxlock2`.
- **Labels.** Results are labelled `<method>@<calib>` (plus `-p<mm>` with a forced prior) and written to
  `results3/spacing/<label>/<scene>.libri.json`, so modes never mix.
- **Extra runs.** They default to `none`, because the study needs only the 5 bus / 2 meeting runs.
- **`table`** prints one bench2 table per spacing; gates are evaluated within a spacing.
  - With only 10 bus geometries, the aligned-stratum G1 criteria read n/a.
  - The meeting has 4 rooms, under the 5 geometries a CI verdict needs, so every meeting CI criterion reads n/a.
  - Crossed breakdown keys are `spacing_x_stratum` and `spacing_x_snr`.
  - `python -m lab.bench2 table --set spacing --results-root results3 --scenes-root scenes3 --corpus libri` also
    reads these results; it was checked, and it mixes the two spacings unless you add `--by spacing`.
- **`spacing`** gives paired per-layout differences, 40 mm − 12 mm, through `lab/stats.py`:
  - each layout is first averaged over its SNRs;
  - percentile bootstrap CI, stratified by E1 stratum on the bus;
  - Wilcoxon p, Holm-corrected across labels.
  - With `--ref`, it also gives the interaction [label − ref](40) − [label − ref](12): what the larger spacing
    buys beyond the reference.
  - Python: `run.spacing_pairs(rows, label, key)` and `run.spacing_vs_ref(rows, label, ref, key)`.
  - With n = 10 bus layouts the smallest two-sided Wilcoxon p is 0.002; with n = 4 rooms it is 0.125, so meeting
    p-values cannot reach 0.05.
- **Cost under load (about 150–190).** Measured on the baseline pass with 2 workers:
  - full bus pass: 12.5 min for 4 cheap methods. That is about 40 s per bus scene and worker, of which scoring
    is about 8–10 s per method;
  - full meeting pass: 4.5 min for 4 methods, about 65 s per room and worker;
  - one method alone on all 48 scenes: 7.4 min (`m4_cdr`, blind pass).

## What to expect from theory (2 omni mics, spherically diffuse noise)

Computed by `tmp/night_spacing-scenes/theory.py`.

| f (Hz) | spacing | sinc coherence | DAS (A+B)/2 broadside gain vs diffuse (dB) | endfire MVDR (superdirective) gain vs diffuse (dB) | its white-noise gain (dB) |
|---|---|---|---|---|---|
| 500 | 12 mm | 0.998 | +0.00 | +6.02 | −19.7 |
| 1000 | 12 mm | 0.992 | +0.02 | +6.01 | −13.7 |
| 2000 | 12 mm | 0.968 | +0.07 | +5.96 | −7.8 |
| 4000 | 12 mm | 0.876 | +0.28 | +5.79 | −2.2 |
| 500 | 40 mm | 0.978 | +0.05 | +5.98 | −9.3 |
| 1000 | 40 mm | 0.913 | +0.19 | +5.86 | −3.6 |
| 2000 | 40 mm | 0.679 | +0.76 | +5.37 | +1.2 |
| 4000 | 40 mm | 0.071 | +2.71 | +3.33 | +3.0 |

- **Plain beamforming.** It gains almost nothing at 12 mm below 4 kHz, and up to +2.7 dB at 40 mm.
- **Superdirectivity.** It promises about 6 dB at both spacings. At 12 mm it pays about 10 dB more white-noise
  amplification, 20·log10(40/12).
- **Where that matters.** The INMP441 self-noise is 33 dBA per mic and is in every scene.
  - In the quiet meeting rooms (noise floor 42–49 dBA, per BENCH2's meeting validation), a WNG of −14 to −20 dB
    lifts the self-noise to the room noise level or above it, so superdirective gain at 12 mm is
    self-noise-limited there.
  - On the bus (rule noise about 72 dBA) there is room for it.
- **Phase-based methods.** Masks, CDR and the TDOA trackers see a 3.3 × smaller phase range at 12 mm (|τ| ≤ 35 µs).
  The board's skew (−63 µs) is then larger than the whole physical TDOA range, so a calibration is mandatory.

## Baseline pass (factory calibration, nominal prior)

**Read this as a snapshot, not a verdict.**
- **Methods.** `m1_das`, `m2_bus_beam`, `m2_meeting_beam` and `m4_cdr`, as of 02:16. Another run is still tuning
  them.
- **Version check.** No file they import changed during the factory passes (bus 02:18:19–02:30:49, meeting
  02:30:49–02:35:20). Checked by mtime, with `tmp/night_spacing-scenes/method_snapshot_start.txt`.
  - The blind `m4_cdr` pass (02:35–02:42) imported its modules at 02:35. `_core/masks.py` changed at 02:38, but
    pool workers keep the version they first imported.
  - **Since the passes**, `m1_das.py`, `m2_bus_beam.py`, `_core/beams.py` (02:40), `_core/masks.py` (02:38) and
    `_core/rtf.py` (02:43) have changed. Re-run with `--skip-existing` off to get current numbers.
- **Commands.**
  - bus: `study/run.py run --methods m0_mica_hpf,m1_das,m2_bus_beam,m4_cdr --scenario bus --calib factory -j 2`;
  - meeting: the same with `m2_meeting_beam`.
- **Files.** Log: `tmp/night_spacing-scenes/baseline.log`. Results: `results3/spacing/<label>/`. Tables:
  `tmp/night_spacing-scenes/baseline_tables.py` and `study/run.py spacing`.

### Bus (10 layouts × 2 SNRs per spacing; 160 results, 0 errors)

| label | spacing | wrstESTOI in | wrstΔESTOI | ΔSNRact (dB) | NAact (dB) | ΔSIRnear (dB) | TAmin (dB) | NM (dB) | RTF |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | 12 mm | 0.232 | +0.001 | +0.00 | +0.12 | −0.02 | −0.17 | +0.04 | 0.000 |
| m0_mica_hpf | 40 mm | 0.235 | +0.000 | +0.00 | +0.12 | −0.02 | −0.17 | +0.04 | 0.000 |
| m1_das | 12 mm | 0.232 | +0.000 | +0.01 | +0.13 | −0.02 | −0.18 | +0.04 | 0.031 |
| m1_das | 40 mm | 0.235 | −0.001 | +0.38 | +0.61 | −0.35 | −0.36 | −0.41 | 0.029 |
| m2_bus_beam | 12 mm | 0.232 | −0.001 | +0.15 | +0.29 | −0.03 | −0.21 | −0.12 | 0.034 |
| m2_bus_beam | 40 mm | 0.235 | −0.002 | +0.55 | +0.84 | −0.34 | −0.44 | −0.50 | 0.034 |
| m4_cdr | 12 mm | 0.232 | −0.001 | +0.15 | +1.33 | +0.17 | −1.33 | −0.09 | 0.038 |
| m4_cdr | 40 mm | 0.235 | +0.002 | +0.36 | +2.49 | +0.46 | −2.41 | −0.11 | 0.036 |

Paired 40 − 12 mm per layout, n = 10. The CI is a 95% bootstrap stratified by E1 stratum; p (Holm) is the
Wilcoxon p, Holm-corrected across the 4 labels.

| metric | m1_das | m2_bus_beam | m4_cdr | m0_mica_hpf |
|---|---|---|---|---|
| worst-partner ΔESTOI | −0.001 [−0.002, +0.001] | −0.001 [−0.003, +0.002] | **+0.003 [+0.001, +0.004]**, p 0.11 | −0.000 |
| ΔSNR_active (dB) | **+0.37 [+0.10, +0.65]**, p 0.20 | **+0.41 [+0.06, +0.75]**, p 0.32 | +0.21 [−0.04, +0.46], p 0.39 | −0.00 |
| NA_active (dB) | **+0.49 [+0.20, +0.77]**, p 0.012 | **+0.55 [+0.14, +0.96]**, p 0.039 | **+1.17 [+0.99, +1.34]**, p 0.008 | −0.00 |
| ΔSIR_near (dB) | −0.33 [−0.67, +0.00] | −0.31 [−0.65, +0.04] | **+0.28 [+0.19, +0.38]**, p 0.08 | −0.00 |
| TA_min (dB) | **−0.18 [−0.26, −0.10]** | **−0.23 [−0.37, −0.09]** | **−1.08 [−1.19, −0.96]** | +0.00 |

Reading:
- **Input side.** The worst-partner input ESTOI is 0.232 at 12 mm and 0.235 at 40 mm. Moving Mic A by 14 mm
  changes it by only +0.003 [+0.001, +0.004], so the pairs are fair.
- **What 40 mm buys today.**
  - Noise attenuation during partner speech improves by +0.5 dB (beamformers) to +1.2 dB (CDR).
  - ΔSNR_active improves by +0.2 to +0.4 dB.
  - Worst-partner ESTOI does not move: |Δ| ≤ 0.003, against a G1 bar of +0.03.
  - At 40 mm the beamformers lose 0.2 dB more target level (TA_min). They also lose about 0.3 dB of
    near-passenger rejection, but that CI touches 0.
- **Effective spacing for M1 is not the array alone.** `m1_das` falls back to Mic A, with output metrics identical
  to M0, in 18 / 20 scenes at 12 mm but only 7 / 20 at 40 mm (`tmp/night_spacing-scenes/fallback_check.py`).
  Its absolute thresholds (S1's 25 µs separability rule, partner peaks within 1.1·d/c) switch it off on a small
  array. Its "spacing gain" is therefore partly a design threshold, not physics. `m2_bus_beam` falls back in
  1 / 20 and 2 / 20 scenes, `m4_cdr` in none.
- **Magnitude.** All gains are small, about 0.5 dB. The theory table says why: a 2-mic DAS gains at most 0.2–0.8 dB
  at 1–2 kHz even at 40 mm.

### Meeting (4 rooms per spacing; 32 results, 0 errors)

| label | spacing | wrstESTOI in | wrstΔESTOI | meanΔESTOI | farΔESTOI | NAact (dB) | T2 far−near (dB) | NM (dB) | RTF |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | 12 mm | 0.345 | −0.000 | +0.001 | +0.001 | +0.03 | −0.01 | +0.00 | 0.001 |
| m0_mica_hpf | 40 mm | 0.345 | +0.000 | +0.001 | +0.001 | +0.03 | −0.01 | +0.00 | 0.001 |
| m1_das | 12 mm | 0.345 | −0.000 | +0.001 | +0.001 | +0.03 | −0.01 | +0.00 | 0.054 |
| m1_das | 40 mm | 0.345 | +0.001 | +0.006 | +0.008 | +0.25 | −0.08 | −0.21 | 0.047 |
| m2_meeting_beam | 12 mm | 0.345 | −0.000 | +0.001 | +0.001 | +0.03 | −0.01 | +0.00 | 0.052 |
| m2_meeting_beam | 40 mm | 0.345 | +0.004 | +0.023 | +0.023 | +0.92 | −0.33 | −0.86 | 0.049 |
| m4_cdr | 12 mm | 0.345 | +0.010 | +0.010 | +0.010 | +1.94 | −0.14 | +0.59 | 0.072 |
| m4_cdr | 40 mm | 0.345 | +0.003 | +0.020 | +0.019 | +3.40 | −0.56 | +0.31 | 0.065 |

Paired 40 − 12 mm per room, n = 4. With 4 rooms the bootstrap CI is only indicative, and no Wilcoxon p can fall
below 0.125.

| metric | m1_das | m2_meeting_beam | m4_cdr |
|---|---|---|---|
| mean ΔESTOI | +0.006 [+0.005, +0.007] | +0.022 [+0.019, +0.026] | +0.009 [+0.007, +0.012] |
| > 4 m ΔESTOI | +0.007 [+0.004, +0.009] | +0.023 [+0.012, +0.032] | +0.009 [+0.004, +0.014] |
| worst-talker ΔESTOI | +0.001 [−0.002, +0.003] | +0.004 [+0.000, +0.008] | −0.007 [−0.014, +0.000] |
| NA_active (dB) | +0.22 [+0.12, +0.30] | +0.89 [+0.63, +1.07] | +1.46 [+1.26, +1.66] |
| T2 far − near (dB) | −0.07 [−0.11, −0.04] | −0.32 [−0.61, −0.06] | −0.41 [−0.68, −0.14] |

Reading:
- **At 12 mm both beamformers are Mic A.**
  - `m2_meeting_beam` is metric-identical to M0 in 4 / 4 rooms at 12 mm and in 0 / 4 at 40 mm.
  - `m1_das` is identical to M0 in all 4 rooms at 12 mm, and in none at 40 mm.
  - Their whole "spacing gain" is therefore the gain of switching on. Mean ΔESTOI rises by +0.02 for
    `m2_meeting_beam`, about the G1 bar (+0.02), and NA_active by +0.9 dB.
  - Why they stay off at 12 mm was not investigated. The docstrings name the likely causes: S1 localization
    thresholds in µs, and "Mic A alone" when a talker is not localized.
- **`m4_cdr` works at both spacings.** At 40 mm it removes about 1.5 dB more noise and doubles the mean ΔESTOI
  (+0.010 → +0.020). Its worst-talker ΔESTOI drops, −0.007, a CI that touches 0. Far talkers lose 0.4 dB relative
  to near ones (T2), still far from the −3 dB G0 limit.
- **Bottom line on the current methods.** 40 mm buys about +0.5 to +1.5 dB of noise attenuation and up to +0.02
  mean ESTOI. Most of the beamformers' benefit comes from not falling back to Mic A. Spacing-aware thresholds,
  for example in units of d/c rather than µs, might recover part of it at 12 mm. That is a methods question this
  set can now answer with paired evidence.

### Blind S0 instead of the factory calibration (`m4_cdr@blind`, all 48 scenes, 0 errors)

Paired per geometry within a spacing; `study/run.py compare --spacing <mm> --ref m4_cdr@factory`. Values are
blind minus factory.

| scenario | spacing | worst ΔESTOI | ΔSNR_active (dB) | NA_active (dB) | mean ΔESTOI (meeting) |
|---|---|---|---|---|---|
| bus | 12 mm | −0.001 [−0.001, +0.000] | −0.02 [−0.05, −0.00] | −0.14 [−0.15, −0.13] | |
| bus | 40 mm | **−0.007 [−0.009, −0.005]** | −0.08 [−0.12, −0.04] | −0.13 [−0.16, −0.09] | |
| meeting | 12 mm | −0.004 [−0.005, −0.002] | −0.14 [−0.17, −0.11] | −0.24 [−0.26, −0.21] | −0.002 [−0.002, −0.002] |
| meeting | 40 mm | −0.008 [−0.017, +0.000] | −0.21 [−0.25, −0.17] | −0.29 [−0.34, −0.22] | −0.005 [−0.007, −0.003] |

- **CDR is robust to the calibration failure.** It is mostly DOA-free, and blind S0 leaves the −63 µs skew
  uncorrected and gets d wrong. Still, the factory calibration is worth 0.1–0.3 dB of noise attenuation and up to
  about 0.008 of worst-partner or worst-talker ESTOI at 40 mm.
- **The spacing effect survives blind S0 for noise attenuation**: NA_active +1.18 [+1.01, +1.34] dB, the same as
  with the factory calibration.
- **It does not survive for the bus worst-partner ESTOI.** Blind gives −0.004 [−0.007, −0.001] at 40 mm − 12 mm,
  against +0.003 with the factory calibration.
- **Not run.** The beamformers were not run blind. They steer on S1 TDOAs, so the uncorrected skew should hurt them
  more.

## Limitations and deviations

1. **Corners and gain are idealized.**
   - The B gain is broadband, −1.3 dB; the board's per-band spread of ±0.6 dB is not modelled.
   - The skew is the current boot's −1.01 samples, the same in every scene. Other boots measured −1.09 to −1.11.
   - The incoherent noise below 60 Hz is not modelled.
   - The factory calib therefore knows the truth exactly. A real factory calibration would carry measurement
     error.
2. **Meeting length versus coverage.**
   - Reaching ~100 s needed `min_turns` = 1, so at least 18 of the 20 talkers speak in exactly one main turn
     (1–6 s utterances, plus back-channels).
   - Per room, 0–1 talker has < 1 s alone and is skipped by the metrics; 4–8 talkers have < 3 s alone.
   - Per-talker ESTOI is therefore noisy, and the T5 returning-talker figures are sparse.
   - For bench-like coverage, set `MEETING_TARGET_S` = 180 and `MEETING_MIN_TURNS` = 3 in `study/scene_spacing.py`
     and rebuild the rooms (`build --only R00,R01,R02,R03`). This gives about 250 s per room.
3. **Small n.**
   - The bus has 10 layouts (2 per stratum), so per-stratum results are anecdotal.
   - The meeting has 4 rooms, so Wilcoxon p ≥ 0.125.
   - The set measures the average spacing effect. It is not powered for per-stratum claims.
4. **Synthetic tails.** All bus layouts and 3 of the 4 rooms use the synthetic tail, whose diffuse coherence is
   exactly sinc at the true spacing. Only R01 uses the full image-source tail. A real bus cabin's late field may
   be less diffuse.
5. **Bus levels.** These are inherited from the bench (`lab.scene_bus`), including its "found, not fixed" item 1:
   near passengers are scaled with the crowd by the SNR rule and are quiet (BENCH2.md "Harness verification").
6. **Not physical: the device.** There is no housing, PCB or body. Moving mics on a real board changes their
   housing diffraction, and these scenes do not model that.
7. **The 25 µs inseparable rule** in the geometry meta is the bench's absolute threshold, not a physical limit. At
   12 mm it flags 6 of 10 layouts.
8. **Reproducibility.**
   - The stored recipes (`*.libri.recipe.json`) and geometries are the reference.
   - Rendering uses `lab.scene2`, so it follows any change there, exactly like `scenes2/`.
   - A rebuild, or a corpus switch, calls `lab.scene_bus` / `lab.scene_meeting` (`make_recipe`, `sample_layout`,
     `room_layout`, `_row`, `_room_truth`), whose owners may still change them. A rebuild then gives the same
     scenes only if those functions are unchanged.
   - Checked at 02:27: rebuilding L04 (lap-flat, whose placement rule depends on the spacing) and R02 into
     `tmp/night_spacing-scenes/rebuild_check/` gave 20 / 20 files identical to `scenes3/spacing/`: geometry JSON,
     RIR stores, recipes and calib files.
