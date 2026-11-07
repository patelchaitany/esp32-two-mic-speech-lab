# Real-device evaluation set (hw/realset)

Built 2026-11-03 from the ESP32 takes in `rec/`, `img/`, `stim/` (recorded by the main agent; nothing here plays,
records or opens the serial port). Owner of the files below: run `real:build`.

| Path | What |
|---|---|
| `calib_session.json` | Board calibration for the current boot, as an `info['calib_json']` dict |
| `calib_session_report.json` | Every estimate behind it and the checks |
| `scenes/` | 18 bus + 3 meeting scenes (components stored once at unit gain; scenes = gains in `scenes.json`), `enroll.npz`, `check.json` |
| `build/calib_session.py`, `build/realscenes.py`, `build/s0_check.py` | Builders and the S0/S1 check (`.venv/bin/python -I hw/realset/build/<file> [build|check]`); `realscenes.load_scene(id)` returns the metrics2 scene dict |
| `../realeval.py` | Runner + table |
| `results/<variant>/<method>/` | Per-scene JSON and the y1 output as 16-bit WAV |

## 1. Session calibration (`calib_session.json`)

```json
{"version": 1, "spacing_m": 0.0119, "tau0_s": -6.346e-05, "hpf_corner_hz": [60, 60], "model": "sinc",
 "gain_db": [0.0, -0.16, 0.25, 0.61, 0.73, 0.80, 0.88, 0.82, 0.75, 0.81, 0.84, 0.51, 0.35, 0.95, 0.78, 0.77, 0.99, 1.79]}
```
(`gain_db` = correction applied to Mic B at the 18 S0 bands 125 Hz–6.35 kHz; `tau0_s` > 0 would mean B late.)

| Quantity | Estimates (samples at 16 kHz / cm / dB) | Used |
|---|---|---|
| skew τ0 | pooled noise of all 18 takes −1.015; per take median −1.016 (−1.030…−1.005); `s3_ambient_after` −1.002 | **−1.015 samples** (−63.5 µs, B leads) |
| spacing | diffuse sinc fits: pooled take noise 1.145, per take 1.12 (1.05–1.31), s3 ambient 1.13 | **1.19 cm** (main agent's value; see below) |
| B re A level | 4 estimates per band (take images, take noise, 5 ms direct path, s3 ambient): −0.4…−1.1 dB from 200 Hz to 5 kHz, −1.6…−2.2 dB at 6.35 kHz | per-band median, sign flipped |
| HPF corners | not measurable from these data | [60, 60] (adds no corner phase) |

Checks (in `calib_session_report.json`):
- Noise cross-spectrum phase after the τ0 correction: median over the 18 takes within ±2° in every band 125 Hz–6.35 kHz
  (before: −11° at 500 Hz, −48° at 2 kHz, −93° at 4 kHz on s3). Per-take scatter is large only at 125–160 Hz
  (low coherence there).
- Direct-path TDOA − τ0 (GCC-PHAT on the fitted responses, + = B later): partners A1–A3/B1–B3 0.375–0.578, median
  0.547 samples, against d/c = 0.555 at 1.19 cm (0.534 at 1.145 cm). 15 of 18 takes are inside ±(d/c + 0.05);
  BE2 0.64, M1 0.625, MF1 0.61 are 0.05–0.09 samples outside (both speakers sit near endfire on Mic A's side, so
  the TDOAs pile up at +d/c and the GCC scatter of ~±0.1 sample pushes some past it). This is why 1.19 cm is used
  rather than the 1.13–1.15 cm diffuse fits: at 1.145 cm, 4 of the 6 partner takes would be outside.
- `calib.load_json` accepts the dict. S0's json override adds its own blind broadband gain drift (±1 dB max).
- **Through S0 itself** (`build/s0_check.py` → `s0_check.json`, 3 bus + 3 meeting scenes): with the json, S0's
  correction leaves the device-noise cross-spectrum phase at −1.8…+1.9° in every band 150–7000 Hz (uncorrected:
  −6/−16/−32/−63/−120° in 150–500/…/4–7k Hz). Blind S0 (1.19 cm or 4 cm prior) returns τ0 = 0 (its skew search is
  {0, +0.5}) and leaves the phase at −12…−120°; its d_hat stays at the prior except meeting/recorded (2.84 cm);
  S0 confidence 0.02–0.08 everywhere.
- **S1 partner TDOAs on the bus scenes** (same file): measured → τ_A = τ_B = −0.03…+0.04 samples (both partners in
  ONE cluster, near 0 rather than at their direct-path +0.55); blind → NaN (no plausible peak); wrong4cm → both at
  −1.04…−1.06 (the uncorrected skew). With τ_A = τ_B, M1's steered DAS stays on its neutral beam (Mic A): on
  L1 0 dB its results are identical to `s0_calib_only` in all three variants (`results/<variant>/m1_das`).

## 2. Bus scenes (`bus/L<k>_snr<±s>_<recorded|quiet>`, 18 scenes, 28.25 s each)

- **Partners are real:** takes A1, B1, A2, B2, A3, B3 (talker A = LibriSpeech 121 F, left speaker; B = 1089 M, right),
  each cut at [onset − 0.3 s, onset + 4.45 s] and joined with 50 ms equal-power crossfades in the quiet margins;
  turn gaps 0.5 s. Image and noise of every take are crossfaded separately (same weights). Partner levels at Mic A
  (in band, active): A −42.1 dBFS, B −48.0 dBFS (B's speaker is 6 dB weaker; kept as recorded).
- **Passengers are simulated** (pyroomacoustics through `lab.scene2.compute_rirs`, synthetic tail, sub-cardioid talkers
  facing random directions, 62–72 dB SPL at 1 m) in a bus-sized room for the session array (1.19 cm), 3 layouts:

  | Layout | Room (m), RT60 | Passengers: distance (m), angle from the mic axis (°), speaker |
  |---|---|---|
  | L1 | 11.75 × 2.56 × 2.21, 0.44 s | 0.94/147/533, 1.49/95/2609, 1.07/77/7018 |
  | L2 | 11.32 × 2.42 × 2.24, 0.37 s | 1.27/53/5442, 1.25/13/533, 2.04/74/4852, 1.10/123/5484 |
  | L3 | 11.20 × 2.56 × 2.23, 0.53 s | 0.93/103/5484, 1.60/166/4852, 1.11/44/1998, 1.29/7/7018, 2.08/161/2033 |

  Speech: LibriSpeech babble pool, test split (`lab.corpus2` 'libri'), utterances ≤ 6 s with 0.4–2.5 s pauses.
  Each passenger is its own group (`near_k`, role `passenger_near`, talker `paxk`, with early-image ref and segs).
- **Bus noise:** DEMAND TBUS ch01 and ch16 at offsets ≥ 20 s apart (independent), made into a 2-ch field with
  coherence sinc(2fd/c) at 1.19 cm by Habets coherence mixing (STFT 512, sqrt-Hann); the second noise contributes
  only its phase at the first one's magnitude, because plain mixing of two non-stationary stretches biased the
  coherence up (fitted 0.97 cm for a 1.19 cm target). Measured on the built scenes: sinc fit 1.10 cm, ρ 1.00,
  skew −1.01/−1.02 samples, B re A −0.77 dB (the real fan noise in the same scene: 1.25 cm, ρ 0.96, −1.014, −0.45 dB).
  Group `bus_noise`, role `engine_road`. Passengers (sum) and bus noise are set equal in band over partner speech,
  then scaled together.
- **Device chain on every simulated part:** 1st-order 60 Hz high-pass per mic, Mic B level per band = −gain_db,
  Mic B advanced by 1.015 samples (τ0), then the takes' 80 Hz 4th-order zero-phase high-pass.
- **SNR rule:** partner speech (A + B images, Mic A, 150–7000 Hz, over partner segs) vs passengers + bus noise at
  −5 / 0 / +5 dB (realized exactly: `scenes/check.json`). The device noise is **not** in the rule.
- **Device noise** (group `device`, role `device_noise`): the six takes' own noise (fans + mic noise + fit residual),
  'recorded' as recorded (≈ −43 dBFS) or 'quiet' at −12 dB. SNR including it: −5.8 / −2.2 / +0.1 dB (recorded) and
  −5.1 / −0.2 / +4.5 dB (quiet) for the −5 / 0 / +5 dB scenes.

## 3. Meeting scenes (`meeting/recorded`, `meeting/quiet`, `meeting/quiet_omeeting`, 38.2 s)

M1–M6 (3575 F, 2300 M, 1995 F, 7729 M, 4970 F, 1320 M at 0, −6, −12, −3, −9, −15 dB) and MF1/MF2 (6829 F, 260 M at
−6 dB, pre-convolved with a 4 m, RT60 0.8 s room = emulated far talkers), in that order, one talker per group
(`talker_00`…`talker_07`, talker ids = take names), pauses 0.31–0.78 s (seeded), 50 ms crossfades, device noise
continuous. Talker levels at Mic A (in band, active): −46.9 (M1) to −62.1 dBFS (M6).
- 'recorded': device noise as recorded (−44 dBFS); per-talker SNR −8.0 (M6) to +6.1 dB (M1).
- 'quiet': device noise −12 dB; per-talker SNR +4.0 to +18.1 dB.
- 'quiet_omeeting': 'quiet' + DEMAND OMEETING ch01/ch16 made diffuse at 1.19 cm (as the bus noise; fit 1.13 cm, ρ 1.00,
  skew −1.014) at −55 dBFS in band (group `omeeting`, role `hvac`); per-talker SNR +1.6 to +12.6 dB. Note: OMEETING
  itself contains distant speech, here spread into a diffuse field.
- `meta.talkers.<id>.dist_m` is **nominal** (0.5 m for M*, 4 m for MF*: the true speaker distance was not measured)
  and `level_db` = 60 + the relative level; they only feed the T2 distance bins and the E7 "soft" flag.

## 4. Metrics (`lab.metrics2`, full compatibility)

`realscenes.load_scene()` returns the BENCH2 scene dict (mix, comp2 in canonical order, groups/roles, refs,
talker_group, segs, meta.scenario/talkers); Σ comp2 = mix exactly. `realeval.py` uses `metrics2.plan` (bus: y1–y4 +
the passenger flip; meeting: y1, y2), `scene2.signed_mix` to flip groups digitally, and `metrics2.score` unchanged.
- **Refs = early images** (direct sound + 50 ms) rebuilt from each take's fitted 4096-tap response: the same
  convolution with the full response reproduces the stored image to −141…−149 dB; refs sit at lag 0 in comp2.
- Added by `realeval.py` (in `m` and `own`): `na_gap_pre/out`, `nm_gap` (noise attenuation where no target talks,
  guards 20 ms / 150 ms, because the bench's idle definition needs ≥ 1 s with 0.1/0.3 s guards and finds none in
  the bus scenes, so the bench's `nm`/`na_idle` are NaN there), and per-talker ESTOI/SDR against the FULL image.

## 5. Enrollment (`scenes/enroll.npz`, `--enroll clean|device`)

- 'clean' = `stim/AE1,AE2.npz['dry']` (A) and `BE1,BE2` (B), concatenated (8.9 s each, 0.25 s leading silence per clip).
- 'device' = Mic A of the AE1+AE2 / BE1+BE2 recordings (80 Hz high-passed as stored), [onset − 0.25 s, onset + 4.45 s]
  each (9.4 s), with the room's fan noise at about 0 dB SNR (images.json: −2.4…+1.4 dB).
Passed as `info['enroll'] = {'A', 'B'}` to every method (only M9 uses it).

## 6. Running

```
.venv/bin/python -I hw/realeval.py run --methods mic_a,m0_mica_hpf,avg [--variant measured|blind|wrong4cm]
        [--set bus|meeting|all] [--enroll clean|device] [-j 2] [--scenes L1_snr+0,meeting] [--skip-existing]
.venv/bin/python -I hw/realeval.py table [--variant measured] [--set bus] [--methods a,b]
```
Variants: `measured` = {spacing 1.19 cm, calib_json}; `blind` = {spacing 1.19 cm, no calib_json};
`wrong4cm` = {spacing 4 cm, no calib_json} (what the methods were tuned with). Methods are stateless per call;
errors are written to the result file and the run goes on. Cost: the 3 baselines on all 21 scenes took 238 s at
-j 2 under load ~165 (63 results, 0 errors; scoring dominates: ~8–11 s per bus scene and method, ~3 s per meeting,
plus the method's own time × 5 runs (bus) or 2 runs (meeting) × 28–38 s of audio).

Smoke results (variant measured, means over scenes; `tmp/night_realeval_smoke.log`):

| | bus: wrst ESTOI (in) | bus: ΔSNR_act | bus: NA_gap | meeting: mean ΔESTOI | meeting: NA_act | meeting T1 capture |
|---|---|---|---|---|---|---|
| mic_a | 0.174 | 0.0 | 0.0 | 0.000 | 0.0 | 0.62 |
| m0_mica_hpf | 0.174 | +0.0 | +0.1 | +0.001 | +0.0 | 0.62 |
| avg | 0.178 | +0.1 | +0.8 | −0.002 | +1.1 | 0.62 |

Input difficulty: bus worst-partner ESTOI at Mic A 0.087 (−5 dB, recorded) … 0.296 (+5 dB, quiet); meeting
per-talker input ESTOI 0.03 (M6, recorded) … 0.78 (M1, quiet). T1 capture 0.62 at the input = the real 15 dB level
spread (5 of 8 talkers within ±3 dB of the median). Closure 150 dB for all three (linear methods decode exactly).

## 7. Limitations (read before trusting a number)

1. **Two speaker positions only.** Both partners and all meeting talkers come from the MacBook's two speakers, both
   near endfire on Mic A's side (TDOA ≈ +d/c). Nothing tests broadside or B-side talkers; spatial methods see an
   easy, fixed geometry. The meeting "8 talkers" are 2 positions × 8 voices.
2. **Real room, simulated crowd.** Passengers and bus noise are simulated (and the bus noise is DEMAND made diffuse);
   only the partners, the room response for them and the device noise are real. The real-room late reverb of the
   partners is in their images (and therefore in the "target" for TA and in comp2), but refs are early images.
3. **Partner level imbalance is real:** B is 5.9 dB below A at Mic A; `balance`/`TAmin` reflect methods, not this.
4. **Bus turn gaps are 0.5 s** (fixed by the 0.3 s / 0.25 s margins): the bench's `nm`/`na_idle` are NaN on the
   bus; use `nm_gap`/`na_gap_*` instead (1.78 s of gap per bus scene, 3.0 s per meeting; the decoded N excludes
   the targets, so their reverb tail does not leak into it, but an AGC's release after a turn falls inside the
   150 ms guard and counts).
5. **The fan noise is not quiet-room noise:** 'recorded' is ≈ −43 dBFS broadband near-diffuse fans (machine loaded);
   'quiet' is a −12 dB scaled copy, not a recording. The image is a 4096-tap LTI fit, so whatever it cannot model
   (time variance within a take, speaker nonlinearity) sits in the "device noise" part during speech: per
   `images.json` the residual during speech is within −5…+2.6 dB of the take's noise-only level (A2, B3, BE1, M2
   are 2–2.6 dB above it). In 'quiet' that part is also scaled by −12 dB, so 'quiet' slightly flatters the image.
6. **Passengers within 0.93–2.08 m** (the room is 2.4–2.6 m wide, so the 0.8–3 m draw never reached 3 m); levels
   relative to the partners follow 62–72 dB SPL physics, then the SNR rule scales the whole crowd.
7. **Nominal meeting distances** (0.5 / 4 m) drive T2 bins: "far" = MF1/MF2 only.
8. 18 bus scenes = 3 layouts × 3 SNRs × 2 device conditions over the SAME partner audio: they are not independent
   samples; no CI/gates are computed (`table` prints plain means).
9. Calibration: spacing chosen as 1.19 cm against diffuse fits of 1.13–1.15 cm (4 % apart); τ0 is a diffuse-phase
   estimate (±0.015 sample spread) and cannot be separated from a residual field anisotropy without an endfire flip.
10. **Scene version.** The scenes were built twice; the current build is from 02:21–02:22 (the first, 02:10, had the
    biased diffuse fields). `hw/realset/results/` was deleted at the rebuild, so every result file there is from the
    current scenes (written 02:22 or later). Do not rebuild while other runs are scoring: `scenes/*.npz` are read
    lazily by each worker.
