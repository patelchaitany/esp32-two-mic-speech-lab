# MEETING_V3: final meeting pipelines (final:meeting, 2026-11-03; dev only, natural speech `--corpus libri`)

Results: `results3/final/dev/<method>/meeting_rNN.libri.json`. Scratch, logs and drivers: `tmp/night_meetv3/`. Test set not touched.

## 1. Outcome (20 dev rooms, paired)
- **Chosen `methods/x_meeting_v3.py`**: x_s0v2 self-calibration when no calib_json (else the json) -> 2-ch offline WPE -> S1 on WPE output -> steered superdirective M2 (WNG >= -3 dB, 400 ms hold, 50 ms xfade) -> **M8 GTCRN, 12 dB limit (replaces M6)** -> M11 (tuned meeting AGC, one static gain per S1 speech segment, -1 dBTP limiter). Returns {out, pre_agc}.
- **Best classical `methods/x_meeting_v3_classic.py`**: same chain with M6 (floor -12 dB, noise update only in >= 300 ms S1-VAD pauses; tuned ns.py).
- Family: `methods/x_meeting_v3_stack.py` (`enhance_cfg`, `run_all`), variants `methods/x_meeting_v3_<tag>.py`.
- Both PASS G0; both FAIL G1 only on T3 (level std / range); WER criterion pending (no ASR model).
- x_meeting_v3 vs classic (paired, 20 rooms): worst-talker dESTOI +0.002 [-0.007, +0.011] (tie), mean +0.027 [+0.025, +0.029], >4 m +0.022 [+0.018, +0.025], NA_active +1.9 dB, but NM +5.0 dB (5.46 [5.01, 5.95], close to the 6 dB cap), T3 std +0.07 dB, capture -0.01. If noise modulation matters more than intelligibility, use the classic.

## 2. 20 dev rooms (means over rooms; d vs raw Mic A)

| method | wrstΔ | meanΔ | <2m | 2-4m | >4m | ok | T2 | TAwrst | T3std | T3rng | soft | turnσ | T1cap | NAact | NAidle | NM | pump | settle | RTF | dly |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 1.000 | +0.0 | -0.1 | 4.14 | 15.8 | -7.3 | 1.3 | 0.53 | +0.0 | +0.0 | -0.0 | 0.0 | 0.00 | 0.000 | +0 |
| x_meeting_v3 | +0.075 | +0.125 | +0.118 | +0.132 | +0.118 | 1.000 | -1.4 | -3.9 | 2.00 | 8.3 | -4.3 | 0.8 | 0.90 | +6.8 | +13.0 | +5.5 | 2.3 | 2.48 | 0.139 | +0 |
| v3_classic | +0.073 | +0.098 | +0.089 | +0.102 | +0.097 | 1.000 | -1.0 | -2.4 | 1.93 | 8.0 | -4.1 | 0.8 | 0.91 | +4.9 | +5.8 | +0.4 | 2.3 | 1.53 | 0.085 | +0 |

paired vs x_meeting_v3_classic (mean [95% CI], per room):
  x_meeting_v3: d_estoi_wrst=+0.002 [-0.007,+0.011]; d_estoi_mean=+0.027 [+0.025,+0.029]; d_estoi_far=+0.022 [+0.018,+0.025]; lvl_std=+0.070 [-0.005,+0.158]; lvl_range=+0.326 [-0.041,+0.762]; capture_rate=-0.010 [-0.020,-0.003]; na_active=+1.911 [+1.681,+2.156]; nm=+5.048 [+4.736,+5.343]

### Gates

| method | G0 | G1 | failing criteria |
|---|---|---|---|
| m0_mica_hpf | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| x_meeting_v3 | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB |
| x_meeting_v3_classic | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB |

| method | gate | criterion | n | point | 95% CI | pass |
| x_meeting_v3 | G0 | worst-talker ESTOI >= input - 0.01 | 20 | 0.0751 | [0.0582, 0.0933] | yes |
| x_meeting_v3 | G0 | dESTOI >= -0.03 for >= 19/20 talkers | 20 | 1 | [1, 1] | yes |
| x_meeting_v3 | G0 | T2: far-bin median TA >= near-bin median - 3 dB | 20 | -1.36 | [-1.58, -1.13] | yes |
| x_meeting_v3 | G0 | NM <= 6 dB | 20 | 5.46 | [5.01, 5.95] | yes |
| x_meeting_v3 | G0 | no NaN or errors | 20 | 0 | — | yes |
| x_meeting_v3 | G0 | RTF < 0.5 | 20 | 0.152 | — | yes |
| x_meeting_v3 | G1 | T3 level std <= 1.5 dB | 20 | 2 | [1.65, 2.37] | NO |
| x_meeting_v3 | G1 | T3 level range <= 6 dB | 20 | 8.31 | [6.71, 10.1] | NO |
| x_meeting_v3 | G1 | mean dESTOI >= +0.02 | 20 | 0.125 | [0.116, 0.135] | yes |
| x_meeting_v3 | G1 | dESTOI >= +0.03 in the > 4 m bin | 20 | 0.118 | [0.106, 0.13] | yes |
| x_meeting_v3 | G1 | WER not worse in any distance bin | 0 | — | — | pending: needs a model that is not downloaded yet |
| x_meeting_v3 | G1 | NA_active >= 3 dB | 20 | 6.78 | [6.33, 7.2] | yes |
| x_meeting_v3_classic | G0 | worst-talker ESTOI >= input - 0.01 | 20 | 0.0733 | [0.0607, 0.0867] | yes |
| x_meeting_v3_classic | G0 | dESTOI >= -0.03 for >= 19/20 talkers | 20 | 1 | [1, 1] | yes |
| x_meeting_v3_classic | G0 | T2: far-bin median TA >= near-bin median - 3 dB | 20 | -0.966 | [-1.15, -0.779] | yes |
| x_meeting_v3_classic | G0 | NM <= 6 dB | 20 | 0.416 | [-0.0492, 0.898] | yes |
| x_meeting_v3_classic | G0 | no NaN or errors | 20 | 0 | — | yes |
| x_meeting_v3_classic | G0 | RTF < 0.5 | 20 | 0.0932 | — | yes |
| x_meeting_v3_classic | G1 | T3 level std <= 1.5 dB | 20 | 1.93 | [1.63, 2.25] | NO |
| x_meeting_v3_classic | G1 | T3 level range <= 6 dB | 20 | 7.99 | [6.58, 9.57] | NO |
| x_meeting_v3_classic | G1 | mean dESTOI >= +0.02 | 20 | 0.098 | [0.0896, 0.106] | yes |
| x_meeting_v3_classic | G1 | dESTOI >= +0.03 in the > 4 m bin | 20 | 0.0968 | [0.0859, 0.108] | yes |
| x_meeting_v3_classic | G1 | WER not worse in any distance bin | 0 | — | — | pending: needs a model that is not downloaded yet |
| x_meeting_v3_classic | G1 | NA_active >= 3 dB | 20 | 4.87 | [4.51, 5.2] | yes |

## 3. Phase 1: 5 dev rooms r00-r04, 12 variants (DESIGN M6 before tune:ns changed ns.py at 03:57; `classic` there = now `x_meeting_v3_core`)

| method | wrstΔ | meanΔ | <2m | 2-4m | >4m | ok | T2 | TAwrst | T3std | T3rng | soft | turnσ | T1cap | NAact | NAidle | NM | pump | settle | RTF | dly |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | +0.000 | +0.000 | +0.000 | +0.001 | -0.000 | 1.000 | +0.0 | -0.1 | 4.18 | 15.1 | -6.3 | 1.2 | 0.44 | +0.0 | +0.0 | -0.0 | 0.0 | 0.00 | 0.000 | +0 |
| v3_core | +0.054 | +0.092 | +0.095 | +0.097 | +0.080 | 1.000 | -1.1 | -3.3 | 1.77 | 7.0 | -2.8 | 0.9 | 0.92 | +7.2 | +12.3 | +4.5 | 2.2 | 2.23 | 0.126 | +0 |
| v3_nons | +0.026 | +0.049 | +0.051 | +0.053 | +0.040 | 1.000 | -0.5 | -1.8 | 1.67 | 6.5 | -2.6 | 0.8 | 0.93 | +1.7 | +1.1 | -0.4 | 2.5 | 0.81 | 0.109 | +0 |
| v3_nobeam | +0.047 | +0.075 | +0.076 | +0.079 | +0.066 | 1.000 | -0.8 | -3.1 | 1.78 | 7.0 | -2.8 | 0.8 | 0.92 | +6.6 | +12.5 | +5.3 | 2.0 | 2.17 | 0.121 | +0 |
| v3_fixed3 | +0.046 | +0.066 | +0.067 | +0.069 | +0.056 | 1.000 | -0.9 | -3.2 | 1.78 | 6.9 | -2.8 | 0.8 | 0.92 | +6.8 | +12.8 | +5.3 | 2.0 | 2.34 | 0.132 | +0 |
| v3_nowpe | +0.032 | +0.055 | +0.058 | +0.060 | +0.044 | 1.000 | -0.9 | -3.1 | 1.86 | 7.5 | -3.3 | 0.9 | 0.92 | +6.8 | +11.6 | +4.1 | 2.2 | 1.42 | 0.064 | +0 |
| v3_s0v2 | +0.057 | +0.096 | +0.100 | +0.104 | +0.079 | 1.000 | -1.2 | -3.7 | 1.79 | 7.1 | -2.9 | 0.8 | 0.92 | +7.7 | +12.3 | +4.0 | 2.2 | 2.25 | 0.153 | +0 |
| v3_gtcrn12 | +0.078 | +0.122 | +0.119 | +0.130 | +0.108 | 1.000 | -1.4 | -3.4 | 1.76 | 6.8 | -2.8 | 0.8 | 0.92 | +6.9 | +13.0 | +5.6 | 2.3 | 2.46 | 0.208 | +0 |
| v3_m6gtcrn9 | +0.083 | +0.125 | +0.123 | +0.133 | +0.111 | 1.000 | -1.6 | -3.8 | 1.61 | 6.3 | -2.6 | 0.8 | 0.92 | +8.5 | +21.0 | +11.6 | 2.4 | 2.69 | 0.209 | +0 |
| v3_dpdf12 | +0.095 | +0.139 | +0.142 | +0.151 | +0.114 | 1.000 | -2.4 | -6.1 | 2.17 | 8.8 | -3.7 | 1.1 | 0.89 | +7.2 | +13.1 | +5.2 | 2.4 | 2.44 | 0.477 | +0 |
| v3_emb | +0.055 | +0.095 | +0.100 | +0.100 | +0.082 | 1.000 | -1.1 | -3.3 | 4.41 | 16.0 | -6.5 | 1.5 | 0.45 | +7.2 | +12.3 | +5.3 | 2.2 | 1.93 | 0.129 | +0 |
| v3_embhyb | +0.055 | +0.093 | +0.097 | +0.098 | +0.081 | 1.000 | -1.1 | -3.3 | 2.39 | 9.5 | -3.6 | 1.1 | 0.84 | +7.2 | +12.3 | +5.0 | 2.3 | 2.13 | 0.127 | +0 |
| v3_dpkeys | +0.054 | +0.094 | +0.098 | +0.099 | +0.081 | 1.000 | -1.1 | -3.3 | 3.45 | 12.0 | -5.7 | 1.3 | 0.63 | +7.2 | +12.3 | +5.3 | 2.1 | 2.05 | 0.123 | +0 |

paired vs x_meeting_v3_core (mean [95% CI], per room):
  v3_nons: d_estoi_wrst=-0.028 [-0.033,-0.021]; d_estoi_mean=-0.043 [-0.047,-0.041]; d_estoi_far=-0.041 [-0.051,-0.033]; lvl_std=-0.096 [-0.204,+0.003]; lvl_range=-0.470 [-1.061,+0.120]; capture_rate=+0.010 [+0.000,+0.030]; na_active=-5.497 [-5.933,-5.061]; nm=-4.896 [-5.249,-4.548]
  v3_nobeam: d_estoi_wrst=-0.007 [-0.012,-0.002]; d_estoi_mean=-0.017 [-0.025,-0.010]; d_estoi_far=-0.014 [-0.028,-0.006]; lvl_std=+0.012 [-0.013,+0.036]; lvl_range=+0.018 [-0.110,+0.188]; capture_rate=+0.000 [+0.000,+0.000]; na_active=-0.629 [-0.924,-0.346]; nm=+0.833 [+0.459,+1.322]
  v3_fixed3: d_estoi_wrst=-0.008 [-0.019,+0.001]; d_estoi_mean=-0.026 [-0.041,-0.015]; d_estoi_far=-0.024 [-0.050,-0.006]; lvl_std=+0.013 [-0.006,+0.033]; lvl_range=-0.056 [-0.188,+0.114]; capture_rate=+0.000 [+0.000,+0.000]; na_active=-0.401 [-0.664,-0.140]; nm=+0.850 [+0.464,+1.320]
  v3_nowpe: d_estoi_wrst=-0.022 [-0.031,-0.013]; d_estoi_mean=-0.037 [-0.042,-0.031]; d_estoi_far=-0.036 [-0.048,-0.024]; lvl_std=+0.093 [+0.007,+0.175]; lvl_range=+0.511 [+0.071,+0.870]; capture_rate=+0.000 [+0.000,+0.000]; na_active=-0.403 [-0.612,-0.193]; nm=-0.340 [-0.607,-0.025]
  v3_s0v2: d_estoi_wrst=+0.003 [+0.000,+0.007]; d_estoi_mean=+0.004 [-0.002,+0.010]; d_estoi_far=-0.001 [-0.008,+0.007]; lvl_std=+0.029 [+0.004,+0.058]; lvl_range=+0.170 [-0.089,+0.429]; capture_rate=+0.000 [+0.000,+0.000]; na_active=+0.451 [+0.114,+0.751]; nm=-0.458 [-0.584,-0.344]
  v3_gtcrn12: d_estoi_wrst=+0.024 [+0.007,+0.040]; d_estoi_mean=+0.029 [+0.025,+0.034]; d_estoi_far=+0.027 [+0.013,+0.039]; lvl_std=-0.007 [-0.050,+0.029]; lvl_range=-0.210 [-0.442,-0.063]; capture_rate=+0.000 [+0.000,+0.000]; na_active=-0.309 [-0.539,-0.093]; nm=+1.178 [+0.869,+1.489]
  v3_m6gtcrn9: d_estoi_wrst=+0.029 [+0.018,+0.041]; d_estoi_mean=+0.033 [+0.030,+0.036]; d_estoi_far=+0.031 [+0.025,+0.035]; lvl_std=-0.157 [-0.285,-0.054]; lvl_range=-0.678 [-1.062,-0.306]; capture_rate=+0.000 [+0.000,+0.000]; na_active=+1.285 [+1.157,+1.474]; nm=+7.168 [+7.071,+7.268]
  v3_dpdf12: d_estoi_wrst=+0.041 [+0.012,+0.064]; d_estoi_mean=+0.047 [+0.040,+0.055]; d_estoi_far=+0.034 [+0.016,+0.050]; lvl_std=+0.409 [+0.009,+0.967]; lvl_range=+1.836 [-0.419,+5.147]; capture_rate=-0.030 [-0.070,+0.020]; na_active=+0.010 [-0.113,+0.103]; nm=+0.777 [+0.070,+1.367]
  v3_emb: d_estoi_wrst=+0.001 [-0.000,+0.003]; d_estoi_mean=+0.003 [+0.002,+0.004]; d_estoi_far=+0.002 [+0.001,+0.004]; lvl_std=+2.643 [+2.377,+2.904]; lvl_range=+8.982 [+6.943,+10.850]; capture_rate=-0.470 [-0.570,-0.380]; na_active=+0.000 [+0.000,+0.000]; nm=+0.828 [+0.220,+1.505]
  v3_embhyb: d_estoi_wrst=+0.001 [-0.001,+0.003]; d_estoi_mean=+0.001 [+0.000,+0.001]; d_estoi_far=+0.000 [-0.000,+0.001]; lvl_std=+0.629 [+0.105,+1.162]; lvl_range=+2.565 [+0.292,+4.820]; capture_rate=-0.080 [-0.150,-0.010]; na_active=+0.000 [+0.000,+0.000]; nm=+0.491 [+0.009,+1.138]
  v3_dpkeys: d_estoi_wrst=+0.000 [-0.002,+0.002]; d_estoi_mean=+0.002 [+0.001,+0.002]; d_estoi_far=+0.001 [+0.000,+0.002]; lvl_std=+1.686 [+1.501,+1.880]; lvl_range=+5.058 [+4.284,+5.908]; capture_rate=-0.290 [-0.320,-0.260]; na_active=+0.000 [+0.000,+0.000]; nm=+0.791 [+0.350,+1.382]

### Stage verdicts (phase 1, paired vs core)
- WPE: keep (removing it: worst -0.022 [-0.031, -0.013], mean -0.037, >4 m -0.036).
- Steered M2: keep, small (no beam: worst -0.007 [-0.012, -0.002], mean -0.017). fixed3 + best-beam selection worse than steered (mean -0.026): dropped.
- NS: largest stage (none: worst -0.028, NA_active -5.5 dB).
- x_s0v2 vs calib.py S0: no cost on sim (worst +0.003 [0.000, +0.007]); needed on the real board (1.19 cm, skew ~ -1 sample): used in both finals.
- DPDFNet 12 dB: best ESTOI but T3 std +0.41, TA_wrst -6.1 dB, T2 -2.4 dB, RTF 0.48: rejected. M6 + GTCRN 9 dB: NM 11.6 dB (fails G0 NM <= 6): rejected.
- **Candidate (c) WeSpeaker-keyed M11: rejected** (T3 std 4.41 dB / capture 0.45, no better than M0; embhyb 2.39; DP-means TDOA keys 3.45) vs 1.77 / 0.92 for per-segment keys. Merged talkers share one gain; offline a per-segment gain wins. ESTOI unchanged by keys.

## 4. Protocol notes and limits
- Driver `tmp/night_meetv3/explore.py` / `explore2.py`: one render per scene, M0 via bench2.run_method, variants via `run_all` (shared prefixes; outputs bit-identical to each module's enhance, checked on a smoke room), scored with metrics2.score as the bench (2 Hagerman runs, no extra/rerun runs). RTF = sum of the variant's own stage times under load ~130-140.
- ns.py changed between phases (md5 in `tmp/night_meetv3/core_versions_phase{1,2}.txt`): phase-1 M6 rows use DESIGN M6; the 20-room classic uses tuned M6.
- Cut for time: AGC parameter variants, GTCRN 9 dB alone, online variant, real-board run (`tmp/night_meetv3/real_meet.py` ready, not run), key-purity diagnostic (`diag_keys.py`, not run).
- At 1.19 cm the steered beam stays on Mic A (hw/spacing/IMPACT.md); on the device the gain comes from WPE + NS + M11.
- Listening examples (fixed rule): full y1 of dev/meeting_r05 and r15 for m0_mica_hpf, x_meeting_v3_classic, x_meeting_v3: `results3/final/dev/<method>/meeting_r{05,15}.libri.wav`.
