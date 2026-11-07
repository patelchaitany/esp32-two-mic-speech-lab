# BUS v3 (night-B final:bus) — candidates, dev results, choice

Status: INCOMPLETE (stage deadline 04:50). Every number below is on the dev set, LibriSpeech (`--corpus libri`),
results in `results3/final/`. Test set never touched. Cut for time: the full dev at SNR -5 (a launch failed on
`--snr -5,0` argument parsing and cost the slot), the extra Hagerman runs (`--extra none` everywhere), M9 (c) runs,
online variants. The 12-scene subset is the only place where all candidates were compared.

## Files
- `methods/x_bus_v3_stack.py` config-driven chain (x_s0v2 calib -> S0/S1 -> spatial m3|m2|mica -> masks from linear
  signals combined once (M4 CDR, M5 fence, CDR-gated M6, M9 union, M8 on target or lin) -> floor/fall rule -> split
  remix (-15 dB, M6 + slow 2:1 compression) -> M11-bus AGC + -1 dBTP limiter; returns out and pre_agc).
- `methods/x_bus_v3.py` CHOSEN = config of `x_bus_v3_m0d20`: Mic A -> DPDFNet-4 mask (20 dB limit) -> remix -> AGC.
- `methods/x_bus_v3_classic.py` best without neural models = config of `x_bus_v3_c_m3` (DESIGN 2.1 Tier 1).
- Variants `methods/x_bus_v3_<tag>.py`: c_m3, c_m2, c_mica, c_m3_fen, c_m3_s0, c_m3_nor, c_m3_f10, c_m3_cdr, c_m3_ns,
  c_m3_df, c_m3_r10 (last two not run), g12_m3, g20_m3, d12_m3 (not run), d20_m3, m0g20, m0d20, t_m3 (M9, OFFLINE_ONLY).
- Logs/tables: `tmp/night_finalbus/` (subset_table.md, subset_compare.md, spacing_table.md, tse_check.json).

## Dev subset: 12 geometries g00-g11 (all 5 strata, 25/40/60 mm), SNR 0, natural speech, 5 runs/scene
Paired differences vs m0_mica_hpf, 95% bootstrap CI over geometries (`bench2 compare`, full output in
`tmp/night_finalbus/subset_compare.md`). TA/NA/dSNR are pre-AGC.

| method | wrst dESTOI [CI] | dSNR_act dB [CI] | TA_min dB | balP90 dB | NA_act dB | NM dB | RDA p10 | RTF | closure dB |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | +0.000 | +0.0 | -0.1 | 0.1 | +0.1 | +0.1 | -0.1 | 0.000 | 150 |
| m4_cdr | +0.001 [-0.001,+0.004] | +0.7 [+0.3,+1.1] | -2.2 | 1.7 | +2.7 | +0.3 | -0.3 | 0.03 | 55 |
| **m8_dpdfnet** (d, raw) | **+0.069 [+0.040,+0.099]** | **+9.3 [+7.4,+11.2]** | -8.1 | 12.2 | +15.1 | +1.1 | -0.7 | 0.45 | 32 |
| m8_gtcrn (d, raw) | +0.025 [+0.015,+0.035] | +6.6 [+5.6,+7.5] | -8.9 | 6.2 | +14.7 | +0.3 | -1.3 | 0.11 | 37 |
| x_bus_v3_c_m3 (classic) | +0.018 [+0.011,+0.025] | +1.9 [+1.0,+2.8] | -4.5 | 6.7 | +5.5 | +4.6 | -4.6 | 0.11 | 24 |
| x_bus_v3_c_m2 | +0.013 [+0.009,+0.018] | +1.7 [+0.9,+2.6] | -4.5 | 5.9 | +5.3 | +4.7 | -3.2 | 0.08 | 26 |
| x_bus_v3_c_mica | +0.014 [+0.010,+0.018] | +1.1 [+0.2,+1.9] | -4.4 | 4.6 | +4.8 | +4.7 | -2.5 | 0.07 | 30 |
| x_bus_v3_c_m3_fen (+M5) | +0.017 [+0.010,+0.025] | +2.6 [+1.7,+3.4] | -4.8 | 6.9 | +6.3 | +3.9 | -4.8 | 0.10 | 24 |
| x_bus_v3_c_m3_s0 (default S0) | +0.020 [+0.013,+0.027] | +1.5 [+0.6,+2.4] | -4.3 | 6.7 | +5.1 | +4.8 | -1.6 | 0.08 | 31 |
| x_bus_v3_c_m3_nor (no remix) | +0.016 [+0.009,+0.023] | +2.0 [+1.0,+3.0] | -4.8 | 7.0 | +5.9 | +5.3 | -4.8 | 0.08 | 25 |
| x_bus_v3_c_m3_f10 (floor -10) | +0.016 [+0.009,+0.023] | +1.7 [+0.9,+2.5] | -4.1 | 6.1 | +5.0 | +3.0 | -4.2 | 0.10 | 26 |
| x_bus_v3_c_m3_cdr (no M6) | +0.007 [-0.001,+0.017] | +1.3 [+0.8,+1.7] | -2.2 | 5.0 | +3.0 | +0.1 | -4.0 | 0.08 | 28 |
| x_bus_v3_c_m3_ns (no M4) | +0.018 [+0.010,+0.026] | +1.5 [+0.7,+2.2] | -2.8 | 6.4 | +3.5 | +5.2 | -5.1 | 0.09 | 24 |
| x_bus_v3_g12_m3 (b, GTCRN 12) | +0.013 [+0.006,+0.022] | +4.8 [+4.2,+5.5] | -7.2 | 6.3 | +11.3 | +0.8 | -4.1 | 0.17 | 25 |
| x_bus_v3_g20_m3 (b, GTCRN 20) | +0.013 [+0.006,+0.022] | +5.1 [+4.4,+5.7] | -7.2 | 6.2 | +11.6 | +0.6 | -4.1 | 0.17 | 25 |
| x_bus_v3_d20_m3 (b, DPDFNet 20) | +0.016 [+0.008,+0.026] | +5.1 [+4.2,+6.0] | -8.5 | 8.5 | +11.8 | +0.7 | -4.0 | 0.39 | 24 |

Passenger ESTOI (paxESTOI) 0.08-0.10 for all (M0 0.082); passenger drop within +-0.01 of M0 (no method suppresses
near passengers' intelligibility measurably). focus% = 0 for every method (RDA p10 < 10 dB everywhere).

## Choice and why
- Decision order (tuning protocol): worst-partner dESTOI, dSNR_active, passenger suppression, partner level drop, G0.
- (d) single-channel Mic A -> DPDFNet wins the two primary metrics by a wide margin (+0.069 ESTOI, +9.3 dB vs
  +0.018 / +1.9 dB for the best classical stack). Every spatial / classical-mask stage IN FRONT of a network
  lowered its ESTOI (b: d20_m3 +0.016, g20_m3 +0.013 vs raw m8 +0.069 / +0.025): the masks remove the cues the
  network uses and their errors compound. -> **x_bus_v3 = Mic A -> DPDFNet (20 dB) -> remix -> M11-bus AGC**
  (config `x_bus_v3_m0d20`; the remix/AGC wrapper's own dev numbers: see "SNR 0 full dev" below if present).
- Classic: M3 (c_m3) over M2 (+0.018 vs +0.013 ESTOI, same dSNR) and over Mic A (+0.014): the linear stage helps a
  little where S1 localizes the partners. M5 fence: same ESTOI, +0.7 dB dSNR (not adopted: 12 scenes, and it was
  negative on the full TTS dev in tune:masks). M4+M6 product: M6 carries the ESTOI gain, M4 the low NM; both kept.
- Self-calibration: x_s0v2 (deployable on the real board) vs default S0 on these simulated scenes: +0.018 vs +0.020
  ESTOI. On sim the default S0's {0, 0.5}-sample skew prior is exactly the truth; x_s0v2 missed the 0.5-sample skew
  on e.g. g05 (estimated 0), which makes M3 and M2 fall back to Mic A for the whole file. Kept x_s0v2 (the real
  board's -1.0..-1.11 sample skew is unrepresentable by default S0).

## Gates (honest)
- G0 FAILS for every candidate including the chosen one: TA_min (pre-AGC partner level) -4.5 dB (classic) and
  -8.1 dB (DPDFNet) vs the -3 dB limit; balance P90 6.7 / 12.2 dB vs the 3 dB cap. NM <= 6 dB, no errors/NaN/clipping,
  RTF < 0.5 pass (DPDFNet RTF 0.39-0.45 under load: marginal). No make-up gain was added to hide the TA drop.
- G1: worst-partner dESTOI +0.069 passes the 0.03 "all" criterion on the subset for DPDFNet; aligned-stratum
  criteria are n/a (3 aligned geometries). dSNR_active passes for all 'b'/'d' candidates and c_m3_fen.
- G0/G1 verdicts on 12 geometries are indicative only.

## Failure cases
- Partner level/balance: the masks (classic) and DPDFNet cut the partners by 4-9 dB pre-AGC and unevenly (balance
  P90 6-12 dB); the M11-bus AGC gate (p_A + p_B > 0.8) opens rarely on these scenes (S1 posteriors), so the AGC does
  not restore it reliably.
- S1 partner TDOAs wrong or one peak -> M3/M2 fallback to Mic A (the classic stack then = c_mica).
- x_s0v2 skew errors on simulated scenes (see above) -> beam fallback.
- Hagerman closure only 24-37 dB for all nonlinear stacks (each signed run re-estimates calibration and masks).
- (c) M9: not evaluated on dev. bench2 passes no enrollment and self-enrollment failed on 4/4 checked dev scenes
  (1.8-2.9 s of confident solo speech < 3 s; `tmp/night_finalbus/tse_check.json`), so t_m3 = c_m3 there.

## RTF (bench median, loaded machine, 1 thread)
c_m3 0.11, c_m2 0.08, c_mica 0.07, g20_m3 0.17, d20_m3 0.39, m8_dpdfnet 0.45 (x_bus_v3 ~ the same DPDFNet cost).
Algorithmic latency (online form): 37 ms classic, 69 ms with DPDFNet (4-frame model delay). The evaluated form is
offline (whole-file S0/S1/M3 statistics).

## Sanity check, 12 mm spacing scenes (study/run.py, 10 layouts at SNR 0, factory calib_json and blind)
See `tmp/night_finalbus/spacing_table.md` (results in `results3/final/spacing/`). No tuning on these.

## SNR 0 full dev (30 geometries, --extra none; raw bench table rows and gate lines, from tmp/night_finalbus/logs/full.log)
| m0_mica_hpf | 30 | 30 | 0 | +0.001 | 0.275 | -0.0 | -0.1 | 0.2 | -0.0 | +0.1 | +0.1 | -0.0 | -0.1 | 0.100 | -0.001 | 0 | 0.000 | +0.0 | 0 | 150 | — | — | +1.7 | +4.1 | 1.06 |
| x_bus_v3 | 30 | 30 | 0 | +0.042 | 0.318 | +3.8 | -5.6 | 6.5 | +5.9 | +10.0 | -0.4 | +3.7 | -1.6 | 0.108 | -0.010 | 0 | 0.242 | +0.0 | 69 | 32 | — | — | +7.0 | +10.4 | 1.10 |
| x_bus_v3_classic | 30 | 30 | 0 | +0.032 | 0.309 | +0.4 | -2.9 | 3.9 | +1.0 | +3.5 | +2.4 | +0.0 | -4.2 | 0.110 | -0.010 | 0 | 0.061 | -0.1 | 37 | 27 | — | — | +2.0 | +7.1 | 1.07 |
| m0_mica_hpf | G0 | worst-partner ESTOI >= input - 0.01 | 30 | 0.000846 | [0.000369, 0.00131] | yes |
| m0_mica_hpf | G0 | both TA_k >= -3 dB | 30 | -0.138 | [-0.158, -0.121] | yes |
| m0_mica_hpf | G0 | balance P90 <= 3 dB | 30 | 0.152 | [0.128, 0.178] | yes |
| m0_mica_hpf | G0 | NM <= 6 dB | 30 | 0.0821 | [0.0549, 0.11] | yes |
| m0_mica_hpf | G0 | no clipping, NaN or errors | 30 | 0 | — | yes |
| m0_mica_hpf | G0 | RTF < 0.5 | 30 | 0.000181 | — | yes |
| m0_mica_hpf | G1 | worst-partner dESTOI >= +0.03 | 30 | 0.000846 | [0.000369, 0.00131] | NO |
| m0_mica_hpf | G1 | worst-partner dESTOI >= +0.05 (aligned) | 6 | 0.000887 | [-0.00065, 0.00236] | NO |
| m0_mica_hpf | G1 | dSNR_active >= +3 dB (aligned) | 6 | -0.025 | [-0.0678, 0.0298] | NO |
| m0_mica_hpf | G1 | dSNR_active >= +1.5 dB (all strata) | 30 | -0.0227 | [-0.0453, 0.00107] | NO |
| x_bus_v3 | G0 | worst-partner ESTOI >= input - 0.01 | 30 | 0.0416 | [0.0345, 0.049] | yes |
| x_bus_v3 | G0 | both TA_k >= -3 dB | 30 | -5.62 | [-6.5, -4.77] | NO |
| x_bus_v3 | G0 | balance P90 <= 3 dB | 30 | 6.5 | [5.83, 7.16] | NO |
| x_bus_v3 | G0 | NM <= 6 dB | 30 | -0.443 | [-0.934, -0.00322] | yes |
| x_bus_v3 | G0 | no clipping, NaN or errors | 30 | 0 | — | yes |
| x_bus_v3 | G0 | RTF < 0.5 | 30 | 0.249 | — | yes |
| x_bus_v3 | G1 | worst-partner dESTOI >= +0.03 | 30 | 0.0416 | [0.0345, 0.049] | yes |
| x_bus_v3 | G1 | worst-partner dESTOI >= +0.05 (aligned) | 6 | 0.0477 | [0.0279, 0.071] | NO |
| x_bus_v3 | G1 | dSNR_active >= +3 dB (aligned) | 6 | 6.35 | [4.7, 7.57] | yes |
| x_bus_v3 | G1 | dSNR_active >= +1.5 dB (all strata) | 30 | 5.92 | [5.28, 6.53] | yes |
| x_bus_v3_classic | G0 | worst-partner ESTOI >= input - 0.01 | 30 | 0.0323 | [0.0271, 0.0387] | yes |
| x_bus_v3_classic | G0 | both TA_k >= -3 dB | 30 | -2.93 | [-3.13, -2.71] | NO |
| x_bus_v3_classic | G0 | balance P90 <= 3 dB | 30 | 3.93 | [2.51, 5.7] | NO |
| x_bus_v3_classic | G0 | NM <= 6 dB | 30 | 2.44 | [1.79, 3.09] | yes |
| x_bus_v3_classic | G0 | no clipping, NaN or errors | 30 | 0 | — | yes |
| x_bus_v3_classic | G0 | RTF < 0.5 | 30 | 0.0753 | — | yes |
| x_bus_v3_classic | G1 | worst-partner dESTOI >= +0.03 | 30 | 0.0323 | [0.0271, 0.0387] | yes |
| x_bus_v3_classic | G1 | worst-partner dESTOI >= +0.05 (aligned) | 6 | 0.0412 | [0.0244, 0.0682] | NO |
| x_bus_v3_classic | G1 | dSNR_active >= +3 dB (aligned) | 6 | 0.272 | [-1.02, 1.69] | NO |
| x_bus_v3_classic | G1 | dSNR_active >= +1.5 dB (all strata) | 30 | 1.03 | [0.543, 1.51] | NO |

12 mm spacing rows:
| m0_mica_hpf@blind | 10 | 10 | 0 | +0.001 | 0.287 | -0.0 | -0.2 | 0.2 | -0.0 | +0.1 | +0.1 | -0.0 | -0.1 | 0.148 | -0.002 | 0 | 0.000 | +0.0 | 0 | 150 | — | — | +1.8 | +3.5 | 1.06 |
| m0_mica_hpf@factory | 10 | 10 | 0 | +0.001 | 0.287 | -0.0 | -0.2 | 0.2 | -0.0 | +0.1 | +0.1 | -0.0 | -0.1 | 0.148 | -0.002 | 0 | 0.000 | +0.0 | 0 | 150 | — | — | +1.8 | +3.5 | 1.06 |
| x_bus_v3@blind | 10 | 10 | 0 | +0.050 | 0.339 | +4.7 | -3.8 | 5.8 | +6.5 | +9.5 | -0.6 | +5.3 | -0.3 | 0.165 | -0.020 | 0 | 0.361 | +0.0 | 69 | 36 | — | — | +7.7 | +10.5 | 1.14 |
| x_bus_v3@factory | 10 | 10 | 0 | +0.050 | 0.339 | +4.7 | -3.8 | 5.8 | +6.5 | +9.5 | -0.6 | +5.3 | +0.4 | 0.165 | -0.020 | 0 | 0.342 | +0.0 | 69 | 31 | — | — | +7.7 | +10.5 | 1.13 |
| x_bus_v3_classic@blind | 10 | 10 | 0 | +0.029 | 0.317 | +0.7 | -2.4 | 3.3 | +1.0 | +3.1 | +0.6 | +0.6 | -3.6 | 0.166 | -0.020 | 0 | 0.081 | +0.0 | 37 | 37 | — | — | +2.3 | +5.7 | 1.08 |
| x_bus_v3_classic@factory | 10 | 10 | 0 | +0.029 | 0.316 | +0.8 | -2.4 | 3.2 | +1.1 | +3.1 | +0.5 | +0.5 | -3.7 | 0.168 | -0.022 | 0 | 0.070 | +0.0 | 37 | 35 | — | — | +2.5 | +5.8 | 1.08 |
| m0_mica_hpf@blind | PASS | FAIL | worst-partner dESTOI >= +0.03; dSNR_active >= +1.5 dB (all strata) |
| m0_mica_hpf@factory | PASS | FAIL | worst-partner dESTOI >= +0.03; dSNR_active >= +1.5 dB (all strata) |
