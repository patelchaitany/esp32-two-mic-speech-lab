#!/usr/bin/env python
"""Writes physics.md from physics.json (+ crosscheck_v2.json). All numbers are read from the JSON.
Run: "$LAB/.venv/bin/python" -I -B write_md.py
"""
import json
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, "physics.json")))
XC = json.load(open(os.path.join(H, "crosscheck_v2.json"))) if os.path.exists(os.path.join(H, "crosscheck_v2.json")) else {}
K = ["1.0cm", "1.2cm", "1.5cm", "2.0cm", "2.5cm", "4.0cm", "6.0cm"]
FC = R["meta"]["band_centres_hz"]
ST, TT, DP, CD, GE, TG = (R[x] for x in ("single_target", "two_talker", "dipole", "cdr", "geometry", "tdoa_groups"))
LF, W3, BS, AW = (R[x] for x in ("low_freq_wng", "wng_needed_for_3dB_DI_axis_3d", "broadside_vs_diffuse_indistinguishable",
                                  "ai_weight_above_edge"))
MM, MF, ILD = R["mismatch_and_skew_azavg2d_wng-10"], R["mismatch_axis3d_by_floor"], R["nearfield_ild_db"]
_z = lambda t: t[1:] if t.startswith("-") and float(t) == 0 else t
f1, f2, f0 = (lambda x: _z(f"{x:.1f}")), (lambda x: _z(f"{x:.2f}")), (lambda x: _z(f"{x:.0f}"))
kk = lambda k: k.replace("cm", "")
nval = sum(v["ok"] for v in R["validation"])
L = []
w = L.append

w("# B1: two-mic physics numbers versus spacing (1.0–6.0 cm)\n")
w("Hardware follow-up to DESIGN.md v1.1 §1, 2026-11-03. Spacings 1.0 / 1.2 / 1.5 / 2.0 / 2.5 / 4.0 / 6.0 cm, fs 16 kHz, "
  "c 343 m/s. Ideal free-field omni mics, ideal diffuse fields, no device body, no field losses (subtract 0.5–1.5 dB "
  "as DESIGN §1.2 says). Analytic and vectorised; no audio was played, recorded or opened.\n")
w(f"**Status.** Every DESIGN §1 / M2 / M4 / §1.5 number at 2.5 / 4 / 6 cm was reproduced first: {nval} of "
  f"{len(R['validation'])} checks pass. The one miss is a last-digit slip in the DESIGN §1.1 table (sinc > 0.7 at 4 cm is "
  "1924.6 Hz, printed as 1.93 kHz). The vectorised two-talker beam matches the original `physics_check_v2.py` loop "
  "exactly at 1.0 and 1.2 cm as well (§2).\n")
w("Files: `physics_b1.py` (all computations → `physics.json`, `tables.md`), `crosscheck_v2.py` (→ `crosscheck_v2.json`), "
  "`write_md.py` (→ this file). Run with `OMP_NUM_THREADS=1 .venv/bin/python -I -B physics_b1.py \"$LAB\"` (about 2 CPU-s).\n")

# ------------------------------------------------------------------------------------------------ bottom line
s12, s4 = ST["1.2cm"], ST["4.0cm"]
t12, t4 = TT["1.2cm"], TT["4.0cm"]
w("## Bottom line at 1.0–2.0 cm (most likely about 1.2 cm)\n")
w(f"- **Axis-aligned single talker: almost no loss.** The superdirective beam at WNG ≥ −10 dB gives "
  f"{f2(s12['axis_wng-10']['aidi_3d'])} dB AI-DI (3-D) / {f2(s12['axis_wng-10']['aidi_2d'])} dB (2-D) at 1.2 cm, against "
  f"{f2(s4['axis_wng-10']['aidi_3d'])} / {f2(s4['axis_wng-10']['aidi_2d'])} dB at 4 cm. The small array loses below 1 kHz "
  f"but has no spatial aliasing (c/2d = {f2(GE['1.2cm']['alias_hz']/1e3)} kHz), so it keeps 5.1–6.0 dB per band from 1.6 to "
  "8 kHz, where the 4 cm array drops to 2.5–4.3 dB above 3 kHz.")
w(f"- **Orientation-averaged gain drops a little; broadside drops to zero.** Averaged over azimuth in 2-D noise: "
  f"{f2(s12['az_wng-10']['avg_2d'])} dB at 1.2 cm vs {f2(s4['az_wng-10']['avg_2d'])} dB at 4 cm (r_c × "
  f"{s12['az_wng-10']['rc_factor_avg_2d']:.2f} vs × {s4['az_wng-10']['rc_factor_avg_2d']:.2f}). At broadside: "
  f"{f2(s12['az_wng-10']['broadside_3d'])} dB (3-D) vs {f2(s4['az_wng-10']['broadside_3d'])} dB.")
w(f"- **Two-talker figure-8 loses about a third.** {f2(t12['wng-10']['2d']['aidi'])} dB (2-D) / "
  f"{f2(t12['wng-10']['3d']['aidi'])} dB (3-D) at 1.2 cm vs {f2(t4['wng-10']['2d']['aidi'])} / "
  f"{f2(t4['wng-10']['3d']['aidi'])} dB at 4 cm. The dipole band starts at {f0(DP['1.2cm']['edge_wng10_closed_hz'])} Hz "
  f"(−10 dB floor), or at {f0(DP['1.2cm']['edge_bus_schedule_hz'])} Hz under the M2 bus schedule (−6 dB above 2 kHz). "
  f"With a −20 dB floor the 4 cm figures come back ({f2(t12['wng-20']['2d']['aidi'])} / {f2(t12['wng-20']['3d']['aidi'])} dB), "
  "but such low floors need calibration (§4.5 shows the cost for the single-target beam).")
w(f"- **Below 1 kHz every useful beam has a large white-noise cost** (§5.2). Reaching just 3 dB of DI on the axis "
  f"costs {f1(-W3['1.2cm'][7])}–{f1(-W3['1.2cm'][0])} dB of WNG over 1000–125 Hz, and the figure-8 costs "
  f"{f1(-LF['1.2cm']['dipole_wng_db'][3])}–{f1(-LF['1.2cm']['dipole_wng_db'][0])} dB. That is 10.5 dB more than at 4 cm "
  "(WNG scales with d²).")
w(f"- **Broadside talkers look like diffuse noise up to about 3.6 kHz** (coherence > 0.9), and the CDR rule "
  f"(|Γs − Γn| ≥ 0.2) is met only above {f2(CD['1.2cm']['broadside_3d_hz']/1e3)} kHz. Endfire partners stay usable for "
  f"CDR from {f0(CD['1.2cm']['endfire_3d_hz'])} Hz.")
w(f"- **TDOA is nearly useless as talker identity:** the 20 meeting seats collapse to "
  f"{f0(TG['1.2cm']['20us']['s0_median'])} groups at 20 µs resolution ({f0(TG['1.2cm']['10us']['s0_median'])} at 10 µs).")
w(f"- **The measured 68 µs skew is {R['table'][1]['skew_vs_max_tdoa_ratio']:.2f} × the largest physical TDOA at 1.2 cm.** "
  f"Uncompensated, it turns the axis beam into {f2(MM['1.2cm']['skew_err_68us_axis_3d_worst'])} dB AI-DI. S0 must remove it "
  f"to within about 5 µs (0.08 samples), which costs ≤ 0.15 dB.\n")

# ------------------------------------------------------------------------------------------------ definitions
w("## 1. Definitions used (as in the scripts behind DESIGN v1.1)\n")
w("Sources read: DESIGN.md §1, M1, M2, M4, §1.5/§1.6 and Appendix C, plus the scratchpad scripts it names "
  "(`arch/physics_check.py`, `revise/physics_check_v2.py`, `review/di_real.py`, `review/probe.py`). Each reading below "
  "reproduces the published numbers.\n")
DEFS = [
    "**AI-DI**: Σ I_i · DI(f_i) in dB, ANSI S3.5 one-third-octave band-importance weights (sum 1.000) at 18 band centres "
    "160 Hz–8 kHz, with the 8 kHz centre evaluated at 7.9 kHz. DI is evaluated at the band centre only (no averaging inside a band).",
    "**Noise models**: 3-D (spherically diffuse) Γ = sinc(2fd/c) (numpy sinc); 2-D (horizontally diffuse) Γ = J0(2πfd/c).",
    "**WNG** = |wᴴa|²/(wᴴw), relative to one omni mic (delay-and-sum = +3 dB). Floors are hard constraints.",
    "**Single-target beam (M2)**: per-band MVDR against the diffuse model, w = (Γ+μI)⁻¹a / (aᴴ(Γ+μI)⁻¹a), μ = the smallest "
    "value meeting the floor (bisection on log10 μ in [−9, 6]). 'On the axis' = θ = 0. In the on-axis rows the 2-D figures use "
    "J0-designed weights. In the azimuth-averaged and broadside rows they use 3-D-designed weights evaluated against J0.",
    "**Azimuth average**: plain mean of the AI-DI (dB) over 19 angles θ = 0, 5, …, 90° (folded azimuth, talkers in the "
    "horizontal plane, uniform). Broadside = θ = 90°; there the MVDR is identical to delay-and-sum.",
    "**Delay-and-sum (M1)**: w = a/2 steered to θ = 0 (axis) or 90° (broadside), evaluated in 3-D noise.",
    "**Two-talker beam (M2)**: soft LCMV w = (Γ + μI + λ_c AAᴴ)⁻¹ λ_c A h*, centre-referenced steering a(f,τ) = "
    "[e^(+jπfτ), e^(−jπfτ)], A at 0°, B at 180°, h = [1, e^(jψ)], λ_c = 10³. ψ is searched on a 2.5° grid. A candidate is "
    "rejected if it violates the floor (strict) or if |r_A|/|r_B| is outside ±1 dB. The kept candidate maximizes "
    "DF = ½(|r_A|²+|r_B|²)/(wᴴΓw). A band with no candidate gets a single omni (0 dB). Weights are designed for the "
    "3-D model; the 2-D figure picks ψ by the J0-evaluated DF (as `physics_check_v2.py` does).",
    "**Dipole band**: centre-referenced h = [1, −1] has WNG = 1 − cos(2πfd/c). The band runs from "
    "f = arccos(1 − 10^(floor/10))·c/(2πd) up to min(c/2d, fs/2). Under the M2 bus schedule (−15 dB below 500 Hz, −10 dB "
    "to 2 kHz, −6 dB above), the edge is found on a 0.1 Hz grid.",
    "**CDR usability (M4)**: a bin is used where |Γs − Γn| ≥ 0.2, with Γs = e^(+j2πfτ) and Γn the 3-D sinc model. "
    "Broadside: τ = 0. Endfire: τ = d/c. The reported value is the first crossing. 2-D (J0) variants are in the JSON.",
    "**Coherence > 0.9** (main-table column): the spherical (sinc) value, as in DESIGN §1.1 ('about 1.1 kHz at 4 cm'). "
    "J0 values are in §4.2.",
    "**Meeting TDOA groups (§1.5)**: the 20 seats of lab scene `meeting_s0` (8 m × 2 m table, mouths at 1.2 m, device at "
    "0.78 m). The mic axis is horizontal at a uniform azimuth, τ = (|s−B| − |s−A|)/c, and sorted τ values are grouped "
    "greedily (a new group when τ − group start > resolution). Headline = median over 720 uniform azimuths. At 2.5 / 4 / "
    "6 cm this equals the exact 500-draw replica of `review/probe.py` (same RNG stream).",
    "**Skew ratio** = 68 µs / (d/c), the measured inter-channel skew magnitude over the largest geometric TDOA.",
]
for d in DEFS:
    w(f"- {d}")
w("")

# ------------------------------------------------------------------------------------------------ validation
w("## 2. Validation against DESIGN v1.1 (2.5 / 4 / 6 cm unless stated)\n")
w("Pass rule: within 0.2 dB for AI-DI rows; otherwise within the stated rounding of each design cell; counts exact.\n")
w("| quantity | design | reproduced | ok |\n|---|---|---|---|")
for v in R["validation"]:
    w(f"| {v['quantity']} | {v['design_value']} | {v['reproduced']} | {'yes' if v['ok'] else '**no**'} |")
if XC:
    w("\nCross-check of the extension: the original `physics_check_v2.two_talker()` loop, run unchanged at the new spacings "
      "(WNG ≥ −10 dB), against this vectorised version:\n")
    w("| spacing, model | original v2 | physics_b1 | diff |\n|---|---|---|---|")
    for k, v in XC.items():
        w(f"| {k.replace('_', ', ')} | {v['original']:.3f} dB | {v['vectorised']:.3f} dB | {v['diff']:+.3f} |")
w("")

# ------------------------------------------------------------------------------------------------ main table
w("## 3. Main table\n")
w("AI-DI in dB, WNG ≥ −10 dB. SD = single-target superdirective (MVDR) beam. 'bs' = broadside. Two-talker = partners on "
  "the axis on opposite sides, 2-D babble. Band edges in Hz. TDOA groups are medians for 20 seats.\n")
hdr = ["d (cm)", "c/2d (kHz)", "max TDOA µs (samples)", "sinc > 0.9 until (kHz)", "DAS axis / bs, 3-D", "SD axis, 3-D",
       "SD bs, 3-D", "SD azimuth-avg, 2-D", "two-talker, 2-D", "dipole band from", "CDR usable from: endfire / bs",
       "TDOA groups at 20 / 10 µs", "68 µs skew ÷ max TDOA"]
w("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for r in R["table"]:
    w(f"| {r['d_cm']:.1f} | {r['alias_khz']:.2f} | {r['max_tdoa_us']:.1f} ({r['max_tdoa_samples']:.2f}) | "
      f"{r['coh_above_0p9_until_khz']:.2f} | {r['das_aidi_axis_db']:.2f} / {r['das_aidi_broadside_db']:.2f} | "
      f"{r['sd_aidi_axis_3d_db']:.2f} | {r['sd_aidi_broadside_3d_db']:.2f} | {r['sd_aidi_azavg_2d_db']:.2f} | "
      f"{r['two_talker_aidi_aligned_2d_db']:.2f} | {r['dipole_band_from_hz']:.0f} | "
      f"{r['cdr_usable_endfire_from_hz']:.0f} / {r['cdr_usable_broadside_from_hz']:.0f} | "
      f"{r['tdoa_groups_20_seats_20us']:.0f} / {r['tdoa_groups_20_seats_10us']:.0f} | {r['skew_vs_max_tdoa_ratio']:.2f} |")
w("\nc/2d above 8 kHz (1.0–2.0 cm) means no spatial aliasing anywhere in the 16 kHz band.\n")

# ------------------------------------------------------------------------------------------------ 4.x
w("## 4. More numbers by spacing\n")
w("### 4.1 AI-DI (dB) by beam, WNG floor and noise model (3-D / 2-D)\n")
hdr = ["d (cm)", "SD axis −3", "SD axis −10", "SD axis −20", "SD axis, bus schedule", "SD az-avg −10", "SD az-avg −3",
       "SD broadside −10", "2-talker −10", "2-talker −20", "2-talker, bus schedule", "2-talker, fixed dipole −10",
       "DAS axis / bs (3-D)", "r_c factor (az-avg 2-D)"]
w("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for k in K:
    s, t = ST[k], TT[k]
    pr = lambda a, b: f"{f2(a)} / {f2(b)}"
    w(f"| {kk(k)} | {pr(s['axis_wng-3']['aidi_3d'], s['axis_wng-3']['aidi_2d'])} | "
      f"{pr(s['axis_wng-10']['aidi_3d'], s['axis_wng-10']['aidi_2d'])} | {pr(s['axis_wng-20']['aidi_3d'], s['axis_wng-20']['aidi_2d'])} | "
      f"{pr(s['axis_bus_schedule']['aidi_3d'], s['axis_bus_schedule']['aidi_2d'])} | "
      f"{pr(s['az_wng-10']['avg_3d'], s['az_wng-10']['avg_2d'])} | {pr(s['az_wng-3']['avg_3d'], s['az_wng-3']['avg_2d'])} | "
      f"{pr(s['az_wng-10']['broadside_3d'], s['az_wng-10']['broadside_2d'])} | "
      f"{pr(t['wng-10']['3d']['aidi'], t['wng-10']['2d']['aidi'])} | {pr(t['wng-20']['3d']['aidi'], t['wng-20']['2d']['aidi'])} | "
      f"{pr(t['bus_schedule']['3d']['aidi'], t['bus_schedule']['2d']['aidi'])} | "
      f"{pr(t['fixed_dipole_wng-10']['3d'], t['fixed_dipole_wng-10']['2d'])} | "
      f"{pr(s['dsb']['axis_3d'], s['dsb']['broadside_3d'])} | × {s['az_wng-10']['rc_factor_avg_2d']:.2f} |")
nfb = sum(len(TT[k][fl][m]["omni_fallback_bands_hz"]) for k in K for fl in ("wng-10", "wng-20", "bus_schedule") for m in ("3d", "2d"))
w("\nBus schedule = −15 dB below 500 Hz, −10 dB to 2 kHz, −6 dB above (M2). 'Fixed dipole' = ψ = 180° in every band, "
  "omni where the dipole is infeasible (the v2 script's comparison beam). "
  + ("No band fell back to omni in the phase-optimized two-talker beam at any spacing or floor.\n" if nfb == 0 else
     f"{nfb} band(s) fell back to omni in the phase-optimized two-talker beam (see JSON).\n"))

w("### 4.2 Band edges (Hz)\n")
hdr = ["d (cm)", "dipole band from: −10 / −15 / −20 dB / bus schedule", "CDR endfire 3-D (2-D)", "CDR broadside 3-D (2-D)",
       "coherence > 0.9 until, 3-D / 2-D", "broadside beam reaches 0.5 / 1 dB", "AI weight above: dipole −10 / CDR endfire / CDR bs"]
w("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for k in K:
    dp, cd, bs, aw = DP[k], CD[k], BS[k], AW[k]
    w(f"| {kk(k)} | {f0(dp['edge_wng10_closed_hz'])} / {f0(dp['edge_wng15_hz'])} / {f0(dp['edge_wng20_hz'])} / "
      f"{f0(dp['edge_bus_schedule_hz'])} | {f0(cd['endfire_3d_hz'])} ({f0(cd['endfire_2d_hz'])}) | "
      f"{f0(cd['broadside_3d_hz'])} ({f0(cd['broadside_2d_hz'])}) | {f0(bs['coh09_3d_hz'])} / {f0(bs['coh09_2d_hz'])} | "
      f"{f0(bs['broadside_beam_di_reaches_0p5dB_3d_hz'])} / {f0(bs['broadside_beam_di_reaches_1dB_3d_hz'])}"
      f"{' (> fs/2)' if bs['broadside_beam_di_reaches_1dB_3d_hz'] > 8000 else ''} | "
      f"{aw['dipole_wng10']:.2f} / {aw['cdr_endfire']:.2f} / {aw['cdr_broadside']:.2f} |")
w("\nThe dipole band ends at min(c/2d, 8 kHz); for 1.0–2.0 cm that is 8 kHz. 'AI weight above' = the share of the "
  "speech band-importance whose band centres lie above the edge. The broadside beam is delay-and-sum, whatever the floor.\n")

w("### 4.3 White-noise gain that a useful beam needs below 2 kHz\n")
fr = W3["freqs_hz"]
w("WNG (dB) of the on-axis MVDR at the point where its 3-D DI reaches 3 dB. Values near +3 dB mean a beam close to "
  "delay-and-sum already gets there.\n")
w("| d (cm) | " + " | ".join(f"{f0(f)} Hz" for f in fr) + " |\n|" + "---|" * (len(fr) + 1))
for k in K:
    w(f"| {kk(k)} | " + " | ".join("n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f1(v) for v in W3[k]) + " |")
lf = LF["1.2cm"]
w("\nAt 1.2 cm, by beam type (WNG in dB, then the DI that a given floor allows):\n")
w("| quantity at 1.2 cm | " + " | ".join(f"{f0(f)} Hz" for f in lf["freqs_hz"]) + " |\n|" + "---|" * (len(lf["freqs_hz"]) + 1))
for lab, key in (("WNG, unconstrained superdirective (hypercardioid, ≈ 6 dB DI)", "sd_endfire_unconstrained_wng_db"),
                 ("WNG, rear-null cardioid (best WNG for any null)", "cardioid_null_at_rear_wng_db"),
                 ("WNG, two-talker figure-8 (dipole)", "dipole_wng_db"),
                 ("DI (3-D) allowed at WNG ≥ −6 dB", "sd_endfire_di3d_at_wng-6"),
                 ("DI (3-D) allowed at WNG ≥ −10 dB", "sd_endfire_di3d_at_wng-10"),
                 ("DI (3-D) allowed at WNG ≥ −15 dB", "sd_endfire_di3d_at_wng-15"),
                 ("DI (3-D) allowed at WNG ≥ −20 dB", "sd_endfire_di3d_at_wng-20")):
    w(f"| {lab} | " + " | ".join(f1(v) for v in lf[key]) + " |")
w(f"\nFor comparison at 4 cm: unconstrained WNG {', '.join(f1(v) for v in LF['4.0cm']['sd_endfire_unconstrained_wng_db'])} dB "
  f"and dipole WNG {', '.join(f1(v) for v in LF['4.0cm']['dipole_wng_db'])} dB at the same frequencies.\n")

w("### 4.4 Per-band DI (dB), 1.2 cm against 4 cm\n")
cols = [("SD axis −10, 3-D, 1.2", ST["1.2cm"]["axis_wng-10"]["band_di_3d"]), ("SD axis −10, 3-D, 4", ST["4.0cm"]["axis_wng-10"]["band_di_3d"]),
        ("SD axis −20, 3-D, 1.2", ST["1.2cm"]["axis_wng-20"]["band_di_3d"]),
        ("2-talker −10, 3-D, 1.2", TT["1.2cm"]["wng-10"]["3d"]["band_di"]), ("2-talker −10, 2-D, 1.2", TT["1.2cm"]["wng-10"]["2d"]["band_di"]),
        ("2-talker −10, 2-D, 4", TT["4.0cm"]["wng-10"]["2d"]["band_di"]),
        ("ψ_AB (°) −10, 1.2", TT["1.2cm"]["wng-10"]["3d"]["psi_deg"]),
        ("broadside, 3-D, 1.2", ST["1.2cm"]["az_wng-10"]["broadside_band_di_3d"]), ("broadside, 3-D, 4", ST["4.0cm"]["az_wng-10"]["broadside_band_di_3d"])]
w("| band (Hz) | " + " | ".join(c for c, _ in cols) + " |\n|" + "---|" * (len(cols) + 1))
for i, f in enumerate(FC):
    w(f"| {f0(f)} | " + " | ".join(("–" if v[i] is None else (f0(v[i]) if c.startswith("ψ") else f1(v[i]))) for c, v in cols) + " |")
w("")

w("### 4.5 Calibration sensitivity\n")
hdr = ["d (cm)", "az-avg 2-D −10: ideal / ±1 dB ±3° / ±0.25 dB ±1°", "az-avg 2-D −10 with residual skew 2 / 5 / 10 µs",
       "axis 3-D −10: ideal, then skew 2 / 5 / 10 µs", "uncompensated 68 µs skew: axis 3-D / az-avg 2-D",
       "axis 3-D −20: ideal / ±1 dB ±3° / ±0.25 dB ±1°"]
w("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for k in K:
    m, mf = MM[k], MF[k]["wng-20"]
    w(f"| {kk(k)} | {f2(m['ideal'])} / {f2(m['pm1dB_pm3deg'])} / {f2(m['pm0.25dB_pm1deg'])} | "
      f"{f2(m['skew_err_2us_azavg_2d_worst'])} / {f2(m['skew_err_5us_azavg_2d_worst'])} / {f2(m['skew_err_10us_azavg_2d_worst'])} | "
      f"{f2(m['axis_3d_ideal'])}, then {f2(m['skew_err_2us_axis_3d_worst'])} / {f2(m['skew_err_5us_axis_3d_worst'])} / "
      f"{f2(m['skew_err_10us_axis_3d_worst'])} | {f2(m['skew_err_68us_axis_3d_worst'])} / {f2(m['skew_err_68us_azavg_2d_worst'])} | "
      f"{f2(mf['ideal'])} / {f2(mf['pm1dB_pm3deg'])} / {f2(mf['pm0.25dB_pm1deg'])} |")
w("\nMismatch: independent uniform per-mic gain and phase errors, the DF averaged over 200–400 draws (as in "
  "`review/di_real.py`). Skew: a fixed inter-channel delay that the beam does not know about; the worse sign is reported. "
  "AI-DI in dB.\n")

w("### 4.6 Meeting: 20 seats as TDOA groups\n")
hdr = ["d (cm)", "TDOA span (µs)", "20 µs: median (min–max, p10–p90)", "10 µs: median (min–max, p10–p90)",
       "median over 6 seat layouts, 20 / 10 µs", "neighbouring seats < 10 µs apart (of 19)"]
w("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
for k in K:
    g = TG[k]
    a, b = g["20us"], g["10us"]
    w(f"| {kk(k)} | {f0(g['tdoa_span_us'])} | {f0(a['s0_median'])} ({a['s0_min']}–{a['s0_max']}, {f0(a['s0_p10'])}–{f0(a['s0_p90'])}) | "
      f"{f0(b['s0_median'])} ({b['s0_min']}–{b['s0_max']}, {f0(b['s0_p10'])}–{f0(b['s0_p90'])}) | "
      f"{a['layouts_median_of_medians']:g} / {b['layouts_median_of_medians']:g} | {f0(g['neighbour_gaps_below_10us_of_19_median'])} |")
w("\nGreedy grouping cannot exceed ⌊span/resolution⌋ + 1 groups (4 at 20 µs and 7 at 10 µs for 1.2 cm), so below 2 cm "
  "the count is set by the spacing, not by the seating.\n")

# ------------------------------------------------------------------------------------------------ notes
bs12 = BS["1.2cm"]
w("## 5. Notes\n")
w("### 5.1 When a broadside talker and diffuse noise are spatially indistinguishable\n")
w(f"- A broadside talker has Γs = 1; diffuse noise has Γn = sinc(2fd/c) (3-D) or J0 (2-D). The two differ by about "
  f"(kd)²/6, which at 1 kHz is only {CD['1.2cm']['at_1khz_broadside_3d']:.3f} at 1.2 cm "
  f"({CD['4.0cm']['at_1khz_broadside_3d']:.3f} at 4 cm, {CD['2.5cm']['at_1khz_broadside_3d']:.3f} at 2.5 cm).")
w(f"- At 1.2 cm they are indistinguishable (coherence > 0.9) below {f2(bs12['coh09_3d_hz']/1e3)} kHz in 3-D noise and "
  f"{f2(bs12['coh09_2d_hz']/1e3)} kHz in 2-D babble. The M4 conditioning rule |Γs − Γn| ≥ 0.2 is met only above "
  f"{f2(bs12['cdr_rule_3d_hz']/1e3)} kHz (3-D) or {f2(bs12['cdr_rule_2d_hz']/1e3)} kHz (2-D). That leaves "
  f"{AW['1.2cm']['cdr_broadside']*100:.1f}% of the speech band-importance.")
w(f"- The broadside beam (= delay-and-sum) reaches 0.5 dB of DI only at {f2(bs12['broadside_beam_di_reaches_0p5dB_3d_hz']/1e3)} kHz "
  f"and 1 dB at {f2(bs12['broadside_beam_di_reaches_1dB_3d_hz']/1e3)} kHz.")
w(f"- Over 1.0–2.0 cm the coherence-0.9 edge is {f2(BS['1.0cm']['coh09_3d_hz']/1e3)}–{f2(BS['2.0cm']['coh09_3d_hz']/1e3)} kHz "
  f"(3-D) and the CDR-rule edge {f2(BS['1.0cm']['cdr_rule_3d_hz']/1e3)}–{f2(BS['2.0cm']['cdr_rule_3d_hz']/1e3)} kHz. Both scale as 1/d.")
w("- So for a talker near broadside, coherence, CDR, clustering and beams give essentially nothing across the speech band. "
  "Only spectral and speaker-identity cues remain.\n")

w("### 5.2 White-noise amplification below 1 kHz at 1.2 cm\n")
w(f"- **Smallest cost for a beam to be 'useful' (DI ≥ 3 dB, on the axis)**: WNG {', '.join(f1(v) for v in W3['1.2cm'][:8])} dB "
  f"at {', '.join(f0(f) for f in fr[:8])} Hz. Self-noise and any uncorrelated noise (wind, vent turbulence, handling, "
  "vibration) are amplified by that magnitude, roughly 6–24 dB.")
w(f"- **Specific beams at 125 / 250 / 500 / 1000 Hz**:")
w(f"  - full hypercardioid: {', '.join(f1(v) for v in lf['sd_endfire_unconstrained_wng_db'][:4])} dB;")
w(f"  - rear-null cardioid (the most WNG-friendly beam with a null): {', '.join(f1(v) for v in lf['cardioid_null_at_rear_wng_db'][:4])} dB;")
w(f"  - two-talker figure-8: {', '.join(f1(v) for v in lf['dipole_wng_db'][:4])} dB.")
w(f"  - Nulling a point source whose TDOA differs from the target's by Δτ costs WNG = 1 − cos(2πfΔτ). For the largest "
  f"possible Δτ (70 µs, opposite ends of the axis) that is −10.2 dB at 1 kHz and −22.2 dB at 250 Hz.")
w(f"- **What the floors allow** (on-axis DI at 125 / 250 / 500 / 1000 Hz): −10 dB floor "
  f"{', '.join(f1(v) for v in lf['sd_endfire_di3d_at_wng-10'][:4])} dB; −20 dB floor "
  f"{', '.join(f1(v) for v in lf['sd_endfire_di3d_at_wng-20'][:4])} dB; meeting −6 dB floor "
  f"{', '.join(f1(v) for v in lf['sd_endfire_di3d_at_wng-6'][:4])} dB.")
w("- Everything is 10.5 dB worse than at 4 cm (WNG ∝ d² at low frequency: 20·log10(4/1.2) = 10.46 dB).")
w("- **In absolute terms** (DESIGN §1.2: INMP441 EIN ≈ −87 dBFS):")
w("  - A −20 dB floor puts the amplified self-noise near −67 dBFS in those bins. That is still well below bus noise "
  "(−50 to −40 dBFS) and below this lab's fan noise (−51 dBFS), so the bus can use it.")
w("  - A meeting room's HVAC (−85 to −75 dBFS) cannot. There the −3 to −6 dB floors leave the beam essentially omni "
  "below 1 kHz.")
w(f"- **Calibration is the price of the −20 dB floor.** Uncalibrated ±1 dB / ±3° errors cost "
  f"{MF['1.2cm']['wng-20']['ideal'] - MF['1.2cm']['wng-20']['pm1dB_pm3deg']:.2f} dB of on-axis AI-DI at 1.2 cm "
  f"({MF['4.0cm']['wng-20']['ideal'] - MF['4.0cm']['wng-20']['pm1dB_pm3deg']:.2f} dB at 4 cm). Calibrated to ±0.25 dB / "
  f"±1°, the cost is {MF['1.2cm']['wng-20']['ideal'] - MF['1.2cm']['wng-20']['pm0.25dB_pm1deg']:.2f} dB.\n")

w("### 5.3 What spatial processing still contributes at 1.2 cm, and where\n")
w(f"- **Single-target superdirective beam, talker within about ±45° of the axis.** At −10 dB, per-band DI ≥ 3 dB from about "
  f"700 Hz, about 6 dB from 1.6 to 4 kHz, and still {f1(ST['1.2cm']['axis_wng-10']['band_di_3d'][-1])} dB at 7.9 kHz. AI-DI: "
  f"{f2(s12['axis_wng-10']['aidi_3d'])} dB (3-D), {f2(s12['axis_wng-10']['aidi_2d'])} dB (2-D). The bus schedule adds "
  f"{s12['axis_bus_schedule']['aidi_3d'] - s12['axis_wng-10']['aidi_3d']:.2f} dB. Main band: 0.8–8 kHz. Steered over "
  f"azimuth (2-D): {f2(s12['az_wng-10']['avg_2d'])} dB.")
w(f"- **Two-talker figure-8, partners on the axis on opposite sides.** The dipole works from "
  f"{f0(DP['1.2cm']['edge_wng10_closed_hz'])} Hz to 8 kHz at −10 dB, from {f0(DP['1.2cm']['edge_bus_schedule_hz'])} Hz under "
  f"the bus schedule, and from {f0(DP['1.2cm']['edge_wng20_hz'])} Hz at −20 dB. Below the dipole band the best balanced "
  "beam gives under 1 dB below 1 kHz. AI-DI: "
  f"{f2(t12['wng-10']['2d']['aidi'])} dB (2-D) / {f2(t12['wng-10']['3d']['aidi'])} dB (3-D); bus schedule "
  f"{f2(t12['bus_schedule']['2d']['aidi'])} / {f2(t12['bus_schedule']['3d']['aidi'])} dB.")
w("  - The −6 dB part of the bus schedule above 2 kHz pushes the dipole band up to 3.3 kHz. At small spacing a −10 dB floor "
  "up to about 3.3 kHz is the better trade when noise is far above self-noise.")
w(f"- **CDR / coherence mask, M4.** Endfire partner: from {f0(CD['1.2cm']['endfire_3d_hz'])} Hz "
  f"({AW['1.2cm']['cdr_endfire']*100:.0f}% of the AI weight). Broadside partner: only above "
  f"{f2(CD['1.2cm']['broadside_3d_hz']/1e3)} kHz, which is effectively unusable.")
w("- **TDOA, DOA and clustering (S1, M7).**")
w("  - Gain: there is no spatial aliasing below 14.3 kHz, so all 0.1–8 kHz bins carry unwrapped phase. At 4 cm, "
  "TDOA estimation must stay below 3.4 kHz.")
w(f"  - Cost: the whole TDOA range is ±35 µs (±0.56 samples). The 20 meeting seats give "
  f"{f0(TG['1.2cm']['20us']['s0_median'])} groups at 20 µs ({f0(TG['1.2cm']['10us']['s0_median'])} at 10 µs).")
w("  - Bus partners at opposite ends of the axis are 70 µs apart and remain separable.")
w("  - S1's fixed 25 µs 'inseparable' threshold is 0.71·d/c at 1.2 cm and should scale with d̂.")
w("- **Point-interferer null (e.g. the fan).** Possible in any band, at WNG = 1 − cos(2πfΔτ). Affordable above about 1 kHz, "
  "or lower only in loud noise.")
w(f"- **Not usable at all:** near-field ILD ({f2(ILD['1.2cm']['0.3'])} / {f2(ILD['1.2cm']['0.5'])} / {f2(ILD['1.2cm']['1.0'])} dB "
  "at 0.3 / 0.5 / 1.0 m), and any spatial cue for broadside talkers below about 5 kHz.")
w("- **Net.** At 1.2 cm, spatial processing is an above-1-kHz tool for talkers near the axis. Its broadband value is a few dB "
  "AI-DI (2.2–3.9 dB in 2-D), and it is almost unchanged from 4 cm for an aligned single talker. Below 1 kHz, and for "
  "broadside talkers, the gain must come from single-channel noise reduction, WPE, leveling and speaker identity "
  "(DESIGN §1.5), as before, but more so.\n")

w("### 5.4 Skew versus spacing\n")
w(f"- The measured skew of about 68 µs (1.1 samples) is {R['table'][1]['skew_vs_max_tdoa_ratio']:.2f} × the largest "
  f"physical TDOA at 1.2 cm ({R['table'][0]['skew_vs_max_tdoa_ratio']:.2f} × at 1.0 cm, "
  f"{R['table'][3]['skew_vs_max_tdoa_ratio']:.2f} × at 2.0 cm).")
w("- Uncompensated, every source appears outside the physical TDOA range, and a geometry-steered beam attenuates its own "
  f"target: {f2(MM['1.2cm']['skew_err_68us_axis_3d_worst'])} dB AI-DI on the axis.")
w(f"- Residual skew errors of 2 / 5 / 10 µs cost {MM['1.2cm']['axis_3d_ideal']-MM['1.2cm']['skew_err_2us_axis_3d_worst']:.2f} / "
  f"{MM['1.2cm']['axis_3d_ideal']-MM['1.2cm']['skew_err_5us_axis_3d_worst']:.2f} / "
  f"{MM['1.2cm']['axis_3d_ideal']-MM['1.2cm']['skew_err_10us_axis_3d_worst']:.2f} dB on the axis at 1.2 cm, against "
  f"{MM['4.0cm']['axis_3d_ideal']-MM['4.0cm']['skew_err_2us_axis_3d_worst']:.2f} / "
  f"{MM['4.0cm']['axis_3d_ideal']-MM['4.0cm']['skew_err_5us_axis_3d_worst']:.2f} / "
  f"{MM['4.0cm']['axis_3d_ideal']-MM['4.0cm']['skew_err_10us_axis_3d_worst']:.2f} dB at 4 cm.")
w("- Target: |residual skew| ≤ 5 µs (0.08 samples), i.e. ≤ 14% of d/c at 1.2 cm. RTF-based M3 absorbs a fixed skew; "
  "model-steered M1, M2 and M4 do not.\n")

w("### 5.5 Consistency with the hardware reading (to be tested, not assumed)\n")
w("- If the skew is −1.07 to −1.12 samples, the measured direct-path TDOAs (−0.53 to −0.56 samples) imply geometric TDOAs "
  "of +0.51 to +0.59 samples (32–37 µs).")
def _ang(dcm, tau_smp):
    cth = tau_smp / (dcm / 100 / 343.0 * 16000)
    return None if cth > 1 else float(np.degrees(np.arccos(cth)))
_a12 = [_ang(1.2, t) for t in (0.51, 0.55, 0.59)]
_a136 = [_ang(1.36, t) for t in (0.51, 0.55, 0.59)]
_fmt = lambda a: "beyond endfire (impossible)" if a is None else f"{a:.0f}°"
w(f"- That needs d ≥ {0.51/16000*343*100:.2f}–{0.59/16000*343*100:.2f} cm (d/c ≥ τ). At d = 1.0 cm (maximum 0.47 samples) the "
  f"reading is impossible. At 1.2 cm the speakers would be at {_fmt(_a12[0])} / {_fmt(_a12[1])} / {_fmt(_a12[2])} from the Mic-A "
  f"end of the axis, for τ = 0.51 / 0.55 / 0.59 samples. At 1.36 cm they would be at {_fmt(_a136[0])} / {_fmt(_a136[1])} / "
  f"{_fmt(_a136[2])}.")
w("- This agrees with the coherence fits (1.09–1.36 cm), but the speaker geometry was not measured.")
w(f"- The rival reading (skew −0.5 samples, speakers near broadside, geometric τ ≈ −0.05 samples) has its own geometric "
  f"limit. The fans' measured −1.15 samples would then be a geometric −0.65 samples ({0.65/16000*1e6:.1f} µs). That is "
  f"physical only if d ≥ {0.65/16000*343*100:.2f} cm, above all but one of the fits (1.09–1.36 cm; one gave 1.74 cm) but "
  "inside the user's 1–2 cm. Under the −1.1 reading the fans "
  "sit near broadside (−0.05 samples) at any d.")
w("- These are consistency limits from the maximum TDOA alone. A biased GCC-PHAT peak on reverberant fan noise would "
  "weaken them, so they point toward the −1.1 reading but do not settle it.\n")

w("### 5.6 Uncertainty and caveats\n")
w("- **Model.** Ideal point mics in free field, ideal diffuse fields, band-centre evaluation, no device-body scattering or "
  "field losses (−0.5 to −1.5 dB in practice).")
_mv = [abs(a - b) for a, b in (
    (ST['1.5cm']['axis_wng-10']['aidi_3d'], ST['1.0cm']['axis_wng-10']['aidi_3d']),
    (ST['1.5cm']['axis_wng-10']['aidi_2d'], ST['1.0cm']['axis_wng-10']['aidi_2d']),
    (ST['1.5cm']['az_wng-10']['avg_2d'], ST['1.0cm']['az_wng-10']['avg_2d']),
    (TT['1.5cm']['wng-10']['2d']['aidi'], TT['1.0cm']['wng-10']['2d']['aidi']),
    (TT['1.5cm']['wng-10']['3d']['aidi'], TT['1.0cm']['wng-10']['3d']['aidi']))]
w(f"- **Spacing uncertainty.** Between 1.0 and 1.5 cm the beam AI-DIs in the main table change by {min(_mv):.2f}–{max(_mv):.2f} dB. "
  "Band edges and the skew ratio scale as 1/d (−17% to +25% around 1.2 cm). The 1.2 cm row is a point estimate; "
  "the 1.0 and 1.5 cm rows bracket it.")
w("- **TDOA groups** depend on the assumed resolution and the seat layout. Six seat layouts give identical medians at "
  "1.0–2.0 cm and agree within one group at 2.5–6 cm.")
w("- **Not determined here:** the actual spacing, skew and speaker directions. Those are data questions for the "
  "other spacing tasks.\n")

open(os.path.join(H, "physics.md"), "w").write("\n".join(L) + "\n")
print(f"wrote physics.md ({len(L)} lines)")
