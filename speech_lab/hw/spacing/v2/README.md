# V2 adversarial check of the B1 spacing physics table

Run with `OMP_NUM_THREADS=1 .venv/bin/python -I -B hw/spacing/v2/<script> <speech_lab path>` (each run takes under 10 s of CPU).

- `v2_physics.py` → `v2_physics.json`: the main recomputation, written from scratch. It uses two independent solvers for the WNG-floored single-target beam: the exact KKT closed form, and diagonal-loading bisection. The two agree to within 4e-13 dB. The script also covers azimuth averages, delay-and-sum, the two-talker soft LCMV, dipole and CDR edges, coherence edges, and TDOA groups for 6 lab meeting layouts.
- `v2_extra.py` → `v2_extra.json`: AI-DI at the meeting floors (−3 and −6 dB) and under the bus WNG schedule. It also covers band-averaged AI-DI, mismatch (analytic expected-power model), skew, the band over which the WNG floor binds, and the neighbouring-seat gaps at each scene's own azimuth.
- `v2_extra2.py` → `v2_extra2.json`: the remaining 2.5 / 4 / 6 cm validation rows (per-band DI, WNG, broadside, off-axis, Mic-A-referenced check) and B1's low-frequency claims at 1.2 cm.
- `v2_psisel.py`, `v2_conv.py`: test how much the two 2-D conventions matter (choosing psi by the J0 or the 3-D DF; J0-designed or 3-D-designed weights).

**Verdict:** every B1 table cell and validation row is reproduced, within 0.01 dB, 1 Hz, or exactly for counts.

**Gaps B1 leaves out:**
- **Meeting floor.** At the DESIGN meeting floor (−3 dB), 1.2 cm is 0.8 to 1.1 dB below 4 cm. At −10 dB the gap is only 0.27 to 0.44 dB.
- **Bus schedule.** With the bus WNG schedule left unchanged, the two-talker beam at 1.2 cm gives 1.41 dB (2-D), against 2.48 dB at 4 cm.
