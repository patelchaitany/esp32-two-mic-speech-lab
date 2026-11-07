# Baselines from the s0v2 night run (moved out of results/)

`hw/realeval.py` runs of the plain methods `m2_bus_beam` (variants measured, wrong4cm) and `m4_cdr` (measured) on the
6 bus scenes `snr+0`, made 2026-11-03 02:27–02:33 for the x_s0v2 A/B (`methods/x_s0v2.md` §3.6).

They were moved here from `results/<variant>/<method>/` so that a later `realeval.py run --skip-existing` does not
silently reuse them after those methods are re-tuned. The x_auto_* and x_gnfix_* results stay in `results/`, because
their names are unique.
