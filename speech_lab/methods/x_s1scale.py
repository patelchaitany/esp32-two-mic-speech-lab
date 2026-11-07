"""Spacing-study helper (x_, study/SPACING_STUDY.md): run a method with S1's absolute-microsecond constants
(methods._core.tracker GRID_STEP 2 us, HIST_STEP 1 us, HIST_SIGMA 6 us, PEAK_SEP 25 us, POST_WIN 20 us; tuned
for 4 cm) scaled by s = d / 4 cm, d = calib_json spacing_m or info['spacing_m'] (s <= 1). At 4 cm nothing
changes. The constants are module globals read at call time; they are patched for the duration of one call
and always restored (try/finally), so other methods in the same process are unaffected. Defaults bound at
function definition (beams.STEER_HYST, rtf._resets tol) are not scaled; beams.NOISE_EXCL is passed explicitly
where a caller exposes it (x_s1sc_m2_meeting_beam). Question: is the 12 mm deficit of S1-steered methods a
threshold artefact (fixed by scaling) or physics (it stays)?"""
import contextlib

from methods._core import tracker

NAMES = ("GRID_STEP", "HIST_STEP", "HIST_SIGMA", "PEAK_SEP", "POST_WIN")
REF_D = 0.04


def scale_of(info):
    cj = info.get("calib_json") or {}
    d = cj.get("spacing_m") or info.get("spacing_m") or REF_D
    return min(1.0, float(d) / REF_D)


@contextlib.contextmanager
def scaled(s):
    old = {n: getattr(tracker, n) for n in NAMES}
    try:
        for n in NAMES:
            setattr(tracker, n, old[n] * s)
        yield
    finally:
        for n, v in old.items():
            setattr(tracker, n, v)
