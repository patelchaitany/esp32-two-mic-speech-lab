"""Tests for the DEMAND real-noise loader (lab.noise_real). Run: .venv/bin/python -m tests.test_noise_real

About 15 s. The geometry test re-fits the array from 120 s of TBUS and PCAFETER (coherence vs sinc), so it
checks the recordings themselves, not only the constants embedded in lab/noise_real.py.
"""
import atexit
import json
import os
import shutil
import tempfile
import time

import numpy as np

from lab import noise_real as nr
from lab.common import FS, PA_TO_DIG, spl_to_pa

_GEOM = []


def _geometry():
    """fit_geometry on 120 s of TBUS + PCAFETER (computed once per process)."""
    if not _GEOM:
        _GEOM.append(nr.fit_geometry(("TBUS", "PCAFETER"), seconds=120))
    return _GEOM[0]


def test_files_and_lengths():
    for env in nr.ENVS:
        assert nr.n_samples_of(env) == 4800064, env
        for c in range(1, 17):
            assert os.path.exists(nr.channel_path(env, c))


def test_pairs_listing():
    nn = nr.pairs()
    assert len(nn) == 33 and len({p["pair"] for p in nn}) == 33
    assert all(p["design_m"] == 0.05 and p["pair"][0] < p["pair"][1] for p in nn)
    assert {p["angle_deg"] for p in nn} <= {0.0, -60.0, -120.0}
    assert sum(p["angle_deg"] == 0.0 for p in nn) == 12  # along the rows; 21 diagonal
    # documented layout: largest distance 21.8 cm (DEMAND.pdf), 120 pairs in total
    D = nr.design_distances()
    assert abs(D.max() - 0.218) < 5e-4 and len(nr.pairs(1.0)) == 120
    # embedded fitted distances are within +-1 cm of the embedded fitted grid
    grid = nr.design_distances(nr.FIT["spacing_m"])
    for p in nr.pairs(1.0):
        a, b = p["pair"]
        assert abs(p["fitted_m"] - grid[a - 1, b - 1]) < 0.015, p
    for p in nn:
        a, b = p["pair"]
        assert abs(p["fitted_m"] - grid[a - 1, b - 1]) < 0.01, p


def test_geometry_from_data():
    """Coherence-fitted distances on fresh data: 5 cm pairs within +-1 cm of the fitted grid, the layout
    (and channel numbering, up to its 180-degree symmetry) recovered blindly by MDS."""
    g = _geometry()
    assert 0.045 < g["grid"]["spacing_m"] < 0.055, g["grid"]
    D = np.array(g["dist_m"])
    grid = nr.design_distances(g["grid"]["spacing_m"])
    for p in nr.pairs():
        a, b = p["pair"]
        assert abs(D[a - 1, b - 1] - grid[a - 1, b - 1]) < 0.01, (p["pair"], D[a - 1, b - 1])
    iu = np.triu_indices(16, 1)
    assert np.abs(D[iu] - grid[iu]).max() < 0.015
    assert g["mds"]["blind_matches_docs"] and g["mds"]["pos_err_cm_max"] < 1.0, g["mds"]
    # the embedded FIT (full 300 s, three environments) agrees with this independent excerpt
    F = np.array(nr.FIT["dist_m"])
    for p in nr.pairs():
        a, b = p["pair"]
        assert abs(F[a - 1, b - 1] - D[a - 1, b - 1]) < 0.01


def test_sinc_beats_j0_on_bus():
    """TBUS coherence follows the 3-D diffuse model (sinc) better than the 2-D one (J0)."""
    S, lev = nr.block_spectra("TBUS", 120)
    G = nr.coherence(S, nr.diffuse_blocks(S, lev))
    es, ej = [], []
    for p in nr.pairs()[:10]:
        a, b = p["pair"]
        es.append(nr.fit_distance(G[a - 1, b - 1], "sinc")[2])
        ej.append(nr.fit_distance(G[a - 1, b - 1], "j0")[2])
    assert np.mean(es) < 0.8 * np.mean(ej), (np.mean(es), np.mean(ej))


def test_real_noise_determinism():
    a = nr.real_noise("TBUS", 32000, np.random.default_rng(5))
    b = nr.real_noise("tbus", 32000, np.random.default_rng(5))
    c = nr.real_noise("TBUS", 32000, np.random.default_rng(6))
    assert a.shape == (2, 32000) and a.dtype == np.float32
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    x1, i1 = nr.real_noise("PCAFETER", 16000, 11, return_info=True)
    x2, i2 = nr.real_noise("PCAFETER", 16000, 11, return_info=True)
    assert np.array_equal(x1, x2) and i1 == i2
    assert tuple(sorted(i1["pair"])) in {p["pair"] for p in nr.pairs()}  # pair=None draws a 5 cm pair


def test_real_noise_content():
    """Rows are the requested channels at the drawn offset, with the sensitivity gains removed."""
    x, info = nr.real_noise("OMEETING", 8000, np.random.default_rng(3), pair=(7, 3), return_info=True)
    (off, n), = info["segments"]
    raw = nr.load_env("OMEETING", off, off + n, (7, 3))
    assert info["pair"] == (7, 3) and n == 8000
    g = 10 ** (-nr.GAIN_DB[[6, 2]] / 20)
    assert np.allclose(x, raw * g[:, None].astype(np.float32), atol=1e-7)
    y = nr.real_noise("OMEETING", 8000, np.random.default_rng(3), pair=(7, 3), equalize=False)
    assert np.array_equal(y, raw)
    try:
        nr.real_noise("TBUS", 100, 0, pair=(3, 3))
    except ValueError:
        return
    raise AssertionError("pair (3, 3) accepted")


def test_long_request_crossfade():
    """Longer than the 300 s recording: chained segments, exact length, no level jump at the joins."""
    N = nr.n_samples_of("PCAFETER")
    n = N + 20 * FS
    x, info = nr.real_noise("PCAFETER", n, np.random.default_rng(1), pair=(1, 2), return_info=True)
    assert x.shape == (2, n) and np.all(np.isfinite(x)) and len(info["segments"]) == 2
    j = info["segments"][0][1]  # first join ends here (crossfade over the 1 s before)
    lv = lambda s: 10 * np.log10(np.mean(x[0, s] ** 2))  # noqa: E731
    around = [lv(slice(j - 2 * FS + k * FS // 2, j - 2 * FS + (k + 1) * FS // 2)) for k in range(8)]
    ref = lv(slice(0, j - 2 * FS))
    assert max(abs(v - ref) for v in around) < 4.0, (ref, around)


def test_field_unit_level_and_hp():
    rng = np.random.default_rng(2)
    f = nr.field(5 * FS, FS, rng, env="TBUS", pair=[2, 6])
    assert f.shape == (2, 5 * FS) and f.dtype == np.float64
    assert abs(np.sqrt(np.mean(f[0] ** 2)) - 1.0) < 1e-9
    raw = nr.real_noise("TBUS", 10 * FS, np.random.default_rng(4), pair=(2, 6))
    hp = nr.real_noise("TBUS", 10 * FS, np.random.default_rng(4), pair=(2, 6), hp_hz=20.0)

    def low(v):  # share of row-0 power below 20 Hz (10 s -> 0.1 Hz bins)
        p = np.abs(np.fft.rfft(v[0])) ** 2
        return np.sum(p[:200]) / np.sum(p)

    # TBUS power is mostly below 20 Hz (peak near 13 Hz); a 4th-order 20 Hz high-pass removes most of it
    assert low(raw) > 0.3 and low(hp) < 0.15 * low(raw), (low(raw), low(hp))
    src = nr.field_source("tbus", 72.0, pair=(1, 2), hp_hz=20)
    assert src["type"] == "field" and src["weighting"] == "A" and src["signal"]["params"] == {
        "env": "TBUS", "pair": [1, 2], "hp_hz": 20.0}
    json.dumps(src)


def test_scene2_field_source():
    """A DEMAND field renders through lab.scene2 (public API) at the requested level, deterministically."""
    from lab import scene2 as s2
    root = tempfile.mkdtemp(prefix="noise_real_test_")
    atexit.register(shutil.rmtree, root, True)
    sid = s2.build_demo(root=root)
    rec = json.load(open(os.path.join(root, "demo", sid.split("/")[1] + ".tts.recipe.json")))
    geom, rirs = s2.load_geometry("demo", rec["geom_id"], root=root)
    T = 20 * FS
    rec.update(n_samples=T, rules=[], groups=[{"name": "real_noise", "role": "engine_road", "sources": [
        nr.field_source("TBUS", 70.0, pair=(5, 6), weighting="Z")]}])
    a, b = s2.render(geom, rirs, rec, impair=False), s2.render(geom, rirs, rec, impair=False)
    x = np.asarray(a["comp2"]["real_noise"], np.float64)
    assert np.array_equal(x, np.asarray(b["comp2"]["real_noise"]))
    lvl = 20 * np.log10(np.sqrt(np.mean(x[0] ** 2)) / PA_TO_DIG / spl_to_pa(0.0))
    assert abs(lvl - 70.0) < 0.05, lvl
    # the two rows keep a 5 cm-like coherence (Re part vs sinc fit on the rendered field)
    X = [np.fft.rfft(v[: T // 512 * 512].reshape(-1, 512) * np.hanning(512), axis=1) for v in x]
    g = np.mean(X[0] * X[1].conj(), 0) / np.sqrt(np.mean(abs(X[0]) ** 2, 0) * np.mean(abs(X[1]) ** 2, 0))
    d, rho, e = nr.fit_distance(g)
    assert 0.035 < d < 0.065, (d, rho, e)


if __name__ == "__main__":
    fails = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.2f} s)")
        except Exception as e:  # report every failure, then exit non-zero
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    raise SystemExit(1 if fails else 0)
