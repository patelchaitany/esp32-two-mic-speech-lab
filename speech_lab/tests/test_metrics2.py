"""Tests for the v2 metrics, statistics and bench runner (DESIGN Appendix B.5, B.6, B.9).
Run: .venv/bin/python -m tests.test_metrics2

Two small scenes are built in a temporary root with the real scene core (lab.scene2) from the generators
below, so scenes2/ and results2/ are not touched:
- bus: partners A/B alternating, 3 near passengers (two of them talking alone in a partner pause), a cone
  passenger, a conversing pair, far babble and engine/road noise at 0 dB SNR;
- meeting: 6 talkers at 1.0-4.8 m (one soft), 8 turns with 2 returning talkers, HVAC at 40 dBA and a fan.
"""
import atexit
import contextlib
import io
import json
import shutil
import tempfile
import time
import types

import numpy as np
from scipy.signal import lfilter

from lab import bench2, metrics2 as M, scene2 as s2, stats
from lab.common import FS
from methods import m0_mica_hpf

_ROOT = []


# ----------------------------------------------------------------------------- fabricated scenes
def _pos(center, r, az_deg, z):
    a = np.radians(az_deg)
    return [center[0] + r * np.cos(a), center[1] + r * np.sin(a), z]


def _utts(corpus, speaker, rs, n, max_dur=3.0):
    pool = [u for u in corpus.select(speaker=speaker) if 1.2 < u.duration < max_dur]
    return [pool[i] for i in rs.choice(len(pool), n, replace=len(pool) < n)]


def _speech(src, talker, level, events):
    s = {"type": "point", "src": src, "level_db": level, "signal": {"kind": "speech", "events": events}}
    return dict(s, talker=talker) if talker else s


def bus_recipe(geom, corpus, seed=0, snr_db=0.0):
    """Test bus generator: partner turns in [0.3, 5.4) s and [15.2, 19.7) s; near passengers n1, n2 and
    conversing talker v1 talk alone in the pause; n3, the cone passenger and v2 overlap the partners."""
    rs, rn = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    partners = [str(s) for s in rs.choice(corpus.speakers_for("partner"), 2, replace=False)]
    babble = [str(s) for s in rs.choice(corpus.speakers_for("babble"), 8, replace=False)]
    ev = {0: [], 1: []}
    for t0, t1 in ((0.3, 5.4), (15.2, 19.7)):
        t, i = int(t0 * FS), 0
        while True:
            u = _utts(corpus, partners[i % 2], rs, 1)[0]
            if t + int(u.duration * FS) > t1 * FS:
                break
            ev[i % 2].append([u.id, t, 0.0])
            t += int(u.duration * FS) + int(rs.uniform(0.2, 0.4) * FS)
            i += 1

    def one(k, at):
        return [[_utts(corpus, babble[k], rs, 1)[0].id, int(at * FS), 0.0]]

    def far(k, t0):
        times = np.arange(t0, 19.0, 2.4)
        return [[u.id, int(t * FS), 0.0] for u, t in zip(_utts(corpus, babble[k], rs, len(times)), times)]

    road = [[40, 0], [100, -6], [400, -10], [800, -10], [2000, -22], [8000, -40]]
    groups = [
        {"name": "partner_A", "role": "partner", "sources": [_speech("pA", "pA", 68.0, ev[0])]},
        {"name": "partner_B", "role": "partner", "sources": [_speech("pB", "pB", 68.0, ev[1])]},
        {"name": "near_1", "role": "passenger_near", "sources": [_speech("n1", "n1", 66.0, one(0, 5.6))]},
        {"name": "near_2", "role": "passenger_near", "sources": [_speech("n2", "n2", 66.0, one(1, 8.8))]},
        {"name": "near_3", "role": "passenger_near", "sources": [_speech("n3", "n3", 66.0, one(2, 2.0))]},
        {"name": "cone", "role": "passenger_cone", "sources": [_speech("cone", "c1", 66.0, one(3, 3.5))]},
        {"name": "conv", "role": "conv_pair", "sources": [_speech("v1", "v1", 66.0, one(4, 12.0)),
                                                         _speech("v2", "v2", 66.0, one(5, 16.0))]},
        {"name": "babble", "role": "babble_far", "sources": [_speech("f1", None, 64.0, far(6, 0.2)),
                                                            _speech("f2", None, 64.0, far(7, 1.3))]},
        {"name": "engine_road", "role": "engine_road", "sources": [
            {"type": "point", "src": "engine", "level_db": 80.0,
             "signal": {"kind": "harmonic", "f0": 40.0, "f0_dev": 3.0, "n_harm": 20, "noise_db": -3.0,
                        "noise_shape": [[50, 0], [300, -3], [600, -30]]}},
            {"type": "diffuse", "model": "sinc", "level_db": 66.0, "signal": {"kind": "noise", "shape": road}}]}]
    return {"n_samples": 20 * FS, "seed": int(rn.integers(2 ** 31)), "impair": s2.draw_impairments(rn, 31.25),
            "groups": groups, "rules": [{"type": "snr", "target": ["partner_A", "partner_B"],
                                         "noise": ["babble", "engine_road"], "snr_db": float(snr_db)}]}


MEETING_TALKERS = [(1.0, 10), (1.6, 100), (2.5, 160), (3.2, 220), (4.4, 200), (4.8, 32)]  # (m, deg)


def meeting_recipe(geom, corpus, seed=0, hvac_dba=40.0):
    """Test meeting generator: turns by talkers 0 1 2 3 4 5 0 2 (two returning), talker 4 soft (53 dB)."""
    rs, rn = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    spk = [str(s) for s in rs.choice(corpus.speakers_for("talker"), len(MEETING_TALKERS), replace=False)]
    ev, t = {k: [] for k in range(len(spk))}, int(0.5 * FS)
    for k in (0, 1, 2, 3, 4, 5, 0, 2):
        for u in _utts(corpus, spk[k], rs, 2):
            ev[k].append([u.id, t, 0.0])
            t += int(u.duration * FS) + int(0.15 * FS)
        t += int(rs.uniform(0.4, 0.7) * FS)
    groups = [{"name": f"talker_{k:02d}", "role": "talker",
               "sources": [_speech(f"t{k}", f"t{k:02d}", 53.0 if k == 4 else 60.0, ev[k])]} for k in range(len(spk))]
    groups += [{"name": "hvac", "role": "hvac", "sources": [
                    {"type": "diffuse", "model": "sinc", "level_db": hvac_dba, "weighting": "A",
                     "signal": {"kind": "noise", "shape": [[50, 0], [500, -6], [4000, -20], [8000, -30]]}}]},
               {"name": "fan", "role": "fan", "sources": [
                   {"type": "point", "src": "fan", "level_db": 50.0,
                    "signal": {"kind": "harmonic", "f0": 120.0, "n_harm": 6, "noise_db": 0.0}}]}]
    return {"n_samples": t + FS, "seed": int(rn.integers(2 ** 31)), "impair": s2.draw_impairments(rn, 0.0),
            "groups": groups}


def _root():
    """Temporary scenes root with the test bus and meeting scenes (built once per process)."""
    if _ROOT:
        return _ROOT[0]
    root = tempfile.mkdtemp(prefix="metrics2_test_")
    atexit.register(shutil.rmtree, root, True)
    c = [5.0, 1.25, 1.0]
    bus_src = {"pA": _pos(c, 0.45, 10, 1.15), "pB": _pos(c, 0.6, 185, 1.15), "n1": _pos(c, 0.9, 95, 1.15),
               "n2": _pos(c, 1.1, 230, 1.15), "n3": _pos(c, 1.4, 130, 1.15), "cone": _pos(c, 1.8, 8, 1.15),
               "v1": _pos(c, 1.0, 60, 1.15), "v2": _pos(c, 1.2, 300, 1.15), "f1": [1.5, 1.0, 1.2],
               "f2": [9.5, 1.6, 1.2], "engine": [11.1, 1.25, 0.5]}
    srcs = [{"name": k, "pos": p, "directivity": s2.facing(p, bus_src["pB" if k == "pA" else "pA"])
             if k in ("pA", "pB") else None} for k, p in bus_src.items()]
    geom = s2.make_geometry("t", "tbus", "bus", [11.5, 2.5, 2.3], 0.35, s2.place_array(c, 0.04, 0.0), srcs, 21,
                            generator="tests.test_metrics2:bus_recipe", meta={"stratum": "aligned"}, root=root)
    s2.add_scene("t", "bus_snr+0", geom, {"seed": 5, "snr_db": 0.0}, root=root)
    cm = [4.5, 3.0, 0.78]
    srcs = [{"name": f"t{k}", "pos": _pos(cm, r, az, 1.2)} for k, (r, az) in enumerate(MEETING_TALKERS)]
    srcs.append({"name": "fan", "pos": [8.5, 5.5, 2.7]})
    geom = s2.make_geometry("t", "tmeet", "meeting", [9.0, 6.0, 3.0], 0.6, s2.place_array(cm, 0.04, 15.0), srcs, 22,
                            generator="tests.test_metrics2:meeting_recipe", root=root)
    s2.add_scene("t", "meet_hvac40", geom, {"seed": 6, "hvac_dba": 40.0}, root=root)
    _ROOT.append(root)
    return root


def _scene(name):
    return s2.load_scene(f"t/{name}", root=_root())


def _method(fn, stages=False):
    """A method object for bench2.run_method from fn(x) -> y or {'out', 'pre_agc'}."""
    def enhance(x, fs, info, return_stages=False):
        return fn(x)
    mod = types.SimpleNamespace(enhance=enhance if stages else (lambda x, fs, info: fn(x)), __name__=fn.__name__)
    return mod


def _outs(sc, fn, extra):
    runs, comps = M.plan(sc, extra)
    return runs, comps, {run: bench2._stages(fn(s2.signed_mix(sc, flip=flip)), sc["mix"].shape[1])[0]
                         for run, flip in runs}


def _lin(x):
    """Fixed FIR on both mics: a linear, time-invariant method."""
    return lfilter([0.6, 0.25, -0.1], [1.0], x[0]) + lfilter([-0.3, 0.15], [1.0], x[1])


def _err_db(est, ref):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((est - ref) ** 2))


# ----------------------------------------------------------------------------- B.5 harness
def test_sum_comp2_equals_mix():
    for name in ("bus_snr+0", "meet_hvac40"):
        sc = _scene(name)
        total = sum(np.asarray(v, np.float64) for v in sc["comp2"].values())
        assert np.max(np.abs(total - sc["mix"])) <= 1e-6


def test_hagerman_linear_recovers_components():
    for name in ("bus_snr+0", "meet_hvac40"):
        sc = _scene(name)
        runs, comps, outs = _outs(sc, _lin, extra=True)
        assert runs[0] == ("y1", ())
        for c, (terms, groups) in comps.items():
            if groups is None:
                continue
            ref = _lin(sum(np.asarray(sc["comp2"][g], np.float64) for g in groups))
            err = _err_db(M.decode(outs, terms, "out"), ref)
            assert err >= 60, f"{name} {c}: decoded within {err:.1f} dB"
        res = M.score(sc, outs, comps)
        assert res["m"]["consistency_db"] >= 60
        if sc["meta"]["scenario"] == "bus":
            assert res["m"]["closure_db"] >= 60
            assert {"near", "rest", "cone", "conv", "far", "engine", "S_nf", "S_tf"} <= set(comps)


def test_consistency_figures():
    sc = _scene("bus_snr+0")
    s = 1 / np.abs(sc["mix"][0]).max()

    def even(x):  # not odd-symmetric: the target-flip and noise-flip estimates of A + B differ
        return x[0] + 0.5 * s * x[0] ** 2

    def odd_mask(x):  # odd-symmetric: both estimates agree exactly, whatever the nonlinearity
        return x[0] * np.minimum(1.0, np.abs(x[0]) * 30 * s)

    for fn, ok in ((even, lambda c: c < 40), (odd_mask, lambda c: c == M.RATIO_CAP_DB)):
        _, comps, outs = _outs(sc, fn, True)
        c = M.score(sc, outs, comps)["m"]["consistency_db"]
        assert ok(c), (fn.__name__, c)
    # the 4-run closure (y1 + y2 + y3 + y4) / 4 holds exactly the A*B*N interaction: 6 ABN for F(x) = x^3
    runs, comps = M.plan(sc)
    A, B, N = np.random.default_rng(2).standard_normal((3, 1000))
    outs = {}
    for run, flip in runs[:4]:  # 'sensor' stands for the whole N group, which is always flipped together
        x = sum((-1.0 if g in flip else 1.0) * v for g, v in (("partner_A", A), ("partner_B", B), ("sensor", N)))
        outs[run] = {"out": x ** 3}
    assert np.allclose(M.decode(outs, comps["closure"][0], "out"), 6 * A * B * N)
    assert np.allclose(M.decode(outs, comps["A"][0], "out"), A ** 3 + 3 * A * (B ** 2 + N ** 2))


# ----------------------------------------------------------------------------- B.9 scoring invariance
def test_partner_polarity_invariance():
    sc = _scene("bus_snr+0")
    ctx = M.SceneCtx(sc)
    for parts in ("refs", "comp2"):
        a, b = ((np.asarray(sc["refs"][t], np.float64) for t in ("pA", "pB")) if parts == "refs" else
                (np.asarray(sc["comp2"][g][0], np.float64) for g in ("partner_A", "partner_B")))
        same, inv = M.partner_fidelity(ctx, a + b), M.partner_fidelity(ctx, a - b)
        for k in ("A", "B"):
            assert abs(same[f"sdr_{k}"] - inv[f"sdr_{k}"]) < 0.1, (parts, k, same[f"sdr_{k}"], inv[f"sdr_{k}"])
            assert abs(same[f"estoi_{k}"] - inv[f"estoi_{k}"]) < 0.005
        if parts == "refs":
            assert min(same["sdr_A"], same["sdr_B"]) > 20 and min(same["estoi_A"], same["estoi_B"]) > 0.95


def test_oracle_vad_gate_fails_g0():
    sc = _scene("bus_snr+0")
    gate = np.zeros(sc["mix"].shape[1])
    for a, b, t in sc["segs"]:
        if t in ("pA", "pB"):
            gate[a:b] = 1.0

    def oracle_gate(x):
        return x[0] * gate

    res, _ = bench2.run_method(_method(oracle_gate), sc)
    m = res["m"]
    assert res["status"] == "ok"
    assert abs(m["dsnr_active"]) <= 0.5 and m["nm"] >= 20, (m["dsnr_active"], m["nm"])
    rows = [dict(res, card={"geom_id": f"g{i}", "stratum": "aligned"}) for i in range(stats.MIN_GEOMS)]
    verdict, fails = stats.gate_summary(stats.evaluate_gates(rows, "bus"), "G0")
    assert verdict == "FAIL" and fails == ["NM <= 6 dB"], fails


def test_passenger_estoi_falls_when_suppressed():
    sc = _scene("bus_snr+0")
    ctx = M.SceneCtx(sc)
    x = np.asarray(sc["mix"][0], np.float64)
    for t, g in (("n1", "near_1"), ("n2", "near_2")):
        iv = ctx.without(t, ["pA", "pB"])
        assert M._total(iv) >= FS
        e_in = M.estoi(x, sc["refs"][t], iv)
        e_sup = M.estoi(x - 0.9 * np.asarray(sc["comp2"][g][0], np.float64), sc["refs"][t], iv)
        e_all = M.estoi(0.1 * x, sc["refs"][t], iv)
        assert e_sup < e_in - 0.1, (t, e_in, e_sup)
        assert abs(e_all - e_in) < 1e-6  # attenuating everything leaves a passenger as intelligible


# ----------------------------------------------------------------------------- metric primitives
def test_sdr_fir_properties():
    sc = _scene("bus_snr+0")
    ctx = M.SceneCtx(sc)
    ref = np.asarray(sc["refs"]["pA"], np.float64)
    iv = ctx.iv["pA"]
    eq = -np.convolve(np.r_[0, 0, 0, 0.8, -0.3, 0.1], ref)[2:2 + len(ref)]  # inverted, 1 sample late, tilted
    assert M.sdr_fir(eq, ref, iv) > 35
    assert M.sdr_fir(eq, ref, iv) == M.sdr_fir(-eq, ref, iv)
    noise = np.random.default_rng(1).standard_normal(len(ref))
    sig = M._msq(M.inband(ref), ctx.mask(iv))
    noise *= np.sqrt(sig / M._msq(M.inband(noise), ctx.mask(iv)) / 10)
    assert abs(M.sdr_fir(ref + noise, ref, iv) - 10.0) < 0.5


def test_delay_safety_net():
    sc = _scene("bus_snr+0")
    for lag in (1, 40):
        def late(x):
            return np.r_[np.zeros(lag), x[0][:-lag]]
        res, _ = bench2.run_method(_method(late), sc)
        m = res["m"]
        assert res["delay"] == lag and res["aligned"] == (lag > 2) and m["delay_ok"] == float(lag <= 2)
        assert abs(m["ta_A"]) < 0.2 and abs(m["d_estoi_min"]) < 0.01, (lag, m["ta_A"], m["d_estoi_min"])


def test_m0_impulse_and_highpass():  # B.6: an LTI method's impulse lands at lag 0 +- 2
    for scenario, fc in (("bus", 120.0), ("meeting", 80.0)):
        x = np.zeros((2, FS))
        x[0, 100] = 1.0
        r = m0_mica_hpf.enhance(x, FS, {"scenario": scenario, "spacing_m": 0.04}, return_stages=True)
        y = r["out"]
        assert y.shape == (FS,) and abs(int(np.argmax(np.abs(y))) - 100) <= 2 and r["pre_agc"] is y
        H = np.abs(np.fft.rfft(y[100:], FS))  # 1 Hz bins
        for f in (fc / 2, fc, 2 * fc, 1000.0):
            want = (f / fc) ** 2 / np.sqrt(1 + (f / fc) ** 4)  # 2nd-order Butterworth high-pass
            assert abs(20 * np.log10(H[int(f)] / want)) < 0.1, (scenario, f)


# ----------------------------------------------------------------------------- bench runner and metrics
def test_bench_bus_m0_end_to_end():
    sc = _scene("bus_snr+0")
    res, y1 = bench2.run_method("m0_mica_hpf", sc, card=s2.scene_card("t/bus_snr+0", root=_root()), extra=True)
    assert res["status"] == "ok", res.get("error")
    m = res["m"]
    json.dumps(bench2._clean(res), allow_nan=False)
    assert y1.shape == (sc["mix"].shape[1],) and res["runs"][0]["run"] == "y1" and len(res["runs"]) == 10
    assert abs(m["d_estoi_min"]) < 0.01 and abs(m["ta_min"]) < 1.0 and abs(m["nm"]) < 1.0 and abs(m["dsnr_active"]) < 1.0
    for k in ("rda_p10", "dsir_near", "dsir_rest", "dsir_conv", "dsir_far", "dsir_engine", "dsir_cone",
              "pass_estoi_med", "conv_estoi_med", "balance_p90", "closure_db", "consistency_db"):
        assert np.isfinite(m[k]), k
    assert abs(m["dsir_near"]) < 1.0 and abs(m["rda_mean"]) < 1.5  # Mic A treats everyone alike
    assert m["focus_c3"] == 0.0 and m["focus_pass"] == 0.0  # no distractor attenuation
    assert res["talkers"]["n3"]["estoi"] is None or not np.isfinite(res["talkers"]["n3"]["estoi"])  # never alone
    for k in ("dsisdr_sum", "br", "pesq"):
        assert np.isfinite(res["legacy"][k])


def test_bench_meeting_end_to_end_and_leveler():
    sc = _scene("meet_hvac40")
    res, _ = bench2.run_method("m0_mica_hpf", sc, extra=True)
    assert res["status"] == "ok", res.get("error")
    m0 = res["m"]
    assert m0["n_talkers"] == 6 and abs(m0["ta_wrst"]) < 1.0 and abs(m0["d_estoi_wrst"]) < 0.01
    assert m0["lvl_std"] > 2.0 and np.isfinite(m0["t2_margin"]) and np.isfinite(m0["consistency_db"])
    # an oracle leveler (gain per turn from the clean talker levels, pre-AGC stage = Mic A) must even out
    # the post-AGC levels without touching T2 or ESTOI
    ctx = M.SceneCtx(sc)
    gain = np.ones(sc["mix"].shape[1])
    for t, a, b, _ in M._turns(ctx):
        s_in = ctx.mic_a((sc["talker_group"][t],))
        gain[a:b] = 10 ** ((-50 - 10 * np.log10(M._msq(s_in, ctx.mask(ctx.alone(t))))) / 20)

    def leveler(x):
        return {"out": x[0] * gain, "pre_agc": x[0]}

    res, _ = bench2.run_method(_method(leveler, stages=True), sc)
    m = res["m"]
    assert res["stages"] and m["lvl_std"] < 0.5 and m["capture_rate"] == 1.0 and m["settle_med"] == 0.0
    assert abs(m["t2_margin"]) < 0.5 and abs(m["d_estoi_wrst"]) < 1e-6 and m["even_pass"] == 1.0
    assert m["pump_std"] > 1.0  # the per-turn gains pump the noise floor


def test_cli_run_table_compare():
    out = tempfile.mkdtemp(prefix="metrics2_results_")
    try:
        args = ["--set", "t", "--scenes-root", _root(), "--results-root", out]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert bench2.main(["run", "--methods", "m0_mica_hpf,mic_a", "-j", "1", "--extra", "none"] + args) == 0
            assert bench2.main(["table", "--by", "stratum,rt60"] + args) == 0
            assert bench2.main(["compare", "--ref", "mic_a"] + args) == 0
        text = buf.getvalue()
        rows = bench2.load_rows("t", "tts", root=out)
        assert len(rows) == 4 and all(r["status"] == "ok" for r in rows), [r.get("error") for r in rows]
        for h in ("### bus", "### meeting", "### bus by stratum", "### meeting by rt60", "| m0_mica_hpf | G0 |",
                  "paired differences vs mic_a", "| d_estoi_min | m0_mica_hpf | 1 |"):
            assert h in text, h
    finally:
        shutil.rmtree(out, True)


def test_bench_records_errors():
    sc = _scene("bus_snr+0")

    def broken(x):
        raise RuntimeError("boom")

    def short(x):
        return x[0][:-5]

    for fn, msg in ((broken, "boom"), (short, "samples")):
        res, y1 = bench2.run_method(_method(fn), sc)
        assert res["status"] == "error" and msg in res["error"] and y1 is None
    assert bench2.result_path("dev/bus_g003_snr+0", "m0", "tts", "R").endswith("R/dev/m0/bus_g003_snr+0.tts.json")


def test_oracle_side_channel_only_for_oracles():
    sc = _scene("bus_snr+0")
    seen = {}

    def spy(name):
        def enhance(x, fs, info):
            seen.setdefault(name, set()).update(info)
            return x[0]
        return types.SimpleNamespace(enhance=enhance, __name__=name)

    for name in ("m_spy", "oracle_spy"):
        assert bench2.run_method(spy(name), sc)[0]["status"] == "ok"
    assert seen["m_spy"] == {"scenario", "spacing_m"} and seen["oracle_spy"] == {"scenario", "spacing_m", "oracle"}
    res, _ = bench2.run_method("oracle2_irm", sc, extra=True)  # a fixed mask: decodes exactly, deterministic
    m = res["m"]
    assert res["status"] == "ok" and res["oracle"] and m["d_estoi_min"] > 0.1 and m["dsnr_active"] > 5
    assert min(m["consistency_db"], m["closure_db"], m["rerun_db"]) >= 100, m
    try:
        from methods import oracle2_irm
        oracle2_irm.enhance(np.array(sc["mix"]), FS, {"scenario": "bus", "spacing_m": 0.04})
        raise AssertionError("oracle2_irm ran without the oracle side channel")
    except RuntimeError:
        pass


def test_method_policy_and_rerun():
    import importlib.util
    d = tempfile.mkdtemp(prefix="policy_test_")
    try:
        mods = {}
        for name, src in (("clean", "import json\nimport numpy as np\nfrom lab import dsp\nfrom lab.common import FS\n"),
                          ("leaky", '"""Docstrings may say scenes2/."""\nfrom lab import scene2\nimport lab.metrics2\n'
                                    'P = "scenes2/dev"\n')):
            path = f"{d}/{name}.py"
            with open(path, "w") as f:
                f.write(src)
            spec = importlib.util.spec_from_file_location(f"policy_{name}", path)
            mods[name] = importlib.util.module_from_spec(spec)
        bad = bench2._file_violations(mods["leaky"].__file__)
        assert not bench2._file_violations(mods["clean"].__file__)
        assert len(bad) == 3 and any("lab.scene2" in v for v in bad) and any("lab.metrics2" in v for v in bad), bad
        assert any("'scenes2'" in v and ":4 " in v for v in bad), bad
        bench2.load_method("m0_mica_hpf")  # the baselines pass
    finally:
        shutil.rmtree(d, True)
    calls = []

    def drifting(x):  # keeps state between calls: the repeated y1 run differs
        calls.append(1)
        return x[0] * (1 + 0.01 * len(calls))
    sc = _scene("bus_snr+0")
    assert bench2.run_method(_method(drifting), sc, extra=True)[0]["m"]["rerun_db"] < 60
    assert bench2.run_method(_method(lambda x: x[0]), sc, extra=True)[0]["m"]["rerun_db"] == M.RATIO_CAP_DB


# ----------------------------------------------------------------------------- statistics and gates
def test_stats_bootstrap_holm_wilcoxon():
    assert stats.bootstrap_ci([2.0] * 7) == (2.0, 2.0, 2.0)
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 0.1, 20), rng.normal(5, 0.1, 20)]
    strata = ["a"] * 20 + ["b"] * 20
    m, lo, hi = stats.bootstrap_ci(x, strata)
    m2, lo2, hi2 = stats.bootstrap_ci(x)
    assert lo < m < hi and (hi - lo) < 0.5 * (hi2 - lo2)  # stratified resampling keeps the 20/20 split
    _, lo, hi = stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], strata=list("abcde"))
    assert hi - lo > 1.0  # single-member strata: falls back to unstratified resampling
    adj = stats.holm([0.01, 0.04, 0.03, 0.005, np.nan])
    assert np.allclose(adj[:4], [0.03, 0.06, 0.06, 0.02]) and np.isnan(adj[4])
    assert stats.wilcoxon_p(np.zeros(10)) == 1.0 and stats.wilcoxon_p(rng.normal(1, 0.2, 30)) < 1e-4


def _rows(values, stratum="aligned", snrs=(0.0, 5.0), **extra):
    """Fake result rows: one geometry per value, the same value at every SNR."""
    return [{"status": "ok", "scenario": "bus", "method": "x",
             "card": {"geom_id": f"g{i}", "stratum": stratum, "snr_db": s},
             "m": {"rtf": 0.1, "clipped": 0.0, **{k: v for k, v in extra.items()}, "d_estoi_min": float(v)}}
            for i, v in enumerate(values) for s in snrs]


def test_gate_rules():
    rng = np.random.default_rng(3)
    base = {"ta_min": -0.5, "balance_p90": 1.0, "nm": 2.0, "dsnr_active": 4.0}
    rows = _rows(0.06 + rng.normal(0, 0.01, 30), **base)
    res = stats.evaluate_gates(rows, "bus")
    assert stats.gate_summary(res, "G0") == ("PASS", []) and stats.gate_summary(res, "G1") == ("PASS", [])
    assert all(r["n"] == 30 for r in res if r["kind"] not in ("never", "rtf"))
    # improvement gate: point estimate above the threshold but the CI includes zero -> fail
    z = rng.normal(0, 0.3, 30)
    rows = _rows(0.04 + z - z.mean(), **base)
    g = {r["desc"]: r for r in stats.evaluate_gates(rows, "bus")}["worst-partner dESTOI >= +0.03"]
    assert g["point"] >= 0.03 and g["lo"] <= 0 and g["pass"] is False
    # no-regression gate on the CI lower bound; stress SNR rows count for G0 only
    rows = _rows(np.full(30, -0.005), **base) + _rows(np.full(30, -0.5), snrs=(-10.0,), **base)
    res = {(r["gate"], r["desc"]): r for r in stats.evaluate_gates(rows, "bus")}
    assert res[("G0", "worst-partner ESTOI >= input - 0.01")]["pass"] is False
    assert abs(res[("G1", "worst-partner dESTOI >= +0.03")]["point"] + 0.005) < 1e-12
    # an error row fails 'no clipping, NaN or errors'; CI criteria need MIN_GEOMS geometries for a verdict
    rows = _rows(np.full(3, 0.1), **base) + [{"status": "error", "scenario": "bus", "card": {"geom_id": "g9"}}]
    res = {r["desc"]: r for r in stats.evaluate_gates(rows, "bus")}
    assert res["no clipping, NaN or errors"]["pass"] is False and res["no clipping, NaN or errors"]["point"] == 1.0
    assert res["NM <= 6 dB"]["pass"] is None and res["NM <= 6 dB"]["n"] == 3


def test_paired_compare():
    rng = np.random.default_rng(4)
    ref = _rows(rng.normal(0, 0.05, 30))
    better = [dict(r, method="y", m=dict(r["m"], d_estoi_min=r["m"]["d_estoi_min"] + 0.04)) for r in ref]
    q = stats.paired(better, ref, "d_estoi_min")
    assert q["n"] == 30 and abs(q["delta"] - 0.04) < 1e-9 and q["lo"] > 0 and q["p"] < 1e-5


if __name__ == "__main__":
    fails = 0
    t_all = time.time()
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.2f} s)")
        except Exception as e:  # report every failure, then exit non-zero
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(("all passed" if not fails else f"{fails} failed") + f" in {time.time() - t_all:.1f} s")
    raise SystemExit(1 if fails else 0)
