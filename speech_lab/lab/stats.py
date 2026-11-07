"""Statistics for the v2 bench (DESIGN 4.5-4.6): per-geometry means, paired bootstrap confidence intervals
(stratified by the E1 stratum on the bus), Wilcoxon signed-rank tests with Holm correction, and the G0/G1
gates with the CI rules. Definitions: BENCH2.md "Metrics".

Rows are per-scene result dicts as written by lab.bench2 (keys 'status', 'scenario', 'card', 'm', 'legacy').
"""
from typing import NamedTuple

import numpy as np
from scipy.stats import wilcoxon

N_BOOT = 10_000
ALPHA = 0.05
STRESS_SNR = -10.0  # bus SNR at or below this is the stress case: G0 includes it, G1 does not
MIN_GEOMS = 5       # CI-based gate criteria need at least this many geometries for a verdict
_NAN = float("nan")


def metric(row, key):
    """Value of `key` in a result row (searched in 'm', 'legacy', then the row itself); NaN if missing."""
    for src in (row.get("m") or {}, row.get("legacy") or {}, row):
        if key in src:
            v = src[key]
            return _NAN if v is None else float(v)
    return _NAN


def card(row):
    return row.get("card") or {}


def geometry(row):
    """Unit of analysis (DESIGN 4.5): the scene's geometry id (scenes of an SNR or HVAC sweep share it)."""
    return card(row).get("geom_id") or row.get("scene")


def is_stress(row):
    snr = card(row).get("snr_db")
    return row.get("scenario") == "bus" and snr is not None and snr <= STRESS_SNR


def per_geometry(rows, key):
    """{geometry: (mean of `key` over its scenes, E1 stratum)}; NaN scenes are skipped and geometries
    without a finite value are left out."""
    vals, strata = {}, {}
    for r in rows:
        v = metric(r, key)
        if np.isfinite(v):
            g = geometry(r)
            vals.setdefault(g, []).append(v)
            strata[g] = card(r).get("stratum")
    return {g: (float(np.mean(v)), strata[g]) for g, v in vals.items()}


def bootstrap_ci(x, strata=None, n_boot=N_BOOT, alpha=ALPHA, seed=0):
    """(mean, lo, hi): percentile bootstrap CI of the mean of x. With strata, resampling is done within each
    stratum, keeping its count (DESIGN 4.5: stratified by E1 stratum); a stratum with a single member would
    have no variance, so then the resampling is not stratified."""
    x = np.asarray(x, float)
    if not len(x):
        return _NAN, _NAN, _NAN
    labels = np.zeros(len(x), int)
    if strata is not None:
        labels = np.unique([str(s) for s in strata], return_inverse=True)[1]
        if np.bincount(labels).min() < 2:
            labels = np.zeros(len(x), int)
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_boot)
    for s in np.unique(labels):
        xs = x[labels == s]
        tot += xs[rng.integers(0, len(xs), (n_boot, len(xs)))].sum(1)
    boot = tot / len(x)
    return float(x.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def wilcoxon_p(d):
    """Two-sided Wilcoxon signed-rank p-value of paired differences; 1.0 when all are zero, NaN if empty."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return _NAN
    if not np.any(d != 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def holm(p):
    """Holm-Bonferroni adjusted p-values; NaN entries stay NaN and do not count as tests."""
    p = np.asarray(p, float)
    out = np.full(len(p), _NAN)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    out[order] = adj
    return out


def paired(rows, ref_rows, key, stratified=True):
    """Paired comparison of `key` between two methods over the geometries both have finite values for.

    Returns {n, delta, lo, hi, p, mean, ref_mean}: the mean per-geometry difference (rows minus ref_rows),
    its bootstrap 95% CI (stratified by E1 stratum when available) and the Wilcoxon p-value."""
    a, b = per_geometry(rows, key), per_geometry(ref_rows, key)
    common = sorted(set(a) & set(b), key=str)
    if not common:
        return {"n": 0, "delta": _NAN, "lo": _NAN, "hi": _NAN, "p": _NAN, "mean": _NAN, "ref_mean": _NAN}
    d = np.array([a[g][0] - b[g][0] for g in common])
    strata = [a[g][1] for g in common]
    use = stratified and any(s is not None for s in strata)
    delta, lo, hi = bootstrap_ci(d, strata if use else None)
    return {"n": len(common), "delta": delta, "lo": lo, "hi": hi, "p": wilcoxon_p(d),
            "mean": float(np.mean([a[g][0] for g in common])), "ref_mean": float(np.mean([b[g][0] for g in common]))}


# ----------------------------------------------------------------------------- gates (DESIGN 4.6)
class Gate(NamedTuple):
    """One gate criterion. kind:
    improve  'delta >= x': point estimate >= x and CI lower bound > 0 (DESIGN 4.5);
    noreg    'delta >= -x': CI lower bound >= -x (DESIGN 4.5);
    cap      'value <= x': CI upper bound <= x (no-regression rule applied to an upper limit);
    floor    'value >= x': CI lower bound >= x;
    never    no scene may have the condition (count of scenes; errors always count);
    rtf      90th percentile of the per-scene RTF < x (reported only for offline-only methods);
    pending  needs a model that is not available yet."""
    gate: str
    desc: str
    key: str
    kind: str
    x: float = _NAN
    where: tuple = ()


GATES = {
    "bus": [
        Gate("G0", "worst-partner ESTOI >= input - 0.01", "d_estoi_min", "noreg", 0.01),
        Gate("G0", "both TA_k >= -3 dB", "ta_min", "noreg", 3.0),
        Gate("G0", "balance P90 <= 3 dB", "balance_p90", "cap", 3.0),
        Gate("G0", "NM <= 6 dB", "nm", "cap", 6.0),
        Gate("G0", "no clipping, NaN or errors", "clipped", "never"),
        Gate("G0", "RTF < 0.5", "rtf", "rtf", 0.5),
        Gate("G1", "worst-partner dESTOI >= +0.03", "d_estoi_min", "improve", 0.03),
        Gate("G1", "worst-partner dESTOI >= +0.05 (aligned)", "d_estoi_min", "improve", 0.05,
             (("stratum", "aligned"),)),
        Gate("G1", "dSNR_active >= +3 dB (aligned)", "dsnr_active", "improve", 3.0, (("stratum", "aligned"),)),
        Gate("G1", "dSNR_active >= +1.5 dB (all strata)", "dsnr_active", "improve", 1.5),
    ],
    "meeting": [
        Gate("G0", "worst-talker ESTOI >= input - 0.01", "d_estoi_wrst", "noreg", 0.01),
        Gate("G0", "dESTOI >= -0.03 for >= 19/20 talkers", "frac_estoi_ok", "floor", 0.95),
        Gate("G0", "T2: far-bin median TA >= near-bin median - 3 dB", "t2_margin", "noreg", 3.0),
        Gate("G0", "NM <= 6 dB", "nm", "cap", 6.0),
        Gate("G0", "no NaN or errors", "error", "never"),
        Gate("G0", "RTF < 0.5", "rtf", "rtf", 0.5),
        Gate("G1", "T3 level std <= 1.5 dB", "lvl_std", "cap", 1.5),
        Gate("G1", "T3 level range <= 6 dB", "lvl_range", "cap", 6.0),
        Gate("G1", "mean dESTOI >= +0.02", "d_estoi_mean", "improve", 0.02),
        Gate("G1", "dESTOI >= +0.03 in the > 4 m bin", "d_estoi_far", "improve", 0.03),
        Gate("G1", "WER not worse in any distance bin", "wer", "pending"),
        Gate("G1", "NA_active >= 3 dB", "na_active", "improve", 3.0),
    ],
}


def _check(kind, x, point, lo, hi):
    if not np.isfinite(point):
        return None
    if kind == "improve":
        return bool(point >= x and lo > 0)
    if kind == "noreg":
        return bool(lo >= -x)
    if kind == "cap":
        return bool(hi <= x)
    if kind == "floor":
        return bool(lo >= x)
    raise ValueError(kind)


def evaluate_gates(rows, scenario, offline_only=False):
    """Gate results for one method's rows of one scenario: a list of dicts {gate, desc, key, kind, x, n,
    point, lo, hi, pass}. pass is None when the criterion cannot be evaluated (no data, pending model,
    RTF of an offline-only method). G1 on the bus leaves out the stress SNR."""
    out = []
    for g in GATES[scenario]:
        sel = [r for r in rows if not (g.gate == "G1" and is_stress(r))]
        sel = [r for r in sel if all(card(r).get(k) == v for k, v in g.where)]
        ok = [r for r in sel if r.get("status") == "ok"]
        res = {"gate": g.gate, "desc": g.desc, "key": g.key, "kind": g.kind, "x": g.x, "n": 0,
               "point": _NAN, "lo": _NAN, "hi": _NAN, "pass": None}
        if g.kind == "pending":
            res["note"] = "pending: needs a model that is not downloaded yet"
        elif g.kind == "never":
            bad = [r for r in sel if r.get("status") != "ok" or (g.key == "clipped" and metric(r, "clipped") > 0)]
            res.update(n=len(sel), point=float(len(bad)), **{"pass": not bad if sel else None})
        elif g.kind == "rtf":
            v = [metric(r, "rtf") for r in ok]
            v = [x for x in v if np.isfinite(x)]
            if v:
                p90 = float(np.percentile(v, 90))
                res.update(n=len(v), point=p90, **{"pass": None if offline_only else bool(p90 < g.x)})
                if offline_only:
                    res["note"] = "offline-only method: RTF reported"
        else:
            pg = per_geometry(ok, g.key)
            if pg:
                vals = [v for v, _ in pg.values()]
                strata = [s for _, s in pg.values()]
                use = scenario == "bus" and not g.where and any(s is not None for s in strata)
                point, lo, hi = bootstrap_ci(vals, strata if use else None)
                verdict = _check(g.kind, g.x, point, lo, hi) if len(vals) >= MIN_GEOMS else None
                res.update(n=len(vals), point=point, lo=lo, hi=hi, **{"pass": verdict})
                if len(vals) < MIN_GEOMS:
                    res["note"] = f"n/a: fewer than {MIN_GEOMS} geometries"
        out.append(res)
    return out


def gate_summary(results, gate):
    """Verdict for one gate (G0 or G1) from evaluate_gates() output, plus the failing criteria: 'FAIL' if any
    criterion fails, 'PASS' if all pass, 'PASS* k/n' if the k evaluable criteria of n pass, 'n/a' if none
    is evaluable."""
    items = [r for r in results if r["gate"] == gate]
    fails = [r["desc"] for r in items if r["pass"] is False]
    known = [r for r in items if r["pass"] is not None]
    if fails:
        return "FAIL", fails
    if not known:
        return "n/a", fails
    return ("PASS" if len(known) == len(items) else f"PASS* {len(known)}/{len(items)}"), fails
