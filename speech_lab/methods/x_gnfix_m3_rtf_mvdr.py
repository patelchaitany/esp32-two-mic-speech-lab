"""x_gnfix_m3_rtf_mvdr: m3_rtf_mvdr with a given calib_json completed by a correctly rotated gamma_n (experimental, x_ prefix).

When info['calib_json'] is a dict without 'gamma_n' (e.g. hw/realset/calib_session.json), x_s0v2.gamma_n_for(x, fs, cfg)
measures the noise coherence on this input's noise frames AFTER that json's own skew/corner correction and adds it.
Without it, methods/_core/calib._override re-blends calib.py's measurement rotated by the BLIND skew (0 / 0.5 sample),
which on this board leaves gamma_n at 0.36 instead of 0.84 at 4 kHz (methods/x_s0v2.md section 4). Everything else,
including the json's spacing, skew and gain, is unchanged. Without a calib_json this is plain m3_rtf_mvdr.
"""
import json

from methods import m3_rtf_mvdr as _base
from methods import x_s0v2


def enhance(x, fs, info, return_stages=False, **kw):
    cfg = info.get("calib_json")
    if cfg is not None:
        if not isinstance(cfg, dict):
            with open(cfg) as fh:
                cfg = json.load(fh)
        if "gamma_n" not in cfg:
            g = x_s0v2.gamma_n_for(x, fs, cfg)
            cfg = dict(cfg, gamma_n={"f_hz": [float(v) for v in x_s0v2.F], "gamma": [float(v) for v in g]})
            info = dict(info, calib_json=cfg)
    return _base.enhance(x, fs, info, return_stages=return_stages, **kw)
