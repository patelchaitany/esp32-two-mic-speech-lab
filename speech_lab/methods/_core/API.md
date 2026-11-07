# methods/_core front end: S0 calibration, S1 tracker, shared helpers

Owner: core (S0/S1). Design: `DESIGN.md` 2.0. Tests: `.venv/bin/python -m tests.test_core`
(`--quick` skips the v1-scene checks). Modules: `common.py` (conventions, STFT, coherence, CDR,
steered response, VAD), `calib.py` (S0), `tracker.py` (S1 and `front_end`), `spk.py` (speaker
identity). `__init__.py` stays empty. Everything is numpy/scipy; nothing reads files except an
optional `calib.json` named in `info`.

## Conventions (fixed; everything below relies on them)

- `x[0]` = Mic A, `x[1]` = Mic B, fs = 16000, float64 internally.
- TDOA `tau` in **seconds**, `tau > 0` = Mic B hears the source later. Plane wave: `X_B = X_A exp(-j 2 pi f tau)`,
  `tau = d cos(theta) / c` with `theta` the angle from the axis (0 deg = beyond Mic A, 180 deg = beyond Mic B), c = 343 m/s.
- Center-referenced steering `a(f, tau) = [exp(+j pi f tau), exp(-j pi f tau)]` = `common.steer(f, tau)`.
  RTFs normalized to Mic A must go through `common.symmetrize_rtf(G)` before center-referenced constraints such as h = [1, -1].
- `Phi_AB = E{X_A conj(X_B)}`; plane-wave coherence `Gamma_s = exp(+j 2 pi f tau)` = `common.plane_coherence(f, tau)`.
- Sign trap: `lab.dsp.gcc_phat(x0, x1)` is positive when x0 lags, so our `tau = lab.dsp.gcc_phat(x_B, x_A)` = `common.tdoa_gcc(x_A, x_B)`.
- STFT: `common.stft(x)` / `common.istft(X, T)` = `lab.dsp` (periodic sqrt-Hann 512, hop 128, 257 bins, 31.25 Hz, 8 ms hop).
  Frame l covers samples `[l*128 - 512, l*128)`; its centre is `l*128 - 256` (`common.frame_centers`). `n_frames(T) = 1 + (T + 1024) // 128`.
  The first 4 and last 8 frames overlap the zero pad (`common.valid_frames(N)`).
- Per-frame arrays in `Track` have exactly one entry per STFT frame.
- Smoothing constants `lambda = exp(-8 ms / tau_avg)`: `LAM_20MS 0.67`, `LAM_50MS 0.85`, `LAM_76MS 0.90`,
  `LAM_160MS 0.95`, `LAM_2S 0.996`; `common.lam(tau_avg_s)` gives the exact value.

## Quick start

```python
from methods._core import common as cm, tracker

def enhance(x, fs, info, return_stages=False):
    fe = tracker.front_end(x, fs, info)          # S0 + S1, offline (whole file); online=True for the causal variant
    X, cal, trk = fe.X, fe.calib, fe.track       # X: (2, 257, N) with X_B already corrected
    f = cm.FREQS                                 # (257,) Hz
    a_A = cm.steer(f, trk.tau_A)                 # (257, 2) steering to partner A (bus)
    Gn = cal.gamma_n                             # (257,) real noise coherence for MVDR / CDR
    ...                                          # build Y (257, N)
    return cm.istft(Y, x.shape[1])               # aligned with x[0]
```

Meeting with WPE: `xh = calib.prefilter(x, 'meeting'); X0 = cm.stft(xh); cal = calib.calibrate(X0, info);
X = calib.apply(X0, cal); Xw = <WPE>(X); trk = tracker.track(Xw, cal, info, x=xh[0])`.

Cost (M1 Pro, one process): S0 + S1 RTF 0.006-0.011 offline, 0.012-0.015 online (32 s bus scene about
0.2 s; 84 s meeting about 0.6 s). 10 min file (machine under heavy load): RTF 0.007-0.009 offline,
0.015-0.018 online; peak RSS about 1.0 GB above the input (1.5-1.7 GB in all).

Timing: S0 only changes X_B, so Mic A after `istft` is the prefiltered Mic A. The causal 2nd-order
high-pass leads speech in a broadband cross-correlation (40 TTS utterances of `corpus/`: -5 to -1 samples,
median -2, at 120 Hz (bus); -3 to 0, median -1, at 80 Hz (meeting)). The v2 scorer (`lab/metrics2.find_delay`)
band-passes to 1-4 kHz before its delay search, where this high-pass shifts the phase by under 0.5 samples.

Short inputs: every entry point accepts any length T >= 0 (T = 0 gives empty outputs and 9 frames). Below
T = 512 samples no frame lies inside the signal: S0 returns the prior (`source='prior'`) and the bus
partner model returns no peaks (NaN TDOAs, zero posteriors, `inseparable_why = ('one_peak',)`).

## common.py

| Name | Signature | Notes |
|---|---|---|
| `FS, C, N_FFT, HOP, N_BINS, HOP_S, FREQS, CHUNK, EPS` | constants | `FREQS` (257,) Hz; `CHUNK` = frames per streaming chunk |
| `stft(x)`, `istft(X, length)` | (..., T) -> (..., 257, N) complex128 and back | bitwise equal to lab.dsp; `stft` frames in chunks (a third of the memory) |
| `n_frames(T)`, `frame_centers(N)`, `frame_of_sample(s)`, `valid_frames(N)`, `bins_between(f_lo, f_hi)` | framing helpers | `bins_between` returns a slice |
| `steer(f, tau)` | -> broadcast(f, tau) + (2,) complex | center-referenced |
| `plane_coherence(f, tau)` | `exp(+j 2 pi f tau)` | |
| `diffuse_model(f, d, model)` | `'sinc'`: sinc(2fd/c) (3-D); `'j0'`: J0(2 pi f d/c) (2-D) | |
| `symmetrize_rtf(G)` | G (..., F) -> (..., F, 2) | unwraps the phase along frequency, then g/sqrt(g1 g2) |
| `tdoa_gcc(x_a, x_b, fs, max_tau, interp)` | -> tau (s), our sign | |
| `hpf(x, fc, fs, order=2)` | causal Butterworth, same on every row | |
| `smooth(P, lam, init=None)` | -> (y, last) | recursive average along the last axis; chunkable |
| `spectra(X)` | -> (\|X_A\|^2, \|X_B\|^2, X_A conj X_B) | X (2, F, N) |
| `coherence(paa, pbb, pab)`, `msc(...)`, `msc_weight(m)` | | `msc_weight` = clip(MSC/(1-MSC+0.05), 0, 20) |
| `smoothed_spectra(X, sl, lams)` | generator of (l0, l1, {lam: (paa, pbb, pab)}) | streams long files in chunks |
| `cdr_doa(Gx, Gn, Gs)` | DOA-dependent CDR (Schwarz-Kellermann prop1), linear | clips \|Gx\| <= 0.999, Gn <= 0.95 |
| `cdr_nodoa(Gx, Gn)` | DOA-free CDR (prop3), linear | same clipping; use only where Gamma_n is well below 1 |
| `srp(G, W, f, grid)` | DESIGN steered response (L, n) | used by `srp_mode='design'` |
| `coherent_fit(G, gn, f, grid)` | -> (J, Q), each (L, n) | per-frame fit G = q e^{j2pi f tau} + (1-q) gn; S1 default |
| `srp_peak(P, grid)` | -> (tau, value) per row | argmax + parabolic refinement |
| `speech_presence(e, valid=None, floor_s=3.0, thr_db=6, slope_db=1.5, bias_db=2.0)` | -> (p, floor) | causal energy VAD: floor = min of the 20 ms-smoothed energy over 3 s + 2 dB; p = 0.5 at 6 dB above it, 0.95 at 10.4 dB |

## calib.py (S0)

```python
calib.prefilter(x, scenario) -> (2, T)          # 2nd-order Butterworth HPF, 120 Hz bus / 80 Hz meeting, both rows
calib.calibrate(X, info) -> Calib               # offline, X = stft(prefilter(x)); info: scenario, spacing_m, [calib_json]
calib.calibrate_online(X, info, block_s=5.0) -> [(start_frame, Calib), ...]   # causal: block k uses frames < start_k
calib.apply(X, cal, inplace=False) -> X'        # X_B *= cal.corr (cal = Calib or online schedule)
calib.identity(d=0.04, scenario='bus', model='sinc', rho=1.0) -> Calib        # no correction, model Gamma_n
calib.load_json(path_or_dict) -> dict
calib.corner_phase(f, f_a, f_b)                 # atan(f_A/f) - atan(f_B/f), rad
calib.BANDS_HZ                                  # 18 one-third-octave centres 125 Hz - 6.35 kHz (base-2)
```

`Calib` fields (all frequencies are `common.FREQS` unless stated):

| Field | Type / unit | Meaning |
|---|---|---|
| `d_hat` | float, m | spacing every method should use (never `info['spacing_m']`) |
| `model` | `'sinc'` or `'j0'` | diffuse-noise model that fitted better |
| `rho` | float in [0.5, 1] | coherence scale (self-noise / uncorrelated-noise cap) |
| `gamma_n` | (257,) real | noise coherence: measured over D (bus) or VAD pauses (meeting), 5-bin smoothed, clipped [-0.3, 0.98], blended with `rho*model(f; d_hat)` with weight N/(N+200) on the measurement |
| `gain_db` | (18,) dB at `BANDS_HZ` | gain correction applied to X_B |
| `phase_deg` | (18,) deg at `BANDS_HZ` | HPF-corner phase correction applied to X_B (excludes the skew) |
| `tau0` | float, s | inter-channel skew, + = Mic B late; `apply()` advances X_B by `tau0`. Blind: 0 or 31.25e-6 only (see deviation 5) |
| `corr` | (257,) complex | full correction: `X_B <- corr * X_B`, `corr = g(f) exp(+j(phi_corner(f) + 2 pi f tau0))` |
| `gain` (property) | (257,) | `abs(corr)` |
| `model_coherence(f)` | method | `rho * model(f; d_hat)` |
| `hpf_hz` | float, Hz | prefilter corner used |
| `confidence` | 0..1 | min(1, N_D/300) * exp(-(fit_rms/0.1)^2) * exp(-(phase_rms/15 deg)^2): about 0.9 on ideal diffuse noise, 0.1-0.8 on v1 bus scenes, <= 0.1 on v1 meetings |
| `source` | `'blind'`, `'json'`, `'prior'` | `'prior'`: fewer than 50 diffuse frames, identity correction |
| `diag` | dict | `n_diffuse, n_noise, d_prior, gain_db_raw, f_a, f_b, tau0_fit, band_phase_resid_deg, skew_applied, phase_frames ('diffuse' = D, 'pauses' = VAD pauses), skew_margin_us (\|tau0_fit - 15.625 us\|), phase_rms_deg, phase_fault, gamma_d, gamma_n_meas, d_fit, fit_rms, fit_rms_models, d_lb, n_direct, d_source ('fit', 'lower_bound', 'prior'), blend_w, vad_p` (+ `json_drift_db, blind_d_hat, blind_tau0` with calib.json) |

S0 steps as implemented: (1) prefilter; (2) D = valid frames whose lam-0.9 MSC over 2.5-6 kHz, measured
**without the frame's own data** (max of the causal value 4 frames earlier and the anti-causal value 4
frames later; online, the anti-causal value comes from a backward pass over each 5 s block, so it uses
past data only and the last ~20 frames of a block never enter D), is below its 30th percentile, and whose
level is above its 20th percentile; (3) per-band gain = sqrt(median_D P_AA,b / median_D P_BB,b), 3-band
[1/4 1/2 1/4] smoothing (linear extrapolation at the edges), clip +-3 dB, log-frequency interpolation to
bins; joint phase fit (skew + two corners) over 100-2000 Hz to sum Phi_AB over the scenario's noise frames
(bus: D; meeting: VAD pauses p < 0.3, the frames of Gamma_n, or D if fewer than 50), weighted by its
magnitude; (4) skew = 31.25 us if the fitted slope `tau0_fit` >= 15.625 us, else 0; the corners are then
refitted with the skew fixed, over 100-2000 Hz, or 100-500 Hz when the joint fit leaves a band residual
above 8 deg; (5) spacing; (6) Gamma_n. The hardware-fault flag `diag['phase_fault']` is set when any band's
corner phase exceeds 10 deg (note: 50/80 Hz corners already give 10.8 deg at 125 Hz). The individual
corners `f_a`, `f_b` are not identifiable (only their difference is; on v2 dev bus 58/90 fits sit on the
30/150 Hz search bounds); use `phase_deg`, which is.

### calib.json (hardware mode)

Pass `info['calib_json'] = <path or dict>`; every key is optional except `version`.

```json
{
  "version": 1,
  "created": "2026-11-05T10:00:00",
  "spacing_m": 0.0398,             "comment_spacing": "acoustic d = c (tau_plus - tau_minus) / 2 from endfire chirps",
  "spacing_caliper_m": 0.040,
  "tau0_s": 3.125e-05,             "comment_tau0": "I2S skew (tau_plus + tau_minus) / 2; + = Mic B late",
  "band_hz": [125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300],
  "gain_db": [0.3, 0.3, "... one value per band: correction applied to X_B"],
  "phase_deg": [-4.1, -3.2, "... one value per band: corner phase applied to X_B, skew excluded"],
  "hpf_corner_hz": [55.0, 72.0],   "comment_hpf": "used for the phase only if phase_deg is absent",
  "model": "sinc",
  "rho": 0.93,
  "gamma_n": {"f_hz": [0, 31.25, "..."], "gamma": [0.98, 0.98, "..."]}
}
```

Overrides: `spacing_m -> d_hat`, `tau0_s -> tau0`, `phase_deg` (or `hpf_corner_hz`) -> corner phase,
`model`, `rho`, `gamma_n` (dict or a 257-list). `gain_db` is the per-band gain plus the blind broadband
drift (median over bands of blind minus json, clipped +-1 dB) when the file has >= 50 diffuse frames.
Without `gamma_n`, the blend is recomputed with the json spacing/model/rho. `comment_*` keys are ignored.
`source` becomes `'json'`; the blind values stay in `diag['blind_d_hat']`, `diag['blind_tau0']`.

## tracker.py (S1)

```python
tracker.front_end(x, fs, info, online=False, **track_kw) -> FrontEnd     # prefilter, stft, S0, apply, S1
tracker.track(X, cal, info, online=False, srp_mode='fit', partners=None, talkers=None,
              x=None, embedder=None, cos_thr=0.6) -> Track
tracker.angle_deg(tau, d_hat) -> deg; tracker.orientation_stratum(theta_a, theta_b) -> str
tracker.inseparable_reasons(n_peaks, tau_a, tau_b, d_hat) -> tuple
```

`FrontEnd`: `x` (2, T) prefiltered input (B uncorrected), `X` (2, 257, N) corrected STFT, `calib` (offline:
whole file; online: last block), `track`, `schedule` (online only; pass it to `calib.apply`).
`track()` arguments: `cal` = Calib or online schedule; `partners` / `talkers` default to the scenario
(bus: partner model; meeting: clustering); `x` = prefiltered Mic A (needed only with an embedder).

`Track` fields (N = number of STFT frames):

| Field | Shape / unit | Meaning |
|---|---|---|
| `tau` | (N,) s | per-frame TDOA (S0-corrected), grid +-1.2 d_hat/c, 2 us steps, parabolic refinement |
| `q` | (N,) 0..1 | coherent (direct-path) fraction at `tau` from the fit; in `'design'` mode the SRP peak value |
| `psi` | (N,) 0..1 | DESIGN directness: mean MSC over 2.5-6 kHz, lam 0.9 |
| `directness` (property) | (N,) | `q` in `'fit'` mode, `psi` in `'design'` mode; used for all selections |
| `cdr_db` | (N,) dB | DOA-free CDR averaged over bins 1-6 kHz with Gamma_n <= 0.7 (floor -20 dB) |
| `level_db` | (N,) dB | mean power per bin over 300-4000 Hz, both mics |
| `vad` | (N,) 0..1 | speech presence (`common.speech_presence` on the 300-4000 Hz energy) |
| `noise_db` | (N,) dB | VAD noise floor |
| `d_hat` | m | spacing used (last block online) |
| `online`, `srp_mode` | bool, str | |
| `tau_A`, `tau_B` | s | bus partner TDOAs, **tau_A >= tau_B** (A = partner toward the Mic-A end); offline: whole file; online: final. Peaks of the energy-weighted histogram (weights capped at their 95th percentile): the highest, plus the highest other one >= 25 us away that reaches 15 % of its height. Equal if only one peak; NaN if none |
| `tau_A_t`, `tau_B_t` | (N,) s | per frame (offline: constant; online: causal, NaN before the first estimate) |
| `p_A`, `p_B` | (N,) 0..1 | partner posteriors (see deviations: calibrated so the 90th percentile reads 1). **One peak** (frames where `tau_A_t == tau_B_t`): the single peak's posterior is `p_A` and `p_B = 0`, so `p_A + p_B` (the M11-bus AGC gate) still works; `p_A` then means "the one localized talker", whichever side it is on |
| `double_talk` | (N,) bool | both posteriors > 0.25; never on one-peak frames. A single-tau indicator almost never fires during real overlaps (synthetic 0.5 s overlaps: 0.00); treat "neither p_k > 0.75" as the AB state |
| `theta_A_deg`, `theta_B_deg` | deg | `arccos(c tau_k / d_hat)`; 0 = beyond Mic A, 90 = broadside |
| `stratum` | str | `'aligned'` (both <= 20 deg from the axis, folded), `'45'` (both 30-60), `'broadside'` (both >= 70; a device lying flat looks the same), `'side_by_side'` (same side, 20-40 deg apart), `'other'`, `'unknown'` (fewer than 2 peaks) |
| `inseparable`, `inseparable_why` | bool, tuple | reasons among `'one_peak'`, `'close_tdoas'` (< 25 us; cannot occur with the peak rule above, kept for callers that pass their own TDOAs), `'both_broadside'` (both \|tau\| < 0.3 d_hat/c) -> use the DESIGN 2.1 fallback |
| `cluster` | (N,) int | meeting talker cluster id; -1 outside speech and before the first cluster exists |
| `change` | (N,) bool | talker-change proposals (onset after >= 300 ms pause, or a 15 us / 4 dB shift held 250 ms) |
| `clusters` | list of dict | `id, tau (s), cdr_db, frames, segments` per cluster |
| `diag` | dict | `hist`, `hist_grid` (partner histogram), `peaks` [(tau, height)] |

Online mode (`online=True`): S0 refreshed every 5 s from past frames (first block: prior or calib.json);
a refresh can change `tau0` (0 <-> 31.25 us), which shifts every later TDOA by 31.25 us (v2 dev: 0.17
changes per bus file, 0.15 per meeting, at most 2); features are causal; partner histogram over the
first 5 s, then 5 s forgetting, peaks and thresholds refreshed every 0.25 s from past frames (20 s
window); posteriors scaled by a running 90th percentile.

## spk.py (speaker identity)

```python
spk.cluster_talkers(trk, online=False, x=None, embedder=None, cos_thr=0.6) -> (labels (N,), summary list)
spk.change_points(trk) -> (N,) bool
spk.DPMeans(tau_thr=15e-6, cdr_thr=4.0, max_clusters=12)   # .assign(feat, weight, update, slack) .update(i, feat, w) .nearest(feat)
spk.CosineClusterer(thr=0.6, max_clusters=30)              # .assign(embedding, weight, update)
spk.dp_means_batch(feats, weights) -> (labels, centroids)
spk.OnnxEmbedder(path, threads=1)    # generic hook (Kaldi-style fbank + onnxruntime), unvalidated
spk.fbank(y, fs=16000, n_mels=80)
```

The embedder is any callable `y (1-D, 16 kHz) -> unit-norm vector`. A validated one exists in
`methods/_core/spk_onnx.py` (another owner): `tracker.front_end(x, fs, info, embedder=spk_onnx.WeSpeakerEmbedder())`
or `tracker.track(Xw, cal, info, x=xh[0], embedder=..., cos_thr=...)`.

- **Embedding route** (embedder and `x` given): inside each speech region (VAD > 0.5, gaps < 300 ms bridged),
  one embedding per 1.5 s of confident speech (vad >= 0.95), clustered online by cosine similarity
  (`cos_thr`, at most 30 clusters). Offline a chunk's label covers the region from the previous chunk's end
  on; online it takes over when the chunk is complete. Regions without 1.5 s of confident speech keep the
  last label. TDOA/CDR change proposals are not used here.
- **Fallback** (no embedder): DP-means on per-segment (directness-weighted median TDOA, median CDR dB) with
  thresholds 15 us / 4 dB, at most 12 clusters; segments = speech regions split at `change_points`.
  Offline: batch DP-means. Online: a segment keeps the previous label for 250 ms, then takes the cluster of
  its first 250 ms (CDR threshold doubled, because the CDR of an onset is still rising); the centroid is
  updated with the whole segment at its end. Segment features skip the first 48 ms.
- Labels are -1 outside speech and before the first cluster exists.

Measured on the v1 meetings (20 TTS talkers, each speaking once): DP-means 6-12 clusters offline over the
8 scenes, 12 (the cap) online on s0 and s12. WeSpeaker embeddings on s0 (40 mm): 13-14 clusters,
1.9 talkers per cluster on average, every talker inside one cluster; `cos_thr` 0.6-0.8 made no
difference there (few 1.5 s chunks per turn). RTF with embeddings 0.02.

## Deviations from DESIGN.md (minimal, with evidence)

1. **S1 TDOA estimator.** The DESIGN 2.0 steered response (MSC-weighted Re{Gamma_x e^{-j2pi f tau}}) is
   biased toward broadside by roughly the coherent fraction q: where Gamma_n is near 1 (all bins below
   about 3 kHz at 4 cm) the phase of Gamma_x = q e^{j theta} + (1 - q) Gamma_n shrinks toward 0, and the MSC
   weights favour exactly those bins. On the v1 bus scenes it put partners at 0.5-0.8 m on the axis near
   tau = 0 (e.g. true +103/-85 us -> -17 us). Default `srp_mode='fit'` fits that two-component model per frame
   on the same bins and grid (`common.coherent_fit`), gives q as a by-product, and uses q (instead of psi)
   to select direct frames. `srp_mode='design'` keeps the original for ablation.
2. **Partner posteriors.** The smoothed indicator (DESIGN) plateaus at 0.3-0.5 during a partner's speech
   because tau is near tau_k on only part of the frames, so the DESIGN thresholds (0.75, 0.85, p_A+p_B > 0.8)
   would never fire. p_k = clip(s_k / max(P90(s_k), 0.15), 0, 1), where s_k is the DESIGN smoothing
   (lam 0.95) of [\|tau - tau_k\| < 20 us and q above its median]; online the P90 is a running 20 s value.
   With one histogram peak (tau_A = tau_B) the DESIGN leaves the assignment open: the peak's indicator
   feeds p_A, p_B = 0 and `double_talk` is False, so p_A + p_B is that talker's posterior (before this fix
   both were 0 for the whole file, and the M11-bus AGC gate p_A + p_B > 0.8 never opened in the one-peak
   half of v2 dev bus).
3. **Spacing fit band 200-7500 Hz** (DESIGN: 200-4000 Hz) with weights 1/max(1 - Gamma^2, 0.05)^2. Over
   200-4000 Hz sinc and J0 differ only by ~0.01-0.02 at 2.5 cm (their low-order terms match after rescaling
   d by 0.82), so on 20-30 s of diffuse noise J0 won 25-62 % of trials at 2.5 cm (wrong d by -4.5 mm);
   with 200-7500 Hz the right model won 160/160 trials and \|d error\| <= 0.9 mm (2.5-6 cm).
4. **Leave-own-frame-out D selection.** Selecting D on the frame's own MSC biases the coherence measured
   on D downward in 2.5-6 kHz (with the 4 kHz fit band J0 then won 6/6 trials at 2.5 cm; with 7.5 kHz it
   left a +0.4 to +1.1 mm bias); the selection statistic now excludes frame l's samples (see S0 steps).
5. **Skew: a two-value decision, and the meeting's phase frames.** DESIGN step 4 takes the slope of the
   step-3 fit as tau0. Coherent energy in the noise frames (engine, wheels, talkers that reach D) adds a
   slope of its own: on v2 dev bus the fitted slope missed the true skew by median 9.6 us, P90 36 us, and
   the bias is mostly fixed per geometry, so more frames do not remove it. The skew is therefore chosen
   between the only values one I2S bus gives with Mic A in the left slot, 0 and +31.25 us (Mic B sampled
   half a frame later): tau0 = 31.25 us if the slope >= 15.625 us, else 0. Both error types leave the same
   31.25 us phase ramp, so the threshold sits midway and there is no further gate. v2 dev (offline / online
   at the last refresh): bus 82/90 / 79/90 right, meeting 20/20 / 20/20. The previous gates (band residual
   <= 8 deg, slope within 6 us of a multiple of 31.25 us, >= 600 diffuse frames, spacing-fit RMS <= 0.08)
   corrected 0 of 45 dev bus skews. Adding only the band-residual gate gives 74/90 (0 false, 16 missed)
   against 82/90 (3 false, 5 missed) without it; a minimum frame count only delays correct skews online
   (time-weighted bus accuracy 0.80 -> 0.70 with >= 200 frames). In the meeting the phase is fitted on the
   VAD pauses (the frames of Gamma_n, DESIGN step 6) instead of D: talker frames in D moved the slope by
   -42 to +34 us (4 false skews in 20 rooms, total phase error <= 2 kHz up to 22.5 deg); on pauses the
   slope still moves by -19 to +45 us, but all 20 decisions are right (smallest margin 3.9 us) and the phase
   error is at most 2.4 deg. Online, D now also leaves out the frame's own data (future part from a backward
   pass over each 5 s block); with the causal-only statistic every burst onset leaked into D, and a
   synthetic pair of partners on the A side got a false skew (slope +17 to +22 us). On hardware, calib.json
   `tau0_s` from the endfire chirps (DESIGN section 5) overrides all of this; any other skew value can only
   come from there.
6. **Spacing fallback** uses the prior only when the fit RMS exceeds 0.15 (the DESIGN leaves "large"
   open); v1 bus fits with RMS 0.10-0.15 were still within 2-6 mm, while the prior can be 15-20 mm off.

## Measured performance

Unit tests (`.venv/bin/python -m tests.test_core`: 12 groups, 165 checks, about 24 s; `--quick` 11 groups, 12 s):

| Check | Result |
|---|---|
| Plane wave, d 2.5/4/6 cm, tau in {-d/c, 0, +d/c} | max \|tau error\| 0.65 us ('fit'), 1.03 us ('design') |
| CDR prop1 / prop3 on the mixing model (CDR -5/0/10 dB) | exact (0.0000 dB) |
| Diffuse field, 40 s, 2.5/4/6 cm (+ sensor noise at -10 dB) | Gamma_n vs rho*sinc above 300 Hz: max \|err\| 0.029-0.041; d_hat within 0.2 mm; sinc selected; rho 0.911-0.912 vs 0.909; no skew |
| 2-D (J0) diffuse field, 30 s, 2.5/4/6 cm | J0 selected; d_hat within 0.12 mm; Gamma_n vs J0 max \|err\| 0.030-0.041 where J0 >= -0.3 (the DESIGN clip) |
| E8-style impairments (+1.5 dB, 31.25 us, corners 50/80 Hz), 40 s, meeting | gain max \|err\| 0.08-0.10 dB; slope 31.3-32.2 us, skew 31.25 applied; corner phase max \|err\| 0.16-0.73 deg; residual after apply: phase <= 1.3 deg, level A/B <= 0.18 dB |
| Same, 15 s, bus, skew 31.25 and 0 us, offline and online | skew right in 12/12; corner phase max \|err\| <= 0.4 deg; d_hat within 0.5 mm |
| Synthetic partner pairs at +6 dB (offline and online) | tau_A/tau_B within 1 us; p_A in A's bursts 0.65-0.73, in B's 0.00 |
| One partner only (one peak), offline / online | peak within 0.2 us; p_A in talk 0.71 / 0.66, in pauses 0.12 / 0.11; p_B = 0; p_A + p_B > 0.8 on 59 % / 53 % of talk frames, 0 % / 2 % of pauses |
| Inputs of T = 0, 1, 100, 511, 512, 1000 samples | no exception in any mode; finite outputs of the right length |
| v1 bus (21 scenes, no skew) | \|d_hat err\| median 1.7 mm (71 % <= 3 mm); partner \|tau err\| median 15.6 us, 74 % <= 30 us ('design' SRP: 60.4 us, 36 %); stratum agreement 9/21; gain err <= 0.11 dB |
| v1 (29 scenes, no skew) | false blind skew in 2 (bus s11 +5 dB: slope 16.5 us; meeting s0 60 mm: 20.2 us) |
| Cost | RTF 0.009 offline, 0.013 online (32 s bus); 96 s meeting 0.009 |

v2 dev (`scenes2/dev`, TTS, 90 bus + 20 meeting scenes; truth from the scene meta, offline / online at the
last refresh; `tmp/fix:core/p10_v2dev_fix.py`, a copy of the reviewer's probe):

| | Bus | Meeting |
|---|---|---|
| Skew decision right | 82/90 / 79/90 (3 false, 5 / 8 missed) | 20/20 / 20/20 |
| Total phase error (corners + skew) <= 2 kHz | median 1.2 deg, P90 6.0 (wrong skews give 22 deg) | median 0.8 deg, max 2.4 |
| Gain error | median 0.05 dB, max 0.17 | median 0.02 dB, max 0.08 |
| d_hat | \|err\| median 4.4 mm; prior used in 40/90 | err median +3.7 mm, max 6.3 (all fits) |
| Partner TDOA, error to the nearest peak | median 26.7 us, 46 % <= 20 us | - |
| One histogram peak | 33/90 offline, 34/90 online | - |
| p_A + p_B > 0.8 while a partner talks (one-peak / two-peak scenes, median) | 0.28 / 0.35 offline, 0.29 / 0.27 online | - |
| ... while no partner talks (one-peak / two-peak) | 0.01 / 0.16 offline, 0.18 / 0.24 online | - |

Before this revision: skew right in 45/90 bus (0 of 45 skews corrected) and 15/20 meetings, phase error
<= 2 kHz median 14.9 deg on the bus, bus d_hat \|err\| median 7.8 mm, nearest-peak partner error 39.6 us,
and p_A = p_B = 0 throughout the 45 one-peak bus files.

Synthetic E8 diffuse fields (20 s, 25/40/60 mm, sinc/J0, white/road (bus) or HVAC (meeting) noise, 33 dBA
self-noise, 3 seeds per cell, 108 files; `tmp/fix:core/p2_s0_diffuse_fix.py`): skew right in 108/108 files
offline and 108/108 online (before: 54/72 bus and 19/36 meeting files offline); total phase error
<= 2.1 deg; bus d_hat within 0.7 mm with the right model. Meeting-level fields (42 dB SPL) still give d_hat
8.7-22 mm too large (see limitations).

## Known limitations

- Blind skew: in reverberant rooms the noise frames carry coherent energy (engine, wheels, talkers that reach
  D) that adds its own slope, fixed per geometry. 9 % of v2 dev bus files (12 % online) get the wrong skew,
  and online S0 can apply a false skew for a 5 s block when one talker dominates the first seconds' noise
  frames (anechoic 3-seat test: first refresh slope +101 us = the talker's TDOA). Every later TDOA then
  shifts by 31.25 us; online DP-means splits clusters across such a change. A skew other than 0 / +31.25 us
  (e.g. a one-sample I2S offset) is never found blindly. Use `calib.json` (`tau0_s`) on hardware.
- Meeting spacing is biased upward (v2 dev median +3.7 mm; synthetic 42 dB SPL HVAC with 33 dBA self-noise
  +9 to +22 mm, J0 rooms read as sinc): a constant rho cannot model a white self-noise floor under sloped
  HVAC noise.
- Bus spacing: the prior (40 mm) is used in 40/90 v2 dev files; on 60 mm arrays the S1 grid (+-1.2 d_hat/c)
  then truncates real TDOAs (up to 175 us) and grid-edge pile-ups can become partner peaks.
- S1 partner model on v2-like scenes (reviewer's diagnosis): peaks often land on near passengers, conversing
  pairs or noise sources rather than the partners; one peak in about a third of the bus files; stratum
  agreement <= 0.06. Pure diffuse noise or uncorrelated channels give two partner peaks (grid edges for the
  latter) that are not flagged inseparable. Do not trust tau_A/tau_B/stratum blindly; check `q`, `inseparable`.
- With one peak, p_A follows whichever talker that peak is (also a passenger), so the AGC gate can open on
  non-partners (v2 dev online: 18 % of no-partner frames in one-peak files).
- `double_talk` almost never fires (single-tau indicator).
- 20 meeting talkers fall into 6-12 DP-means clusters; use the embedding route (spk_onnx) when available.
