# 2-mic speech lab: v2 bench

The v2 bench replaces the v1 bench (`BENCH.md`, `lab/scenes.py`, `scenes/`), which stays read-only.
Sections are added by the agent that owns each part. Run everything from the lab folder with `.venv/bin/python`.

## Scene format

Owner: `lab/scene2.py`, `lab/corpus2.py`, `tests/test_scene2.py`. Implements DESIGN §4.1 E5–E9 and §4.2.

### Concepts

- A **scene** is a **geometry** plus one **recipe** per corpus.
  - Geometry: the room, the array, every point source and its RIR pair.
  - Recipe: utterance choices and timings, noise models and seeds, levels, the SNR rule and the E8 impairments.
- Scenes are rendered deterministically when loaded. A 32 s bus scene renders in about 0.5 s and a 6 min meeting in about 2.5 s. Rendered audio is never written to disk.
- A **scene id** is `"<set>/<name>"`, for example `"dev/bus_g003_snr+0"`.
  - Sets: `dev`, `wide`, `test`, `smoke`, plus `demo` (built here).
  - Names must not contain `.` or `/`.
- Several scenes may share one geometry, for example an SNR sweep.
- Switching corpus re-renders the scene on the same geometry. The recipe for the new corpus is made by the geometry's generator from the stored `variant` (see "Corpus switch").

### Files

All files live in `scenes2/<set>/`.

| File | Content |
|---|---|
| `<geom_id>.geom.npz` | `store`: float32 `(S, 2, n)` RIRs. For `tail="synthetic"` only the image-source part (`n = n_x`) is stored, and the stochastic tail is re-synthesized from the seed on load, bit-identically (`test_geometry_store_roundtrip`). For other tails the full RIR is stored |
| `<geom_id>.geom.json` | Geometry dict (below) |
| `<name>.<corpus>.recipe.json` | Recipe dict (below), one file per corpus (`tts`, later `libri`, ...) |

Measured sizes:
- bus geometry with 30 sources at RT60 0.6 s: 0.43 MB (synthetic tail) or 2.4 MB (`tail="ism"`);
- meeting geometry with 21 sources at RT60 0.7 s: 0.39 MB (synthetic tail);
- recipe: 3–30 kB.

The budget for all of `scenes2/` is 600 MB.

### Geometry dict (`.geom.json`, returned by `make_geometry` / `load_geometry`)

| Key | Meaning |
|---|---|
| `geom_id`, `set`, `scenario` | Identity. `scenario` is `"bus"` or `"meeting"` |
| `generator` | `"module:function"` that makes recipes for this geometry (see "Writing a generator") |
| `seed` | RIR seed. `pra.random.seed(numpy=seed, libroom=seed)` is called inside `compute_rirs` (E9) |
| `fs` | 16000 |
| `room` | `dims`, `rt60`, `tail` (`synthetic` / `ism` / `none`), `max_order`, `e_abs`, `rand_disp`, `exact_direct`, `edited` (an `edit_images` hook was used), `seed`, `n_rir` (RIR length), `n_x` (synthetic only) |
| `mics` | `[[x,y,z] Mic A, [x,y,z] Mic B]` in m, rounded to 1 µm |
| `spacing_m`, `center`, `array_azimuth_deg`, `array_elevation_deg` | True array geometry. Mic A = center + (d/2)·u (DESIGN §2.0) |
| `sources` | List of `{name, pos, directivity, dist_m, dist_a_m, theta_deg, fold_deg, tau_us, direct_a}` |
| `meta` | Generator-specific JSON: strata, distance bins, table options, and so on |

Fields of each `sources` entry:
- `theta_deg`: angle from the mic axis u (0° = Mic A side), 0–180°. `fold_deg` is folded to 0–90° (the E1 strata).
- `tau_us`: exact near-field TDOA (r_B − r_A)/c in µs. It is positive when Mic B hears later, the same sign as S1's τ.
- `direct_a`: sample index of the direct path at Mic A inside the RIR. It equals r_A/c·fs + 40.
- `directivity`: `None` (omni) or `{azimuth_deg, colatitude_deg, p}`. This is a pyroomacoustics `CardioidFamily` with response p + (1 − p)·cos: p = 1 omni, 0.75 sub-cardioid (the talker default), 0.5 cardioid.

### Recipe dict (`.recipe.json`)

Top level:

| Key | Meaning |
|---|---|
| `scene_id`, `set`, `geom_id`, `scenario`, `corpus`, `variant` | Filled in by `add_scene`; the generator does not set them |
| `n_samples` | Scene length T in samples |
| `seed` | Root seed for all render-time randomness. Source k of group g uses `default_rng([seed, crc32(g), crc32(k)])`; the sensor uses `[seed, crc32("sensor")]` |
| `impair` | E8 truth `{gain_db: [A, B], hp_hz: [A, B], skew_us}`. Use `draw_impairments(rng)`: ±1 dB, 50–80 Hz, skew 0 or 31.25 µs |
| `sensor_dba` | Optional. Self-noise level, default 33 (dBA SPL-equivalent, INMP441) |
| `groups` | Ordered list `{name, role, sources: [...]}`. The order is the canonical summation order. `sensor` is reserved and appended last by the renderer |
| `rules` | Optional list of `{type: "snr", target: [groups], noise: [groups], snr_db, scale: "noise" (default) or "target"}` |
| `meta` | Optional generator JSON (voices, turn plan, ...) |

**Roles** are free strings. Use this vocabulary:
- bus: `partner`, `passenger_near`, `passenger_cone`, `conv_pair`, `babble_far`, `engine_road`, `transient`, `handling`, `probe`;
- meeting: `talker`, `hvac`, `fan`;
- reserved: `sensor`.

Give every partner, near passenger, probe and meeting talker its own group. A conversing pair is one `conv_pair` group with two talkers.

Recommended names, so metrics code can rely on them:
- bus partners: groups `partner_A` and `partner_B`;
- meeting talkers: groups `talker_00` to `talker_19`;
- talker ids: short and unique, for example `pA`, `pB`, `near1`, `t07`.

**Source types**

| `type` | Fields | Rendered as |
|---|---|---|
| `point` | `src` (a geometry source name), `level_db`, `signal`, optional `talker` (id), optional `weighting` (`"Z"` or `"A"`) | Dry signal convolved with that source's RIR pair |
| `diffuse` | `level_db`, `signal` (kind `noise`), `model` (`"sinc"`, 3-D, default; or `"j0"`, 2-D), optional `weighting` | 2-ch noise with coherence sinc(2fd/c) or J0(2πfd/c) at the true spacing |
| `uncorrelated` | `level_db`, `signal` (kind `noise`), optional `weighting` | Independent noise per mic: handling, vibration, wind (E6c) |
| `field` | `level_db`, `signal` (kind `func` returning `(2, T)`) | A 2-ch field given directly, for example recorded 2-ch noise (E6a) |

Only a `point` source with a `speech` signal may carry a `talker` id. Such a talker gets a ref, segs and a `talker_group` entry. Talker ids are unique within a scene.

**Signal kinds.** Each kind produces a unit-level dry signal; `level_db` then scales it.

| `kind` | Fields | Unit level |
|---|---|---|
| `speech` | `events: [[utt_id, start_sample, gain_db], ...]`, all from one speaker | Each utterance has active RMS 1 (`lab.common.active_rms`), × 10^(gain_db/20). `start_sample` is the dry onset; the sound reaches Mic A `direct_a` samples later |
| `noise` | `shape` (optional `[[f_hz, dB], ...]`, interpolated over log f, flat outside the points; default white), `bursts`, `env` | Gaussian noise, RMS 1 |
| `harmonic` | `f0`, `f0_dev` (Hz), `f0_rate` (Hz), `n_harm`, `rolloff` (amplitude 1/k^rolloff), optional `noise_db` + `noise_shape` (added noise relative to the tones), `bursts`, `env` | RMS 1. For engine, fan hum and brake squeal |
| `func` | `func: "module:function"`, `params` | `fn(T, fs, rng, **params)` returns `(T,)` (point) or `(2, T)` (field), already at unit level; it must be deterministic given `rng` |

- `bursts: [[start_sample, length, gain_db], ...]` restricts `noise` and `harmonic` signals to those spans. Each span is multiplied by an envelope with peak 1:
  - `env: "decay"` (default): 2 ms attack, then exponential decay to −43 dB at the end of the span;
  - `env: "gate"`: flat with 10 ms raised-cosine fades.
- `level_db` meaning:
  - `point`: the dry signal's level in dB SPL at 1 m on the source axis. pyroomacoustics 0.10.1 RIRs have direct-path gain 1/r, so the direct sound at r m is `level_db − 20·log10(r)`.
  - `diffuse` and `uncorrelated`: the level at each mic.
  - `field`: the level of the unit-level field.
  - With bursts, the level under the envelope peak.
  - `weighting: "A"` makes `level_db` an A-weighted level, for example HVAC at 35/40/45 dBA.

**SNR rule.** The rule is evaluated after all groups are rendered:
- P_t = mean in-band power of the `target` groups at Mic A, over the samples where any target talker is active (from segs). In-band means 150–7000 Hz: `butter(4, [150, 7000], "band")` applied with `sosfiltfilt`, as in v1;
- P_n = the same for the `noise` groups, over the same samples;
- the noise groups are then scaled by √(P_t/(P_n·10^(snr/10))). With `scale: "target"`, the target groups are scaled instead.

Rules use the acoustic (pre-microphone) Mic-A signals, so `impair=False` renders the identical acoustic scene. `meta.rules[i].realized_snr_db` is measured on the recorded comp2. With impairments on it differs from `snr_db` by about 0.1 dB, because of the high-pass filter.

### Render pipeline (per scene, per group)

1. Build the dry signals and scale them to `level_db`. Convolve point sources with their RIRs. For talkers, also convolve with the early RIR at Mic A, `h_A[: ceil(direct_a) + 800]` (direct path plus 50 ms).
2. Apply the SNR rules (group gains).
3. Add the `sensor` group: white Gaussian noise, independent per mic, at 33 dBA SPL-equivalent.
4. Apply the E8 chain identically to every group, on each channel, in this order:
   - first-order high-pass (`butter(1, hp_hz[m], "highpass")`, causal);
   - × 10^(gain_db[m]/20);
   - Mic B only: exact fractional delay by `skew_us`, as linear phase on the FFT of the whole signal (or of each padded block of a Sparse signal).

   Refs pass through Mic A's part of the chain, so a ref is exactly the early part of its talker's comp2 at Mic A.
5. Convert to digital (× `lab.common.PA_TO_DIG`) and store as float32.
6. Compute `mix` as the float32 accumulation of `comp2` in dict order, so `sum(comp2.values()) == mix` holds bit for bit in that order and within 1e-6 in any order.

### Scene dict (returned by `load_scene`)

| Key | Type | Meaning |
|---|---|---|
| `id` | str | `"<set>/<name>"` |
| `fs` | int | 16000 |
| `mix` | float32 `(2, T)`, read-only | Method input (row 0 = Mic A), digital full scale 1.0 |
| `comp2` | `{group: (2, T) float32}` | Every group after level rules, impairments and digital scaling, in canonical order, ending with `"sensor"`. Σ comp2 = mix |
| `groups` | `{group: role}` | Same keys as `comp2` |
| `refs` | `{talker_id: (T,) float32}` | Early image (direct path + 50 ms) at Mic A, same digital scale and Mic-A chain as comp2, aligned with comp2 (lag 0) |
| `talker_group` | `{talker_id: group}` | Group holding each talker |
| `segs` | `[(start, end, talker_id)]` | In mix samples: utterance extent shifted by the talker's direct-path delay (`round(direct_a)`), clipped to T, sorted by start |
| `meta` | dict | See below |

`comp2` and `refs` values are numpy arrays or `lab.scene2.Sparse`:
- Sparse is used when a signal's non-zero blocks cover ≤ 50% of T, for example meeting talkers. A 6 min meeting then holds 209 MB of arrays instead of 1.5 GB.
- `np.asarray(v)` or `v.dense()` gives the ndarray. `v[c]` gives channel c, as a Sparse.
- `v[a:b]`, `v[c, a:b]` and `v[:, a:b]` return dense slices without materializing the rest.
- Arithmetic, and any other ndarray attribute (`v.sum()`, `v.astype(...)`), act on the dense array.
- `v.blocks` is a list of `(start, array)`; `v.support()` lists the `(start, end)` intervals.
- All arrays are read-only (renders are cached), so copy before modifying. Benches should pass `np.array(sc["mix"])`, a writable copy, to methods.

`meta` keys:
- Generator `geom.meta` keys and `variant` keys are merged at the top level, for example `stratum` and `snr_db`.
- Core keys, which take precedence: `scene_id`, `set`, `scenario`, `corpus`, `geom_id`, `generator`, `n_samples`, `duration_s`, `fs`, `spacing_m` (true), `mics`, `center`, `array_azimuth_deg`, `array_elevation_deg`, `room`, `rt60`, `sources` (with geometric truth), `talkers`, `impair` (`{gain_db, hp_hz, skew_us, applied}`; zeros when `impair=False`), `sensor_dba`, `rules` (with `gain_db` and `realized_snr_db`), `group_gain_db`, `peak` (max |mix|), `clipped` (peak ≥ 1; nothing is clipped, values stay linear), `seeds` (`{geometry, recipe}`), `variant`, `recipe_meta`.
- `talkers` is `{talker_id: {group, src, speaker, level_db, utts, dist_m, theta_deg, fold_deg, tau_us}}`.

### API (`lab.scene2`)

```python
load_scene(scene_id, corpus="tts", impair=True, root=None) -> dict      # recipe made on first use if missing
list_scenes(set_name, scenario=None, corpus=None, root=None, **filters) -> [scene_id, ...]
scene_card(scene_id, root=None) -> dict          # filter fields: geom meta + variant + scene_id, set, scenario,
                                                 # geom_id, corpora, spacing_m, rt60, tail, duration_s
signed_mix(scene, flip=(), drop=()) -> float32 (2, T)   # Hagerman runs; flip=() and drop=() gives mix exactly
clear_cache()
make_geometry(set_name, geom_id, scenario, dims, rt60, mics, sources, seed, generator, meta=None, root=None,
              **rir_opts) -> geom dict           # rir_opts: compute_rirs options below
compute_rirs(dims, rt60, srcs, mics, seed, max_order=18, tail="synthetic", directivity=None,
             edit_images=None, rand_disp=0.05, exact_direct=True) -> (rirs (S,2,L) float32, store, params)
add_image(room, pos, gain, mirror_axis=2)        # inside edit_images: extra image of the room's source
add_scene(set_name, name, geom, variant, corpus="tts", root=None) -> scene_id
ensure_recipe(scene_id, corpus="tts", root=None) -> recipe dict
load_geometry(set_name, geom_id, root=None) -> (geom dict, rirs)
render(geom, rirs, recipe, impair=True) -> scene dict
draw_impairments(rng, skew_us=None) -> impair dict
place_array(center, spacing, azimuth_deg, elevation_deg=0.0) -> [micA, micB]
facing(pos, target, p=0.75) -> directivity dict  # source at pos facing target
source_truth(pos, mics) -> {dist_m, dist_a_m, theta_deg, fold_deg, tau_us, direct_a}
a_weight(f) -> A-weighting amplitude (1 at 1 kHz)
Sparse                                           # see above
FRAC = 40, EARLY = 800, ROOT = "<LAB>/scenes2"
```

Notes on the API:
- `list_scenes` filters: `key=value` tests equality, `key=(v1, v2)` tests membership, `key=callable` is a predicate, all on `scene_card` fields. Example: `list_scenes("dev", "bus", stratum="aligned", snr_db=(0.0, 5.0))`.
- Caching: the last 2 renders and 4 geometries are cached per process.
- `compute_rirs` details:
  - `edit_images(room, k)` is called once per source k, after `room.image_source_model()` and before `room.compute_rir()`. The room holds only that source (`room.sources[0]`), so you can edit `.damping`, look up first-order images with `.orders_xyz` (for example `[0, 0, -1]` is the floor), or call `add_image` (E7 table reflection).
  - `exact_direct=True` resets the order-0 image to the true position. pyroomacoustics' randomized ISM otherwise moves the direct path by up to `rand_disp` per axis, which matters for 0.3 m partners.
  - `tail="ism"` uses the Sabine order (E6b). Measured: 0.5 s per source and about 1 GB peak RSS at RT60 0.6 s (bus) or 0.9 s (meeting). `"synthetic"` takes about 3 ms per source.
- Directivity uses the image-source method in shoebox rooms only, as in pyroomacoustics 0.10.1, and is frequency-independent (DESIGN §1.6 item 14).
- Absorption is broadband (`pra.inverse_sabine`) and air absorption is off, as in v1.
- The geometry JSON does not store an `edit_images` hook. Record what the hook did in `meta`; geometries with `edited: true` are reproducible only through their generator.

### Writing a generator

```python
from lab import scene2 as s2
geom = s2.make_geometry("dev", "bus_dev_g003", "bus", dims, rt60, mics, sources, seed=1003,
                        generator="lab.scene_bus:make_recipe", meta={"stratum": "aligned", "dist_bin": "0.3-0.45"},
                        tail="synthetic")
for snr in (-10, -5, 0, 5, 10):
    s2.add_scene("dev", f"bus_g003_snr{snr:+d}", geom, {"seed": 1003, "snr_db": float(snr)})   # corpus "tts"

# in lab/scene_bus.py
def make_recipe(geom, corpus, seed, snr_db):        # keyword args = the variant dict
    rs = np.random.default_rng([seed, 1])            # speech choices only
    rn = np.random.default_rng([seed, 2])            # everything corpus-independent
    ...
    return {"n_samples": T, "seed": int(rn.integers(2**31)), "impair": s2.draw_impairments(rn),
            "groups": [...], "rules": [{"type": "snr", "target": [...], "noise": [...], "snr_db": snr_db}],
            "meta": {...}}
```

- Contract: `make_recipe(geom, corpus, **variant)` must be importable by name and deterministic.
- Take speech choices from their own generator and everything else (noise, impairments, render seed) from another. A corpus switch then changes only the speech. `test_corpus_switch_same_geometry` checks this for `demo_recipe`.
- Pick speakers with `corpus.speakers_for(role, split)`, where split is `"test"` for the test set and `"dev"` otherwise. Pick utterances with `corpus.select(...)`. Never hard-code TTS file names.
- `lab.scene2.demo_geometry` / `demo_recipe` / `build_demo` are a complete small example. `python -m lab.scene2 --demo` builds `scenes2/demo/` (about 50 kB) and `--list <set>` lists a set.
- Parallel generation: seeds are applied inside each call, so worker processes are safe. Files are written atomically.

### Corpus switch

`load_scene(sid, corpus="libri")` with no `libri` recipe calls `ensure_recipe`. It reads the stored `variant` from an existing recipe, calls the geometry's generator with `corpus2.load_corpus("libri")`, validates the result and writes `<name>.libri.recipe.json`. The geometry, impairments and noise seeds stay the same.

### Corpora (`lab.corpus2`)

`load_corpus(name="tts") -> Corpus`:
- `.utts`: list of `Utt(id, speaker, text, path, duration, pool, source, split, gender, style)`. `path` is relative to the lab folder, `duration` is in seconds, `pool` is `"target"` or `"babble"`, `split` is `"dev"`, `"test"` or `"all"`.
- `.speakers`: `{speaker: {pool, split, gender, quality, roles, n_utts, total_s}}`.
- `.speakers_for(role, split=None)`: role is `"partner"`, `"talker"` or `"babble"`. Split `"dev"` also matches `"all"`.
- `.select(speaker=, pool=, split=, role=, style=, min_dur=, max_dur=)` returns a list of Utt.
- `.audio(utt_id)`: float32 at 16 kHz with active RMS 1, read-only and cached.
- `utt_id in corpus` tests membership.
- Pools are speaker-level, so target speakers never appear in babble.

**TTS voice eligibility (`corpus="tts"`, built from `corpus/manifest.json`)**

| Voices | Roles | Notes |
|---|---|---|
| 8 natural English voices: Samantha, Daniel, Karen, Moira, Tessa, Rishi, Aman, Tara | `partner`, `talker` | 50 lines each: 22 bus dialogue, 20 meeting, 8 babble |
| 12 natural voices of European locales reading English: Anna, Thomas, Xander, Ellen, Nora, Alva, Sara, Satu, Alice, Mónica, Joana, Zosia | `talker` | A meeting needs 20 distinct voices and only 8 natural English voices exist. 8 babble lines each (about 17 s) |
| The other 27 natural non-English voices | `babble` | Bus interferers: near passengers, conversing pairs, far babble |
| Excluded: Fred, Kathy, Ralph, Junior (formant synthesis) and all 16 Eloquence novelty voices (`"Name (English (US/UK))"`) | — | Not natural speech (DESIGN §4.1) |

- `style` is `dialogue`, `meeting` or `babble`; ids are file stems such as `samantha__bus03`.
- TTS has no dev/test speaker split (`split="all"`): too few voices. It is for smoke tests only; G1 is decided on natural speech (E0).

**Drop-in manifest for other corpora** (for example LibriSpeech) goes in `corpora/<name>/manifest.json`:

```json
{"name": "libri", "source": "libri",
 "speakers": {"1089": {"pool": "target", "split": "dev", "gender": "M"},
              "1688": {"pool": "babble", "split": "test", "gender": "F", "roles": ["babble"]}},
 "utterances": [{"id": "1089-134686-0000", "speaker": "1089", "text": "HE HOPED THERE ...",
                 "path": "corpora/libri/test-clean/1089/134686/1089-134686-0000.flac", "duration": 10.435}]}
```

- Roles default to `["partner", "talker"]` for the target pool and `["babble"]` for the babble pool.
- Audio must be 16 kHz; mono is preferred, and the first channel is used otherwise.
- `corpus2.register(name, path)` points a name at another manifest for the current process (tests).

### Tests

Run `.venv/bin/python -m tests.test_scene2`: 17 tests, about 1.5 s. They build the demo in a temporary root and cover:
- Σ comp2 = mix (exact);
- render determinism;
- bit-identical RIR regeneration (E9) and the stored-geometry round trip;
- impairment truth (Welch transfer function per mic: gain, high-pass, skew);
- refs at lag 0 relative to comp2, at the direct-path delay relative to the dry speech, with segs consistent;
- SNR rule and sensor 33 dBA;
- A-weighted levels and diffuse coherence (sinc and J0);
- Hagerman 4-run decode within 1e-6;
- Sparse behaviour;
- list filters;
- corpus switch via a drop-in manifest;
- directivity and `add_image`;
- tail modes (RT60);
- recipe validation;
- render time.

### Differences from v1 that affect levels

v1 (`lab/scenes.py`) scaled point sources by 4π, assuming RIR gain 1/(4πr). pyroomacoustics 0.10.1 RIRs have gain 1/r, as measured: the direct-path filter sums to 1/r. So every v1 point source was 22 dB louder than its stated dB SPL:
- v1 meeting HVAC (diffuse, not scaled) was 22 dB too low relative to the talkers;
- the v1 sensor self-noise was 22 dB too low relative to all point sources.

v2 uses the physical scale.

## Bus scenes

Owner: `lab/scene_bus.py`, `tests/test_scene_bus.py`. Implements DESIGN §4.1 E1–E6 and E8–E9 for the bus case on top of the scene core above. Speech is TTS until a natural-speech corpus is approved (E0); the same recipes re-render with any corpus through the corpus switch.

### What is simulated

**Cabin and device**
- Shoebox cabin 11.5 × 2.5 × 2.3 m, each dimension scaled by U(0.95, 1.05). Nominal RT60 0.15–0.6 s (E3), broadband absorption from `pra.inverse_sabine`.
- Device at seated height: center 0.85–1.2 m above the floor (held, on a seat back), mic axis horizontal. Lap-flat stratum: 0.5–0.78 m, axis tilted ±10° by the lap.
- True spacing 25, 40 or 60 mm (balanced, E8). Methods are told 40 mm.

**Partners (E1, E2, E5).** The stratum fixes how the partners sit relative to the mic axis. Angles are to the axis, folded to 0–90°.

| Stratum | Layout | Acceptance rule |
|---|---|---|
| `aligned` | Opposite ends of the device, facing each other across it | both folds ≤ 20°; partners ≥ 110° apart as seen from the device |
| `45` | Same, other orientation | both folds 30–60°; ≥ 110° apart |
| `broadside` | Same, other orientation | both folds ≥ 70°; ≥ 110° apart |
| `side_by_side` | Two neighbours (seated, 15% standing), mouths 0.42–0.62 m apart, device 0.3–0.95 m in front | same side of the broadside plane and \|θ_A − θ_B\| 20–40° (the S1 tracker rule), not one of the three strata above, and the pair subtends 20–40° |
| `lap_flat` | Device in one partner's lap; the other partner sits beside the owner (70%) or stands in front (30%) | both \|τ\| ≤ 0.28·d/c (DESIGN: < 0.3·d/c) |

- Distances 0.3–1.0 m from the device center. In the three opposed strata, each set uses the 6 bin pairs (A, B) = (1,1), (2,2), (3,3), (1,2), (2,3), (3,1) once per stratum, bins 0.3–0.45 / 0.45–0.65 / 0.65–1.0 m. Side-by-side and lap-flat distances follow from the layout.
- Partners face each other: `CardioidFamily` p = 0.75 (`lab.scene2.facing`). Mouth heights: seated 1.05–1.32 m, standing 1.45–1.72 m.
- Lombard level U(62, 72) dB SPL at 1 m per partner, plus ±1.5 dB per utterance.
- Turn-taking (`_dialogue`): turns of 1–2 utterances (TTS `dialogue` lines of 0.9–6.5 s). Transitions overlap by 0.15–0.7 s (double talk, 20%), leave a 0.1–0.6 s gap (60%) or a 0.8–2 s pause (20%). In 40% of turns longer than 2.5 s, the listener interjects one of their 3 shortest lines at −8 to −4 dB (back-channel stand-in, see limitations). A talker never overlaps themselves.
- Partners are silent for the first 1.5 s (lead-in) plus 0–0.4 s, and stop 0.25 s before the end.

**Interferers (E4).** Every passenger speaks at U(62, 72) dB SPL at 1 m, the partners' distribution. Passengers face random azimuths (p = 0.75); conversing pairs face each other. There are 16–25 passengers in all, with mouths at least 0.42 m apart.

| Group (role) | Talker ids | Placement | Speech |
|---|---|---|---|
| `near_1`..`near_3` (`passenger_near`) | `n1`..`n3` | 0.6–1.4 m from the device (`near_1` at 0.6–0.85 m), random azimuth, 30% standing | Talk spurts of 1–3 utterances, activity 25–55% |
| `cone` (`passenger_cone`) | `cone` | 1.5–2.0 m, angle to the axis within ±2° of partner A's (A's cone of confusion), on A's side of the axis when the cabin allows | Spurts, activity 25–55% |
| `conv_1`, `conv_2` (`conv_pair`) | `c1a`, `c1b`, `c2a`, `c2b` | 1–2 pairs, both members 0.6–1.2 m from the device, 0.45–1.0 m apart | Alternating turns (as the partners, without back-channels) |
| `babble_far` (`babble_far`) | none (no refs) | All remaining passengers, beyond max(1.2 m, the farthest near or pair talker) | Spurts, activity 15–50% |

All interferer speech can start during the lead-in. Babble speakers are drawn without replacement while the corpus has enough (27 TTS babble voices); otherwise voices repeat.

**Noise (E4, E6).** Levels are physical before the SNR rule.

| Group | Sources |
|---|---|
| `engine_road` | Engine at the rear wall (0.45–0.8 m high): harmonic f0 28–45 Hz ±1.5–5 Hz, 40 harmonics, plus low-pass noise at −6 dB, 66–74 dBA at 1 m. Four wheel arches (front and rear axle, 0.3–0.45 m high, 0.15 m from the side walls): independent road noise, 60–66 dBA at 1 m each. Diffuse road noise, 58–66 dBA at the mics, coherence `sinc` or `j0` (balanced per stratum and spacing) |
| `transient` | 1–3 events at least 2 s apart: door hiss (0.4–1.0 s), door thump (0.12–0.3 s; both at the front or middle door), brake hiss (0.3–0.8 s) or brake squeal (3 harmonics at f0 1.2–3.2 kHz, 0.4–1.5 s; at a wheel arch). Fast-weighted maximum LZFmax 85–97 dB SPL at the device ([96]). The level is set through the source's measured octave-band gain to Mic A, and capped at 103 dB SPL at the device under the envelope peak |
| `handling` | Only in a third of the geometries (E6c): 2–5 spatially uncorrelated bursts below 500 Hz (taps: decay envelope; rubs: gated), 72–84 dB SPL per mic under the envelope peak |
| `sensor` | Added by the core: 33 dBA self-noise |

- Point-source noise spectra roll off below 30 Hz.
- E6(a) hook: the variant key `recorded` = `{"files": [...], "level_db": dBA, "channels": [0, 1], "replace": bool}` adds a `field` source (`lab.scene_bus:recorded_noise`) to `engine_road`. It plays a random excerpt of a recorded 2-channel noise (one multichannel file or two mono files, 16 kHz, paths relative to the lab folder). With `"replace": true` it replaces the synthetic engine, wheel and road noise. No set uses it yet: the real-noise download is pending approval.

**SNR rule (render-time parameter `snr_db`).** The rule sets the in-band (150–7000 Hz) SNR at Mic A, measured over partner-active samples, between the partners (`partner_A` + `partner_B`) and every continuous interferer: `near_*`, `cone`, `conv_*`, `babble_far`, `engine_road`.
- These groups are scaled together (`scale: "noise"`), so their relative levels stay physical.
- `transient`, `handling` and `sensor` keep their physical level relative to the partners and are not in the rule (see limitations).
- All SNR variants of a geometry share every signal; only the rule gain differs (`test_snr_exact_and_scaling`).

**Other choices**
- E8: per-mic gain ±1 dB and high-pass corners 50–80 Hz from `draw_impairments`. The 31.25 µs skew is on in 3 of 6 geometries per stratum.
- E9: seeds are applied inside the core. Regenerating a geometry and its recipes is bit-identical, checked on 5 geometries of all three sets (both tails) against the built files.

### Sets

| Set | Geometries | Scenes | SNR (dB) | Duration | Seeds | Partner speakers (TTS) |
|---|---|---|---|---|---|---|
| `dev` | 30: 6 per stratum | 90 | −5, 0, +5 | 24 s | 10000–10029 | Karen, Samantha, Tessa, Daniel |
| `test` | 30: 6 per stratum | 110 | −5, 0, +5; the `sweep` subset (10 geometries, 2 per stratum, one per tail) also −10 and +10 | 24 s | 20000–20029 | Moira, Tara, Aman, Rishi |
| `smoke` | 3: aligned/40 mm, side_by_side/60 mm, lap_flat/25 mm (handling); 16 passengers, 1 pair, synthetic tail | 3 | 0 | 12 s | 30000–30002 | dev speakers |

**Balance within each stratum of dev and test (6 geometries)**
- Spacing × tail: each of 25/40/60 mm × {synthetic, ism} once, so half the geometries use the full image-source tail (E6b).
- RT60: each of the 6 sub-bins of 0.15–0.6 s (width 0.075 s) once; each tail covers all three thirds.
- Handling in 2 geometries (one per tail), skew in 3, diffuse road model 3 `sinc` + 3 `j0`.
- The opposed strata use the distance-bin pairs listed above.
- `.venv/bin/python -m lab.scene_bus design dev` prints the design table.

**Speakers**
- TTS has no speaker split, so `partner_speakers()` halves the 8 natural English voices deterministically (alternating within gender): dev and test partners are disjoint.
- A corpus with `dev`/`test` labels (for example LibriSpeech) uses its own labelled partners.
- Babble voices are shared between sets (27 TTS voices).

**Names**
- Geometry id `bus_<set>_g<NN>`, scene id `<set>/bus_g<NN>_snr<±S>`, for example `dev/bus_g07_snr-5` or `test/bus_g08_snr+10`. NN interleaves the strata: g00 aligned, g01 45, g02 broadside, g03 side_by_side, g04 lap_flat, g05 aligned, and so on.
- Variant: `{seed, snr_db, seconds}`; `recorded` optional.

**Filtering** (`scene_card` fields = geometry meta + variant):
```python
s2.list_scenes("test", "bus", sweep=True)                                   # 50 sweep scenes
s2.list_scenes("dev", "bus", stratum="aligned", snr_db=0.0,
               partner_dist_max_m=lambda v: v <= 0.6)                        # G2 cell: 2 scenes in dev
```

**Geometry meta keys**
- Design: `stratum`, `replicate`, `spacing_mm`, `rt60_bin`, `handling`, `sweep`, `skew_us`, `road_model`.
- Partners: `dist_A_m`, `dist_B_m`, `dist_bin_A`, `dist_bin_B`, `partner_dist_max_m`, `theta_A_deg`, `theta_B_deg`, `fold_A_deg`, `fold_B_deg`, `tau_A_us`, `tau_B_us` (exact near-field, > 0: Mic B later), `partner_angle_deg` (angle between the partners seen from the device), `angle_class` (the angle rule; lap-flat gives `broadside`).
- `inseparable` and `inseparable_why`: the DESIGN S1 "spatially inseparable" truth, `close_tdoas` (\|τ_A − τ_B\| < 25 µs) or `both_broadside` (both \|τ\| < 0.3·d/c).
- Layout: `device_height_m`, `lap_owner` and `pose` (lap-flat; `pose` also for side-by-side), `n_passengers`, `n_conv_pairs`, `n_far`, `nearest_passenger_m`, `cone_dist_m`, `cone_dtheta_deg`, `rc_sabine_m`.
- `people` (talker ids by role) and `noise_src` (engine, wheels, doors).
- `acoustics` (measured on the stored RIRs at Mic A):
  - `edt_s`, `t20_s`, `t30_s` and `rc_eff_m` (r·10^(DRR/20)): medians over the far passengers;
  - `drr_A_db`, `drr_B_db`: partner DRR, direct path ±0.5 ms;
  - `gain_db`: per source, the level at Mic A relative to `level_db`, from Σh²;
  - `band_gain_db`: the same in octave bands 31.5 Hz–4 kHz, for the noise sources.

Recipe `meta` holds the speakers per talker id, the Lombard levels, the passenger activities, the engine, wheel and road levels, the transient list (`kind`, `src`, `start`, `length`, `lfmax_target_db`, `capped`), the handling bursts and `recorded`.

### Build, validate, test

```
.venv/bin/python -m lab.scene_bus build dev test smoke -j 4     # 186 s wall
.venv/bin/python -m lab.scene_bus validate dev test smoke -j 3  # 96 s wall; renders every scene twice
.venv/bin/python -m tests.test_scene_bus                         # 11 tests, about 8 s
```

**Build cost**
- Geometry CPU time: dev 345 s, test 389 s, smoke 1 s.
- The slowest geometry took 74.5 s: a full-order image-source tail at RT60 ≈ 0.6 s, 29–34 sources, under load. Its RIRs are computed twice: once to measure `acoustics`, once inside `make_geometry`.
- Recipes take about 1 s per set. Peak memory is about 1.2 GB per worker for the full-order image-source tail.

**Disk**
- dev 30.8 MB, test 32.6 MB, smoke 1.2 MB: 64.6 MB in all (budget 350 MB).
- Geometry npz: 0.34–0.49 MB with the synthetic tail, 0.69–2.56 MB with the full image-source tail.
- Recipes 13–22 kB; geometry JSON 11–14 kB.

**Tests** (`tests/test_scene_bus.py`; temporary root)
- Design balance.
- Every design layout against its stratum, distance, cone, passenger and separation rules.
- Layout determinism.
- Groups, refs and talker ids.
- Exact SNR and pure noise-gain scaling between SNR variants.
- Transient and handling levels.
- Turn-taking statistics over 20 recipes.
- Corpus switch with a LibriSpeech-style manifest, and the dev/test partner split.
- The E6(a) hook.
- Measured acoustics, and a full-order image-source room.
- Render time.

### Validation (all 63 geometries, 203 scenes)

**Partner angles**

| Stratum | n | fold A (°) | fold B (°) | Angle rule | Inseparable truth |
|---|---|---|---|---|---|
| aligned | 13 | 4.1–19.0 (median 12.1) | 0.8–20.0 (12.2) | 13 × aligned | 0 |
| 45 | 12 | 31.1–59.3 (46.2) | 32.1–56.5 (40.8) | 12 × 45 | 0 |
| broadside | 12 | 71.9–88.3 (81.6) | 71.1–88.8 (79.5) | 12 × broadside | 9 |
| side_by_side | 13 | 17.4–89.3 (36.1) | 12.9–89.8 (55.5) | 13 × side_by_side; \|θ_A − θ_B\| 21.1–37.7°; subtended 27.4–39.4° | 1 |
| lap_flat | 13 | 74.1–89.3 (78.9) | 74.6–88.1 (82.2) | 13 × broadside | 13 |

**TDOA truth \|τ\| in µs (and \|τ\|/(d/c))**

| Stratum | 25 mm | 40 mm | 60 mm |
|---|---|---|---|
| aligned | 69.6–72.7 (0.96–1.00) | 110.3–116.3 (0.95–1.00) | 164.4–174.9 (0.94–1.00) |
| 45 | 37.2–62.0 (0.51–0.85) | 64.3–99.9 (0.55–0.86) | 104.0–144.2 (0.59–0.82) |
| broadside | 3.4–22.4 (0.05–0.31) | 2.5–37.7 (0.02–0.32) | 6.8–42.9 (0.04–0.25) |
| side_by_side | 28.0–67.2 (0.38–0.92) | 0.4–113.7 (0.00–0.97) | 7.0–168.3 (0.04–0.96) |
| lap_flat | 0.9–19.7 (0.01–0.27) | 5.6–31.0 (0.05–0.27) | 5.9–47.8 (0.03–0.27) |

All lap-flat partners have \|τ\| < 0.3·d/c.

**Distances (partners per bin 0.3–0.45 / 0.45–0.65 / 0.65–1.0 m)**
- aligned 9/9/8, 45 8/8/8, broadside 8/8/8.
- side_by_side 0/1/25 (0.54–0.93 m); lap_flat 2/12/12 (0.45–0.99 m).

**Interferers**
- Nearest passenger 0.60–0.85 m (median 0.68).
- Cone passenger 1.50–1.99 m, its angle within −1.9° to +2.0° of A's.
- 16–25 passengers (median 20).

**Room acoustics, measured / nominal RT60**

| Tail | EDT | T20 | T30 | rc_eff / rc_Sabine | Image-source order |
|---|---|---|---|---|---|
| synthetic (33 rooms) | 1.02 (0.62–1.20) | 1.06 (1.01–1.98) | 1.03 (1.01–1.52) | 1.19 (0.90–2.34) | 18 |
| ism, full order (30 rooms) | 1.22 (0.76–1.39) | 2.21 (1.86–2.39) | 2.23 (2.04–2.34) | 1.16 (0.90–1.60) | 35–117 (Sabine order, median 76; no cap needed) |

Examples (s):

| Room | Tail | RT60 nominal | EDT | T20 | T30 | rc_eff / Sabine (m) |
|---|---|---|---|---|---|---|
| bus_test_g11 | synthetic | 0.153 | 0.095 | 0.303 | 0.232 | 2.80 / 1.20 |
| bus_test_g01 | synthetic | 0.384 | 0.402 | 0.404 | 0.394 | 0.81 / 0.74 |
| bus_dev_g03 | synthetic | 0.560 | 0.560 | 0.567 | 0.565 | 0.58 / 0.63 |
| bus_dev_g18 | ism | 0.174 | 0.176 | 0.325 | 0.355 | 1.46 / 1.09 |
| bus_test_g19 | ism | 0.331 | 0.395 | 0.717 | 0.720 | 0.87 / 0.83 |
| bus_test_g20 | ism | 0.527 | 0.660 | 1.163 | 1.170 | 0.77 / 0.65 |

**SNR (realized − target)**
- Recorded mics: −0.19 to +0.14 dB (median +0.03).
- Ideal mics (`impair=False`): 0.000 dB.
- Partners against everything else, including transients, handling and sensor: −24.1 to +0.1 dB (median −4.2).

| SNR | Scenes | Rule gain (dB) | Rule noise at Mic A (dBA) | Partners at Mic A (dBA) |
|---|---|---|---|---|
| −10 | 10 | −4.9 to +6.4 (median +0.8) | 72.7–82.6 (76.7) | 62.8–71.9 (67.2) |
| −5 | 60 | −9.9 to +5.8 (−2.5) | 66.2–81.0 (73.2) | 60.3–76.1 (68.1) |
| 0 | 63 | −14.9 to +0.8 (−7.6) | 61.2–76.0 (68.2) | 60.3–76.1 (68.1) |
| +5 | 60 | −19.9 to −4.2 (−12.5) | 56.2–71.0 (63.2) | 60.3–76.1 (68.1) |
| +10 | 10 | −24.9 to −13.6 (−19.2) | 52.7–62.6 (56.7) | 62.8–71.9 (67.2) |

At physical levels (rule gain 0 dB) the continuous noise is about 75.8 dBA at the device (median). This matches the measured in-bus Leq of 75.3 ± 2.6 dBA [96]; the natural SNR is then about −7.6 dB.

Composition of the in-band rule noise (median share, range): engine_road 0.51 (0.28–0.73), babble_far 0.19 (0.05–0.38), passenger_near 0.11 (0.02–0.46), conv_pair 0.11 (0.03–0.45), passenger_cone 0.02 (0.00–0.10).

**Transients** (134 events, 10 capped; peaks −35 to −11 dBFS). LZFmax measured − target, uncapped events:

| Kind | n | Median (range) | 5–95% |
|---|---|---|---|
| door_hiss | 35 | −0.2 (−0.8 to +0.5) dB | −0.7 to +0.1 |
| brake_hiss | 36 | −0.3 (−1.0 to +0.4) | −0.8 to +0.3 |
| door_thump | 24 | −0.5 (−3.5 to +5.2) | −3.5 to +3.1 |
| brake_squeal | 29 | −2.1 (−7.6 to +2.1) | −6.6 to +1.8 |

**Handling** (21 geometries): inter-mic correlation −0.10 to +0.08; 99.0–99.3% of the power is below 500 Hz.

**Turn-taking**
- Utterances per partner per scene: 1–6 (median 3–4; 1 only in the 12 s smoke scenes). Back-channel stand-ins: 0–3 (median 1).
- Activity per partner 0.14–0.67 (median 0.42); double talk 0–20% of partner-active time (median 8%).
- First partner onset at 1.52–1.89 s.
- Each near passenger speaks 0.1–4.7 s (median 1.7) while both partners are silent, and about 0 s with no other ref talker.

**Peaks and cost**
- Peak \|mix\| 0.040–0.288 (median 0.134); nothing clipped.
- Render 0.30–1.34 s per 24 s scene (median 0.63; 3 workers in parallel).

### Known limitations and deviations

1. **Full-order image-source tail (E6b): RT60 accuracy.**
   - In this long, flat cabin the specular image-source decay has two slopes. T20 and T30 are 1.9–2.4 × the nominal (Sabine) RT60; EDT is 1.22 ×.
   - The reverberant energy still matches: rc_eff ≈ 1.16 × the Sabine critical distance, versus 1.19 × for the synthetic tail.
   - The absorption is therefore not recalibrated. Lowering it until T20 matched would raise the DRR by 2–4 dB and make the synthetic-versus-image-source ablation compare reverberation levels instead of tail coherence.
   - Use `meta.acoustics` for measured values. The image-source RIRs end at 1.1 × the nominal RT60 (core), so T30 is measured on a truncated decay.
2. **Synthetic tail below about 0.25 s nominal.** The order-18 image-source part, up to about 120 ms, dominates, so T20 is up to 2 × nominal and the critical distance up to 2.3 × Sabine. The effective lower end of E3 is about 0.2–0.3 s of measured T20.
3. **SNR definition.**
   - Transients and handling are sporadic events at physical levels, so they are excluded from the rule. Otherwise one 97 dB brake squeal would set the whole crowd's level.
   - As a result the SNR against "everything else" is lower: median −4.2 dB, down to −24 dB at +10 dB SNR when an event falls inside partner speech.
   - Passengers are drawn from the partners' Lombard distribution but scaled with the crowd by the rule (median −7.6 dB at 0 dB SNR). Their levels relative to each other, to the engine and to the road stay physical.
4. **Distance bins.** Side-by-side partners are 0.54–0.93 m away (20–40° subtended with at least 0.42 m between mouths); lap-flat partners are 0.45–0.99 m away. The 0.3–0.45 m bin comes from the three opposed strata (25 partners) plus 2 lap-flat partners. Only 2 dev scenes fall in the G2 cell (aligned, 0 dB, both partners ≤ 0.6 m). DESIGN's `wide` set (≥ 30 per reported cell) is not built; adding it is one entry in `SETS`.
5. **Back-channels.** TTS has no short tokens such as "mm-hmm" or "yeah", and LibriSpeech has none either. The listener's 3 shortest lines (mostly 1.1–2 s), at −8 to −4 dB, stand in. Real back-channels and interruptions need a conversational corpus (DESIGN E0, AMI).
6. **Speech material (TTS).**
   - 27 babble voices cover up to 25 distinct interferers per scene.
   - Each babble voice has only 8 lines (14–24 s), so lines repeat within a scene.
   - The TTS dev/test partner split is the generator's own: corpus2 marks TTS as `all`.
7. **Physics not modelled.**
   - Bodies, seats, the lap and the device housing: no shadowing or scattering.
   - Absorption is uniform and broadband, with no air absorption.
   - Directivity is frequency-independent (DESIGN §1.6 item 14). It acts on image sources only; the synthetic tail inherits it only through its energy match.
   - Point-source noise rolls off below 30 Hz, because the image-source model has a large near-DC gain that a real, leaky cabin does not.
8. **Transient levels.** Noise-like events land within ±1 dB of their LZFmax target. Thumps scatter by −3.5 to +5 dB (low-frequency modes). Tonal squeals sit at a median of −2 dB and down to −7.6 dB: that is the single-frequency room response, and a diffuse field gives a −1.6 dB median. 10 of 134 events hit the 103 dB SPL envelope cap and stay below their target.
9. **Passenger solo regions (B6).** The conversing pairs talk 70–90% of the time, so a near passenger is almost never the only ref talker. Speech while both partners are silent is 0.1–4.7 s per near passenger per scene (median 1.7 s). B6 should use partner-silent regions and pool over passengers or scenes.
10. **Not built: B7 probe ("bubble") scenes.** They need extra probe sources per geometry; the role `probe` is reserved.
11. **E6(a) untested on real recordings.** The hook is tested only with corpus files standing in for a recording (real noise is pending approval). A recorded field keeps the coherence of its recording pair (DEMAND: 5 cm), not the scene's spacing.
12. **Cone passenger side.** It sits on partner A's side of the axis when the cabin allows. Otherwise it goes to the other side of the same cone, which is identical in TDOA.

## Meeting scenes

Owner: `lab/scene_meeting.py`, `tests/test_scene_meeting.py`. Implements the meeting case of DESIGN §4.1 (E5, E7, E8, E9 and the set sizes) on top of the scene format above, with the per-talker structure the §4.4 metrics need.

### What is simulated

One scene per room. Every random choice is seeded from the room seed, and RIRs are seeded through `compute_rirs` (E9).

- **Room.** 12 × 7 × 3 m, each dimension scaled by 0.92–1.08. RT60 0.5–0.9 s, stratified over the rooms of a set. Broadband absorption from `pra.inverse_sabine`, randomized image sources (5 cm).
- **Table.** 8 × 2 m along the room's x axis, top at 0.72–0.76 m, centred in the room within ±0.3 / ±0.2 m.
- **Device.** At the table centre (±0.1 m), mics 1 / 2 / 10 / 20 cm above the top. The mic axis lies along the table (0° or 180°, ±3°) in half the rooms and is random in the other half (`axis_mode`: `"table"` or `"random"`). True spacing 25 / 40 / 60 mm.
- **20 talkers.** The talker ids `t00`–`t19` are shuffled, so the index carries no information.
  - Table seats: 8 per long side at 1 m pitch and 2 per end; mouths 0.35–0.5 m beyond the table edge, 1.12–1.25 m high (0.36–0.53 m above the top).
  - A second row of chairs 1.3–1.8 m behind the long edges (3–6 talkers). One of them sits in a far corner, 5.5–6.2 m from the device.
  - In half the rooms, a standing presenter (mouth 1.55–1.7 m) 5.0–5.7 m away at the far end, facing away from the device (±20°).
  - The rest sit at 13–17 of the 20 table seats.
  - Measured over dev + test (800 talkers): nearest 1.40 m; the farthest talker in each room 5.51–6.15 m (40/40 rooms ≥ 5.5 m); 135 talkers in the < 2 m bin, 424 in 2–4 m, 241 in > 4 m.
- **Directivity (E5).** Every talker is sub-cardioid (`CardioidFamily`, p = 0.75). Seated talkers face a random point on the table's long axis within ±3 m of its centre. The presenter's direct path to the device is 158–172° off-axis, so it is 5.7–6.0 dB down.
- **Table reflection (E7), first order.** `table_hook` is the `edit_images` hook. For every source it appends an explicit table image: the mirror image in the tabletop plane, with amplitude 0.9 and pyroomacoustics' directivity convention through `s2.add_image`. That image is visible at a mic only if its specular point lies on the tabletop. The hook also hides the first-order floor image (`orders_xyz == [0, 0, -1]`), per mic through `room.visibility`, when the floor-reflected path is blocked by the table. All hook parameters are in `meta.table`, so `rebuild_rirs(geom)` recomputes the stored RIRs bit for bit from the JSON alone.
- **Tails (E6b).** Half the rooms use the synthetic tail; the other half use the full-order image-source tail (`tail="ism"`). See limitation 3.
- **Conversation.** See "Conversation model" below.
  - About 3.4 min: 2 s of noise only before the first speech, and RT60 + 2 s after the last.
  - Every talker has ≥ 3 turns, plus interruptions and back-channels; 10–15 % of the speech time is overlapped.
  - 2–3 soft talkers at 52–55 dB SPL at 1 m; the others at 57–66.
- **Noise.**
  - `hvac`: diffuse, sinc coherence, A-weighted level 35 / 40 / 45 dBA at each mic. The spectrum is 0 dB at 31.5 Hz, −2 dB at 63 Hz and −5 dB at 125 Hz, then falls 4–6 dB per octave.
  - `fan`: projector fan, a point source 0.3–0.5 m below the ceiling above the table. Harmonics of 120 Hz (12 of them, amplitudes falling as 1/k^0.8–1.2) plus broadband noise within ±3 dB of the tones, 43–47 dB SPL (Z) at 1 m.
  - `sensor`: 33 dBA self-noise, added by the scene core.
- **E8 impairments.** Drawn by `s2.draw_impairments`: gain ±1 dB, high-pass corner 50–80 Hz per mic, skew 0 or 31.25 µs.

### Sets

| Set | Rooms | Seeds | Scene length | Turns per talker |
|---|---|---|---|---|
| `smoke` | 2 | 490000–490001 | 46–48 s; each talker's shortest utterances | ≥ 1 |
| `dev` | 20 | 410000–410019 | 188–213 s | ≥ 3 |
| `test` | 20 | 430000–430019 | 187–220 s | ≥ 3 |

- **Naming.** Scene ids are `<set>/meeting_rNN`. Geometries are `scenes2/<set>/meeting_<set>_rNN.geom.{json,npz}`, one scene per geometry.
- **Balanced factors**, shuffled independently in dev and in test (`set_design`; `smoke` has one room per tail):

  | Factor | Values | Rooms per value (dev / test) |
  |---|---|---|
  | `spacing_mm` | 25 / 40 / 60 | 7 / 7 / 6 and 7 / 6 / 7 |
  | `mic_height_cm` | 1 / 2 / 10 / 20 | 5 each |
  | `hvac_dba` | 35 / 40 / 45 | 7 / 6 / 7 and 6 / 7 / 7 |
  | `axis_mode` | table / random | 10 / 10 |
  | `presenter` | yes / no | 10 / 10 |
  | `tail` | synthetic / ism | 10 / 10 |

  RT60 is stratified: one room per 0.02 s stratum of 0.5–0.9 s.
- **Disk.** 50 MB for all 42 rooms: 0.37–0.42 MB per synthetic-tail geometry and 1.4–2.5 MB per full image-source geometry (`.npz`), plus 18 kB of geometry JSON and 11–20 kB of recipe per room.
- **Build time.** 90 s with 3 workers. Full image-source rooms take up to 19 s each; the slowest (RT60 0.88 s) peaked at 0.97 GB RSS.
- **Selecting scenes.** `list_scenes` filters on geometry meta and variant keys, for example `list_scenes("dev", "meeting", mic_height_cm=(1, 2), axis_mode="table")`.
  - Use `spacing_mm` rather than `spacing_m`: `spacing_m` is measured from the µm-rounded mic positions, for example 0.024999.

### Scene contents

| Item | Content |
|---|---|
| Groups | `talker_00` … `talker_19` (role `talker`, talker id `tNN`, geometry source `tNN`), then `hvac`, `fan` and `sensor`. Every talker has its own comp2 (a `Sparse`), ref and segs |
| Back-channels and second utterances | Events of the talker's own speech source, so they are in its segs and ref |
| Geometry meta (`scene["meta"]`, and filterable through `scene_card`) | `room_index`, `spacing_mm`, `mic_height_cm`, `axis_mode`, `presenter`, `n_row2`, `table {on, x, y, z, refl}`, `min_dist_m`, `max_dist_m`, `n_bin {"<2", "2-4", ">4"}`, `rc_m` (DESIGN §1.4, omni), `t30_s` and `edt_s` (room medians), `n_table_image`, `n_floor_zeroed`, `n_floor_under`, `fan_truth`, `talker_geo` |
| `talker_geo[tid]` | `seat` (`table` / `row2` / `presenter`), `dist_m`, `dist_bin`, `theta_deg`, `fold_deg`, `tau_us`, `azimuth_deg` (room frame, 0° = +x along the table), `elevation_deg`, `mouth_above_table_m`, `off_axis_deg` (device direction re the talker's facing), `dir_gain_db`, `table_image`, `floor_blocked`, `floor_under`, `drr_db`, `c50_db`, `edt_s`, `t30_s`, `drr_theory_db` |
| Measures in `talker_geo` | All at Mic A, from the stored RIRs. DRR uses a ±2.5 ms window around the direct path, which contains the table image (≤ 0.2 ms later). C50 is measured from the direct path. `drr_theory_db` = 20·log10(r_c/r) |
| Variant | `seed`, `hvac_dba`, `target_s` (180, or 40 for smoke), `min_turns` (3, or 1 for smoke) |
| `meta.recipe_meta` | `speakers {tid: speaker}`, `level_db {tid}`, `soft [tids]`, `hvac {dba, slope_db_oct}`, `fan {level_db, f0}`, `turns [[tid, kind, start_s, end_s, [utt ids]]]` (dry times; `kind` is `turn`, `interrupt` or `backchannel`), `n_turns {tid}` (main turns), `n_interrupts`, `n_backchannels`, `overlap_ratio`, `text_repeats` |

### Conversation model (`make_recipe`)

**Random streams**
- `rn` draws everything corpus-independent: talker levels, the HVAC slope, fan parameters, impairments and the render seed.
- `rs` draws the speakers (20 distinct ones from `speakers_for("talker", split)`; split `"test"` for the test set, `"dev"` otherwise) and the utterances.
- `rt` draws the structure: turn order, gaps, interruptions and back-channel placement.

**Turns**
- Each talker gets `min_turns` main turns. Extra turns go to talkers with random activity weights, but only while the expected length is below the target; no extra turns are drawn for TTS.
- Order: the next talker is drawn with probability proportional to its remaining turns, never the same talker twice in a row.
- Gaps between turns are 0.1 s + Exp(mean 0.35 s), capped at 2 s.
- 15 % of turn changes are interruptions: they start 0.3–1.2 s before the previous turn ends, never over a third talker.

**Utterances**
- A turn is one utterance of 1–6 s. In 20 % of turns whose first utterance is shorter than 2.5 s, a second one follows after a 0.15–0.45 s pause.
- Choice: never repeated within a talker; then the least-used text in the scene; then TTS style `meeting`.
- Level jitter is ±1 dB per turn.

**Back-channels**
- One per 28 s of conversation.
- A back-channel is one of the talker's 4 shortest utterances at −2 to −6 dB, placed inside a stretch of another talker's turn where nobody else talks.
- TTS has no back-channel tokens, so these are short whole sentences (1.3–2.5 s).

`P_INTERRUPT` and `BC_EVERY_S` were set on 20 seeds outside every set (470000+). The result was 11.8 % overlap with TTS and 11.6 % with a scratch LibriSpeech test-clean manifest.

### Validation (dev / test, TTS, `--report`)

| Check | dev | test |
|---|---|---|
| T30 / RT60, synthetic tail (200 talker RIRs) | 1.01 [0.99, 1.05] | 1.01 [0.98, 1.05] |
| T30 / RT60, full image-source tail | 1.61 [1.39, 1.78] | 1.63 [1.45, 1.80] |
| EDT / RT60, full image-source tail | 1.06 [0.74, 1.23] | 1.09 [0.73, 1.22] |
| Farthest talker per room | 5.90 [5.52, 6.15] m, 20/20 rooms ≥ 5.5 m | 5.81 [5.51, 6.11] m, 20/20 |
| DRR, < 2 / 2–4 / > 4 m (median) | +3.5 / −1.3 / −5.2 dB | +3.7 / −1.3 / −5.3 dB |
| DRR theory, omni, no table (DESIGN §1.4) | −3.1 / −8.7 / −12.8 dB | −3.1 / −8.6 / −12.6 dB |
| DRR minus theory, mic 1 / 2 / 10 / 20 cm | +8.8 / +8.6 / +5.5 / +5.6 dB | +8.7 / +8.9 / +5.6 / +5.8 dB |
| C50, < 2 / 2–4 / > 4 m | 8.6 / 5.4 / 4.0 dB | 8.5 / 5.2 / 3.6 dB |
| Talker level at Mic A (own non-overlapped segs), < 2 / 2–4 / > 4 m | 61.5 / 59.5 / 56.9 dB SPL | 61.8 / 58.9 / 56.0 dB SPL |
| Talker level spread per scene (max − min) | 15.7 [11.6, 21.0] dB | 16.0 [12.0, 19.5] dB |
| In-band SNR per talker (150–7000 Hz, vs hvac + fan + sensor), < 2 / 2–4 / > 4 m | 13.1 / 10.9 / 8.5 dB (minimum −5.6) | 13.7 / 10.7 / 7.8 dB (minimum −4.4) |
| HVAC dBA at both mics minus target | 0.00 dB | 0.00 dB |
| Fan at Mic A | 47.0 dB(Z), 42.8 dBA | 46.8 dB(Z), 42.0 dBA |
| Noise floor at Mic A (hvac + fan + sensor) | 45.5 [41.7, 48.6] dBA | 44.5 [42.7, 48.2] dBA |
| Duration; overlap (from segs); interruptions; back-channels | 205 s; 11 % [10, 15]; 8; 7 | 202 s; 11 % [8, 14]; 8; 7 |
| Noise-only lead / tail (minimum over scenes) | 2.01 / 2.51 s | 2.01 / 2.53 s |
| Peak | 0.014 | 0.013 |
| Render time per scene (1 process) | 1.2 [0.9, 1.8] s | 1.0 [0.9, 1.7] s |

**Determinism (E9).** Checked with the scratch script `tmp/harness_scenes-meeting/check_repro.py`:
- `rebuild_rirs` reproduces the stored RIRs of all 42 rooms bit for bit;
- a fresh in-process build of dev rooms r00 (full image-source tail) and r01 (synthetic) gives identical geometry JSON, RIR store and recipe to the worker-built set;
- re-rendering a scene is bit-identical, and Σ comp2 == mix.

**B.10, table comb notch** (`--b10`, `notch_check`; mouth 0.42 m above the top):

| Talker distance | Mics 1 cm | Mics 2 cm | Mics 10 cm | Mics 20 cm |
|---|---|---|---|---|
| 3 m | no notch below 7.9 kHz (predicted 62 kHz) | none (predicted 31 kHz) | **6132 Hz** (predicted 6188; DESIGN 6.2 kHz ± 5 %) | 3112 Hz (predicted 3099) |
| 1.5 m | none (predicted 32 kHz) | none (predicted 16 kHz) | 3183 Hz (predicted 3187) | 1606 Hz (predicted 1602) |

These reproduce DESIGN §5 (3.2 / 1.6 kHz at 1.5 m; above 15 kHz within 2 cm). `test_floor_image_zeroed_and_table_image_added` checks that the hook adds exactly the table image (delay ±1 sample, energy within 5 % of 0.9²·(r_d/r_t)²) and removes exactly the floor image, with nothing else changed.

**Corpus switch (`corpus="libri"`, `corpora/libri/manifest.json` from `lab/corpus_libri.py`).** `--report` was run on temporary copies of all 40 dev and test rooms, so no libri recipe was written to `scenes2/`.
- Every dev room drew 20 `dev` speakers and every test room 20 `test` speakers.
- Durations 246–282 s (dev median 264, test 270); overlap 12 / 13 % (range 9–16 %); ≥ 3 turns per talker; 7–10 back-channels; 0–1 repeated texts per scene.
- Acoustic truth is unchanged. Talker levels at Mic A are 0.5–1 dB lower than with TTS, and the level spread per scene is 15.9 / 15.4 dB.
- Render time 1.4–1.6 s median.
- `load_scene(sid, corpus="libri")` writes `<name>.libri.recipe.json` next to the TTS recipe on first use.

### CLI and API

```
.venv/bin/python -m lab.scene_meeting --build smoke dev test -j 3   # about 90 s, 50 MB in scenes2/
.venv/bin/python -m lab.scene_meeting --report dev [--corpus tts]   # validation table, about 2.5 min per 20 scenes
.venv/bin/python -m lab.scene_meeting --b10                         # table comb-notch check
.venv/bin/python -m tests.test_scene_meeting                        # 11 tests, about 14 s
```

`lab.scene_meeting`:
- `set_design(set)` gives the room designs;
- `room_layout(design)` gives positions only, no RIRs;
- `build_room(design, root=None, corpus="tts")` writes a room and returns its scene id; `build_set(set, workers=3)` builds a whole set;
- `make_recipe(geom, corpus, seed, hvac_dba, target_s, min_turns)` is the generator;
- `table_hook(table, refl=0.9, add_table=True, zero_floor=True)` and `table_paths(src, mic, table)` implement E7;
- `rebuild_rirs(geom)`, `notch_check(dist, mouth, mic)`, `acoustic_truth(geom, rirs)`, `scene_stats(sid)` and `report(set)` serve validation;
- constants: `SETS`, `TABLE_REFL`, `TALKER_P`, `P_TWO_UTTS`, `P_INTERRUPT`, `BC_EVERY_S`.

### Limitations and deviations from DESIGN

1. **Floor-image rule.**
   - DESIGN E7 says to zero the first-order floor image "whose specular point lies under the table". We zero it when either leg of the floor-reflected path crosses the tabletop.
   - This includes every DESIGN case. It also covers the case the DESIGN rule misses: a mic above the table centre sees the floor only past the table edge, so every first-order floor path to it is blocked.
   - The DESIGN rule alone would have kept a physically blocked floor reflection for 13 % of the talkers: dev 348 / 400 and test 347 / 400 have the specular point under the table; the rest are mostly in the second row.
   - Floor zeroing uses per-mic `visibility` instead of `damping`, so a decision that differs between the two mics stays exact.
2. **First-order table model only.** Higher-order table interactions are not modelled: a ceiling or wall reflection that then bounces off the table is missing, and the table does not shadow higher-order images that arrive from below. The reverberant field is therefore not boundary-enhanced.
   - At 1–2 cm, the coherent table image adds +5.6 dB of direct energy (measured DRR − theory: +8.7 dB at 1–2 cm, +5.6 dB at 10–20 cm).
   - A real boundary mic also gains about +2.6 dB of diffuse energy. Treat the simulated DRR at 1–2 cm as about 2.5–3 dB optimistic relative to 10–20 cm when reading the mic-height ablation (DESIGN §4.8).
3. **Full image-source tail decays slower than RT60.**
   - In these flat rooms, purely specular image sources with uniform absorption give T30 ≈ 1.6 × RT60, with EDT ≈ RT60. The late field is a non-diffuse horizontal field: a double slope.
   - The synthetic tail matches RT60 exactly.
   - Calibrating the absorption so that T30 = RT60 was tried on 3 dev layouts and rejected. Relative to the synthetic tail it raised DRR by 2.0–3.3 dB and C50 by 3.8–4.5 dB, and EDT fell to 0.2–0.7 × RT60.
   - With nominal absorption the two tails agree within 0.3 dB in DRR and 0.7 dB in C50; the full image-source EDT is about 10 % longer on median (up to 22 %). E6b therefore compares tail structure, not reverberant energy.
   - `rt60` is the design (Sabine) value for both tails. Use `meta.t30_s` / `edt_s` or the `tail` card field when the decay matters.
4. **DRR is 5–9 dB above DESIGN §1.4.** The DESIGN table is for omni talkers without a table. Talker directivity (+2.3 dB) and the table image (+2.6 to +5.6 dB) explain most of the difference; the hidden floor image and the 5 ms direct window add the rest.
   - The > 4 m bin has a median of −5.2 dB, not the −13 to −17 dB DESIGN §0 quotes for 6 m.
   - The presenter, facing away, is the hard case: DRR median −11.2 / −11.6 dB (dev / test), range −15.0 to −9.6 dB, at 5.0–5.7 m.
5. **Length.** Scenes are 187–220 s with TTS, not 3 min: 60 main turns of 2–5 s sentences set the floor. With `libri` they are 246–282 s, because utterances of 1–6 s are drawn for turns. (DESIGN E7 asked for 5–10 min; this task set about 3 min.)
6. **Set size.** 20 rooms per set (this task) instead of ≥ 30 (DESIGN §4.5), and no `wide` set.
7. **TTS limits.**
   - 12 of the 20 talker voices are accented, with only 8 short sentences each. The talker with the least speech has 5.4–6.9 s per scene (median talker about 10 s), so DNSMOS's ≥ 9 s per talker (T9) is not always reached.
   - 12–20 texts per scene (median 16) repeat across talkers, never within a talker, because the voices share sentence lists.
   - Some TTS voices put about 90 % of their power in 150–400 Hz around a steady f0. When a low-frequency comb notch of the early RIR hits f0, their early-image level can sit up to about 7 dB below the white-noise RIR gain (seen for a presenter). This is physically consistent, but stronger than it would be for natural speech.
8. **Fan level.**
   - At 43–47 dB SPL (Z) at 1 m, the reverberant field puts the fan at about 47 dB(Z) / 42 dBA at the mics: louder than HVAC in the 35 and 40 dBA rooms.
   - The noise floor at Mic A is therefore 42–49 dBA, and the HVAC factor is partly masked.
   - Lower `fan` `level_db` in `make_recipe` if the HVAC factor should dominate. Recipes would then need regenerating; geometries would not.
9. **Not modelled.**
   - Talker directivity is frequency-independent (DESIGN §1.6 item 14) and the mics are omni.
   - No bodies, chairs or laptops; no air absorption.
   - Smoke scenes have 1 turn per talker, so they exercise no returning talkers.
10. **Corpus requirements.** A drop-in corpus needs ≥ 20 `talker` speakers in each of the `dev` and `test` splits; `make_recipe` raises `ValueError` otherwise. LibriSpeech test-clean has 40 speakers, so a 20/20 split leaves no spare.

## Metrics

Owner: `lab/metrics2.py`, `lab/stats.py`, `tests/test_metrics2.py`. Implements DESIGN §4.2–4.6 and Appendix B.5, B.6 and B.9. Every definition below is what the code computes. "Resolved ambiguities and deviations" says where it departs from the DESIGN wording and why.

### Hagerman runs and separated components

The method runs once per Hagerman run. Each run's input is `scene2.signed_mix` with some groups negated, and the outputs are decoded into separated components for both stages (below).

**Bus.** A = `partner_A`, B = `partner_B`, N = every other group, `sensor` included.

| Run | Input | Scenes |
|---|---|---|
| `y1` | A + B + N (the mix itself) | all |
| `y2`, `y3`, `y4` | A − B − N, −A + B − N, −A − B + N | all |
| `near` | mix with every `passenger_near` group negated | all with near passengers |
| `probe` | mix with the `probe` group negated | probe scenes |
| `cone`, `conv`, `far`, `engine` | mix with `passenger_cone`, `conv_pair`, `babble_far` or `engine_road` negated | extra subset |
| `cons` | A + B − N | extra subset |

Decoders:
- A = (y1 + y2 − y3 − y4)/4, B = (y1 − y2 + y3 − y4)/4, N = (y1 − y2 − y3 + y4)/4;
- for the 2-run flip of a group g, g = (y1 − y_g)/2;
- rest = N − near.

**Meeting.** S = every `talker` group, N = everything else. y1 = S + N and y2 = S − N, so S = (y1 + y2)/2 and N = (y1 − y2)/2. The extra subset adds `cons` = −S + N.

**Extra subset.** With `run --extra subset` (the default), the extra runs go to one geometry in five, at least one, per scenario and E1 stratum. Geometries are picked by CRC32 of their id, which gives 9 of the 50 dev geometries (19 of the 110 dev scenes). `--extra all` and `--extra none` override this. Runs per scene:
- bus: 5, or 6 with a probe; 10 with the extras;
- meeting: 2; 3 with the extras.

On the extra subset the bench also repeats `y1` after all other runs (one more `enhance` call, not timed) and reports `rerun_db` = 10·log10(‖y1‖² / ‖y1 − y1_repeat‖²), capped at 150. Anything below 150 means the method's output depends on earlier calls (state kept between calls, unseeded randomness). That breaks Hagerman decoding, and a method that remembers earlier runs' inputs could decode the scene's components itself; methods must be stateless between `enhance` calls (DESIGN B.8).

**Stages.** `pre` is the method's `pre_agc` output when it returns `{"out", "pre_agc"}` for `return_stages=True`; otherwise `pre` is `out`. Every component is decoded for both stages.

**Consistency (nonlinearity) figures**, on the final output, capped at 150 dB:
- `consistency_db` (extra subset; DESIGN §4.2) compares the target-flip and noise-flip estimates of the target: 10·log10(‖S_nf‖² / ‖S_nf − S_tf‖²).
  - Bus: S_nf = (y1 + y_cons)/2 and S_tf = (y1 − y4)/2, which equals the decoded A + B.
  - Meeting: S = (y1 + y2)/2 is compared with S_tf = (y1 − y_cons)/2.
- `closure_db` (bus, every scene, no extra run): 10·log10(‖y1‖² / ‖(y1 + y2 + y3 + y4)/4‖²). The closure term is zero for a linear method; for F(x) = x³ it is exactly 6·A·B·N (tested).
- Limits:
  - Any odd-symmetric method (F(−x) = −F(x), which includes every gain or mask computed from magnitudes) scores `consistency_db` = 150 by construction.
  - The closure only holds three-way A·B·N products, which are rare because the partners seldom overlap.
  - Measured on the test bus scene (consistency / closure): x + c·x² gives 3.1 / 13.7 dB, a memoryless odd mask 150 / 150, a linear FIR 150 / 150.
  - A high figure therefore rules out only non-odd and three-way nonlinearity.

**Delay and alignment (B10, B.6)**
- `delay` is the lag in samples (> 0: output late) that maximizes the sum over talkers of the squared normalized cross-correlation between a decoded target component (final output) and the talker's early image:
  - bus: A and B, each over its partner's speech;
  - meeting: S, over each talker's alone samples;
  - at most 30 s per talker, search range ±100 ms.
- Both signals are band-passed to 1–4 kHz first. The S0 high-pass leads by up to about 20 samples near 150 Hz, and with full-band correlation M0 measured −2 to −4 samples on the smoke scenes. At 1–4 kHz the filter's phase delay is under 0.5 samples, and M0 measures 0.
- `delay_ok` = 1 when |delay| ≤ 2. Beyond that, every run output is advanced by `delay` before scoring and `aligned` is true. This is a safety net, not a licence.
- `delay_found` is false, and the delay 0, when no talker correlates above 0.1.

### Common definitions

- **Level L:** mean square in the scene SNR band, 150–7000 Hz (the SNR rule's 4th-order Butterworth band-pass, zero-phase), in dB at digital scale.
- **Level change ΔL:** L of an output component minus L of the same groups' comp2 at Mic A, over the same samples. It is clipped to ±60 dB and NaN where the input has no power.
- **Activity** comes from `segs`; "k active" is the union of k's segments.
  - Bus: partner-active means A or B is active.
  - Meeting: speech-active means any talker is active.
- **Alone** (meeting): k is the only talker with segments that is active.
- **Partners silent** (bus passengers): j is active and both partners are inactive; other passengers may talk.
- **Idle:** samples at least 0.1 s before and 0.3 s after any partner (bus) or talker (meeting) activity, from 0.5 s into the scene on. NA_idle and NM need at least 1 s of idle samples, else they are NaN.
- **Windows:** non-overlapping 1 s windows. A talker counts in a window where it is active for at least 0.25 s.
- **SDR** (dB, capped at 150) is mir_eval's zero-padded BSS-Eval projection with a non-causal lag range. Polarity, small delays and spectral tilt are not penalized.
  - The output and the reference are band-limited as above, with 0.1 s of context around each interval.
  - Each scored interval is followed by 512 zeros.
  - The output is projected by least squares onto the reference through a 512-tap FIR at lags −128…+383 samples (8 ms lead, 24 ms lag).
  - SDR = ‖projection‖² / ‖output − projection‖².
- **ESTOI:** `pystoi.stoi(ref, y, 16000, extended=True)` on the concatenation of the scored intervals.
- SDR and ESTOI are NaN with less than 1 s of audio. ESTOI is also NaN when pystoi keeps fewer than 30 frames.
- `*_in` values are the same measures on raw Mic A (`mix[0]`).

### Bus metrics (DESIGN §4.3)

All values are per scene, in `m` of the result file.

| ID | Keys | Definition |
|---|---|---|
| B1 | `sdr_A`, `sdr_B`, `estoi_A`, `estoi_B` (and `*_in_*`) | y1 against the partner's early image, over the samples where that partner talks and the other partner is silent (all of its speech if that is under 1 s). Passengers may talk there |
| B1 | `d_estoi_min` (**worst-partner ΔESTOI**), `d_sdr_min` | Minimum over the two partners of ESTOI_k − ESTOI_in,k; the same for SDR |
| B1 | `estoi_wrst`, `estoi_wrst_in`, `d_estoi_wrst` | min_k ESTOI_k, min_k ESTOI_in,k and their difference (report only) |
| B2 | `ta_A`, `ta_B`, `ta_min`, `ta_mean` | ΔL of the separated partner component, pre stage, over that partner's speech. `ta_out_*` is the final stage |
| B2 | `balance_med`, `balance_p90` | For each 1 s window where partner k is active: \|TA_k(window) − the median window TA of the other partner\|. Median and 90th percentile over all such window–partner pairs. `balance` = \|TA_A − TA_B\| |
| B3 | `na_active`, `na_idle` | −ΔL of the separated N, pre stage, over partner-active and idle samples |
| B3 | `dsnr_active` | `na_active` + `ta_mean` |
| B3 | `nm` | NA_idle − NA_active on the final output, so an AGC's off-attenuation counts. `nm_pre` is the pre-stage value |
| B4 | `dsir_near`, `dsir_rest` (all scenes); `dsir_cone`, `dsir_conv`, `dsir_far`, `dsir_engine` (extra subset) | `ta_mean` − ΔL of the separated group (pre stage) over partner-active samples. `rest` = N − near |
| B5 | `rda_mean`, `rda_p10`, `rda_n` | Final output. For each 1 s window with at least 0.25 s of near-passenger speech: TA_p − ΔL_near. ΔL_near is over the near passengers' speech in the window. TA_p is the mean TA of the partners active in the window, or the scene's final-stage mean partner TA when neither is. Reported: mean, 10th percentile, number of windows |
| B6 | `pass_estoi_med`, `pass_estoi_in_med`, `pass_drop_med` | ESTOI of y1 (and of raw Mic A) against passenger j's early image, over j's speech with both partners silent. Taken for the 3 `passenger_near` talkers nearest the array center; the median is over those that have a value, and the drop is the median of in − out. `conv_*` gives the same for the conversing-pair talkers (report only) |
| B7 | `probe_rel_db`, `probe_dist_m` | Probe scenes: ΔL of the separated probe (final output) over the probe's speech, minus the final-stage mean partner TA. `table` lists the mean per probe distance and the bubble, the value at 0.6 m minus the value at 2 m |
| B8, B9 | `pending` | The hooks `metrics2.score_wer` and `metrics2.score_dnsmos` return None until the models are approved. Every result file carries a `pending` note |
| B10 | `rtf`, `rtf_y1`, `rtf_max`, `delay`, `delay_ok`, `algo_latency_ms` | RTF = wall time of one `enhance` call / audio duration; `rtf` is the median over the scene's runs. `algo_latency_ms` is the method module's optional `ALGO_LATENCY_MS` |
| — | `closure_db`, `consistency_db`, `rerun_db`, `peak`, `clipped` | Hagerman figures and the repeated-run check (above); the peak of y1 and whether it reaches 1.0 |

**Composite: two-talker focus** (`focus_pass`, criteria `focus_c1` to `focus_c4`). A scene passes when all four hold:
1. `d_estoi_min` ≥ 0;
2. `pass_estoi_med` ≤ 0.25;
3. `rda_p10` ≥ 10 dB;
4. `ta_min` ≥ −3 dB.

The composite is NaN when any criterion is NaN, for example when no near passenger has 1 s of speech with the partners silent. `table` shows `focus%`: the percentage of passing scenes among those with a defined composite (mean over geometries).

### Meeting metrics (DESIGN §4.4)

Every per-talker value uses the talker's alone samples. Talkers with less than 1 s alone are skipped and counted in `n_talkers_skipped`. S_in is the sum of all talker groups at Mic A.

| ID | Keys | Definition |
|---|---|---|
| T7 | `talkers.<id>.estoi`, `talkers.<id>.d_estoi`, `estoi_mean`, `d_estoi_mean`, `d_estoi_min` | ESTOI of y1 against the talker's early image, and ΔESTOI against raw Mic A |
| T7 | `estoi_wrst`, `estoi_wrst_in`, `d_estoi_wrst` (**worst-talker ΔESTOI**) | Minimum over talkers of ESTOI_out and of ESTOI_in, and min_out − min_in |
| T7 | `d_estoi_near`, `d_estoi_mid`, `d_estoi_far` | Mean ΔESTOI per distance bin from the array center: < 2, 2–4, ≥ 4 m |
| T7 | `frac_estoi_ok` | Fraction of talkers with ΔESTOI ≥ −0.03 |
| T2 | `talkers.<id>.ta`, `ta_med_near`, `ta_med_mid`, `ta_med_far`, `ta_wrst` | ΔL of the separated speech S, pre stage, over the talker's alone samples; the median per bin; the minimum |
| T2 | `t2_margin`, `t2_ok` | Median TA of the farthest non-empty bin minus that of the nearest non-empty bin (normally far − near; NaN with only one bin). OK if ≥ −3 dB |
| T3 | `talkers.<id>.level_out_db`, `lvl_std`, `lvl_range` | Active speech level of the in-band separated speech S (final output) over the talker's alone samples, by `lab.common.active_rms` (20 ms frames within 30 dB of the loudest). Std and range across talkers |
| T3 | `lvl_soft_rel`, `turn_lvl_std` | Mean level relative to the median for the E7 soft talkers (≤ 55.5 dB SPL at 1 m). Mean, over talkers with at least 2 turns, of the std of their per-turn active levels |
| T1 | `capture_rate` | Fraction of talkers with ΔESTOI ≥ −0.03 and \|L_k − median\| ≤ 3 dB. The WER condition is pending |
| T4 | `na_active`, `na_idle`, `nm`, `nm_pre`, `dsnr_active` | As B3, with speech-active and idle defined by all talkers. `dsnr_active` = `na_active` + the mean talker TA |
| T4 | `pump_std`, `pump_step_max` | Per turn, ΔL of the separated N (final output) over the turn. Std over turns, and the largest step between consecutive turns |
| T5 | `settle_med`, `settle_p90`, `settle_first_med`, `settle_return_med` | Turns of at least 2 s. Gain = 10·log10 of the energy of S (final output) over that of S_in, in 400 ms windows every 50 ms; windows more than 20 dB below the turn's loudest input window are ignored. The final gain is the median over windows that start in the turn's last second. Settling time = start of the first window after which every window stays within ±3 dB of the final gain (0 if all do). Median and P90 over turns; medians for first turns and for returning talkers |
| T6, T9 | `pending` | As B8 and B9 |
| T8 | — | Not implemented (optional; it needs frozen linear stages) |

A turn is a run of one talker's segments less than 1 s apart, with no other talker's segment starting in between. A turn is "returning" if its talker spoke before.

**Composite: even capture** (`even_pass`, criteria `even_c1` to `even_c3`): `capture_rate` = 1, `d_estoi_wrst` ≥ 0 and `t2_ok`.

In both composites, ESTOI ties within 1e-6 count as "≥". The ESTOI of a scaled copy differs by about 1e-10.

### Legacy columns (reported, never used for decisions)

- Bus, in `legacy`: v1 `lab.metrics.score_bus` on y1 against the summed partner early images.
  - `dsisdr_sum`: SI-SDR improvement.
  - `br`: v1 background reduction, capped at 40 dB.
  - `pesq`, `pesq_in`: PESQ-WB.
  - `estoi_sum`, `estoi_sum_in`: ESTOI against the summed reference.
- Meeting, in `legacy`, per talker over the alone samples:
  - `loud_std`: std of the full-band active level of y1;
  - `gap_snr`: median talker level minus the y1 level over idle samples;
  - `dsisdr`: mean SI-SDR improvement over raw Mic A.

### Statistics (`lab/stats.py`, DESIGN §4.5)

- The unit of analysis is the geometry: a metric is first averaged over the scenes of one geometry (its SNRs), skipping NaN scenes.
- `bootstrap_ci`: 10,000 resamples of the geometries, percentile 95% CI of the mean. On the bus, resampling is within E1 strata (counts kept), unless a stratum has a single geometry.
- `paired(rows, ref_rows, key)`: per-geometry differences (method − reference) over the geometries both have. It returns the mean, the bootstrap CI and the two-sided Wilcoxon signed-rank p (1.0 when all differences are 0).
- `compare` Holm-corrects the p-values across methods, separately for each metric.

### Gates (`lab/stats.py`, DESIGN §4.6)

Gates are computed on the Δ metrics, which already compare with raw Mic A. `compare` gives the paired differences against `m0_mica_hpf`.

| Kind | Passes when |
|---|---|
| `improve` (Δ ≥ x) | point estimate ≥ x and CI lower bound > 0 (§4.5) |
| `noreg` (Δ ≥ −x) | CI lower bound ≥ −x (§4.5) |
| `cap` (value ≤ x) | CI upper bound ≤ x |
| `floor` (value ≥ x) | CI lower bound ≥ x |
| `never` | no scene has an error (non-finite or wrong-length output included) and, on the bus, no scene clips |
| `rtf` | 90th percentile over scenes of `rtf` < x. A method with `OFFLINE_ONLY = True` (Tier 3) gets its RTF reported, with no verdict |

| Gate | Bus | Meeting |
|---|---|---|
| G0 | `d_estoi_min` noreg 0.01; `ta_min` noreg 3; `balance_p90` cap 3; `nm` cap 6; never clipping, NaN or errors; RTF < 0.5 | `d_estoi_wrst` noreg 0.01; `frac_estoi_ok` floor 0.95 (19 of 20 talkers); `t2_margin` noreg 3; `nm` cap 6; never NaN or errors; RTF < 0.5 |
| G1 | `d_estoi_min` improve 0.03, and 0.05 in the aligned stratum; `dsnr_active` improve 3 (aligned) and 1.5 (all strata) | `lvl_std` cap 1.5; `lvl_range` cap 6; `d_estoi_mean` improve 0.02; `d_estoi_far` improve 0.03; WER pending; `na_active` improve 3 |

- G0 uses every scene ("always"). G1 leaves out the bus stress SNR (−10 dB or lower).
- Aligned-stratum criteria resample within that stratum.
- A CI criterion needs at least 5 geometries for a verdict; otherwise it shows "n/a".
- Verdicts:
  - FAIL if any criterion fails;
  - PASS if all pass;
  - "PASS* k/n" if only k of the n criteria could be evaluated; meeting G1 is at best PASS* 5/6 until WER exists;
  - "n/a" if none could.
- The DESIGN decides gates on natural speech (E0). On TTS scenes the verdicts are indicative only.
- G2 is not automated. `table --snr 0 --strata aligned`, and the scene-card field `partner_dist_max_m` with `list_scenes`, select its cell.

### Resolved ambiguities and deviations

1. **B1 and T7 on y1, not on a separated component** (orchestrator guidance). ESTOI or SDR of a separated speech component ignores the noise left in the output, so a method that does nothing would score as well as one that removes noise.
2. **Worst partner = minimum of the per-partner ΔESTOI.** The literal "min out ≥ min in" lets a method trade one partner against the other; the per-partner minimum implies it. The meeting keeps the literal min_out − min_in, because T1 and G0 already check every talker.
3. **SDR in band (150–7000 Hz).** Full-band SDR is dominated by engine energy below 150 Hz. On the demo scene the 120 Hz high-pass alone raised full-band SDR of partner A by 8.9 dB (−10.4 → −1.6 dB); in band, M0 changes SDR by 0.0 dB.
4. **B6 regions: partners silent, not "alone".** On 12 dev bus scenes at 0 dB, 0 of 36 nearest-passenger instances had 1 s of speech with no other reference talker active. 24 of 36 (67%) had 1 s with both partners silent. The bus-scene section reports the same (its item 9). Other passengers count as background there, like the engine.
5. **Near-passenger flip on every bus scene**, one run more than the DESIGN's 4. B5 RDA is a criterion of the focus composite, so it needs the nearest-interferer component on every scene. The DESIGN's other finer splits and the consistency run stay on the 20% subset.
6. **Stage per metric.**
   - Pre-AGC stage: TA, NA, ΔSNR, ΔSIR and T2 (the DESIGN's "level change before AGC").
   - Final output: NM, RDA, the probe, T3, T4 pumping and T5, where an AGC's gating, off-attenuation and pumping are audible.
7. **Balance.** Partners mostly alternate, so a 1 s window rarely holds both. Each window's active-partner TA is compared with the other partner's median window TA.
8. **RDA partner term.** The partners' TA in the same window when they talk, else their scene-level TA, so windows where only passengers talk count too.
9. **Idle guards** (0.1 s before, 0.3 s after, first 0.5 s skipped, at least 1 s needed). They keep the DESIGN's "partner pauses" to steady-state pauses, without onset and hangover frames.
10. **T3 uses the active speech level**, not the mean square, because utterances with more pauses would otherwise read quieter.
11. **T2 bins.** The farthest and nearest non-empty bins stand in for "far" and "near" when a room has no talker beyond 4 m or within 2 m.
12. **"19 of 20 talkers"** becomes the fraction of scored talkers with ΔESTOI ≥ −0.03, with a floor of 0.95, so that scenes with skipped talkers are handled.
13. **Cap and floor gates** apply the §4.5 no-regression rule (a CI bound) to absolute limits. The RTF gate uses the 90th percentile of per-scene RTF, so that single timing outliers on a loaded machine do not decide it.
14. **Result files carry the corpus** (`<name>.<corpus>.json`), because the same scene is scored on TTS now and on natural speech later.
15. **Not tested on real data:** B7 (no probe scenes exist yet), WER and DNSMOS (models not downloaded).

### Tests

`.venv/bin/python -m tests.test_metrics2` runs 18 tests in about 25–40 s (machine load). Two scenes are built with the scene core in a temporary root:
- a 20 s bus scene: partners, 3 near passengers, a cone passenger, a conversing pair, far babble, engine and road noise at 0 dB SNR, with skew on;
- a 44 s meeting: 6 talkers at 1.0–4.8 m (one soft), 8 turns with two returning talkers, HVAC at 40 dBA and a fan.

| Check | Requirement | Measured |
|---|---|---|
| B.5: Σ comp2 = mix | ≤ 1e-6 | passes on both scenes |
| B.5: linear method (fixed FIR on both mics), decoded components against the method applied to each component | ≥ 60 dB | ≥ 133 dB for every component (A, B, N, near, rest, cone, conv, far, engine, S) |
| B.5: consistency and closure of the linear method | ≥ 60 dB | 150 dB (the cap) |
| Non-odd method x + c·x²; odd mask | consistency < 40 dB; = 150 dB | 3.1 dB; 150 dB |
| Closure algebra, F(x) = x³ | = 6·A·B·N | exact |
| B.9: partner B inverted in an ideal output (early images; full partner components) | ΔSDR < 0.1 dB, ΔESTOI < 0.005 | 0 dB and 2e-16; 3e-10 dB and 7e-11 |
| B.9: oracle-VAD gate on Mic A | ΔSNR_active within ±0.5 dB, NM ≥ 20 dB, G0 fails | −0.005 dB; 60 dB (the cap); G0 fails on NM only |
| B6: passenger suppressed by 90% in y1; everything scaled by 0.1 | ESTOI falls by more than 0.1; unchanged | passes |
| SDR of an inverted, delayed, tilted copy; of the reference plus noise at 10 dB | > 35 dB; 10 ± 0.5 dB | 82.7 dB; 10.04 dB |
| Delay of 1 and 40 samples | measured exactly; shifted only for 40 | 1 (no shift); 40 (shifted, TA 0.000 dB) |
| B.6: M0 impulse | peak at lag 0 ± 2 samples; Butterworth magnitude within 0.1 dB | lag 0 |

The tests also cover:
- the bench end to end: M0 on the bus with extras (10 runs) and on the meeting;
- an oracle leveler: T3 std < 0.5 dB, capture 100%, settling 0 s;
- error recording, and the CLI run → table → compare path;
- the oracle side channel: `info["oracle"]` reaches a method named `oracle*` and no other; `oracle2_irm` decodes exactly (consistency, closure and `rerun_db` ≥ 100 dB) and refuses to run without the side channel;
- the code policy (imports of `lab.scene2` / `lab.metrics2` and a `"scenes2/…"` literal are flagged, a docstring mentioning `scenes2/` is not) and `rerun_db` (< 60 dB for a method that keeps state between calls, 150 dB for a stateless one);
- statistics: stratified bootstrap (narrower than unstratified; singleton fallback), Holm, Wilcoxon, the gate rules and paired comparison.

## Bench usage

Owner: `lab/bench2.py`, `methods/m0_mica_hpf.py`.

```
.venv/bin/python -m lab.bench2 run --methods m0_mica_hpf[,other] --set dev [--scenario bus|meeting]
        [--snr -5,0] [--strata aligned,45] [--limit N] [--corpus tts] [-j 3] [--extra subset|all|none]
        [--save-audio] [--skip-existing]
.venv/bin/python -m lab.bench2 table --set dev [--methods a,b] [--scenario bus] [--by stratum,snr]
        [--snr 0] [--strata aligned]
.venv/bin/python -m lab.bench2 compare --set dev [--methods a,b] [--ref m0_mica_hpf] [--metrics d_estoi_min,nm]
```

**`run`**
- Each worker job renders one scene once and runs every requested method on it, so the methods share the render and the input-side metrics.
- Jobs run longest scene first, with at most `-j` workers (default 3, the team limit).
- It prints one line per scene and method, then the `table` of the methods it ran.
- A failing scene does not stop the run: the error and traceback go into its result file (`status: "error"`).
- `--skip-existing` skips scene–method pairs that already have an `ok` result. `--limit N` takes the first N matching scenes.

**`table`** prints, per scenario:
- the decision columns, then the legacy columns (prefix `L:`);
- optional breakdowns;
- the B7 probe table, when probe scenes exist;
- the G0 and G1 verdicts, with each criterion's estimate and CI.

Values are means over geometries of per-geometry means. `--by` keys:
- `stratum`, `spacing`, `snr`, `tail`, `corpus`;
- `dist_bin`: E2 bin of the farther partner, from `partner_dist_max_m`;
- `rt60`: the card's `rt60_bin`;
- any scene-card field, for example `hvac_dba` or `mic_height_cm`.

**`compare`** prints, per scenario and metric, each method's paired per-geometry difference against `--ref`, with its 95% CI, the Wilcoxon p and the Holm-corrected p.

**Result files**
- `results2/<set>/<method>/<name>.<corpus>.json`, and with `--save-audio` also `<name>.<corpus>.wav`: the final output y1 as float32, not normalized.
- Top-level keys: `bench` (`bench2/2` since the harness verification), `scene`, `set`, `method`, `scenario`, `corpus`, `card` (the scene card), `status`, `error`, `traceback`, `runs` (name, flipped groups and RTF of each run), `extra`, `stages`, `algo_latency_ms`, `offline_only`, `oracle` (true for oracle methods), `time_s`, `delay`, `delay_found`, `aligned`, `m` (the metrics above), `legacy`, `talkers` (per talker), `pending`.
- NaN is written as null. A result is about 8 kB for a bus scene and 20 kB for a meeting.

**Method contract, as the bench applies it**
- Inputs: a fresh writable float32 copy of the (signed) mix, `fs` = 16000, and `info` = `{"scenario", "spacing_m": 0.04}`.
- `return_stages=True` is passed when `enhance` accepts it (or `**kwargs`). The return value is an array or `{"out": y, "pre_agc": y0}`. `pre_agc` must be the signal that actually enters the method's last level-control stage: TA, NA, ΔSNR, ΔSIR and T2 are read from it, and the bench cannot check that it is genuine (a make-up gain before the AGC raises TA_k 1:1; "Harness verification").
- The output must have exactly T samples and be finite; otherwise the scene is an error.
- Optional module attributes: `ALGO_LATENCY_MS` (reported for B10) and `OFFLINE_ONLY = True` (Tier 3: RTF reported, not gated).
- The method must be deterministic and stateless between calls, so that every Hagerman run sees the same algorithm: no module-level caches of signals, adaptive states or calibrations carried from one `enhance` call to the next. `rerun_db` checks this on the extra subset.
- **Code policy** (`bench2.policy_violations`, checked when a named method is loaded; a violation is a `PolicyError` and the scene is an error). Methods run in the bench process next to the rendered scene and scene2's render cache, so their code (the method module and every `methods.*` module loaded with it, oracles excluded) may import only numpy, scipy, soundfile, pyroomacoustics, onnxruntime, the standard library, `lab.dsp`, `lab.common` and `methods.*`, and may not contain string literals (docstrings excepted) naming `scenes2`, `scenes/`, `results2`, `corpus/` or `corpora/`. Dynamic imports are not detected; they are still forbidden.
- **Oracles.** Only methods whose name starts with `oracle` get a third `info` key, `info["oracle"]` (`bench2.oracle_view`): the unmodified scene (`id`, `fs`, `mix`, `comp2`, `groups`, `refs`, `talker_group`, `segs` and a copy of `meta`), the same object for every Hagerman run. Oracles are labelled cheating upper bounds, never candidates; their gate verdicts mean nothing.
- RTF is measured with one BLAS thread per worker. `lab.bench2` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1 unless they are already set. Methods that use onnxruntime should create their sessions with `intra_op_num_threads=1`, so that timings compare.

**M0** (`methods/m0_mica_hpf.py`)
- Mic A through a causal 2nd-order Butterworth high-pass at 120 Hz (bus) or 80 Hz (meeting): the S0 prefilter of `methods/_core/calib.py`.
- The filter is written out in the method, so the baseline does not depend on core modules.
- It returns stages (pre = out) and sets `ALGO_LATENCY_MS = 0`. Its measured delay is 0 samples on the test and smoke scenes.

**Cost** (scorer only, one process, this Mac)
- The method's own time is runs × duration × RTF. Runs per scene:
  - bus: 5 (the 4 Hagerman runs + near; 6 with a probe), 10 on the extra subset;
  - meeting: 2, 3 on the extra subset.
- Scoring time per method and scene, including the legacy PESQ:
  - 12 s smoke bus scene: 0.8 s;
  - 20 s test bus scene: 1.0–1.7 s;
  - 48 s smoke meeting: 0.8 s;
  - 6 min meeting with 20 talkers: 8–13 s, depending on machine load. About 3 s of this is the input side, computed once per scene for all methods.
- Peak memory for that meeting is 1.23 GB, mostly the scene render.
- Measured full passes, and what a method adds to them: "Baselines", "Cost of a pass".

**Python API**
```python
from lab import bench2, metrics2, stats
res, y1 = bench2.run_method("m0_mica_hpf", scene, ctx=None, card=None, extra=False)  # never raises
runs, comps = metrics2.plan(scene, extra=False)       # [(run, flipped groups)], {component: (terms, groups)}
ctx = metrics2.SceneCtx(scene)                        # input side, shared between methods
res = metrics2.score(scene, outs, comps, ctx)         # outs = {run: {"out": y, "pre": y0}}
rows = bench2.load_rows("dev", corpus="tts")
stats.evaluate_gates(rows_of_one_method, "bus")
stats.paired(rows, ref_rows, "d_estoi_min")
```

## Harness verification (real v2 scenes)

Owner of the whole harness since this pass: harness:verify-baseline. Every check below ran on the built `scenes2/` sets, not on fabricated scenes. The evidence scripts are in `tmp/harness:verify-baseline/` (run from the lab folder with `PYTHONPATH=.`): `verify.py <check>` for the table below, `pax_level.py` for passenger levels, `tables.py` for the baseline tables.

| Check (DESIGN) | Scenes | Requirement | Measured |
|---|---|---|---|
| Σ comp2 = mix (B.5) | 8 bus (dev −5/0/+5 dB, test −10/+10 dB) + 2 meetings, impairments on and off | ≤ 1e-6 | `signed_mix` equals `mix` bit for bit in 20 of 20 renders; float64 sum within 1.4e-8 (≤ 1.5e-7 of the peak) |
| Hagerman decoding of LTI methods (B.5): `avg`, the M0 high-pass, a fixed endfire fractional-delay DAS | 4 dev bus scenes with all extra runs, 1 meeting | ≥ 60 dB | every decoded component (A, B, N, near, cone, conv, far, engine, rest, S) matches the method applied to that component within 128–151 dB (float32 limit); closure and consistency 150 dB |
| Partner B inverted in an ideal output (B.9) | 5 dev bus scenes | ΔSDR < 0.1 dB, ΔESTOI < 0.005 | ≤ 2.2e-3 dB and ≤ 5.3e-6 (ideal = early images: SDR 51–90 dB, ESTOI 1.0000) |
| Oracle-VAD gate on Mic A (B.9) | 10 dev bus geometries at 0 dB | ΔSNR_active ±0.5 dB, NM ≥ 20 dB, G0 fails | ΔSNR_active −0.003 to 0.000 dB, NM 60 dB (cap), worst-partner ΔESTOI 0.0000; G0 FAIL on "NM ≤ 6 dB" only; focus 0 of 9 scenes (RDA P10 ≈ 0 dB); legacy BR +40 dB |
| Ideal output: the partners' own Mic-A comp2 | 3 dev bus scenes | near-perfect | worst-partner ESTOI 0.95–0.98 (input 0.09–0.46), TA 0.000 dB, NA_active and ΔSIR_near +60 dB (cap), NM 0, consistency and closure 150 dB |
| Far-talker cut hidden by an AGC (DESIGN §4.4 legacy warning) | 5 dev meetings | T2 catches it | pre stage −10 dB for talkers ≥ 4 m, final stage re-leveled by an oracle AGC: T2 margin −10.0 dB in 5 of 5 rooms, G0 FAIL (T2, NM, ESTOI); T3 std 0.7–1.0 dB (the leveling itself is credited, as intended); pumping std 6.4–9.6 dB, NM +6.1 to +15.1 dB |
| Partner TDOA truth vs GCC-PHAT (300–3500 Hz) on the partner components | all 63 bus geometries at 0 dB, impairments off and on (skew added to the truth) | as designed | 252 measurements: median error 1.1 µs, P90 5.7 µs, max 20.7 µs. The largest errors come with a strong early reflection (−4 dB at 1.4 ms); there the direct-path TDOA read from the RIRs matches the truth within 0.4 µs. Truth \|τ\|/(d/c): aligned 0.94–1.00, 45° 0.51–0.86, broadside 0.02–0.32, side-by-side 0.00–0.97, lap-flat 0.01–0.27 |
| refs time-aligned with Mic A | 4 bus + 2 meetings, 80 talkers | lag 0 | lag 0 for all 80 |
| E8 impairments | 3 bus + 1 meeting, rendered with impairments on and off | gain, high-pass, skew as drawn | per-mic transfer function within 0.10 dB and 0.7° (median) of gain × 1st-order high-pass × skew; group gains identical on and off; SNR rule realized within 0.1 dB (on) and exactly (off) |
| What reaches a method | every bench call | x, fs, info only | a fresh writable float32 copy that shares no memory with the scene (writing into it does not reach the scene); `info` = {scenario, spacing_m} |

**Fixed in this pass** (`lab/bench2.py`, version `bench2/2`; results written before carry `bench2/1`, decision metrics unchanged)
1. **Leak path: method code could reach the scene.** Methods run in the bench worker that rendered the scene, and `lab.scene2` keeps the last two renders in a cache; nothing stopped method code from importing `lab.scene2` or `lab.metrics2`, or reading `scenes2/`. The bench now checks every named method's code statically (`bench2.policy_violations`, "Bench usage"). All 22 non-oracle methods in `methods/` on 2026-11-03 pass.
2. **State kept between calls.** Hagerman decoding assumes every run sees the same algorithm. A module-level adaptive state or streaming cache carried from one `enhance` call to the next corrupts the decoded components silently, and a method that remembered earlier inputs could decode A, B and N itself ((x_y1 + x_y2)/2 = A). The extra subset now repeats y1 last and reports `rerun_db` (150 = identical; the tables show it as `rerun`). Smoke check of both new checks on the team's model-free methods (`s0_calib_only`, M1–M7, M10, M11 and their online variants, 15 methods, smoke set with all extra runs): no policy violation, `rerun_db` 150 for all, |delay| ≤ 1; their consistency is 150 dB (odd-symmetric, as expected) and closure 15–57 dB.
3. **Oracle side channel and `methods/oracle2_irm.py`** (task 3; "Bench usage" for the contract).

**Found, not fixed** (decisions for the DESIGN and scene owners; listed in the harness report)
1. **Near passengers are quiet.** The bus SNR rule scales the near, cone and conversing passengers together with the crowd and the engine (bus deviation 3). At 0 dB SNR a near passenger's active level is −11.0 dB re the partners' (median over dev scenes of the 3-passenger median; loudest passenger −8.1 dB; range −18.5 to −6.8 dB); −6.0 dB at −5 dB SNR and −16.0 dB at +5 dB. Without the rule's scaling (−7.2 dB median at 0 dB) they would sit about 4 dB below the partners. Input passenger ESTOI (partners silent) is 0.074 median and 0.21 at most on dev. Consequences: focus criterion 2 (passenger ESTOI ≤ 0.25) already holds at the input on every scene; the G2 "passenger ESTOI drop ≥ 0.15" cannot be met; B4–B6 measure the suppression of passengers who are already masked. DESIGN E4 asked for passenger levels from the partners' Lombard distribution precisely because v1 passengers were quieter than the partners. A fix (not done, because it changes every bus recipe while method agents are tuning): keep `near_*`, `cone` and `conv_*` at physical level and let the rule scale only `babble_far` and `engine_road`, in a new set rather than in place of dev and test.
2. **The focus composite is out of reach, even for the oracle.** `oracle2_irm` reaches a mean RDA of +11.5 dB, but its RDA P10 is +4.8 dB (dev), so focus criterion 3 (RDA P10 ≥ 10 dB, also a G2 criterion) fails on every scene and focus% is 0 for all four baselines. The lowest 1 s windows are those where a passenger overlaps a partner, and a time-frequency mask cannot remove overlapping speech. Until DESIGN recalibrates the 10 dB threshold (§4.7), compare RDA P10 and ΔSIR_near as numbers; focus% has no dynamic range on these scenes.
3. **TA_k is absolute and `pre_agc` is self-reported.** A make-up gain before the AGC raises TA_k one for one while ΔSNR_active does not move, so G0's "both TA_k ≥ −3 dB" can be met by gain alone. Read TA together with ΔSNR_active and the balance.
4. **The delay safety net does not penalize.** Beyond ±2 samples the bench shifts the output and scores it anyway (`delay_ok` = 0); no gate fails. All four baselines measure 0 (`avg` ±1 on some scenes: its output is centre-referenced).
5. **The oracle IRM gates.** It passes bus G1 but fails G0 on NM (+41 dB bus, +28 dB meeting): a ratio mask removes the noise in pauses. It bounds ESTOI and ΔSNR, not G0-compliant processing.

## Baselines

The reference numbers every method is compared against. TTS corpus, E8 impairments on, `info["spacing_m"]` = 0.04, default extra subset, bench `bench2/2`, 2026-11-03.
- dev: bus 90 scenes on 30 geometries (6 per E1 stratum) at −5 / 0 / +5 dB; meeting 20 rooms.
- test: bus 110 scenes on 30 geometries at −5 / 0 / +5 dB plus −10 / +10 dB on the 10 sweep geometries; meeting 20 rooms. Run once here as the reference for the final evaluator; nobody tunes on it.
- Methods: `mic_a` (raw Mic A: every Δ is 0 by construction, a scorer sanity check), `m0_mica_hpf` (DESIGN M0, the `compare` reference), `avg` ((A + B)/2: an uncalibrated broadside delay-and-sum) and `oracle2_irm` (a cheating upper bound, below).
- Every value is a mean over geometries of per-geometry means, as `table` computes it; G0 and G1 follow "Gates". Result files: `results2/<set>/<method>/`. To reprint: `.venv/bin/python -m lab.bench2 table --set dev --by stratum,snr` and `.venv/bin/python -m lab.bench2 compare --set dev --ref m0_mica_hpf`.
- On TTS the gate verdicts are indicative only (DESIGN §4.6 decides on natural speech).

**`methods/oracle2_irm.py`: ideal ratio mask, a labelled cheat.** It reads the true scene from `info["oracle"]`, which the bench gives only to methods named `oracle*`. Target T = the comp2 of every `partner` group (bus) or `talker` group (meeting) at Mic A, reverberation included; N = mix − T (interferers, noise and sensor). Mask m = (|T|² / (|T|² + |N|²))^0.5 on the `lab.dsp` STFT (512/128), applied to Mic A. The mask comes from the unmodified scene, so every Hagerman run gets the same one: the oracle is a fixed linear time-varying filter that decodes exactly (consistency, closure and `rerun_db` 150 dB on every extra-subset scene). The v1 oracle in DESIGN §0 masked against the early image (late reverberation counted as noise) and scored the summed partners, so its 0.398 → 0.877 is not comparable.

### Bus, dev

| method | geoms | wrstESTOI in | wrstΔESTOI | wrstΔSDR | TAmin | balP90 | ΔSNRact | NM | ΔSIRnear | RDAp10 | paxESTOI in | paxESTOI | focus% | RTF | delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mic_a | 30 | 0.308 | -0.000 | +0.0 | -0.0 | 0.0 | +0.0 | -0.0 | +0.0 | -0.0 | 0.085 | 0.085 | 0 | 0.0000 | +0.0 |
| m0_mica_hpf | 30 | 0.308 | +0.001 | -0.2 | -0.4 | 0.3 | -0.2 | +0.1 | -0.0 | -0.2 | 0.085 | 0.086 | 0 | 0.0002 | +0.0 |
| avg | 30 | 0.308 | -0.013 | +0.3 | -0.3 | 0.3 | +1.1 | -0.8 | -0.0 | -0.2 | 0.085 | 0.081 | 0 | 0.0000 | -0.0 |
| oracle2_irm | 30 | 0.308 | +0.366 | +8.5 | -1.5 | 1.1 | +12.6 | +41.1 | +7.0 | +4.8 | 0.085 | -0.003 | 0 | 0.0021 | +0.0 |

`paxESTOI in` / `paxESTOI`: B6 on raw Mic A and on the output. NAact: `avg` +1.2 dB, `oracle2_irm` +13.8 dB, M0 +0.1 dB.

| method | G0 | G1 | failing criteria |
|---|---|---|---|
| mic_a | PASS | FAIL | worst-partner dESTOI >= +0.03; worst-partner dESTOI >= +0.05 (aligned); dSNR_active >= +3 dB (aligned); dSNR_active >= +1.5 dB (all strata) |
| m0_mica_hpf | PASS | FAIL | worst-partner dESTOI >= +0.03; worst-partner dESTOI >= +0.05 (aligned); dSNR_active >= +3 dB (aligned); dSNR_active >= +1.5 dB (all strata) |
| avg | FAIL | FAIL | worst-partner ESTOI >= input - 0.01; worst-partner dESTOI >= +0.03; worst-partner dESTOI >= +0.05 (aligned); dSNR_active >= +3 dB (aligned); dSNR_active >= +1.5 dB (all strata) |
| oracle2_irm | FAIL | PASS | NM <= 6 dB |

Gate details for M0 (point, 95% CI): worst-partner ΔESTOI +0.0010 [+0.0005, +0.0015]; TA_min −0.38 [−0.41, −0.35] dB (the 120 Hz high-pass removes in-band energy near 150 Hz); balance P90 0.26 [0.23, 0.29] dB; NM +0.09 [+0.06, +0.11] dB; ΔSNR_active −0.16 [−0.21, −0.12] dB (aligned −0.19 dB); RTF P90 0.0002.

### Bus, dev, by E1 stratum (`mic_a` is 0 everywhere)

| method | group | geoms | wrstESTOI in | wrstΔESTOI | ΔSNRact | ΔSIRnear | RDAp10 | TAmin | focus% |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | aligned | 6 | 0.311 | +0.001 | -0.2 | -0.0 | -0.2 | -0.4 | 0 |
| avg | aligned | 6 | 0.311 | -0.035 | +0.9 | -0.1 | -0.5 | -0.6 | 0 |
| oracle2_irm | aligned | 6 | 0.311 | +0.354 | +12.6 | +7.0 | +4.8 | -1.5 | 0 |
| m0_mica_hpf | 45 | 6 | 0.255 | +0.001 | -0.2 | -0.0 | -0.2 | -0.4 | 0 |
| avg | 45 | 6 | 0.255 | -0.021 | +1.1 | -0.1 | -0.4 | -0.3 | 0 |
| oracle2_irm | 45 | 6 | 0.255 | +0.363 | +11.5 | +7.8 | +5.4 | -1.9 | 0 |
| m0_mica_hpf | broadside | 6 | 0.337 | +0.001 | -0.1 | -0.0 | -0.2 | -0.4 | 0 |
| avg | broadside | 6 | 0.337 | +0.004 | +0.7 | +0.1 | -0.1 | -0.1 | 0 |
| oracle2_irm | broadside | 6 | 0.337 | +0.382 | +11.0 | +6.7 | +4.4 | -1.3 | 0 |
| m0_mica_hpf | side_by_side | 6 | 0.293 | +0.001 | -0.2 | -0.1 | -0.2 | -0.4 | 0 |
| avg | side_by_side | 6 | 0.293 | -0.018 | +1.1 | +0.0 | -0.2 | -0.2 | 0 |
| oracle2_irm | side_by_side | 6 | 0.293 | +0.375 | +13.5 | +6.5 | +4.9 | -1.7 | 0 |
| m0_mica_hpf | lap_flat | 6 | 0.345 | +0.001 | -0.2 | -0.0 | -0.2 | -0.3 | 0 |
| avg | lap_flat | 6 | 0.345 | +0.004 | +1.5 | -0.0 | -0.2 | -0.1 | 0 |
| oracle2_irm | lap_flat | 6 | 0.345 | +0.359 | +14.5 | +7.2 | +4.7 | -1.0 | 0 |

### Bus, dev, by SNR

| method | group | geoms | wrstESTOI in | wrstΔESTOI | ΔSNRact | ΔSIRnear | RDAp10 | TAmin | focus% |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | -5.0 | 30 | 0.169 | +0.000 | -0.1 | -0.0 | -0.2 | -0.4 | 0 |
| avg | -5.0 | 30 | 0.169 | -0.011 | +0.7 | -0.0 | -0.2 | -0.3 | 0 |
| oracle2_irm | -5.0 | 30 | 0.169 | +0.488 | +11.8 | +8.5 | +6.1 | -2.4 | 0 |
| m0_mica_hpf | 0.0 | 30 | 0.302 | +0.001 | -0.2 | -0.0 | -0.2 | -0.4 | 0 |
| avg | 0.0 | 30 | 0.302 | -0.014 | +1.1 | -0.0 | -0.2 | -0.3 | 0 |
| oracle2_irm | 0.0 | 30 | 0.302 | +0.365 | +12.3 | +7.0 | +4.8 | -1.3 | 0 |
| m0_mica_hpf | 5.0 | 30 | 0.454 | +0.001 | -0.2 | -0.0 | -0.2 | -0.4 | 0 |
| avg | 5.0 | 30 | 0.454 | -0.015 | +1.5 | -0.0 | -0.2 | -0.3 | 0 |
| oracle2_irm | 5.0 | 30 | 0.454 | +0.246 | +13.8 | +5.6 | +3.7 | -0.7 | 0 |

### Meeting, dev

| method | geoms | wrstESTOI in | wrstΔESTOI | meanΔESTOI | farΔESTOI | T1cap% | T2far−near | T3std | T3range | NAact | NM | pumpStd | even% | RTF | delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mic_a | 20 | 0.279 | -0.000 | -0.000 | +0.000 | 52 | +0.0 | 4.2 | 16.0 | +0.0 | +0.0 | 0.0 | 0 | 0.0000 | +0.0 |
| m0_mica_hpf | 20 | 0.279 | +0.001 | +0.001 | +0.001 | 53 | -0.0 | 4.2 | 16.1 | +0.0 | +0.0 | 0.0 | 0 | 0.0002 | +0.0 |
| avg | 20 | 0.279 | -0.005 | -0.015 | -0.016 | 38 | -0.0 | 4.2 | 16.1 | +0.2 | -0.0 | 0.0 | 0 | 0.0000 | +0.0 |
| oracle2_irm | 20 | 0.279 | +0.325 | +0.186 | +0.203 | 52 | -0.1 | 4.4 | 16.7 | +4.7 | +27.7 | 1.5 | 0 | 0.0043 | +0.0 |

| method | G0 | G1 | failing criteria |
|---|---|---|---|
| mic_a | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| m0_mica_hpf | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| avg | FAIL | FAIL | dESTOI >= -0.03 for >= 19/20 talkers; T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| oracle2_irm | FAIL | FAIL | NM <= 6 dB; T3 level std <= 1.5 dB; T3 level range <= 6 dB |

M0's G1 numbers (point, 95% CI): T3 std 4.21 [3.91, 4.51] dB, range 16.1 [14.9, 17.2] dB, mean ΔESTOI +0.0006, > 4 m ΔESTOI +0.0007, NA_active +0.03 dB. The 52–53% capture rate of `mic_a` and M0 is the level spread alone: T1 needs |L_k − median| ≤ 3 dB.

### Paired differences against M0, dev (95% bootstrap CI over geometries)

| metric | `mic_a` | `avg` | `oracle2_irm` |
|---|---|---|---|
| bus worst-partner ΔESTOI | −0.0010 [−0.0015, −0.0005] | −0.014 [−0.018, −0.011] | +0.366 [+0.348, +0.384] |
| bus worst-partner ΔSDR (dB) | +0.21 [+0.16, +0.25] | +0.48 [+0.26, +0.71] | +8.7 [+7.7, +9.7] |
| bus ΔSNR_active (dB) | +0.16 [+0.12, +0.21] | +1.24 [+0.93, +1.56] | +12.8 [+11.7, +14.0] |
| bus NM (dB) | −0.09 [−0.11, −0.06] | −0.90 [−1.21, −0.58] | +41.0 [+37.5, +44.0] |
| bus ΔSIR_near (dB) | +0.04 [−0.01, +0.08] | +0.02 [−0.06, +0.09] | +7.1 [+6.7, +7.5] |
| bus RDA P10 (dB) | +0.20 [+0.16, +0.24] | −0.04 [−0.12, +0.03] | +5.1 [+4.7, +5.4] |
| meeting worst-talker ΔESTOI | −0.0009 [−0.0013, −0.0005] | −0.006 [−0.011, −0.002] | +0.324 [+0.284, +0.367] |
| meeting mean ΔESTOI | −0.0006 [−0.0007, −0.0004] | −0.015 [−0.020, −0.011] | +0.186 [+0.165, +0.208] |
| meeting > 4 m ΔESTOI | −0.0007 [−0.0009, −0.0005] | −0.017 [−0.023, −0.011] | +0.202 [+0.178, +0.228] |
| meeting T2 margin (dB) | +0.00 [−0.01, +0.01] | −0.05 [−0.10, +0.01] | −0.12 [−0.16, −0.07] |
| meeting NA_active (dB) | −0.03 [−0.04, −0.03] | +0.21 [+0.01, +0.40] | +4.7 [+4.5, +5.0] |
| meeting capture rate | −0.003 [−0.008, 0] | −0.145 [−0.213, −0.083] | −0.005 [−0.013, 0] |

### Bus, test (reference only)

| method | geoms | wrstESTOI in | wrstΔESTOI | wrstΔSDR | TAmin | balP90 | ΔSNRact | NM | ΔSIRnear | RDAp10 | paxESTOI in | paxESTOI | focus% | RTF | delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mic_a | 30 | 0.321 | -0.000 | +0.0 | -0.0 | 0.0 | -0.0 | +0.0 | +0.0 | -0.0 | 0.067 | 0.067 | 0 | 0.0000 | +0.0 |
| m0_mica_hpf | 30 | 0.321 | +0.000 | -0.0 | -0.2 | 0.1 | -0.0 | +0.1 | +0.1 | +0.0 | 0.067 | 0.067 | 0 | 0.0002 | +0.0 |
| avg | 30 | 0.321 | -0.016 | +0.2 | -0.2 | 0.3 | +1.2 | -1.0 | -0.0 | -0.2 | 0.067 | 0.066 | 0 | 0.0000 | +0.0 |
| oracle2_irm | 30 | 0.321 | +0.368 | +9.1 | -1.3 | 0.9 | +13.6 | +41.6 | +7.2 | +5.3 | 0.067 | 0.005 | 0 | 0.0018 | +0.0 |

| method | G0 | G1 | failing criteria |
|---|---|---|---|
| mic_a | PASS | FAIL | worst-partner dESTOI >= +0.03; worst-partner dESTOI >= +0.05 (aligned); dSNR_active >= +3 dB (aligned); dSNR_active >= +1.5 dB (all strata) |
| m0_mica_hpf | PASS | FAIL | worst-partner dESTOI >= +0.03; worst-partner dESTOI >= +0.05 (aligned); dSNR_active >= +3 dB (aligned); dSNR_active >= +1.5 dB (all strata) |
| avg | FAIL | FAIL | worst-partner ESTOI >= input - 0.01; worst-partner dESTOI >= +0.03; worst-partner dESTOI >= +0.05 (aligned); dSNR_active >= +3 dB (aligned); dSNR_active >= +1.5 dB (all strata) |
| oracle2_irm | FAIL | PASS | NM <= 6 dB |

### Bus, test, by E1 stratum

| method | group | geoms | wrstESTOI in | wrstΔESTOI | ΔSNRact | ΔSIRnear | RDAp10 | TAmin | focus% |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | aligned | 6 | 0.311 | -0.000 | +0.1 | +0.1 | -0.0 | -0.2 | 0 |
| avg | aligned | 6 | 0.311 | -0.042 | +0.6 | -0.1 | -0.5 | -0.3 | 0 |
| oracle2_irm | aligned | 6 | 0.311 | +0.375 | +11.8 | +7.3 | +5.5 | -1.4 | 0 |
| m0_mica_hpf | 45 | 6 | 0.345 | +0.000 | -0.1 | +0.2 | +0.0 | -0.2 | 0 |
| avg | 45 | 6 | 0.345 | -0.032 | +0.9 | -0.1 | -0.4 | -0.3 | 0 |
| oracle2_irm | 45 | 6 | 0.345 | +0.387 | +14.9 | +7.1 | +4.9 | -1.0 | 0 |
| m0_mica_hpf | broadside | 6 | 0.301 | +0.002 | +0.0 | +0.1 | +0.0 | -0.2 | 0 |
| avg | broadside | 6 | 0.301 | +0.006 | +0.8 | +0.0 | -0.1 | -0.1 | 0 |
| oracle2_irm | broadside | 6 | 0.301 | +0.371 | +11.4 | +7.5 | +5.7 | -1.5 | 0 |
| m0_mica_hpf | side_by_side | 6 | 0.311 | +0.000 | +0.0 | +0.1 | +0.0 | -0.2 | 0 |
| avg | side_by_side | 6 | 0.311 | -0.019 | +1.2 | -0.0 | -0.2 | -0.1 | 0 |
| oracle2_irm | side_by_side | 6 | 0.311 | +0.356 | +13.9 | +7.2 | +5.3 | -1.5 | 0 |
| m0_mica_hpf | lap_flat | 6 | 0.338 | +0.000 | -0.1 | +0.1 | -0.0 | -0.2 | 0 |
| avg | lap_flat | 6 | 0.338 | +0.007 | +2.6 | +0.1 | -0.0 | -0.0 | 0 |
| oracle2_irm | lap_flat | 6 | 0.338 | +0.351 | +15.9 | +7.0 | +5.1 | -1.3 | 0 |

### Bus, test, by SNR (−10 and +10 dB: the 10 sweep geometries only)

| method | group | geoms | wrstESTOI in | wrstΔESTOI | ΔSNRact | ΔSIRnear | RDAp10 | TAmin | focus% |
|---|---|---|---|---|---|---|---|---|---|
| m0_mica_hpf | -10.0 | 10 | 0.084 | -0.000 | +0.0 | +0.1 | -0.0 | -0.2 | 0 |
| avg | -10.0 | 10 | 0.084 | -0.015 | +0.8 | -0.1 | -0.4 | -0.2 | 0 |
| oracle2_irm | -10.0 | 10 | 0.084 | +0.581 | +13.9 | +10.4 | +7.8 | -3.2 | 0 |
| m0_mica_hpf | -5.0 | 30 | 0.182 | +0.000 | +0.0 | +0.1 | +0.0 | -0.2 | 0 |
| avg | -5.0 | 30 | 0.182 | -0.015 | +0.9 | -0.0 | -0.2 | -0.2 | 0 |
| oracle2_irm | -5.0 | 30 | 0.182 | +0.495 | +12.7 | +8.8 | +6.7 | -2.0 | 0 |
| m0_mica_hpf | 0.0 | 30 | 0.312 | +0.001 | -0.0 | +0.1 | +0.0 | -0.2 | 0 |
| avg | 0.0 | 30 | 0.312 | -0.017 | +1.2 | -0.0 | -0.2 | -0.2 | 0 |
| oracle2_irm | 0.0 | 30 | 0.312 | +0.365 | +13.1 | +7.1 | +5.3 | -1.2 | 0 |
| m0_mica_hpf | 5.0 | 30 | 0.459 | +0.001 | -0.0 | +0.1 | +0.0 | -0.2 | 0 |
| avg | 5.0 | 30 | 0.459 | -0.018 | +1.6 | -0.0 | -0.2 | -0.2 | 0 |
| oracle2_irm | 5.0 | 30 | 0.459 | +0.241 | +14.4 | +5.6 | +4.0 | -0.6 | 0 |
| m0_mica_hpf | 10.0 | 10 | 0.595 | +0.001 | -0.1 | +0.1 | -0.0 | -0.2 | 0 |
| avg | 10.0 | 10 | 0.595 | -0.021 | +2.2 | -0.1 | -0.4 | -0.2 | 0 |
| oracle2_irm | 10.0 | 10 | 0.595 | +0.137 | +18.6 | +4.3 | +2.8 | -0.3 | 0 |

### Meeting, test (reference only)

| method | geoms | wrstESTOI in | wrstΔESTOI | meanΔESTOI | farΔESTOI | T1cap% | T2far−near | T3std | T3range | NAact | NM | pumpStd | even% | RTF | delay |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mic_a | 20 | 0.267 | -0.000 | -0.000 | -0.000 | 55 | +0.0 | 4.1 | 16.1 | -0.0 | -0.0 | 0.0 | 0 | 0.0000 | +0.0 |
| m0_mica_hpf | 20 | 0.267 | +0.001 | +0.001 | +0.001 | 55 | +0.0 | 4.1 | 16.1 | +0.0 | -0.0 | 0.0 | 0 | 0.0002 | +0.0 |
| avg | 20 | 0.267 | -0.011 | -0.016 | -0.023 | 38 | -0.1 | 4.1 | 16.1 | +0.6 | +0.0 | 0.0 | 0 | 0.0000 | +0.0 |
| oracle2_irm | 20 | 0.267 | +0.329 | +0.187 | +0.210 | 55 | -0.2 | 4.3 | 16.7 | +4.7 | +27.9 | 1.5 | 0 | 0.0026 | +0.0 |

| method | G0 | G1 | failing criteria |
|---|---|---|---|
| mic_a | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| m0_mica_hpf | PASS | FAIL | T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| avg | FAIL | FAIL | worst-talker ESTOI >= input - 0.01; dESTOI >= -0.03 for >= 19/20 talkers; T3 level std <= 1.5 dB; T3 level range <= 6 dB; mean dESTOI >= +0.02; dESTOI >= +0.03 in the > 4 m bin; NA_active >= 3 dB |
| oracle2_irm | FAIL | FAIL | NM <= 6 dB; T3 level std <= 1.5 dB; T3 level range <= 6 dB |

### Calibration: what a ΔESTOI is worth

- **Input ESTOI per dB of SNR (bus dev).** Least-squares slope of each partner's input ESTOI (raw Mic A against its early image, the B1 regions) over −5 / 0 / +5 dB, per geometry: **0.0294 per dB** [0.0284, 0.0304] over 60 partner-geometries (median 0.0295); by stratum 0.028 (45°) to 0.032 (broadside); worst partner 0.0285. The test set gives 0.0289. DESIGN used 0.034 from v1. In SNR terms: the G1 targets +0.03 and +0.05 worst-partner ΔESTOI are about +1.0 and +1.7 dB of effective SNR, and the G2 target +0.08 about +2.7 dB.
- **Input levels.** Mean worst-partner input ESTOI: 0.169 / 0.302 / 0.454 at −5 / 0 / +5 dB (dev); test adds 0.084 at −10 dB and 0.595 at +10 dB. Input SDR rises only 0.53 dB per dB of SNR, because the early-image reference counts late reverberation as distortion. Meeting input: worst talker 0.279 (dev) and 0.267 (test), talker mean 0.54; level spread across talkers 4.2 dB std and 16 dB range.
- **Oracle headroom (dev).** Bus: worst-partner ΔESTOI +0.37 (+0.49 at −5 dB, +0.25 at +5 dB; aligned +0.35), ΔSNR_active +12.6 dB, ΔSIR_near +7.0 dB, RDA P10 +4.8 dB. Meeting: worst-talker ΔESTOI +0.33, mean +0.19, > 4 m +0.20, NA_active +4.7 dB. Spatial methods are not bounded by this single-channel mask, but no two-mic method is expected near it (DESIGN §1).
- **Cheap two-mic gain.** `avg` gains +1.1 dB ΔSNR_active (dev; +1.5 dB in the lap-flat stratum, where the partners are near broadside) but loses worst-partner ESTOI (−0.013; −0.035 in the aligned stratum, where the endfire comb filter hits the partners) and fails G0. Spatial processing has to be calibrated and steered (S0, S1) to beat M0.

### Cost of a pass

Measured on this Mac while other agents loaded it (load average 60–180 on 10 cores), so expect less on a quiet machine.

| Pass | Workers | Wall time | CPU time |
|---|---|---|---|
| dev, `m0_mica_hpf` alone | 3 | 3 min 03 s | 346 s |
| dev, `mic_a` + `avg` + `oracle2_irm` in one run | 4 | 4 min 48 s | 764 s |
| test, all four in one run | 4 | 5 min 09 s | 1048 s |

- Fixed cost per scene: rendering about 1 s (bus) or 2.5 s (meeting), once per job; scoring per method about 2–3 s (bus) or 4–7 s (meeting), the first method of a job also paying the input side.
- A method's own time comes on top. Per dev pass it processes **22,750 s of audio** (bus 12,960 s: 90 scenes × 24 s × 5 runs plus 6 more calls on the 15 extra-subset scenes; meeting 9,790 s: 20 rooms × about 204 s × 2 runs plus 2 more calls on 4 rooms). That is 22,750 × RTF CPU seconds, or about **126 × RTF minutes of wall time with 3 workers**: +13 min at RTF 0.1, +32 min at 0.25, +63 min at 0.5. The test pass is 25,360 s (141 × RTF min).
- Iteration subsets (dev): `--scenario bus --snr 0` is 30 scenes, 6 per stratum, 5 of them with extra runs: 4,320 s of method audio (about 24 × RTF min with 3 workers) plus about 1 min. `--scenario meeting --limit 5` is rooms r00–r04 without extra runs: 2,026 s (about 11 × RTF min) plus about 1 min.
- Memory: about 1.3 GB per worker on meetings; disk: about 1.1 MB of results per method and dev pass.

### Tuning protocol

- **Iterate on a dev subset, into a scratch results root.** Bus: `.venv/bin/python -m lab.bench2 run --methods <m>,m0_mica_hpf --set dev --scenario bus --snr 0 -j 3 --results-root tmp/<label>/r` (30 scenes, all five strata; listing M0 in the same run costs only its scoring and gives `compare` its reference in that root). Meeting: `--scenario meeting --limit 5`. Then `table` and `compare --ref m0_mica_hpf` with the same `--results-root` (and `--snr 0`), and read the by-stratum table (`--by stratum`), not only the means. `--extra none` saves the extra runs (about 17% of the method time) while iterating; keep them for full passes.
- **Run the full dev set** (all SNRs, both scenarios, default extra subset, default results root `results2/`) before reporting any number, comparing methods or claiming a gate: the bootstrap CIs need all 30 bus geometries (6 per stratum for the aligned-stratum criteria) and all 20 rooms. Also run it after any change to shared `methods/_core` code the method depends on. Check `err`, `delay` (|delay| ≤ 2), `rerun` (150) and RTF in the table.
- **Never run or tune on test.** The test set is run once per milestone by the final evaluator only; the test numbers above are its M0 reference, not targets. Do not choose parameters, thresholds or variants from test results, and do not run your method on `--set test`.
- Gates on TTS scenes are indicative; DESIGN decides G1 on natural speech (E0) once that corpus is approved.
