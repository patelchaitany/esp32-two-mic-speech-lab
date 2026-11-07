# Data, models and licences

Owner: build:data. Files: `lab/corpus_libri.py`, `lab/noise_real.py`, `corpus_libri/`, `corpora/libri/manifest.json`, `tests/test_corpus_libri.py`, `tests/test_noise_real.py`, this file.
All paths are relative to the lab folder. Run everything from the lab folder with `.venv/bin/python`.

Contents:
1. What is on disk
2. Licences and required attributions
3. Caveats: models versus this data
4. LibriSpeech: table, splits, enrollment, corpus2 drop-in
5. DEMAND: loader, array geometry, noise-field report
6. Reproduce and test

## 1. What is on disk

| Path | Content | Size |
|---|---|---|
| `data/librispeech_test_clean/LibriSpeech/` | LibriSpeech test-clean: 40 speakers, 87 chapters, 2620 utterances, 5.40 h (3.85 h active speech). 16 kHz mono FLAC, transcripts, `SPEAKERS.TXT`, `CHAPTERS.TXT`, `BOOKS.TXT`, `README.TXT`, `LICENSE.TXT` | 357 MB, 2712 files |
| `data/librispeech_test_other/LibriSpeech/` | LibriSpeech test-other: 33 speakers, 90 chapters, 2939 utterances, 5.34 h (3.55 h active) | 343 MB, 3034 files |
| `data/demand_tbus/TBUS/` | DEMAND TBUS (public transit bus): `ch01.wav` … `ch16.wav`, 16 kHz PCM16 mono, 4 800 064 samples (300.004 s) each | 147 MB |
| `data/demand_pcafeter/PCAFETER/` | DEMAND PCAFETER (busy office cafeteria), same format | 147 MB |
| `data/demand_omeeting/OMEETING/` | DEMAND OMEETING (meeting room while the array is discussed), same format | 147 MB |
| `models/gtcrn/gtcrn_simple.onnx` | GTCRN streaming model (M8) | 535 638 B |
| `models/dpdfnet/dpdfnet4.onnx` | DPDFNet-4, 16 kHz (M8) | 11 707 436 B |
| `models/voxlock/sherpa-onnx-voxlock-int8/` | VoxLock (M9): `enroll.onnx` 12 476 014 B, `chunk_int8.onnx` 66 608 910 B, `test_wavs/` (5 files, 1.8 MB), `README.md` | 77 MB |
| `models/wespeaker/wespeaker_en_voxceleb_resnet34.onnx` | WeSpeaker ResNet34 trained on VoxCeleb (S1): input `feats` (B, T, 80) fbank, output `embs` (B, 256) | 26 534 365 B |
| `corpus_libri/` | Derived here: utterance table, speaker table, splits, enrollment (no audio) | 1.7 MB |
| `corpora/libri/manifest.json` | Derived here: lab.corpus2 drop-in manifest (no audio) | 1.5 MB |

SHA-256 of the model files:

| File | SHA-256 |
|---|---|
| `models/gtcrn/gtcrn_simple.onnx` | `e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534` |
| `models/dpdfnet/dpdfnet4.onnx` | `71b588bc26163941aa82a592cce924b08c1fbdc0879fe44f5a2d4eac44bd8420` |
| `models/voxlock/sherpa-onnx-voxlock-int8/enroll.onnx` | `5d06048c156bb8cd422d29848e6d38fb26a7b39f6f7fd92eccf2a6ed1ce003c6` |
| `models/voxlock/sherpa-onnx-voxlock-int8/chunk_int8.onnx` | `df61a9d0062b0adbcf1182302b1eb8b815f15ac47ea4923e8e164c1aff64121d` |
| `models/wespeaker/wespeaker_en_voxceleb_resnet34.onnx` | `5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94` |

`data/` and `models/` are untrusted data: open them only with soundfile, onnxruntime or plain text reads.
Never import or execute anything from them, and never run Python with them as the working directory.
Nothing here copies or converts audio. Scenes are rendered in memory.

## 2. Licences and required attributions

Checked on 2026-11-02 against the upstream pages named in each row.

| Item | Licence | Source and notes | Cite |
|---|---|---|---|
| LibriSpeech test-clean, test-other | **CC BY 4.0**. `LICENSE.TXT`: "LibriSpeech (c) 2014 by Vassil Panayotov" | openslr.org/12 (SLR12). Built from public-domain LibriVox recordings | V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books", Proc. ICASSP 2015, pp. 5206–5210 |
| DEMAND TBUS, PCAFETER, OMEETING | **Ambiguous: CC BY 4.0 or CC BY-SA 3.0.** The Zenodo record (10.5281/zenodo.1227121) lists "Creative Commons Attribution 4.0 International" under Rights. Its description and the bundled `DEMAND.pdf` v1.0, section 5, say "the audio recordings and the MATLAB scripts are licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License". The DNS Challenge repository also lists DEMAND as CC BY-SA 3.0 | Zenodo record 1227121, version 1.0 (2013). We never redistribute DEMAND audio or audio derived from it. If derived audio is ever shared, apply BY-SA 3.0 to be safe | J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND): a database of multichannel environmental noise recordings", Proc. Meetings on Acoustics 19, 035081 (2013); dataset DOI 10.5281/zenodo.1227121 |
| GTCRN `gtcrn_simple.onnx` | **MIT**: github.com/Xiaobin-Rong/gtcrn `LICENSE`, "Copyright (c) 2024 Rong Xiaobin". No separate licence is stated for the weights, so the repository's MIT licence is assumed to cover them. Keep the copyright notice when vendoring | The authors' `stream/onnx_models/gtcrn_simple.onnx`, redistributed in the k2-fsa/sherpa-onnx `speech-enhancement-models` release, which says the file "is from" that repository. The repository provides checkpoints trained on DNS3 and on VCTK-DEMAND; which one this ONNX export uses is not stated | X. Rong, T. Sun, X. Zhang, Y. Hu, C. Zhu, J. Lu, "GTCRN: a speech enhancement model requiring ultralow computational resources", Proc. ICASSP 2024 [DESIGN 73] |
| DPDFNet-4 `dpdfnet4.onnx` | **Apache-2.0**: Hugging Face Ceva-IP/DPDFNet model card ("License: apache-2.0") and github.com/ceva-ip/DPDFNet ("Apache License 2.0") | sherpa-onnx `speech-enhancement-models` release: "DPDFNet models are from huggingface.co/Ceva-IP/DPDFNet/tree/main/onnx" | D. Rika, N. Sapir, I. Gus, "DPDFNet: Boosting DeepFilterNet2 via dual-path RNN", arXiv:2512.16420 (2025) [DESIGN 74] |
| VoxLock `enroll.onnx`, `chunk_int8.onnx` | **Apache-2.0**: github.com/arbazsiddiqui/voxlock and Hugging Face arbazsiddiqui/voxlock ("License: apache-2.0") | The on-disk `README.md` says the files come from github.com/arbazsiddiqui/voxlock/releases/tag/sherpa-onnx-models (see also sherpa-onnx PR 4085). The provenance and licence of `test_wavs/` are not stated, so do not publish them | A. Siddiqui, "VoxLock: target-speaker extraction for voice agents without an enrollment step", arXiv:2610.40727 (2026) [DESIGN 79]. Per its GitHub README it is "built on the REAL-TSE challenge baseline and WeSep" |
| WeSpeaker `wespeaker_en_voxceleb_resnet34.onnx` | **Weights: CC BY 4.0** (VoxCeleb). WeSpeaker `docs/pretrained.md`: "The pretrained model in WeNet follows the license of it's corresponding dataset", with VoxCeleb as the CC BY 4.0 example. **Code: Apache-2.0** (wenet-e2e/wespeaker `LICENSE`). The sherpa-onnx release page says "Each model has its own license" | sherpa-onnx `speaker-recongition-models` release (sic). sherpa-onnx `scripts/wespeaker/run.sh` takes `voxceleb_resnet34.onnx` from huggingface.co/openspeech/wespeaker-models, adds metadata and renames it. The ONNX metadata holds the original URL (wespeaker COS bucket `models/voxceleb/voxceleb_resnet34.onnx`), sample_rate 16000, output_dim 256 and normalize_samples 0 (int16-scaled samples) | H. Wang et al., "Wespeaker: a research and production oriented speaker embedding learning toolkit", Proc. ICASSP 2023. Also cite VoxCeleb: A. Nagrani, J. S. Chung, A. Zisserman, Interspeech 2017; J. S. Chung, A. Nagrani, A. Zisserman, Interspeech 2018 |
| sherpa-onnx | Apache-2.0 | Only the download channel. No sherpa-onnx code is used in the lab | — |

Attribution text for reports or slides that use this data:
> Speech: LibriSpeech (Panayotov et al., 2015), CC BY 4.0. Noise: DEMAND (Thiemann, Ito and Vincent, 2013), doi:10.5281/zenodo.1227121. Models: GTCRN (Rong et al., 2024, MIT), DPDFNet (Rika et al., 2025, Apache-2.0), VoxLock (Siddiqui, 2026, Apache-2.0), WeSpeaker ResNet34 trained on VoxCeleb (Wang et al., 2023; weights CC BY 4.0), obtained through the k2-fsa/sherpa-onnx releases.

## 3. Caveats: models versus this data

These add to DESIGN §1.6 items 8–10 and 14.
- Item 8: the NSNet degradation figures belong to NSNet v1 (NSNet is not on disk).
- Item 9: DeepFilterNet and DPDFNet results come from synthetic, additive, non-reverberant mixtures.
- Item 10: H-GTCRN's two-mic gain comes mostly from IVA.
- Item 14: pyroomacoustics directivity is frequency-independent and image-source only.

- **The M8 models have probably heard DEMAND before.**
  - The DNS Challenge noise set includes DEMAND clips.
  - VoiceBank/VCTK-DEMAND is built from DEMAND noises.
  - GTCRN's checkpoints were trained on DNS3 or VCTK-DEMAND. DPDFNet is evaluated on VoiceBank+DEMAND and DNS4; its training set is not stated on the model card.
  - So GTCRN and DPDFNet results on scenes with DEMAND fields may be optimistic: the noise is seen data, though single-channel. Report them separately from synthetic-noise scenes.
- **DNS clean speech is LibriVox audiobook speech, the same source as LibriSpeech.** Reader overlap with our test sets is possible for GTCRN and DPDFNet. It was not checked.
- **VoxLock's model card says it was "trained on LibriSpeech and the AMI Meeting Corpus".** The subsets are not stated on the card, the README or the arXiv abstract.
  - If test-clean or test-other were used, M9 results on `corpus="libri"` scenes are optimistic.
  - Confirm with the author's training recipe before M9 decisions, and add a non-LibriSpeech check if possible.
- **WeSpeaker ResNet34 is trained on VoxCeleb only,** so it has no LibriSpeech overlap.
  - Sanity check on our enrollment samples (`tmp/build:data/emb_check.py`). It uses the fbank front end of `methods/_core/spk.py` (owned by build:spk, not yet validated against WeSpeaker), so treat the numbers as indicative:
    - Enrollment pairs: EER 0 %. Same-speaker cosine is at least 0.76 (mean 0.89); different-speaker cosine is at most 0.54 (mean 0.12).
    - 120 scene utterances (3 per target speaker) were identified among the 40 enrolled speakers with 100 % accuracy.
  - It took 0.65 s per embedding (single thread, 4–22 s clips).

## 4. LibriSpeech (`lab/corpus_libri.py`)

### Pools and splits (speaker level, disjoint)

| Pool | Source | Roles | dev | test |
|---|---|---|---|---|
| target | test-clean, 40 speakers | `partner`, `talker` (bus partners, meeting talkers) | 20: 10 F / 10 M, 1305 scene utterances, 2.55 h | 20: 10 F / 10 M, 1237 scene utterances, 2.54 h |
| babble | test-other, 33 speakers | `babble` (every bus interferer) | 16: 8 F / 8 M, 1517 utterances, 2.60 h | 17: 9 F / 8 M, 1422 utterances, 2.74 h |

How the splits are made:
- Gender comes from `SPEAKERS.TXT`.
- Within each pool and gender, speakers are sorted by median F0 and paired with their neighbour.
- A seeded coin flip per pair sends one speaker to dev and the other to test (seed 20261102), so both splits cover the same voice range. The odd speaker left over goes to test.
- `corpus_libri/splits.json` holds the lists.

| Split | Speakers |
|---|---|
| target dev | 1284 F, 2094 F, 237 F, 2830 M, 3729 F, 4446 F, 4507 F, 4992 F, 5105 M, 5639 M, 5683 F, 61 M, 7021 M, 7127 M, 7176 M, 8224 M, 8230 M, 8455 M, 8463 F, 8555 F |
| target test | 1089 M, 1188 M, 121 F, 1221 F, 1320 M, 1580 F, 1995 F, 2300 M, 260 M, 2961 F, 3570 F, 3575 F, 4077 M, 4970 F, 5142 F, 672 M, 6829 F, 6930 M, 7729 M, 908 M |
| babble dev | 1688 M, 2414 M, 3331 F, 3538 F, 367 F, 3764 F, 3997 F, 4198 M, 4294 F, 4350 M, 6128 F, 6432 M, 7902 M, 8131 M, 8188 M, 8280 F |
| babble test | 1998 F, 2033 M, 2609 M, 3005 M, 3080 F, 3528 F, 4852 M, 533 F, 5442 F, 5484 F, 5764 M, 6070 F, 6938 F, 7018 M, 7105 M, 7975 M, 8461 F |

F0 check of the `SPEAKERS.TXT` genders (median F0 over voiced 40 ms frames):
- **Target speakers are consistent.** F 161–216 Hz, M 69–150 Hz.
- **Three babble speakers labelled M have female-range F0:**
  - 1688: 203 Hz (dev);
  - 7105: 220 Hz (test);
  - 8188: 170 Hz, ambiguous (dev).
- The labels are kept as published. By voice, babble dev is about 10 F / 6 M and babble test 10 F / 7 M.
- The F0 pairing puts 1688 and 7105 in the same pair, so they ended up in different splits.

### Utterance table `corpus_libri/utterances.tsv`

Tab-separated, one header line, 5559 rows sorted by id. Load it with `load_table()`.

| Column | Meaning |
|---|---|
| `id`, `subset`, `speaker`, `gender`, `chapter` | LibriSpeech identity; gender from `SPEAKERS.TXT` |
| `pool`, `split` | `target` or `babble`; `dev` or `test` (the speaker's) |
| `use` | `scene`, or `enroll` (reserved, never in scenes) |
| `path` | FLAC path relative to the lab folder, inside `data/` |
| `duration_s` | File length (frames / 16000) |
| `active_s` | Energy-VAD active speech: 20 ms frames within 30 dB of the loudest frame **and** ≥ 10 dB above the noise floor (5th percentile of frame levels, digital silence excluded) |
| `active_rms_db` | RMS over the VAD-active frames, dB re full scale 1.0 (a full-scale sine is −3.01 dB) |
| `norm_rms_db` | `lab.common.active_rms` (−30 dB rule only): the level `lab.corpus2` divides every utterance by |
| `floor_db`, `snr_db` | Noise floor and active level minus floor |
| `peak_dbfs`, `clip_n` | Sample peak, and the number of samples with \|x\| ≥ 0.999 |
| `f0_hz`, `n_voiced` | Median F0 and the number of voiced frames |
| `text` | Transcript |

`corpus_libri/speakers.tsv` has one row per speaker:
- pool, split, gender, minutes in `SPEAKERS.TXT`;
- n_utts, total_s and active_s, plus scene-only counts;
- chapters and seconds per chapter, n_chapters;
- median F0 and the F0-implied gender;
- the enrollment chapter and `enroll_chapter_scene_share`.

Measured over the table:

| | target (test-clean) | babble (test-other) |
|---|---|---|
| Duration p5 / median / p95 / max | 2.3 / 5.8 / 17.8 / 35.0 s | 2.3 / 5.2 / 15.8 / 34.5 s |
| Active-speech fraction (median) | 0.71 | 0.67 |
| Active level p5 / median / p95 | −28.7 / −23.3 / −19.4 dB | −29.4 / −23.2 / −18.8 dB |
| SNR estimate p5 / median / p95 | 20.9 / 36.6 / 59.7 dB | 17.8 / 39.4 / 67.6 dB |
| Clipped files (`clip_n` > 0) | 11, all speaker 2094 (chapter 142345) | 0 |

`norm_rms_db` versus `active_rms_db`:
- In noisy files the −30 dB rule also counts noise frames, so `lab.corpus2`'s "unit active RMS" leaves speech slightly quieter.
- 9.4 % of target utterances differ by more than 1 dB and 1.2 % by more than 3 dB (max 5.5 dB; 7.1 dB in babble).

### Enrollment `corpus_libri/enrollment.json`

Each target speaker has 2 reserved samples, 80 in total, built from 83 utterances.
- **Active speech:** each sample has 8.0–14.8 s (median 10.0 s); file length 9.5–22.5 s.
- **Chapter:** samples are taken from the speaker's chapter with the least material that still offers two eligible utterances. Within it, the highest-SNR ones are chosen (SNR ≥ 21.6 dB, median 38.7 dB).
- **Clipped files** are never used. This replaced two clipped picks of speaker 2094.
- **Effect on scene material:**
  - For the 29 speakers with more than one chapter, the enrollment chapter holds a median 28 % of the remaining scene material (max 51 %, speaker 1580).
  - The other 11 speakers have a single chapter, so the share is 1.0 (1188, 2094, 2300, 3575, 3729, 4507, 5639, 672, 7729, 8230, 8455).
- **Speaker 61:** its longest utterance has only 7.3 s of active speech. Each of its two samples is therefore a run of 2 and 3 consecutive utterances from chapter 70970.
- **Exclusion from scenes:** the reserved utterances are absent from `corpora/libri/manifest.json`, so `lab.corpus2` and every scene generator cannot pick them. `tests/test_corpus_libri.py` checks this.
- **For M9 explicit enrollment:** `enrollment_audio(speaker)` returns the 2 samples. The scene's talker-to-speaker map is `scene["meta"]["talkers"][tid]["speaker"]`.

### Drop-in for the v2 scene core

`corpora/libri/manifest.json` is written exactly in the BENCH2.md "Scene format" drop-in format and location. `lab/corpus2.py` was not changed.
- Top level: `{"name": "libri", "source": "libri", "speakers": {...}, "utterances": [...]}`.
- Speakers carry `pool`, `split`, `gender`, `quality: "natural"` and `roles`.
- Utterances carry `id`, `speaker`, `text`, `path` (into `data/`) and `duration`.
- It holds 5476 scene utterances; the 83 enrollment utterances are left out.

```python
from lab import corpus2, corpus_libri as cl, scene2
c = corpus2.load_corpus("libri")                 # 73 speakers, 5476 utterances
c.speakers_for("partner", "dev")                 # 20 target dev speakers (same for "talker")
c.speakers_for("babble", "test")                 # 17 babble test speakers
sc = scene2.load_scene("dev/<name>", corpus="libri")   # makes <name>.libri.recipe.json on first use
enr = cl.enrollment_audio("1089")                # [2 x float32 (T,)], unit active RMS
rows = cl.load_table()                           # every utterance with VAD / level / chapter fields
```

Notes for scene generators:
- **Durations.** LibriSpeech utterances are long: median 5.4 s, up to 35 s.
- **demo_recipe short-utterance pool.** Target speakers 2961 (test) and 8224 (dev) have no utterance shorter than 3.5 s. `lab.scene2.demo_recipe` draws only from utterances shorter than 3.5 s, so its pool would be empty if it picked them (seed 3 does not).
- **Clipping.** Only speaker 2094 has clipped files: 11 scene utterances, 3–497 samples at full scale each. Filter on `clip_n` from the table if needed.

## 5. DEMAND (`lab/noise_real.py`)

### Format and geometry

Each environment is 16 mono WAV files of 300.004 s at 16 kHz.

**Documented layout** (`DEMAND.pdf` v1.0 §3.1, Fig. 1):
- 16 microphones (Sony ECM-C10 electret) in 4 staggered rows, built to ±2 mm.
- Every microphone is 5 cm from its immediate neighbours.
- The array is horizontal, on a tripod about 1.4 m above the floor.
- Channels are numbered row by row: 1–4, 5–8, 9–12, 13–16. Rows 2 and 4 are shifted by 2.5 cm, so rows are 4.33 cm apart and the largest distance is 21.8 cm.
- `design_xy()` gives the positions.

**Fitted from the recordings** (`fit_geometry()`, 300 s of all three environments):
- **Method:**
  - Per environment, the 30 % most diffuse 2 s blocks are kept: mean 1–6 kHz MSC over all pairs at or below its 30th percentile, and level at or above its 20th percentile.
  - Re(coherence) of all 120 pairs is fitted with ρ·sinc(2fd/c) over 150–7000 Hz.
  - The per-pair fits are combined with weights 1/rmse².
- **Grid:** the fitted grid spacing is **5.08 cm**. A two-parameter fit gives a column spacing of 5.08 cm and a row spacing of 4.40 cm (design 4.33).
- **Pair distances against the fitted grid:**
  - the 33 nearest-neighbour pairs are within **0.47 cm**;
  - all 120 pairs are within 1.34 cm (rms 0.32 cm).
- **Layout and numbering:**
  - Classical MDS of the fitted distances is planar (third/second eigenvalue 0.09).
  - It matches the layout to 0.26 cm mean (0.88 cm max) per microphone.
  - A blind, label-free match of the MDS points to the layout positions returns exactly the documented channel numbering.
  - Distances cannot distinguish the layout's 180° symmetry (channel k ↔ 17 − k), and they do not need to.
- **Per environment, mean fitted distance of the 5 cm pairs** (with the mean fit rmse and ρ over all 120 pairs):
  - TBUS 4.92 cm (rmse 0.074, ρ 0.98);
  - PCAFETER 5.47 cm (rmse 0.072, ρ 1.00);
  - OMEETING 4.98 cm (rmse 0.134, ρ 0.92; some long pairs off by up to 10 cm).
- **J0 (2-D diffuse model) fits worse everywhere.** For example, TBUS rmse is 0.116 against 0.074 for sinc, and J0 underestimates d by about 1 cm.
- **Orientation dependence.** The apparent 5 cm distance depends on the pair's orientation, differently in each environment:
  - TBUS: row pairs 5.23 cm, diagonals 4.68–4.81 cm;
  - OMEETING: 3.98 cm or 5.83 cm depending on the diagonal.

  The fields are anisotropic. A blind spacing estimate from noise coherence (S0 step 5) can be off by about ±5 % (TBUS, PCAFETER) to ±20 % (OMEETING) from orientation alone.

`pairs()` lists the 33 pairs at 5 cm (12 along rows, 21 diagonal), the closest DEMAND spacing to our 4 cm. Fitted distances in cm:

1-2 5.29, 1-5 5.00, 2-3 5.25, 2-5 4.80, 2-6 5.25, 3-4 5.46, 3-6 4.69, 3-7 4.99, 4-7 4.76, 4-8 5.09, 5-6 5.55, 5-9 5.26, 5-10 5.35, 6-7 5.43, 6-10 5.26, 6-11 5.33, 7-8 5.38, 7-11 4.99, 7-12 5.17, 8-12 4.83, 9-10 5.42, 9-13 4.85, 10-11 5.45, 10-13 4.94, 10-14 4.95, 11-12 5.48, 11-14 4.84, 11-15 5.05, 12-15 5.16, 12-16 4.96, 13-14 5.41, 14-15 5.25, 15-16 5.36.

**Channel sensitivities.**
- `GAIN_DB` holds the offsets: −0.65, 0.10, 0.73, −0.42, 1.89, 0.25, −0.15, 0.21, 1.70, −1.42, −1.15, −0.33, 0.29, −0.14, −0.51, −0.49 dB for ch01…ch16.
- Each is the median over the 3 environments and the 125–4000 Hz octave bands.
- They agree across environments (median difference 0.17 dB, max 0.76 dB), so they are capsule sensitivities, not field effects.
- `real_noise()` removes them by default, because the scene renderer adds its own E8 per-mic gain error.

### Noise-field report (`python -m lab.noise_real --analyze`, full 300 s)

Levels are digital: DEMAND is not SPL-calibrated (the microphone preamp gain was +20 dB). They are averaged in power over the 16 raw channels.

| | TBUS | PCAFETER | OMEETING |
|---|---|---|---|
| Leq, Z-weighted | −20.8 dBFS | −44.4 dBFS | −54.4 dBFS |
| Share of power below 20 Hz | **82 %** (peak near 13 Hz) | 22 % | 10 % |
| Leq, A-weighted | −47.1 dBFS(A) | −51.3 dBFS(A) | −59.4 dBFS(A) |
| Leq, 150–7000 Hz (the SNR-rule band) | −44.2 dBFS | −48.0 dBFS | −55.7 dBFS |
| 2 s block level, 150–7000 Hz, p10 / p50 / p90 | −51.5 / −47.8 / −42.5 | −51.2 / −49.0 / −46.6 | −66.7 / −59.0 / −52.0 |
| PSD tilt over 125–4000 Hz octave bands (pink = −3 dB/oct) | −5.9 dB/oct | −5.1 dB/oct | −6.1 dB/oct |
| Octave bands 63 / 125 / 250 / 500 / 1k / 2k / 4k Hz, dB re total | −16.5 / −25.5 / −29.6 / −26.5 / −33.3 / −36.2 / −40.1 | −12.6 / −11.3 / −9.4 / −6.6 / −13.7 / −17.0 / −19.9 | −16.2 / −8.5 / −6.0 / −4.9 / −10.6 / −18.0 / −21.7 |

Coherence of the 33 pairs at 5 cm against the 3-D diffuse model sinc(2fd/c), using all blocks (long term):

| Re(Γ) rmse vs sinc at 5 cm (J0 in brackets); mean \|Im Γ\| | TBUS | PCAFETER | OMEETING |
|---|---|---|---|
| 50–200 Hz | 0.007 (0.007); 0.05 | 0.006 (0.005); 0.04 | 0.081 (0.081); 0.04 |
| 200–500 Hz | 0.012 (0.009); 0.03 | 0.004 (0.010); 0.02 | 0.012 (0.014); 0.07 |
| 500–2000 Hz | 0.048 (0.121); 0.06 | 0.040 (0.096); 0.06 | 0.131 (0.140); **0.24** |
| 2000–4000 Hz | 0.093 (0.290); 0.09 | 0.121 (0.222); 0.09 | 0.196 (0.329); 0.13 |
| 4000–7000 Hz | 0.082 (0.185); 0.05 | 0.149 (0.169); 0.08 | 0.224 (0.258); 0.14 |
| 100–7000 Hz | **0.077** (0.206); 0.06 | 0.119 (0.169); 0.08 | 0.192 (0.253); 0.15 |
| Per-pair sinc fit: d mean (range), ρ, rmse | 4.96 (4.55–5.38) cm, 0.99, 0.066 | 5.44 (4.88–5.85) cm, 1.00, 0.095 | 4.74 (2.58–6.58) cm, 0.90, 0.133 |

Short-term diffuseness (`block_diffuseness()`, 2 s blocks × 33 pairs):
- Each block's Re-coherence rmse against sinc (at the pair's fitted distance) is compared with a **matched null**: Gaussian frames with the same per-frame, per-bin power and exactly sinc coherence.
- The matched null is needed because level fluctuations alone inflate the estimator's spread: about 0.12 instead of 0.055 for stationary noise.

| | TBUS | PCAFETER | OMEETING |
|---|---|---|---|
| Block rmse p10 / p50 / p90 | 0.121 / 0.163 / 0.229 | 0.131 / 0.177 / 0.244 | 0.192 / 0.299 / 0.437 |
| Matched null p50 / p95 | 0.119 / 0.187 | 0.115 / 0.157 | 0.128 / 0.177 |
| Share beyond the null p95 (5 % if ideally diffuse) | **29 %** | 67 % | 95 % |
| MSC 2.5–6 kHz p50 (null p50 / p95), share beyond p95 | 0.107 (0.064 / 0.116), 41 % | 0.125 (0.063 / 0.096), 74 % | 0.148 (0.067 / 0.112), 73 % |

**Does real bus noise follow the diffuse model the methods assume?**
- **Long term, yes, well.**
  - Over 5 minutes the TBUS 5 cm coherence follows the 3-D sinc model: mean Re(Γ) within 0.025 of sinc in every band, rmse 0.08 over 0.1–7 kHz, \|Im Γ\| ≈ 0.05, ρ ≈ 0.99.
  - It follows sinc much better than J0, and better than the cafeteria and the meeting room do.
- **Short term, mostly.** Once level fluctuations are accounted for, 60–70 % of 2 s block-pair cases are consistent with an ideal diffuse field: 29 % lie beyond the null's 95th percentile by coherence rmse and 41 % by 2.5–6 kHz MSC, against 5 % expected. The rest carry directional events.
- **But the bus noise has almost no spatial leverage at our scale.**
  - 82 % of its power is infrasound below 20 Hz.
  - About 90 % of the power above 45 Hz is below 350 Hz (63–250 Hz octave bands). Below 500 Hz any 4–5 cm pair is coherent (sinc ≥ 0.96) whatever the field.
  - Spatial post-filters therefore cannot remove it; DESIGN §1.6 item 5 already says this for engine noise.
- **PCAFETER** has a sinc-like long-term coherence but at an apparent 5.4 cm: a partly horizontal (2-D) field of talkers near the array height. It is non-diffuse in 67 % of block-pair cases (nearby talkers).
- **OMEETING** fits neither model: speech discussing the array is directional (\|Im Γ\| 0.24 at 0.5–2 kHz).

### Using real noise in scenes

```python
from lab import noise_real as nr
x = nr.real_noise("TBUS", n_samples, rng)       # float32 (2, n): a random 5 cm pair, random order and offset
x, info = nr.real_noise("PCAFETER", n, 7, pair=(5, 6), return_info=True)   # info: pair, segments, distances, gains
nr.pairs()                                       # the 33 pairs at 5 cm with design / fitted distance and angle
src = nr.field_source("TBUS", 72.0)              # recipe source: {"type": "field", "weighting": "A", ...}
```

- **`real_noise(env, n_samples, rng, pair=None, equalize=True, xfade_s=1.0, hp_hz=None, return_info=False)`**
  - Deterministic for a given rng (Generator or int).
  - With `pair=None` it draws one of the 33 pairs at 5 cm and a random order. `pair=(a, b)` uses DEMAND channel numbers 1–16, with row 0 = Mic A = channel a.
  - The offset is uniform. Requests longer than 300 s chain random segments with 1 s equal-power crossfades, so the content repeats.
  - `hp_hz` applies an optional zero-phase 4th-order high-pass, identical on both rows, so coherence is unchanged.
- **`field(T, fs, rng, env, pair, equalize, hp_hz)`** is the scene2 `field` signal (kind `func`). Mic A has RMS 1.
- **`field_source(env, level_db, pair=None, weighting="A", hp_hz=None)`** builds the recipe source dict. Use A-weighting or `hp_hz`: TBUS's Z-weighted level is infrasound. `tests/test_noise_real.py` renders such a source through `lab.scene2.render` at the requested level.
- **Spacing mismatch.** The pair carries the coherence of its own ≈5 cm spacing, while the scene geometry may be 2.5, 4 or 6 cm. A 4 cm device would have its first coherence zero at 4.3 kHz instead of 3.4 kHz, and blind spacing estimates (S0) on such scenes will be pulled towards 5 cm. Keep this in mind when E6(a) scenes are scored on calibration accuracy.
- **Recording conditions.** The DEMAND array was horizontal at 1.4 m on a tripod. Our device is handheld or on a seat in the bus and on a table in the meeting.

## 6. Reproduce and test

| Command | What it does | Time |
|---|---|---|
| `.venv/bin/python -m lab.corpus_libri --build` | Measures all 5559 utterances (3 worker processes) and rewrites `corpus_libri/*` and `corpora/libri/manifest.json`. Deterministic | about 20 s |
| `.venv/bin/python -m lab.corpus_libri --summary` | Counts and hours per pool and split | < 1 s |
| `.venv/bin/python -m lab.noise_real --analyze [--json PATH]` | Geometry fit and the report above for all three environments | about 1 min |
| `.venv/bin/python -m tests.test_corpus_libri` | 9 tests: files, table against disk, durations against audio, reproducible VAD, disjoint and balanced splits, enrollment reserved and absent from the corpus, drop-in loads in corpus2, scene2 demo rendered with `corpus="libri"` | about 3 s |
| `.venv/bin/python -m tests.test_noise_real` | 9 tests: files, pair listing, geometry re-fitted from 120 s of TBUS and PCAFETER (5 cm pairs within ±1 cm of the fitted grid, blind numbering), sinc better than J0 on the bus, determinism, content and gains, crossfaded long requests, field level and high-pass, scene2 `field` render | about 7 s |

Evidence kept in `tmp/build:data/`:
- `demand_report.json`: the full `--analyze` output;
- `emb_check.py`: the WeSpeaker enrollment check, about 2.5 min.
