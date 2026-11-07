# Speaker → ESP32 hardware tests

## Session 1: 2026-11-03, under load
- **Playback:** MacBook Pro built-in stereo speakers at system volume 75 %. The test sounds play on the left speaker (`_L`), the right speaker (`_R`), or both.
- **Capture:** ESP32-C3 with 2 × INMP441. ch0 = Mic A, ch1 = Mic B; 16 kHz, 24-bit.
- **Timing:** each sound lasts 4.2 s or less, inside a 12 s recording.
- **Machine state:** the two build runs were active, with load average 150–200. The laptop fans were loud, and they shaped every result below.

### Files
| Path | Content |
|---|---|
| `hw/esp32.py` | Capture over the `mic_recorder` serial protocol. `play_record()` starts `afplay` when the board confirms recording. |
| `hw/stimuli.py` | Builds the test sounds into `hw/stim/` (`<name>.wav` played at 48 kHz, `<name>.npz` 16 kHz references) |
| `hw/session.py` | Runs the session and writes `hw/rec/<name>.wav` (float32, unscaled, DC kept) plus `.json` metadata |
| `hw/analyze.py` | Produces `hw/analysis.json` and `hw/irs.npz` (room responses, 5 ms before to 600 ms after the direct sound) |

### Sounds and recordings
| Recording | What played |
|---|---|
| `ambient` | Nothing |
| `sweep_L`, `sweep_R`, `sweep_L2` | 3 s exponential sweep, 50–7800 Hz, −10 dBFS (`sweep_L2` repeats `sweep_L`) |
| `talkA_L` | LibriSpeech 121 (F, test split, has enrollment clips) |
| `talkB_R` | LibriSpeech 1089 (M, test split, has enrollment clips) |
| `talkAB_LR` | A and B at the same time |
| `noise_LR` | DEMAND TBUS ch01 on the left, ch04 on the right |
| `talkA_noise` | A plus the bus noise |
| `talkC_R_quiet` | LibriSpeech 3575 (F) at −15 dB |

Speech is set to −22 dBFS active RMS with a peak of −6 dBFS or less. Bus noise is −24 dBFS RMS.

### Findings (numbers in `hw/analysis.json`)
1. **The capture chain is sound.**
   - Both mics are alive.
   - No samples were dropped: the per-window sub-sample lag moves smoothly, with a largest step of 0.47 samples.
   - The Mac and ESP32 clocks differ by 17 ppm (one sample per about 4 s). This only matters for lab references.
2. **Mic B is about 1 dB less sensitive than Mic A.**
   - The direct sound at B re A is −0.3 to −1.9 dB per band, with the same pattern for both speakers and for the repeat.
   - S0's gain calibration has to remove this; DESIGN's ±1 dB impairment covers it.
3. **The two mics are offset in time by about half a sample.**
   - The direct-sound time difference is −0.53, −0.56 and −0.55 samples for left, right and the left repeat (B hears first, about 34 µs).
   - Two speakers about 25 cm apart give the same value. That fits the expected half-frame I2S slot offset (31.25 µs, DESIGN E-factor "skew"), with both speakers near broadside.
   - It cannot be separated from geometry with this setup. One endfire sweep, or a diffuse-field phase measurement in a quiet room, would settle it.
4. **The loudest noise was the laptop fans.**
   - Ambient level after the 80 Hz high-pass is −51 dBFS (about 69 dB SPL equivalent).
   - It is strongly coherent (MSC 0.97–0.99 at 0.5–2 kHz) and arrives from one direction: B hears it 1.15 samples first, a different direction from the speakers.
   - Per-mic noise below 60 Hz is incoherent (MSC 0.04) at −35 dBFS. This is electrical or mechanical, not acoustic, and the 80 Hz high-pass in S0 is essential.
5. **The room has RT60 of about 0.4–0.5 s.**
   - DRR is 6 dB from the left speaker and 0 dB from the right.
   - The responses are 29–49 dB above their noise floor.
6. **The room changes between takes.** Comparing sweep_L with its repeat, after sub-sample alignment:
   - direct sound matches to −28 dB;
   - reflections at 5–50 ms only to −11 dB;
   - the late tail only to −3.5 dB.

   As a result, a mixture recording equals the sum of its separately recorded parts only to −10 dB (both mixtures tested). Recording-based references are therefore good to about 10 dB SDR. Hagerman or superposition scoring on real takes would hit the same ceiling.
7. **The speakers limit what can be tested.**
   - Talker A is 11 dB above the fan noise and talker B 7 dB. Talker C at −15 dB is below it.
   - The played bus noise is 20 dB below the fan noise over 150–7000 Hz, because MacBook speakers cannot reproduce bus rumble.
   - So hardware bus tests must add real bus noise digitally, using the multichannel DEMAND recordings from 5 cm pairs, rather than through the speakers.
8. **Late-tail coherence measures the fans here, not diffuse reverb.**
   - The tail is coherent (0.98) with the fans' time difference, so a diffuse-field spacing fit is invalid in this session.
   - The mic spacing is still unknown; the bench assumes 4 cm.

### Implications
- S0 must estimate the gain mismatch (about 1 dB) and the skew (about 0.5 sample); both are already in DESIGN.
- A strong directional machine noise next to the device is a realistic case. In the bus pipeline the spatial stage can null it when its direction differs from the partners' (here 0.6 samples apart).
- For the hardware tests, score with these:
  1. single-take noise-only stretches, for noise reduction;
  2. a direct-sound reference (speech convolved with the stable first 5 ms of the measured response), for distortion;
  3. listening.

  Treat superposition-based SDR as accurate only to about 10 dB.

## Session 2 (planned: after the builds finish, fans quiet)
- Repeat the sweeps. In a quiet room the late reverb approximates a diffuse field, which gives the mic spacing and skew from the coherence and its phase.
- Record the bus-partner conversation and a meeting-style sequence of talkers. Add DEMAND bus noise and babble digitally.
- Run the final bus and meeting enhancers, and produce before/after audio.
- Keep each sound at 4.5 s or less.

## Superseded (2026-11-03): see `hw/SPACING.md`
- **Findings 3 and 8 are superseded by `hw/SPACING.md`.** The −0.53 to −0.56-sample values are not a half-frame skew with broadside speakers: they combine a one-frame skew (≈ −1.1) with the geometric delay of speakers near Mic A's endfire (≈ +0.6). The channel skew is about one I2S frame (Mic B early: −1.12 samples in this session, −1.01 to −1.03 after later resets), and the spacing is about 1.3 cm (1.2–1.5), not 4 cm. The late tail, fans included, is near-diffuse and carries the same skew, so its coherence is usable, but at this spacing it gives only an isotropic-equivalent spacing.
- For the same reason, finding 4's "arrives from one direction" (its −1.15 samples is the channel skew) and the skew of "about 0.5 sample" under Implications are superseded too.
- Session 2: take the spacing from calipers and two endfire sweeps in one boot, not from diffuse coherence (`hw/SPACING.md` §5).
