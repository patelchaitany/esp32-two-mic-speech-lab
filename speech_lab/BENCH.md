# 2-mic speech lab — bench contract

Hardware simulated: ESP32-C3 + 2x INMP441 (omni MEMS), sample-synchronous, 16 kHz. Row 0 = Mic A, row 1 = Mic B.
True spacing in scenes: 25 / 40 / 60 mm, random array orientation. Methods are told a NOMINAL spacing of 40 mm
(the user's estimate) — robust methods must tolerate the mismatch. Mic gains differ by up to +/-1 dB.

## Method API
`methods/<name>.py` must define `enhance(x, fs, info) -> y`:
- `x`: float32 array (2, T); `fs` = 16000; `info = {"scenario": "bus"|"meeting", "spacing_m": 0.04}`.
- return mono float array of length T, time-aligned with the input (no added latency beyond a few samples;
  the scorer aligns up to +/-100 ms but keep it aligned).
- Use ONLY x, fs, info. Never read scene files, references, positions or segments.
- Offline processing is fine (whole file available); prefer algorithms that could run online/causally.
- Keep runtime real-time factor < 0.5 on this Mac (a 30 s file should process in < 15 s).
- Allowed deps: numpy, scipy, pyroomacoustics, onnxruntime, soundfile, and `lab.dsp` helpers
  (`stft`/`istft` with sqrt-Hann WOLA, perfect reconstruction; `gcc_phat`; `diffuse_coherence`; `recursive_avg`).
  Do NOT pip install anything into the shared venv.

## Run
    .venv/bin/python -m lab.bench --methods <name>[,<name2>] --set dev [--only bus] [--save-audio]
Prints a table; per-scene JSON (and WAV with --save-audio) under results/<set>/<name>/.
Python: `.venv/bin/python` (3.12). Working dir: the speech_lab folder.

## Scenes (dev set: 13 bus + 5 meeting; held-out "test" set exists — do not tune on it)
- BUS: 11.5 x 2.5 x 2.3 m cabin, RT60 0.35-0.6 s. Two target talkers 0.3-0.8 m from the device on roughly
  opposite sides, alternating (Lombard, 66-70 dB SPL @1 m). 16-25 babble passengers >= 1 m away talking
  intermittently, engine at the rear, diffuse road noise. In-band SNR -5 / 0 / +5 dB at Mic A.
- MEETING: ~12 x 7 x 3 m room, RT60 0.5-0.9 s, device at the centre of an 8 x 2 m table, 20 seated talkers
  1.2-4.6 m away speaking in turn (15% overlaps, 57-66 dB SPL @1 m), HVAC diffuse noise 40-45 dBA, projector fan.

## Metrics (higher is better unless noted)
BUS: dSISDR = SI-SDR improvement over Mic A (dB, ref = both targets' direct+50 ms image at Mic A);
ESTOI (0-1); wrstT = ESTOI of the weaker of the two targets (both must survive); PESQ-WB;
BR = background reduction (dB) = target gain minus gain in target-silent regions (gain-invariant, capped at 40).
MEETING: ESTOI mean / worst talker; dSISDR (per-talker mean, ref direct+50 ms); loudσ = std of per-talker
output loudness in dB (LOWER is better, even capture); gapSNR = median talker level minus noise floor (dB); PESQ-WB.
Baselines: mic_a (bus ESTOI 0.398, BR 0; meeting ESTOI 0.650, worst 0.476, loudσ 2.6, gapSNR 14.8), avg.
