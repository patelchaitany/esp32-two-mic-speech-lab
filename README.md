# ESP32 two-mic speech lab

Speech enhancement for an ESP32-C3 with two INMP441 microphones on one I2S bus (16 kHz), for two use cases:

- **Bus:** keep the two people talking near the device, suppress the bus and other passengers.
- **Meeting:** capture all of about 20 talkers in a large room, at even levels.

## What's here

| Path | Content |
|---|---|
| `mic_recorder/` | ESP32-C3 firmware: streams both mics over USB serial. SCK = GPIO4, WS = GPIO5, SD = GPIO6. Mic A (L/R to GND) is the left channel, Mic B (L/R to 3V3) the right. |
| `record.py` | Records a stereo WAV from the board. |
| `speech_lab/enhance.py` | Runs the final pipelines on a recording (see below). |
| `speech_lab/methods/` | All methods: self-calibration (`x_s0v2.py`), beams, masks, WPE, leveling, neural post-filters, and the final pipelines `x_bus_v3*.py`, `x_meeting_v3*.py`. |
| `speech_lab/lab/` | Simulation bench: scene generation, metrics, statistics, CLI (`python -m lab.bench2`). |
| `speech_lab/hw/` | Board measurements: spacing and skew (`SPACING.md`), hardware tests, the real-device test set and its results (`realset/RESULTS_REAL.md`, `realset/BIZARRE.md`). |
| `speech_lab/study/` | 12 mm vs 40 mm spacing study (`SPACING_STUDY.md`) and how the final pipelines were chosen (`final/BUS_V3.md`, `final/MEETING_V3.md`). |
| `speech_lab/results3/` | Per-scene results of the held-out tests and the spacing study (JSON). |
| `speech_lab/report_page/` | The results report with before/after audio. Open `index.html` through a local web server. |
| `speech_lab/DESIGN.md`, `BENCH2.md`, `DATA.md` | Design, bench and data documentation. |

## Results (3 Nov 2026)

- **Meeting: mostly works.** On 8 held-out simulated rooms, `x_meeting_v3` raised average-talker intelligibility (ESTOI) by +0.12 and the worst talker's by +0.06, and cut the level spread from 3.9 to 1.5 dB. On the real board the gain is about a third.
- **Bus: half works.** `x_bus_v3` gives +5.4 dB SNR and +0.03 worst-partner ESTOI on the held-out test, but nearby passengers stay as intelligible as before. On the real board, a fixed beam along the mic axis plus DPDFNet gave +0.12, with the partners in line with the mics.
- **The board:** the mics are about 1.3 cm apart. Mic B is sampled one I2S frame (62.5 µs) late, plus 0–7.5 µs that changes at each boot, so the pipelines self-calibrate from room noise.
- **Spacing:** moving the mics to 4 cm adds little (+0.004 ESTOI bus, +0.013 meeting).

## Setup

Python 3.12:

```bash
python3 -m venv speech_lab/.venv
speech_lab/.venv/bin/pip install -r requirements.txt
```

The datasets (LibriSpeech test-clean/test-other, DEMAND) and the ONNX models (GTCRN, DPDFNet, VoxLock, WeSpeaker) are not in the repo. `speech_lab/DATA.md` lists the sources, licences and expected paths (`speech_lab/data/`, `speech_lab/models/`).

## Use

```bash
python3 record.py -o ~/voice.wav
cd speech_lab
.venv/bin/python enhance.py ~/voice.wav -m meeting
```

Modes: `meeting`, `meeting-classic`, `bus`, `bus-classic`, `bus-endfire`. Each run prints the calibration it measured. Keep at least 0.5 s without speech in the recording.

## Not included

Downloaded datasets and models, raw board recordings and rendered scene audio (all `.wav`/`.npz` except the report's clips), earlier bench results (`results/`, `results2/`), and tuning scratch (`tmp/`).
