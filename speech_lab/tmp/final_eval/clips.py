"""Build the clip list JSON for the report page from report_page/audio (+ metrics)."""
import json, os, sys
import soundfile as sf
A = "report_page/audio"
def dur(f):
    return round(sf.info(os.path.join(A, f)).duration, 2)
def row(label, kind, f):
    stem = os.path.splitext(f)[0]
    return {"label": label, "kind": kind, "src": f"audio/{f}", "spec": f"spec/{stem}.png", "dur": dur(f)}
extra = json.load(open("tmp/final_eval/sim_bus_clip_raw.json")) if os.path.exists("tmp/final_eval/sim_bus_clip_raw.json") else None
if extra:
    extra["rows"] = [row(r["label"], r["kind"], r["f"]) for r in extra["rows"]]
clips = [
 {"title": "Real board · bus · partners in line with the mics", "tags": [{"text": "real recording", "cls": "ok"}, {"text": "0 dB SNR", "cls": "na"}, {"text": "fan noise as recorded", "cls": "na"}],
  "caption": "Two LibriSpeech talkers played from the MacBook toward Mic A's end, the real fan noise, plus simulated passengers and bus noise. Listen for the passengers: the axis beam pushes them back; the AI-only pipeline only removes the steady noise.",
  "rows": [row("Mic A as recorded", "raw", "real_bus_micA.wav"), row("Axis beam + DPDFNet (bus-endfire)", "out", "real_bus_endfire.wav"), row("Bus v3, no beam (bus)", "out", "real_bus_v3.wav")],
  "metrics": [{"k": "worst partner ESTOI", "v": "+0.110 axis · +0.005 v3"}, {"k": "SNR gain", "v": "+8.7 dB · +3.2 dB"}, {"k": "passenger SIR", "v": "+8.0 dB · +2.5 dB"}]},
]
if extra:
    clips.append(extra)
clips += [
 {"title": "Real board · meeting · eight talkers", "tags": [{"text": "real recording", "cls": "ok"}, {"text": "fan noise as recorded", "cls": "na"}],
  "caption": "Eight talkers from 0 to −15 dB, two of them pre-reverberated to sound far away. This is the hardest real scene (loudest fans). Listen for the fan noise dropping and the quiet talkers coming up; intelligibility gains here are small.",
  "rows": [row("Mic A as recorded", "raw", "real_meet_micA.wav"), row("Meeting v3 (meeting)", "out", "real_meet_v3.wav"), row("Meeting v3, no neural model (meeting-classic)", "out", "real_meet_classic.wav")],
  "metrics": [{"k": "average talker ESTOI", "v": "+0.019 v3 · +0.023 classic"}, {"k": "noise removed", "v": "7.0 dB · 2.7 dB"}]},
 {"title": "Simulated meeting · median held-out room", "tags": [{"text": "simulation", "cls": "part"}, {"text": "20 talkers", "cls": "na"}, {"text": "30 s excerpt", "cls": "na"}],
  "caption": "Room r03 of the held-out test (the median by worst-talker gain), 66–96 s: ten talkers between 1.5 and 5.5 m from the device. Listen for the far talkers getting louder and drier, and the levels evening out.",
  "rows": [row("Mic A as recorded", "raw", "sim_meet_micA.wav"), row("Meeting v3", "out", "sim_meet_v3.wav"), row("Meeting v3, no neural model", "out", "sim_meet_classic.wav")],
  "metrics": [{"k": "whole room, worst talker", "v": "+0.056 v3"}, {"k": "average talker", "v": "+0.118"}]},
]
print(json.dumps(clips, ensure_ascii=False))
