p = "report_page/index.html"
s = open(p).read()
svg = open("tmp/final_eval/drift.svg").read()
R = {}
R["TBD_DRIFT_FIG"] = f"""<figure>
      <div class="svgbox">{svg}</div>
      <figcaption>Mic B's skew estimated from 10 s of room noise: first boot (copper), the 8 resets and later ambient takes (dark), and the drift monitor every 10 minutes (green). Sweeps from the same speaker at 01:55 and 03:26 measured the same direct-path delay, so the wander is in the noise-based estimate, not the board.</figcaption>
    </figure>"""
R["TBD_EXP_DRIFT"] = """<article class="exp"><p class="q">Does the skew drift overnight?</p>
        <p>No. Two sweeps from the same speaker 1.5 hours apart gave the same direct-sound delay. Noise-based estimates every 10 minutes wandered a little as the fan noise changed, well inside the ±5 µs target.</p>
        <p class="ans">direct path −0.516 sample at 01:55 and at 03:26 · 27 noise readings 02:05–06:25: −0.979…−1.032</p></article>"""
R["TBD_MEETING"] = """<div class="prose">
      <p>The meeting pipeline cleans and levels everyone instead of picking talkers. It removes reverberation (WPE), steers a beam toward whoever is speaking, suppresses noise with a small neural network (GTCRN), and gives each speech segment its own gain so quiet and far talkers come up to the same level. Self-calibration runs first.</p>
    </div>
    <h3>Held-out simulated test</h3>
    <div class="tblwrap"><table>
      <thead><tr><th>Pipeline</th><th class="n">Worst talker</th><th class="n">Average talker</th><th class="n">Far (&gt; 4 m)</th><th class="n">Level spread</th><th class="n">Talkers captured</th><th>Do-no-harm gate</th></tr></thead>
      <tbody>
        <tr><td>Mic A as recorded</td><td class="n">0</td><td class="n">0</td><td class="n">0</td><td class="n">3.9 dB</td><td class="n">55 %</td><td><span class="chip ok">pass</span></td></tr>
        <tr><td>GTCRN only<span class="sub">neural denoiser on Mic A</span></td><td class="n">+0.036</td><td class="n">+0.074</td><td class="n">+0.072</td><td class="n">4.1 dB</td><td class="n">53 %</td><td><span class="chip no">fail</span><span class="sub">noise pumping 5.9 dB</span></td></tr>
        <tr class="hl"><td><strong>Meeting v3</strong> (chosen)<span class="sub">x_meeting_v3</span></td><td class="n">+0.061<span class="sub">+0.052…+0.069</span></td><td class="n">+0.120<span class="sub">+0.114…+0.127</span></td><td class="n">+0.120<span class="sub">+0.113…+0.128</span></td><td class="n">1.5 dB</td><td class="n">93 %</td><td><span class="chip part">borderline</span><span class="sub">noise modulation 5.4 dB, CI up to 6.3 (limit 6)</span></td></tr>
        <tr><td>Meeting v3, no neural model<span class="sub">x_meeting_v3_classic</span></td><td class="n">+0.056<span class="sub">+0.051…+0.061</span></td><td class="n">+0.092<span class="sub">+0.086…+0.098</span></td><td class="n">+0.090<span class="sub">+0.084…+0.097</span></td><td class="n">1.5 dB</td><td class="n">92 %</td><td><span class="chip ok">pass</span></td></tr>
        <tr><td>Oracle<span class="sub">uses the clean speech; the ceiling</span></td><td class="n">+0.264</td><td class="n">+0.174</td><td class="n">+0.189</td><td class="n">4.0 dB</td><td class="n">55 %</td><td><span class="chip na">n/a</span></td></tr>
      </tbody></table></div>
    <p class="note">Change in ESTOI, an intelligibility score from 0 to 1, against Mic A; ranges are paired 95 % confidence intervals. 8 simulated rooms with 20 LibriSpeech talkers each, none used for tuning. Level spread is the standard deviation of talker levels (target 1.5 dB; both v3 pipelines sit right at it). Talkers captured: within ±3 dB of the median level and not less intelligible. On the 20 tuning rooms v3 scored +0.075 / +0.125 / +0.118, so it did not overfit.</p>
    <h3>Real board</h3>
    <div class="tblwrap"><table>
      <thead><tr><th>Pipeline</th><th class="n">Worst talker</th><th class="n">Average talker</th><th class="n">Far talkers</th><th class="n">Level spread</th><th class="n">Talkers captured</th></tr></thead>
      <tbody>
        <tr><td>Mic A as recorded</td><td class="n">0</td><td class="n">0</td><td class="n">0</td><td class="n">4.2 dB</td><td class="n">62 %</td></tr>
        <tr class="hl"><td><strong>Meeting v3</strong></td><td class="n">+0.023</td><td class="n">+0.038</td><td class="n">+0.004</td><td class="n">3.5 dB</td><td class="n">75 %</td></tr>
        <tr><td>Meeting v3, no neural model</td><td class="n">+0.015</td><td class="n">+0.028</td><td class="n">+0.013</td><td class="n">2.7 dB</td><td class="n">75 %</td></tr>
        <tr><td>Axis beam + DPDFNet<span class="sub">only for talkers on the mic axis</span></td><td class="n">+0.006</td><td class="n">+0.076</td><td class="n">+0.022</td><td class="n">5.4 dB</td><td class="n">71 %</td></tr>
      </tbody></table></div>
    <p class="note">3 scenes: 8 LibriSpeech talkers played from the MacBook at 0 to −15 dB, two of them pre-reverberated to sound 4 m away; real fan noise as recorded, 12 dB quieter, and quieter with added meeting babble. Self-calibration from the recording itself gave the same numbers as the measured calibration (+0.039 average). Gains are smaller than in simulation because the fans were loud and the "far" talkers still came from the nearby laptop speakers, so dereverberation had nothing to remove.</p>"""
R["TBD_SPACING"] = """<div class="prose">
      <p><strong>Not needed for the bus; a small gain for the meeting.</strong> The same simulated scenes were rendered at 12 mm and at 40 mm and run through the same methods, so the difference isolates the spacing.</p>
    </div>
    <div class="tblwrap"><table>
      <thead><tr><th>Measure</th><th class="n">12 mm</th><th class="n">40 mm</th><th class="n">40 − 12 mm</th></tr></thead>
      <tbody>
        <tr><td>Bus, best system (CDR → GTCRN), worst-partner ESTOI change</td><td class="n">+0.025</td><td class="n">+0.029</td><td class="n">+0.004<span class="sub">−0.000…+0.008</span></td></tr>
        <tr><td>Bus, same system, SNR gain</td><td class="n">+5.5 dB</td><td class="n">+5.7 dB</td><td class="n">+0.19 dB<span class="sub">−0.18…+0.56</span></td></tr>
        <tr><td>Bus, best classical beam (RTF-MVDR), SNR gain difference</td><td class="n"></td><td class="n"></td><td class="n">+0.48 dB<span class="sub">+0.13…+0.83</span></td></tr>
        <tr><td>Meeting, steered beam, average ESTOI change</td><td class="n">0<span class="sub">falls back to Mic A in 4/4 rooms</span></td><td class="n">+0.022</td><td class="n">+0.022<span class="sub">+0.019…+0.026</span></td></tr>
        <tr><td>Meeting, best system at each spacing, average ESTOI change</td><td class="n">+0.080</td><td class="n">+0.093</td><td class="n">+0.013<span class="sub">+0.009…+0.017</span></td></tr>
      </tbody></table></div>
    <div class="prose">
      <ul>
        <li><strong>Why 12 mm loses direction.</strong> Delays between the mics shrink to about 45 % of their true value in reverberant noise (85 % at 40 mm), so the direction tracker called the two bus partners inseparable in 9 of 10 layouts (6 of 10 at 40 mm).</li>
        <li><strong>Physics.</strong> 4 cm gives a beam about 1 dB more (+0.8–1.0 dB bus, +0.7–1.1 dB meeting), 3× finer direction resolution and about 6× less damage from the per-boot skew. 25 mm is within 0.1–0.3 dB of 40 mm.</li>
        <li><strong>Neural denoisers don't care.</strong> GTCRN on Mic A alone gained +0.024 / +0.026 at 12 / 40 mm.</li>
        <li><strong>Recommendation:</strong> keep 1.2 cm for now. If you revise the board, use 25–40 mm on one rigid strip, then calibrate; it helps the meeting and makes calibration more forgiving.</li>
      </ul>
    </div>"""
R["TBD_NEXT"] = """<ol class="steps">
      <li><div><strong>Tap test and calipers (5 minutes).</strong> While recording, tap next to the mic whose L/R pin goes to GND: channel 0 should jump by more than 10 dB. If channel 1 jumps, the labels are swapped and the one-frame skew comes from how the ESP32 pairs the I2S slots. Then measure the port spacing.<span class="val">Settles which mic is which and why Mic B is one frame late.</span></div></li>
      <li><div><strong>Calibrate at every start.</strong> Use 0.5–1 s of room noise after each boot to estimate the skew; store only the integer frame, Mic B's gain and the caliper spacing. <code>enhance.py</code> already does this per file.<span class="val">Without it, the beams in these pipelines remove the talker instead of the noise.</span></div></li>
      <li><div><strong>Record real people.</strong> A 3–5 minute meeting around a table, and a two-person conversation with other talkers nearby. Put talkers on Mic B's side and side-on to the board too; tonight's speakers sat only on Mic A's side.<span class="val">The real-board numbers rest on two speaker positions.</span></div></li>
      <li><div><strong>Decide the bus approach.</strong> With this board, try <code>bus-endfire</code> with Mic A's end pointed at the partners. If both partners can't be in line with the mics, two mics 1.2 cm apart won't separate them from passengers; plan 3 or more mics, a wider array, or close-talk mics.<span class="val">This is the one requirement no method met in simulation.</span></div></li>
      <li><div><strong>Add a speech-recognition test.</strong> Word error rate is still pending in the gates; it decides whether the neural denoisers help or hurt transcription.<span class="val">Intelligibility scores and transcription accuracy can disagree.</span></div></li>
    </ol>"""
R["TBD_RUN"] = """<div class="prose"><p>Record with your <code>record.py</code> as before (left = Mic A, right = Mic B), then from the lab folder:</p></div>
    <div class="cmd"><pre>.venv/bin/python enhance.py ~/voice.wav -m meeting</pre><button type="button">Copy</button></div>
    <div class="cmd"><pre>.venv/bin/python enhance.py ~/voice.wav -m bus-endfire</pre><button type="button">Copy</button></div>
    <div class="cmd"><pre>afplay ~/voice_meeting.wav</pre><button type="button">Copy</button></div>
    <div class="prose">
      <ul>
        <li>Modes: <code>meeting</code>, <code>meeting-classic</code>, <code>bus</code>, <code>bus-classic</code>, <code>bus-endfire</code>. Each run prints the board calibration it measured from the recording; keep at least 0.5 s without speech in it.</li>
        <li>Calibration only: <code>.venv/bin/python -m methods.x_s0v2 file.wav</code>.</li>
        <li>Write-ups: <code>hw/SPACING.md</code> (board), <code>hw/realset/RESULTS_REAL.md</code> and <code>BIZARRE.md</code> (real recordings), <code>study/SPACING_STUDY.md</code>, <code>study/final/BUS_V3.md</code> and <code>MEETING_V3.md</code> (pipeline choices). Held-out results are in <code>results3/final/test/</code>.</li>
      </ul>
    </div>"""
R["TBD_METHOD"] = """<div class="prose">
      <ul>
        <li><strong>Real board.</strong> ESP32-C3 with two INMP441 mics, 16 kHz, 24-bit over USB serial. LibriSpeech speech played from the MacBook speakers in clips of 4.2 s or less. A clean reference was fitted from each recording, so real takes can be scored. Bus noise and passengers were added digitally, because laptop speakers can't reproduce bus rumble. The build jobs kept the laptop fans loud all night (−41 to −59 dBFS), so each real scene exists with the fans as recorded and 12 dB quieter.</li>
        <li><strong>Simulation.</strong> Simulated rooms with LibriSpeech speech and DEMAND noise. Bus: 30 tuning layouts and 30 held-out layouts at −5 and 0 dB SNR. Meeting: 20 tuning rooms and 8 held-out rooms with 20 talkers. Pipelines were chosen on the tuning sets only.</li>
        <li><strong>Scores.</strong> ESTOI intelligibility (0–1), SNR gain during speech, passenger suppression, level spread across talkers, each against Mic A with paired 95 % bootstrap intervals. Gates: G0 means do no harm, G1 means a meaningful gain.</li>
        <li><strong>Not done.</strong> No word-error-rate test (no speech recognizer downloaded), no listening panel, no real people. The overnight run stopped at 06:27 before its final tests; the held-out tests and the real-board runs of the final pipelines were run again this evening on the frozen code.</li>
      </ul>
    </div>"""
R["TBD_FOOTER"] = """<p>Built from the lab folder on 3 Nov 2026: overnight runs 01:45–06:27 IST, held-out re-run 19:20 IST.</p>
    <p>Board drawing and timing figure are to scale; all other numbers trace to the files named in “How to run it”.</p>"""
for k, v in R.items():
    assert k in s, k
    s = s.replace(k, v)
open(p, "w").write(s)
print("ok")
