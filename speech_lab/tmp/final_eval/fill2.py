import json
p = "report_page/index.html"
s = open(p).read()
clips = open("tmp/final_eval/clips.json").read().strip()
assert "</" not in clips
R = {}
R["TBD_CLIPS"] = clips
R["TBD_POINTS"] = """
        <li><strong>Meeting: works.</strong> On 8 held-out simulated rooms with 20 talkers each, the meeting pipeline raised the average talker's intelligibility (ESTOI) by +0.12 and the worst talker's by +0.06, and cut the level spread between talkers from 3.9 to 1.5 dB. On the real board the gain is about a third of that (+0.04 average).</li>
        <li><strong>Bus: only half works.</strong> Noise goes down (+5.4 dB SNR, worst partner +0.03 ESTOI on the held-out test), but nearby passengers stay as intelligible as before. With the mics 1.3 cm apart, direction can't tell a partner from a passenger, and voice-print extraction followed whoever was louder.</li>
        <li><strong>One bus setup worked on the board:</strong> Mic A's end pointed at the partners, with the axis beam and DPDFNet (+0.12 worst-partner ESTOI, passengers −8.6 dB). It needs the partners in line with the mics, and was only tested that way.</li>
        <li><strong>Calibrate at every boot.</strong> The mics are 1.3 cm apart, and Mic B is sampled one I2S frame (62.5 µs) late plus 0–7.5 µs that changes per boot. Self-calibration from 0.5 s of room noise handles it; the original calibration made results worse.</li>
        <li><strong>Moving the mics to 4 cm is optional:</strong> +0.004 ESTOI for the bus and +0.01 to +0.02 for the meeting.</li>
      """
V = [("Meeting · simulated test", "ok", "works", "+0.12 average and +0.06 worst-talker ESTOI; level spread 3.9 → 1.5 dB."),
     ("Meeting · real board", "part", "partly", "+0.04 average ESTOI; talkers captured 62 → 75 %. Loud fans limit it."),
     ("Bus · simulated test", "part", "partly", "+0.03 worst-partner ESTOI, +5.4 dB SNR; passengers not suppressed."),
     ("Bus · real board", "part", "partly", "Bus v3 +0.01. Partners in line with the mics, axis beam + DPDFNet: +0.12."),
     ("Self-calibration", "ok", "works", "Matches the measured calibration; the skew held steady through the night."),
     ("Mics at 4 cm", "na", "optional", "Bus +0.004, meeting +0.013 average ESTOI in a paired simulation.")]
R["TBD_VERDICTS"] = "".join(f'<div class="verdict"><span class="what">{w}</span><span class="chip {c}">{t}</span><span class="why">{y}</span></div>' for w, c, t, y in V)
R["TBD_BUS"] = """<div class="prose">
      <p>The bus goal is selective: keep the two partners, remove the bus <em>and</em> the other passengers. With the mics 1.3 cm apart, direction can't reliably separate a partner from a passenger, so the pipeline chosen on the tuning set uses no beam. It runs a neural denoiser (DPDFNet) on Mic A, remixes a −15 dB residual back in, and levels the result with an AGC gated on the partners.</p>
    </div>
    <h3>Held-out simulated test</h3>
    <div class="tblwrap"><table>
      <thead><tr><th>Pipeline</th><th class="n">Worst partner</th><th class="n">SNR gain</th><th class="n">Passenger SIR</th><th class="n">Passenger ESTOI</th><th class="n">Partner level, worst</th><th>Do-no-harm gate</th></tr></thead>
      <tbody>
        <tr><td>Mic A as recorded</td><td class="n">0</td><td class="n">0</td><td class="n">0</td><td class="n">0.082</td><td class="n">−0.1 dB</td><td><span class="chip ok">pass</span></td></tr>
        <tr class="hl"><td><strong>Bus v3</strong> (chosen)<span class="sub">x_bus_v3</span></td><td class="n">+0.032<span class="sub">+0.027…+0.037</span></td><td class="n">+5.4 dB<span class="sub">+5.1…+5.7</span></td><td class="n">+3.8 dB<span class="sub">+3.3…+4.2</span></td><td class="n">0.090</td><td class="n">−6.3 dB</td><td><span class="chip no">fail</span><span class="sub">partner level and balance</span></td></tr>
        <tr><td>Bus v3, no neural model<span class="sub">x_bus_v3_classic</span></td><td class="n">+0.025<span class="sub">+0.020…+0.030</span></td><td class="n">+1.7 dB<span class="sub">+1.2…+2.2</span></td><td class="n">−1.4 dB<span class="sub">−2.5…−0.4</span></td><td class="n">0.091</td><td class="n">−3.3 dB</td><td><span class="chip no">fail</span></td></tr>
        <tr><td>DPDFNet only<span class="sub">neural denoiser on Mic A</span></td><td class="n">+0.050<span class="sub">+0.038…+0.063</span></td><td class="n">+9.8 dB<span class="sub">+9.2…+10.4</span></td><td class="n">+5.9 dB<span class="sub">+5.0…+6.8</span></td><td class="n">0.084</td><td class="n">−8.9 dB</td><td><span class="chip no">fail</span></td></tr>
        <tr><td>Oracle<span class="sub">uses the clean speech; the ceiling</span></td><td class="n">+0.454</td><td class="n">+15.3 dB</td><td class="n">+10.8 dB</td><td class="n">0.007</td><td class="n">−1.1 dB</td><td><span class="chip na">n/a</span></td></tr>
      </tbody></table></div>
    <p class="note">30 held-out layouts at −5 and 0 dB SNR (60 scenes), LibriSpeech speech, paired 95 % intervals against Mic A. Passenger ESTOI is how intelligible the nearby passengers remain (lower is better): no real pipeline lowers it, while the oracle takes it to 0.007. DPDFNet alone beat the chosen pipeline here (+0.050 vs +0.032) but drops the weaker partner by up to 9 dB and leaves the two partners 11 dB apart; Bus v3 trades some intelligibility for steadier levels. On the tuning set Bus v3 scored +0.042 at 0 dB.</p>
    <h3>Real board</h3>
    <div class="tblwrap"><table>
      <thead><tr><th>Pipeline</th><th class="n">Worst partner</th><th class="n">SNR gain</th><th class="n">Passenger SIR</th><th class="n">Partner level, worst</th></tr></thead>
      <tbody>
        <tr><td>Mic A as recorded</td><td class="n">0</td><td class="n">0</td><td class="n">0</td><td class="n">−0.1 dB</td></tr>
        <tr><td><strong>Bus v3</strong></td><td class="n">+0.011</td><td class="n">+3.1 dB</td><td class="n">+2.6 dB</td><td class="n">−9.1 dB</td></tr>
        <tr><td>Bus v3, no neural model</td><td class="n">+0.006</td><td class="n">+0.5 dB</td><td class="n">+0.3 dB</td><td class="n">−2.6 dB</td></tr>
        <tr><td>DPDFNet only</td><td class="n">+0.014</td><td class="n">+4.7 dB</td><td class="n">+3.6 dB</td><td class="n">−14.9 dB</td></tr>
        <tr><td>Axis beam<span class="sub">fixed, toward Mic A's end</span></td><td class="n">+0.043</td><td class="n">+2.0 dB</td><td class="n">+1.9 dB</td><td class="n">−1.1 dB</td></tr>
        <tr class="hl"><td><strong>Axis beam + DPDFNet</strong><span class="sub">bus-endfire; layout L1, 6 scenes</span></td><td class="n">+0.117</td><td class="n">+9.6 dB</td><td class="n">+8.6 dB</td><td class="n">−8.2 dB</td></tr>
      </tbody></table></div>
    <div class="prose">
      <ul>
        <li><strong>Setup.</strong> 18 scenes: real partner recordings with the real fan noise, plus simulated passengers and bus noise, in 3 layouts at −5, 0 and +5 dB. Self-calibration gave the same Bus v3 numbers as the measured calibration.</li>
        <li><strong>Why the axis beam wins here.</strong> Every real talker sat in line with the mics on Mic A's side, because the sound came from the MacBook. Turned 180°, the same beam removed the partners (−0.08 ESTOI, −4 dB). With the original default calibration it scored −0.06 to −0.13.</li>
        <li><strong>What failed.</strong> Beams steered by the direction tracker gave the same output as Mic A. Voice-print extraction (VoxLock) made things worse (−0.05 to −0.12) with clean enrollment, or passed the audio through unchanged with self-enrollment.</li>
      </ul>
    </div>"""
for k, v in R.items():
    assert k in s, k
    s = s.replace(k, v)
open(p, "w").write(s)
import re
print("left:", sorted(set(re.findall(r"TBD_[A-Z_]+", s))))
