p = "report_page/index.html"
s = open(p).read()
F = [
 # 1 meeting oracle row: paired values like the other rows
 ('<tr><td>Oracle<span class="sub">uses the clean speech; the ceiling</span></td><td class="n">+0.264</td><td class="n">+0.174</td><td class="n">+0.189</td><td class="n">4.0 dB</td><td class="n">55 %</td><td><span class="chip na">n/a</span></td></tr>',
  '<tr><td>Oracle<span class="sub">uses the clean speech; the ceiling</span></td><td class="n">+0.070<span class="sub">+0.057…+0.083</span></td><td class="n">+0.173</td><td class="n">+0.189</td><td class="n">4.0 dB</td><td class="n">55 %</td><td><span class="chip na">n/a</span></td></tr>'),
 # 2 meeting v3 gate label
 ('<td><span class="chip part">borderline</span><span class="sub">noise modulation 5.4 dB, CI up to 6.3 (limit 6)</span></td>',
  '<td><span class="chip no">fail</span><span class="sub">noise modulation 5.4 dB, CI up to 6.3 (limit 6)</span></td>'),
 # 3 GTCRN label
 ('<span class="sub">noise pumping 5.9 dB</span>', '<span class="sub">noise modulation 5.9 dB</span>'),
 # 4 DPDFNet level drop wording
 ('but drops the weaker partner by up to 9 dB and leaves the two partners 11 dB apart',
  'but drops the weaker partner by about 9 dB on average and leaves the two partners up to 11 dB apart (90th percentile)'),
 # 5 room noise range
 ('<span class="v num">−41 to −59 dBFS</span>', '<span class="v num">−41 to −62 dBFS</span>'),
 ('kept the laptop fans loud all night (−41 to −59 dBFS)', 'kept the laptop fans running all night (−41 to −62 dBFS)'),
 # 6 Mic B gain stability
 ('For direct sound near the mic axis; −0.7 dB in diffuse noise. It varied by at most 0.03 dB across boots, so a factory constant works.',
  'For direct sound near the mic axis; −0.7 dB in diffuse noise. Across 8 boots it varied by 0.03 dB or less (std) at 0.5–2 kHz and 0.14–0.35 dB in other bands, so a factory constant works.'),
 # 7 skew value
 ('<span class="v num">−62.5 µs + 0…7.5</span>', '<span class="v num">−62.5 to −70 µs</span>'),
 # 8/9 drift card
 ('<article class="exp"><p class="q">Does the skew drift overnight?</p>\n        <p>No. Two sweeps from the same speaker 1.5 hours apart gave the same direct-sound delay. Noise-based estimates every 10 minutes wandered a little as the fan noise changed, well inside the ±5 µs target.</p>\n        <p class="ans">direct path −0.516 sample at 01:55 and at 03:26 · 27 noise readings 02:05–06:25: −0.979…−1.032</p></article>',
  '<article class="exp"><p class="q">Does the skew drift overnight?</p>\n        <p>Not measurably. Two sweeps from the same speaker 1.5 hours apart gave the same direct-sound delay. Noise-based estimates every 10 minutes until 06:25 wandered by about ±0.03 sample as the fan noise changed, inside the ±5 µs target. Only the first 1.5 hours are checked with a sweep.</p>\n        <p class="ans">direct path −0.516 sample at 01:55 and at 03:26 · self-calibration on 27 noise readings 02:05–06:25: −0.979…−1.032 (monitor\'s simpler fit −0.970…−1.042)</p></article>'),
 ('Sweeps from the same speaker at 01:55 and 03:26 measured the same direct-path delay, so the wander is in the noise-based estimate, not the board.',
  'Sweeps from the same speaker at 01:55 and 03:26 measured the same direct-path delay, so over that span the wander is in the noise-based estimate, not the board.'),
]
for a, b in F:
    assert s.count(a) == 1, ("missing or repeated", a[:80], s.count(a))
    s = s.replace(a, b)
# verdict strip + bottom line wording
F2 = [
 ('<span class="what">Meeting · simulated test</span><span class="chip ok">works</span><span class="why">+0.12 average and +0.06 worst-talker ESTOI; level spread 3.9 → 1.5 dB.</span>',
  '<span class="what">Meeting · simulated test</span><span class="chip ok">mostly works</span><span class="why">+0.12 average and +0.06 worst-talker ESTOI; level spread 3.9 → 1.5 dB. The no-neural version passes do-no-harm; v3 slightly exceeds the noise-modulation limit.</span>'),
 ('<span class="what">Self-calibration</span><span class="chip ok">works</span><span class="why">Matches the measured calibration; the skew held steady through the night.</span>',
  '<span class="what">Self-calibration</span><span class="chip ok">works</span><span class="why">Matched the measured calibration for both v3 pipelines; the skew is stable within a boot.</span>'),
 ('<li><strong>Meeting: works.</strong> On 8 held-out simulated rooms with 20 talkers each, the meeting pipeline raised the average talker\'s intelligibility (ESTOI) by +0.12 and the worst talker\'s by +0.06, and cut the level spread between talkers from 3.9 to 1.5 dB. On the real board the gain is about a third of that (+0.04 average).</li>',
  '<li><strong>Meeting: mostly works.</strong> On 8 held-out simulated rooms with 20 talkers each, the meeting pipeline raised the average talker\'s intelligibility (ESTOI) by +0.12 and the worst talker\'s by +0.06, close to the oracle\'s +0.07, and cut the level spread between talkers from 3.9 to 1.5 dB. The neural version slightly exceeds the noise-modulation limit; the classical version passes. On the real board the gain is about a third (+0.04 average).</li>'),
]
for a, b in F2:
    assert s.count(a) == 1, ("missing", a[:80])
    s = s.replace(a, b)
open(p, "w").write(s)
print("fixed", len(F) + len(F2))
