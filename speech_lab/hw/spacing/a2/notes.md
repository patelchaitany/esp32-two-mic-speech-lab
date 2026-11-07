# A2 – Is an inter-channel sampling skew expected on ESP32-C3 + 2 × INMP441 (one I2S bus)?

Date: 2026-11-03. Scope: primary sources only (datasheets, TRM, ESP-IDF source/docs, GitHub issues), plus one
cheap read-only check of the session-1 recordings (`startup_check.py` in this folder).
Convention (as in the task): tau = arrival at B minus arrival at A, in samples at 16 kHz. The measured
skew of about −1.1 samples means B's sample n is taken about 68 µs later in real time than A's sample n.
One frame is 62.5 µs and half a frame is 31.25 µs.

## 1. What the primary sources say

### Microphone (TDK/InvenSense INMP441, DS-INMP441-00 rev 1.1, 2014)
URL: https://microcontroller-course.web.cern.ch/Project_Material/resources/INMP441.pdf
(TDK's own link https://invensense.tdk.com/wp-content/uploads/2015/02/INMP441.pdf now redirects to a navigation page.)

- **Synchronizing Microphones (p. 13).** "Stereo INMP441 microphones are synchronized by the WS signal, so audio
  captured from two microphones sharing the same clock will be in sync." The datasheet adds that if the mics are enabled
  separately, synchronization can take up to 0.35 ms after enable, while internal data paths are flushed.
  - The datasheet does **not** say when the sample is taken, or whether the left-slot and right-slot words of
    one frame carry the same sampling instant.
- **I2S interface (p. 11–12).**
  - The format is I2S, 24-bit, two's complement, with 64 SCK cycles per WS frame (fSCK = 64 × fWS).
  - The MSB is "delayed by one SCK cycle from the start of each half-frame". This 1-bit delay is the same for both channels.
  - L/R low means the mic transmits in the left channel (WS low); L/R high means the right channel (WS high).
  - SD tri-states immediately after the LSB. A 100 kΩ pull-down on SD is recommended.
- **Digital filter (Table 1).**
  - Group delay is 17.2/fS: 1078 µs at fs = 16 kHz.
  - The ADMP441 datasheet gives 17.25/fS.
  - Both mics use the same digital filter and the same clock, so the group delay cancels in the A–B difference.
  - Pass band is 0.423·fS. The high-pass corner is 3.7 Hz at 48 kHz and scales with fs (about 1.2 Hz at 16 kHz).
- **Start-up (p. 10).**
  - After power-up, the output is zero for 2^18 SCK cycles.
  - After SCK stops (standby), normal operation resumes 2^14 SCK cycles after SCK restarts.
  - After CHIPEN goes high, it takes 2^17 SCK cycles.
  - At SCK = 1.024 MHz these are 256 ms, 16 ms and 128 ms.

### Original ADI part (ADMP441 Rev. B, 2012), the same design
URL: https://www.analog.com/media/en/technical-documentation/obsolete-data-sheets/ADMP441.pdf
- It has the same I2S, L/R and start-up text, and group delay 17.25/fS.
- It has **no** "Synchronizing microphones" section. TDK added that section later, in the INMP441 sheet.

### Successor I2S mic (TDK ICS-43434, DS-000069 v1.2): the explicit statement
URL: https://cdn.sparkfun.com/assets/a/9/c/9/1/DS-000069-ICS-43434-v1.2.pdf (p. 12)
- "The two microphones will synchronously sample the acoustic signals at the beginning of the I2S frame
  (WS falling edge)." In left mode, data are output after the WS falling edge; in right mode, after the WS rising edge.
- Group delay is "2/fS", measured from acoustic input to digital output, including the filter and the I2S serial output.

### TDK TDM mic (ICS-52000, DS-000121 v1.3)
URL: https://product.tdk.com/system/files/dam/doc/product/sw_piezo/mic/mems-mic/data_sheet/ds-000121-ics-52000-v1.3.pdf
- "All microphones in an array sample their acoustic signals synchronously, enabling precise array processing."

**Conclusion (mic side).** For mics that share one bus, TDK's documented design intent is simultaneous sampling,
referenced to the WS falling edge. That gives an expected inter-channel skew of **0**, provided the host pairs the left
and right words of the **same** frame.
- No primary source supports a half-frame (31.25 µs) skew.
- An internal one-frame latency difference between L mode and R mode in the INMP441 is not documented. It is
  not ruled out either: the INMP441 sheet only says "in sync", while the ICS-43434 sheet is explicit.
- Community opinion is split and unmeasured. In EE.SE Q615444, one answer assumes the mics latch at the same WS edge;
  another says the channels are "out of sync by one word length".
  - URL: https://electronics.stackexchange.com/questions/615444/i2s-microphone-data-synchronization-for-left-and-right-channel

### Host (ESP32-C3 I2S, ESP-IDF 5.x)
- **ESP32-C3 TRM v1.4, chapter 29.** URL: https://documentation.espressif.com/esp32-c3_technical_reference_manual_en.pdf
  - Standard mode is a 2-slot TDM.
  - I2S_RX_WS_IDLE_POL: "0: WS remains low when receiving left channel data, and remains high when receiving right
    channel data".
  - I2S_RX_MSB_SHIFT=1 means WS changes one BCK before the MSB (Philips).
  - RX start (I2S_RX_START) in master mode is described with no statement about which slot is captured first, or about
    re-alignment.
  - The TRM has no documented sampling offset between slots. The RX only captures bits; the acoustic sampling happens
    in the mics.
- **ESP-IDF source** (v5.3, `components/hal/i2s_hal.c`, `components/esp_driver_i2s/include/driver/i2s_std.h`,
  `components/hal/esp32c3/include/hal/i2s_ll.h`, `components/esp_driver_i2s/i2s_std.c`):
  - URLs: https://raw.githubusercontent.com/espressif/esp-idf/release/v5.3/components/hal/i2s_hal.c ,
    https://raw.githubusercontent.com/espressif/esp-idf/release/v5.3/components/esp_driver_i2s/include/driver/i2s_std.h ,
    https://raw.githubusercontent.com/espressif/esp-idf/release/v5.3/components/hal/esp32c3/include/hal/i2s_ll.h ,
    https://raw.githubusercontent.com/espressif/esp-idf/release/v5.3/components/esp_driver_i2s/i2s_std.c
  - The Philips default on HW version 2 (C3) sets ws_pol = false, bit_shift = true, left_align = true, big_endian = false
    and slot_mask = BOTH.
  - The HAL calls `i2s_ll_rx_set_ws_idle_pol(ws_pol)` and `i2s_ll_rx_enable_msb_shift(bit_shift)`.
  - `i2s_ll_rx_select_std_slot` sets `rx_tdm_tot_chan_num = 1` (2 slots).
  - `i2s_ll_rx_start` does `rx_update`, then `rx_start`. It does not wait for a WS edge in software.
  - In full duplex, "the later initialized channel must be slave", so the RX can end up as an internal slave on
    shared BCK/WS.
- **ESP-IDF docs, ESP32-C3.** URL: https://docs.espressif.com/projects/esp-idf/en/stable/esp32c3/api-reference/peripherals/i2s.html
  - The STD RX table shows stereo data in line order. Mono-left gives the 1st, 3rd, … words; mono-right gives the
    2nd, 4th, …. So the buffer is [L, R, L, R …] with the left word (WS low) first.
  - `i2s_channel_enable` starts BCLK/WS in master mode, and disable stops them. Stopping them puts the INMP441 into standby.
  - TDM RX is supported (4 slots at 32-bit).
  - PDM: soc_caps for C3 in v5.3 has only PDM TX. On master, raw PDM RX (1 line) is supported. There is no PDM-to-PCM
    converter on C3.
  - URLs: https://raw.githubusercontent.com/espressif/esp-idf/release/v5.3/components/soc/esp32c3/include/soc/soc_caps.h ,
    https://raw.githubusercontent.com/espressif/esp-idf/master/components/soc/esp32c3/include/soc/soc_caps.h
- **The 1-bit Philips delay and the byte selection cannot create a skew.**
  - The MSB delay is 1 SCK = 0.98 µs at 1.024 MHz, the same in both slots.
  - A wrong msb_shift scales or garbles the data by one bit. It does not move it in time.
  - Keeping bytes 1..3 of each little-endian 32-bit word keeps exactly the INMP441's 24-bit word (bits 25–32 are tri-state).

### Field reports relevant to determinism
- **esp-idf #11135 (ESP32-C3, IDF v5.0.1).** https://github.com/espressif/esp-idf/issues/11135
  - Setup: RX master, Philips stereo, 16 kHz, two ICS-43434.
  - In about 1 in 10 restarts the channel came up wrongly: a 2-bit data misalignment (×4 gain) that lasted until the next restart.
  - The maintainer suspected BCLK being delayed relative to WS at init. The problem stopped on IDF master in April 2023.
    No root-cause commit was identified.
  - So C3 RX initialisation has been non-deterministic at bit level in an older IDF.
- **esp-idf #9513 (ESP32-S3, the same I2S IP generation).** https://github.com/espressif/esp-idf/issues/9513
  - In TDM slave full-duplex mode, "Sometimes first channel appear when WS is High, some times when WS is LOW" after reboots.
  - The maintainers attributed it to the internal MCLK/BCLK ratio at high sample rates. Closed in 2022 (commit af0c0bd).
  - So a slot-level slip has been seen in **slave** mode.
- **arduino-esp32 #8207 (XIAO ESP32-C3 + INMP441).** https://github.com/espressif/arduino-esp32/issues/8207
  - After stopping and restarting the I2S channel (or after light sleep), the INMP441 gives about 100–250 ms of loud,
    decaying noise. This is consistent with the datasheet's standby and start-up behaviour plus its digital high-pass.
- **Legacy-driver L/R swaps on the original ESP32.**
  - esp-idf #6625: https://github.com/espressif/esp-idf/issues/6625 (fixed 2021).
  - These show that channel-order bugs exist in some driver versions. They are not C3-specific.
- **No published measurement found.** No ESP32 + INMP441/ICS-43434 stereo DOA project reports a measured inter-channel
  delay. Checked: https://github.com/alanfellerv/Audio-Source-Localisationbot ,
  https://github.com/EkinUgurr/sound-localization-esp32 , and the gfai 2017 MEMS-array paper:
  https://www.gfaitech.com/fileadmin/gfaitech/documents/publications/english/IN-Pflug-Krischker-AspectsOfTheUseOfMemsMicrophones-2017.pdf
  The gfai paper only notes that the INMP441 is "not recommended for new designs", with the ICS-43434 as successor.
- **Not readable.** esp32.com forum threads (bot check) were not read and are not cited.

## 2. What could make about −1.1 samples (68 µs)?

Pairing algebra (words on the line: L0 R0 L1 R1 …; the firmware always takes ch0 = 1st word and ch1 = 2nd word of each pair):

| Mic timing model | Pairing (L_n, R_n) | Pairing shifted by one slot (R_n, L_n+1), labels swapped |
|---|---|---|
| A. Both sample at the WS falling edge (TDK claim) | 0 | **−1 frame**, ch0 = 3V3 mic |
| B. Each mic samples at its own slot start | −0.5 | −0.5, labels swapped |
| C. L-mode outputs the previous frame's sample, R-mode the current one (undocumented) | **−1 frame**, no swap | 0, labels swapped |

So under every model the skew comes in steps of half a frame, and **−1 frame needs either**:
- a host-side one-slot pairing offset, which also swaps the channel labels; or
- an undocumented one-frame latency asymmetry inside the INMP441.

The pairing offset could come from:
- WS polarity inverted: `ws_pol=true`, GPIO `ws_inv`, or `setInverted(…, ws=true)` in Arduino ESP_I2S;
- RX started on the right slot, for example as an internal slave in full duplex;
- the firmware dropping or adding one 4-byte word.

The half-frame model gives −0.5 under both pairings. It cannot give −1.1.

### Cross-check with the session-1 numbers (direct-path limits, c = 343 m/s)

| d | Largest physical TDOA | Skew range allowed by the speaker TDOAs (−0.53…−0.56) and the fan TDOAs (−1.00…−1.15) |
|---|---|---|
| 1.0 cm | 0.466 samples | −0.996 … −0.684 samples |
| 1.2 cm | 0.560 samples | −1.090 … −0.590 samples |
| 1.5 cm | 0.700 samples | −1.230 … −0.450 samples |
| 2.0 cm | 0.933 samples | −1.463 … −0.217 samples |

- **Skew 0 is impossible for d ≤ 2 cm.** The fan would need d ≥ 2.47 cm.
- **Skew −0.5 needs d ≥ 1.39 cm**, with the fan at endfire on the B side.
- **Skew −1.0 needs d ≥ 1.01 cm**, with the speakers at endfire on the A side.

With the diffuse-coherence spacing of about 1.2 cm and the early-reflection phase slopes of −1.07 to −1.12, one frame
is favoured. The leftover ≈ −0.1 sample (≈ 6 µs) is within the fit systematics:
- high-pass corner spread;
- residual phase rms of 8–18°;
- port differences of a few µs.

### Per-take stability (`startup_check.py`, read-only)
- GCC-PHAT lag between the channels on the fan noise (10.0–11.9 s, 150–7000 Hz) in all 10 takes:
  −1.094, −1.078, −1.078, −1.109, −1.094, −1.094, −1.109, −1.125, −1.047, −1.000 samples.
  - There is no ±1-sample jump, so the pairing did not slip during session 1.
- No take starts with an INMP441 start-up signature (no zero run, no decaying burst).
  - So the I2S clock probably ran continuously between takes, or the firmware discards the warm-up.
  - Whether the takes spanned reboots or I2S restarts is therefore **unknown**. Stability across restarts is not yet shown.

## 3. Answers

1. **Mechanism.** The measured ≈ 68 µs is best explained as **one frame (62.5 µs) plus a small residual (≈ 3–9 µs)**.
   - TDK documents simultaneous sampling, which gives 0. So the frame offset is either:
     - (a) a host-side one-slot re-pairing, which would mean ch0 is physically the L/R = 3V3 mic; or
     - (b) an undocumented L-versus-R one-frame latency difference in the INMP441.
   - A half-frame skew is unsupported by the documents and by the data.
   - The Philips 1-bit delay, the byte selection and the group delay (identical for both mics) contribute ≤ 1 µs.
   - **Discriminating test (no audio playback needed):** tap or scratch the L/R = GND mic, or cover its port, and
     see whether ch0 or ch1 responds. Also check the firmware's `ws_pol`, `ws_inv`, slot mask, and whether TX+RX run in full duplex.
2. **Determinism.**
   - For a fixed firmware in RX-only master mode, the integer part is expected to be deterministic. The mics
     resynchronise to WS within 0.35 ms after enable, and the RX slot order is set by the RX state machine and `rx_ws_idle_pol`.
   - This is not guaranteed:
     - C3 RX init has shown random bad starts at bit level (IDF 5.0.1);
     - S3 slave and full-duplex has shown slot-order flips across reboots.
   - S0 should therefore re-check the integer part (and the channel identity) at every start. This is cheap.
     The fractional residual can come from calib.json.
3. **Firmware options**
   - **Check the slot configuration first.**
     - Use `I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO)` unchanged
       (ws_pol = false, bit_shift = true, slot_mask = BOTH).
     - Set no GPIO `ws_inv` or `bclk_inv`.
     - Create an RX-only channel (`i2s_new_channel(&cfg, NULL, &rx)`), so the RX is the master that generates WS
       and is never an internal full-duplex slave.
     - If the tap test shows ch0 is the 3V3 mic, fixing the polarity or pairing should bring the skew to 0, as TDK documents.
   - **Exact integer removal.** Emit (ch0[n], ch1[n−1]), holding the previous frame's second word. This removes
     62.5 µs with no interpolation. It is valid for both remaining explanations (re-pairing, or the internal L/R
     latency), but only if the offset stays at +1 frame, so verify it at boot. The same can be done on the host as a
     1-sample delay of ch1.
   - **Stable start.**
     - Keep BCLK/WS running between recordings: do not call `i2s_channel_disable` per take.
     - After any (re)start, discard at least 256 ms after power-up (2^18 SCK cycles) and at least 16 ms after an SCK
       restart (2^14 cycles). In practice discard at least 0.5 s: the digital high-pass is about 1.2 Hz at 16 kHz,
       and #8207 reports 100–250 ms of noise.
     - Use an IDF newer than v5.0.1 (#11135).
   - **Self-report at boot.**
     - Log or send the integer lag measured on 1–2 s of ambient noise (GCC-PHAT or cross-spectrum slope), and the
       per-channel level, in the REC header.
     - S0 compares these with calib.json.
   - **Hardware alternatives with documented synchronous sampling.**
     - ICS-43434 pair: same I2S wiring, and TDK states both sample at the WS falling edge. The host pairing still needs checking.
     - ICS-52000 TDM chain: the C3 supports TDM RX with up to 4 × 32-bit slots, so 3–4 mics fit on one data line.
       It needs the WS→WSO daisy chain and its start-up sequence.
     - Stereo audio ADC or codec with analog MEMS mics: one converter samples both channels. It needs MCLK and I2C
       configuration and adds parts.
     - PDM mics:
       - The C3 has no PDM-to-PCM converter. Raw PDM RX is only in newer IDF (master caps: 1 line), so decimation
         would be in software, which is costly.
       - The two PDM channels on one line are captured on opposite clock edges. This is sub-µs and would need checking.
     - Not possible on the C3: one mic per I2S controller (SOC_I2S_NUM = 1).

## 4. Calibration implications (short)
- **Model the skew as integer + fraction.**
  - Remove one frame exactly, then let S0 fit only a fractional residual. Search ±0.3 sample; about −0.1 sample is expected.
  - After integer removal, every apparent TDOA must lie within ±d·fs/c: 0.47–0.93 samples for d = 1–2 cm.
    A cluster about ±1 sample away means a slot slip (and a label swap). That is a cheap per-start detector.
- **Do the tap test before trusting the channel labels.** If ch0 is the 3V3 mic:
  - swap the labels;
  - the "Mic B 1 dB less sensitive" finding then refers to the GND mic;
  - the geometric TDOA signs flip.
- **Update the bench and the hardware notes.**
  - The bench E8 skew set (0 / 31.25 µs) and HW_TESTS finding 3 (half-frame skew, speakers at broadside) are not
    supported by the documents.
  - Add 62.5 µs, and a ±1-frame-slip case with label swap.
- **Spacing.**
  - The coherence magnitude (sinc) does not depend on the skew, so d ≈ 1.2 cm from the magnitude is not confounded.
  - Phase-based estimates must separate the skew: τ0 = (τ₊ + τ₋)/2 from endfire chirps.
- **Size of the effect at d ≈ 1.2 cm.**
  - The skew (62.5 µs) is 1.8 × the endfire TDOA (35 µs).
  - A 3 µs residual is 9 % of the endfire TDOA, so S0's ±3 µs target matters.
- **Absolute latency.** The group delay of 17.2–17.25/fs = 1.08 ms is common to both mics. It only affects absolute
  latency, for example AEC reference alignment.
