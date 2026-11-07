// ESP32-C3 + 2x INMP441 on one I2S bus -> stereo audio streamed over USB.
// Mic A (L/R = GND) = left channel, Mic B (L/R = 3V3) = right channel.
//
// Commands from the host (newline-terminated):
//   r / r<seconds>  record (default 5 s). Reply: "REC <rate> <channels> <bytes_per_sample> <frames>\n"
//                   then exactly frames*channels*3 bytes of 24-bit little-endian PCM,
//                   or "ERR <reason>\n" if a wiring fault was found at boot.
//   d               print pin diagnostics, ending with "DIAG END".

#include <driver/i2s_std.h>
#include <driver/gpio.h>

#define PIN_SCK 4
#define PIN_WS 5
#define PIN_SD 6

#define SAMPLE_RATE 16000
#define DEFAULT_SECONDS 5
#define MAX_SECONDS 60
#define BLOCK_FRAMES 256

static i2s_chan_handle_t rx;
static int32_t in_buf[BLOCK_FRAMES * 2];       // 32-bit stereo frames from I2S
static uint8_t out_buf[BLOCK_FRAMES * 2 * 3];  // 24-bit packed stereo frames

static uint32_t frames_left = 0;
static String cmd;
static String fault;  // non-empty = wiring fault, I2S not started

// SCK and WS are outputs: an undriven pin must follow the internal pull resistors.
// If it doesn't, it is shorted to 3V3 or GND and must not be driven.
static void check_output_pin(int pin, const char *name) {
  gpio_num_t g = (gpio_num_t)pin;
  gpio_reset_pin(g);
  gpio_set_direction(g, GPIO_MODE_INPUT);
  gpio_set_pull_mode(g, GPIO_PULLDOWN_ONLY);
  delay(2);
  int pd = gpio_get_level(g);
  gpio_set_pull_mode(g, GPIO_PULLUP_ONLY);
  delay(2);
  int pu = gpio_get_level(g);
  gpio_set_pull_mode(g, GPIO_FLOATING);
  if (pd == 1) fault += String(name) + " (GPIO" + pin + ") is stuck HIGH - shorted to 3V3? ";
  if (pu == 0) fault += String(name) + " (GPIO" + pin + ") is stuck LOW - shorted to GND? ";
}

static void start_i2s() {
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  chan_cfg.dma_desc_num = 8;
  chan_cfg.dma_frame_num = 512;  // 8 x 32 ms of slack if USB stalls
  i2s_std_config_t std_cfg = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = {
      .mclk = I2S_GPIO_UNUSED,
      .bclk = (gpio_num_t)PIN_SCK,
      .ws = (gpio_num_t)PIN_WS,
      .dout = I2S_GPIO_UNUSED,
      .din = (gpio_num_t)PIN_SD,
      .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
    },
  };
  if (i2s_new_channel(&chan_cfg, NULL, &rx) != ESP_OK
      || i2s_channel_init_std_mode(rx, &std_cfg) != ESP_OK
      || i2s_channel_enable(rx) != ESP_OK) {
    fault = "I2S driver failed to start";
  }
}

static void diag() {
  if (fault.length()) {
    Serial.println("FAULT: " + fault);
    Serial.println("DIAG END");
    return;
  }
  gpio_input_enable((gpio_num_t)PIN_SCK);
  gpio_input_enable((gpio_num_t)PIN_WS);
  gpio_input_enable((gpio_num_t)PIN_SD);
  const int pins[3] = { PIN_SCK, PIN_WS, PIN_SD };
  const char *names[3] = { "SCK", "WS", "SD" };
  uint32_t hi[3] = { 0 }, edges[3] = { 0 }, prev = REG_READ(GPIO_IN_REG);
  const uint32_t N = 200000;
  uint32_t t0 = micros();
  for (uint32_t i = 0; i < N; i++) {
    uint32_t v = REG_READ(GPIO_IN_REG);
    for (int p = 0; p < 3; p++) {
      hi[p] += (v >> pins[p]) & 1;
      edges[p] += ((v ^ prev) >> pins[p]) & 1;
    }
    prev = v;
  }
  uint32_t us = micros() - t0;
  for (int p = 0; p < 3; p++)
    Serial.printf("%-3s GPIO%d: high %3u%%, ~%u kHz\n", names[p], pins[p], hi[p] * 100 / N, edges[p] * 500 / us);
  Serial.println("Expect SCK ~1024 kHz, WS ~16 kHz, SD toggling when mics are alive.");
  Serial.println("DIAG END");
}

static void handle_command() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmd.length() && cmd[0] == 'r' && frames_left == 0) {
        if (fault.length()) {
          Serial.println("ERR " + fault);
        } else {
          int secs = cmd.length() > 1 ? cmd.substring(1).toInt() : DEFAULT_SECONDS;
          if (secs < 1 || secs > MAX_SECONDS) secs = DEFAULT_SECONDS;
          frames_left = (uint32_t)secs * SAMPLE_RATE;
          Serial.printf("REC %d 2 3 %u\n", SAMPLE_RATE, frames_left);
        }
      } else if (cmd == "d" && frames_left == 0) {
        diag();
      }
      cmd = "";
    } else if (cmd.length() < 16) {
      cmd += c;
    }
  }
}

void setup() {
  Serial.setTxBufferSize(16384);
  Serial.begin(921600);  // native USB ignores the baud rate

  check_output_pin(PIN_SCK, "SCK");
  check_output_pin(PIN_WS, "WS");
  if (fault.length() == 0) start_i2s();
}

void loop() {
  handle_command();
  if (fault.length()) {
    delay(10);
    return;
  }

  // Always read, so the DMA buffer holds fresh audio when a recording starts.
  size_t got = 0;
  i2s_channel_read(rx, in_buf, sizeof(in_buf), &got, portMAX_DELAY);
  if (frames_left == 0) return;

  uint32_t frames = got / 8;
  if (frames > frames_left) frames = frames_left;

  // INMP441 sends 24-bit samples left-aligned in 32-bit slots: keep the top 3 bytes.
  const uint8_t *src = (const uint8_t *)in_buf;
  uint8_t *dst = out_buf;
  for (uint32_t i = 0; i < frames * 2; i++) {
    dst[0] = src[1];
    dst[1] = src[2];
    dst[2] = src[3];
    src += 4;
    dst += 3;
  }
  Serial.write(out_buf, frames * 6);
  frames_left -= frames;
}
