#!/usr/bin/env python3
"""Record audio from the ESP32-C3 mic_recorder sketch and save a stereo WAV.

Left channel = Mic A (L/R to GND), right channel = Mic B (L/R to 3V3).
Standard library only (no pyserial needed).
"""
import argparse
import glob
import math
import os
import select
import sys
import termios
import time
import tty
import wave
from array import array


def find_port():
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*",
                    "/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    sys.exit("No ESP32 serial port found. Pass one with -p.")


def open_port(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    tty.setraw(fd)
    attrs = termios.tcgetattr(fd)
    attrs[2] |= termios.CLOCAL | termios.CREAD
    attrs[4] = attrs[5] = termios.B115200  # ignored by native USB
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


def read_some(fd, timeout):
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return b""
    try:
        return os.read(fd, 65536)
    except BlockingIOError:
        return b""


def capture(fd, seconds):
    time.sleep(0.3)
    while read_some(fd, 0.2):  # drop boot logs / stale bytes
        pass

    buf = b""
    for _ in range(6):  # retry in case opening the port rebooted the board
        os.write(fd, f"r{seconds}\n".encode())
        deadline = time.time() + 1.5
        while time.time() < deadline and b"REC " not in buf:
            buf += read_some(fd, 0.1)
            err = buf.find(b"ERR ")
            if err >= 0 and b"\n" in buf[err:]:
                msg = buf[err + 4:buf.index(b"\n", err)].decode(errors="replace").strip()
                sys.exit(f"Board reports a wiring fault: {msg}")
        start = buf.find(b"REC ")
        if start >= 0:
            while b"\n" not in buf[start:]:
                chunk = read_some(fd, 2)
                if not chunk:
                    sys.exit("Header was cut off.")
                buf += chunk
            break
    else:
        sys.exit("No response from board. Is mic_recorder uploaded?")

    end = buf.index(b"\n", start)
    rate, channels, width, frames = map(int, buf[start + 4:end].split())
    data = bytearray(buf[end + 1:])
    total = frames * channels * width

    print(f"Recording {frames / rate:.1f} s ...", flush=True)
    while len(data) < total:
        chunk = read_some(fd, 2)
        if not chunk:
            sys.exit(f"Stream stalled after {len(data)} of {total} bytes.")
        data += chunk
    return rate, channels, bytes(data[:total])


def decode_24bit(data):
    out = array("i", bytes(len(data) // 3 * 4))
    for i in range(len(out)):
        b = data[3 * i:3 * i + 3]
        v = b[0] | (b[1] << 8) | (b[2] << 16)
        out[i] = v - (1 << 24) if v & 0x800000 else v
    return out


def dbfs(x):
    return 20 * math.log10(x / 8388608) if x > 0 else float("-inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-p", "--port", help="serial port (default: auto-detect)")
    ap.add_argument("-s", "--seconds", type=int, default=5)
    ap.add_argument("-o", "--out", default="recording.wav")
    ap.add_argument("--raw", action="store_true", help="no auto gain (24-bit -> 16-bit at unity)")
    ap.add_argument("--split", action="store_true", help="also write one mono WAV per mic")
    ap.add_argument("--diag", action="store_true", help="print pin/clock diagnostics and exit")
    args = ap.parse_args()

    port = args.port or find_port()
    fd = open_port(port)
    if args.diag:
        time.sleep(0.3)
        while read_some(fd, 0.2):
            pass
        os.write(fd, b"d\n")
        buf, deadline = b"", time.time() + 3
        while time.time() < deadline and b"DIAG END" not in buf:
            buf += read_some(fd, 0.1)
        os.close(fd)
        print(buf.decode(errors="replace").replace("DIAG END", "").strip() or "No response.")
        return
    try:
        rate, channels, data = capture(fd, args.seconds)
    finally:
        os.close(fd)

    samples = decode_24bit(data)
    chans = [samples[c::channels] for c in range(channels)]

    # Remove DC offset and report levels per mic.
    peak = 0
    for name, ch in zip(("Mic A (left) ", "Mic B (right)"), chans):
        mean = sum(ch) / len(ch)
        for i in range(len(ch)):
            ch[i] = int(ch[i] - mean)
        p = max(abs(min(ch)), abs(max(ch)))
        rms = math.sqrt(sum(v * v for v in ch) / len(ch))
        peak = max(peak, p)
        warn = "  <-- silent? check wiring" if p < 64 else ""
        print(f"{name}: peak {dbfs(p):6.1f} dBFS, rms {dbfs(rms):6.1f} dBFS{warn}")

    gain = 1 / 256  # 24-bit -> 16-bit
    if not args.raw and peak > 0:
        gain = min(0.89 * 32767 / peak, 64 / 256)  # normalize to -1 dBFS, max +48 dB
    print(f"Gain applied: {20 * math.log10(gain * 256):+.1f} dB")

    pcm = [array("h", (max(-32768, min(32767, round(v * gain))) for v in ch)) for ch in chans]

    def write_wav(path, chs):
        inter = array("h", bytes(2 * len(chs) * len(chs[0])))
        for c, ch in enumerate(chs):
            inter[c::len(chs)] = ch
        with wave.open(path, "wb") as w:
            w.setnchannels(len(chs))
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(inter.tobytes())
        print(f"Saved {path}")

    write_wav(args.out, pcm)
    if args.split:
        stem = os.path.splitext(args.out)[0]
        write_wav(f"{stem}_micA.wav", pcm[:1])
        write_wav(f"{stem}_micB.wav", pcm[1:2])


if __name__ == "__main__":
    main()
