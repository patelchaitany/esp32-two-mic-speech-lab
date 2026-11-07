"""ESP32-C3 two-mic capture (mic_recorder protocol) with optional speaker playback.

Channel 0 = Mic A (L/R to GND), channel 1 = Mic B (L/R to 3V3).
Samples are returned as float32 at 24-bit full scale (1.0 = 0 dBFS): no gain, no DC removal.
"""
import glob
import os
import select
import subprocess
import termios
import time
import tty

import numpy as np


def find_port():
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        ports = sorted(glob.glob(pattern))
        if ports:
            return ports[0]
    raise RuntimeError("No ESP32 serial port found.")


def _open(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    tty.setraw(fd)
    attrs = termios.tcgetattr(fd)
    attrs[2] |= termios.CLOCAL | termios.CREAD
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


def _read(fd, timeout):
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return b""
    try:
        return os.read(fd, 65536)
    except BlockingIOError:
        return b""


def record(seconds=5, port=None, on_start=None):
    """Record `seconds` of stereo audio. `on_start()` runs as soon as the board confirms the recording began.

    Returns (fs, x[n, 2] float32, meta)."""
    fd = _open(port or find_port())
    try:
        time.sleep(0.3)
        while _read(fd, 0.2):  # drop boot logs and stale bytes
            pass
        buf = b""
        for _ in range(6):  # retry: opening the port can reboot the board
            t_cmd = time.monotonic()
            os.write(fd, f"r{seconds}\n".encode())
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and b"REC " not in buf:
                buf += _read(fd, 0.05)
                err = buf.find(b"ERR ")
                if err >= 0 and b"\n" in buf[err:]:
                    raise RuntimeError("Board reports a wiring fault: " + buf[err + 4:buf.index(b"\n", err)].decode(errors="replace"))
            start = buf.find(b"REC ")
            if start >= 0:
                while b"\n" not in buf[start:]:
                    chunk = _read(fd, 2)
                    if not chunk:
                        raise RuntimeError("Header was cut off.")
                    buf += chunk
                break
        else:
            raise RuntimeError("No response from board. Is mic_recorder uploaded?")
        t_rec = time.monotonic()
        end = buf.index(b"\n", start)
        rate, channels, width, frames = map(int, buf[start + 4:end].split())
        if on_start:
            on_start()
        data = bytearray(buf[end + 1:])
        total = frames * channels * width
        while len(data) < total:
            chunk = _read(fd, 2)
            if not chunk:
                raise RuntimeError(f"Stream stalled after {len(data)} of {total} bytes.")
            data += chunk
        t_end = time.monotonic()
    finally:
        os.close(fd)
    b = np.frombuffer(bytes(data[:total]), np.uint8).reshape(-1, 3).astype(np.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    v = np.where(v & 0x800000, v - (1 << 24), v)
    x = (v.astype(np.float32) / 8388608.0).reshape(-1, channels)
    meta = {"fs": rate, "channels": channels, "frames": frames, "t_cmd": t_cmd, "t_rec": t_rec,
            "t_end": t_end, "stream_s": t_end - t_rec, "extra_bytes": len(data) - total}
    return rate, x, meta


def play_record(stim_path, seconds=6, port=None):
    """Start a recording, then play `stim_path` through the default output with afplay (None = silence)."""
    state = {}

    def start():
        if stim_path:
            state["t_play"] = time.monotonic()
            state["proc"] = subprocess.Popen(["afplay", stim_path])

    fs, x, meta = record(seconds, port, on_start=start)
    if "proc" in state:
        meta["afplay_rc"] = state["proc"].wait(timeout=20)
        meta["t_play"] = state["t_play"]
        meta["play_after_rec_s"] = state["t_play"] - meta["t_rec"]
    return fs, x, meta
