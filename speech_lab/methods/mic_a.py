"""Baseline: reference microphone (Mic A) only, no processing."""
def enhance(x, fs, info):
    return x[0]
