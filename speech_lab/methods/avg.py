"""Baseline: average of both mics (delay-and-sum steered to broadside)."""
def enhance(x, fs, info):
    return 0.5 * (x[0] + x[1])
