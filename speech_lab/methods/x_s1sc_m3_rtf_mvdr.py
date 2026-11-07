"""Spacing-study variant (x_): methods.m3_rtf_mvdr with S1's microsecond constants scaled by d / 4 cm (methods/x_s1scale.py).
Identical to m3_rtf_mvdr at 4 cm."""
from methods import m3_rtf_mvdr, x_s1scale


def enhance(x, fs, info, return_stages=False):
    with x_s1scale.scaled(x_s1scale.scale_of(info)):
        return m3_rtf_mvdr.enhance(x, fs, info, return_stages=return_stages)
