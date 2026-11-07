"""Spacing-study variant (x_): methods.m1_das with S1's microsecond constants scaled by d / 4 cm (methods/x_s1scale.py).
Identical to m1_das at 4 cm."""
from methods import m1_das, x_s1scale


def enhance(x, fs, info, return_stages=False):
    with x_s1scale.scaled(x_s1scale.scale_of(info)):
        return m1_das.enhance(x, fs, info, return_stages=return_stages)
