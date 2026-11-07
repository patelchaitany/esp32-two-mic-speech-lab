"""Spacing-study variant (x_): methods.m5_fence with S1's microsecond constants scaled by d / 4 cm (methods/x_s1scale.py).
Identical to m5_fence at 4 cm."""
from methods import m5_fence, x_s1scale


def enhance(x, fs, info, return_stages=False):
    with x_s1scale.scaled(x_s1scale.scale_of(info)):
        return m5_fence.enhance(x, fs, info, return_stages=return_stages)
