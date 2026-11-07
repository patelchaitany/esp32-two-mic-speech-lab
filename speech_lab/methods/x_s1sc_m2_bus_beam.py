"""Spacing-study variant (x_): methods.m2_bus_beam with S1's microsecond constants scaled by d / 4 cm (methods/x_s1scale.py).
Identical to m2_bus_beam at 4 cm."""
from methods import m2_bus_beam, x_s1scale


def enhance(x, fs, info, return_stages=False):
    with x_s1scale.scaled(x_s1scale.scale_of(info)):
        return m2_bus_beam.enhance(x, fs, info, return_stages=return_stages)
