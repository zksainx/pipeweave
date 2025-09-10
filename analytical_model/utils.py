#!/usr/bin/env python3
"""Common utility functions for analytical model calculators."""


def ceil_div(x: int, y: int) -> int:
    """Ceiling division"""
    return (x + y - 1) // y


def gcd(a: int, b: int) -> int:
    """Greatest common divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b
    return a
