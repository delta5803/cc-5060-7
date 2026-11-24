"""Wavelet-packet path geometry (no state, no transform).

Kept in its own module so the explicit decomposition tree can depend on path
arithmetic without importing the compatibility engine in ``wpt``.

Convention: ``'a'`` = low-pass branch, ``'d'`` = high-pass branch, root ``''``.
Frequencies are cyclic (half-band) frequencies in ``[0, 0.5] * sr``.
"""
from __future__ import annotations

import numpy as np
import pywt


def all_paths(depth: int) -> list[str]:
    """Natural-order node paths at a fixed depth."""
    if depth == 0:
        return [""]
    out = [""]
    for _ in range(depth):
        out = [p + b for p in out for b in ("a", "d")]
    return out


def freq_order_paths(depth: int) -> list[str]:
    """Gray-code order in which fixed-depth leaves sort by ascending band."""
    if depth == 0:
        return [""]
    n = 2 ** max(depth, 4)  # pywt needs enough samples for the full tree
    wp = pywt.WaveletPacket(np.zeros(n), "db1", mode="periodization",
                            maxlevel=depth)
    return [n.path for n in wp.get_level(depth, "freq")]


def band_index(path: str) -> int:
    """Frequency-sorted index 0..2**d-1 of a node within its depth.

    Matches pywt's 'freq' ordering.  Recursively, child index = (parent << 1)
    with the new low bit equal to the parent's low bit on an 'a' branch and
    its complement on a 'd' branch.
    """
    k = 0
    for branch in path:
        bit = k & 1
        k = (k << 1) | (bit if branch == "a" else 1 - bit)
    return k


def path_from_band_index(index: int, depth: int) -> str:
    """Inverse of band_index: build the natural-order path for band k."""
    bits = [(index >> (depth - 1 - i)) & 1 for i in range(depth)]
    out = []
    for i, bit in enumerate(bits):
        prev = 0 if i == 0 else bits[i - 1]
        out.append("a" if bit == prev else "d")
    return "".join(out)


def nominal_band(path: str, sr: int) -> tuple[float, float]:
    """Nominal dyadic band [lo, hi) in Hz for a node.

    WPT frequency ordering is a Gray-code permutation of the uniform dyadic
    grid, not recursive halving: a 'd' branch does not always keep the upper
    half of its parent's band.  The band is therefore read from the node's
    frequency index.  These are *nominal* bands; real wavelet filters overlap
    them (see leakage_profile).
    """
    depth = len(path)
    width = (0.5 * sr) / (2 ** depth) if depth else 0.5 * sr
    k = band_index(path) if depth else 0
    return k * width, (k + 1) * width


def descendants(path: str, max_depth: int) -> list[str]:
    if len(path) >= max_depth:
        return [path]
    out = []
    for p in all_paths(max_depth - len(path)):
        out.append(path + p)
    return out
