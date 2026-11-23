"""Wavelet packet engine built on PyWavelets.

The engine deliberately exposes the *non-ideal* geometry of the representation:
nominal dyadic frequency bands, decimated time support, edge-extension behaviour
and measured basis-function support/ringing.  WPT coefficients are not a
brick-wall filter bank, and nothing in this module pretends they are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

# Path convention: 'a' = low-pass branch, 'd' = high-pass branch, root = ''.
# Frequencies are cyclic (half-band) frequencies in [0, 0.5] * sr.


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


def descendants(path: str, max_depth: int) -> list[str]:
    if len(path) >= max_depth:
        return [path]
    out = []
    for p in all_paths(max_depth - len(path)):
        out.append(path + p)
    return out


@dataclass
class NodeInfo:
    path: str
    depth: int
    n_coeffs: int
    band_hz: tuple[float, float]
    decimation: int            # input samples per coefficient step
    nominal_step_sec: float    # nominal time between two coefficients
    nominal_span_sec: float    # nominal window covered by one coefficient
    support_samples: tuple[int, int] | None      # interior basis support
    support_sec: float | None
    edge_coeffs: tuple[int, int] | None          # coeffs touching each edge
    edge_support_samples: tuple[int, int] | None  # union incl. extension wrap


class WPTEngine:
    """Fixed wavelet packet decomposition of one signal."""

    def __init__(self, signal: np.ndarray, wavelet: str, max_level: int,
                 mode: str, sr: int):
        self.signal = np.asarray(signal, dtype=np.float64).copy()
        self.n = self.signal.size
        if wavelet not in pywt.wavelist(kind="discrete"):
            raise ValueError(f"unknown wavelet {wavelet!r}")
        if mode not in pywt.Modes.modes:
            raise ValueError(f"unknown extension mode {mode!r}")
        if max_level < 1:
            raise ValueError("max_level must be >= 1")
        self.wavelet = wavelet
        self.max_level = max_level
        self.mode = mode
        self.sr = sr
        self.wp = pywt.WaveletPacket(self.signal, wavelet, mode=mode,
                                     maxlevel=max_level)

    # ------------------------------------------------------------------ basic
    def coeffs(self, path: str) -> np.ndarray:
        return self.wp[path].data

    def set_coeffs(self, path: str, data: np.ndarray) -> None:
        node = self.wp[path]
        if data.shape != node.data.shape:
            raise ValueError(f"shape mismatch for {path}: "
                             f"{data.shape} vs {node.data.shape}")
        node.data = np.asarray(data, dtype=np.float64)

    def node_info(self, path: str, measure_support: bool = True) -> NodeInfo:
        depth = len(path)
        n_coeffs = self.coeffs(path).size
        dec = 2 ** depth
        band = nominal_band(path, self.sr)
        step = dec / self.sr
        span = (self.n / n_coeffs) / self.sr if n_coeffs else 0.0
        support = edge = edge_support = None
        if measure_support:
            lo, hi, elo, ehi, union = self._measure_support(path)
            support = (lo, hi)
            edge = (elo, ehi)
            edge_support = tuple(union)
        return NodeInfo(path=path, depth=depth, n_coeffs=n_coeffs,
                        band_hz=band, decimation=dec,
                        nominal_step_sec=step, nominal_span_sec=span,
                        support_samples=support,
                        support_sec=None if support is None
                        else (support[1] - support[0] + 1) / self.sr,
                        edge_coeffs=edge,
                        edge_support_samples=edge_support)

    # ------------------------------------------------------------- time maps
    def time_window_to_indices(self, path: str, t0: float, t1: float
                               ) -> tuple[int, int]:
        """Map a time range in seconds to a coefficient index range [i0, i1).

        The coefficient array includes extension padding (its length differs
        from the ideal N / 2**depth under non-periodization modes), so time
        is mapped with the actual coefficient density n_coeffs / duration.
        The extra padded coefficients are physically tied to the signal
        boundaries and are included when the window reaches them.  Real basis
        functions still spread over the measured support window.
        """
        n_coeffs = self.coeffs(path).size
        duration = self.n / self.sr
        i0 = int(np.floor(t0 / duration * n_coeffs))
        i1 = int(np.ceil(t1 / duration * n_coeffs))
        i0 = max(0, min(i0, n_coeffs))
        i1 = max(0, min(i1, n_coeffs))
        if i1 < i0:
            i1 = i0
        return i0, i1

    def indices_to_time_window(self, path: str, i0: int, i1: int
                               ) -> tuple[float, float]:
        n_coeffs = self.coeffs(path).size
        duration = self.n / self.sr
        return (i0 / n_coeffs * duration,
                i1 / n_coeffs * duration)

    # ------------------------------------------------------------- reconstruction
    def reconstruct(self, leaf_paths: list[str] | None = None) -> np.ndarray:
        """Reconstruct from a valid leaf basis (or the full deepest tree).

        Coarse leaves are pushed down to max_level by forward-transforming
        their coefficients.  This guarantees no node is counted together with
        its ancestor/descendant: the pushed-down leaves form a disjoint
        partition of the deepest level.
        """
        if leaf_paths is None:
            leaf_paths = all_paths(self.max_level)
        self._validate_leaf_basis(leaf_paths)
        zero = pywt.WaveletPacket(np.zeros_like(self.signal), self.wavelet,
                                  mode=self.mode, maxlevel=self.max_level)
        for leaf in leaf_paths:
            data = self.coeffs(leaf)
            depth = self.max_level - len(leaf)
            if depth == 0:
                zero[leaf].data = data.copy()
            else:
                sub = pywt.WaveletPacket(data, self.wavelet, mode=self.mode,
                                         maxlevel=depth)
                for fn in sub.get_level(depth, "natural"):
                    zero[leaf + fn.path].data = fn.data.copy()
        return zero.reconstruct(update=False)

    def node_contribution(self, path: str) -> np.ndarray:
        """Signal reconstructed from the subtree under one node only.

        Implemented as deepest-level zeroing, so it matches reconstruct()
        exactly and works for internal nodes too.
        """
        backup = {p: self.coeffs(p).copy()
                  for p in all_paths(self.max_level)}
        try:
            keep = set(descendants(path, self.max_level))
            for p in backup:
                if p not in keep:
                    self.wp[p].data = np.zeros_like(backup[p])
            return self.wp.reconstruct(update=False)
        finally:
            for p, d in backup.items():
                self.wp[p].data = d

    def _validate_leaf_basis(self, leaf_paths: list[str]) -> None:
        if not leaf_paths:
            raise ValueError("empty leaf basis")
        # disjointness: no path is a prefix of another
        s = sorted(leaf_paths, key=len)
        for i, p in enumerate(s):
            for q in s[i + 1:]:
                if q.startswith(p):
                    raise ValueError(
                        f"invalid basis: {p} is an ancestor of {q}; a node "
                        "and its descendants cannot both be reconstructed")
        # coverage: every deepest leaf has exactly one ancestor in the basis
        covered = set()
        for leaf in leaf_paths:
            covered.update(descendants(leaf, self.max_level))
        full = set(all_paths(self.max_level))
        if covered != full:
            missing = sorted(full - covered)
            raise ValueError(f"incomplete leaf basis, missing under {missing[:4]}")

    # ------------------------------------------------------------- support/edges
    def _measure_support(self, path: str, tol: float = 1e-9
                         ) -> tuple[int, int, int, int, tuple[int, int]]:
        """Measure real basis-function support via impulse responses.

        Returns (interior_min, interior_max, n_edge_left, n_edge_right,
        union_support).  Every coefficient impulse is measured.  Under
        symmetric/reflective extension an impulse near a boundary is mirrored
        but stays local; under periodic extension an edge coefficient wraps to
        the opposite side and spans the whole array.  Interior support is the
        median impulse span, i.e. the filter-derived basis width, while the
        edge counts expose how many coefficients are boundary-shaped.
        """
        n_coeffs = self.coeffs(path).size
        spans = []
        edge_l = edge_r = 0
        union_lo, union_hi = self.n, 0
        edge_band = max(2, n_coeffs // 32)
        for i in range(n_coeffs):
            # fresh packet every time: reusing one leaves stale coefficients
            # behind at other nodes and contaminates the impulse
            pkt = pywt.WaveletPacket(np.zeros(self.n), self.wavelet,
                                     mode=self.mode, maxlevel=self.max_level)
            d = np.zeros(n_coeffs)
            d[i] = 1.0
            pkt[path].data = d
            imp = pkt.reconstruct(update=False)
            peak = np.max(np.abs(imp))
            idx = np.flatnonzero(np.abs(imp) > max(tol, 1e-7 * peak))
            if idx.size == 0:
                continue
            lo_i, hi_i = int(idx[0]), int(idx[-1])
            union_lo = min(union_lo, lo_i)
            union_hi = max(union_hi, hi_i)
            spans.append((lo_i, hi_i, hi_i - lo_i + 1))
            if i < edge_band and lo_i == 0:
                edge_l += 1
            if i >= n_coeffs - edge_band and hi_i >= self.n - 1:
                edge_r += 1
        widths = np.array([s[2] for s in spans])
        # the modal width is the filter-derived interior basis width; edge
        # impulses are narrower (symmetric) or wrap to full length (periodic)
        vals, counts = np.unique(widths, return_counts=True)
        modal_width = int(vals[int(np.argmax(counts))])
        typical = next(s for s in spans if s[2] == modal_width)
        # interior support: the typical basis width, anchored at a central
        # impulse rather than the union across all shifted positions
        center = typical[0] + modal_width // 2
        lo = max(0, center - modal_width // 2)
        hi = min(self.n - 1, lo + modal_width - 1)
        return lo, hi, edge_l, edge_r, (union_lo, union_hi)

    # ------------------------------------------------------------- introspection
    def tree_table(self) -> list[NodeInfo]:
        out = []
        for depth in range(self.max_level + 1):
            for p in all_paths(depth):
                # support measurement only for modest trees; callers can ask
                out.append(self.node_info(p, measure_support=False))
        return out

    def leaf_table(self, measure_support: bool = True) -> list[NodeInfo]:
        return [self.node_info(p, measure_support=measure_support)
                for p in all_paths(self.max_level)]

    def leakage_profile(self, path: str, n_bins: int | None = None
                        ) -> np.ndarray:
        """Fraction of a node's basis energy landing in each nominal band.

        Bins follow the node's *own* depth by default (2**depth dyadic bands).
        Uses a white-noise argument via the node contribution's FFT.  Rows
        sum to 1; an ideal brick wall would be a one-hot row.  Under edge-
        redundant extension modes the profile is biased near boundaries and
        periodization is preferred for an unbiased leakage picture.
        """
        depth = len(path)
        n_bins = n_bins or 2 ** depth
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(self.n)
        eng = WPTEngine(noise, self.wavelet, self.max_level, self.mode,
                        self.sr)
        contribution = eng.node_contribution(path)
        spec = np.fft.rfft(contribution)
        freqs = np.fft.rfftfreq(self.n, 1.0 / self.sr)
        widths = (0.5 * self.sr) / n_bins
        energy = np.zeros(n_bins)
        for k in range(n_bins):
            lo = k * widths
            hi = (k + 1) * widths if k < n_bins - 1 else 0.5 * self.sr
            m = (freqs >= lo) & (freqs <= hi)
            energy[k] = np.sum(np.abs(spec[m]) ** 2)
        total = energy.sum()
        return energy / total if total > 0 else energy
