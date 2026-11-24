"""Explicit, versioned wavelet-packet decomposition tree.

This module is the new boundary between the transform and everything that
edits it.  It separates three things that used to be entangled in a single
mutable ``BasisSpec``/``WPTEngine`` pair:

* a :class:`TreeSpec` — the *basis-function configuration* (wavelet, depth,
  extension mode, sample rate, length);
* a :class:`DecompositionTree` — the *decomposition structure*: a frozen leaf
  cut plus the stable identities of the nodes that cut materialises;
* a :class:`TreeVersion` — the content hash of the structure, so an inverse
  transform can insist on reconstructing exactly the version it was given.

The tree is deliberately **sparse**: it stores the leaf cut and derives the
materialised node set from it.  It never flattens the tree to the deepest
level, because pre-flattening erases which coarse node an edit was owned by
(and therefore its identity for undo / save / migration).

Node identity is deterministic: ``NodeId(structural fingerprint, path)``.  The
structural fingerprint depends on the transform configuration, not on time or
allocation order, so an identity survives save/load and undo/redo unchanged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pywt

from .paths import all_paths, descendants


# --------------------------------------------------------------------- errors
class StaleTreeVersion(RuntimeError):
    """Raised when an inverse transform is asked for a mismatched version."""


# ----------------------------------------------------------------------- spec
@dataclass(frozen=True)
class TreeSpec:
    """Basis-function configuration (no edit state, no coefficients)."""
    wavelet: str
    max_level: int
    mode: str
    sr: int
    n: int

    def __post_init__(self) -> None:
        if self.wavelet not in pywt.wavelist(kind="discrete"):
            raise ValueError(f"unknown wavelet {self.wavelet!r}")
        if self.mode not in pywt.Modes.modes:
            raise ValueError(f"unknown extension mode {self.mode!r}")
        if self.max_level < 1:
            raise ValueError("max_level must be >= 1")
        if self.sr <= 0 or self.n <= 0:
            raise ValueError("sr and n must be positive")

    def canonical(self) -> str:
        return (f"{self.wavelet}|{self.max_level}|{self.mode}|"
                f"{self.sr}|{self.n}")

    def fingerprint(self) -> str:
        """Structural identity: same value => same coefficient arrays/shapes."""
        return hashlib.sha1(self.canonical().encode()).hexdigest()


@dataclass(frozen=True)
class NodeId:
    """Stable identity of one node within one structural configuration."""
    structural: str
    path: str

    def __str__(self) -> str:  # deterministic, used in save/undo comparisons
        return f"{self.structural[:12]}:{self.path or '<root>'}"


@dataclass(frozen=True)
class TreeVersion:
    """A frozen (spec, leaf cut) pair with a content-hash identity."""
    spec: TreeSpec
    leaves: tuple[str, ...]

    @property
    def structural(self) -> str:
        return self.spec.fingerprint()

    @property
    def id(self) -> str:
        body = self.structural + "|" + ",".join(self.leaves)
        return hashlib.sha1(body.encode()).hexdigest()


# ---------------------------------------------------------------- validation
def validate_leaf_basis(leaves: list[str], max_level: int) -> None:
    """Reject overlap (ancestor+descendant) and incomplete coverage."""
    if not leaves:
        raise ValueError("empty leaf basis")
    s = sorted(leaves, key=len)
    for i, p in enumerate(s):
        for q in s[i + 1:]:
            if q.startswith(p):
                raise ValueError(
                    f"invalid basis: {p} is an ancestor of {q}; a node "
                    "and its descendants cannot both be reconstructed")
    covered: set[str] = set()
    for leaf in leaves:
        if len(leaf) > max_level:
            raise ValueError(f"leaf {leaf} deeper than max_level {max_level}")
        covered.update(descendants(leaf, max_level))
    full = set(all_paths(max_level))
    if covered != full:
        missing = sorted(full - covered)
        raise ValueError(f"incomplete leaf basis, missing under {missing[:4]}")


# ----------------------------------------------------------------------- tree
class DecompositionTree:
    """A sparse, versioned decomposition structure.

    ``leaves`` is the cut used for reconstruction.  Everything else (the
    materialised nodes, their identities) is derived.  Mutating operations
    return a *new* tree with a *new* version; the old tree object is unchanged,
    which is what lets callers keep an original tree for non-destructive
    evaluation.
    """

    def __init__(self, spec: TreeSpec, leaves: list[str] | None = None):
        if leaves is None:
            leaves = all_paths(spec.max_level)
        validate_leaf_basis(list(leaves), spec.max_level)
        self.spec = spec
        self._leaves: tuple[str, ...] = tuple(sorted(leaves))
        self._version = TreeVersion(spec, self._leaves)
        # Sparse materialisation: every strict prefix of a leaf plus the leaf.
        nodes: set[str] = set()
        for leaf in self._leaves:
            for i in range(len(leaf) + 1):
                nodes.add(leaf[:i])
        self._nodes: frozenset[str] = frozenset(nodes)

    # ------------------------------------------------------------ structure
    @property
    def version(self) -> TreeVersion:
        return self._version

    @property
    def leaves(self) -> list[str]:
        return list(self._leaves)

    @property
    def nodes(self) -> list[str]:
        """Materialised nodes (internal + leaves), never a full deepest tree."""
        return sorted(self._nodes, key=lambda p: (len(p), p))

    def has_node(self, path: str) -> bool:
        return path in self._nodes

    def is_leaf(self, path: str) -> bool:
        return path in self._leaves

    def node_id(self, path: str) -> NodeId:
        return NodeId(self.spec.fingerprint(), path)

    def covering_leaf(self, path: str) -> str | None:
        """Unique current leaf comparable to ``path`` (ancestor *or* descendant).

        This is the "ancestor origin" relation used when a tree changes: an
        edit whose origin is comparable to exactly one new leaf can be
        remapped onto it; otherwise it has no ancestor source and is rejected.
        """
        if path == "":
            return None  # root is never a leaf (max_level >= 1)
        for lf in self._leaves:
            if lf == path or path.startswith(lf) or lf.startswith(path):
                return lf
        return None

    def remap_targets(self, path: str) -> list[str]:
        """Leaves an edit on ``path`` should migrate onto in this cut."""
        if path == "":
            return list(self._leaves)
        under = [lf for lf in self._leaves if lf.startswith(path)]
        if under:
            return sorted(under)
        cover = self.covering_leaf(path)
        return [cover] if cover is not None else []

    def children(self, path: str) -> list[str]:
        return [path + b for b in ("a", "d")]

    # ------------------------------------------------------------ transforms
    def _derived(self, leaves: list[str]) -> "DecompositionTree":
        return DecompositionTree(self.spec, leaves)

    def subdivide(self, node: str) -> "DecompositionTree":
        """Refine one node one level; returns a new versioned tree."""
        if len(node) >= self.spec.max_level:
            raise ValueError(f"{node} is already at the maximum depth")
        leaves = list(self._leaves)
        if node in leaves:
            i = leaves.index(node)
            leaves[i:i + 1] = list(self.children(node))
            return self._derived(leaves)
        under = [lf for lf in leaves if lf.startswith(node)]
        if not under:
            raise ValueError(f"{node} has no covering leaf in the basis")
        splittable = [lf for lf in under if len(lf) < self.spec.max_level]
        if not splittable:
            raise ValueError(f"subtree under {node} is already at max depth")
        out: list[str] = []
        for lf in leaves:
            if lf in splittable:
                out.extend(self.children(lf))
            else:
                out.append(lf)
        return self._derived(out)

    def coarsen(self, pair_parent: str) -> "DecompositionTree":
        a, d = pair_parent + "a", pair_parent + "d"
        leaves = list(self._leaves)
        if a not in leaves or d not in leaves:
            raise ValueError(f"{a}/{d} are not both current leaves")
        i = leaves.index(a)
        leaves[i:i + 2] = [pair_parent]
        return self._derived(leaves)

    def with_leaves(self, leaves: list[str]) -> "DecompositionTree":
        return self._derived(list(leaves))

    # ------------------------------------------------------- reconstruction
    def reconstruct(self, engine, *,
                    expect_version: str | None = None) -> np.ndarray:
        """Reconstruct using *this tree's frozen version*.

        The engine must describe the same signal geometry.  Passing an
        ``expect_version`` that does not match the tree raises instead of
        silently reconstructing a different cut — the inverse transform never
        reads an implicit/ambient tree.
        """
        if expect_version is not None and expect_version != self.version.id:
            raise StaleTreeVersion(
                f"tree is at version {self.version.id[:12]}, "
                f"caller expects {str(expect_version)[:12]}")
        if (engine.wavelet, engine.max_level, engine.mode,
                engine.sr, engine.n) != (self.spec.wavelet, self.spec.max_level,
                                         self.spec.mode, self.spec.sr,
                                         self.spec.n):
            raise ValueError("engine geometry does not match tree spec")
        return engine.reconstruct(list(self._leaves),
                                  tree_version=self.version.id)


# ------------------------------------------------------------- version helper
def version_id_for(wavelet: str, max_level: int, mode: str, sr: int, n: int,
                   leaves: list[str]) -> str:
    """Content hash for a (config, cut) pair, without building a tree."""
    spec = TreeSpec(wavelet, max_level, mode, sr, n)
    return TreeVersion(spec, tuple(sorted(leaves))).id
