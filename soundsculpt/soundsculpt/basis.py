"""Basis selection, stroke model, coefficient-edit state and rebase/retarget.

Two stroke kinds carry *different* semantics on purpose:

* BandStroke remembers a physical frequency band in Hz.  After a basis change
  it must keep covering that same band, even if the covering nodes change.
* SubtreeStroke remembers a node in the decomposition tree.  After a basis
  change it follows that node's descendants; if the node no longer exists the
  stroke is held unresolved until the user picks a new target.

Both strokes also carry an optional time window in seconds and a linear gain.

Three pieces of state are kept apart so a tree change only invalidates what
the change actually touched:

* :class:`BasisSpec` — basis-function configuration + the editable leaf cut
  (delegating cut arithmetic to :class:`~soundsculpt.tree.DecompositionTree`);
* :class:`DecompositionTree` — structural identity and version (see ``tree``);
* :class:`EditState` — the coefficient edits actually applied, each keyed by a
  stable :class:`NodeId`.  Undo and save both read this same identity.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from .wpt import (WPTEngine, all_paths, nominal_band, descendants)
from .tree import DecompositionTree, TreeSpec
from .edits import EditState, MigrationReport, NodeEdit


# --------------------------------------------------------------------- basis
@dataclass
class BasisSpec:
    """Basis-function configuration plus the editable leaf cut.

    The cut operations delegate to :class:`DecompositionTree`, but ``BasisSpec``
    keeps the historical field layout so saved projects and callers that pass
    ``BasisSpec(wavelet, max_level, mode, leaves)`` keep working.  The
    structural tree (a versioned :class:`~soundsculpt.tree.DecompositionTree`)
    is built on demand via :meth:`tree`.
    """
    wavelet: str = "db4"
    max_level: int = 5
    mode: str = "symmetric"
    leaves: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.leaves:
            self.leaves = all_paths(self.max_level)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BasisSpec":
        return cls(d["wavelet"], int(d["max_level"]), d["mode"], list(d["leaves"]))

    def tree(self, sr: int, n: int) -> DecompositionTree:
        """Build the explicit, versioned tree for this basis and geometry."""
        return DecompositionTree(TreeSpec(self.wavelet, self.max_level,
                                          self.mode, sr, n), list(self.leaves))

    def subdivide(self, node: str) -> None:
        """Refine one node one level, whether or not it is a current leaf.

        Delegates to :meth:`DecompositionTree.subdivide`, so the cut rules live
        in exactly one place.
        """
        self.leaves = self._cut().subdivide(node).leaves

    def coarsen(self, pair_parent: str) -> None:
        """Merge a sibling pair back into their parent (local coarsening)."""
        self.leaves = self._cut().coarsen(pair_parent).leaves

    def _cut(self) -> DecompositionTree:
        # Structural cut arithmetic does not depend on sr/n, so a nominal
        # geometry is enough here; the real tree is built where audio is known.
        return DecompositionTree(TreeSpec(self.wavelet, self.max_level,
                                          self.mode, 1, 1 << self.max_level),
                                 list(self.leaves))


# -------------------------------------------------------------------- strokes
@dataclass
class BandStroke:
    """Circle a fixed physical frequency band (Hz) over a time window."""
    id: str
    band_hz: tuple[float, float]
    gain: float
    t0: float = 0.0
    t1: float | None = None  # None => end of file
    kind: str = "band"

    def to_dict(self) -> dict:
        return {"kind": "band", "id": self.id,
                "band_hz": list(self.band_hz), "gain": self.gain,
                "t0": self.t0, "t1": self.t1}

    @classmethod
    def from_dict(cls, d: dict) -> "BandStroke":
        return cls(d["id"], tuple(d["band_hz"]), float(d["gain"]),
                   float(d.get("t0", 0.0)), d.get("t1"))


@dataclass
class SubtreeStroke:
    """Push down the whole subtree rooted at a decomposition node."""
    id: str
    node_path: str
    gain: float
    t0: float = 0.0
    t1: float | None = None
    origin_band_hz: tuple[float, float] | None = None
    kind: str = "subtree"

    def to_dict(self) -> dict:
        return {"kind": "subtree", "id": self.id, "node_path": self.node_path,
                "gain": self.gain, "t0": self.t0, "t1": self.t1,
                "origin_band_hz": (list(self.origin_band_hz)
                                   if self.origin_band_hz else None)}

    @classmethod
    def from_dict(cls, d: dict) -> "SubtreeStroke":
        ob = d.get("origin_band_hz")
        return cls(d["id"], d["node_path"], float(d["gain"]),
                   float(d.get("t0", 0.0)), d.get("t1"),
                   tuple(ob) if ob else None)


def stroke_from_dict(d: dict):
    return BandStroke.from_dict(d) if d["kind"] == "band" else SubtreeStroke.from_dict(d)


# ------------------------------------------------------------------ resolution
@dataclass
class CoeffEdit:
    path: str
    i0: int
    i1: int
    gain: float
    stroke_id: str


@dataclass
class UnresolvedStroke:
    stroke_id: str
    reason: str
    candidates: list[str]


@dataclass
class ResolutionReport:
    edits: list[CoeffEdit] = field(default_factory=list)
    unresolved: list[UnresolvedStroke] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _band_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def resolve_strokes(engine: WPTEngine, basis: BasisSpec,
                    strokes: list) -> ResolutionReport:
    report = ResolutionReport()
    duration = engine.n / engine.sr
    for st in strokes:
        t1 = duration if st.t1 is None else min(st.t1, duration)
        if isinstance(st, BandStroke):
            hits = []
            for leaf in basis.leaves:
                lb = nominal_band(leaf, engine.sr)
                ov = _band_overlap(st.band_hz, lb)
                if ov <= 0:
                    continue
                i0, i1 = engine.time_window_to_indices(leaf, st.t0, t1)
                if i1 > i0:
                    report.edits.append(CoeffEdit(leaf, i0, i1, st.gain, st.id))
                    hits.append((leaf, lb, ov))
            if not hits:
                report.unresolved.append(UnresolvedStroke(
                    st.id, "no leaf intersects the requested band",
                    _band_candidates(engine, basis, st.band_hz)))
            else:
                covered = _coverage_fraction(st.band_hz, hits)
                report.notes.append(
                    f"{st.id}: band {_fmt_band(st.band_hz)} covered by "
                    f"{len(hits)} leaf/leaves, nominal coverage {covered:.1%}")
        else:  # SubtreeStroke
            if len(st.node_path) > basis.max_level:
                report.unresolved.append(UnresolvedStroke(
                    st.id,
                    f"node {st.node_path} is deeper than new max_level "
                    f"{basis.max_level}",
                    _band_candidates(engine, basis,
                                     st.origin_band_hz
                                     or nominal_band(st.node_path, engine.sr))))
                continue
            # Resolve the node against the actual leaf basis:
            #  - leaves at/under the node, or
            #  - a coarser leaf that contains the node (after retargeting
            #    onto an internal node).
            target_leaves = [lf for lf in basis.leaves
                             if lf == st.node_path
                             or lf.startswith(st.node_path)
                             or st.node_path.startswith(lf)]
            if not target_leaves:
                report.unresolved.append(UnresolvedStroke(
                    st.id, f"node {st.node_path} has no covering leaf",
                    _band_candidates(engine, basis,
                                     st.origin_band_hz
                                     or nominal_band(st.node_path, engine.sr))))
                continue
            for leaf in target_leaves:
                i0, i1 = engine.time_window_to_indices(leaf, st.t0, t1)
                if i1 > i0:
                    report.edits.append(CoeffEdit(leaf, i0, i1, st.gain, st.id))
            report.notes.append(
                f"{st.id}: subtree {st.node_path} follows "
                f"{len(target_leaves)} current leaf/leaves {target_leaves}")
    return report


def _band_candidates(engine: WPTEngine, basis: BasisSpec,
                     band: tuple[float, float], k: int = 4) -> list[str]:
    scored = []
    for leaf in basis.leaves:
        lb = nominal_band(leaf, engine.sr)
        scored.append((_band_overlap(band, lb), leaf))
    scored.sort(reverse=True)
    return [p for _, p in scored[:k]]


def _coverage_fraction(band: tuple[float, float],
                       hits: list) -> float:
    lo = min(h[1][0] for h in hits)
    hi = max(h[1][1] for h in hits)
    union = max(0.0, hi - lo)
    target = band[1] - band[0]
    return min(1.0, union / target) if target else 0.0


def _fmt_band(b: tuple[float, float]) -> str:
    return f"{b[0]:.0f}-{b[1]:.0f} Hz"


def apply_edits(engine: WPTEngine, edits: list[CoeffEdit],
                basis: BasisSpec, *,
                tree_version: str | None = None) -> np.ndarray:
    # Compose gains per (path, coefficient) so overlapping strokes multiply
    # rather than overwrite.  Edits start from the engine's current data.
    staged: dict[str, np.ndarray] = {}
    for e in edits:
        if e.path not in staged:
            staged[e.path] = engine.coeffs(e.path).copy()
        staged[e.path][e.i0:e.i1] *= e.gain
    for path, data in staged.items():
        engine.set_coeffs(path, data)
    # Reconstruction is pinned to an explicit tree version whenever one is
    # supplied, so the inverse transform cannot read an ambient/different cut.
    return engine.reconstruct(basis.leaves, tree_version=tree_version)


# -------------------------------------------------------------------- project
class Project:
    """A sound-sculpting session: original audio + basis + strokes."""

    def __init__(self, signal: np.ndarray, sr: int, audio_name: str,
                 basis: BasisSpec | None = None,
                 strokes: list | None = None):
        self.original = np.asarray(signal, dtype=np.float64).copy()
        self.sr = sr
        self.audio_name = audio_name
        self.basis = basis or BasisSpec()
        self.strokes: list = strokes or []
        self.retarget_log: list[dict] = []
        # The explicit decomposition tree is the single source of truth for
        # the leaf cut and its version; renders read *this* tree, never an
        # ambient default.  Rebuilt whenever the basis geometry changes.
        self.tree = self.basis.tree(self.sr, self.original.size)
        # Coefficient-edit state (node-identity keyed) separate from strokes.
        self.edits = EditState(self.tree.version)

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "audio_name": self.audio_name,
            "sr": self.sr,
            "n": int(self.original.size),
            "basis": self.basis.to_dict(),
            "tree_version": self.tree.version.id,
            "strokes": [s.to_dict() for s in self.strokes],
            "edits": self.edits.to_dict(),
            "retarget_log": self.retarget_log,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, audio_dir: str | None = None) -> "Project":
        from .wavio import read_wav
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        audio_dir = audio_dir or os.path.dirname(path)
        sig, sr = read_wav(os.path.join(audio_dir, d["audio_name"]))
        if sr != d["sr"] or sig.size != d["n"]:
            raise ValueError("audio file does not match project metadata")
        obj = cls(sig, sr, d["audio_name"], BasisSpec.from_dict(d["basis"]),
                  [stroke_from_dict(s) for s in d["strokes"]])
        obj.retarget_log = d.get("retarget_log", [])
        if "edits" in d:
            obj.edits = EditState.from_dict(d["edits"])
        return obj

    # ---------------------------------------------------------------- rendering
    def build_engine(self) -> WPTEngine:
        return WPTEngine(self.original, self.basis.wavelet,
                         self.basis.max_level, self.basis.mode, self.sr)

    def sync_tree(self) -> DecompositionTree:
        """Rebuild the explicit tree from the current basis + geometry.

        Rebuilds the tree object only.  Callers that change the cut are
        responsible for migrating edit state (see :meth:`subdivide`,
        :meth:`coarsen`, :meth:`rebase`); :meth:`_ensure_tree` handles direct
        ``basis.leaves`` mutations.
        """
        self.tree = self.basis.tree(self.sr, self.original.size)
        return self.tree

    def _ensure_tree(self) -> MigrationReport | None:
        """Re-sync if ``basis.leaves`` was mutated behind the tree's back.

        Returns the migration report when a re-sync happened, else ``None``.
        This keeps direct ``proj.basis.subdivide(...)`` callers working while
        still pinning every render to one explicit tree version.
        """
        if sorted(self.tree.leaves) != sorted(self.basis.leaves):
            new_tree = self.basis.tree(self.sr, self.original.size)
            self.edits, migration = self.edits.migrate(new_tree)
            self.tree = new_tree
            return migration
        return None

    def subdivide(self, node: str) -> MigrationReport:
        """Refine a node and migrate edit state to the new tree version."""
        new_tree = self.tree.subdivide(node)
        self.basis.leaves = new_tree.leaves
        self.edits, migration = self.edits.migrate(new_tree)
        self.tree = new_tree
        return migration

    def coarsen(self, pair_parent: str) -> MigrationReport:
        """Merge a leaf sibling pair and migrate edit state to the new version."""
        new_tree = self.tree.coarsen(pair_parent)
        self.basis.leaves = new_tree.leaves
        self.edits, migration = self.edits.migrate(new_tree)
        self.tree = new_tree
        return migration

    def render(self) -> tuple[np.ndarray, ResolutionReport]:
        self._ensure_tree()
        engine = self.build_engine()
        report = resolve_strokes(engine, self.basis, self.strokes)
        if report.unresolved:
            # Never silently drop strokes: render the last fully-resolved
            # intent by raising; callers surface this to the user.
            ids = ", ".join(u.stroke_id for u in report.unresolved)
            raise UnresolvedError(ids, report)
        # Node edits (identity-keyed) compose with stroke edits; both are
        # staged in one pass so overlapping gains multiply consistently.
        edits = list(report.edits) + self.edits.resolve(engine, self.tree)
        out = apply_edits(engine, edits, self.basis,
                          tree_version=self.tree.version.id)
        return out, report

    def dry_run(self) -> ResolutionReport:
        self._ensure_tree()
        engine = self.build_engine()
        return resolve_strokes(engine, self.basis, self.strokes)

    # ------------------------------------------------------- explicit edit state
    def add_node_edit(self, path: str, gain: float, t0: float = 0.0,
                      t1: float | None = None, tag: str = "") -> NodeEdit:
        """Record a coefficient edit keyed by the current tree's node identity.

        This is the identity-preserving path used by undo and save: the record
        is bound to a :class:`NodeId`, not to a transient array offset.
        """
        self._ensure_tree()
        return self.edits.add(self.tree, path, gain, t0, t1, tag)

    def undo(self, tag: str | None = None, path: str | None = None
             ) -> NodeEdit | None:
        """Undo the most recent edit for ``tag`` / node ``path`` (identity-keyed).

        The comparison is by NodeId, the same identity the saved record stores,
        so undo cannot accidentally hit a different node with a similar offset.
        """
        node = self.tree.node_id(path) if path is not None else None
        return self.edits.undo(tag=tag, node=node)

    def evaluate_tree(self, tree: DecompositionTree | None = None,
                      state: EditState | None = None) -> np.ndarray:
        """Non-destructive evaluation against a specific tree version.

        The project's live basis/strokes are not mutated: a separate engine is
        built for the tree's own geometry, the (optionally migrated) edit state
        is resolved against the supplied tree, and reconstruction is pinned to
        that tree's version and leaf cut.
        """
        tree = tree or self.tree
        state = state or self.edits
        if state.version.id != tree.version.id:
            from .tree import StaleTreeVersion
            raise StaleTreeVersion(
                f"edit state is bound to tree {state.version.id[:12]} but "
                f"evaluation tree is {tree.version.id[:12]}")
        engine = WPTEngine(self.original, tree.spec.wavelet, tree.spec.max_level,
                           tree.spec.mode, tree.spec.sr)
        edits = state.resolve(engine, tree)
        cut = BasisSpec(tree.spec.wavelet, tree.spec.max_level, tree.spec.mode,
                        leaves=tree.leaves)
        return apply_edits(engine, edits, cut, tree_version=tree.version.id)

    # ------------------------------------------------------------------ rebase
    def _remap_leaves(self, old_leaves: list[str], new_level: int) -> list[str]:
        """Map a custom leaf basis onto a new maximum depth.

        A leaf shallower than the new depth is pushed to its descendants at
        the new depth; a leaf deeper than it is replaced by its ancestor leaf
        at the new depth (the subtree stroke resolution machinery then flags
        any stroke that needs an explicit retarget).
        """
        remapped = set()
        for leaf in old_leaves:
            if len(leaf) == new_level:
                remapped.add(leaf)
            elif len(leaf) < new_level:
                remapped.update(descendants(leaf, new_level))
            else:
                remapped.add(leaf[:new_level])
        return sorted(remapped)

    def rebase(self, wavelet: str | None = None, max_level: int | None = None,
               mode: str | None = None) -> ResolutionReport:
        """Change wavelet packet basis and re-resolve every stroke.

        A custom (locally subdivided) leaf basis is preserved: same-depth
        leaves carry over, shallower ones expand to descendants, deeper ones
        collapse to their ancestor.  Subtree strokes that cannot be mapped
        stay on the project but are reported unresolved; call retarget().
        """
        new_level = (max_level if max_level is not None
                     else self.basis.max_level)
        custom = self.basis.leaves != all_paths(self.basis.max_level)
        if custom and new_level == self.basis.max_level:
            # wavelet/mode change only: keep the leaf cut exactly as-is
            new_leaves = list(self.basis.leaves)
        elif custom:
            new_leaves = self._remap_leaves(self.basis.leaves, new_level)
        else:
            new_leaves = all_paths(new_level)
        new_basis = BasisSpec(
            wavelet or self.basis.wavelet, new_level,
            mode or self.basis.mode, leaves=new_leaves)
        old = self.basis
        old_tree = self.tree
        self.basis = new_basis
        self.sync_tree()
        engine = self.build_engine()

        # Migrate coefficient-edit state along the ancestor path; rejected
        # old-node edits are rolled back rather than applied to a stranger.
        new_edits, migration = self.edits.migrate(self.tree)
        self.edits = new_edits

        # Record origin bands for subtree strokes (once).
        for st in self.strokes:
            if isinstance(st, SubtreeStroke) and st.origin_band_hz is None:
                st.origin_band_hz = nominal_band(st.node_path, self.sr)

        report = resolve_strokes(engine, new_basis, self.strokes)
        self.retarget_log.append({
            "from": old.to_dict(), "to": new_basis.to_dict(),
            "old_version": old_tree.version.id,
            "new_version": self.tree.version.id,
            "migration": migration.to_dict(),
            "unresolved": [u.stroke_id for u in report.unresolved]})
        return report

    def retarget(self, stroke_id: str, new_node_path: str) -> None:
        """Point an unresolved subtree stroke at a new node."""
        for st in self.strokes:
            if isinstance(st, SubtreeStroke) and st.id == stroke_id:
                if len(new_node_path) > self.basis.max_level:
                    raise ValueError("target node deeper than current max_level")
                self.retarget_log.append(
                    {"retarget": stroke_id, "old": st.node_path,
                     "new": new_node_path})
                st.node_path = new_node_path
                st.origin_band_hz = nominal_band(new_node_path, self.sr)
                return
        raise KeyError(f"no subtree stroke with id {stroke_id}")

    def add_stroke(self, stroke) -> None:
        if any(s.id == stroke.id for s in self.strokes):
            raise ValueError(f"stroke id {stroke.id} already exists")
        self.strokes.append(stroke)


class UnresolvedError(RuntimeError):
    def __init__(self, ids: str, report: ResolutionReport):
        super().__init__(f"unresolved strokes: {ids}")
        self.report = report


# Convenience: build a band stroke that circles one node's nominal band.
def band_stroke_for_node(stroke_id: str, path: str, sr: int,
                         gain: float, t0: float = 0.0,
                         t1: float | None = None) -> BandStroke:
    return BandStroke(stroke_id, nominal_band(path, sr), gain, t0, t1)
