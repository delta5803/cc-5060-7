"""Basis selection, stroke model and rebase/retarget logic.

Two stroke kinds carry *different* semantics on purpose:

* BandStroke remembers a physical frequency band in Hz.  After a basis change
  it must keep covering that same band, even if the covering nodes change.
* SubtreeStroke remembers a node in the decomposition tree.  After a basis
  change it follows that node's descendants; if the node no longer exists the
  stroke is held unresolved until the user picks a new target.

Both strokes also carry an optional time window in seconds and a linear gain.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from .wpt import (WPTEngine, all_paths, nominal_band, descendants)


# --------------------------------------------------------------------- basis
@dataclass
class BasisSpec:
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

    def subdivide(self, node: str) -> None:
        """Refine one node one level, whether or not it is a current leaf.

        If it is a leaf, replace it by its two children.  If it is an
        internal node, replace every current leaf under it by that leaf's
        own two children (the whole subtree gains one level of editable
        nodes).  Reconstruction stays exact throughout.
        """
        if len(node) >= self.max_level:
            raise ValueError(f"{node} is already at the maximum depth")
        if node in self.leaves:
            i = self.leaves.index(node)
            self.leaves[i:i + 1] = [node + "a", node + "d"]
            return
        under = [lf for lf in self.leaves if lf.startswith(node)]
        if not under:
            raise ValueError(f"{node} has no covering leaf in the basis")
        # leaves already at the maximum depth cannot be split further; only
        # the leaves that still have room gain a level
        splittable = [lf for lf in under if len(lf) < self.max_level]
        if not splittable:
            raise ValueError(f"subtree under {node} is already at max depth")
        new_leaves = []
        for lf in self.leaves:
            if lf in splittable:
                new_leaves.extend([lf + "a", lf + "d"])
            else:
                new_leaves.append(lf)
        self.leaves = new_leaves

    def coarsen(self, pair_parent: str) -> None:
        """Merge a sibling pair back into their parent (local coarsening)."""
        a, d = pair_parent + "a", pair_parent + "d"
        if a not in self.leaves or d not in self.leaves:
            raise ValueError(f"{a}/{d} are not both current leaves")
        i = self.leaves.index(a)
        self.leaves[i:i + 2] = [pair_parent]


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
                basis: BasisSpec) -> np.ndarray:
    # Compose gains per (path, coefficient) so overlapping strokes multiply
    # rather than overwrite.  Edits start from the engine's current data.
    staged: dict[str, np.ndarray] = {}
    for e in edits:
        if e.path not in staged:
            staged[e.path] = engine.coeffs(e.path).copy()
        staged[e.path][e.i0:e.i1] *= e.gain
    for path, data in staged.items():
        engine.set_coeffs(path, data)
    return engine.reconstruct(basis.leaves)


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

    # ------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "audio_name": self.audio_name,
            "sr": self.sr,
            "n": int(self.original.size),
            "basis": self.basis.to_dict(),
            "strokes": [s.to_dict() for s in self.strokes],
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
        return obj

    # ---------------------------------------------------------------- rendering
    def build_engine(self) -> WPTEngine:
        return WPTEngine(self.original, self.basis.wavelet,
                         self.basis.max_level, self.basis.mode, self.sr)

    def render(self) -> tuple[np.ndarray, ResolutionReport]:
        engine = self.build_engine()
        report = resolve_strokes(engine, self.basis, self.strokes)
        if report.unresolved:
            # Never silently drop strokes: render the last fully-resolved
            # intent by raising; callers surface this to the user.
            ids = ", ".join(u.stroke_id for u in report.unresolved)
            raise UnresolvedError(ids, report)
        out = apply_edits(engine, report.edits, self.basis)
        return out, report

    def dry_run(self) -> ResolutionReport:
        engine = self.build_engine()
        return resolve_strokes(engine, self.basis, self.strokes)

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
        self.basis = new_basis
        engine = self.build_engine()

        # Record origin bands for subtree strokes (once).
        for st in self.strokes:
            if isinstance(st, SubtreeStroke) and st.origin_band_hz is None:
                st.origin_band_hz = nominal_band(st.node_path, self.sr)

        report = resolve_strokes(engine, new_basis, self.strokes)
        self.retarget_log.append({
            "from": old.to_dict(), "to": new_basis.to_dict(),
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
