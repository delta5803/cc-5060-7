"""Coefficient-edit state, migration and undo.

Separated from the basis configuration (:class:`~soundsculpt.basis.BasisSpec`)
and from the decomposition structure (:class:`~soundsculpt.tree.DecompositionTree`)
on purpose.  This module owns *what was actually changed in the coefficients*:

* each edit is keyed by a stable :class:`~soundsculpt.tree.NodeId`, so undo and
  save/load agree on node identity;
* an edit stores its intent in seconds (``t0``/``t1``) so it can be re-resolved
  against a different node geometry after a tree change;
* :meth:`EditState.migrate` moves edits along the ancestor path (remap) or
  rejects them when the new tree has neither an ancestor nor a descendant of
  the edited node.  Rejected old-node edits are rolled back — they are not
  silently applied to an unrelated node, and sibling branches are never
  touched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tree import DecompositionTree, NodeId, TreeVersion


# ------------------------------------------------------------------- records
@dataclass
class CoeffEdit:
    """One resolved coefficient range edit, ready to apply to an engine."""
    path: str
    i0: int
    i1: int
    gain: float
    stroke_id: str
    node_id: NodeId | None = None


@dataclass(frozen=True)
class NodeEdit:
    """A persistent edit keyed by node identity, in time-window form."""
    node: NodeId
    t0: float
    t1: float | None
    gain: float
    tag: str = ""
    op_id: str = ""          # logical operation; migration splits share it

    @property
    def path(self) -> str:
        return self.node.path

    def to_dict(self) -> dict:
        return {"node": {"structural": self.node.structural,
                         "path": self.node.path},
                "t0": self.t0, "t1": self.t1, "gain": self.gain,
                "tag": self.tag, "op_id": self.op_id}

    @classmethod
    def from_dict(cls, d: dict) -> "NodeEdit":
        nd = d["node"]
        return cls(NodeId(nd["structural"], nd["path"]),
                   float(d["t0"]), d.get("t1"), float(d["gain"]),
                   d.get("tag", ""), d.get("op_id", ""))


# ------------------------------------------------------------------ migration
@dataclass
class MigrationAction:
    old_path: str
    new_paths: list[str]
    origin: str            # 'same' | 'ancestor' | 'descendant'
    action: str            # 'remap' | 'reject'
    reason: str = ""

    def to_dict(self) -> dict:
        return {"old_path": self.old_path, "new_paths": list(self.new_paths),
                "origin": self.origin, "action": self.action,
                "reason": self.reason}


@dataclass
class MigrationReport:
    actions: list[MigrationAction] = field(default_factory=list)

    @property
    def remapped(self) -> list[MigrationAction]:
        return [a for a in self.actions if a.action == "remap"]

    @property
    def rejected(self) -> list[MigrationAction]:
        return [a for a in self.actions if a.action == "reject"]

    def to_dict(self) -> dict:
        return {"actions": [a.to_dict() for a in self.actions],
                "remapped": len(self.remapped),
                "rejected": len(self.rejected)}


def _origin(old_path: str, new_paths: list[str]) -> str:
    """Classify how old node relates to its new targets.

    ``ancestor``   — the old node is an ancestor of every target (a split):
                     the edit's source is higher in the tree.
    ``descendant`` — a new target is an ancestor of the old node (a merge):
                     the edit's source is lower; it maps up to the cover.
    ``same``       — the node survived unchanged.
    """
    if not new_paths:
        return "none"
    if len(new_paths) == 1 and new_paths[0] == old_path:
        return "same"
    if all(t.startswith(old_path) and t != old_path for t in new_paths):
        return "ancestor"
    if all(old_path.startswith(t) and t != old_path for t in new_paths):
        return "descendant"
    return "mixed"


# ---------------------------------------------------------------------- state
class EditState:
    """Persistent coefficient edits bound to one tree version."""

    def __init__(self, version: TreeVersion, edits: list[NodeEdit] | None = None):
        self.version = version
        self.edits: list[NodeEdit] = list(edits) if edits else []

    # --------------------------------------------------------------- mutation
    def add(self, tree: DecompositionTree, path: str, gain: float,
            t0: float = 0.0, t1: float | None = None,
            tag: str = "", op_id: str = "") -> NodeEdit:
        if not tree.is_leaf(path):
            raise ValueError(
                f"cannot edit {path!r}: not a leaf of the current tree "
                f"(leaves={tree.leaves[:6]}{'...' if len(tree.leaves) > 6 else ''})")
        if not op_id:
            op_id = f"op{len({e.op_id for e in self.edits}) + 1}"
        edit = NodeEdit(tree.node_id(path), t0, t1, gain, tag, op_id)
        self.edits.append(edit)
        return edit

    def undo(self, tag: str | None = None,
             node: NodeId | None = None) -> NodeEdit | None:
        """Remove the most recent logical edit matching ``tag`` / ``node``.

        Migration splits one logical edit into several per-leaf records that
        share an ``op_id``; undo removes the whole group so a split edit is
        undone atomically rather than one leaf at a time.
        """
        for i in range(len(self.edits) - 1, -1, -1):
            e = self.edits[i]
            if (tag is None or e.tag == tag) and (node is None or e.node == node):
                group = e.op_id
                removed = [x for x in self.edits
                           if x.op_id == group and (tag is None or x.tag == tag)]
                self.edits = [x for x in self.edits if x not in removed]
                return e
        return None

    # -------------------------------------------------------------- resolving
    def resolve(self, engine, tree: DecompositionTree) -> list[CoeffEdit]:
        """Resolve every record into concrete coefficient ranges on ``tree``."""
        duration = engine.n / engine.sr
        out: list[CoeffEdit] = []
        for e in self.edits:
            if not tree.is_leaf(e.path):
                continue  # stale record; migrate() is the path that rebinds
            t1 = duration if e.t1 is None else min(e.t1, duration)
            i0, i1 = engine.time_window_to_indices(e.path, e.t0, t1)
            if i1 > i0:
                out.append(CoeffEdit(e.path, i0, i1, e.gain, e.tag or "edit",
                                     e.node))
        return out

    # -------------------------------------------------------------- migration
    def migrate(self, new_tree: DecompositionTree
                ) -> tuple["EditState", MigrationReport]:
        """Rebind edits to ``new_tree`` by ancestor origin.

        An edit may only survive when the old node still exists as an
        *ancestor* of the new leaves (a split, or an unchanged node): the new
        leaves partition the old node's subspace, so copying the same time
        window onto each descendant leaf preserves the intent exactly.

        If the new tree only has a coarser *ancestor of the old node* (a merge
        / depth decrease), remapping the fine edit upward would apply it to the
        siblings it never touched.  That edit is **rejected** — the old-node
        edit is undone — and no sibling coefficient is modified.  The receiver
        is left untouched so callers may keep it for non-destructive evaluation.
        """
        report = MigrationReport()
        new_edits: list[NodeEdit] = []
        for e in self.edits:
            targets = new_tree.remap_targets(e.path)
            if not targets:
                report.actions.append(MigrationAction(
                    e.path, [], "none", "reject",
                    "no ancestor or descendant in the new tree"))
                continue
            origin = _origin(e.path, targets)
            if origin in ("same", "ancestor"):
                for t in targets:
                    new_edits.append(NodeEdit(new_tree.node_id(t), e.t0, e.t1,
                                              e.gain, e.tag, e.op_id))
                report.actions.append(MigrationAction(
                    e.path, targets, origin, "remap"))
            else:  # 'descendant' (merge up) or 'mixed': not faithful
                report.actions.append(MigrationAction(
                    e.path, targets, origin, "reject",
                    "old node is finer than every new leaf; remapping it up "
                    "would alter sibling branches"))
        return EditState(new_tree.version, new_edits), report

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"version_id": self.version.id,
                "structural": self.version.structural,
                "leaves": list(self.version.leaves),
                "edits": [e.to_dict() for e in self.edits]}

    @classmethod
    def from_dict(cls, d: dict) -> "EditState":
        version = LoadedVersion(d.get("version_id", ""),
                                d.get("structural", ""),
                                tuple(d.get("leaves", [])))
        return cls(version, [NodeEdit.from_dict(x) for x in d.get("edits", [])])

    def identity_index(self) -> dict[str, int]:
        """Node-identity -> edit count, the map undo and save both agree on."""
        idx: dict[str, int] = {}
        for e in self.edits:
            idx[str(e.node)] = idx.get(str(e.node), 0) + 1
        return idx


@dataclass(frozen=True)
class LoadedVersion:
    """A version record restored from disk (no live TreeSpec required).

    It exposes the same ``id`` / ``structural`` / ``leaves`` surface as
    :class:`~soundsculpt.tree.TreeVersion` so loaded state compares equal to a
    freshly built tree by content hash, not by object identity.
    """
    id: str
    structural: str
    leaves: tuple[str, ...]
