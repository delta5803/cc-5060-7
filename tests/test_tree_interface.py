"""Tests for the explicit, versioned decomposition-tree interface.

These cover the refactor contract, not the signal-processing claims (those live
in test_wpt.py):

* the inverse transform reads an explicit tree version and refuses a stale one;
* a tree switch migrates edits along the ancestor path (remap) or rejects them;
* rejected edits roll back cleanly and never touch sibling branches;
* undo and persistence agree on stable node identity;
* the tree stays sparse (no pre-flattening), so coarse edit ownership survives.
"""
import numpy as np
import pytest

from soundsculpt.basis import Project, BasisSpec, SubtreeStroke, band_stroke_for_node
from soundsculpt.tree import (DecompositionTree, TreeSpec, NodeId, TreeVersion,
                              StaleTreeVersion, validate_leaf_basis)
from soundsculpt.edits import EditState, NodeEdit


@pytest.fixture
def signal():
    return np.random.default_rng(7).standard_normal(8192)


def _coarse_project(signal):
    # level-2 cut inside a level-4 tree, so 'ad' is a real, subdividable leaf
    return Project(signal, 44100, "x.wav",
                   BasisSpec("db4", 4, "symmetric",
                             leaves=["aa", "ad", "da", "dd"]))


# --------------------------------------------------------------- tree basics
def test_tree_is_sparse_not_flattened(signal):
    p = _coarse_project(signal)
    # coarser leaves: materialised nodes are only root, a, d and the four
    # level-2 leaves -- a full deepest tree would have 1+2+4+8+16 nodes.
    assert p.tree.nodes == ["", "a", "d", "aa", "ad", "da", "dd"]
    assert "aaaa" not in p.tree.nodes
    assert p.tree.leaves == ["aa", "ad", "da", "dd"]


def test_tree_version_changes_with_cut_but_not_identity(signal):
    p = _coarse_project(signal)
    v0 = p.tree.version.id
    t1 = p.tree.subdivide("ad")
    assert t1.version.id != v0
    # structural fingerprint is about geometry, not the cut
    assert t1.version.structural == p.tree.version.structural
    # old tree object is untouched (non-destructive)
    assert p.tree.leaves == ["aa", "ad", "da", "dd"]
    assert p.tree.version.id == v0


def test_leaf_basis_validation():
    with pytest.raises(ValueError, match="ancestor"):
        validate_leaf_basis(["a", "ad", "da", "dd"], 4)
    with pytest.raises(ValueError, match="incomplete"):
        validate_leaf_basis(["aaa", "aad"], 4)
    assert True


# --------------------------------------------------- explicit version in inverse
def test_reconstruct_pins_to_explicit_version(signal):
    p = _coarse_project(signal)
    eng = p.build_engine()
    rec = eng.reconstruct(p.tree.leaves, tree_version=p.tree.version.id)
    assert np.max(np.abs(rec - signal)) < 1e-10
    with pytest.raises(StaleTreeVersion):
        eng.reconstruct(p.tree.leaves, tree_version="deadbeef")


def test_coarse_leaf_is_reconstructed_without_flattening(signal):
    # A coarse leaf must reconstruct from *its own* coefficients; reconstructing
    # the fully-split level-4 tree must agree to floating point (no silent
    # disagreement about where the coarse edit is owned).
    from soundsculpt.wpt import all_paths
    p = _coarse_project(signal)
    eng = p.build_engine()
    coarse = eng.reconstruct(p.tree.leaves, tree_version=p.tree.version.id)
    full = eng.reconstruct(all_paths(4))
    assert np.max(np.abs(coarse - signal)) < 1e-10
    assert np.max(np.abs(full - signal)) < 1e-10
    assert np.max(np.abs(coarse - full)) < 1e-10


# ------------------------------------------------------------- migration rules
def test_subdivide_remaps_edit_along_ancestor_path(signal):
    p = _coarse_project(signal)
    p.add_node_edit("ad", 0.5, tag="k")
    before = p.evaluate_tree()
    rep = p.subdivide("ad")
    assert [a.action for a in rep.actions] == ["remap"]
    assert rep.actions[0].origin == "ancestor"
    assert [e.path for e in p.edits.edits] == ["ada", "add"]
    after = p.evaluate_tree()
    # same intent at a finer cut reproduces the same audio
    assert np.max(np.abs(before - after)) < 1e-10


def test_merge_up_rejects_and_rolls_back_without_sibling_bleed(signal):
    # fine cut with only 'ada' edited; coarsening merges ada+add into ad.
    p = Project(signal, 44100, "x.wav",
                BasisSpec("db4", 4, "symmetric",
                          leaves=["aa", "ada", "add", "da", "dd"]))
    p.add_node_edit("ada", 0.5, tag="fine")
    edited = p.evaluate_tree()
    assert np.max(np.abs(edited - signal)) > 1e-3

    rep = p.coarsen("ad")
    assert [a.action for a in rep.actions] == ["reject"]
    assert rep.actions[0].origin == "descendant"
    assert p.edits.edits == []                     # old-node edit undone
    after = p.evaluate_tree()
    # the coarsened render is the *original* signal: no sibling was touched
    assert np.max(np.abs(after - signal)) < 1e-10
    assert np.max(np.abs(after - edited)) > 1e-3


def test_no_comparable_node_rejects(signal):
    # full level-5 tree: 'adaad' is a real deepest leaf
    p = Project(signal, 44100, "x.wav", BasisSpec("db4", 5, "symmetric"))
    p.add_node_edit("adaad", 0.5, tag="deep")
    # shrink the tree above the edited node: no comparable leaf remains
    rep = p.rebase("db4", 3, "symmetric")
    assert p.edits.edits == []
    log = p.retarget_log[-1]
    assert log["migration"]["rejected"] == 1


def test_rebase_same_depth_keeps_edit_on_same_node(signal):
    p = _coarse_project(signal)
    p.add_node_edit("ad", 0.5, tag="k")
    p.rebase("sym6", 4, "symmetric")
    assert [e.path for e in p.edits.edits] == ["ad"]
    assert p.retarget_log[-1]["migration"]["remapped"] == 1


# ------------------------------------------------------ non-destructive eval
def test_original_tree_kept_for_non_destructive_evaluation(signal):
    p = _coarse_project(signal)
    p.add_node_edit("ad", 0.5, tag="k")
    old_tree, old_state = p.tree, p.edits
    old_audio = p.evaluate_tree(old_tree, old_state)
    p.subdivide("ad")                       # live tree moves on
    # the frozen old (tree, state) pair still evaluates to the same audio
    replay = p.evaluate_tree(old_tree, old_state)
    assert np.max(np.abs(replay - old_audio)) < 1e-12
    # and a mismatched (old tree, new state) pair is refused
    with pytest.raises(StaleTreeVersion):
        p.evaluate_tree(old_tree, p.edits)


# ------------------------------------------------------------- identity / undo
def test_undo_and_save_share_node_identity(signal, tmp_path):
    from soundsculpt.wavio import write_wav
    p = _coarse_project(signal)
    e = p.add_node_edit("ad", 0.5, tag="k")
    assert isinstance(e.node, NodeId)
    assert e.node.structural == p.tree.spec.fingerprint()
    assert e.node.path == "ad"

    write_wav(str(tmp_path / "x.wav"), signal, 44100)
    p.save(str(tmp_path / "p.json"))
    loaded = Project.load(str(tmp_path / "p.json"))
    # same node identity after a save/load round-trip
    assert loaded.edits.identity_index() == p.edits.identity_index()
    assert loaded.tree.version.id == p.tree.version.id
    # undo on the loaded project removes exactly that identity's edit
    undone = loaded.undo(tag="k")
    assert undone is not None and str(undone.node) == str(e.node)
    assert loaded.edits.edits == []


def test_undo_is_identity_scoped_not_offset_scoped(signal):
    p = _coarse_project(signal)
    a = p.add_node_edit("aa", 0.5, tag="a")
    b = p.add_node_edit("ad", 0.5, tag="b")
    # undo targeting node 'aa' must not touch 'ad' even though windows overlap
    got = p.undo(path="aa")
    assert got is not None and got.node == a.node
    assert [e.node for e in p.edits.edits] == [b.node]


def test_split_edit_undoes_atomically(signal):
    p = _coarse_project(signal)
    e = p.add_node_edit("ad", 0.5, tag="k")
    p.rebase(max_level=5)          # one record splits into many per-leaf ones
    assert len(p.edits.edits) > 1
    assert {x.op_id for x in p.edits.edits} == {e.op_id}
    undone = p.undo(tag="k")
    # the whole logical edit disappears, not just the last split fragment
    assert undone.op_id == e.op_id
    assert p.edits.edits == []


# ------------------------------------------------ boundary extension preserved
def test_boundary_extension_roundtrip_preserved():
    from soundsculpt.signals import boundary_impulses
    for mode in ("symmetric", "periodization", "zero", "reflect"):
        p = Project(boundary_impulses(4096, 44100), 44100, "b.wav",
                    BasisSpec("db4", 4, mode))
        rec = p.build_engine().reconstruct(p.tree.leaves,
                                           tree_version=p.tree.version.id)
        assert np.max(np.abs(rec - p.original)) < 1e-10, mode


# --------------------------------------------------------- render integration
def test_render_composes_node_edits_and_roundtrips_when_undone(signal):
    p = _coarse_project(signal)
    leaf = p.tree.leaves[0]
    p.add_node_edit(leaf, 0.3, tag="r")
    edited, _ = p.render()
    assert np.max(np.abs(edited - signal)) > 1e-3
    p.undo(tag="r")
    undone, _ = p.render()
    assert np.max(np.abs(undone - signal)) < 1e-10


def test_scale_switch_keeps_full_length_edit_equivalent(signal):
    # a full-length subtree edit has the same intent at any depth
    p = Project(signal, 44100, "x.wav",
                BasisSpec("db4", 3, "symmetric",
                          leaves=["aa", "ad", "da", "dd"]))
    p.add_node_edit("ad", 0.4, tag="k")
    before, _ = p.render()
    p.rebase(max_level=5)
    after, _ = p.render()
    assert np.max(np.abs(before - after)) < 1e-10
    assert all(e.path.startswith("ad") for e in p.edits.edits)


# ------------------------------------------------------------- web contract
def test_web_payload_exposes_tree_version_and_edits(signal):
    from soundsculpt.web import tree_payload, spectrum_payload
    p = _coarse_project(signal)
    p.add_node_edit("ad", 0.5, tag="k")
    payload = tree_payload(p)
    assert payload["tree_version"] == p.tree.version.id
    assert len(payload["edits"]) == 1
    spec = spectrum_payload(p, window=24)
    assert len(spec["spectrogram"]) == 24
    assert spec["tree_version"] == p.tree.version.id
    assert spec["bands"]  # multiscale bands present for the editor to draw


def test_web_edit_endpoint_rejects_non_leaf_and_reports_identity(signal):
    # exercise the same calls the HTTP handler makes, without a socket
    p = _coarse_project(signal)
    p.basis.subdivide("ad")          # direct mutation -> resync on next render
    p._ensure_tree()
    with pytest.raises(ValueError, match="not a leaf"):
        p.add_node_edit("ad", 0.5, tag="bad")
    rec = p.add_node_edit("ada", 0.5, tag="ok")
    assert rec.node.path == "ada"
    assert p.undo(tag="ok").node == rec.node
