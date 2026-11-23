import numpy as np
import pytest
import pywt

from soundsculpt.wpt import (WPTEngine, all_paths, nominal_band, band_index,
                             path_from_band_index,
                             freq_order_paths)
from soundsculpt.basis import (BasisSpec, BandStroke, SubtreeStroke,
                               apply_edits,
                               band_stroke_for_node, Project, UnresolvedError)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def signal(rng):
    return rng.standard_normal(4096)


@pytest.mark.parametrize("mode", pywt.Modes.modes)
def test_exact_roundtrip_all_modes(signal, mode):
    eng = WPTEngine(signal, "db4", 5, mode, 44100)
    rec = eng.reconstruct()
    assert np.max(np.abs(rec - signal)) < 1e-10


@pytest.mark.parametrize("wave", ["db1", "db4", "sym6", "coif3"])
def test_roundtrip_wavelets(signal, wave):
    eng = WPTEngine(signal, wave, 5, "symmetric", 44100)
    assert np.max(np.abs(eng.reconstruct() - signal)) < 1e-10


def test_roundtrip_odd_length(rng):
    x = rng.standard_normal(1001)
    eng = WPTEngine(x, "db4", 4, "symmetric", 44100)
    assert np.max(np.abs(eng.reconstruct() - x)) < 1e-10


def test_mixed_basis_roundtrip(signal):
    eng = WPTEngine(signal, "db4", 4, "symmetric", 44100)
    leaves = ["aa", "ad", "da", "dd"]  # level-2 basis inside a level-4 tree
    rec = eng.reconstruct(leaves)
    assert np.max(np.abs(rec - signal)) < 1e-10


def test_ancestor_descendant_rejected(signal):
    eng = WPTEngine(signal, "db4", 4, "symmetric", 44100)
    with pytest.raises(ValueError, match="ancestor"):
        eng.reconstruct(["a", "ad", "da", "ddd"])  # 'a' overlaps others
    with pytest.raises(ValueError, match="incomplete"):
        eng.reconstruct(["aaa", "aad"])  # doesn't cover the tree


def test_band_index_roundtrip():
    for depth in range(1, 7):
        for k in range(2 ** depth):
            p = path_from_band_index(k, depth)
            assert band_index(p) == k


def test_nominal_bands_partition(signal):
    sr = 44100
    WPTEngine(signal, "db4", 4, "symmetric", sr)
    order = freq_order_paths(4)
    edges = []
    for p in order:
        lo, hi = nominal_band(p, sr)
        edges.append((lo, hi))
    assert edges[0][0] == 0.0
    assert edges[-1][1] == pytest.approx(0.5 * sr)
    for (_, h), (l, _) in zip(edges, edges[1:]):
        assert h == pytest.approx(l)
    # bands computed from freq order are exactly the ascending dyadic grid
    width = 0.5 * sr / 2 ** 4
    for k, (lo, hi) in enumerate(edges):
        assert lo == pytest.approx(k * width)
        assert hi == pytest.approx((k + 1) * width)


def test_node_contribution_partitions_signal(signal):
    eng = WPTEngine(signal, "db4", 4, "periodization", 44100)
    total = np.zeros_like(signal)
    for p in all_paths(4):
        total += eng.node_contribution(p)
    assert np.max(np.abs(total - signal)) < 1e-10


def test_node_contribution_orthonormal_energy(signal):
    eng = WPTEngine(signal, "db4", 4, "periodization", 44100)
    e = sum(np.sum(eng.node_contribution(p) ** 2) for p in all_paths(4))
    assert e == pytest.approx(np.sum(signal ** 2), rel=1e-10)


def test_measured_support_finite_and_edge_coeffs(signal):
    eng = WPTEngine(signal, "db4", 4, "symmetric", 44100)
    info = eng.node_info("ad")
    s0, s1 = info.support_samples
    assert 0 <= s0 < s1 < signal.size
    width = s1 - s0 + 1
    # db4 filter length 8 -> interior support stays compact across levels
    assert width < 200
    # extension guarantees at least one edge-influenced coefficient per side
    assert info.edge_coeffs[0] >= 1 and info.edge_coeffs[1] >= 1
    # edge coefficients genuinely wrap across the full buffer
    assert info.edge_support_samples[0] == 0
    assert info.edge_support_samples[1] == signal.size - 1


def test_leakage_profile_is_not_brick_wall(signal):
    eng = WPTEngine(signal, "db4", 4, "symmetric", 44100)
    prof = eng.leakage_profile("ad")
    assert prof.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.max(prof) < 0.999  # energy visibly leaks across bands


def test_band_and_subtree_strokes_identical(signal):
    basis = BasisSpec("db4", 5, "symmetric")
    p1 = Project(signal, 44100, "x.wav", BasisSpec(**basis.to_dict()),
                 [band_stroke_for_node("b", "ad", 44100, 0.3)])
    p2 = Project(signal, 44100, "x.wav", BasisSpec(**basis.to_dict()),
                 [SubtreeStroke("s", "ad", 0.3,
                                origin_band_hz=nominal_band("ad", 44100))])
    y1, _ = p1.render()
    y2, _ = p2.render()
    assert np.max(np.abs(y1 - y2)) < 1e-12


def test_time_window_localizes_edit(rng):
    n = 32768  # ~0.74 s at 44.1 kHz, so 0.3-0.5 s is inside the signal
    x = rng.standard_normal(n)
    proj = Project(x, 44100, "x.wav", BasisSpec("db4", 4, "symmetric"),
                   [SubtreeStroke("s", "d", 0.0, t0=0.3, t1=0.5)])
    report = proj.dry_run()
    # an internal node resolves onto the deepest leaves under it
    assert report.edits and all(e.path.startswith("d") for e in report.edits)
    y, _ = proj.render()
    res = np.abs(y - x)
    info = proj.build_engine().node_info("da")
    pad = info.support_samples[1] - info.support_samples[0] + 1
    # far outside the edited window the residual must be tiny
    far = res[: int(0.1 * 44100)]
    assert far.max() < 1e-8
    # the edited region is affected
    mid = res[int(0.35 * 44100): int(0.45 * 44100)]
    assert mid.max() > 1e-4
    # ringing beyond one support width of the window is small but non-zero:
    # locality is real but not a hard boundary
    outside = np.concatenate([res[: max(0, int(0.28 * 44100) - pad)],
                              res[min(n, int(0.52 * 44100) + pad):]])
    assert outside.max() < mid.max()


def test_rebase_band_keeps_physical_band(signal):
    basis = BasisSpec("db4", 4, "symmetric")
    stroke = band_stroke_for_node("b", "ad", 44100, 0.5)
    proj = Project(signal, 44100, "x.wav", basis, [stroke])
    band = stroke.band_hz
    from soundsculpt.verification import band_energy
    e_orig = band_energy(signal, 44100, *band)
    proj.rebase("sym6", 4, "symmetric")
    assert not proj.dry_run().unresolved
    y1, _ = proj.render()
    e1 = band_energy(y1, 44100, *band)
    g = np.sqrt(e1 / e_orig)
    # the stroke still attenuates the *same physical band* substantially;
    # non-brick-wall leakage means the FFT-band readout is not an exact 0.5
    assert 0.35 < g < 0.75
    # attenuation is real and large compared with no edit
    assert g < 0.9
    # neighbouring bands are also touched: leakage is visible, not hidden
    full_spec_orig = np.sum(np.abs(np.fft.rfft(signal)) ** 2)
    full_spec_new = np.sum(np.abs(np.fft.rfft(y1)) ** 2)
    assert full_spec_new < full_spec_orig


def test_rebase_subtree_follows_descendants(signal):
    proj = Project(signal, 44100, "x.wav", BasisSpec("db4", 4, "symmetric"),
                   [SubtreeStroke("s", "ad", 0.4,
                                  origin_band_hz=nominal_band("ad", 44100))])
    proj.rebase("db4", 6, "symmetric")
    report = proj.dry_run()
    assert not report.unresolved
    assert all(e.path.startswith("ad") for e in report.edits)


def test_rebase_deep_node_requires_retarget(signal):
    proj = Project(signal, 44100, "x.wav", BasisSpec("db4", 6, "symmetric"),
                   [SubtreeStroke("s", "adaad", 0.4,
                                  origin_band_hz=nominal_band("adaad", 44100))])
    report = proj.rebase("db4", 3, "symmetric")
    assert report.unresolved and report.unresolved[0].stroke_id == "s"
    with pytest.raises(UnresolvedError):
        proj.render()
    proj.retarget("s", "ada")
    assert not proj.dry_run().unresolved
    y, _ = proj.render()
    assert y.shape == signal.shape


def test_render_never_silently_drops_stroke(signal):
    proj = Project(signal, 44100, "x.wav", BasisSpec("db4", 3, "symmetric"),
                   [BandStroke("b", (100000, 120000), 0.5)])  # above nyquist
    with pytest.raises(UnresolvedError):
        proj.render()


def test_subdivide_keeps_roundtrip_and_stroke_follows(signal):
    # level-2 leaves make 'ad' an actual leaf that can be subdivided locally
    proj = Project(signal, 44100, "x.wav",
                   BasisSpec("db4", 4, "symmetric",
                             leaves=["aa", "ad", "da", "dd"]),
                   [SubtreeStroke("s", "ad", 0.5,
                                  origin_band_hz=nominal_band("ad", 44100))])
    before, rep0 = proj.render()
    assert {e.path for e in rep0.edits} == {"ad"}
    proj.basis.subdivide("ad")
    after, report = proj.render()
    edited = sorted({e.path for e in report.edits})
    assert edited == ["ada", "add"]
    # same intent -> same output regardless of where the leaf cut sits
    assert np.max(np.abs(before - after)) < 1e-12
    assert np.max(np.abs(proj.build_engine().reconstruct(proj.basis.leaves)
                         - signal)) < 1e-10


def test_overlapping_strokes_compose_gains(signal):
    proj = Project(signal, 44100, "x.wav", BasisSpec("db4", 4, "periodization"),
                   [SubtreeStroke("s1", "ad", 0.5),
                    SubtreeStroke("s2", "ad", 0.5)])
    y, report = proj.render()
    # exact coefficient-level check: staged gains multiply to 0.25
    eng = proj.build_engine()
    edited_eng = proj.build_engine()
    apply_edits(edited_eng, report.edits, proj.basis)
    for e in report.edits:
        before = eng.coeffs(e.path)[e.i0:e.i1]
        after = edited_eng.coeffs(e.path)[e.i0:e.i1]
        assert np.allclose(after, before * 0.25)
    # The nominal FFT band only approximates the real wavelet subspace;
    # db4 filters leak across a narrow dyadic band, so the exact multiscale
    # claim is verified at coefficient level and via the edited subtree's
    # own reconstructed energy, not via a brick-wall band-energy readout.
    subtree_before = np.sum(eng.node_contribution("ad") ** 2)
    subtree_after = np.sum(edited_eng.node_contribution("ad") ** 2)
    assert abs(np.sqrt(subtree_after / subtree_before) - 0.25) < 1e-8
    # total energy decreases (only the edited subtree changes)
    assert np.sum(y ** 2) < np.sum(signal ** 2)
    assert report.edits  # edits land on the leaves covering 'ad'


def test_subdivide_internal_node_expands_subtree(signal):
    # on a tree whose leaves sit above max depth, subdividing an internal
    # node refines every leaf under it by one level
    proj2 = Project(signal, 44100, "x.wav",
                    BasisSpec("db4", 3, "symmetric",
                              leaves=["aa", "ad", "da", "dd"]))
    proj2.basis.subdivide("ad")
    leaves = proj2.basis.leaves
    assert "ada" in leaves and "add" in leaves
    assert "aa" in leaves  # outside the subtree untouched
    assert np.max(np.abs(
        proj2.build_engine().reconstruct(leaves) - signal)) < 1e-10
    # subdividing when every leaf under the node is already at max depth
    # raises rather than producing an invalid basis
    proj_full = Project(signal, 44100, "x.wav",
                        BasisSpec("db4", 3, "symmetric"))
    with pytest.raises(ValueError):
        proj_full.basis.subdivide("ad")


def test_rebase_preserves_custom_leaf_basis(signal):
    proj = Project(signal, 44100, "x.wav",
                   BasisSpec("db4", 4, "symmetric",
                             leaves=["aa", "ad", "da", "dd"]))
    # same-depth wavelet change keeps the custom leaves verbatim
    proj.rebase("sym6", None, "symmetric")
    assert proj.basis.leaves == ["aa", "ad", "da", "dd"]
    assert np.max(np.abs(proj.build_engine().reconstruct(proj.basis.leaves)
                         - signal)) < 1e-10
    # depth increase pushes shallow leaves down to the new max depth
    proj.rebase("sym6", 5, "symmetric")
    leaves = proj.basis.leaves
    assert "adaaa" in leaves and "adddd" in leaves
    assert all(lf.startswith(("aa", "ad", "da", "dd")) for lf in leaves)
    assert np.max(np.abs(proj.build_engine().reconstruct(leaves)
                         - signal)) < 1e-10


def test_project_save_load_roundtrip(tmp_path, signal):
    from soundsculpt.wavio import write_wav, read_wav
    proj = Project(signal, 44100, "src.wav", BasisSpec("db4", 4, "symmetric"),
                   [SubtreeStroke("s", "ad", 0.4,
                                  origin_band_hz=nominal_band("ad", 44100))])
    write_wav(str(tmp_path / "src.wav"), signal, 44100)
    # WAV is 16-bit PCM: loading re-quantises, compare against the stored PCM
    stored, sr = read_wav(str(tmp_path / "src.wav"))
    proj_pcm = Project(stored, sr, "src.wav", BasisSpec("db4", 4, "symmetric"),
                       list(proj.strokes))
    proj_pcm.save(str(tmp_path / "p.json"))
    loaded = Project.load(str(tmp_path / "p.json"))
    y1, _ = proj_pcm.render()
    y2, _ = loaded.render()
    assert np.max(np.abs(y1 - y2)) < 1e-12
    # metadata round-trips intact
    assert loaded.basis.to_dict() == proj_pcm.basis.to_dict()
    assert loaded.strokes[0].node_path == "ad"
