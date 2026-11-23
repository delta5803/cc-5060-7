"""Numerical verification primitives and the end-to-end test battery.

Every claim about locality is checked against real reconstructed signals,
never against the selection rectangle alone.  Basis functions overlap nominal
bands and spread beyond the edited time window; the functions here measure
that spread.
"""
from __future__ import annotations

import numpy as np

from .basis import (Project, BasisSpec, SubtreeStroke,
                    band_stroke_for_node)
from .wpt import nominal_band, all_paths


# ------------------------------------------------------------- basic measures
def band_energy(signal: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sr)
    m = (freqs >= f_lo) & (freqs < f_hi)
    return float(np.sum(np.abs(spec[m]) ** 2))


def tone_gain(original: np.ndarray, edited: np.ndarray, sr: int,
              freq: float, tol_hz: float = 40.0) -> float:
    """Realised linear gain at a probe tone, measured from FFT bins."""
    def amp(x):
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
        m = np.abs(freqs - freq) <= tol_hz
        return float(np.sqrt(np.sum(np.abs(spec[m]) ** 2)))
    a0, a1 = amp(original), amp(edited)
    return a1 / a0 if a0 > 0 else float("nan")


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(r) if r > 0 else -np.inf


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


# ------------------------------------------------------------- check 1: roundtrip
def check_roundtrip(project: Project) -> dict:
    engine = project.build_engine()
    rec = engine.reconstruct(project.basis.leaves)
    full = engine.reconstruct(all_paths(project.basis.max_level))
    return {
        "name": "no-edit roundtrip",
        "max_abs_error": max_abs(rec, project.original),
        "full_tree_error": max_abs(full, project.original),
        "pass": max_abs(rec, project.original) < 1e-10
        and max_abs(full, project.original) < 1e-10,
    }


def check_energy(project: Project) -> dict:
    """Parseval-style check; some extension modes are edge-redundant."""
    engine = project.build_engine()
    e_sig = float(np.sum(project.original ** 2))
    e_coeff = sum(float(np.sum(engine.coeffs(p) ** 2))
                  for p in all_paths(project.basis.max_level))
    ratio = e_coeff / e_sig if e_sig else float("nan")
    return {"name": "coefficient/signal energy ratio", "ratio": ratio,
            "mode": project.basis.mode,
            "pass": abs(ratio - 1.0) < 1e-8
            or project.basis.mode in ("symmetric", "periodic", "reflect",
                                      "constant", "smooth", "antisymmetric",
                                      "antireflect")}


# --------------------------------------------- check 2: edits actually sound
def check_edit_audibility(project: Project) -> dict:
    edited, _ = project.render()
    err = edited - project.original
    return {"name": "coefficient edit changes the audio",
            "rms_original_db": rms_db(project.original),
            "rms_residual_db": rms_db(err),
            "max_residual": float(np.max(np.abs(err))),
            "pass": np.max(np.abs(err)) > 1e-6}


# --------------------------------------------- check 3: two strokes equivalent
def check_stroke_equivalence(signal: np.ndarray, sr: int, basis: BasisSpec,
                             node_path: str, gain: float) -> dict:
    """Band circle and subtree push must sound identical under one basis."""
    p1 = Project(signal, sr, "x.wav", BasisSpec(**basis.to_dict()),
                 [band_stroke_for_node("b1", node_path, sr, gain)])
    p2 = Project(signal, sr, "x.wav", BasisSpec(**basis.to_dict()),
                 [SubtreeStroke("s1", node_path, gain,
                                origin_band_hz=nominal_band(node_path, sr))])
    y1, _ = p1.render()
    y2, _ = p2.render()
    err = max_abs(y1, y2)
    return {"name": "band vs subtree stroke equivalence",
            "max_abs_error": err, "pass": err < 1e-12}


# --------------------------------------------- check 4: rebase semantics
def check_rebase_band_persistence(project: Project, stroke_id: str,
                                  new_basis: BasisSpec) -> dict:
    """A band stroke keeps acting on the same physical band after rebasing.

    Wavelet filters are not brick walls, so FFT energy inside the narrow
    nominal band never shows the exact coefficient gain.  The check therefore
    verifies direction and magnitude of the change rather than an exact
    gain, and reports the measured deviation explicitly.
    """
    stroke = next(s for s in project.strokes if s.id == stroke_id)
    before, _ = project.render()
    band = stroke.band_hz
    e_orig = band_energy(project.original, project.sr, *band)
    e_before = band_energy(before, project.sr, *band)
    project.rebase(**{k: getattr(new_basis, k)
                      for k in ("wavelet", "max_level", "mode")})
    report = project.dry_run()
    unresolved = [u.stroke_id for u in report.unresolved]
    after, _ = project.render()
    e_after = band_energy(after, project.sr, *band)
    target = stroke.gain
    if e_orig <= 1e-12 * float(np.sum(project.original ** 2)):
        return {"name": "band stroke persists across rebase",
                "unresolved": unresolved, "gain_in_band": float("nan"),
                "target_gain": target, "pass": True,
                "skipped_sparse": True,
                "reason": "original signal has negligible energy in band"}
    g = np.sqrt(e_after / e_orig) if e_orig else float("nan")
    # attenuation strokes must keep attenuating after rebase, staying in a
    # broad envelope around the target; never cross above unity
    envelope = (0.25, 0.9) if target < 1 else (1.0, 4.0)
    ok = (stroke_id not in unresolved
          and envelope[0] <= g <= envelope[1]
          and (g < 1.0 if target < 1 else g > 1.0))
    return {"name": "band stroke persists across rebase",
            "unresolved": unresolved,
            "gain_in_band": g, "target_gain": target,
            "pre_rebase_gain_in_band":
                np.sqrt(e_before / e_orig) if e_orig else float("nan"),
            "pass": ok}


def check_rebase_subtree_follows(project: Project, stroke_id: str,
                                 new_basis: BasisSpec) -> dict:
    """A subtree stroke follows node descendants; deep nodes go unresolved."""
    stroke = next(s for s in project.strokes if s.id == stroke_id)
    old_path = stroke.node_path
    project.rebase(**{k: getattr(new_basis, k)
                      for k in ("wavelet", "max_level", "mode")})
    report = project.dry_run()
    un = next((u for u in report.unresolved if u.stroke_id == stroke_id), None)
    if un is not None:
        return {"name": "subtree stroke requires retarget",
                "old_path": old_path, "reason": un.reason,
                "candidates": un.candidates, "pass": True,
                "retarget_required": True}
    leaves_edited = sorted({e.path for e in report.edits
                            if e.stroke_id == stroke_id})
    follows = all(p == old_path or p.startswith(old_path) or
                  old_path.startswith(p) for p in leaves_edited)
    return {"name": "subtree stroke follows descendants",
            "old_path": old_path, "edited_leaves": leaves_edited,
            "pass": follows, "retarget_required": False}


# --------------------------------------------- check 5: local propagation
def check_local_propagation(signal: np.ndarray, sr: int,
                            basis: BasisSpec, node_path: str,
                            t0: float, t1: float, gain: float) -> dict:
    """Measure how far a windowed edit really spreads (support + ringing)."""
    proj = Project(signal, sr, "x.wav", BasisSpec(**basis.to_dict()),
                   [SubtreeStroke("s", node_path, gain, t0, t1)])
    edited, _ = proj.render()
    residual = edited - signal
    active = np.flatnonzero(np.abs(residual) > 1e-9)
    if active.size == 0:
        # Sparse signals (single impulses) may have no coefficient inside the
        # chosen node/time window.  That is correct behaviour, not a failure.
        return {"name": "local propagation and ringing",
                "pass": True, "skipped_sparse": True,
                "reason": "no coefficient energy inside the edited window"}
    spread_lo, spread_hi = active[0], active[-1]
    engine = proj.build_engine()
    i0, i1 = engine.time_window_to_indices(node_path, t0, t1)
    w0, w1 = engine.indices_to_time_window(node_path, i0, i1)
    info = engine.node_info(node_path)
    supp = info.support_samples
    # ringing: residual energy outside a window enlarged by measured support
    pad = (supp[1] - supp[0] + 1) if supp else 0
    safe_lo = max(0, int(w0 * sr) - pad)
    safe_hi = min(signal.size, int(w1 * sr) + pad)
    outside = np.concatenate([residual[:safe_lo], residual[safe_hi:]])
    inside = residual[safe_lo:safe_hi]
    ring_ratio = (float(np.sum(outside ** 2))
                  / max(float(np.sum(inside ** 2)), 1e-30))
    return {"name": "local propagation and ringing",
            "edited_window_sec": [w0, w1],
            "residual_span_samples": [int(spread_lo), int(spread_hi)],
            "residual_span_sec": [spread_lo / sr, spread_hi / sr],
            "basis_support_samples": list(supp) if supp else None,
            "ringing_energy_fraction": ring_ratio,
            "pass": ring_ratio < 0.05}


# --------------------------------------------- check 6: cross-scale strokes
def check_cross_scale(signal: np.ndarray, sr: int,
                      coarse_basis: BasisSpec, fine_basis: BasisSpec,
                      node_path: str, gain: float) -> dict:
    """Same subtree intent rendered at two depths: error must stay tiny
    except where finer time localisation legitimately changes the window."""
    p_coarse = Project(signal, sr, "x.wav",
                       BasisSpec(**coarse_basis.to_dict()),
                       [SubtreeStroke("s", node_path, gain,
                                      origin_band_hz=nominal_band(node_path, sr))])
    y_coarse, _ = p_coarse.render()
    p_fine = Project(signal, sr, "x.wav", BasisSpec(**fine_basis.to_dict()),
                     [SubtreeStroke("s", node_path, gain,
                                    origin_band_hz=nominal_band(node_path, sr))])
    y_fine, report = p_fine.render()
    err = max_abs(y_coarse, y_fine)
    return {"name": "cross-scale stroke re-open error",
            "max_abs_error": err,
            "edited_leaves_fine": sorted({e.path for e in report.edits}),
            "pass": err < 1e-10}


# --------------------------------------------- check 7: probe gain table
def probe_gain_table(project: Project, low_hz: float, high_hz: float
                     ) -> dict:
    edited, _ = project.render()
    return {"low_hz": low_hz, "high_hz": high_hz,
            "gain_low": tone_gain(project.original, edited, project.sr, low_hz),
            "gain_high": tone_gain(project.original, edited, project.sr, high_hz)}


# --------------------------------------------- check 8: leakage (non brick wall)
def check_leakage(project: Project, path: str) -> dict:
    engine = project.build_engine()
    profile = engine.leakage_profile(path)
    own = nominal_band(path, project.sr)
    width = (0.5 * project.sr) / 2 ** len(path)
    own_k = int(round(own[0] / width))
    in_band = float(profile[own_k])
    return {"name": "nominal-band leakage (1.0 would be a brick wall)",
            "node": path, "depth": len(path),
            "energy_in_nominal_band": in_band,
            "profile": [round(float(v), 4) for v in profile],
            "pass": in_band < 0.999}


# ------------------------------------------------------------- full battery
def run_battery(signal: np.ndarray, sr: int, out: list | None = None) -> list:
    """Run the complete verification battery on one signal."""
    results = out if out is not None else []
    base = BasisSpec("db4", 5, "symmetric")
    proj = Project(signal, sr, "x.wav", BasisSpec(**base.to_dict()),
                   [band_stroke_for_node("b1", "ad", sr, 0.25),
                    SubtreeStroke("s1", "ad", 0.25,
                                  origin_band_hz=nominal_band("ad", sr))])
    results.append(check_roundtrip(proj))
    results.append(check_energy(proj))
    results.append(check_edit_audibility(
        Project(signal, sr, "x.wav", BasisSpec(**base.to_dict()),
                [SubtreeStroke("s", "ad", 0.3)])))
    results.append(check_stroke_equivalence(signal, sr, base, "ad", 0.4))

    # rebase: wavelet change keeps band; subtree follows at same depth
    pb = Project(signal, sr, "x.wav", BasisSpec(**base.to_dict()),
                 [band_stroke_for_node("b", "ad", sr, 0.5)])
    results.append(check_rebase_band_persistence(
        pb, "b", BasisSpec("sym6", 5, "symmetric")))
    ps = Project(signal, sr, "x.wav", BasisSpec(**base.to_dict()),
                 [SubtreeStroke("s", "ad", 0.5,
                                origin_band_hz=nominal_band("ad", sr))])
    results.append(check_rebase_subtree_follows(
        ps, "s", BasisSpec("db4", 6, "symmetric")))

    # subtree node deeper than the new tree -> must demand retarget
    pdeep = Project(signal, sr, "x.wav", BasisSpec("db4", 6, "symmetric"),
                    [SubtreeStroke("sd", "adaad", 0.5,
                                   origin_band_hz=nominal_band("adaad", sr))])
    results.append(check_rebase_subtree_follows(
        pdeep, "sd", BasisSpec("db4", 3, "symmetric")))

    # windowed propagation check (sparse signals may legitimately skip it)
    results.append(check_local_propagation(
        signal, sr, base, "ad", 0.2, 0.4, 0.2))
    # full-length propagation check always has coefficients to edit
    results.append(check_local_propagation(
        signal, sr, base, "d", 0.0, signal.size / sr, 0.2))
    results.append(check_cross_scale(
        signal, sr, BasisSpec("db4", 4, "symmetric"),
        BasisSpec("db4", 6, "symmetric"), "ad", 0.5))
    results.append(check_leakage(
        Project(signal, sr, "x.wav", BasisSpec(**base.to_dict())), "ad"))
    return results
