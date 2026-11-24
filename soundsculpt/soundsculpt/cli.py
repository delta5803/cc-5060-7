"""Command-line interface for soundsculpt."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .wavio import read_wav, write_wav
from .signals import (demo_mix, chirp, impulse, boundary_impulses,
                      band_limited_noise, probe_pair)
from .basis import (Project, BasisSpec, BandStroke, SubtreeStroke,
                    UnresolvedError, band_stroke_for_node)
from .wpt import nominal_band, all_paths, freq_order_paths
from . import verification as vf


SIGNALS = {
    "demo": lambda n, sr: (demo_mix(n, sr), sr),
    "chirp": lambda n, sr: (chirp(n, sr), sr),
    "impulse": lambda n, sr: (impulse(n, sr), sr),
    "boundary": lambda n, sr: (boundary_impulses(n, sr), sr),
    "noise": lambda n, sr: (band_limited_noise(n, sr, 2400, 3000), sr),
}


def _load_project(path: str) -> Project:
    return Project.load(path)


def cmd_init(args) -> None:
    if args.signal == "wav":
        sig, sr = read_wav(args.wav)
        name = os.path.basename(args.wav)
        audio_out = os.path.join(os.path.dirname(args.project) or ".", name)
        if os.path.abspath(audio_out) != os.path.abspath(args.wav):
            write_wav(audio_out, sig, sr)
    else:
        sr = args.sr
        sig, sr = SIGNALS[args.signal](args.n, sr)
        name = os.path.basename(args.project).replace(".json", "") + "_src.wav"
        audio_out = os.path.join(os.path.dirname(args.project) or ".", name)
        write_wav(audio_out, sig, sr)
    proj = Project(sig, sr, name,
                   BasisSpec(args.wavelet, args.level, args.mode))
    proj.save(args.project)
    print(f"created {args.project} ({sig.size} samples @ {sr} Hz), "
          f"audio {name}, basis {args.wavelet}/L{args.level}/{args.mode}")


def _print_tree(proj: Project, show_support: bool) -> None:
    engine = proj.build_engine()
    print(f"wavelet={proj.basis.wavelet} level={proj.basis.max_level} "
          f"mode={proj.basis.mode} N={engine.n} sr={engine.sr}")
    print("current leaf basis:", " ".join(proj.basis.leaves))
    for depth in range(proj.basis.max_level + 1):
        for p in all_paths(depth):
            info = engine.node_info(p, measure_support=show_support)
            lo, hi = info.band_hz
            tag = "LEAF" if p in proj.basis.leaves else "    "
            line = (f"  {'  ' * depth}{tag} {p or '<root>':<6} "
                    f"band={lo:8.1f}-{hi:8.1f} Hz n={info.n_coeffs:5d} "
                    f"step={info.nominal_step_sec * 1e3:6.3f} ms")
            if show_support and info.support_samples:
                s0, s1 = info.support_samples
                el, er = info.edge_coeffs
                line += (f" support={s0}..{s1} "
                         f"({info.support_sec * 1e3:.2f} ms) "
                         f"edge_coeffs=L{el}/R{er}")
            print(line)


def cmd_tree(args) -> None:
    proj = _load_project(args.project)
    _print_tree(proj, args.support)


def cmd_bands(args) -> None:
    proj = _load_project(args.project)
    engine = proj.build_engine()
    order = freq_order_paths(proj.basis.max_level)
    print("freq-ordered deepest leaves (nominal bands; filters overlap):")
    for k, p in enumerate(order):
        lo, hi = nominal_band(p, engine.sr)
        n = engine.coeffs(p).size
        print(f"  k={k:2d} {p:5s} {lo:9.1f} - {hi:9.1f} Hz  coeffs={n}")


def cmd_leakage(args) -> None:
    proj = _load_project(args.project)
    res = vf.check_leakage(proj, args.node)
    print(json.dumps(res, indent=2))


def cmd_stroke_band(args) -> None:
    proj = _load_project(args.project)
    if args.node:
        st = band_stroke_for_node(args.id, args.node, proj.sr, args.gain,
                                  args.t0, args.t1)
    else:
        st = BandStroke(args.id, (args.f0, args.f1), args.gain,
                        args.t0, args.t1)
    proj.add_stroke(st)
    proj.save(args.project)
    print(f"added band stroke {st.id} -> {st.band_hz[0]:.0f}-"
          f"{st.band_hz[1]:.0f} Hz gain={st.gain}")


def cmd_stroke_subtree(args) -> None:
    proj = _load_project(args.project)
    st = SubtreeStroke(args.id, args.node, args.gain, args.t0, args.t1,
                       nominal_band(args.node, proj.sr))
    proj.add_stroke(st)
    proj.save(args.project)
    print(f"added subtree stroke {st.id} -> node {args.node} "
          f"gain={st.gain}")


def cmd_list_strokes(args) -> None:
    proj = _load_project(args.project)
    report = proj.dry_run()
    for s in proj.strokes:
        print(f"  {s.kind:7s} {s.id} gain={s.gain} "
              f"window=[{s.t0},{s.t1}] "
              + (f"band={s.band_hz}" if s.kind == "band"
                 else f"node={s.node_path}"))
    for n in report.notes:
        print("  -", n)
    for u in report.unresolved:
        print(f"  UNRESOLVED {u.stroke_id}: {u.reason}; candidates={u.candidates}")


def cmd_render(args) -> None:
    proj = _load_project(args.project)
    try:
        out, report = proj.render()
    except UnresolvedError as e:
        print("render blocked; unresolved strokes:", file=sys.stderr)
        for u in e.report.unresolved:
            print(f"  {u.stroke_id}: {u.reason} candidates={u.candidates}",
                  file=sys.stderr)
        sys.exit(2)
    write_wav(args.out, out, proj.sr)
    residual = out - proj.original
    print(f"rendered {args.out}")
    print(f"  residual rms={vf.rms_db(residual):.2f} dBFS "
          f"max={np.max(np.abs(residual)):.6f}")
    for n in report.notes:
        print("  -", n)


def cmd_rebase(args) -> None:
    proj = _load_project(args.project)
    nb = BasisSpec(args.wavelet or proj.basis.wavelet,
                   args.level or proj.basis.max_level,
                   args.mode or proj.basis.mode)
    report = proj.rebase(nb.wavelet, nb.max_level, nb.mode)
    proj.save(args.project)
    print(f"rebased to {proj.basis.wavelet}/L{proj.basis.max_level}/"
          f"{proj.basis.mode}")
    for n in report.notes:
        print("  -", n)
    if report.unresolved:
        print("UNRESOLVED strokes (use 'retarget'):")
        for u in report.unresolved:
            print(f"  {u.stroke_id}: {u.reason}; candidates={u.candidates}")


def cmd_retarget(args) -> None:
    proj = _load_project(args.project)
    proj.retarget(args.stroke_id, args.node)
    proj.save(args.project)
    report = proj.dry_run()
    bad = [u.stroke_id for u in report.unresolved]
    print(f"retargeted {args.stroke_id} -> {args.node}; "
          f"unresolved now: {bad or 'none'}")


def cmd_edit(args) -> None:
    """Apply a gain edit to one leaf node, keyed by its stable identity."""
    proj = _load_project(args.project)
    try:
        rec = proj.add_node_edit(args.node, args.gain, args.t0,
                                 args.t1, tag=args.tag or "")
    except ValueError as e:
        print(f"edit rejected: {e}", file=sys.stderr)
        sys.exit(2)
    proj.save(args.project)
    print(f"edit {rec.node} gain={rec.gain} "
          f"window=[{rec.t0},{rec.t1 if rec.t1 is not None else 'end'}] "
          f"tag={rec.tag or '-'} (tree {proj.tree.version.id[:12]})")


def cmd_undo(args) -> None:
    """Undo the most recent edit by node identity or tag."""
    proj = _load_project(args.project)
    got = proj.undo(tag=args.tag, path=args.node)
    if got is None:
        print("nothing to undo", file=sys.stderr)
        sys.exit(1)
    proj.save(args.project)
    print(f"undid edit on {got.node} (gain={got.gain})")


def cmd_edits(args) -> None:
    """List the persisted coefficient-edit state and its node identities."""
    proj = _load_project(args.project)
    print(f"tree version {proj.tree.version.id[:12]} "
          f"({len(proj.tree.leaves)} leaves)")
    for e in proj.edits.edits:
        print(f"  {e.node} gain={e.gain} "
              f"window=[{e.t0},{e.t1 if e.t1 is not None else 'end'}] "
              f"tag={e.tag or '-'}")
    idx = proj.edits.identity_index()
    print(f"identity index: {idx}")


def cmd_subdivide(args) -> None:
    proj = _load_project(args.project)
    migration = proj.subdivide(args.node)
    proj.save(args.project)
    report = proj.dry_run()
    print(f"subdivided {args.node}; leaves={proj.basis.leaves}")
    print(f"  edit migration: remapped={len(migration.remapped)} "
          f"rejected={len(migration.rejected)}")
    for u in report.unresolved:
        print(f"  UNRESOLVED {u.stroke_id}: {u.reason}")


def cmd_audition(args) -> None:
    proj = _load_project(args.project)
    engine = proj.build_engine()
    contrib = engine.node_contribution(args.node)
    write_wav(args.out, contrib, proj.sr)
    info = engine.node_info(args.node)
    print(f"node {args.node} contribution -> {args.out} "
          f"(rms {vf.rms_db(contrib):.2f} dBFS, band "
          f"{info.band_hz[0]:.0f}-{info.band_hz[1]:.0f} Hz)")


def cmd_probe(args) -> None:
    # Build a probe file matching the project's length, apply the project's
    # strokes through a scratch project, measure high/low realised gain.
    proj = _load_project(args.project)
    n = proj.original.size
    tone, lo, hi = probe_pair(n, proj.sr, args.low, args.high)
    scratch = Project(tone, proj.sr, "probe.wav",
                      BasisSpec(**proj.basis.to_dict()),
                      list(proj.strokes))
    try:
        edited, _ = scratch.render()
    except UnresolvedError:
        print("probe blocked by unresolved strokes", file=sys.stderr)
        sys.exit(2)
    if args.out:
        write_wav(args.out, edited, proj.sr)
    table = vf.probe_gain_table(scratch, lo, hi)
    table["gain_low_db"] = 20 * np.log10(table["gain_low"])
    table["gain_high_db"] = 20 * np.log10(table["gain_high"])
    print(json.dumps(table, indent=2))


def cmd_diff(args) -> None:
    """Before/after differential evidence for a tree reconstruction change.

    Runs, on a clean project built from a synthetic signal or WAV:
      1. boundary short-clip before/after (tree swap keeps the clip identical
         within floating point);
      2. a scale switch (depth change) before/after, reporting reconstruction
         error and which node identities the migrated edits landed on.
    Emits a JSON report so the numbers are execution evidence, not prose.
    """
    from .signals import boundary_impulses as _bi
    sr = args.sr
    n = args.n
    if args.signal == "wav":
        sig, sr = read_wav(args.wav)
    elif args.signal == "boundary":
        sig = _bi(n, sr)
    else:
        sig, _ = SIGNALS[args.signal](n, sr)

    report = {"signal": args.signal, "n": int(sig.size), "sr": sr}

    # (1) boundary short clip: cut a short window near an edge and compare the
    # reconstruction of the same source under the same basis before/after a
    # no-op tree "swap" (rebuild + resync).  The clip must be bit-identical.
    clip_n = min(sig.size, int(0.05 * sr))
    edge_clip = sig[sig.size - clip_n:].copy()
    proj = Project(sig, sr, "diff.wav",
                   BasisSpec(args.wavelet, args.level, args.mode))
    base_tree = proj.tree.version.id
    v_before = proj.build_engine().reconstruct(
        proj.tree.leaves, tree_version=proj.tree.version.id)
    proj.sync_tree()
    v_after = proj.build_engine().reconstruct(
        proj.tree.leaves, tree_version=proj.tree.version.id)
    clip_err = float(np.max(np.abs(v_before[-clip_n:] - v_after[-clip_n:])))
    report["boundary_short_clip"] = {
        "clip_samples": clip_n, "version_id": base_tree[:12],
        "max_abs_diff": clip_err, "pass": clip_err < 1e-12,
    }

    # (2) scale switch: apply a full-length edit at a coarse leaf, record its
    # node ownership + reconstruction error, switch depth, and record the
    # migrated ownership + reconstruction error.  A full-length subtree edit
    # has the same intent at any depth, so the two renders must agree to
    # floating point; a *windowed* edit legitimately differs because the
    # coefficient-to-time map changes with depth, and that difference is
    # reported rather than hidden.
    proj2 = Project(sig, sr, "diff.wav",
                    BasisSpec(args.wavelet, 3, args.mode,
                              leaves=["aa", "ad", "da", "dd"]))
    proj2.add_node_edit("ad", 0.4, tag="diff")          # full length
    y_coarse = proj2.render()[0] if not proj2.dry_run().unresolved else None
    own_before = sorted(str(e.node) for e in proj2.edits.edits)
    proj2.rebase(max_level=args.level)
    y_fine = proj2.render()[0] if not proj2.dry_run().unresolved else None
    own_after = sorted(str(e.node) for e in proj2.edits.edits)
    rec = proj2.build_engine().reconstruct(proj2.tree.leaves,
                                           tree_version=proj2.tree.version.id)
    # windowed variant reported for contrast (not an equality claim)
    pw = Project(sig, sr, "diff.wav",
                 BasisSpec(args.wavelet, 3, args.mode,
                           leaves=["aa", "ad", "da", "dd"]))
    pw.add_node_edit("ad", 0.4, t0=0.2, t1=0.4, tag="w")
    yw_coarse = pw.render()[0]
    pw.rebase(max_level=args.level)
    yw_fine = pw.render()[0]
    windowed_diff = float(np.max(np.abs(yw_coarse - yw_fine)))
    full_diff = (float(np.max(np.abs(y_coarse - y_fine)))
                 if y_coarse is not None and y_fine is not None else None)
    rec_err = float(np.max(np.abs(rec - sig)))
    under = all(p.split(":", 1)[1].startswith("ad") for p in own_after)
    report["scale_switch"] = {
        "from_depth": 3, "to_depth": args.level,
        "reconstruction_error_signal": rec_err,
        "full_length_render_max_diff": full_diff,
        "windowed_render_max_diff_reported_not_equal": windowed_diff,
        "ownership_before": own_before,
        "ownership_after": own_after,
        "under_ancestor": under,
        "pass": (full_diff is not None and full_diff < 1e-10
                 and rec_err < 1e-10 and under),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    ok = (report["boundary_short_clip"]["pass"]
          and report["scale_switch"]["pass"])
    print(f"\nreport -> {args.out}; equivalent={ok}")
    if not ok:
        sys.exit(1)


def cmd_verify(args) -> None:
    sr = args.sr
    n = args.n
    signals = {
        "impulse": impulse(n, sr),
        "boundary": boundary_impulses(n, sr),
        "chirp": chirp(n, sr),
        "noise": band_limited_noise(n, sr, 2400, 3000),
        "demo": demo_mix(n, sr),
    }
    all_results = {}
    failed = 0
    for name, sig in signals.items():
        res = vf.run_battery(sig, sr)
        all_results[name] = res
        failed += sum(1 for r in res if not r.get("pass", False))
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=float)
    for name, res in all_results.items():
        print(f"== {name} ==")
        for r in res:
            mark = "ok " if r.get("pass") else "FAIL"
            print(f"  [{mark}] {r['name']}")
    print(f"\nreport -> {args.report}; failures={failed}")
    if failed:
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soundsculpt",
                                description="wavelet-packet sound sculpting")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("init")
    q.add_argument("project")
    q.add_argument("--signal", choices=list(SIGNALS) + ["wav"], default="demo")
    q.add_argument("--wav")
    q.add_argument("--n", type=int, default=32768)
    q.add_argument("--sr", type=int, default=44100)
    q.add_argument("--wavelet", default="db4")
    q.add_argument("--level", type=int, default=5)
    q.add_argument("--mode", default="symmetric")
    q.set_defaults(func=cmd_init)

    q = sub.add_parser("tree"); q.add_argument("project")
    q.add_argument("--support", action="store_true")
    q.set_defaults(func=cmd_tree)

    q = sub.add_parser("bands"); q.add_argument("project")
    q.set_defaults(func=cmd_bands)

    q = sub.add_parser("leakage"); q.add_argument("project")
    q.add_argument("node"); q.set_defaults(func=cmd_leakage)

    q = sub.add_parser("stroke-band"); q.add_argument("project")
    q.add_argument("id"); q.add_argument("--gain", type=float, required=True)
    q.add_argument("--node"); q.add_argument("--f0", type=float)
    q.add_argument("--f1", type=float); q.add_argument("--t0", type=float, default=0.0)
    q.add_argument("--t1", type=float, default=None)
    q.set_defaults(func=cmd_stroke_band)

    q = sub.add_parser("stroke-subtree"); q.add_argument("project")
    q.add_argument("id"); q.add_argument("node")
    q.add_argument("--gain", type=float, required=True)
    q.add_argument("--t0", type=float, default=0.0)
    q.add_argument("--t1", type=float, default=None)
    q.set_defaults(func=cmd_stroke_subtree)

    q = sub.add_parser("strokes"); q.add_argument("project")
    q.set_defaults(func=cmd_list_strokes)

    q = sub.add_parser("render"); q.add_argument("project"); q.add_argument("out")
    q.set_defaults(func=cmd_render)

    q = sub.add_parser("rebase"); q.add_argument("project")
    q.add_argument("--wavelet"); q.add_argument("--level", type=int)
    q.add_argument("--mode"); q.set_defaults(func=cmd_rebase)

    q = sub.add_parser("retarget"); q.add_argument("project")
    q.add_argument("stroke_id"); q.add_argument("node")
    q.set_defaults(func=cmd_retarget)

    q = sub.add_parser("edit"); q.add_argument("project")
    q.add_argument("node"); q.add_argument("--gain", type=float, required=True)
    q.add_argument("--t0", type=float, default=0.0)
    q.add_argument("--t1", type=float, default=None)
    q.add_argument("--tag", default="")
    q.set_defaults(func=cmd_edit)

    q = sub.add_parser("undo"); q.add_argument("project")
    q.add_argument("--node"); q.add_argument("--tag")
    q.set_defaults(func=cmd_undo)

    q = sub.add_parser("edits"); q.add_argument("project")
    q.set_defaults(func=cmd_edits)

    q = sub.add_parser("subdivide"); q.add_argument("project"); q.add_argument("node")
    q.set_defaults(func=cmd_subdivide)

    q = sub.add_parser("audition"); q.add_argument("project")
    q.add_argument("node"); q.add_argument("out")
    q.set_defaults(func=cmd_audition)

    q = sub.add_parser("probe"); q.add_argument("project")
    q.add_argument("--low", type=float, required=True)
    q.add_argument("--high", type=float, required=True)
    q.add_argument("--out"); q.set_defaults(func=cmd_probe)

    q = sub.add_parser("verify")
    q.add_argument("report"); q.add_argument("--n", type=int, default=32768)
    q.add_argument("--sr", type=int, default=44100)
    q.set_defaults(func=cmd_verify)

    q = sub.add_parser("diff")
    q.add_argument("out")
    q.add_argument("--signal", choices=list(SIGNALS) + ["wav"], default="chirp")
    q.add_argument("--wav"); q.add_argument("--n", type=int, default=16384)
    q.add_argument("--sr", type=int, default=44100)
    q.add_argument("--wavelet", default="db4")
    q.add_argument("--level", type=int, default=5)
    q.add_argument("--mode", default="periodization")
    q.set_defaults(func=cmd_diff)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
