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


def cmd_subdivide(args) -> None:
    proj = _load_project(args.project)
    proj.basis.subdivide(args.node)
    proj.save(args.project)
    report = proj.dry_run()
    print(f"subdivided {args.node}; leaves={proj.basis.leaves}")
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
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
