#!/usr/bin/env python3
"""End-to-end calibration: try every hand-eye method, pick the winner by Step4
verification metrics, then run Step5 reports on the winner.

Single command for any capture dataset:
  python3 run_calibration.py --root_folder ./<your>/session --intrinsics_dir ./intrinsics

Outputs (under <root>/calib_compare/):
  <METHOD>/                 — Step3+Step4 outputs per method
  comparison.json           — parsed metrics + ranking
  comparison.md             — human-readable comparison table

Outputs (under <root>/calib_out/):
  Winning method's calibration + Step4 verification + Step5 reports.
  This is the directory downstream object-pose code should read.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ALL_METHODS = ["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"]


def run_step(label, cmd, log_path):
    print(f"  $ {label}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return proc.returncode, dt


def parse_step4_metrics(metrics_path):
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        m = json.load(f)

    def gp(*path, default=None):
        cur = m
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    return {
        "cross_cam_mean_mm": gp("cross_camera", "mean_mm"),
        "cross_cam_max_mm": gp("cross_camera", "max_mm"),
        "reproj_mean_px": gp("reprojection", "mean_px"),
        "handeye_pos_std_mm": gp("handeye", "board_position_std_mm"),
        "handeye_pos_max_mm": gp("handeye", "board_position_max_mm"),
        "board_reproj_mean_px": gp("board_reprojection", "mean_px"),
        "pose_rep_mean_mm": gp("pose_repeatability", "mean_dt_mm"),
        "pose_rep_max_mm": gp("pose_repeatability", "max_dt_mm"),
        "pose_rep_mean_deg": gp("pose_repeatability", "mean_dr_deg"),
        "mesh_rmse_mm": gp("mesh_alignment", "mean_rmse_mm"),
        "dim_err_mean_mm": gp("dimension_accuracy", "mean_abs_err_mm"),
        "all_pass": all([
            gp("cross_camera", "pass", default=False),
            gp("reprojection", "pass", default=False),
            gp("handeye", "pass", default=False),
            gp("pose_repeatability", "pass", default=False),
            gp("mesh_alignment", "pass", default=False),
            gp("dimension_accuracy", "pass", default=False),
        ]),
    }


def composite_score(s):
    """Composite score (lower = better) for downstream object-pose accuracy.

    Weights chosen for multi-camera object pose estimation use case:
      - cross-camera consistency dominates (this is what limits final accuracy)
      - pose repeatability is the second-most important (consistency across robot poses)
      - hand-eye stability and depth/3D consistency are secondary
    """
    if s is None:
        return float("inf")
    def f(v):
        return 0.0 if v is None else float(v)
    return (
        2.0 * f(s["cross_cam_mean_mm"])
        + 1.0 * f(s["cross_cam_max_mm"])
        + 1.0 * f(s["pose_rep_mean_mm"])
        + 0.3 * f(s["pose_rep_max_mm"])
        + 5.0 * f(s["pose_rep_mean_deg"])
        + 2.0 * f(s["handeye_pos_std_mm"])
        + 1.0 * f(s["mesh_rmse_mm"])
    )


def fmt_row(name, s, score):
    if s is None:
        return f"| {name:<11} | FAILED | | | | | | |"
    return (
        f"| {name:<11} "
        f"| {score:6.2f} "
        f"| {s['cross_cam_mean_mm']:5.2f} / {s['cross_cam_max_mm']:5.2f} "
        f"| {s['pose_rep_mean_mm']:5.2f} / {s['pose_rep_max_mm']:5.2f} "
        f"| {s['pose_rep_mean_deg']:5.3f} "
        f"| {s['handeye_pos_std_mm']:5.2f} "
        f"| {s['reproj_mean_px']:5.3f} "
        f"| {s['mesh_rmse_mm']:5.2f} "
        f"| {'PASS' if s['all_pass'] else 'FAIL'} |"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_folder", required=True, help="Path to session/ directory containing meta.json + camN/")
    ap.add_argument("--intrinsics_dir", required=True, help="Path to intrinsics/ with camN.npz files")
    ap.add_argument("--methods", default=",".join(ALL_METHODS),
                    help=f"Comma-separated handeye methods (default: {','.join(ALL_METHODS)})")
    ap.add_argument("--compare_out", default=None,
                    help="Per-method outputs (default: <root>/calib_compare)")
    ap.add_argument("--final_out", default=None,
                    help="Winner output (default: <root>/calib_out)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Reuse existing per-method outputs if Step4 metrics already present")
    ap.add_argument("--enable_ba", action="store_true",
                    help="Enable bundle adjustment refinement (experimental — currently degrades on this dataset)")
    ap.add_argument("--skip_step5", action="store_true",
                    help="Skip Step5 report export on the winner")
    args = ap.parse_args()

    root = Path(args.root_folder).resolve()
    intr = Path(args.intrinsics_dir).resolve()
    if not (root / "meta.json").exists():
        sys.exit(f"[ERR] meta.json not found in {root}")
    if not intr.exists():
        sys.exit(f"[ERR] intrinsics dir not found: {intr}")

    compare_out = Path(args.compare_out).resolve() if args.compare_out else root / "calib_compare"
    final_out = Path(args.final_out).resolve() if args.final_out else root / "calib_out"
    compare_out.mkdir(parents=True, exist_ok=True)

    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    print(f"[INFO] Methods: {methods}")
    print(f"[INFO] Per-method outputs:  {compare_out}")
    print(f"[INFO] Final winner output: {final_out}\n")

    results = {}
    timings = {}

    for method in methods:
        out = compare_out / method
        out.mkdir(parents=True, exist_ok=True)
        metrics_path = out / "verification_metrics.json"

        print(f"━━━ {method} ━━━")
        timings[method] = {}

        if args.skip_existing and metrics_path.exists():
            print(f"  [skip] reusing existing output")
        else:
            rc, dt = run_step(
                f"Step3 ({method})",
                [sys.executable, "Step3_calibration.py",
                 "--root_folder", str(root), "--intrinsics_dir", str(intr),
                 "--out_dir", str(out), "--handeye_method", method],
                out / "_step3.log",
            )
            timings[method]["step3_s"] = dt
            print(f"    Step3: {dt_show(dt)}{' ✗ rc=' + str(rc) if rc else ''}")
            if rc != 0:
                results[method] = None
                continue

            if args.enable_ba:
                rc, dt = run_step(
                    "Bundle adjustment",
                    [sys.executable, "bundle_adjust.py",
                     "--root_folder", str(root), "--intrinsics_dir", str(intr),
                     "--calib_dir", str(out), "--quiet"],
                    out / "_ba.log",
                )
                timings[method]["ba_s"] = dt
                print(f"    BA:    {dt_show(dt)}{' ✗ rc=' + str(rc) if rc else ''}")
                # Non-fatal: if BA fails, fall back to non-BA result

            rc, dt = run_step(
                "Step4 verify",
                [sys.executable, "Step4_verify.py",
                 "--root_folder", str(root), "--intrinsics_dir", str(intr),
                 "--calib_dir", str(out)],
                out / "_step4.log",
            )
            timings[method]["step4_s"] = dt
            print(f"    Step4: {dt_show(dt)}{' ✗ rc=' + str(rc) if rc else ''}")
            if rc != 0:
                results[method] = None
                continue

        m = parse_step4_metrics(metrics_path)
        results[method] = m
        if m is None:
            print(f"    [WARN] no metrics produced")
        else:
            print(f"    cross={m['cross_cam_mean_mm']:.2f}mm "
                  f"pose_rep={m['pose_rep_mean_mm']:.2f}mm/{m['pose_rep_mean_deg']:.3f}° "
                  f"mesh={m['mesh_rmse_mm']:.2f}mm  pass={m['all_pass']}")
        print()

    # Rank
    scored = [(name, results[name], composite_score(results[name])) for name in methods]
    scored.sort(key=lambda x: x[2])

    if scored[0][1] is None:
        sys.exit("[ERR] All methods failed; no winner could be selected.")
    winner_name = scored[0][0]
    winner_metrics = scored[0][1]
    winner_score = scored[0][2]

    # Build comparison report
    comparison = {
        "methods": methods,
        "ranking": [n for n, _, _ in scored],
        "scores": {n: float(s) for n, _, s in scored},
        "results": {n: r for n, r in results.items()},
        "timings": timings,
        "winner": {
            "name": winner_name,
            "score": float(winner_score),
            "metrics": winner_metrics,
        },
    }
    with open(compare_out / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    md = []
    md.append("# Calibration: hand-eye method comparison\n")
    md.append(f"- root_folder: `{root}`")
    md.append(f"- methods evaluated: {', '.join(methods)}\n")
    md.append("Composite score (lower = better) weights downstream object-pose accuracy: "
              "cross-cam consistency (most), pose repeatability, hand-eye stability, mesh RMSE.\n")
    md.append("| Method | Score | cross mean/max (mm) | pose_rep mean/max (mm) | "
              "pose_rep rot (°) | HE pos std (mm) | reproj (px) | mesh RMSE (mm) | All Pass |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for name, s, sc in scored:
        md.append(fmt_row(name, s, sc))
    md.append("")
    md.append(f"**Winner: {winner_name}**  (composite score {winner_score:.2f})")
    md.append(f"- cross-cam: mean {winner_metrics['cross_cam_mean_mm']:.2f}mm / max {winner_metrics['cross_cam_max_mm']:.2f}mm")
    md.append(f"- pose repeatability: {winner_metrics['pose_rep_mean_mm']:.2f}mm / {winner_metrics['pose_rep_mean_deg']:.3f}°")
    md.append(f"- mesh alignment RMSE: {winner_metrics['mesh_rmse_mm']:.2f}mm")
    with open(compare_out / "comparison.md", "w") as f:
        f.write("\n".join(md))

    print("\n" + "\n".join(md))

    # Copy winner to final_out
    src = compare_out / winner_name
    if final_out.exists():
        shutil.rmtree(final_out)
    shutil.copytree(src, final_out)
    print(f"\n[FINAL] Winner '{winner_name}' copied to: {final_out}")

    # Step5 export reports on the winner
    if not args.skip_step5:
        print(f"\n━━━ Step5 (winner: {winner_name}) ━━━")
        step5_log = final_out / "_step5.log"
        rc, dt = run_step(
            "Step5 export reports",
            [sys.executable, "Step5_export_reports.py",
             "--root_folder", str(root), "--intrinsics_dir", str(intr),
             "--calib_dir", str(final_out)],
            step5_log,
        )
        print(f"    Step5: {dt_show(dt)}{' ✗ rc=' + str(rc) if rc else ' ✓'}")
        if rc != 0:
            print(f"    [WARN] Step5 failed; see {step5_log}")

    print(f"\n[DONE] Best calibration available at: {final_out}")
    print(f"       Comparison report: {compare_out / 'comparison.md'}")


def dt_show(seconds):
    if seconds < 1.0:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds//60)}m{int(seconds%60)}s"


if __name__ == "__main__":
    main()
