#!/usr/bin/env python3
"""
Generate Zeus calibration waypoints for hand-eye / multi-view capture.

Output format is directly compatible with:
  Step2_capture_cube_poses.py --joint_file <json>

The generated JSON is a list of [d1,d2,d3,d4,d5,d6] rows.
In this project, these 6 values are commonly used as TCP-like pose values
(x, y, z, rz, ry, rx) in mm/deg. Keep your Zeus server interpretation aligned.

명령어:
  python make_zeus_calib_waypoints.py \
  --seed_pose 450,-120,380,180,0,180 \
  --out_file joints_handeye_calib.json \
  --translation_scale 1.0 \
  --rotation_scale 1.0
"""

import argparse
import json
from typing import List, Tuple


Pose6 = List[float]


# Curated pose deltas around one taught seed pose.
# Order: [dx_mm, dy_mm, dz_mm, drz_deg, dry_deg, drx_deg]
# The sequence is arranged to gradually expand motion range.
DEFAULT_DELTAS: List[Pose6] = [
    # Phase A: rotation diversity at almost same position
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 15, 0, 0],
    [0, 0, 0, -15, 0, 0],
    [0, 0, 0, 0, 20, 0],
    [0, 0, 0, 0, -20, 0],
    [0, 0, 0, 0, 0, 15],
    [0, 0, 0, 0, 0, -15],
    [0, 0, 0, 12, 12, 0],
    [0, 0, 0, -12, -12, 0],

    # Phase B: XY translation ring + mild orientation compensation
    [50, 0, 0, 0, 10, 0],
    [35, 35, 0, 8, 8, 0],
    [0, 50, 0, 10, 0, 0],
    [-35, 35, 0, 8, -8, 0],
    [-50, 0, 0, 0, -10, 0],
    [-35, -35, 0, -8, -8, 0],
    [0, -50, 0, -10, 0, 0],
    [35, -35, 0, -8, 8, 0],

    # Phase C: depth / height diversity (important for hand-eye conditioning)
    [60, 0, 35, 0, 15, 0],
    [0, 60, 35, 15, 0, 0],
    [-60, 0, 35, 0, -15, 0],
    [0, -60, 35, -15, 0, 0],
    [60, 0, -35, 0, 20, 0],
    [0, 60, -35, 20, 0, 0],
    [-60, 0, -35, 0, -20, 0],
    [0, -60, -35, -20, 0, 0],
    [0, 0, 55, 0, 12, 0],
    [0, 0, -55, 0, -12, 0],
]


def parse_pose6(text: str) -> Pose6:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 6:
        raise ValueError("seed_pose must have 6 comma-separated values: x,y,z,rz,ry,rx")
    return vals


def add_pose(a: Pose6, b: Pose6) -> Pose6:
    return [float(a[i] + b[i]) for i in range(6)]


def scale_delta(delta: Pose6, t_scale: float, r_scale: float) -> Pose6:
    out = delta[:]
    out[0] *= t_scale
    out[1] *= t_scale
    out[2] *= t_scale
    out[3] *= r_scale
    out[4] *= r_scale
    out[5] *= r_scale
    return out


def round_pose(p: Pose6, pos_decimals: int, rot_decimals: int) -> Pose6:
    return [
        round(float(p[0]), pos_decimals),
        round(float(p[1]), pos_decimals),
        round(float(p[2]), pos_decimals),
        round(float(p[3]), rot_decimals),
        round(float(p[4]), rot_decimals),
        round(float(p[5]), rot_decimals),
    ]


def build_waypoints(
    seed_pose: Pose6,
    translation_scale: float,
    rotation_scale: float,
    repeats_per_waypoint: int,
    append_stop_marker: bool,
    pos_decimals: int,
    rot_decimals: int,
) -> List[Pose6]:
    if repeats_per_waypoint < 1:
        raise ValueError("repeats_per_waypoint must be >= 1")

    waypoints: List[Pose6] = []
    for d in DEFAULT_DELTAS:
        sd = scale_delta(d, translation_scale, rotation_scale)
        wp = round_pose(add_pose(seed_pose, sd), pos_decimals, rot_decimals)
        for _ in range(repeats_per_waypoint):
            waypoints.append(wp)

    if append_stop_marker:
        # Step2_capture_cube_poses.py treats all-zeros row as end marker.
        waypoints.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return waypoints


def compute_stats(waypoints: List[Pose6], has_stop_marker: bool) -> Tuple[int, int]:
    total = len(waypoints)
    effective = total - 1 if has_stop_marker and total > 0 else total
    return total, effective


def main():
    parser = argparse.ArgumentParser(
        description="Generate a robust Zeus waypoint set for camera/hand-eye calibration."
    )
    parser.add_argument(
        "--seed_pose",
        required=True,
        help="Reference 6D pose (x,y,z,rz,ry,rx) in mm/deg, comma-separated.",
    )
    parser.add_argument(
        "--out_file",
        default="joints_handeye_calib.json",
        help="Output JSON path (Step2 --joint_file compatible).",
    )
    parser.add_argument(
        "--translation_scale",
        type=float,
        default=1.0,
        help="Scale factor for translation deltas.",
    )
    parser.add_argument(
        "--rotation_scale",
        type=float,
        default=1.0,
        help="Scale factor for rotation deltas.",
    )
    parser.add_argument(
        "--repeats_per_waypoint",
        type=int,
        default=1,
        help="How many times each waypoint is repeated.",
    )
    parser.add_argument(
        "--no_stop_marker",
        action="store_true",
        help="Do not append [0,0,0,0,0,0] end marker.",
    )
    parser.add_argument(
        "--meta_file",
        default="joints_handeye_calib.meta.json",
        help="Optional metadata JSON path.",
    )
    parser.add_argument("--pos_decimals", type=int, default=3)
    parser.add_argument("--rot_decimals", type=int, default=3)
    args = parser.parse_args()

    seed = parse_pose6(args.seed_pose)
    append_stop = not args.no_stop_marker

    waypoints = build_waypoints(
        seed_pose=seed,
        translation_scale=float(args.translation_scale),
        rotation_scale=float(args.rotation_scale),
        repeats_per_waypoint=int(args.repeats_per_waypoint),
        append_stop_marker=append_stop,
        pos_decimals=int(args.pos_decimals),
        rot_decimals=int(args.rot_decimals),
    )

    with open(args.out_file, "w") as f:
        json.dump(waypoints, f, indent=2)

    total_n, effective_n = compute_stats(waypoints, append_stop)
    meta = {
        "seed_pose": seed,
        "translation_scale": float(args.translation_scale),
        "rotation_scale": float(args.rotation_scale),
        "repeats_per_waypoint": int(args.repeats_per_waypoint),
        "append_stop_marker": bool(append_stop),
        "total_rows": int(total_n),
        "effective_capture_rows": int(effective_n),
        "notes": [
            "Use with Step2_capture_cube_poses.py --use_robot --joint_file <out_file>.",
            "This list is optimized for calibration diversity, not for shortest motion.",
            "Verify all waypoints are collision-safe in your Zeus controller before running.",
        ],
    }
    with open(args.meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] {args.out_file}")
    print(f"[SAVE] {args.meta_file}")
    print(f"[INFO] total rows={total_n}, effective capture rows={effective_n}")
    print("[INFO] If path is too aggressive, reduce --translation_scale / --rotation_scale.")


if __name__ == "__main__":
    main()
