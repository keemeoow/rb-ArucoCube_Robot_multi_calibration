#!/usr/bin/env python3
"""
Generate Zeus calibration cycle waypoints for repeated:
  내려놓기(place_pose) -> 위로 상승(capture_pose) -> 촬영

Output JSON is compatible with:
  newStep2_capture_cube_poses.py --cycle_file <json>

Output format (recommended):
[
  {
    "cycle_index": 0,
    "place_pose_6dof": [x,y,z,rz,ry,rx],
    "capture_pose_6dof": [x,y,z,rz,ry,rx]
  },
  ...,
  {"action":"stop"}   # optional
]

Example:
  python newmake_zeus_calib_waypoints.py \
    --seed_pose 450,-120,380,180,0,180 \
    --out_file newjoints_handeye_cycle.json \
    --down_dz_mm -60 \
    --translation_scale 1.0 \
    --rotation_scale 1.0
"""

import argparse
import json
from typing import Dict, List, Tuple


Pose6 = List[float]


# Same diverse motion template as the original generator.
# [dx_mm, dy_mm, dz_mm, drz_deg, dry_deg, drx_deg]
DEFAULT_DELTAS: List[Pose6] = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 15, 0, 0],
    [0, 0, 0, -15, 0, 0],
    [0, 0, 0, 0, 20, 0],
    [0, 0, 0, 0, -20, 0],
    [0, 0, 0, 0, 0, 15],
    [0, 0, 0, 0, 0, -15],
    [0, 0, 0, 12, 12, 0],
    [0, 0, 0, -12, -12, 0],

    [50, 0, 0, 0, 10, 0],
    [35, 35, 0, 8, 8, 0],
    [0, 50, 0, 10, 0, 0],
    [-35, 35, 0, 8, -8, 0],
    [-50, 0, 0, 0, -10, 0],
    [-35, -35, 0, -8, -8, 0],
    [0, -50, 0, -10, 0, 0],
    [35, -35, 0, -8, 8, 0],

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
    out = [float(x) for x in delta]
    out[0] *= float(t_scale)
    out[1] *= float(t_scale)
    out[2] *= float(t_scale)
    out[3] *= float(r_scale)
    out[4] *= float(r_scale)
    out[5] *= float(r_scale)
    return out


def round_pose(p: Pose6, pos_decimals: int, rot_decimals: int) -> Pose6:
    return [
        round(float(p[0]), int(pos_decimals)),
        round(float(p[1]), int(pos_decimals)),
        round(float(p[2]), int(pos_decimals)),
        round(float(p[3]), int(rot_decimals)),
        round(float(p[4]), int(rot_decimals)),
        round(float(p[5]), int(rot_decimals)),
    ]


def make_place_from_capture(capture_pose: Pose6, down_dz_mm: float) -> Pose6:
    place = [float(x) for x in capture_pose]
    place[2] = float(place[2] + down_dz_mm)
    return place


def build_cycle_waypoints(
    seed_pose: Pose6,
    down_dz_mm: float,
    translation_scale: float,
    rotation_scale: float,
    repeats_per_waypoint: int,
    append_stop_marker: bool,
    pos_decimals: int,
    rot_decimals: int,
) -> List[Dict[str, object]]:
    if repeats_per_waypoint < 1:
        raise ValueError("repeats_per_waypoint must be >= 1")

    out: List[Dict[str, object]] = []
    cycle_idx = 0

    for d in DEFAULT_DELTAS:
        sd = scale_delta(d, translation_scale, rotation_scale)
        cap_pose = round_pose(add_pose(seed_pose, sd), pos_decimals, rot_decimals)
        place_pose = round_pose(make_place_from_capture(cap_pose, down_dz_mm), pos_decimals, rot_decimals)

        for _ in range(repeats_per_waypoint):
            out.append({
                "cycle_index": int(cycle_idx),
                "place_pose_6dof": [float(x) for x in place_pose],
                "capture_pose_6dof": [float(x) for x in cap_pose],
            })
            cycle_idx += 1

    if append_stop_marker:
        out.append({"action": "stop"})

    return out


def compute_stats(cycles: List[Dict[str, object]], has_stop_marker: bool) -> Tuple[int, int]:
    total = len(cycles)
    effective = total - 1 if has_stop_marker and total > 0 else total
    return total, effective


def extract_capture_pose_list(cycles: List[Dict[str, object]]) -> List[Pose6]:
    out: List[Pose6] = []
    for row in cycles:
        if str(row.get("action", "")).strip().lower() in ["stop", "quit", "end"]:
            break
        p = row.get("capture_pose_6dof")
        if isinstance(p, list) and len(p) == 6:
            out.append([float(x) for x in p])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate Zeus cycle waypoint set for robot-assisted calibration."
    )
    parser.add_argument(
        "--seed_pose",
        required=True,
        help="Reference 6D pose (x,y,z,rz,ry,rx) in mm/deg, comma-separated.",
    )
    parser.add_argument(
        "--out_file",
        default="newjoints_handeye_cycle.json",
        help="Output cycle JSON path.",
    )
    parser.add_argument(
        "--down_dz_mm",
        type=float,
        default=-60.0,
        help="Place pose z offset from capture pose (negative => 내려놓기).",
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
        help="How many times each cycle waypoint is repeated.",
    )
    parser.add_argument(
        "--no_stop_marker",
        action="store_true",
        help="Do not append {'action':'stop'} end marker.",
    )
    parser.add_argument(
        "--meta_file",
        default="newjoints_handeye_cycle.meta.json",
        help="Optional metadata JSON path.",
    )
    parser.add_argument("--pos_decimals", type=int, default=3)
    parser.add_argument("--rot_decimals", type=int, default=3)

    # Optional legacy export for old Step2 script
    parser.add_argument(
        "--legacy_capture_pose_file",
        default="",
        help="Optional output path for legacy [[d1..d6], ...] capture-only list.",
    )

    args = parser.parse_args()

    seed = parse_pose6(args.seed_pose)
    append_stop = not args.no_stop_marker

    cycles = build_cycle_waypoints(
        seed_pose=seed,
        down_dz_mm=float(args.down_dz_mm),
        translation_scale=float(args.translation_scale),
        rotation_scale=float(args.rotation_scale),
        repeats_per_waypoint=int(args.repeats_per_waypoint),
        append_stop_marker=append_stop,
        pos_decimals=int(args.pos_decimals),
        rot_decimals=int(args.rot_decimals),
    )

    with open(args.out_file, "w") as f:
        json.dump(cycles, f, indent=2)

    total_n, effective_n = compute_stats(cycles, append_stop)

    meta = {
        "seed_pose_capture": seed,
        "down_dz_mm": float(args.down_dz_mm),
        "translation_scale": float(args.translation_scale),
        "rotation_scale": float(args.rotation_scale),
        "repeats_per_waypoint": int(args.repeats_per_waypoint),
        "append_stop_marker": bool(append_stop),
        "total_rows": int(total_n),
        "effective_cycle_rows": int(effective_n),
        "format": {
            "type": "capture_cycle_v1",
            "required_fields": ["place_pose_6dof", "capture_pose_6dof"],
        },
        "notes": [
            "Use with newStep2_capture_cube_poses.py --use_robot --cycle_file <out_file>",
            "Each cycle executes place -> capture pose, then cameras are captured.",
            "Verify all waypoints are collision-safe in the Zeus controller before running.",
        ],
    }

    with open(args.meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] {args.out_file}")
    print(f"[SAVE] {args.meta_file}")
    print(f"[INFO] total rows={total_n}, effective cycle rows={effective_n}")

    if args.legacy_capture_pose_file:
        legacy = extract_capture_pose_list(cycles)
        with open(args.legacy_capture_pose_file, "w") as f:
            json.dump(legacy, f, indent=2)
        print(f"[SAVE] {args.legacy_capture_pose_file}")
        print("[INFO] legacy capture-only list generated for old Step2 compatibility")


if __name__ == "__main__":
    main()
