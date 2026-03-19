#!/usr/bin/env python3
"""
Generate Zeus waypoints for Place-and-Capture calibration.

Workflow:
  Robot places the ArUco cube at different positions on the workspace,
  then moves up so the gripper camera + fixed cameras can all capture.

Output format: JSON list of {"place": [x,y,z,rz,ry,rx], "capture": [x,y,z,rz,ry,rx]}
  - place:   TCP pose where robot releases the cube on the workspace
  - capture: TCP pose where robot moves to for image capture (above the cube)

Compatible with: new2_Step2_capture_cube_poses.py --waypoint_file <json>

Usage:
  python new2_make_zeus_calib_waypoints.py \
    --seed_place 450,-120,200,180,0,180 \
    --capture_z_offset 200 \
    --out_file new2_waypoints.json
"""

import argparse
import json
from typing import List, Tuple, Dict

Pose6 = List[float]


# ─── Place position deltas ───
# [dx_mm, dy_mm, dz_mm, drz_deg, dry_deg, drx_deg]
# Cube is placed at seed + delta on the workspace surface
DEFAULT_PLACE_DELTAS: List[Pose6] = [
    # Phase A: Center position, cube rotation diversity
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 30, 0, 0],
    [0, 0, 0, 60, 0, 0],
    [0, 0, 0, 90, 0, 0],

    # Phase B: XY position ring (cube at different table spots)
    [60, 0, 0, 0, 0, 0],
    [-60, 0, 0, 0, 0, 0],
    [0, 60, 0, 0, 0, 0],
    [0, -60, 0, 0, 0, 0],
    [40, 40, 0, 0, 0, 0],
    [-40, 40, 0, 0, 0, 0],
    [40, -40, 0, 0, 0, 0],
    [-40, -40, 0, 0, 0, 0],

    # Phase C: Position + rotation combinations
    [60, 0, 0, 45, 0, 0],
    [-60, 0, 0, -45, 0, 0],
    [0, 60, 0, 30, 0, 0],
    [0, -60, 0, -30, 0, 0],
    [40, 40, 0, 60, 0, 0],
    [-40, -40, 0, 90, 0, 0],

    # Phase D: Wider positions for fixed camera diversity
    [80, 0, 0, 0, 0, 0],
    [-80, 0, 0, 0, 0, 0],
    [0, 80, 0, 45, 0, 0],
    [0, -80, 0, -45, 0, 0],
]


# ─── Capture angle deltas (applied to capture position) ───
# [drz_deg, dry_deg, drx_deg] relative to place orientation
# These create different gripper camera viewpoints for the same cube placement
DEFAULT_CAPTURE_ANGLE_DELTAS: List[List[float]] = [
    [0, 0, 0],        # straight down (default)
]

# For multi-angle capture per placement, use:
MULTI_ANGLE_CAPTURE_DELTAS: List[List[float]] = [
    [0, 0, 0],        # straight down
    [10, 0, 0],       # slight tilt
    [-10, 0, 0],
    [0, 10, 0],
    [0, -10, 0],
]


def parse_pose6(text: str) -> Pose6:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 6:
        raise ValueError("Pose must have 6 comma-separated values: x,y,z,rz,ry,rx")
    return vals


def add_pose(a: Pose6, b: Pose6) -> Pose6:
    return [float(a[i] + b[i]) for i in range(6)]


def round_pose(p: Pose6, pos_dec: int = 3, rot_dec: int = 3) -> Pose6:
    return [
        round(float(p[0]), pos_dec), round(float(p[1]), pos_dec),
        round(float(p[2]), pos_dec), round(float(p[3]), rot_dec),
        round(float(p[4]), rot_dec), round(float(p[5]), rot_dec),
    ]


def build_waypoints(
    seed_place: Pose6,
    capture_z_offset: float,
    capture_xy_offset: Tuple[float, float] = (0.0, 0.0),
    place_deltas: List[Pose6] = None,
    capture_angle_deltas: List[List[float]] = None,
    t_scale: float = 1.0,
    r_scale: float = 1.0,
    pos_dec: int = 3,
    rot_dec: int = 3,
) -> List[Dict[str, Pose6]]:
    """
    Generate place + capture waypoint pairs.

    For each place delta:
      place_pose = seed_place + scaled_delta
      capture_pose = place_pose + [capture_xy_offset_x, capture_xy_offset_y, capture_z_offset, angle_deltas...]

    For each capture angle delta (if multi_angle):
      generates an additional waypoint with rotated capture pose.
    """
    if place_deltas is None:
        place_deltas = DEFAULT_PLACE_DELTAS
    if capture_angle_deltas is None:
        capture_angle_deltas = DEFAULT_CAPTURE_ANGLE_DELTAS

    waypoints = []

    for pd in place_deltas:
        # Scale translation and rotation deltas
        scaled = [
            pd[0] * t_scale, pd[1] * t_scale, pd[2] * t_scale,
            pd[3] * r_scale, pd[4] * r_scale, pd[5] * r_scale,
        ]
        place_pose = round_pose(add_pose(seed_place, scaled), pos_dec, rot_dec)

        for cad in capture_angle_deltas:
            capture_delta = [
                capture_xy_offset[0], capture_xy_offset[1], capture_z_offset,
                cad[0] * r_scale, cad[1] * r_scale, cad[2] * r_scale,
            ]
            capture_pose = round_pose(add_pose(place_pose, capture_delta), pos_dec, rot_dec)

            waypoints.append({
                "place": place_pose,
                "capture": capture_pose,
            })

    return waypoints


def main():
    parser = argparse.ArgumentParser(
        description="Generate Place-and-Capture waypoints for multi-camera calibration."
    )
    parser.add_argument(
        "--seed_place", required=True,
        help="Reference 6D pose for placing cube (x,y,z,rz,ry,rx in mm/deg).",
    )
    parser.add_argument(
        "--capture_z_offset", type=float, default=200.0,
        help="Height above place position for capture (mm). Default: 200.",
    )
    parser.add_argument(
        "--capture_xy_offset", type=str, default="0,0",
        help="XY offset for capture position (dx,dy in mm). Default: 0,0.",
    )
    parser.add_argument(
        "--out_file", default="new2_waypoints.json",
        help="Output JSON path.",
    )
    parser.add_argument("--translation_scale", type=float, default=1.0)
    parser.add_argument("--rotation_scale", type=float, default=1.0)
    parser.add_argument(
        "--multi_angle", action="store_true",
        help="Generate multiple capture angles per place position.",
    )
    parser.add_argument("--pos_decimals", type=int, default=3)
    parser.add_argument("--rot_decimals", type=int, default=3)
    parser.add_argument(
        "--meta_file", default="new2_waypoints.meta.json",
        help="Metadata JSON path.",
    )
    args = parser.parse_args()

    seed = parse_pose6(args.seed_place)
    cap_xy = [float(x) for x in args.capture_xy_offset.split(",")]
    if len(cap_xy) != 2:
        cap_xy = [0.0, 0.0]

    capture_angles = MULTI_ANGLE_CAPTURE_DELTAS if args.multi_angle else DEFAULT_CAPTURE_ANGLE_DELTAS

    waypoints = build_waypoints(
        seed_place=seed,
        capture_z_offset=args.capture_z_offset,
        capture_xy_offset=tuple(cap_xy),
        capture_angle_deltas=capture_angles,
        t_scale=args.translation_scale,
        r_scale=args.rotation_scale,
        pos_dec=args.pos_decimals,
        rot_dec=args.rot_decimals,
    )

    with open(args.out_file, "w") as f:
        json.dump(waypoints, f, indent=2)

    meta = {
        "seed_place": seed,
        "capture_z_offset": args.capture_z_offset,
        "capture_xy_offset": cap_xy,
        "translation_scale": args.translation_scale,
        "rotation_scale": args.rotation_scale,
        "multi_angle": args.multi_angle,
        "total_waypoints": len(waypoints),
        "notes": [
            "Use with: new2_Step2_capture_cube_poses.py --waypoint_file <out_file>",
            "place: TCP pose where robot releases cube on workspace",
            "capture: TCP pose where robot moves to for gripper+fixed camera capture",
            "Verify all poses are collision-safe before running.",
        ],
    }
    with open(args.meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] {args.out_file}")
    print(f"[SAVE] {args.meta_file}")
    print(f"[INFO] {len(waypoints)} waypoints generated")
    print(f"[INFO] Capture Z offset: {args.capture_z_offset} mm")
    if args.multi_angle:
        print(f"[INFO] Multi-angle mode: {len(MULTI_ANGLE_CAPTURE_DELTAS)} angles per placement")


if __name__ == "__main__":
    main()
