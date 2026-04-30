#!/usr/bin/env python3
"""
Generate auto-capture waypoints for hand-eye + multi-camera calibration.

Workflow:
  1) Run a manual session once with server/robot_calb.py + Step2_capture.py
     and record ~5-9 cube placements (each with at least 1 'c' capture).
  2) Run this script with --from_session pointing at that session dir to
     extract per-set cube centers + place_joints, then generate a hemispherical
     capture pattern around each cube position.
  3) The output JSON uses an extended format (placements -> captures list)
     that requires server-side --auto modifications to consume.
     Format is kept human-readable for review before committing server changes.

TCP convention assumed:
  - Position [x, y, z, rz, ry, rx] in mm/deg, ZYX intrinsic Euler:
        R = Rz(rz) @ Ry(ry) @ Rx(rx)
  - Gripper z-axis (Tool 3) points away from the robot flange toward the
    workspace; a downward-pointing gripper has rx near +/-180 and ry near 0.
  - Gripper camera is mounted on the gripper and looks along the gripper z-axis.
  Verify by jogging to one generated pose before running --auto.

Examples:
  # Auto-extract from existing manual session, generate ~24 captures/placement
  python tools/generate_capture_waypoints.py \
      --from_session ./data/session \
      --output ./data/session/capture_waypoints_auto.json

  # Custom pattern
  python tools/generate_capture_waypoints.py \
      --from_session ./data/session \
      --output ./out.json \
      --distances 280,380 \
      --elevations 90,55 \
      --azimuths 0,90,180,270 \
      --rz_spins 0,90 \
      --min_tcp_z 60
"""

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def lookat_tcp(
    cube_xyz: List[float],
    distance_mm: float,
    elevation_deg: float,
    azimuth_deg: float,
    rz_spin_deg: float = 0.0,
) -> List[float]:
    """Compute TCP pose [x,y,z, rz,ry,rx] (mm, deg).

    Gripper z-axis (column 3 of rotation matrix in world frame) points from TCP
    toward cube center, so the gripper camera (mounted along gripper z) views
    the cube near image center.

    elevation_deg = 90 -> TCP directly above cube (top-down).
    elevation_deg = 0  -> TCP at cube height, offset horizontally.
    azimuth_deg  = 0   -> TCP offset in +x direction (xy-plane projection).
    rz_spin_deg        -> extra rotation about gripper z-axis (axis diversity).
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    # Unit vector from cube toward TCP.
    ux = math.cos(el) * math.cos(az)
    uy = math.cos(el) * math.sin(az)
    uz = math.sin(el)

    cube = np.array(cube_xyz[:3], dtype=float)
    tcp_pos = cube + distance_mm * np.array([ux, uy, uz])

    # Gripper z-axis (in world): from TCP toward cube center.
    z_axis = -np.array([ux, uy, uz])

    # Reference vector for x-axis. Avoid degeneracy when z is parallel to ref.
    world_ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(z_axis, world_ref))) > 0.95:
        world_ref = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(world_ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    # rz_spin: rotate about the gripper's own z-axis (body frame).
    if rz_spin_deg != 0.0:
        R = R @ Rotation.from_euler("z", rz_spin_deg, degrees=True).as_matrix()

    rz, ry, rx = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)

    return [
        round(float(tcp_pos[0]), 3),
        round(float(tcp_pos[1]), 3),
        round(float(tcp_pos[2]), 3),
        round(float(rz), 4),
        round(float(ry), 4),
        round(float(rx), 4),
    ]


def generate_pattern(
    cube_xyz: List[float],
    distances: List[float],
    elevations: List[float],
    azimuths: List[float],
    rz_spins: List[float],
    min_tcp_z: float,
    workspace_x: Optional[Tuple[float, float]],
    workspace_y: Optional[Tuple[float, float]],
    max_radius_xy: Optional[float],
) -> Tuple[List[dict], List[dict]]:
    """Generate hemisphere capture poses around one cube. Returns (accepted, rejected)."""
    accepted: List[dict] = []
    rejected: List[dict] = []

    for distance in distances:
        for el in elevations:
            # At elevation ~90 the TCP position is the same regardless of azimuth.
            azs = [0.0] if el >= 89.5 else azimuths
            for az in azs:
                for spin in rz_spins:
                    tcp = lookat_tcp(cube_xyz, distance, el, az, spin)
                    rec = {
                        "capture_tcp": tcp,
                        "meta": {
                            "distance_mm": distance,
                            "elevation_deg": el,
                            "azimuth_deg": az,
                            "rz_spin_deg": spin,
                        },
                    }
                    reasons: List[str] = []
                    if tcp[2] < min_tcp_z:
                        reasons.append(f"tcp_z {tcp[2]:.1f} < {min_tcp_z}")
                    if workspace_x and not (workspace_x[0] <= tcp[0] <= workspace_x[1]):
                        reasons.append(f"tcp_x {tcp[0]:.1f} out of {workspace_x}")
                    if workspace_y and not (workspace_y[0] <= tcp[1] <= workspace_y[1]):
                        reasons.append(f"tcp_y {tcp[1]:.1f} out of {workspace_y}")
                    if max_radius_xy is not None:
                        r = math.hypot(tcp[0], tcp[1])
                        if r > max_radius_xy:
                            reasons.append(f"radius_xy {r:.1f} > {max_radius_xy}")
                    if reasons:
                        rec["reject_reasons"] = reasons
                        rejected.append(rec)
                    else:
                        accepted.append(rec)
    return accepted, rejected


def load_placements_from_session(session_dir: str) -> Tuple[dict, List[dict]]:
    """Extract placements from a previous manual session.

    - meta.json provides set_cube_center_6dof per set_index (most reliable).
    - capture_waypoints.json provides place_joints per set_index plus the
      shared set_joints/set_tcp.
    Set indices missing place_joints (e.g. captured without prior 'go') are
    skipped with a warning.
    """
    wp_path = os.path.join(session_dir, "capture_waypoints.json")
    meta_path = os.path.join(session_dir, "meta.json")
    if not os.path.exists(wp_path):
        sys.exit(f"[ERROR] capture_waypoints.json not found: {wp_path}")
    if not os.path.exists(meta_path):
        sys.exit(f"[ERROR] meta.json not found: {meta_path}")

    with open(wp_path) as f:
        wp_data = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    set_cube_centers: dict = {}
    for c in meta.get("captures", []):
        si = c.get("set_index")
        sc = c.get("set_cube_center_6dof")
        if si is not None and sc is not None:
            set_cube_centers.setdefault(si, sc)

    place_joints_map: dict = {}
    for wp in wp_data.get("waypoints", []):
        si = wp.get("set_index")
        pj = wp.get("place_joints")
        if si is not None and pj is not None:
            place_joints_map.setdefault(si, pj)

    placements = []
    for si in sorted(set_cube_centers.keys()):
        if si not in place_joints_map:
            print(f"[WARN] set_index={si} has cube_center but no place_joints; skipping")
            continue
        placements.append({
            "set_index": si,
            "place_joints": place_joints_map[si],
            "set_cube_center": set_cube_centers[si],
        })

    header = {
        "set_joints": wp_data.get("set_joints"),
        "set_tcp": wp_data.get("set_tcp"),
    }
    return header, placements


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_session", type=str,
                     help="Manual session dir (containing capture_waypoints.json + meta.json)")
    src.add_argument("--seed", type=str,
                     help="Hand-crafted seed JSON {set_joints, set_tcp, placements:[...]}")

    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path (extended format)")

    parser.add_argument("--distances", type=parse_csv_floats, default=[250.0, 350.0],
                        help="TCP-to-cube distances in mm (default: 250,350)")
    parser.add_argument("--elevations", type=parse_csv_floats, default=[90.0, 60.0, 30.0],
                        help="Elevation angles in deg, 90=top-down (default: 90,60,30)")
    parser.add_argument("--azimuths", type=parse_csv_floats, default=[0.0, 90.0, 180.0, 270.0],
                        help="Azimuth angles in deg around cube (default: 0,90,180,270)")
    parser.add_argument("--rz_spins", type=parse_csv_floats, default=[0.0, 90.0, 180.0],
                        help="Extra rotations about gripper z-axis (default: 0,90,180)")

    parser.add_argument("--min_tcp_z", type=float, default=50.0,
                        help="Reject poses with TCP z below this (mm, default: 50)")
    parser.add_argument("--workspace_x", type=parse_csv_floats, default=None,
                        help="TCP x bounds 'min,max' in mm (optional)")
    parser.add_argument("--workspace_y", type=parse_csv_floats, default=None,
                        help="TCP y bounds 'min,max' in mm (optional)")
    parser.add_argument("--max_radius_xy", type=float, default=None,
                        help="Reject poses with sqrt(x^2+y^2) above this (mm, optional)")

    parser.add_argument("--preview", action="store_true",
                        help="Print summary + sample poses without writing the file")

    args = parser.parse_args()

    if args.from_session:
        header, placements = load_placements_from_session(args.from_session)
    else:
        with open(args.seed) as f:
            seed = json.load(f)
        header = {
            "set_joints": seed["set_joints"],
            "set_tcp": seed.get("set_tcp"),
        }
        placements = seed["placements"]

    workspace_x = tuple(args.workspace_x) if args.workspace_x else None
    workspace_y = tuple(args.workspace_y) if args.workspace_y else None

    print(f"[INFO] {len(placements)} placements")
    print(f"[INFO] Pattern: dist={args.distances} el={args.elevations} "
          f"az={args.azimuths} rz_spin={args.rz_spins}")
    print(f"[INFO] Filters: min_tcp_z={args.min_tcp_z} workspace_x={workspace_x} "
          f"workspace_y={workspace_y} max_radius_xy={args.max_radius_xy}")

    output_placements = []
    pose_index = 0
    total_accepted = 0
    total_rejected = 0

    for p in placements:
        cube_xyz = p["set_cube_center"][:3]
        accepted, rejected = generate_pattern(
            cube_xyz,
            args.distances, args.elevations, args.azimuths, args.rz_spins,
            min_tcp_z=args.min_tcp_z,
            workspace_x=workspace_x,
            workspace_y=workspace_y,
            max_radius_xy=args.max_radius_xy,
        )

        captures = []
        for cap in accepted:
            captures.append({"pose_index": pose_index, **cap})
            pose_index += 1

        output_placements.append({
            "set_index": p["set_index"],
            "place_joints": p["place_joints"],
            "set_cube_center": p["set_cube_center"],
            "captures": captures,
        })

        total_accepted += len(accepted)
        total_rejected += len(rejected)
        print(f"  set_index={p['set_index']}: cube=("
              f"{cube_xyz[0]:.1f}, {cube_xyz[1]:.1f}, {cube_xyz[2]:.1f}) -> "
              f"{len(accepted)} accepted, {len(rejected)} rejected")

        if args.preview and rejected:
            for rj in rejected[:3]:
                print(f"    REJECT meta={rj['meta']} reasons={rj['reject_reasons']}")

    print(f"\n[INFO] Total: {total_accepted} captures across "
          f"{len(output_placements)} placements ({total_rejected} rejected)")

    if args.preview:
        print("\n[PREVIEW] First placement, first 5 captures:")
        if output_placements and output_placements[0]["captures"]:
            for cap in output_placements[0]["captures"][:5]:
                tcp = cap["capture_tcp"]
                m = cap["meta"]
                print(f"  pose_idx={cap['pose_index']} "
                      f"d={m['distance_mm']} el={m['elevation_deg']} "
                      f"az={m['azimuth_deg']} rz_spin={m['rz_spin_deg']}\n"
                      f"    TCP=[{tcp[0]:8.1f}, {tcp[1]:8.1f}, {tcp[2]:7.1f},"
                      f" rz={tcp[3]:7.2f} ry={tcp[4]:7.2f} rx={tcp[5]:7.2f}]")
        return

    output_data = {
        "format_version": 2,
        "set_joints": header["set_joints"],
        "set_tcp": header["set_tcp"],
        "placements": output_placements,
        "_meta": {
            "generator": "tools/generate_capture_waypoints.py",
            "pattern": {
                "distances": args.distances,
                "elevations": args.elevations,
                "azimuths": args.azimuths,
                "rz_spins": args.rz_spins,
            },
            "filters": {
                "min_tcp_z": args.min_tcp_z,
                "workspace_x": list(workspace_x) if workspace_x else None,
                "workspace_y": list(workspace_y) if workspace_y else None,
                "max_radius_xy": args.max_radius_xy,
            },
        },
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()
