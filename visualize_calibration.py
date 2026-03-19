#!/usr/bin/env python3
"""
Visualize calibration results in robot base coordinate frame.

Shows:
  - Robot base frame (origin)
  - Fixed camera positions & orientations (cam0, cam1, cam3)
  - Gripper camera position at each capture (cam2)
  - Robot gripper (TCP) positions at each capture
  - Cube positions at each capture
  - Fused point cloud (optional)

Usage:
  python visualize_calibration.py \
    --root_folder ./data/session_manual \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/session_manual/calib_out_kinematic \
    --show_pointcloud \
    --frame_idx 0
"""

import os
import json
import argparse
import numpy as np
import cv2

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt
from config import CubeConfig
from robot_comm import euler_deg_to_matrix


def load_intrinsics(intr_dir, cam_idx):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64)
    ds = float(d["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in d else 0.001
    if np.isnan(ds):
        ds = 0.001
    return K, D, ds


def create_camera_frustum(T, K, w=640, h=480, scale=0.05, color=[1, 0, 0]):
    """Create a camera frustum (pyramid wireframe) at pose T."""
    import open3d as o3d

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 4 corners of image plane at distance=scale
    corners_cam = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float64)

    pts_cam = []
    pts_cam.append([0, 0, 0])  # camera center
    for u, v in corners_cam:
        x = (u - cx) / fx * scale
        y = (v - cy) / fy * scale
        pts_cam.append([x, y, scale])

    pts_cam = np.array(pts_cam, dtype=np.float64)

    # Transform to world frame
    R = T[:3, :3]
    t = T[:3, 3]
    pts_world = pts_cam @ R.T + t

    # Lines: center to each corner + rectangle
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # rectangle
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def create_axis_frame(T, size=0.05):
    """Create a coordinate frame at pose T."""
    import open3d as o3d
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


def create_cube_wireframe(T, side_m=0.03, color=[1, 1, 0]):
    """Create a cube wireframe at pose T."""
    import open3d as o3d

    s = side_m / 2.0
    corners = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float64)

    R = T[:3, :3]
    t = T[:3, 3]
    corners_world = corners @ R.T + t

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def create_trajectory_line(positions, color=[0, 1, 0]):
    """Create a line connecting trajectory positions."""
    import open3d as o3d

    if len(positions) < 2:
        return None

    lines = [[i, i + 1] for i in range(len(positions) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(positions))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def depth_to_points(depth_u16, K, depth_scale, z_min, z_max, stride=4):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth_u16.shape[:2]
    pts, pix = [], []
    for v in range(0, h, stride):
        for u in range(0, w, stride):
            d = int(depth_u16[v, u])
            if d == 0:
                continue
            z = float(d) * float(depth_scale)
            if z < z_min or z > z_max:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts.append([x, y, z])
            pix.append((v, u))
    if len(pts) == 0:
        return np.empty((0, 3), np.float64), []
    return np.asarray(pts, np.float64), pix


def main():
    parser = argparse.ArgumentParser(description="Visualize calibration in robot base frame")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--show_pointcloud", action="store_true",
                        help="Also show fused point cloud")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Frame index for point cloud")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--z_min", type=float, default=0.2)
    parser.add_argument("--z_max", type=float, default=1.5)
    parser.add_argument("--save_ply", action="store_true",
                        help="Save visualization as PLY")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        print("[ERROR] open3d required. Install: pip install open3d")
        return

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    calib_dir = args.calib_dir or os.path.join(root, "calib_out_kinematic")

    # Load calibration
    with open(os.path.join(calib_dir, "calibration_summary.json")) as f:
        summary = json.load(f)

    transforms = {}
    for key, flat in summary["transforms"].items():
        transforms[key] = np.array(flat, dtype=np.float64).reshape(4, 4)

    gripper_cam_idx = summary["gripper_cam_idx"]
    ref_fixed = summary["ref_fixed_cam_idx"]
    fixed_cam_ids = summary["fixed_cam_ids"]
    all_cam_ids = summary["all_cam_ids"]

    T_gTc = transforms.get("T_gripper_cam")

    # Load meta
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)

    # Load intrinsics
    K_map, D_map, ds_map = {}, {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], ds_map[ci] = load_intrinsics(intr_dir, ci)

    # ─── Build visualization geometries ───
    geometries = []

    # Camera colors
    cam_colors = {
        0: [1, 0, 0],      # cam0: red
        1: [0, 0, 1],      # cam1: blue
        2: [0, 1, 0],      # cam2 (gripper): green
        3: [1, 0.5, 0],    # cam3: orange
    }

    # 1. Robot base frame (origin)
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    geometries.append(base_frame)
    print("[VIS] Robot base frame (origin, RGB=XYZ)")

    # 2. Fixed camera positions
    for ci in fixed_cam_ids:
        key = f"T_base_C{ci}"
        T = transforms.get(key)
        if T is None:
            continue

        color = cam_colors.get(ci, [0.5, 0.5, 0.5])
        frustum = create_camera_frustum(T, K_map[ci], scale=0.06, color=color)
        geometries.append(frustum)

        cam_frame = create_axis_frame(T, size=0.04)
        geometries.append(cam_frame)

        t = T[:3, 3]
        print(f"[VIS] cam{ci} (FIXED): pos=[{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm  color={color}")

    # 3. Robot gripper trajectory + gripper camera at each capture
    gripper_positions = []
    gripper_cam_positions = []
    cube_positions = []

    cfg = CubeConfig()
    cube_target = ArucoCubeTarget(cfg)

    for cap in meta["captures"]:
        rp = cap.get("robot_pose_6dof")
        if rp is None:
            continue

        T_bTg = euler_deg_to_matrix(*rp)
        gripper_pos = T_bTg[:3, 3].tolist()
        gripper_positions.append(gripper_pos)

        # Gripper frame
        gripper_frame = create_axis_frame(T_bTg, size=0.02)
        geometries.append(gripper_frame)

        # Gripper camera position
        if T_gTc is not None:
            T_base_cam2 = T_bTg @ T_gTc
            gripper_cam_positions.append(T_base_cam2[:3, 3].tolist())

            # Gripper camera frustum (smaller)
            frustum = create_camera_frustum(
                T_base_cam2, K_map[gripper_cam_idx],
                scale=0.03, color=[0, 0.8, 0])
            geometries.append(frustum)

        # Cube position from fixed cameras
        eid = cap["event_id"]
        for ci in fixed_cam_ids:
            ci_str = str(ci)
            cinfo = cap["cams"].get(ci_str)
            if not cinfo or not cinfo.get("saved"):
                continue

            rgb_path = os.path.join(root, cinfo["rgb_path"])
            img = cv2.imread(rgb_path)
            if img is None:
                continue

            ok, rvec, tvec, used = cube_target.solve_pnp_cube(
                img, K_map[ci], D_map[ci], min_markers=1, reproj_thr_mean_px=10.0)
            if not ok:
                continue

            T_cam_O = rodrigues_to_Rt(rvec, tvec)
            T_base_cam = transforms.get(f"T_base_C{ci}")
            if T_base_cam is not None:
                T_base_O = T_base_cam @ T_cam_O
                cube_positions.append(T_base_O[:3, 3].tolist())

                # Cube wireframe (only once per event from best camera)
                cube_wf = create_cube_wireframe(T_base_O, cfg.cube_side_m, color=[1, 1, 0])
                geometries.append(cube_wf)
                break  # one per event

    # 4. Gripper trajectory line
    if len(gripper_positions) >= 2:
        traj = create_trajectory_line(gripper_positions, color=[0.5, 0.5, 0.5])
        if traj:
            geometries.append(traj)
    print(f"[VIS] Gripper trajectory: {len(gripper_positions)} poses")

    # 5. Gripper camera trajectory
    if len(gripper_cam_positions) >= 2:
        traj = create_trajectory_line(gripper_cam_positions, color=[0, 0.6, 0])
        if traj:
            geometries.append(traj)
    print(f"[VIS] Gripper camera trajectory: {len(gripper_cam_positions)} poses")

    # 6. Cube positions
    print(f"[VIS] Cube positions: {len(cube_positions)}")

    # 7. Optional: fused point cloud
    if args.show_pointcloud:
        print(f"\n[VIS] Loading point cloud (frame={args.frame_idx})...")
        all_pts, all_cols = [], []

        for ci in fixed_cam_ids:
            rgb_path = os.path.join(root, f"cam{ci}", f"rgb_{args.frame_idx:05d}.jpg")
            depth_path = os.path.join(root, f"cam{ci}", f"depth_{args.frame_idx:05d}.png")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue

            rgb_bgr = cv2.imread(rgb_path)
            depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth_u16 is None:
                continue

            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            pts_cam, pix = depth_to_points(
                depth_u16, K_map[ci], ds_map[ci],
                args.z_min, args.z_max, args.stride)

            if pts_cam.shape[0] == 0:
                continue

            T = transforms.get(f"T_base_C{ci}")
            if T is None:
                continue

            R, t = T[:3, :3], T[:3, 3]
            pts_base = pts_cam @ R.T + t
            cols = np.array([rgb[v, u] for v, u in pix], dtype=np.float64)

            all_pts.append(pts_base)
            all_cols.append(cols)
            print(f"  cam{ci}: {pts_base.shape[0]} points")

        if all_pts:
            P = np.concatenate(all_pts)
            C = np.concatenate(all_cols)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(P)
            pcd.colors = o3d.utility.Vector3dVector(C)
            geometries.append(pcd)
            print(f"  Total: {P.shape[0]} points")

    # ─── Legend ───
    print("\n=== Legend ===")
    print("  Large RGB axes  : Robot base frame (origin)")
    print("  Red frustum     : cam0 (fixed, right)")
    print("  Blue frustum    : cam1 (fixed, front)")
    print("  Orange frustum  : cam3 (fixed, left)")
    print("  Green frustums  : cam2 (gripper) at each capture")
    print("  Small axes      : Gripper TCP at each capture")
    print("  Yellow wireframe: Cube positions")
    print("  Gray line       : Gripper trajectory")
    print("  Green line      : Gripper camera trajectory")

    # ─── Save PLY ───
    if args.save_ply and args.show_pointcloud and all_pts:
        out_path = os.path.join(root, "visualization_base_frame.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"\n[SAVE] {out_path}")

    # ─── Visualize ───
    print("\n[VIS] Opening viewer... (close window to exit)")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Calibration - Robot Base Frame",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
