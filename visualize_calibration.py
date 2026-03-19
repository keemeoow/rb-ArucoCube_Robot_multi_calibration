#!/usr/bin/env python3
"""
Visualize calibration results in robot base coordinate frame.

Shows:
  - Robot base frame (origin) with XYZ axes + labels
  - Fixed camera positions with coordinate axes + labels
  - Gripper camera positions at each capture
  - Robot gripper (TCP) positions at each capture
  - Cube positions
  - Ground grid (XY plane)

Usage:
  python visualize_calibration.py \
    --root_folder ./data/session_manual \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/session_manual/calib_out_direct
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
    return K, D


def create_axis_lines(T, size=0.1, thickness=3):
    """Create thick XYZ axis lines at pose T. Returns list of LineSet."""
    import open3d as o3d

    origin = T[:3, 3]
    R = T[:3, :3]

    axes = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R, G, B = X, Y, Z

    for i, color in enumerate(colors):
        direction = R[:, i] * size
        end = origin + direction

        pts = np.array([origin, end])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector([[0, 1]])
        ls.colors = o3d.utility.Vector3dVector([color])
        axes.append(ls)

    return axes


def create_3d_text(text, position, scale=0.02, color=[1, 1, 1]):
    """Create 3D text as a point cloud with text rendered on a plane."""
    import open3d as o3d

    # Use a simple sphere as label marker + store text info
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=scale * 0.3)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    return sphere


def create_text_label(text, position, T=None, axis_size=0.1, color=[1, 1, 1]):
    """Create text labels for axes (X, Y, Z) at end of axis lines."""
    import open3d as o3d

    labels = []
    if T is not None:
        origin = T[:3, 3]
        R = T[:3, :3]
        axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        axis_names = ["X", "Y", "Z"]

        for i in range(3):
            end_pos = origin + R[:, i] * axis_size * 1.15
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sphere.translate(end_pos)
            sphere.paint_uniform_color(axis_colors[i])
            labels.append(sphere)

    # Main label
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    labels.append(sphere)

    return labels


def create_ground_grid(size=1.0, spacing=0.1, z=0.0, color=[0.3, 0.3, 0.3]):
    """Create a ground grid on XY plane at z height."""
    import open3d as o3d

    lines = []
    points = []
    idx = 0

    n = int(size / spacing)

    # Lines parallel to X axis
    for i in range(-n, n + 1):
        y = i * spacing
        points.append([-size, y, z])
        points.append([size, y, z])
        lines.append([idx, idx + 1])
        idx += 2

    # Lines parallel to Y axis
    for i in range(-n, n + 1):
        x = i * spacing
        points.append([x, -size, z])
        points.append([x, size, z])
        lines.append([idx, idx + 1])
        idx += 2

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


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


def create_trajectory_line(positions, color=[0.5, 0.5, 0.5]):
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


def main():
    parser = argparse.ArgumentParser(description="Visualize calibration in robot base frame")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--axis_size", type=float, default=0.08,
                        help="Size of coordinate axes (meters)")
    parser.add_argument("--grid_size", type=float, default=0.8,
                        help="Ground grid size (meters)")
    parser.add_argument("--grid_spacing", type=float, default=0.1,
                        help="Ground grid line spacing (meters)")
    args = parser.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        print("[ERROR] open3d required. Install: pip install open3d")
        return

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    calib_dir = args.calib_dir or os.path.join(root, "calib_out_direct")

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

    K_map = {}
    for ci in all_cam_ids:
        K_map[ci], _ = load_intrinsics(intr_dir, ci)

    cfg = CubeConfig()
    cube_target = ArucoCubeTarget(cfg)

    geometries = []
    labels_3d = []  # (text, position, color) for overlay

    # Camera colors
    cam_colors = {
        0: [1.0, 0.2, 0.2],    # cam0: red
        1: [0.2, 0.4, 1.0],    # cam1: blue
        2: [0.2, 0.9, 0.2],    # cam2 (gripper): green
        3: [1.0, 0.6, 0.1],    # cam3: orange
    }
    cam_names = {
        0: "cam0 (R)",
        1: "cam1 (F)",
        2: "cam2 (G)",
        3: "cam3 (L)",
    }

    # ─── 1. Ground grid (XY plane) ───
    grid = create_ground_grid(
        size=args.grid_size, spacing=args.grid_spacing, z=0.0,
        color=[0.25, 0.25, 0.25])
    geometries.append(grid)

    # Highlight X and Y axes on grid
    x_axis_line = o3d.geometry.LineSet()
    x_axis_line.points = o3d.utility.Vector3dVector([
        [-args.grid_size, 0, 0], [args.grid_size, 0, 0]])
    x_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    x_axis_line.colors = o3d.utility.Vector3dVector([[0.6, 0.15, 0.15]])
    geometries.append(x_axis_line)

    y_axis_line = o3d.geometry.LineSet()
    y_axis_line.points = o3d.utility.Vector3dVector([
        [0, -args.grid_size, 0], [0, args.grid_size, 0]])
    y_axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    y_axis_line.colors = o3d.utility.Vector3dVector([[0.15, 0.6, 0.15]])
    geometries.append(y_axis_line)

    print("[VIS] Ground grid (XY plane)")

    # ─── 2. Robot base frame (origin) ───
    T_base = np.eye(4)
    base_axes = create_axis_lines(T_base, size=args.axis_size * 2.0)
    geometries.extend(base_axes)

    # Base label
    labels_3d.append(("ROBOT BASE", T_base[:3, 3] + np.array([0, 0, args.axis_size * 2.2]), [1, 1, 1]))

    # Axis end labels for base
    labels_3d.append(("X", np.array([args.axis_size * 2.2, 0, 0]), [1, 0, 0]))
    labels_3d.append(("Y", np.array([0, args.axis_size * 2.2, 0]), [0, 1, 0]))
    labels_3d.append(("Z", np.array([0, 0, args.axis_size * 2.2]), [0, 0, 1]))

    print("[VIS] Robot base frame (origin)")

    # ─── 3. Fixed cameras ───
    for ci in fixed_cam_ids:
        key = f"T_base_C{ci}"
        T = transforms.get(key)
        if T is None:
            continue

        color = cam_colors.get(ci, [0.5, 0.5, 0.5])

        # Coordinate axes
        axes = create_axis_lines(T, size=args.axis_size)
        geometries.extend(axes)

        # Camera label
        label_pos = T[:3, 3] + T[:3, :3] @ np.array([0, 0, -args.axis_size * 1.3])
        labels_3d.append((cam_names.get(ci, f"cam{ci}"), label_pos, color))

        # Label sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(T[:3, 3])
        sphere.paint_uniform_color(color)
        geometries.append(sphere)

        t = T[:3, 3]
        print(f"[VIS] {cam_names.get(ci, f'cam{ci}')}: [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm")

    # ─── 4. Gripper TCP + camera at each capture ───
    gripper_positions = []

    for cap in meta["captures"]:
        rp = cap.get("robot_pose_6dof")
        if rp is None:
            continue

        T_bTg = euler_deg_to_matrix(*rp)
        gripper_pos = T_bTg[:3, 3]
        gripper_positions.append(gripper_pos.tolist())

        # Gripper TCP axes (smaller)
        axes = create_axis_lines(T_bTg, size=args.axis_size * 0.4)
        geometries.extend(axes)

        # Gripper camera
        if T_gTc is not None:
            T_base_cam2 = T_bTg @ T_gTc
            axes_gc = create_axis_lines(T_base_cam2, size=args.axis_size * 0.3)
            geometries.extend(axes_gc)

            # Small green sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sphere.translate(T_base_cam2[:3, 3])
            sphere.paint_uniform_color([0.2, 0.9, 0.2])
            geometries.append(sphere)

    # Gripper trajectory
    if len(gripper_positions) >= 2:
        traj = create_trajectory_line(gripper_positions, color=[0.5, 0.5, 0.5])
        if traj:
            geometries.append(traj)

    print(f"[VIS] Gripper poses: {len(gripper_positions)}")

    # ─── 5. Cube positions ───
    cube_count = 0
    K_map_full, D_map_full = {}, {}
    for ci in all_cam_ids:
        K_map_full[ci], D_map_full[ci] = load_intrinsics(intr_dir, ci)

    for cap in meta["captures"]:
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
                img, K_map_full[ci], D_map_full[ci],
                min_markers=1, reproj_thr_mean_px=10.0)
            if not ok:
                continue

            T_cam_O = rodrigues_to_Rt(rvec, tvec)
            T_base_cam = transforms.get(f"T_base_C{ci}")
            if T_base_cam is not None:
                T_base_O = T_base_cam @ T_cam_O
                cube_wf = create_cube_wireframe(T_base_O, cfg.cube_side_m, color=[1, 1, 0])
                geometries.append(cube_wf)
                cube_count += 1
                break

    print(f"[VIS] Cube positions: {cube_count}")

    # ─── 6. Create label annotations ───
    for text, pos, color in labels_3d:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        geometries.append(sphere)

    # ─── Legend ───
    print("\n=== Legend ===")
    print("  Large RGB axes  : Robot base (origin)")
    print("  Red axes+sphere : cam0 (fixed, right)")
    print("  Blue axes+sphere: cam1 (fixed, front)")
    print("  Orange axes+sph : cam3 (fixed, left)")
    print("  Small RGB axes  : Gripper TCP at each capture")
    print("  Small green axes: Gripper camera (cam2) at each capture")
    print("  Yellow wireframe: Cube positions")
    print("  Gray grid       : XY ground plane")
    print("  Gray line       : Gripper trajectory")
    print()
    print("  Axis colors: Red=X, Green=Y, Blue=Z")

    # ─── Visualize ───
    print("\n[VIS] Opening viewer...")
    print("  Mouse: rotate / Scroll: zoom / Shift+mouse: pan")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Calibration - Robot Base Frame", width=1400, height=900)

    for g in geometries:
        vis.add_geometry(g)

    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])
    opt.line_width = 3.0
    opt.point_size = 5.0

    # Set viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.5, 0.7])
    ctr.set_lookat([0, 0.3, 0])
    ctr.set_up([0, 0, 1])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
