# Step5_pointcloud.py
"""
Step 5: 멀티카메라 포인트클라우드 정합, 시각화, 검증.

캘리브레이션 결과를 이용하여:
  1. 각 카메라의 뎁스 이미지 -> 컬러 포인트클라우드 생성
  2. T_base_Ci 변환으로 로봇 좌표계에 정합
  3. 멀티뷰 포인트클라우드 합성 + 시각화
  4. 정합 품질 검증 (카메라 간 겹치는 영역 비교)
  5. PLY/OBJ 내보내기 (시뮬레이션용)

실행 (저장된 이미지 사용):
  python Step5_pointcloud.py \
    --root_folder ./data/session \
    --calib_dir ./data/session/calib_out \
    --intrinsics_dir ./intrinsics \
    --event_id 0

실행 (라이브 카메라):
  python Step5_pointcloud.py \
    --calib_dir ./data/session/calib_out \
    --intrinsics_dir ./intrinsics \
    --live
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from robot_comm import euler_deg_to_matrix


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64)
    return K, D


def load_calib(calib_dir: str) -> Dict[str, np.ndarray]:
    transforms = {}
    for f in os.listdir(calib_dir):
        if f.endswith(".npy"):
            transforms[f.replace(".npy", "")] = np.load(os.path.join(calib_dir, f))
    return transforms


def depth_to_pointcloud(color_bgr, depth_u16, K, z_min=0.1, z_max=1.5, stride=2):
    """Convert aligned color + depth images to colored point cloud.

    Args:
        color_bgr: (H, W, 3) uint8 BGR image
        depth_u16: (H, W) uint16 depth in mm
        K: (3, 3) camera intrinsic matrix
        z_min, z_max: depth range in meters
        stride: downsample factor (1=full, 2=half, etc.)

    Returns:
        points: (N, 3) float64 in meters
        colors: (N, 3) float64 in [0, 1] RGB
    """
    h, w = depth_u16.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel grid
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us = us.astype(np.float64)
    vs = vs.astype(np.float64)

    # Depth in meters
    z = depth_u16[0:h:stride, 0:w:stride].astype(np.float64) / 1000.0

    # Filter by depth range
    mask = (z > z_min) & (z < z_max)

    z_valid = z[mask]
    u_valid = us[mask]
    v_valid = vs[mask]

    # Back-project to 3D
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    points = np.stack([x, y, z_valid], axis=1)

    # Colors (BGR -> RGB, normalized)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    colors = color_rgb[0:h:stride, 0:w:stride][mask].astype(np.float64) / 255.0

    return points, colors


def create_o3d_pointcloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_camera_lineset(T, scale=0.05, color=(0, 0, 1)):
    """Create camera frustum visualization."""
    s = scale
    # Camera corners in camera frame
    pts_cam = np.array([
        [0, 0, 0],
        [-s, -s * 0.75, s * 1.5],
        [s, -s * 0.75, s * 1.5],
        [s, s * 0.75, s * 1.5],
        [-s, s * 0.75, s * 1.5],
    ], dtype=np.float64)

    R = T[:3, :3]
    t = T[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors_ls = [color for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors_ls)
    return ls


def create_coord_frame(T, size=0.05):
    """Create coordinate frame at transform T."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


# ══════════════════════════════════════════════════════════════
# Registration verification
# ══════════════════════════════════════════════════════════════

def verify_registration(pcds_base: Dict[int, o3d.geometry.PointCloud], voxel_size=0.005):
    """Verify registration quality by checking overlap between camera point clouds."""
    print("\n" + "=" * 60)
    print("[VERIFY] Registration quality")
    print("=" * 60)

    cam_ids = sorted(pcds_base.keys())
    if len(cam_ids) < 2:
        print("  Need 2+ cameras for verification")
        return

    for i in range(len(cam_ids)):
        for j in range(i + 1, len(cam_ids)):
            ci, cj = cam_ids[i], cam_ids[j]
            pcd_i = pcds_base[ci]
            pcd_j = pcds_base[cj]

            if len(pcd_i.points) == 0 or len(pcd_j.points) == 0:
                continue

            # Downsample for faster computation
            pcd_i_ds = pcd_i.voxel_down_sample(voxel_size)
            pcd_j_ds = pcd_j.voxel_down_sample(voxel_size)

            # Compute distances from each point in i to nearest in j
            dists = np.asarray(pcd_i_ds.compute_point_cloud_distance(pcd_j_ds))

            overlap_1cm = float(np.sum(dists < 0.01)) / max(len(dists), 1) * 100
            overlap_2cm = float(np.sum(dists < 0.02)) / max(len(dists), 1) * 100

            print(f"  cam{ci} vs cam{cj}:")
            print(f"    points: {len(pcd_i_ds.points)} vs {len(pcd_j_ds.points)}")
            print(f"    mean dist: {np.mean(dists)*1000:.2f}mm")
            print(f"    median:    {np.median(dists)*1000:.2f}mm")
            print(f"    overlap <10mm: {overlap_1cm:.1f}%")
            print(f"    overlap <20mm: {overlap_2cm:.1f}%")

            if overlap_1cm > 30:
                # Try ICP refinement
                result = o3d.pipelines.registration.registration_icp(
                    pcd_i_ds, pcd_j_ds,
                    max_correspondence_distance=0.02,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                print(f"    ICP fitness: {result.fitness:.3f}")
                print(f"    ICP RMSE:    {result.inlier_rmse*1000:.2f}mm")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-camera point cloud fusion & verification")
    parser.add_argument("--root_folder", type=str, default=None,
                        help="Capture session folder (for saved images)")
    parser.add_argument("--calib_dir", required=True,
                        help="Calibration output directory")
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--event_id", type=int, default=None,
                        help="Specific event to visualize (default: latest)")
    parser.add_argument("--live", action="store_true",
                        help="Live capture from cameras")

    parser.add_argument("--z_min", type=float, default=0.15, help="Min depth (m)")
    parser.add_argument("--z_max", type=float, default=1.2, help="Max depth (m)")
    parser.add_argument("--stride", type=int, default=2, help="Downsample stride")
    parser.add_argument("--voxel_size", type=float, default=0.003,
                        help="Voxel size for downsampling (m)")

    parser.add_argument("--export_ply", type=str, default=None,
                        help="Export fused point cloud to PLY file")
    parser.add_argument("--export_mesh", type=str, default=None,
                        help="Export reconstructed mesh to OBJ file")

    args = parser.parse_args()

    # Load calibration
    transforms = load_calib(args.calib_dir)
    print(f"[INFO] Calibration transforms: {list(transforms.keys())}")

    # Load summary for camera info
    summary_path = os.path.join(args.calib_dir, "calibration_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    gripper_cam_idx = summary.get("gripper_cam_idx")
    all_cam_ids = summary.get("all_cam_ids", [])
    if not all_cam_ids:
        all_cam_ids = sorted([int(k.split("C")[1]) for k in transforms if k.startswith("T_base_C")])

    print(f"[INFO] Cameras: {all_cam_ids}, gripper=cam{gripper_cam_idx}")

    # ─── Load or capture depth + color images ───
    frames: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # ci -> (color_bgr, depth_u16)

    if args.live:
        # Live capture
        from camera import RealSenseCamera

        dm_path = os.path.join(args.intrinsics_dir, "device_map.json")
        with open(dm_path, "r") as f:
            dm = json.load(f)
        serial_to_idx = dm.get("serial_to_idx", {})

        devs = RealSenseCamera.list_devices()
        cams = {}
        for serial, idx_str in serial_to_idx.items():
            if serial in devs:
                ci = int(idx_str)
                cam = RealSenseCamera(
                    serial=serial, width=640, height=480, fps=15,
                    use_color=True, use_depth=True,
                    align_depth_to_color=True, warmup_frames=15,
                )
                cam.start()
                cams[ci] = cam
                print(f"[INFO] cam{ci} started ({serial})")

        import time
        time.sleep(1.0)  # settle

        for ci, cam in cams.items():
            color, depth, _ = cam.get_latest()
            if color is not None and depth is not None:
                frames[ci] = (color, depth)
                print(f"[INFO] cam{ci}: captured {color.shape} + {depth.shape}")

        for cam in cams.values():
            cam.stop()

    elif args.root_folder:
        # Load from saved session
        meta_path = os.path.join(args.root_folder, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Find target event
        captures = meta.get("captures", [])
        if not captures:
            raise RuntimeError("No captures in meta.json")

        if args.event_id is not None:
            cap = next((c for c in captures if c.get("event_id") == args.event_id), None)
            if cap is None:
                raise RuntimeError(f"Event {args.event_id} not found")
        else:
            cap = captures[-1]  # latest

        eid = cap.get("event_id", 0)
        print(f"[INFO] Using event_id={eid}")

        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            rgb_path = os.path.join(args.root_folder, cinfo.get("rgb_path", ""))
            depth_path = cinfo.get("depth_path")

            if depth_path is None:
                print(f"[WARN] cam{ci}: no depth image saved (use --save_depth in Step2)")
                continue

            depth_path = os.path.join(args.root_folder, depth_path)
            color = cv2.imread(rgb_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if color is not None and depth is not None:
                frames[ci] = (color, depth)
                print(f"[INFO] cam{ci}: loaded {color.shape} + {depth.shape}")
            else:
                print(f"[WARN] cam{ci}: failed to load images")
    else:
        raise RuntimeError("Specify --root_folder (saved) or --live (camera)")

    if not frames:
        raise RuntimeError("No frames loaded")

    # ─── Create point clouds per camera ───
    print("\n" + "=" * 60)
    print("[STEP 1] Per-camera point clouds")
    print("=" * 60)

    pcds_cam: Dict[int, o3d.geometry.PointCloud] = {}
    pcds_base: Dict[int, o3d.geometry.PointCloud] = {}

    cam_colors = {0: (0.2, 0.4, 1.0), 1: (0.2, 0.8, 0.2),
                  2: (1.0, 0.3, 0.3), 3: (1.0, 0.7, 0.1)}

    for ci, (color, depth) in frames.items():
        K, D = load_intrinsics(args.intrinsics_dir, ci)

        points, colors = depth_to_pointcloud(
            color, depth, K,
            z_min=args.z_min, z_max=args.z_max, stride=args.stride,
        )

        if len(points) == 0:
            print(f"  cam{ci}: 0 points (check depth range)")
            continue

        pcd = create_o3d_pointcloud(points, colors)

        # Voxel downsample
        pcd = pcd.voxel_down_sample(args.voxel_size)
        pcds_cam[ci] = pcd
        print(f"  cam{ci}: {len(pcd.points)} points")

        # Transform to robot base frame
        T_key = f"T_base_C{ci}"
        if T_key in transforms:
            pcd_base = o3d.geometry.PointCloud(pcd)
            pcd_base.transform(transforms[T_key])
            pcds_base[ci] = pcd_base
        else:
            print(f"  cam{ci}: no {T_key}, skipping base transform")

    # ─── Fuse all point clouds ───
    print("\n" + "=" * 60)
    print("[STEP 2] Fusion in robot base frame")
    print("=" * 60)

    fused = o3d.geometry.PointCloud()
    for ci, pcd in pcds_base.items():
        fused += pcd

    if len(fused.points) > 0:
        fused = fused.voxel_down_sample(args.voxel_size)
        print(f"  Fused: {len(fused.points)} points")

        # Remove statistical outliers
        fused, inlier_idx = fused.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  After outlier removal: {len(fused.points)} points")
    else:
        print("  [WARN] No points in fused cloud")

    # ─── Verify registration ───
    verify_registration(pcds_base, voxel_size=args.voxel_size)

    # ─── Export ───
    if args.export_ply and len(fused.points) > 0:
        o3d.io.write_point_cloud(args.export_ply, fused)
        print(f"\n[EXPORT] Point cloud: {args.export_ply} ({len(fused.points)} points)")

    if args.export_mesh and len(fused.points) > 1000:
        print(f"\n[MESH] Reconstructing mesh...")
        fused.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01, max_nn=30))
        fused.orient_normals_consistent_tangent_plane(k=15)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            fused, depth=9)
        # Remove low-density vertices (noise)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        o3d.io.write_triangle_mesh(args.export_mesh, mesh)
        print(f"[EXPORT] Mesh: {args.export_mesh} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} faces)")

    # ─── Visualize ───
    print("\n" + "=" * 60)
    print("[VIS] 3D Visualization (Open3D)")
    print("=" * 60)
    print("  Controls: mouse drag=rotate, scroll=zoom, shift+drag=pan")
    print("  Press Q or close window to exit")

    vis_geoms = []

    # Robot base frame
    vis_geoms.append(create_coord_frame(np.eye(4), size=0.1))

    # Camera positions + frustums
    cam_color_map = {0: (0, 0, 1), 1: (0, 0.7, 0), 2: (1, 0, 0), 3: (1, 0.5, 0)}
    for ci in all_cam_ids:
        T_key = f"T_base_C{ci}"
        if T_key in transforms:
            T = transforms[T_key]
            vis_geoms.append(create_coord_frame(T, size=0.03))
            color = cam_color_map.get(ci, (0.5, 0.5, 0.5))
            vis_geoms.append(create_camera_lineset(T, scale=0.03, color=color))

    # Fused point cloud
    if len(fused.points) > 0:
        vis_geoms.append(fused)

    # Per-camera point clouds (colored by camera for verification)
    # Uncomment below to show per-camera clouds instead of fused:
    # for ci, pcd in pcds_base.items():
    #     vis_geoms.append(pcd)

    o3d.visualization.draw_geometries(
        vis_geoms,
        window_name="Multi-Camera Point Cloud (Robot Base Frame)",
        width=1280, height=720,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
