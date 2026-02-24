# Step4_verify_and_fuse.py
"""
Step 4: Verify calibration quality and optionally fuse depth point clouds.

Verification:
  - Reprojection consistency across cameras
  - Hand-eye verification (bTg @ gTc ≈ bTt @ inv(tTc))
  - ICP alignment quality between camera point clouds

Optional: Fuse depth from all cameras into a single point cloud in robot base frame.

Usage:
  python Step4_verify_and_fuse.py \
    --root_folder ./data/session_01 \
    --intrinsics_dir ./intrinsics \
    --frame_idx 0 \
    --save_ply --eval_icp
"""

import os
import json
import argparse
from typing import Dict, Optional

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt, inv_T
from config import CubeConfig
from robot_comm import euler_deg_to_matrix


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64)
    ds = float(d["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in d else 0.001
    if np.isnan(ds):
        ds = 0.001
    return K, D, ds


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


def compute_pose_error(T1, T2):
    pos_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    R_err = T1[:3, :3].T @ T2[:3, :3]
    ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
    return pos_err, np.rad2deg(ang)


def main():
    parser = argparse.ArgumentParser(description="Verify calibration and fuse point clouds")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--calib_dir", type=str, default=None, help="Default: <root>/calib_out")

    parser.add_argument("--frame_idx", type=int, default=0, help="Frame to use for point cloud fusion")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--z_min", type=float, default=0.2)
    parser.add_argument("--z_max", type=float, default=1.5)

    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--eval_icp", action="store_true")
    parser.add_argument("--icp_dist", type=float, default=0.02)
    parser.add_argument("--visualize", action="store_true", help="Open3D visualization")

    parser.add_argument("--coord_frame", type=str, default="base",
                        choices=["base", "ref_cam"],
                        help="Coordinate frame for fusion: 'base' (robot) or 'ref_cam'")

    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    calib_dir = args.calib_dir or os.path.join(root, "calib_out")

    # Load calibration summary
    summary_path = os.path.join(calib_dir, "calibration_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    gripper_cam_idx = summary.get("gripper_cam_idx")
    ref_fixed = summary.get("ref_fixed_cam_idx")
    fixed_cam_ids = summary.get("fixed_cam_ids", [])
    all_cam_ids = summary.get("all_cam_ids", [])

    print(f"[INFO] Gripper cam: cam{gripper_cam_idx}")
    print(f"[INFO] Ref fixed cam: cam{ref_fixed}")
    print(f"[INFO] Fixed cams: {fixed_cam_ids}")

    # Load transforms
    transforms = {}
    for key, flat in summary["transforms"].items():
        transforms[key] = np.array(flat, dtype=np.float64).reshape(4, 4)

    T_gTc = transforms.get("T_gripper_cam")
    if T_gTc is not None:
        print(f"\n[INFO] Hand-eye transform (T_gripper_cam):")
        t = T_gTc[:3, 3]
        print(f"  Translation: [{t[0]*100:.2f}, {t[1]*100:.2f}, {t[2]*100:.2f}] cm")
        print(f"  Distance: {np.linalg.norm(t)*100:.2f} cm")

    # ─── Verification: cube reprojection consistency ───
    print(f"\n{'='*60}")
    print("Verification: Cross-camera cube consistency")
    print(f"{'='*60}")

    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    K_map, D_map, ds_map = {}, {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], ds_map[ci] = load_intrinsics(intr_dir, ci)

    # For each frame, compute cube position in base frame from each camera
    # They should agree
    n_checked = 0
    pos_diffs = []

    for cap in meta["captures"][:20]:  # Check first 20 frames
        eid = cap["event_id"]
        cube_positions_base = {}

        for ci_str, cinfo in cap["cams"].items():
            ci = int(ci_str)
            if not cinfo.get("saved"):
                continue

            rgb_path = os.path.join(root, cinfo["rgb_path"])
            img = cv2.imread(rgb_path)
            if img is None:
                continue

            ok, rvec, tvec, used = cube.solve_pnp_cube(
                img, K_map[ci], D_map[ci], min_markers=1, reproj_thr_mean_px=10.0)
            if not ok:
                continue

            T_cam_O = rodrigues_to_Rt(rvec, tvec)

            # Transform to base frame
            T_base_cam = None
            if ci == gripper_cam_idx:
                rp = cap.get("robot_pose_6dof")
                if rp and T_gTc is not None:
                    T_bTg = euler_deg_to_matrix(*rp)
                    T_base_cam = T_bTg @ T_gTc
            else:
                key = f"T_base_C{ci}"
                T_base_cam = transforms.get(key)

            if T_base_cam is not None:
                T_base_O = T_base_cam @ T_cam_O
                cube_positions_base[ci] = T_base_O[:3, 3]

        if len(cube_positions_base) >= 2:
            positions = list(cube_positions_base.values())
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    diff = np.linalg.norm(positions[i] - positions[j]) * 1000  # mm
                    pos_diffs.append(diff)
            n_checked += 1

    if pos_diffs:
        print(f"  Checked {n_checked} frames")
        print(f"  Cube position agreement (pairwise):")
        print(f"    Mean: {np.mean(pos_diffs):.2f} mm")
        print(f"    Median: {np.median(pos_diffs):.2f} mm")
        print(f"    Max: {np.max(pos_diffs):.2f} mm")
        print(f"    Std: {np.std(pos_diffs):.2f} mm")
    else:
        print("  [WARN] Not enough data for cross-camera verification")

    # ─── Hand-eye verification ───
    if T_gTc is not None and len([c for c in meta["captures"] if c.get("robot_pose_6dof")]) > 0:
        print(f"\n{'='*60}")
        print("Verification: Hand-eye (bTg @ gTc consistency)")
        print(f"{'='*60}")

        gripper_frames = {}
        robot_Ts = {}

        for cap in meta["captures"]:
            eid = cap["event_id"]
            rp = cap.get("robot_pose_6dof")
            if rp is None:
                continue
            robot_Ts[eid] = euler_deg_to_matrix(*rp)

            ci_str = str(gripper_cam_idx)
            cinfo = cap["cams"].get(ci_str)
            if cinfo and cinfo.get("saved"):
                rgb_path = os.path.join(root, cinfo["rgb_path"])
                img = cv2.imread(rgb_path)
                if img is not None:
                    ok, rvec, tvec, _ = cube.solve_pnp_cube(
                        img, K_map[gripper_cam_idx], D_map[gripper_cam_idx],
                        min_markers=1, reproj_thr_mean_px=10.0)
                    if ok:
                        gripper_frames[eid] = rodrigues_to_Rt(rvec, tvec)

        common = sorted(set(robot_Ts.keys()) & set(gripper_frames.keys()))
        if len(common) >= 2:
            rTs = [robot_Ts[eid] for eid in common]
            cTs = [gripper_frames[eid] for eid in common]

            # Average board pose in base
            T_bTt_list = [(rTs[i] @ T_gTc) @ cTs[i] for i in range(len(rTs))]
            from utils_pose import robust_se3_average
            T_bTt = robust_se3_average(T_bTt_list)

            pos_errs, rot_errs = [], []
            for i in range(len(rTs)):
                T_bTc_est = rTs[i] @ T_gTc
                T_bTc_gt = T_bTt @ np.linalg.inv(cTs[i])
                pe, re = compute_pose_error(T_bTc_est, T_bTc_gt)
                pos_errs.append(pe * 1000)
                rot_errs.append(re)

            print(f"  Frames: {len(common)}")
            print(f"  Position error: mean={np.mean(pos_errs):.2f}mm, max={np.max(pos_errs):.2f}mm")
            print(f"  Rotation error: mean={np.mean(rot_errs):.2f}°, max={np.max(rot_errs):.2f}°")

    # ─── Point cloud fusion ───
    if args.save_ply or args.eval_icp or args.visualize:
        try:
            import open3d as o3d
        except ImportError:
            print("[ERROR] open3d not installed. Skip point cloud fusion.")
            print("  Install: pip install open3d")
            return

        print(f"\n{'='*60}")
        print(f"Point cloud fusion (frame={args.frame_idx}, coord={args.coord_frame})")
        print(f"{'='*60}")

        per_cam_pcd = {}
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

            pts_cam, pix = depth_to_points(depth_u16, K_map[ci], ds_map[ci],
                                           args.z_min, args.z_max, args.stride)
            if pts_cam.shape[0] == 0:
                continue

            # Transform to chosen frame
            if args.coord_frame == "base":
                key = f"T_base_C{ci}"
                T = transforms.get(key)
                if T is None:
                    print(f"[WARN] cam{ci}: no T_base_C{ci}, skipping")
                    continue
            else:  # ref_cam
                if ci == ref_fixed:
                    T = np.eye(4, dtype=np.float64)
                else:
                    key = f"T_C{ref_fixed}_C{ci}"
                    # T_Cref_Ci transforms points FROM Ci TO Cref
                    # pts are in Ci frame, so: pts_ref = T_Cref_Ci @ pts_Ci
                    # Actually T_Cref_Ci is the pose of Ci in Cref frame
                    # So pts_ref = T_Cref_Ci @ pts_cam (homogeneous)
                    T = transforms.get(key)
                    if T is None:
                        continue

            R = T[:3, :3]
            t = T[:3, 3]
            pts_out = pts_cam @ R.T + t.reshape(1, 3)
            cols = np.array([rgb[v, u] for (v, u) in pix], dtype=np.float64)

            all_pts.append(pts_out)
            all_cols.append(cols)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_out)
            pcd.colors = o3d.utility.Vector3dVector(cols)
            per_cam_pcd[ci] = pcd

            print(f"  cam{ci}: {pts_out.shape[0]} points")

        if len(all_pts) == 0:
            print("[WARN] No points collected for fusion.")
            return

        P = np.concatenate(all_pts)
        C = np.concatenate(all_cols)
        fused = o3d.geometry.PointCloud()
        fused.points = o3d.utility.Vector3dVector(P)
        fused.colors = o3d.utility.Vector3dVector(C)

        if args.save_ply:
            out_ply = os.path.join(root, f"fused_{args.coord_frame}_frame{args.frame_idx:05d}.ply")
            o3d.io.write_point_cloud(out_ply, fused)
            print(f"[SAVE] {out_ply}")

        if args.eval_icp and ref_fixed in per_cam_pcd:
            ref_pcd = per_cam_pcd[ref_fixed].voxel_down_sample(0.005)
            for ci, pcd in per_cam_pcd.items():
                if ci == ref_fixed:
                    continue
                src_down = pcd.voxel_down_sample(0.005)
                reg = o3d.pipelines.registration.registration_icp(
                    src_down, ref_pcd, args.icp_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                print(f"  ICP cam{ci}->cam{ref_fixed}: fitness={reg.fitness:.4f} rmse={reg.inlier_rmse:.6f}")

        if args.visualize:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([axis, fused])

    print("\n[DONE] Step4 verification complete.")


if __name__ == "__main__":
    main()
