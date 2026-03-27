# Step4_verify.py
"""
Step 4: 캘리브레이션 정확도 검증 및 3D 시각화.

검증 항목:
  1. 교차 카메라 일관성: 같은 큐브를 여러 카메라로 봤을 때 위치 차이
  2. 재투영 오차: 큐브 마커를 이미지에 역투영하여 검출 결과와 비교
  3. Hand-eye 일관성: 보드 위치가 모든 프레임에서 일정한지
  4. 3D 시각화: 카메라, 큐브, 로봇 위치를 3D로 표시

실행:
  python Step4_verify.py \
    --root_folder ./data/session \
    --calib_dir ./data/session/calib_out \
    --intrinsics_dir ./intrinsics
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt, inv_T
from config import CubeConfig
from robot_comm import euler_deg_to_matrix


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def rotation_error_deg(Ra, Rb):
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def load_calib(calib_dir: str):
    """Load all calibration .npy files from directory."""
    transforms = {}
    for f in os.listdir(calib_dir):
        if f.endswith(".npy"):
            name = f.replace(".npy", "")
            transforms[name] = np.load(os.path.join(calib_dir, f))
    return transforms


def draw_frame(ax, T, label="", scale=30.0, lw=1.5):
    """Draw a coordinate frame (RGB = XYZ) at transform T."""
    o = T[:3, 3] * 1000.0  # m -> mm
    R = T[:3, :3]
    colors = ['r', 'g', 'b']
    for i, c in enumerate(colors):
        d = R[:, i] * scale
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                  color=c, linewidth=lw, arrow_length_ratio=0.15)
    if label:
        ax.text(o[0], o[1], o[2], f"  {label}", fontsize=7)


def draw_camera(ax, T, label="", scale=20.0, color='blue'):
    """Draw camera as a pyramid frustum."""
    o = T[:3, 3] * 1000.0
    R = T[:3, :3]

    # Camera frustum corners (in camera frame, pointing +Z)
    s = scale
    corners_cam = np.array([
        [-s, -s*0.75, s*1.5],
        [ s, -s*0.75, s*1.5],
        [ s,  s*0.75, s*1.5],
        [-s,  s*0.75, s*1.5],
    ], dtype=np.float64)

    corners_world = (R @ corners_cam.T).T + o
    # Draw frustum lines
    for c in corners_world:
        ax.plot3D([o[0], c[0]], [o[1], c[1]], [o[2], c[2]],
                  color=color, linewidth=0.8, alpha=0.6)
    # Draw rectangle
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D([corners_world[i, 0], corners_world[j, 0]],
                  [corners_world[i, 1], corners_world[j, 1]],
                  [corners_world[i, 2], corners_world[j, 2]],
                  color=color, linewidth=0.8, alpha=0.6)
    if label:
        ax.text(o[0], o[1], o[2], f"  {label}", fontsize=7, color=color)


# ══════════════════════════════════════════════════════════════
# Verification tests
# ══════════════════════════════════════════════════════════════

def test_cross_camera_consistency(meta, transforms, all_cam_ids, gripper_cam_idx):
    """Test: same cube seen from different cameras -> same position in base frame."""
    print("\n" + "=" * 60)
    print("[TEST 1] Cross-camera consistency")
    print("=" * 60)

    errors_mm = []
    n_events = 0

    for cap in meta.get("captures", []):
        eid = cap.get("event_id", -1)

        # Collect cube positions in base frame from each camera
        positions_base = []
        cam_labels = []

        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            cpnp = cinfo.get("cube_pnp")
            if not cpnp or not cpnp.get("ok"):
                continue

            T_cam_cube = np.asarray(cpnp["T_cam_cube_4x4"], dtype=np.float64)

            # Get T_base_cam for this camera
            T_base_cam_key = f"T_base_C{ci}"
            if T_base_cam_key not in transforms:
                continue

            T_base_cam = transforms[T_base_cam_key]
            T_base_cube = T_base_cam @ T_cam_cube
            positions_base.append(T_base_cube[:3, 3] * 1000.0)  # mm
            cam_labels.append(ci)

        if len(positions_base) < 2:
            continue

        n_events += 1
        positions = np.array(positions_base)
        mean_pos = positions.mean(axis=0)

        for i, (pos, ci) in enumerate(zip(positions, cam_labels)):
            err = np.linalg.norm(pos - mean_pos)
            errors_mm.append(err)

    if errors_mm:
        print(f"  Events with 2+ cameras: {n_events}")
        print(f"  Position error (vs mean):")
        print(f"    mean:   {np.mean(errors_mm):.2f} mm")
        print(f"    median: {np.median(errors_mm):.2f} mm")
        print(f"    max:    {np.max(errors_mm):.2f} mm")
        print(f"    std:    {np.std(errors_mm):.2f} mm")
        ok = np.mean(errors_mm) < 5.0
        print(f"  Result: {'PASS' if ok else 'FAIL'} (threshold: 5mm)")
    else:
        print("  [SKIP] Not enough multi-camera observations")
        ok = None

    return errors_mm


def test_reprojection(meta, transforms, intrinsics_dir, all_cam_ids, root_folder):
    """Test: project cube model through calibrated transforms back onto images."""
    print("\n" + "=" * 60)
    print("[TEST 2] Reprojection verification")
    print("=" * 60)

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)
    errors_px = []

    for cap in meta.get("captures", []):
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            cpnp = cinfo.get("cube_pnp")
            if not cpnp or not cpnp.get("ok"):
                continue
            err = cpnp.get("reproj_mean_px")
            if err is not None:
                errors_px.append(float(err))

    if errors_px:
        print(f"  Total observations: {len(errors_px)}")
        print(f"  Reprojection error (from metadata):")
        print(f"    mean:   {np.mean(errors_px):.3f} px")
        print(f"    median: {np.median(errors_px):.3f} px")
        print(f"    max:    {np.max(errors_px):.3f} px")
        print(f"    <1px:   {sum(1 for e in errors_px if e < 1.0)}/{len(errors_px)}")
        ok = np.mean(errors_px) < 2.0
        print(f"  Result: {'PASS' if ok else 'FAIL'} (threshold: 2px)")
    else:
        print("  [SKIP] No reprojection data")

    return errors_px


def test_handeye_consistency(meta, transforms, gripper_cam_idx):
    """Test: T_base_board should be constant (board is fixed)."""
    print("\n" + "=" * 60)
    print("[TEST 3] Hand-eye consistency (board stability)")
    print("=" * 60)

    T_gTc = transforms.get("T_gripper_cam")
    if T_gTc is None:
        print("  [SKIP] T_gripper_cam not found")
        return []

    T_base_board_list = []
    for cap in meta.get("captures", []):
        # Need robot pose
        T_B_G = None
        if "robot_pose_matrix_4x4" in cap:
            try:
                T_B_G = np.asarray(cap["robot_pose_matrix_4x4"], dtype=np.float64)
            except Exception:
                pass
        if T_B_G is None and "robot_pose_6dof" in cap:
            try:
                T_B_G = euler_deg_to_matrix(*cap["robot_pose_6dof"])
            except Exception:
                pass
        if T_B_G is None:
            continue

        # Need charuco from gripper camera
        gi_data = cap.get("cams", {}).get(str(gripper_cam_idx), {})
        ch = gi_data.get("charuco")
        if not ch or not ch.get("ok"):
            continue

        T_cam_board = np.asarray(ch["T_cam_board_4x4"], dtype=np.float64)
        T_base_board = T_B_G @ T_gTc @ T_cam_board
        T_base_board_list.append(T_base_board)

    if len(T_base_board_list) < 2:
        print("  [SKIP] Not enough ChArUco observations")
        return []

    # Compute consistency
    positions = np.array([T[:3, 3] * 1000.0 for T in T_base_board_list])
    mean_pos = positions.mean(axis=0)

    pos_errors = [np.linalg.norm(p - mean_pos) for p in positions]
    rot_errors = [rotation_error_deg(T[:3, :3], T_base_board_list[0][:3, :3])
                  for T in T_base_board_list]

    print(f"  Frames: {len(T_base_board_list)}")
    print(f"  Board position stability:")
    print(f"    std: {np.std(pos_errors):.2f} mm")
    print(f"    max: {np.max(pos_errors):.2f} mm")
    print(f"  Board rotation stability:")
    print(f"    mean: {np.mean(rot_errors):.3f} deg")
    print(f"    max:  {np.max(rot_errors):.3f} deg")
    ok = np.std(pos_errors) < 3.0 and np.mean(rot_errors) < 1.0
    print(f"  Result: {'PASS' if ok else 'FAIL'} (pos<3mm, rot<1deg)")

    return pos_errors


# ══════════════════════════════════════════════════════════════
# 3D Visualization
# ══════════════════════════════════════════════════════════════

def visualize_3d(meta, transforms, gripper_cam_idx, all_cam_ids):
    """3D plot of robot base, cameras, cube positions, gripper poses."""
    print("\n" + "=" * 60)
    print("[VIS] 3D Visualization")
    print("=" * 60)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Robot base (origin)
    T_origin = np.eye(4)
    draw_frame(ax, T_origin, label="Robot Base", scale=40.0, lw=2.5)

    # 2. Fixed cameras
    cam_colors = {0: 'blue', 1: 'green', 3: 'orange'}
    for ci in all_cam_ids:
        key = f"T_base_C{ci}"
        if key not in transforms:
            continue
        T = transforms[key]
        tag = "Gripper" if ci == gripper_cam_idx else "Fixed"
        color = cam_colors.get(ci, 'purple')
        if ci == gripper_cam_idx:
            color = 'red'
        draw_camera(ax, T, label=f"cam{ci} ({tag})", color=color)
        draw_frame(ax, T, scale=20.0, lw=1.0)

    # 3. Cube positions per event
    T_gTc = transforms.get("T_gripper_cam")
    cube_positions = []

    for cap in meta.get("captures", []):
        eid = cap.get("event_id", -1)

        # From fixed cameras
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            cpnp = cinfo.get("cube_pnp")
            if not cpnp or not cpnp.get("ok"):
                continue
            key = f"T_base_C{ci}"
            if key not in transforms:
                continue
            T_cam_cube = np.asarray(cpnp["T_cam_cube_4x4"], dtype=np.float64)
            T_base_cube = transforms[key] @ T_cam_cube
            cube_positions.append(T_base_cube[:3, 3] * 1000.0)
            break  # one per event is enough

    if cube_positions:
        pts = np.array(cube_positions)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='cyan', s=30, marker='s', alpha=0.7, label='Cube positions')

    # 4. Gripper poses per event
    gripper_positions = []
    for cap in meta.get("captures", []):
        T_B_G = None
        if "robot_pose_matrix_4x4" in cap:
            try:
                T_B_G = np.asarray(cap["robot_pose_matrix_4x4"], dtype=np.float64)
            except Exception:
                pass
        if T_B_G is None and "robot_pose_6dof" in cap:
            try:
                T_B_G = euler_deg_to_matrix(*cap["robot_pose_6dof"])
            except Exception:
                pass
        if T_B_G is not None:
            gripper_positions.append(T_B_G[:3, 3] * 1000.0)

    if gripper_positions:
        pts = np.array(gripper_positions)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='red', s=15, marker='^', alpha=0.5, label='Gripper poses')

    # 5. Board position (average)
    T_base_O = transforms.get("T_base_O")
    if T_base_O is not None:
        draw_frame(ax, T_base_O, label="Cube (avg)", scale=25.0, lw=2.0)

    # Formatting
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Calibration Result - Robot Base Frame")
    ax.legend(loc='upper left', fontsize=8)

    # Equal aspect ratio
    all_pts = []
    if cube_positions:
        all_pts.extend(cube_positions)
    if gripper_positions:
        all_pts.extend(gripper_positions)
    for ci in all_cam_ids:
        key = f"T_base_C{ci}"
        if key in transforms:
            all_pts.append(transforms[key][:3, 3] * 1000.0)
    all_pts.append(np.zeros(3))

    if all_pts:
        pts = np.array(all_pts)
        center = pts.mean(axis=0)
        max_range = max(pts.max(axis=0) - pts.min(axis=0)) / 2.0 * 1.2
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    return fig


def visualize_errors(cross_errors, reproj_errors, handeye_errors):
    """Plot error distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cross-camera
    if cross_errors:
        axes[0].hist(cross_errors, bins=20, color='steelblue', edgecolor='black')
        axes[0].axvline(np.mean(cross_errors), color='red', linestyle='--',
                        label=f'mean={np.mean(cross_errors):.2f}mm')
        axes[0].set_xlabel("Position error (mm)")
        axes[0].set_title("Cross-camera consistency")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[0].transAxes)

    # Reprojection
    if reproj_errors:
        axes[1].hist(reproj_errors, bins=20, color='seagreen', edgecolor='black')
        axes[1].axvline(np.mean(reproj_errors), color='red', linestyle='--',
                        label=f'mean={np.mean(reproj_errors):.3f}px')
        axes[1].set_xlabel("Reprojection error (px)")
        axes[1].set_title("Reprojection error")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[1].transAxes)

    # Hand-eye consistency
    if handeye_errors:
        axes[2].hist(handeye_errors, bins=20, color='coral', edgecolor='black')
        axes[2].axvline(np.mean(handeye_errors), color='red', linestyle='--',
                        label=f'mean={np.mean(handeye_errors):.2f}mm')
        axes[2].set_xlabel("Board position error (mm)")
        axes[2].set_title("Hand-eye consistency")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Calibration verification & visualization")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--gripper_cam_idx", type=int, default=None)
    parser.add_argument("--no_show", action="store_true", help="Save plots without showing")
    args = parser.parse_args()

    root = args.root_folder
    calib_dir = args.calib_dir or os.path.join(root, "calib_out")

    # Load meta
    meta_path = os.path.join(root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load calibration summary
    summary_path = os.path.join(calib_dir, "calibration_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    # Load transforms
    transforms = load_calib(calib_dir)
    print(f"[INFO] Loaded transforms: {list(transforms.keys())}")

    # Camera info
    gripper_cam_idx = args.gripper_cam_idx
    if gripper_cam_idx is None:
        gripper_cam_idx = summary.get("gripper_cam_idx") or meta.get("gripper_cam_idx")

    all_cam_ids = sorted({
        int(k) for cap in meta.get("captures", [])
        for k in cap.get("cams", {}).keys()
    })
    print(f"[INFO] Cameras: {all_cam_ids}, gripper=cam{gripper_cam_idx}")

    # For gripper camera, compute T_base_C from hand-eye + robot poses
    T_gTc = transforms.get("T_gripper_cam")
    if T_gTc is not None and gripper_cam_idx is not None:
        # Use first robot pose to get approximate gripper camera position
        for cap in meta.get("captures", []):
            T_B_G = None
            if "robot_pose_matrix_4x4" in cap:
                try:
                    T_B_G = np.asarray(cap["robot_pose_matrix_4x4"], dtype=np.float64)
                except Exception:
                    pass
            if T_B_G is None and "robot_pose_6dof" in cap:
                try:
                    T_B_G = euler_deg_to_matrix(*cap["robot_pose_6dof"])
                except Exception:
                    pass
            if T_B_G is not None:
                T_base_gripper_cam = T_B_G @ T_gTc
                key = f"T_base_C{gripper_cam_idx}"
                if key not in transforms:
                    transforms[key] = T_base_gripper_cam
                break

    # ─── Run tests ───
    cross_err = test_cross_camera_consistency(meta, transforms, all_cam_ids, gripper_cam_idx)
    reproj_err = test_reprojection(meta, transforms, args.intrinsics_dir, all_cam_ids, root)
    he_err = test_handeye_consistency(meta, transforms, gripper_cam_idx)

    # ─── Print calibration summary ───
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    if summary:
        print(f"  Hand-eye method: {summary.get('selected_handeye_method', 'N/A')}")
        print(f"  Data source: {summary.get('handeye_data_source', 'N/A')}")
        print(f"  Robot poses: {summary.get('num_robot_poses', 0)}")
        print(f"  Hand-eye events: {summary.get('num_handeye_events', 0)}")
        print(f"  ChArUco frames: {summary.get('num_charuco_frames', 0)}")

    # Print transforms
    print("\n  Transforms:")
    for name, T in transforms.items():
        pos = T[:3, 3] * 1000.0
        print(f"    {name}: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm")

    # ─── Visualize ───
    save_dir = os.path.join(calib_dir, "verify")
    os.makedirs(save_dir, exist_ok=True)

    fig_3d = visualize_3d(meta, transforms, gripper_cam_idx, all_cam_ids)
    fig_3d.savefig(os.path.join(save_dir, "3d_overview.png"), dpi=150)
    print(f"\n[SAVE] {os.path.join(save_dir, '3d_overview.png')}")

    fig_err = visualize_errors(cross_err, reproj_err, he_err)
    fig_err.savefig(os.path.join(save_dir, "error_histograms.png"), dpi=150)
    print(f"[SAVE] {os.path.join(save_dir, 'error_histograms.png')}")

    if not args.no_show:
        plt.show()

    print("\n[DONE] Verification complete")


if __name__ == "__main__":
    main()
