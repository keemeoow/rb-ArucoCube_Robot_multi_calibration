#!/usr/bin/env python3
"""
Step 3 (Direct Registration): Hand-eye 없이 카메라-로봇 정합.

원리:
  각 캡처에서 로봇 TCP 위치(base frame)와 카메라 PnP 큐브 위치(cam frame)는
  동일한 큐브를 가리킨다. 이 3D-3D 대응관계로 Procrustes 정합하면
  회전 다양성 없이도 T_base_cam을 구할 수 있다.

  Robot TCP (x,y) ≈ Cube (x,y) in base frame
  Camera PnP → Cube position in camera frame
  → Procrustes: cam frame → base frame

Usage:
  python Step3_direct_registration.py \
    --root_folder ./data/session_manual \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 2 \
    --ref_fixed_cam_idx 0
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt, inv_T
from config import CubeConfig
from robot_comm import euler_deg_to_matrix


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_intrinsics(intr_dir, cam_idx):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def procrustes_rigid(src, dst):
    """
    Solve rigid transform: dst = R @ src + t
    src, dst: (N, 3) arrays
    Returns: R (3,3), t (3,), residual
    """
    assert src.shape == dst.shape
    n = src.shape[0]

    # Center
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # SVD
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])  # ensure proper rotation
    R = Vt.T @ D @ U.T

    t = dst_mean - R @ src_mean

    # Residual
    transformed = (R @ src.T).T + t
    residual = np.sqrt(np.mean(np.sum((transformed - dst) ** 2, axis=1)))

    return R, t, residual


def main():
    parser = argparse.ArgumentParser(
        description="Direct registration calibration (no hand-eye rotation diversity needed)")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gripper_cam_idx", type=int, default=2)
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=0)
    parser.add_argument("--min_markers", type=int, default=1)
    parser.add_argument("--reproj_max_px", type=float, default=6.0)
    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    out_dir = args.out_dir or os.path.join(root, "calib_out_direct")
    ensure_dir(out_dir)

    ref_fixed = args.ref_fixed_cam_idx
    gripper_cam_idx = args.gripper_cam_idx

    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)

    all_cam_ids = sorted({
        int(k)
        for cap in meta.get("captures", [])
        for k, v in cap.get("cams", {}).items()
        if v.get("saved")
    })
    fixed_cam_ids = [ci for ci in all_cam_ids if ci != gripper_cam_idx]

    print(f"[INFO] All cams: {all_cam_ids}")
    print(f"[INFO] Fixed cams: {fixed_cam_ids}")
    print(f"[INFO] Gripper cam: cam{gripper_cam_idx}")
    print(f"[INFO] Ref fixed cam: cam{ref_fixed}")

    # Load intrinsics
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics(intr_dir, ci)

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    # ─── Step A: Collect per-camera PnP ───
    print(f"\n{'='*60}")
    print("[STEP A] Per-camera cube PnP")
    print(f"{'='*60}")

    pnp_per_cam: Dict[int, Dict[int, np.ndarray]] = {ci: {} for ci in all_cam_ids}
    robot_T: Dict[int, np.ndarray] = {}

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue

        # Robot TCP
        rp = cap.get("robot_pose_6dof")
        if rp:
            robot_T[eid] = euler_deg_to_matrix(*rp)

        # Per-camera PnP
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in pnp_per_cam:
                continue
            if not cinfo.get("saved"):
                continue

            rgb_path = os.path.join(root, cinfo["rgb_path"])
            img = cv2.imread(rgb_path)
            if img is None:
                continue

            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img, K_map[ci], D_map[ci],
                use_ransac=False, min_markers=args.min_markers,
                reproj_thr_mean_px=args.reproj_max_px,
                return_reproj=True)

            if ok and reproj is not None:
                pnp_per_cam[ci][eid] = rodrigues_to_Rt(rvec, tvec)

    # Filter IPPE ambiguity: remove frames where cube z-sign is inconsistent
    for ci in all_cam_ids:
        if len(pnp_per_cam[ci]) < 3:
            continue
        z_vals = {eid: T[2, 3] for eid, T in pnp_per_cam[ci].items()}
        z_signs = [1 if z > 0 else -1 for z in z_vals.values()]
        majority_sign = 1 if sum(z_signs) > 0 else -1
        bad_eids = [eid for eid, z in z_vals.items()
                    if (1 if z > 0 else -1) != majority_sign]
        for eid in bad_eids:
            del pnp_per_cam[ci][eid]
        if bad_eids:
            print(f"  cam{ci}: removed {len(bad_eids)} IPPE-ambiguous frames: {bad_eids}")

    for ci in all_cam_ids:
        print(f"  cam{ci}: {len(pnp_per_cam[ci])} valid PnP frames")
    print(f"  Robot poses: {len(robot_T)}")

    # ─── Step B: Fixed camera relative transforms ───
    print(f"\n{'='*60}")
    print(f"[STEP B] Fixed camera extrinsics (ref=cam{ref_fixed})")
    print(f"{'='*60}")

    T_Cref_Ci: Dict[int, np.ndarray] = {ref_fixed: np.eye(4)}

    for ci in fixed_cam_ids:
        if ci == ref_fixed:
            continue

        common = sorted(set(pnp_per_cam[ref_fixed].keys()) & set(pnp_per_cam[ci].keys()))
        if len(common) == 0:
            print(f"  [WARN] cam{ci}: no common frames with ref")
            continue

        Ts = []
        for eid in common:
            T_ref_O = pnp_per_cam[ref_fixed][eid]
            T_ci_O = pnp_per_cam[ci][eid]
            Ts.append(T_ref_O @ inv_T(T_ci_O))

        # Robust average
        T_avg = Ts[0].copy()
        if len(Ts) > 1:
            from utils_pose import robust_se3_average
            T_avg = robust_se3_average(Ts)

        T_Cref_Ci[ci] = T_avg
        np.save(os.path.join(out_dir, f"T_C{ref_fixed}_C{ci}.npy"), T_avg)
        print(f"  [SAVE] T_C{ref_fixed}_C{ci}.npy  frames={len(common)}")

    # ─── Step C: Independent Procrustes for EACH camera ───
    print(f"\n{'='*60}")
    print("[STEP C] Independent Procrustes for each camera")
    print(f"{'='*60}")
    print("  Each camera independently registered to robot base via cube positions")
    print("  (No dependency on relative camera transforms)")

    T_base_Ci: Dict[int, np.ndarray] = {}
    T_base_camref = None

    for ci in fixed_cam_ids:
        common_events = sorted(set(robot_T.keys()) & set(pnp_per_cam[ci].keys()))
        print(f"\n  --- cam{ci} ---")
        print(f"  Common events: {len(common_events)}")

        if len(common_events) < 3:
            print(f"  [WARN] Not enough points (<3), will use relative transform later")
            continue

        pts_cam = []
        pts_base = []
        for eid in common_events:
            pts_cam.append(pnp_per_cam[ci][eid][:3, 3])
            pts_base.append(robot_T[eid][:3, 3])

        pts_cam = np.array(pts_cam)
        pts_base = np.array(pts_base)

        R_bc, t_bc, residual = procrustes_rigid(pts_cam, pts_base)

        T_base_ci = np.eye(4, dtype=np.float64)
        T_base_ci[:3, :3] = R_bc
        T_base_ci[:3, 3] = t_bc
        T_base_Ci[ci] = T_base_ci

        if ci == ref_fixed:
            T_base_camref = T_base_ci

        pts_tr = (R_bc @ pts_cam.T).T + t_bc
        errors = np.linalg.norm(pts_tr - pts_base, axis=1) * 1000
        print(f"  Residual: {residual * 1000:.2f} mm")
        print(f"  Per-frame: mean={np.mean(errors):.2f}mm, max={np.max(errors):.2f}mm")
        print(f"  Position: [{t_bc[0]*1000:.1f}, {t_bc[1]*1000:.1f}, {t_bc[2]*1000:.1f}] mm")

        np.save(os.path.join(out_dir, f"T_base_C{ci}.npy"), T_base_ci)
        print(f"  [SAVE] T_base_C{ci}.npy")

    # Fallback for cameras with too few events: use relative transform
    if T_base_camref is None and ref_fixed in T_base_Ci:
        T_base_camref = T_base_Ci[ref_fixed]

    for ci in fixed_cam_ids:
        if ci in T_base_Ci:
            continue
        if T_base_camref is not None and ci in T_Cref_Ci:
            T_base_Ci[ci] = T_base_camref @ T_Cref_Ci[ci]
            np.save(os.path.join(out_dir, f"T_base_C{ci}.npy"), T_base_Ci[ci])
            print(f"\n  cam{ci}: [SAVE] T_base_C{ci}.npy (from relative, fallback)")

    common_events = sorted(set(robot_T.keys()) & set(pnp_per_cam[ref_fixed].keys()))
    residual = 0.0

    # ─── Step E: Verify with all cameras ───
    print(f"\n{'='*60}")
    print("[STEP E] Cross-camera verification")
    print(f"{'='*60}")

    # For each frame, compute cube position in base from each camera
    # They should agree
    pos_errors = []
    for cap in meta["captures"]:
        eid = int(cap["event_id"])
        cube_base_positions = {}

        for ci in fixed_cam_ids:
            if eid not in pnp_per_cam[ci]:
                continue
            if ci not in T_base_Ci:
                continue

            T_cam_cube = pnp_per_cam[ci][eid]
            T_base_cube = T_base_Ci[ci] @ T_cam_cube
            cube_base_positions[ci] = T_base_cube[:3, 3]

        # Compare cube positions from different cameras
        cams = list(cube_base_positions.keys())
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                err = np.linalg.norm(
                    cube_base_positions[cams[i]] - cube_base_positions[cams[j]]) * 1000
                pos_errors.append(err)

    if pos_errors:
        print(f"  Cube position agreement (pairwise):")
        print(f"    Mean:   {np.mean(pos_errors):.2f} mm")
        print(f"    Median: {np.median(pos_errors):.2f} mm")
        print(f"    Max:    {np.max(pos_errors):.2f} mm")
        print(f"    Std:    {np.std(pos_errors):.2f} mm")

    # ─── Step F: Gripper camera (optional, rough estimate) ───
    print(f"\n{'='*60}")
    print("[STEP F] Gripper camera estimate")
    print(f"{'='*60}")

    # Use: T_base_cam2 = T_base_gripper @ T_gripper_cam
    # And: T_base_cam2 @ T_cam2_cube = T_base_cube
    # Also: T_base_camref @ T_camref_cube = T_base_cube
    # So: T_base_gripper @ T_gripper_cam = T_base_camref @ T_camref_cube @ inv(T_cam2_cube)

    common_gripper = sorted(
        set(robot_T.keys()) &
        set(pnp_per_cam[gripper_cam_idx].keys()) &
        set(pnp_per_cam[ref_fixed].keys())
    )
    print(f"  Common events (robot + gripper + ref): {len(common_gripper)}")

    if len(common_gripper) >= 3:
        T_gTc_list = []
        for eid in common_gripper:
            T_bg = robot_T[eid]  # already in meters

            T_cam2_cube = pnp_per_cam[gripper_cam_idx][eid]
            T_camref_cube = pnp_per_cam[ref_fixed][eid]

            # T_base_cube from ref camera (most accurate)
            T_base_cube = T_base_camref @ T_camref_cube

            # T_base_cam2 = T_base_cube @ inv(T_cam2_cube)
            T_base_cam2 = T_base_cube @ inv_T(T_cam2_cube)

            # T_gripper_cam = inv(T_base_grip) @ T_base_cam2
            T_gTc = inv_T(T_bg) @ T_base_cam2
            T_gTc_list.append(T_gTc)

        from utils_pose import robust_se3_average
        T_gTc_avg = robust_se3_average(T_gTc_list)

        t_gc = T_gTc_avg[:3, 3]
        print(f"  T_gripper_cam translation: [{t_gc[0]*1000:.1f}, {t_gc[1]*1000:.1f}, {t_gc[2]*1000:.1f}] mm")
        print(f"  T_gripper_cam distance: {np.linalg.norm(t_gc)*1000:.1f} mm")

        # Check consistency
        gc_positions = np.array([T[:3, 3] for T in T_gTc_list])
        gc_std = np.std(gc_positions, axis=0) * 1000
        print(f"  Consistency (std): [{gc_std[0]:.1f}, {gc_std[1]:.1f}, {gc_std[2]:.1f}] mm")

        np.save(os.path.join(out_dir, "T_gripper_cam.npy"), T_gTc_avg)
        print(f"  [SAVE] T_gripper_cam.npy")
    else:
        T_gTc_avg = None
        print("  [WARN] Not enough common events for gripper camera estimate")

    # ─── Save summary ───
    summary = {
        "method": "direct_procrustes_registration",
        "gripper_cam_idx": gripper_cam_idx,
        "ref_fixed_cam_idx": ref_fixed,
        "fixed_cam_ids": [int(x) for x in fixed_cam_ids],
        "all_cam_ids": [int(x) for x in all_cam_ids],
        "procrustes_residual_mm": float(residual * 1000),
        "procrustes_n_points": len(common_events),
        "cross_camera_error_mm": {
            "mean": float(np.mean(pos_errors)) if pos_errors else None,
            "median": float(np.median(pos_errors)) if pos_errors else None,
            "max": float(np.max(pos_errors)) if pos_errors else None,
        },
        "transforms": {},
    }

    for ci, T in T_base_Ci.items():
        summary["transforms"][f"T_base_C{ci}"] = T.reshape(-1).tolist()
    for ci, T in T_Cref_Ci.items():
        if ci != ref_fixed:
            summary["transforms"][f"T_C{ref_fixed}_C{ci}"] = T.reshape(-1).tolist()
    if T_gTc_avg is not None:
        summary["transforms"]["T_gripper_cam"] = T_gTc_avg.reshape(-1).tolist()

    # Save T_base_O (average cube position in base frame)
    T_base_O_list = []
    for eid in common_events:
        T_base_O_list.append(T_base_camref @ pnp_per_cam[ref_fixed][eid])
    T_base_O = robust_se3_average(T_base_O_list) if len(T_base_O_list) > 1 else T_base_O_list[0]
    np.save(os.path.join(out_dir, "T_base_O.npy"), T_base_O)
    summary["transforms"]["T_base_O"] = T_base_O.reshape(-1).tolist()

    with open(os.path.join(out_dir, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] calibration_summary.json")

    print(f"\n{'='*60}")
    print("[DONE] Direct registration calibration complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
