# Step3_calibrate_all.py
"""
Step 3: Unified calibration — computes ALL transforms in one script.

Given captures from Step2 (cube seen by fixed cams + gripper cam + robot poses):

  (A) Fixed camera extrinsics:
      T_Cref_Ci for each fixed camera i relative to a reference fixed camera.
      Method: per-frame cube PnP → T_Ci_O, then T_Cref_Ci = T_Cref_O @ inv(T_Ci_O)

  (B) Hand-eye calibration (eye-in-hand):
      T_gripper_cam (gTc): transform from gripper (EE) to gripper camera.
      Method: AX=XB using robot TCP poses + cube poses from gripper camera.

  (C) Fixed cameras in robot base frame:
      T_base_Cref (bTcref): reference fixed camera in robot base frame.
      Method: from bTg @ gTc @ cTo and the known cube poses.
      Then T_base_Ci = T_base_Cref @ T_Cref_Ci for all fixed cameras.

Usage:
  python Step3_calibrate_all.py \
    --root_folder ./data/session_01 \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 0 \
    --ref_fixed_cam_idx 1 \
    --robot_poses_file robot_poses.json

  OR if robot_poses are embedded in meta.json (from Step2 --use_robot):
  python Step3_calibrate_all.py \
    --root_folder ./data/session_01 \
    --intrinsics_dir ./intrinsics
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt, inv_T
from config import CubeConfig, CalibrationConfig
from utils_pose import robust_se3_average, se3_distance
from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def compute_pose_error(T1, T2):
    """Position error (m) and rotation error (deg)."""
    pos_err = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    R_err = T1[:3, :3].T @ T2[:3, :3]
    ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
    return pos_err, np.rad2deg(ang)


def average_transforms(T_list):
    """Simple SVD-based rotation average + translation mean."""
    positions = np.array([T[:3, 3] for T in T_list])
    Rs = np.array([T[:3, :3] for T in T_list])
    R_sum = Rs.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    T_avg = np.eye(4, dtype=np.float64)
    T_avg[:3, :3] = R_mean
    T_avg[:3, 3] = positions.mean(axis=0)
    return T_avg


def main():
    parser = argparse.ArgumentParser(description="Unified multi-camera + hand-eye calibration")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)

    parser.add_argument("--gripper_cam_idx", type=int, default=None,
                        help="Camera index for the gripper camera. Auto-detected from device_map if not set.")
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=None,
                        help="Reference fixed camera index. Default: first fixed camera.")

    parser.add_argument("--robot_poses_file", type=str, default=None,
                        help="JSON with robot TCP poses [[x,y,z,rz,ry,rx], ...] in mm/deg. "
                             "If not set, reads from meta.json robot_pose_6dof fields.")

    # Detection params
    parser.add_argument("--min_markers", type=int, default=1)
    parser.add_argument("--reproj_max_px", type=float, default=10.0)
    parser.add_argument("--use_ransac", action="store_true", default=True)

    # Hand-eye method
    parser.add_argument("--handeye_method", type=int, default=4,
                        help="OpenCV hand-eye method (0-4). 4=PARK")

    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir

    # ─── Load meta ───
    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)

    gripper_cam_idx = args.gripper_cam_idx
    if gripper_cam_idx is None:
        gripper_cam_idx = meta.get("gripper_cam_idx")
    if gripper_cam_idx is None:
        # Try from device_map
        map_path = os.path.join(intr_dir, "device_map.json")
        if os.path.exists(map_path):
            with open(map_path) as f:
                dm = json.load(f)
            gripper_cam_idx = dm.get("gripper_cam_idx")

    if gripper_cam_idx is None:
        print("[WARN] gripper_cam_idx not set. Hand-eye calibration will be skipped.")

    # Discover all cam indices from captures
    all_cam_ids = set()
    for cap in meta["captures"]:
        for k, v in cap["cams"].items():
            if v.get("saved"):
                all_cam_ids.add(int(k))
    all_cam_ids = sorted(all_cam_ids)

    fixed_cam_ids = [ci for ci in all_cam_ids if ci != gripper_cam_idx]

    ref_fixed = args.ref_fixed_cam_idx
    if ref_fixed is None:
        ref_fixed = fixed_cam_ids[0] if fixed_cam_ids else None
    if ref_fixed is not None and ref_fixed not in fixed_cam_ids:
        raise RuntimeError(f"ref_fixed_cam_idx={ref_fixed} not in fixed cameras {fixed_cam_ids}")

    print(f"[INFO] All cameras: {all_cam_ids}")
    print(f"[INFO] Gripper camera: cam{gripper_cam_idx}")
    print(f"[INFO] Fixed cameras: {fixed_cam_ids}")
    print(f"[INFO] Reference fixed camera: cam{ref_fixed}")

    # ─── Load intrinsics ───
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics(intr_dir, ci)
        print(f"[INFO] cam{ci}: loaded intrinsics")

    # ─── Load robot poses (if available) ───
    robot_T_per_event: Dict[int, np.ndarray] = {}
    if args.robot_poses_file:
        with open(args.robot_poses_file) as f:
            rp_list = json.load(f)
        for i, rp in enumerate(rp_list):
            robot_T_per_event[i] = euler_deg_to_matrix(*rp)
        print(f"[INFO] Loaded {len(rp_list)} robot poses from file")
    else:
        # Extract from meta.json
        for cap in meta["captures"]:
            eid = cap["event_id"]
            rp = cap.get("robot_pose_6dof")
            if rp is not None and len(rp) == 6:
                robot_T_per_event[eid] = euler_deg_to_matrix(*rp)
        if robot_T_per_event:
            print(f"[INFO] Loaded {len(robot_T_per_event)} robot poses from meta.json")

    # ─── Cube PnP per camera per frame ───
    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    out_dir = ensure_dir(os.path.join(root, "calib_out"))

    # T_Ci_O[cam_idx][event_id] = 4x4 transform (Object->Camera)
    T_Ci_O: Dict[int, Dict[int, np.ndarray]] = {ci: {} for ci in all_cam_ids}

    for cap in meta["captures"]:
        eid = cap["event_id"]
        for ci_str, cinfo in cap["cams"].items():
            ci = int(ci_str)
            if not cinfo.get("saved") or not cinfo.get("cube_visible", True):
                continue

            rgb_path = os.path.join(root, cinfo["rgb_path"])
            img = cv2.imread(rgb_path)
            if img is None:
                continue

            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img, K_map[ci], D_map[ci],
                use_ransac=args.use_ransac,
                min_markers=args.min_markers,
                reproj_thr_mean_px=args.reproj_max_px,
                return_reproj=True)

            if ok and reproj is not None:
                T_Ci_O[ci][eid] = rodrigues_to_Rt(rvec, tvec)

    for ci in all_cam_ids:
        print(f"[INFO] cam{ci}: {len(T_Ci_O[ci])} valid PnP frames")

    # ═══════════════════════════════════════════════════════════════
    # (A) Fixed camera extrinsics: T_Cref_Ci
    # ═══════════════════════════════════════════════════════════════
    T_Cref_Ci: Dict[int, np.ndarray] = {}

    if ref_fixed is not None and len(fixed_cam_ids) > 0:
        print(f"\n{'='*60}")
        print(f"(A) Fixed camera extrinsics (ref=cam{ref_fixed})")
        print(f"{'='*60}")

        ref_frames = T_Ci_O.get(ref_fixed, {})
        T_Cref_Ci[ref_fixed] = np.eye(4, dtype=np.float64)

        for ci in fixed_cam_ids:
            if ci == ref_fixed:
                continue

            ci_frames = T_Ci_O.get(ci, {})
            common = sorted(set(ref_frames.keys()) & set(ci_frames.keys()))

            if len(common) == 0:
                print(f"[WARN] cam{ci}: no common frames with ref cam{ref_fixed}")
                continue

            T_list = []
            for fid in common:
                T_ref_O = ref_frames[fid]
                T_ci_O = ci_frames[fid]
                T_ref_ci = T_ref_O @ inv_T(T_ci_O)
                T_list.append(T_ref_ci)

            T_avg, stats = robust_se3_average(T_list, return_stats=True)
            T_Cref_Ci[ci] = T_avg

            npy_path = os.path.join(out_dir, f"T_C{ref_fixed}_C{ci}.npy")
            np.save(npy_path, T_avg)
            print(f"[SAVE] {npy_path}  ({len(T_list)} frames, "
                  f"rot_std={stats['rotation_std_deg']:.3f}°, "
                  f"trans_std={stats['translation_std_mm']:.2f}mm)")

    # ═══════════════════════════════════════════════════════════════
    # (B) Hand-eye calibration: T_gripper_cam (gTc)
    # ═══════════════════════════════════════════════════════════════
    T_gTc: Optional[np.ndarray] = None

    if gripper_cam_idx is not None and len(robot_T_per_event) > 0:
        print(f"\n{'='*60}")
        print(f"(B) Hand-eye calibration (gripper cam=cam{gripper_cam_idx})")
        print(f"{'='*60}")

        gripper_frames = T_Ci_O.get(gripper_cam_idx, {})

        # Collect paired (robot_T_base_gripper, cam_T_cam_object) for common events
        common_events = sorted(set(robot_T_per_event.keys()) & set(gripper_frames.keys()))
        print(f"[INFO] Common events (robot + gripper cam): {len(common_events)}")

        if len(common_events) < 3:
            print("[WARN] Not enough common events for hand-eye calibration (need >= 3)")
        else:
            # Filter by reprojection quality (already filtered by reproj_max_px above)
            robot_Ts = [robot_T_per_event[eid] for eid in common_events]
            cam_Ts = [gripper_frames[eid] for eid in common_events]

            # AX = XB formulation
            # A = inv(robot_Ts[i]) @ robot_Ts[i+1]  (gripper motion)
            # B = cam_Ts[i+1] @ inv(cam_Ts[i])       (camera motion)
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []

            for i in range(len(robot_Ts) - 1):
                A = np.linalg.inv(robot_Ts[i]) @ robot_Ts[i + 1]
                B = cam_Ts[i + 1] @ np.linalg.inv(cam_Ts[i])
                R_gripper2base.append(A[:3, :3])
                t_gripper2base.append(A[:3, 3])
                R_target2cam.append(B[:3, :3])
                t_target2cam.append(B[:3, 3])

            # Try multiple methods, pick best
            methods = {
                0: "TSAI",
                1: "PARK",
                2: "HORAUD",
                3: "ANDREFF",
                4: "DANIILIDIS",
            }
            best_method = args.handeye_method
            best_T = None

            for method_id, method_name in methods.items():
                try:
                    R_gc, t_gc = cv2.calibrateHandEye(
                        R_gripper2base, t_gripper2base,
                        R_target2cam, t_target2cam,
                        method=method_id)

                    T_gc = np.eye(4, dtype=np.float64)
                    T_gc[:3, :3] = R_gc
                    T_gc[:3, 3] = t_gc.flatten()

                    # Validate: det(R) should be 1, R^T R should be I
                    det_R = np.linalg.det(R_gc)
                    ortho_err = np.linalg.norm(R_gc.T @ R_gc - np.eye(3))

                    # Compute verification error
                    pos_errors, rot_errors = [], []
                    T_bTt_list = [(robot_Ts[i] @ T_gc) @ cam_Ts[i] for i in range(len(robot_Ts))]
                    T_bTt = average_transforms(T_bTt_list)

                    for i in range(len(robot_Ts)):
                        T_bTc_est = robot_Ts[i] @ T_gc
                        T_bTc_gt = T_bTt @ np.linalg.inv(cam_Ts[i])
                        pe, re = compute_pose_error(T_bTc_est, T_bTc_gt)
                        pos_errors.append(pe)
                        rot_errors.append(re)

                    mean_pos = np.mean(pos_errors) * 1000  # mm
                    mean_rot = np.mean(rot_errors)  # deg

                    print(f"  [{method_name}] det(R)={det_R:.6f} ortho_err={ortho_err:.6f} "
                          f"pos_err={mean_pos:.2f}mm rot_err={mean_rot:.2f}°")

                    if method_id == best_method:
                        best_T = T_gc

                except Exception as e:
                    print(f"  [{method_name}] FAILED: {e}")

            if best_T is not None:
                T_gTc = best_T
                gTc_path = os.path.join(out_dir, "T_gripper_cam.npy")
                np.save(gTc_path, T_gTc)
                print(f"\n[SAVE] {gTc_path} (method={methods[best_method]})")
                print(f"  T_gTc =\n{T_gTc}")

                t = T_gTc[:3, 3]
                print(f"  Translation: X={t[0]*100:.2f}cm Y={t[1]*100:.2f}cm Z={t[2]*100:.2f}cm "
                      f"dist={np.linalg.norm(t)*100:.2f}cm")
            else:
                print("[ERROR] All hand-eye methods failed!")

    # ═══════════════════════════════════════════════════════════════
    # (C) Fixed cameras in robot base frame: T_base_Ci
    # ═══════════════════════════════════════════════════════════════
    T_base_Ci: Dict[int, np.ndarray] = {}

    if T_gTc is not None and ref_fixed is not None and len(robot_T_per_event) > 0:
        print(f"\n{'='*60}")
        print(f"(C) Fixed cameras in robot base frame")
        print(f"{'='*60}")

        # For each event where both the gripper cam AND a fixed cam see the cube:
        # T_base_Ci = T_base_gripper @ T_gripper_cam @ T_cam_cube @ inv(T_fixedcam_cube)
        #           = T_bTg @ gTc @ cTo @ inv(T_fi_O)

        for fi in fixed_cam_ids:
            fi_frames = T_Ci_O.get(fi, {})
            gripper_frames = T_Ci_O.get(gripper_cam_idx, {})

            # Events where robot pose, gripper cam, AND this fixed cam all have data
            common = sorted(
                set(robot_T_per_event.keys()) &
                set(gripper_frames.keys()) &
                set(fi_frames.keys())
            )

            if len(common) == 0:
                print(f"[WARN] cam{fi}: no common events with gripper+robot. "
                      f"Using indirect method via cube.")
                # Fallback: use T_base_cube from gripper, then T_base_fi = T_base_O @ inv(T_fi_O)
                common_gr = sorted(set(robot_T_per_event.keys()) & set(gripper_frames.keys()))
                if len(common_gr) == 0:
                    print(f"[WARN] cam{fi}: cannot compute base transform. Skipping.")
                    continue

                # Compute average T_base_O from gripper observations
                T_bO_list = []
                for eid in common_gr:
                    T_bTg = robot_T_per_event[eid]
                    T_cTo = gripper_frames[eid]  # T_gripper_cam_O
                    T_bTo = T_bTg @ T_gTc @ T_cTo
                    T_bO_list.append(T_bTo)
                T_base_O = robust_se3_average(T_bO_list)

                # Now for each frame where fi sees the cube
                T_bf_list = []
                for eid in fi_frames.keys():
                    T_fi_O = fi_frames[eid]
                    T_base_fi = T_base_O @ inv_T(T_fi_O)
                    T_bf_list.append(T_base_fi)

                if len(T_bf_list) > 0:
                    T_base_Ci[fi] = robust_se3_average(T_bf_list)
                continue

            T_bf_list = []
            for eid in common:
                T_bTg = robot_T_per_event[eid]
                T_cam_O = gripper_frames[eid]
                T_fi_O = fi_frames[eid]

                # Method: T_base_O = T_bTg @ gTc @ cam_O
                # Then:   T_base_fi = T_base_O @ inv(T_fi_O)
                T_base_O = T_bTg @ T_gTc @ T_cam_O
                T_base_fi = T_base_O @ inv_T(T_fi_O)
                T_bf_list.append(T_base_fi)

            T_avg, stats = robust_se3_average(T_bf_list, return_stats=True)
            T_base_Ci[fi] = T_avg
            print(f"[INFO] cam{fi}: T_base_C{fi} from {len(T_bf_list)} frames "
                  f"(rot_std={stats['rotation_std_deg']:.3f}°, trans_std={stats['translation_std_mm']:.2f}mm)")

        # Save all base transforms
        for ci, T in T_base_Ci.items():
            npy_path = os.path.join(out_dir, f"T_base_C{ci}.npy")
            np.save(npy_path, T)
            print(f"[SAVE] {npy_path}")

        # Also save gripper camera in base (it's T_bTg @ gTc, but varies per frame)
        # Save the relationship instead
        print(f"\n[INFO] To get gripper camera in base at runtime:")
        print(f"       T_base_cam = T_base_gripper(from_robot) @ T_gripper_cam")
        print(f"       T_gripper_cam is saved as T_gripper_cam.npy")

    # ═══════════════════════════════════════════════════════════════
    # Summary JSON
    # ═══════════════════════════════════════════════════════════════
    summary = {
        "gripper_cam_idx": gripper_cam_idx,
        "ref_fixed_cam_idx": ref_fixed,
        "fixed_cam_ids": fixed_cam_ids,
        "all_cam_ids": all_cam_ids,
        "transforms": {},
    }

    # Fixed cam extrinsics
    for ci, T in T_Cref_Ci.items():
        key = f"T_C{ref_fixed}_C{ci}"
        summary["transforms"][key] = T.reshape(-1).tolist()

    # Hand-eye
    if T_gTc is not None:
        summary["transforms"]["T_gripper_cam"] = T_gTc.reshape(-1).tolist()

    # Base transforms
    for ci, T in T_base_Ci.items():
        key = f"T_base_C{ci}"
        summary["transforms"][key] = T.reshape(-1).tolist()

    summary_path = os.path.join(out_dir, "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] {summary_path}")

    print("\n" + "=" * 60)
    print("Step3 calibration COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {out_dir}")
    print(f"  Transforms saved:")
    for k in summary["transforms"]:
        print(f"    - {k}")


if __name__ == "__main__":
    main()
