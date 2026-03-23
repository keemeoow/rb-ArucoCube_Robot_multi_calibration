# Step3_in_calibration.py
"""
Step 3B: Eye-in-hand calibration using ChArUco board captures.

Solves for T_gripper_cam using:
  - Robot TCP poses (T_base_gripper) from each capture
  - ChArUco board poses (T_cam_board) from each capture
  - cv2.calibrateHandEye (AX=XB)

Requires rotation diversity in robot poses.

Usage:
  python Step3_in_calibration.py \
    --charuco_folder ./data/charuco_session \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 2

  # Combine with cube calibration:
  python Step3_in_calibration.py \
    --charuco_folder ./data/charuco_session \
    --cube_calib_dir ./data/session_manual/calib_out_direct \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 2
"""

import os
import json
import argparse
from typing import Dict, List, Optional

import cv2
import numpy as np

from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def rotation_error_deg(Ra, Rb):
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main():
    parser = argparse.ArgumentParser(
        description="Eye-in-hand calibration from ChArUco captures"
    )
    parser.add_argument("--charuco_folder", required=True,
                        help="Folder with meta_charuco.json from Step2B")
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--gripper_cam_idx", type=int, default=2)
    parser.add_argument("--cube_calib_dir", type=str, default=None,
                        help="Optional: cube calibration dir to merge T_gripper_cam into")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--handeye_method", type=str, default="AUTO",
                        help="AUTO / TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS")

    args = parser.parse_args()

    charuco_folder = args.charuco_folder
    out_dir = args.out_dir or os.path.join(charuco_folder, "calib_out_handeye")
    ensure_dir(out_dir)

    # Load charuco meta
    meta_path = os.path.join(charuco_folder, "meta_charuco.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"[INFO] ChArUco captures: {len(meta['captures'])}")
    print(f"[INFO] Gripper camera: cam{args.gripper_cam_idx}")

    # Collect valid captures
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_board2cam_list = []
    t_board2cam_list = []
    reproj_list = []
    events_used = []

    for cap in meta["captures"]:
        if not cap.get("saved"):
            continue

        # Robot pose
        T_bg = None
        if "T_base_gripper_4x4" in cap:
            T_bg = np.array(cap["T_base_gripper_4x4"], dtype=np.float64)
        elif "robot_pose_6dof" in cap:
            T_bg = euler_deg_to_matrix(*cap["robot_pose_6dof"])

        if T_bg is None:
            continue

        # Board pose
        T_cb = None
        if "T_cam_board_4x4" in cap:
            T_cb = np.array(cap["T_cam_board_4x4"], dtype=np.float64)
        elif "rvec" in cap and "tvec" in cap:
            rvec = np.array(cap["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.array(cap["tvec"], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            T_cb = np.eye(4, dtype=np.float64)
            T_cb[:3, :3] = R
            T_cb[:3, 3] = tvec.flatten()

        if T_cb is None:
            continue

        R_gripper2base_list.append(T_bg[:3, :3])
        t_gripper2base_list.append(T_bg[:3, 3].reshape(3, 1))
        R_board2cam_list.append(T_cb[:3, :3])
        t_board2cam_list.append(T_cb[:3, 3].reshape(3, 1))
        reproj_list.append(cap.get("reproj_error_px", 1.0))
        events_used.append(cap["event_id"])

    n = len(events_used)
    print(f"[INFO] Valid captures for hand-eye: {n}")

    if n < 5:
        raise RuntimeError(f"Need at least 5 captures (have {n}). Add more with rotation diversity.")

    # Check rotation diversity
    Rs = np.array(R_gripper2base_list)
    rot_angles = []
    for i in range(1, n):
        rot_angles.append(rotation_error_deg(Rs[i], Rs[0]))
    max_rot = max(rot_angles) if rot_angles else 0
    mean_rot = np.mean(rot_angles) if rot_angles else 0

    print(f"[INFO] Rotation diversity: mean={mean_rot:.1f}deg, max={max_rot:.1f}deg")
    if max_rot < 15:
        print(f"[WARN] Low rotation diversity ({max_rot:.1f}deg). Results may be inaccurate.")
        print(f"[WARN] Recommended: >30 deg range in multiple axes")

    # Build method map
    methods = {}
    cand = {
        "TSAI": "CALIB_HAND_EYE_TSAI",
        "PARK": "CALIB_HAND_EYE_PARK",
        "HORAUD": "CALIB_HAND_EYE_HORAUD",
        "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
        "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
    }
    for name, cv_attr in cand.items():
        if hasattr(cv2, cv_attr):
            methods[name] = int(getattr(cv2, cv_attr))
    if not methods:
        methods = {"TSAI": 0, "PARK": 1, "HORAUD": 2, "ANDREFF": 3, "DANIILIDIS": 4}

    method_sel = args.handeye_method.strip().upper()

    print(f"\n{'='*60}")
    print("[STEP] Hand-eye calibration (AX=XB)")
    print(f"{'='*60}")

    method_results = {}
    method_iter = methods.items() if method_sel == "AUTO" else [(method_sel, methods.get(method_sel))]

    for mname, mcode in method_iter:
        if mcode is None:
            continue
        try:
            R_gc, t_gc = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base_list,
                t_gripper2base=t_gripper2base_list,
                R_target2cam=R_board2cam_list,
                t_target2cam=t_board2cam_list,
                method=int(mcode),
            )

            T_gc = np.eye(4, dtype=np.float64)
            T_gc[:3, :3] = np.asarray(R_gc).reshape(3, 3)
            T_gc[:3, 3] = np.asarray(t_gc).reshape(3)

            # Consistency check: T_base_board should be constant
            T_base_board_list = []
            for i in range(n):
                T_bg = np.eye(4, dtype=np.float64)
                T_bg[:3, :3] = R_gripper2base_list[i]
                T_bg[:3, 3] = t_gripper2base_list[i].flatten()

                T_cb = np.eye(4, dtype=np.float64)
                T_cb[:3, :3] = R_board2cam_list[i]
                T_cb[:3, 3] = t_board2cam_list[i].flatten()

                T_base_board = T_bg @ T_gc @ T_cb
                T_base_board_list.append(T_base_board)

            # Measure consistency
            positions = np.array([T[:3, 3] for T in T_base_board_list])
            pos_std = np.std(positions, axis=0) * 1000  # mm
            pos_mean_std = float(np.mean(pos_std))

            rot_devs = []
            for T in T_base_board_list:
                rot_devs.append(rotation_error_deg(T[:3, :3], T_base_board_list[0][:3, :3]))
            rot_mean = float(np.mean(rot_devs))

            score = pos_mean_std + 10.0 * rot_mean

            method_results[mname] = {
                "T_gripper_cam": T_gc,
                "score": score,
                "pos_std_mm": pos_mean_std,
                "rot_mean_deg": rot_mean,
                "pos_std_xyz_mm": pos_std.tolist(),
            }

            t_gc_mm = T_gc[:3, 3] * 1000
            print(f"  [{mname}] score={score:.3f}  "
                  f"board_pos_std={pos_mean_std:.2f}mm  board_rot_mean={rot_mean:.3f}deg  "
                  f"offset=[{t_gc_mm[0]:.1f}, {t_gc_mm[1]:.1f}, {t_gc_mm[2]:.1f}]mm")

        except Exception as e:
            print(f"  [{mname}] FAILED: {e}")

    if not method_results:
        raise RuntimeError("All hand-eye methods failed")

    # Select best
    best_method = min(method_results, key=lambda k: method_results[k]["score"])
    best = method_results[best_method]
    T_gc = best["T_gripper_cam"]

    print(f"\n[BEST] {best_method}")
    t_mm = T_gc[:3, 3] * 1000
    print(f"  T_gripper_cam translation: [{t_mm[0]:.2f}, {t_mm[1]:.2f}, {t_mm[2]:.2f}] mm")
    print(f"  Distance: {np.linalg.norm(t_mm):.2f} mm")
    print(f"  Board position std: {best['pos_std_mm']:.2f} mm")
    print(f"  Board rotation consistency: {best['rot_mean_deg']:.3f} deg")

    # Save
    np.save(os.path.join(out_dir, "T_gripper_cam.npy"), T_gc)
    print(f"\n[SAVE] T_gripper_cam.npy")

    # Save summary
    summary = {
        "calibration_type": "charuco_eye_in_hand",
        "gripper_cam_idx": args.gripper_cam_idx,
        "n_captures": n,
        "rotation_diversity_deg": float(max_rot),
        "selected_method": best_method,
        "all_methods": {
            k: {
                "score": v["score"],
                "pos_std_mm": v["pos_std_mm"],
                "rot_mean_deg": v["rot_mean_deg"],
                "pos_std_xyz_mm": v["pos_std_xyz_mm"],
            }
            for k, v in method_results.items()
        },
        "T_gripper_cam": T_gc.flatten().tolist(),
        "T_gripper_cam_translation_mm": (T_gc[:3, 3] * 1000).tolist(),
        "T_gripper_cam_distance_mm": float(np.linalg.norm(T_gc[:3, 3] * 1000)),
    }

    with open(os.path.join(out_dir, "handeye_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] handeye_summary.json")

    # Optionally merge into cube calibration
    if args.cube_calib_dir and os.path.isdir(args.cube_calib_dir):
        dst = os.path.join(args.cube_calib_dir, "T_gripper_cam.npy")
        np.save(dst, T_gc)
        print(f"\n[MERGE] T_gripper_cam.npy → {dst}")
        print(f"  Replaced cube-based hand-eye with ChArUco-based result")

    print(f"\n{'='*60}")
    print("Eye-in-hand calibration COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
