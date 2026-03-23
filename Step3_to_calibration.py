# Step3_to_calibration.py
"""
Step 3 (Refined): Kinematics + ArUco cube unified calibration.

Compared to Step3_calibrate_all.py:
  1) Uses absolute poses for cv2.calibrateHandEye (correct input convention).
  2) Robust + weighted SE(3) averaging using reprojection confidence.
  3) Supports robot poses from either:
       - meta.json (robot_pose_matrix_4x4 / robot_pose_6dof)
       - external robot_poses_file
  4) Produces diagnostics for hand-eye consistency and per-transform stability.

Usage:
  python Step3_to_calibration.py \
    --root_folder ./data/session_01 \
    --intrinsics_dir ./intrinsics \
    --gripper_cam_idx 0 \
    --ref_fixed_cam_idx 1
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt, inv_T
from config import CubeConfig
from utils_pose import robust_se3_average, se3_distance
from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def weighted_se3_average(T_list: List[np.ndarray], w_list: Optional[List[float]] = None) -> np.ndarray:
    if len(T_list) == 0:
        raise ValueError("T_list is empty")

    if w_list is None:
        w = np.ones((len(T_list),), dtype=np.float64)
    else:
        w = np.asarray(w_list, dtype=np.float64)
        w = np.maximum(w, 1e-12)
    w = w / (w.sum() + 1e-12)

    ts = np.asarray([T[:3, 3] for T in T_list], dtype=np.float64)
    t_mean = (w[:, None] * ts).sum(axis=0)

    Rs = np.asarray([T[:3, :3] for T in T_list], dtype=np.float64)
    R_mean = (w[:, None, None] * Rs).sum(axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_mean
    return T


def robust_weighted_se3_average(
    T_list: List[np.ndarray],
    w_list: Optional[List[float]] = None,
    k_mad: float = 2.5,
    max_iters: int = 3,
    return_stats: bool = False,
):
    if len(T_list) == 0:
        raise ValueError("T_list is empty")

    if w_list is None:
        w = np.ones((len(T_list),), dtype=np.float64)
    else:
        w = np.asarray(w_list, dtype=np.float64)
        w = np.maximum(w, 1e-12)

    idx = np.arange(len(T_list), dtype=int)
    T_curr = weighted_se3_average(T_list, w)

    for _ in range(max_iters):
        res = np.array([se3_distance(T_list[i], T_curr) for i in idx], dtype=np.float64)
        med = np.median(res)
        mad = np.median(np.abs(res - med)) + 1e-12
        thr = med + k_mad * 1.4826 * mad
        keep = res <= thr
        if keep.sum() < max(3, int(0.35 * len(idx))):
            break
        new_idx = idx[keep]
        if len(new_idx) == len(idx):
            break
        idx = new_idx
        T_curr = weighted_se3_average([T_list[i] for i in idx], [w[i] for i in idx])

    T_final = weighted_se3_average([T_list[i] for i in idx], [w[i] for i in idx])

    if not return_stats:
        return T_final

    trans_devs = []
    rot_devs = []
    for T in [T_list[i] for i in idx]:
        trans_devs.append(float(np.linalg.norm(T[:3, 3] - T_final[:3, 3]) * 1000.0))
        rot_devs.append(rotation_error_deg(T[:3, :3], T_final[:3, :3]))

    stats = {
        "num_frames": int(len(T_list)),
        "num_inliers": int(len(idx)),
        "inlier_ratio": float(len(idx) / max(len(T_list), 1)),
        "translation_std_mm": float(np.std(trans_devs)) if len(trans_devs) else 0.0,
        "rotation_std_deg": float(np.std(rot_devs)) if len(rot_devs) else 0.0,
    }
    return T_final, stats


def try_parse_pose6(obj: Any) -> Optional[List[float]]:
    if obj is None:
        return None
    if isinstance(obj, list) and len(obj) == 6:
        try:
            return [float(x) for x in obj]
        except Exception:
            return None
    if isinstance(obj, dict):
        if all(k in obj for k in ["x", "y", "z", "rz", "ry", "rx"]):
            try:
                return [
                    float(obj["x"]), float(obj["y"]), float(obj["z"]),
                    float(obj["rz"]), float(obj["ry"]), float(obj["rx"]),
                ]
            except Exception:
                return None
        if all(k in obj for k in ["x", "y", "z", "roll", "pitch", "yaw"]):
            try:
                return [
                    float(obj["x"]), float(obj["y"]), float(obj["z"]),
                    float(obj["yaw"]), float(obj["pitch"]), float(obj["roll"]),
                ]
            except Exception:
                return None
        for k in ["robot_pose_6dof", "tcp_pose_6dof", "pose_6dof", "pose", "tcp_pose"]:
            if k in obj:
                p = try_parse_pose6(obj[k])
                if p is not None:
                    return p
    return None


def try_parse_T44(obj: Any) -> Optional[np.ndarray]:
    if obj is None:
        return None
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
        return None
    if isinstance(obj, dict):
        for k in ["T_B_G", "robot_pose_matrix_4x4", "matrix", "transform", "T"]:
            if k in obj:
                T = try_parse_T44(obj[k])
                if T is not None:
                    return T
    return None


def load_robot_poses_from_meta(meta: dict) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue

        T = None
        if "robot_pose_matrix_4x4" in cap:
            T = try_parse_T44(cap.get("robot_pose_matrix_4x4"))
        if T is None:
            p6 = try_parse_pose6(cap.get("robot_pose_6dof"))
            if p6 is not None:
                T = euler_deg_to_matrix(*p6)
        if T is not None:
            out[eid] = T.astype(np.float64)
    return out


def load_robot_poses_from_file(path: str) -> Dict[int, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)

    entries: List[Any]
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        if isinstance(data.get("poses"), list):
            entries = data["poses"]
        else:
            # map style: {"0": [...], "1": [...]} or mixed
            entries = []
            for k, v in data.items():
                try:
                    eid = int(k)
                    entries.append({"event_id": eid, "pose": v})
                except Exception:
                    continue
    else:
        raise RuntimeError(f"Unsupported robot_poses_file format: {type(data)}")

    out: Dict[int, np.ndarray] = {}
    for i, rec in enumerate(entries):
        eid = i
        T = None
        if isinstance(rec, dict):
            if "event_id" in rec:
                try:
                    eid = int(rec["event_id"])
                except Exception:
                    continue
            T = try_parse_T44(rec)
            if T is None:
                T = try_parse_T44(rec.get("pose"))
            if T is None:
                p6 = try_parse_pose6(rec)
                if p6 is None:
                    p6 = try_parse_pose6(rec.get("pose"))
                if p6 is not None:
                    T = euler_deg_to_matrix(*p6)
        else:
            p6 = try_parse_pose6(rec)
            if p6 is not None:
                T = euler_deg_to_matrix(*p6)

        if T is not None:
            out[eid] = T.astype(np.float64)

    if len(out) == 0:
        raise RuntimeError("No valid robot pose found in robot_poses_file")
    return out


def build_method_map() -> Dict[str, int]:
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
    if len(methods) == 0:
        # fallback ids used in old code
        methods = {"TSAI": 0, "PARK": 1, "HORAUD": 2, "ANDREFF": 3, "DANIILIDIS": 4}
    return methods


def main():
    parser = argparse.ArgumentParser(description="Refined multi-camera + hand-eye calibration")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: <root_folder>/calib_out_kinematic")

    parser.add_argument("--gripper_cam_idx", type=int, default=None,
                        help="Gripper camera index. If omitted, auto from meta/device_map")
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=None,
                        help="Reference fixed camera index. Default: first fixed cam")

    parser.add_argument("--robot_poses_file", type=str, default=None,
                        help="Optional robot pose JSON (overrides/merges meta poses)")

    parser.add_argument("--min_markers", type=int, default=1)
    parser.add_argument("--reproj_max_px", type=float, default=6.0)
    parser.add_argument("--use_ransac", action="store_true", default=True)
    parser.add_argument("--handeye_method", type=str, default="AUTO",
                        help="AUTO / TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS")

    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    out_dir = args.out_dir or os.path.join(root, "calib_out_kinematic")
    ensure_dir(out_dir)

    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)

    # camera discovery
    all_cam_ids = sorted({
        int(k)
        for cap in meta.get("captures", [])
        for k, v in cap.get("cams", {}).items()
        if v.get("saved")
    })
    if len(all_cam_ids) == 0:
        raise RuntimeError("No saved camera data in meta.json")

    gripper_cam_idx = args.gripper_cam_idx
    if gripper_cam_idx is None:
        gripper_cam_idx = meta.get("gripper_cam_idx")
    if gripper_cam_idx is None:
        dm_path = os.path.join(intr_dir, "device_map.json")
        if os.path.exists(dm_path):
            with open(dm_path, "r") as f:
                dm = json.load(f)
            gripper_cam_idx = dm.get("gripper_cam_idx")

    if gripper_cam_idx is None:
        raise RuntimeError("gripper_cam_idx is required (argument or device_map/meta)")

    fixed_cam_ids = [ci for ci in all_cam_ids if ci != gripper_cam_idx]
    if len(fixed_cam_ids) == 0:
        raise RuntimeError("No fixed cameras found")

    ref_fixed = args.ref_fixed_cam_idx
    if ref_fixed is None:
        ref_fixed = fixed_cam_ids[0]
    if ref_fixed not in fixed_cam_ids:
        raise RuntimeError(f"ref_fixed_cam_idx={ref_fixed} not in fixed cams={fixed_cam_ids}")

    print(f"[INFO] all cams: {all_cam_ids}")
    print(f"[INFO] gripper cam: cam{gripper_cam_idx}")
    print(f"[INFO] fixed cams: {fixed_cam_ids}")
    print(f"[INFO] ref fixed cam: cam{ref_fixed}")

    # intrinsics
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics(intr_dir, ci)
        print(f"[INFO] cam{ci}: intrinsics loaded")

    # robot poses
    robot_T_per_event = load_robot_poses_from_meta(meta)
    print(f"[INFO] robot poses from meta: {len(robot_T_per_event)}")

    if args.robot_poses_file:
        rp2 = load_robot_poses_from_file(args.robot_poses_file)
        robot_T_per_event.update(rp2)
        print(f"[INFO] robot poses merged from file: +{len(rp2)}")

    if len(robot_T_per_event) == 0:
        raise RuntimeError("No robot poses available. Need robot_pose_6dof/matrix or --robot_poses_file")

    # PnP per cam/event
    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    pnp_obs: Dict[int, Dict[int, dict]] = {ci: {} for ci in all_cam_ids}

    print("\n" + "=" * 60)
    print("[STEP-A] Per-camera cube PnP")
    print("=" * 60)

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in pnp_obs:
                continue
            if not cinfo.get("saved"):
                continue

            rgb_path = os.path.join(root, cinfo.get("rgb_path", ""))
            img = cv2.imread(rgb_path)
            if img is None:
                continue

            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img,
                K_map[ci],
                D_map[ci],
                use_ransac=args.use_ransac,
                min_markers=args.min_markers,
                reproj_thr_mean_px=args.reproj_max_px,
                return_reproj=True,
            )
            if not ok or reproj is None:
                continue

            pnp_obs[ci][eid] = {
                "T_C_O": rodrigues_to_Rt(rvec, tvec),
                "err_mean": float(reproj["err_mean"]),
                "n_points": int(reproj["n_points"]),
                "used_ids": [int(x) for x in used],
            }

    for ci in all_cam_ids:
        errs = [r["err_mean"] for r in pnp_obs[ci].values()]
        mean_err = float(np.mean(errs)) if errs else float("nan")
        print(f"[INFO] cam{ci}: valid={len(pnp_obs[ci])}  reproj_mean={mean_err:.3f}px")

    # (A) fixed extrinsics
    print("\n" + "=" * 60)
    print(f"[STEP-B] Fixed camera extrinsics (ref=cam{ref_fixed})")
    print("=" * 60)

    T_Cref_Ci: Dict[int, np.ndarray] = {ref_fixed: np.eye(4, dtype=np.float64)}
    fixed_stats = {}

    for ci in fixed_cam_ids:
        if ci == ref_fixed:
            continue

        common = sorted(set(pnp_obs[ref_fixed].keys()) & set(pnp_obs[ci].keys()))
        if len(common) == 0:
            print(f"[WARN] cam{ci}: no common frames with ref")
            continue

        Ts, ws = [], []
        for eid in common:
            T_ref_O = pnp_obs[ref_fixed][eid]["T_C_O"]
            T_ci_O = pnp_obs[ci][eid]["T_C_O"]
            T_ref_ci = T_ref_O @ inv_T(T_ci_O)

            er = pnp_obs[ref_fixed][eid]["err_mean"]
            ei = pnp_obs[ci][eid]["err_mean"]
            w = 1.0 / max(er * ei, 1e-9)

            Ts.append(T_ref_ci)
            ws.append(w)

        T_avg, st = robust_weighted_se3_average(Ts, ws, return_stats=True)
        T_Cref_Ci[ci] = T_avg
        fixed_stats[f"T_C{ref_fixed}_C{ci}"] = st

        np.save(os.path.join(out_dir, f"T_C{ref_fixed}_C{ci}.npy"), T_avg)
        print(f"[SAVE] T_C{ref_fixed}_C{ci}.npy  frames={len(common)}  "
              f"rot_std={st['rotation_std_deg']:.3f}deg  trans_std={st['translation_std_mm']:.2f}mm")

    # (B) hand-eye
    print("\n" + "=" * 60)
    print(f"[STEP-C] Hand-eye (gripper cam=cam{gripper_cam_idx})")
    print("=" * 60)

    common_he = sorted(set(robot_T_per_event.keys()) & set(pnp_obs[gripper_cam_idx].keys()))
    print(f"[INFO] hand-eye common events: {len(common_he)}")
    if len(common_he) < 5:
        raise RuntimeError("Not enough common events for robust hand-eye (need >= 5)")

    R_gripper2base = [robot_T_per_event[eid][:3, :3] for eid in common_he]
    t_gripper2base = [robot_T_per_event[eid][:3, 3].reshape(3, 1) for eid in common_he]
    R_target2cam = [pnp_obs[gripper_cam_idx][eid]["T_C_O"][:3, :3] for eid in common_he]
    t_target2cam = [pnp_obs[gripper_cam_idx][eid]["T_C_O"][:3, 3].reshape(3, 1) for eid in common_he]
    w_he = [1.0 / max(pnp_obs[gripper_cam_idx][eid]["err_mean"], 1e-9) for eid in common_he]

    method_map = build_method_map()
    method_sel = str(args.handeye_method or "AUTO").strip().upper()

    method_results = {}
    method_iter = method_map.items() if method_sel == "AUTO" else [(method_sel, method_map.get(method_sel))]

    for mname, mcode in method_iter:
        if mcode is None:
            print(f"[WARN] hand-eye method not available: {mname}")
            continue
        try:
            R_gc, t_gc = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=int(mcode),
            )
            T_gTc = np.eye(4, dtype=np.float64)
            T_gTc[:3, :3] = np.asarray(R_gc, dtype=np.float64).reshape(3, 3)
            T_gTc[:3, 3] = np.asarray(t_gc, dtype=np.float64).reshape(3)

            T_B_O_list = []
            for eid in common_he:
                T_B_G = robot_T_per_event[eid]
                T_Cg_O = pnp_obs[gripper_cam_idx][eid]["T_C_O"]
                T_B_O_list.append(T_B_G @ T_gTc @ T_Cg_O)

            T_B_O_avg, st_bo = robust_weighted_se3_average(T_B_O_list, w_he, return_stats=True)

            trans_mm, rot_deg = [], []
            for T in T_B_O_list:
                trans_mm.append(float(np.linalg.norm(T[:3, 3] - T_B_O_avg[:3, 3]) * 1000.0))
                rot_deg.append(rotation_error_deg(T[:3, :3], T_B_O_avg[:3, :3]))

            mean_trans = float(np.mean(trans_mm)) if trans_mm else 1e9
            mean_rot = float(np.mean(rot_deg)) if rot_deg else 1e9
            score = mean_trans + 10.0 * mean_rot

            method_results[mname] = {
                "T_gTc": T_gTc,
                "T_B_O_avg": T_B_O_avg,
                "score": float(score),
                "mean_trans_mm": mean_trans,
                "mean_rot_deg": mean_rot,
                "stability": st_bo,
            }

            print(f"  [{mname}] score={score:.3f}  "
                  f"B_O trans={mean_trans:.2f}mm  rot={mean_rot:.3f}deg")
        except Exception as e:
            print(f"  [{mname}] FAILED: {e}")

    if len(method_results) == 0:
        raise RuntimeError("All hand-eye methods failed")

    best_method = min(method_results.keys(), key=lambda k: method_results[k]["score"])
    best = method_results[best_method]
    T_gTc = best["T_gTc"]
    T_B_O_avg = best["T_B_O_avg"]

    np.save(os.path.join(out_dir, "T_gripper_cam.npy"), T_gTc)
    np.save(os.path.join(out_dir, "T_base_O.npy"), T_B_O_avg)
    print(f"[SAVE] T_gripper_cam.npy (best method={best_method})")
    print(f"[SAVE] T_base_O.npy")

    # event-wise T_B_O from selected hand-eye (for base camera estimation)
    T_B_O_by_event = {}
    for eid in common_he:
        T_B_O_by_event[eid] = robot_T_per_event[eid] @ T_gTc @ pnp_obs[gripper_cam_idx][eid]["T_C_O"]

    # (C) fixed cams in base
    print("\n" + "=" * 60)
    print("[STEP-D] Fixed cameras in robot base frame")
    print("=" * 60)

    T_base_Ci: Dict[int, np.ndarray] = {}
    base_stats = {}

    for ci in fixed_cam_ids:
        common = sorted(set(pnp_obs[ci].keys()) & set(T_B_O_by_event.keys()))
        if len(common) == 0:
            # fallback: use global base->object average
            common = sorted(pnp_obs[ci].keys())
            if len(common) == 0:
                print(f"[WARN] cam{ci}: no pnp frames")
                continue
            Ts, ws = [], []
            for eid in common:
                T_Ci_O = pnp_obs[ci][eid]["T_C_O"]
                T_B_Ci = T_B_O_avg @ inv_T(T_Ci_O)
                Ts.append(T_B_Ci)
                ws.append(1.0 / max(pnp_obs[ci][eid]["err_mean"], 1e-9))
        else:
            Ts, ws = [], []
            for eid in common:
                T_Ci_O = pnp_obs[ci][eid]["T_C_O"]
                T_B_O = T_B_O_by_event[eid]
                T_B_Ci = T_B_O @ inv_T(T_Ci_O)
                Ts.append(T_B_Ci)
                ws.append(1.0 / max(pnp_obs[ci][eid]["err_mean"], 1e-9))

        T_avg, st = robust_weighted_se3_average(Ts, ws, return_stats=True)
        T_base_Ci[ci] = T_avg
        base_stats[f"T_base_C{ci}"] = st

        np.save(os.path.join(out_dir, f"T_base_C{ci}.npy"), T_avg)
        print(f"[SAVE] T_base_C{ci}.npy  frames={len(Ts)}  "
              f"rot_std={st['rotation_std_deg']:.3f}deg  trans_std={st['translation_std_mm']:.2f}mm")

    # summary
    summary = {
        "gripper_cam_idx": int(gripper_cam_idx),
        "ref_fixed_cam_idx": int(ref_fixed),
        "fixed_cam_ids": [int(x) for x in fixed_cam_ids],
        "all_cam_ids": [int(x) for x in all_cam_ids],
        "selected_handeye_method": best_method,
        "num_robot_poses": int(len(robot_T_per_event)),
        "num_handeye_events": int(len(common_he)),
        "diagnostics": {
            "fixed_extrinsics": fixed_stats,
            "handeye_methods": {
                k: {
                    "score": float(v["score"]),
                    "mean_trans_mm": float(v["mean_trans_mm"]),
                    "mean_rot_deg": float(v["mean_rot_deg"]),
                    "stability": v["stability"],
                }
                for k, v in method_results.items()
            },
            "base_transforms": base_stats,
        },
        "transforms": {},
    }

    for ci, T in T_Cref_Ci.items():
        summary["transforms"][f"T_C{ref_fixed}_C{ci}"] = T.reshape(-1).tolist()
    summary["transforms"]["T_gripper_cam"] = T_gTc.reshape(-1).tolist()
    summary["transforms"]["T_base_O"] = T_B_O_avg.reshape(-1).tolist()
    for ci, T in T_base_Ci.items():
        summary["transforms"][f"T_base_C{ci}"] = T.reshape(-1).tolist()

    summary_path = os.path.join(out_dir, "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] {summary_path}")

    print("\n" + "=" * 60)
    print("Refined calibration COMPLETE")
    print("=" * 60)
    print(f"  output dir: {out_dir}")
    print("  saved transforms:")
    for k in summary["transforms"].keys():
        print(f"    - {k}")


if __name__ == "__main__":
    main()
