# Step3_calibrate_grasped_cube_fk.py
"""
Step 3 (new): Robot grasps cube, then calibrate fixed cameras using
kinematics + fixed-cam PnP only.

[명령어]:
python Step3_calibrate_grasped_cube_fk.py \
  --root_folder ./data/session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_fixed_cam_idx 1 \
  --pair_mode consecutive \
  --min_markers 1 \
  --reproj_max_px 6.0 \
  --min_motion_rot_deg 2.0

Core equations (cube rigidly attached to gripper):
  T_base_cube(k) = T_base_grip(k) * T_grip_cube

For each fixed camera i:
  T_base_cube(k) = T_base_Ci * T_Ci_cube(k)

So:
  T_base_Ci = [T_base_grip(k) * T_grip_cube] * inv(T_Ci_cube(k))

Compared to the old route (with gripper camera hand-eye), this removes
one PnP error source in the final fixed-camera base transform estimation.

결과물:
  - T_grip_cube.npy
  - T_base_C{idx}.npy for fixed cameras
  - T_C{ref}_C{idx}.npy (derived from base transforms)
  - calibration_summary_grasped_cube.json
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt, inv_T
from config import CubeConfig
from utils_pose import se3_distance
from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_intrinsics(intr_dir: str, cam_idx: int):
    p = os.path.join(intr_dir, f"cam{cam_idx}.npz")
    d = np.load(p, allow_pickle=True)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def parse_cam_idx_csv(text: str) -> List[int]:
    s = str(text or "").strip()
    if s == "":
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return sorted(set(out))


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
        for k in ["T_B_G", "robot_pose_matrix_4x4", "matrix", "transform", "T", "pose_matrix"]:
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
        raise RuntimeError("No valid robot poses found in robot_poses_file")
    return out


def build_motion_pairs(
    robot_T_per_event: Dict[int, np.ndarray],
    pnp_by_cam: Dict[int, Dict[int, dict]],
    fixed_cam_ids: List[int],
    pair_mode: str,
    min_motion_rot_deg: float,
):
    pairs = []

    for ci in fixed_cam_ids:
        events = sorted(set(robot_T_per_event.keys()) & set(pnp_by_cam.get(ci, {}).keys()))
        if len(events) < 2:
            continue

        idx_pairs: List[Tuple[int, int]] = []
        if pair_mode == "all":
            for a in range(len(events)):
                for b in range(a + 1, len(events)):
                    idx_pairs.append((a, b))
        else:
            # consecutive
            for a in range(len(events) - 1):
                idx_pairs.append((a, a + 1))

        for ia, ib in idx_pairs:
            e0 = events[ia]
            e1 = events[ib]

            G0 = robot_T_per_event[e0]
            G1 = robot_T_per_event[e1]
            C0 = pnp_by_cam[ci][e0]["T_C_O"]
            C1 = pnp_by_cam[ci][e1]["T_C_O"]

            # A X = X B
            # A = inv(G0) G1
            # B = inv(C0) C1
            A = inv_T(G0) @ G1
            B = inv_T(C0) @ C1

            rotA = rotation_error_deg(A[:3, :3], np.eye(3, dtype=np.float64))
            rotB = rotation_error_deg(B[:3, :3], np.eye(3, dtype=np.float64))
            if rotA < min_motion_rot_deg or rotB < min_motion_rot_deg:
                continue

            e0_err = float(pnp_by_cam[ci][e0]["err_mean"])
            e1_err = float(pnp_by_cam[ci][e1]["err_mean"])
            w = 1.0 / max(e0_err * e1_err, 1e-9)

            pairs.append({
                "cam_idx": int(ci),
                "e0": int(e0),
                "e1": int(e1),
                "A": A,
                "B": B,
                "weight": float(w),
                "rotA_deg": float(rotA),
                "rotB_deg": float(rotB),
                "pnp_err_pair": [e0_err, e1_err],
            })

    return pairs


def solve_grip_cube_from_pairs(
    pairs: List[dict],
    max_iters: int = 5,
    k_mad: float = 2.5,
):
    if len(pairs) < 6:
        raise RuntimeError(f"Not enough motion pairs for robust solve: {len(pairs)} (<6)")

    idx = np.arange(len(pairs), dtype=int)

    def solve_from_indices(sel_idx: np.ndarray) -> np.ndarray:
        # --- Rotation: solve a ~= R * b (Park-Martin style) ---
        M = np.zeros((3, 3), dtype=np.float64)
        for pi in sel_idx:
            A = pairs[int(pi)]["A"]
            B = pairs[int(pi)]["B"]
            w = float(pairs[int(pi)]["weight"])

            a, _ = cv2.Rodrigues(A[:3, :3])
            b, _ = cv2.Rodrigues(B[:3, :3])
            av = a.reshape(3)
            bv = b.reshape(3)
            M += w * np.outer(av, bv)

        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # --- Translation: (R_A - I) t = R t_B - t_A ---
        L_list = []
        r_list = []
        for pi in sel_idx:
            A = pairs[int(pi)]["A"]
            B = pairs[int(pi)]["B"]
            w = np.sqrt(float(pairs[int(pi)]["weight"]))

            L = A[:3, :3] - np.eye(3, dtype=np.float64)
            r = R @ B[:3, 3] - A[:3, 3]

            L_list.append(w * L)
            r_list.append(w * r)

        L_all = np.concatenate(L_list, axis=0)
        r_all = np.concatenate(r_list, axis=0)

        t, *_ = np.linalg.lstsq(L_all, r_all, rcond=None)

        X = np.eye(4, dtype=np.float64)
        X[:3, :3] = R
        X[:3, 3] = t.reshape(3)
        return X

    X = solve_from_indices(idx)
    inlier_idx = idx.copy()

    for _ in range(max_iters):
        residual_metric = []
        for pi in inlier_idx:
            A = pairs[int(pi)]["A"]
            B = pairs[int(pi)]["B"]
            E = inv_T(A @ X) @ (X @ B)
            t_mm = float(np.linalg.norm(E[:3, 3]) * 1000.0)
            r_deg = rotation_error_deg(E[:3, :3], np.eye(3, dtype=np.float64))
            residual_metric.append(t_mm + 20.0 * r_deg)

        residual_metric = np.asarray(residual_metric, dtype=np.float64)
        med = np.median(residual_metric)
        mad = np.median(np.abs(residual_metric - med)) + 1e-12
        thr = med + k_mad * 1.4826 * mad

        keep = residual_metric <= thr
        if keep.sum() < max(6, int(0.35 * len(inlier_idx))):
            break

        new_inlier_idx = inlier_idx[keep]
        if len(new_inlier_idx) == len(inlier_idx):
            break

        inlier_idx = new_inlier_idx
        X = solve_from_indices(inlier_idx)

    # final residuals
    trans_mm = []
    rot_deg = []
    for pi in inlier_idx:
        A = pairs[int(pi)]["A"]
        B = pairs[int(pi)]["B"]
        E = inv_T(A @ X) @ (X @ B)
        trans_mm.append(float(np.linalg.norm(E[:3, 3]) * 1000.0))
        rot_deg.append(rotation_error_deg(E[:3, :3], np.eye(3, dtype=np.float64)))

    stats = {
        "num_pairs": int(len(pairs)),
        "num_inliers": int(len(inlier_idx)),
        "inlier_ratio": float(len(inlier_idx) / max(len(pairs), 1)),
        "motion_eq_trans_mm_mean": float(np.mean(trans_mm)) if trans_mm else 0.0,
        "motion_eq_trans_mm_std": float(np.std(trans_mm)) if trans_mm else 0.0,
        "motion_eq_rot_deg_mean": float(np.mean(rot_deg)) if rot_deg else 0.0,
        "motion_eq_rot_deg_std": float(np.std(rot_deg)) if rot_deg else 0.0,
    }

    return X, stats, [int(x) for x in inlier_idx.tolist()]


def main():
    parser = argparse.ArgumentParser(description="Calibrate with robot-grasped cube (new version)")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: <root_folder>/calib_out_grasped_cube")

    parser.add_argument("--gripper_cam_idx", type=int, default=None,
                        help="Camera idx of gripper cam. Excluded from fixed cams.")
    parser.add_argument("--fixed_cam_idxs", type=str, default="",
                        help="Optional fixed cam idx CSV (e.g. 1,2,3). Empty=auto")
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=None,
                        help="Reference fixed camera for T_Cref_Ci output")

    parser.add_argument("--robot_poses_file", type=str, default=None,
                        help="Optional robot pose JSON (merged with meta poses)")

    parser.add_argument("--min_markers", type=int, default=1)
    parser.add_argument("--reproj_max_px", type=float, default=6.0)
    parser.add_argument("--use_ransac", action="store_true", default=True)

    parser.add_argument("--pair_mode", choices=["consecutive", "all"], default="consecutive")
    parser.add_argument("--min_motion_rot_deg", type=float, default=2.0,
                        help="Minimum relative rotation per pair (deg)")

    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    out_dir = args.out_dir or os.path.join(root, "calib_out_grasped_cube")
    ensure_dir(out_dir)

    meta_path = os.path.join(root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

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

    fixed_cam_ids = parse_cam_idx_csv(args.fixed_cam_idxs)
    if len(fixed_cam_ids) == 0:
        fixed_cam_ids = [ci for ci in all_cam_ids if ci != gripper_cam_idx]
    fixed_cam_ids = sorted(set(fixed_cam_ids))
    if len(fixed_cam_ids) == 0:
        raise RuntimeError("No fixed cameras selected")

    ref_fixed = args.ref_fixed_cam_idx
    if ref_fixed is None:
        ref_fixed = fixed_cam_ids[0]
    if ref_fixed not in fixed_cam_ids:
        raise RuntimeError(f"ref_fixed_cam_idx={ref_fixed} not in fixed cams={fixed_cam_ids}")

    print(f"[INFO] all cams: {all_cam_ids}")
    print(f"[INFO] gripper cam idx: {gripper_cam_idx}")
    print(f"[INFO] fixed cams: {fixed_cam_ids}")
    print(f"[INFO] ref fixed cam: {ref_fixed}")

    # load intrinsics for fixed cams
    K_map, D_map = {}, {}
    for ci in fixed_cam_ids:
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
        raise RuntimeError("No robot poses available")

    # fixed-cam PnP per event
    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)
    pnp_by_cam: Dict[int, Dict[int, dict]] = {ci: {} for ci in fixed_cam_ids}

    print("\n" + "=" * 60)
    print("[STEP-1] Fixed camera PnP (cube pose)")
    print("=" * 60)

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue

        for ci in fixed_cam_ids:
            cinfo = cap.get("cams", {}).get(str(ci))
            if not cinfo or not cinfo.get("saved"):
                continue

            rgb_rel = cinfo.get("rgb_path")
            if not rgb_rel:
                continue
            rgb_abs = os.path.join(root, rgb_rel)
            img = cv2.imread(rgb_abs)
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

            pnp_by_cam[ci][eid] = {
                "T_C_O": rodrigues_to_Rt(rvec, tvec),
                "err_mean": float(reproj["err_mean"]),
                "n_points": int(reproj["n_points"]),
                "used_ids": [int(x) for x in used],
            }

    for ci in fixed_cam_ids:
        errs = [x["err_mean"] for x in pnp_by_cam[ci].values()]
        mean_err = float(np.mean(errs)) if errs else float("nan")
        print(f"[INFO] cam{ci}: valid={len(pnp_by_cam[ci])}  reproj_mean={mean_err:.3f}px")

    # solve T_grip_cube from AX=XB motions
    print("\n" + "=" * 60)
    print("[STEP-2] Solve T_grip_cube via motion equations")
    print("=" * 60)

    motion_pairs = build_motion_pairs(
        robot_T_per_event=robot_T_per_event,
        pnp_by_cam=pnp_by_cam,
        fixed_cam_ids=fixed_cam_ids,
        pair_mode=args.pair_mode,
        min_motion_rot_deg=args.min_motion_rot_deg,
    )
    print(f"[INFO] motion pairs: {len(motion_pairs)}")

    T_grip_cube, grip_cube_stats, inlier_pair_idx = solve_grip_cube_from_pairs(motion_pairs)
    np.save(os.path.join(out_dir, "T_grip_cube.npy"), T_grip_cube)
    print(f"[SAVE] {os.path.join(out_dir, 'T_grip_cube.npy')}")
    print(f"[INFO] T_grip_cube motion residual: "
          f"trans_mean={grip_cube_stats['motion_eq_trans_mm_mean']:.2f}mm, "
          f"rot_mean={grip_cube_stats['motion_eq_rot_deg_mean']:.3f}deg")

    # event-wise base cube from FK
    T_base_cube_by_event = {}
    for eid, T_B_G in robot_T_per_event.items():
        T_base_cube_by_event[eid] = T_B_G @ T_grip_cube

    # solve fixed camera in base
    print("\n" + "=" * 60)
    print("[STEP-3] Solve T_base_Ci (only 1 PnP term)")
    print("=" * 60)

    T_base_Ci: Dict[int, np.ndarray] = {}
    base_stats: Dict[str, dict] = {}
    consistency_stats: Dict[str, dict] = {}

    for ci in fixed_cam_ids:
        common = sorted(set(robot_T_per_event.keys()) & set(pnp_by_cam[ci].keys()))
        if len(common) == 0:
            print(f"[WARN] cam{ci}: no common events")
            continue

        Ts = []
        ws = []
        for eid in common:
            T_base_cube = T_base_cube_by_event[eid]
            T_Ci_cube = pnp_by_cam[ci][eid]["T_C_O"]
            T_base_C = T_base_cube @ inv_T(T_Ci_cube)
            Ts.append(T_base_C)
            ws.append(1.0 / max(pnp_by_cam[ci][eid]["err_mean"], 1e-9))

        T_avg, st = robust_weighted_se3_average(Ts, ws, return_stats=True)
        T_base_Ci[ci] = T_avg
        base_stats[f"T_base_C{ci}"] = st

        out_npy = os.path.join(out_dir, f"T_base_C{ci}.npy")
        np.save(out_npy, T_avg)
        print(f"[SAVE] {out_npy}  frames={len(common)}  "
              f"rot_std={st['rotation_std_deg']:.3f}deg  trans_std={st['translation_std_mm']:.2f}mm")

        # consistency: FK cube vs camera-derived cube
        trans_mm = []
        rot_deg = []
        for eid in common:
            T_fk = T_base_cube_by_event[eid]
            T_cam = T_avg @ pnp_by_cam[ci][eid]["T_C_O"]
            trans_mm.append(float(np.linalg.norm(T_fk[:3, 3] - T_cam[:3, 3]) * 1000.0))
            rot_deg.append(rotation_error_deg(T_fk[:3, :3], T_cam[:3, :3]))

        consistency_stats[f"cam{ci}"] = {
            "num_events": int(len(common)),
            "cube_consistency_trans_mm_mean": float(np.mean(trans_mm)),
            "cube_consistency_trans_mm_max": float(np.max(trans_mm)),
            "cube_consistency_rot_deg_mean": float(np.mean(rot_deg)),
            "cube_consistency_rot_deg_max": float(np.max(rot_deg)),
        }

    if ref_fixed not in T_base_Ci:
        raise RuntimeError(f"ref_fixed_cam_idx={ref_fixed} transform was not estimated")

    # derive T_Cref_Ci for compatibility
    T_Cref_Ci = {ref_fixed: np.eye(4, dtype=np.float64)}
    T_base_Cref = T_base_Ci[ref_fixed]
    for ci, T_base_C in T_base_Ci.items():
        if ci == ref_fixed:
            continue
        T_Cref_C = inv_T(T_base_Cref) @ T_base_C
        T_Cref_Ci[ci] = T_Cref_C
        np.save(os.path.join(out_dir, f"T_C{ref_fixed}_C{ci}.npy"), T_Cref_C)
        print(f"[SAVE] {os.path.join(out_dir, f'T_C{ref_fixed}_C{ci}.npy')}")

    # summary
    summary = {
        "mode": "robot_grasped_cube",
        "formula": "T_base_Ci = (T_base_grip * T_grip_cube) * inv(T_Ci_cube)",
        "gripper_cam_idx": None if gripper_cam_idx is None else int(gripper_cam_idx),
        "fixed_cam_ids": [int(x) for x in fixed_cam_ids],
        "ref_fixed_cam_idx": int(ref_fixed),
        "num_robot_poses": int(len(robot_T_per_event)),
        "num_motion_pairs": int(len(motion_pairs)),
        "num_motion_pair_inliers": int(len(inlier_pair_idx)),
        "diagnostics": {
            "grip_cube_solve": grip_cube_stats,
            "base_cam_stability": base_stats,
            "cube_fk_vs_cam_consistency": consistency_stats,
        },
        "transforms": {},
    }

    summary["transforms"]["T_grip_cube"] = T_grip_cube.reshape(-1).tolist()
    for ci, T in T_base_Ci.items():
        summary["transforms"][f"T_base_C{ci}"] = T.reshape(-1).tolist()
    for ci, T in T_Cref_Ci.items():
        summary["transforms"][f"T_C{ref_fixed}_C{ci}"] = T.reshape(-1).tolist()

    # save selected per-event base cube poses for debug
    ev_keys = sorted(T_base_cube_by_event.keys())
    per_event = {
        str(int(eid)): T_base_cube_by_event[eid].reshape(-1).tolist()
        for eid in ev_keys
    }
    with open(os.path.join(out_dir, "T_base_cube_per_event.json"), "w") as f:
        json.dump(per_event, f, indent=2)
    summary_path = os.path.join(out_dir, "calibration_summary_grasped_cube.json")
    summary_compat_path = os.path.join(out_dir, "calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_compat_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVE] {summary_path}")
    print(f"[SAVE] {summary_compat_path}")
    print("\n" + "=" * 60)
    print("Step3 grasped-cube calibration COMPLETE")
    print("=" * 60)
    print(f"  output dir: {out_dir}")
    print("  saved transforms:")
    for k in summary["transforms"].keys():
        print(f"    - {k}")


if __name__ == "__main__":
    main()
