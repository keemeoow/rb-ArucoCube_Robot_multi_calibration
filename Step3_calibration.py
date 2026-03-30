# Step3_calibration.py
"""
Step 3: 통합 캘리브레이션 (ChArUco hand-eye + ArUco cube multi-camera).

Step2_capture_capture.py의 meta.json을 입력으로 받아:
  A) 메타데이터에서 cube PnP 읽기 (고정 + 그리퍼 카메라)
  B) 고정 카메라 간 상대 위치 계산 (cube PnP)
  C) Hand-eye: 그리퍼 카메라 ChArUco 보드 데이터 사용
  D) 고정 카메라의 로봇 베이스 좌표 계산

출력:
  T_gripper_cam.npy    - 그리퍼 -> 카메라
  T_base_O.npy         - 로봇 베이스 -> 큐브 평균 위치
  T_base_C{i}.npy      - 로봇 베이스 -> 고정 카메라
  T_C{ref}_C{i}.npy    - 기준 카메라 -> 다른 고정 카메라

실행:
  python Step3_calibration.py \
    --root_folder ./data/session \
    --intrinsics_dir ./intrinsics
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Any

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


def weighted_se3_average(T_list, w_list=None):
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


def robust_weighted_se3_average(T_list, w_list=None, k_mad=2.5, max_iters=3,
                                 return_stats=False):
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

    trans_devs, rot_devs = [], []
    for T in [T_list[i] for i in idx]:
        trans_devs.append(float(np.linalg.norm(T[:3, 3] - T_final[:3, 3]) * 1000.0))
        rot_devs.append(rotation_error_deg(T[:3, :3], T_final[:3, :3]))

    stats = {
        "num_frames": int(len(T_list)),
        "num_inliers": int(len(idx)),
        "inlier_ratio": float(len(idx) / max(len(T_list), 1)),
        "translation_std_mm": float(np.std(trans_devs)) if trans_devs else 0.0,
        "rotation_std_deg": float(np.std(rot_devs)) if rot_devs else 0.0,
    }
    return T_final, stats


def try_parse_pose6(obj):
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
                return [float(obj["x"]), float(obj["y"]), float(obj["z"]),
                        float(obj["rz"]), float(obj["ry"]), float(obj["rx"])]
            except Exception:
                return None
        for k in ["robot_pose_6dof", "tcp_pose_6dof", "pose_6dof", "pose"]:
            if k in obj:
                p = try_parse_pose6(obj[k])
                if p is not None:
                    return p
    return None


def try_parse_T44(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        arr = np.asarray(obj, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
    if isinstance(obj, dict):
        for k in ["T_B_G", "robot_pose_matrix_4x4", "matrix", "transform"]:
            if k in obj:
                T = try_parse_T44(obj[k])
                if T is not None:
                    return T
    return None


def load_robot_poses_from_meta(meta):
    out = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        T = None
        if "robot_pose_matrix_4x4" in cap:
            T = try_parse_T44(cap.get("robot_pose_matrix_4x4"))
        if T is None and "capture_pose_matrix_4x4" in cap:
            T = try_parse_T44(cap.get("capture_pose_matrix_4x4"))
        if T is None:
            p6 = try_parse_pose6(cap.get("robot_pose_6dof"))
            if p6 is None:
                p6 = try_parse_pose6(cap.get("capture_pose_6dof"))
            if p6 is not None:
                T = euler_deg_to_matrix(*p6)
        if T is not None:
            out[eid] = T.astype(np.float64)
    return out


def build_method_map():
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
        methods = {"TSAI": 0, "PARK": 1, "HORAUD": 2, "ANDREFF": 3, "DANIILIDIS": 4}
    return methods


def main():
    parser = argparse.ArgumentParser(description="Unified calibration (ChArUco hand-eye + cube multi-cam)")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gripper_cam_idx", type=int, default=None)
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=None)
    parser.add_argument("--handeye_method", type=str, default="AUTO")
    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    out_dir = args.out_dir or os.path.join(root, "calib_out")
    ensure_dir(out_dir)

    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)

    # ─── Camera discovery ───
    all_cam_ids = sorted({
        int(k) for cap in meta.get("captures", [])
        for k, v in cap.get("cams", {}).items() if v.get("saved")
    })
    if not all_cam_ids:
        raise RuntimeError("No saved camera data in meta.json")

    gripper_cam_idx = args.gripper_cam_idx
    if gripper_cam_idx is None:
        gripper_cam_idx = meta.get("gripper_cam_idx")
    if gripper_cam_idx is None:
        dm_path = os.path.join(intr_dir, "device_map.json")
        if os.path.exists(dm_path):
            with open(dm_path, "r") as f:
                gripper_cam_idx = json.load(f).get("gripper_cam_idx")
    if gripper_cam_idx is None:
        raise RuntimeError("gripper_cam_idx required")

    fixed_cam_ids = [ci for ci in all_cam_ids if ci != gripper_cam_idx]
    ref_fixed = args.ref_fixed_cam_idx or (fixed_cam_ids[0] if fixed_cam_ids else None)

    print(f"[INFO] all cams: {all_cam_ids}")
    print(f"[INFO] gripper: cam{gripper_cam_idx}, fixed: {fixed_cam_ids}, ref: cam{ref_fixed}")

    # intrinsics
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics(intr_dir, ci)

    # robot poses
    robot_T = load_robot_poses_from_meta(meta)
    print(f"[INFO] robot poses: {len(robot_T)}")

    # ══════════════════════════════════════════════════════════
    # STEP A: Read cube PnP from metadata (all cameras)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[STEP-A] Cube PnP from metadata")
    print("=" * 60)

    pnp_obs: Dict[int, Dict[int, dict]] = {ci: {} for ci in all_cam_ids}
    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in pnp_obs or not cinfo.get("saved"):
                continue

            cpnp = cinfo.get("cube_pnp")
            if cpnp and cpnp.get("ok"):
                err = float(cpnp.get("reproj_mean_px", 99.0))
                # Gripper camera: more lenient (views cube from above, multi-face)
                # Fixed cameras: strict (reject oblique false matches)
                max_err = 5.0 if ci == gripper_cam_idx else 3.0
                if err > max_err:
                    continue
                T44 = cpnp.get("T_cam_cube_4x4")
                if T44 is not None:
                    pnp_obs[ci][eid] = {
                        "T_C_O": np.asarray(T44, dtype=np.float64),
                        "err_mean": err,
                        "n_points": int(cpnp.get("n_points", 4)),
                        "used_ids": cpnp.get("used_ids", []),
                    }
                    continue

            # Fallback: re-detect from image, try per-marker PnP
            rgb_path = os.path.join(root, cinfo.get("rgb_path", ""))
            img = cv2.imread(rgb_path)
            if img is None:
                continue
            thr = 5.0 if ci == gripper_cam_idx else 3.0
            asp = 0.0 if ci == gripper_cam_idx else 0.3

            # First try all markers together
            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img, K_map[ci], D_map[ci],
                use_ransac=True, min_markers=1,
                reproj_thr_mean_px=thr, return_reproj=True,
                min_aspect=asp)
            if ok and reproj and reproj["err_mean"] <= thr:
                pnp_obs[ci][eid] = {
                    "T_C_O": rodrigues_to_Rt(rvec, tvec),
                    "err_mean": float(reproj["err_mean"]),
                    "n_points": int(reproj["n_points"]),
                    "used_ids": [int(x) for x in used],
                }
                continue

            # Per-marker PnP: try each marker, store all IPPE solutions
            corners_list, ids = cube.detect(img)
            if ids is None:
                continue
            candidates = []  # [(T_C_O, err, marker_id), ...]
            for c, mid in zip(corners_list, ids):
                mid = int(mid)
                if mid not in cube.cfg.id_to_face:
                    continue
                obj = cube.model.marker_corners_in_rig(mid)
                img_pts = c.reshape(4, 2).astype(np.float64)
                if mid == 3:
                    img_pts = img_pts[[1, 2, 3, 0]]
                if asp > 0:
                    ew = np.linalg.norm(img_pts[1] - img_pts[0])
                    eh = np.linalg.norm(img_pts[3] - img_pts[0])
                    if min(ew, eh) / (max(ew, eh) + 1e-6) < asp:
                        continue
                # Get ALL solutions (IPPE returns 2)
                n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                    obj.reshape(-1, 1, 3), img_pts.reshape(-1, 1, 2),
                    K_map[ci], D_map[ci], flags=cv2.SOLVEPNP_IPPE)
                for si in range(n_sol):
                    err_val = float(reproj_errs[si][0]) if reproj_errs is not None else 99.0
                    if err_val <= thr:
                        T_sol = rodrigues_to_Rt(rvecs[si], tvecs[si])
                        candidates.append((T_sol, err_val, mid))

            if candidates:
                # Store all candidates; Step D will pick the right one
                # using T_base_O rotation from gripper camera
                best = min(candidates, key=lambda x: x[1])
                pnp_obs[ci][eid] = {
                    "T_C_O": best[0],
                    "err_mean": best[1],
                    "n_points": 4,
                    "used_ids": [best[2]],
                    "_candidates": candidates,  # all IPPE solutions
                }

    for ci in all_cam_ids:
        errs = [r["err_mean"] for r in pnp_obs[ci].values()]
        tag = "G" if ci == gripper_cam_idx else "F"
        print(f"  cam{ci}({tag}): {len(pnp_obs[ci])} frames, "
              f"reproj={np.mean(errs):.3f}px" if errs else f"  cam{ci}({tag}): 0 frames")

    # ══════════════════════════════════════════════════════════
    # STEP A-2: Read ChArUco board from gripper camera metadata
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[STEP-A2] ChArUco board from gripper camera")
    print("=" * 60)

    charuco_obs: Dict[int, dict] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        gi_data = cap.get("cams", {}).get(str(gripper_cam_idx), {})
        ch = gi_data.get("charuco")
        if ch and ch.get("ok"):
            T44 = ch.get("T_cam_board_4x4")
            if T44 is not None:
                charuco_obs[eid] = {
                    "T_cam_board": np.asarray(T44, dtype=np.float64),
                    "reproj": float(ch.get("reproj_error_px", 1.0)),
                    "n_corners": int(ch.get("n_corners", 0)),
                }

    # Fallback: detect ChArUco from saved gripper camera images
    if len(charuco_obs) == 0:
        print("  No ChArUco in metadata, detecting from saved images...")
        from charuco_utils import CharucoTarget
        from config import CharucoBoardConfig
        charuco_det = CharucoTarget(CharucoBoardConfig())
        g_K, g_D = K_map[gripper_cam_idx], D_map[gripper_cam_idx]

        for cap in meta.get("captures", []):
            eid = int(cap.get("event_id", -1))
            if eid < 0:
                continue
            gi_data = cap.get("cams", {}).get(str(gripper_cam_idx), {})
            rgb_rel = gi_data.get("rgb_path", "")
            if not rgb_rel:
                continue
            img = cv2.imread(os.path.join(root, rgb_rel))
            if img is None:
                continue
            try:
                ch_ok, ch_rvec, ch_tvec, ch_n, ch_reproj = charuco_det.estimate_pose(
                    img, g_K, g_D)
            except Exception as e:
                print(f"    event={eid}: ERROR {e}")
                continue
            if ch_ok and ch_rvec is not None and ch_n >= 4:
                charuco_obs[eid] = {
                    "T_cam_board": rodrigues_to_Rt(ch_rvec, ch_tvec),
                    "reproj": float(ch_reproj) if ch_reproj else 1.0,
                    "n_corners": int(ch_n),
                }
                print(f"    event={eid}: OK {ch_n} corners, reproj={ch_reproj:.3f}px")
            else:
                print(f"    event={eid}: no board (corners={ch_n if ch_ok else 0})")

    ch_reprs = [v["reproj"] for v in charuco_obs.values()]
    if ch_reprs:
        print(f"  ChArUco total: {len(charuco_obs)} frames, reproj={np.mean(ch_reprs):.3f}px")
    else:
        print(f"  ChArUco: 0 frames (will fallback to cube PnP for hand-eye)")

    # ══════════════════════════════════════════════════════════
    # STEP B: Fixed camera extrinsics (cube PnP)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"[STEP-B] Fixed camera extrinsics (ref=cam{ref_fixed})")
    print("=" * 60)

    T_Cref_Ci = {ref_fixed: np.eye(4, dtype=np.float64)}
    fixed_stats = {}

    for ci in fixed_cam_ids:
        if ci == ref_fixed:
            continue
        common = sorted(set(pnp_obs[ref_fixed].keys()) & set(pnp_obs[ci].keys()))
        if not common:
            print(f"  [WARN] cam{ci}: no common frames with ref")
            continue

        Ts, ws = [], []
        for eid in common:
            T_ref_O = pnp_obs[ref_fixed][eid]["T_C_O"]
            T_ci_O = pnp_obs[ci][eid]["T_C_O"]
            T_ref_ci = T_ref_O @ inv_T(T_ci_O)
            w = 1.0 / max(pnp_obs[ref_fixed][eid]["err_mean"] * pnp_obs[ci][eid]["err_mean"], 1e-9)
            Ts.append(T_ref_ci)
            ws.append(w)

        T_avg, st = robust_weighted_se3_average(Ts, ws, return_stats=True)
        T_Cref_Ci[ci] = T_avg
        fixed_stats[f"T_C{ref_fixed}_C{ci}"] = st
        np.save(os.path.join(out_dir, f"T_C{ref_fixed}_C{ci}.npy"), T_avg)
        print(f"  T_C{ref_fixed}_C{ci}: {len(common)}fr "
              f"rot={st['rotation_std_deg']:.3f}deg trans={st['translation_std_mm']:.2f}mm")

    # ══════════════════════════════════════════════════════════
    # STEP C: Hand-eye (ChArUco preferred, cube PnP fallback)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"[STEP-C] Hand-eye (gripper=cam{gripper_cam_idx})")
    print("=" * 60)

    # Choose data source: ChArUco (preferred) or cube PnP (fallback)
    use_charuco = len(charuco_obs) >= 5
    if use_charuco:
        common_he = sorted(set(robot_T.keys()) & set(charuco_obs.keys()))
        print(f"  Using ChArUco board ({len(common_he)} common events)")
        R_target2cam = [charuco_obs[eid]["T_cam_board"][:3, :3] for eid in common_he]
        t_target2cam = [charuco_obs[eid]["T_cam_board"][:3, 3].reshape(3, 1) for eid in common_he]
        w_he = [1.0 / max(charuco_obs[eid]["reproj"], 1e-9) for eid in common_he]
    else:
        common_he = sorted(set(robot_T.keys()) & set(pnp_obs[gripper_cam_idx].keys()))
        print(f"  Fallback: cube PnP ({len(common_he)} common events)")
        R_target2cam = [pnp_obs[gripper_cam_idx][eid]["T_C_O"][:3, :3] for eid in common_he]
        t_target2cam = [pnp_obs[gripper_cam_idx][eid]["T_C_O"][:3, 3].reshape(3, 1) for eid in common_he]
        w_he = [1.0 / max(pnp_obs[gripper_cam_idx][eid]["err_mean"], 1e-9) for eid in common_he]

    if len(common_he) < 5:
        raise RuntimeError(f"Not enough events for hand-eye ({len(common_he)} < 5)")

    R_gripper2base = [robot_T[eid][:3, :3] for eid in common_he]
    t_gripper2base = [robot_T[eid][:3, 3].reshape(3, 1) for eid in common_he]

    method_map = build_method_map()
    method_sel = str(args.handeye_method or "AUTO").strip().upper()
    method_iter = method_map.items() if method_sel == "AUTO" else [(method_sel, method_map.get(method_sel))]

    method_results = {}
    for mname, mcode in method_iter:
        if mcode is None:
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

            # Consistency: T_base_target should be constant
            T_B_tgt_list = []
            for eid in common_he:
                T_B_G = robot_T[eid]
                if use_charuco:
                    T_cam_tgt = charuco_obs[eid]["T_cam_board"]
                else:
                    T_cam_tgt = pnp_obs[gripper_cam_idx][eid]["T_C_O"]
                T_B_tgt_list.append(T_B_G @ T_gTc @ T_cam_tgt)

            T_B_tgt_avg, st_bo = robust_weighted_se3_average(T_B_tgt_list, w_he, return_stats=True)

            trans_mm, rot_deg = [], []
            for T in T_B_tgt_list:
                trans_mm.append(float(np.linalg.norm(T[:3, 3] - T_B_tgt_avg[:3, 3]) * 1000.0))
                rot_deg.append(rotation_error_deg(T[:3, :3], T_B_tgt_avg[:3, :3]))

            score = float(np.mean(trans_mm)) + 10.0 * float(np.mean(rot_deg))
            method_results[mname] = {
                "T_gTc": T_gTc,
                "score": score,
                "mean_trans_mm": float(np.mean(trans_mm)),
                "mean_rot_deg": float(np.mean(rot_deg)),
                "stability": st_bo,
            }
            print(f"  [{mname}] score={score:.3f} trans={np.mean(trans_mm):.2f}mm rot={np.mean(rot_deg):.3f}deg")
        except Exception as e:
            print(f"  [{mname}] FAILED: {e}")

    if not method_results:
        raise RuntimeError("All hand-eye methods failed")

    best_method = min(method_results, key=lambda k: method_results[k]["score"])
    T_gTc = method_results[best_method]["T_gTc"]

    np.save(os.path.join(out_dir, "T_gripper_cam.npy"), T_gTc)
    print(f"  [BEST] {best_method} -> T_gripper_cam.npy")

    # ══════════════════════════════════════════════════════════
    # STEP C-2: Compute T_base_board (from hand-eye + ChArUco)
    # ══════════════════════════════════════════════════════════
    T_base_board_list, w_bb = [], []
    for eid in common_he:
        if eid not in charuco_obs:
            continue
        T_B_G = robot_T[eid]
        T_cam_board = charuco_obs[eid]["T_cam_board"]
        T_base_board_list.append(T_B_G @ T_gTc @ T_cam_board)
        w_bb.append(1.0 / max(charuco_obs[eid]["reproj"], 1e-9))

    T_base_board = weighted_se3_average(T_base_board_list, w_bb) if T_base_board_list else None
    if T_base_board is not None:
        ts = np.array([T[:3, 3] for T in T_base_board_list])
        print(f"  T_base_board: {len(T_base_board_list)} frames, "
              f"pos_std=[{np.std(ts[:,0])*1000:.1f},{np.std(ts[:,1])*1000:.1f},{np.std(ts[:,2])*1000:.1f}]mm")

    # Compute T_base_O from gripper camera cube PnP
    common_cube = sorted(set(robot_T.keys()) & set(pnp_obs[gripper_cam_idx].keys()))
    T_B_O_by_event = {}
    T_B_O_list, w_bo = [], []
    for eid in common_cube:
        T_B_G = robot_T[eid]
        T_Cg_O = pnp_obs[gripper_cam_idx][eid]["T_C_O"]
        T_B_O = T_B_G @ T_gTc @ T_Cg_O
        T_B_O_by_event[eid] = T_B_O
        T_B_O_list.append(T_B_O)
        w_bo.append(1.0 / max(pnp_obs[gripper_cam_idx][eid]["err_mean"], 1e-9))

    T_B_O_avg = weighted_se3_average(T_B_O_list, w_bo) if T_B_O_list else np.eye(4)
    np.save(os.path.join(out_dir, "T_base_O.npy"), T_B_O_avg)
    print(f"  T_base_O.npy ({len(common_cube)} events)")

    # ══════════════════════════════════════════════════════════
    # STEP D: Fixed cameras in robot base frame (cube PnP chaining)
    # T_base_Ci = T_B_O[eid] @ inv(T_Ci_O[eid])
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[STEP-D] Fixed cameras in robot base frame (cube PnP)")
    print("=" * 60)

    T_base_Ci = {}
    base_stats = {}

    def _pick_candidate(ci, eid, T_B_O, T_ref=None):
        """Pick best IPPE solution for a fixed camera observation.
        T_ref: approximate T_base_Ci from pass 1 (if available)."""
        candidates = pnp_obs[ci][eid].get("_candidates")
        if not candidates or len(candidates) <= 1:
            return pnp_obs[ci][eid]["T_C_O"]
        best_T, best_score = None, 1e9
        for T_sol, err_sol, _ in candidates:
            T_B_Ci_sol = T_B_O @ inv_T(T_sol)
            if T_ref is not None:
                score = rotation_error_deg(T_B_Ci_sol[:3, :3], T_ref[:3, :3])
            else:
                cam_z = T_B_Ci_sol[:3, 2]
                score = err_sol + max(cam_z[2], 0.0) * 50.0
            if score < best_score:
                best_score = score
                best_T = T_sol
        return best_T

    for ci in fixed_cam_ids:
        common = sorted(set(pnp_obs[ci].keys()) & set(T_B_O_by_event.keys()))
        if not common:
            print(f"  [WARN] cam{ci}: no overlap")
            continue

        # Pass 1: rough estimate (z-axis heuristic)
        Ts1, ws1 = [], []
        for eid in common:
            T_C_O = _pick_candidate(ci, eid, T_B_O_by_event[eid])
            Ts1.append(T_B_O_by_event[eid] @ inv_T(T_C_O))
            ws1.append(1.0 / max(pnp_obs[ci][eid]["err_mean"], 1e-9))
        T_rough = robust_weighted_se3_average(Ts1, ws1)

        # Pass 2: refine with rotation consistency
        Ts2, ws2 = [], []
        for eid in common:
            T_C_O = _pick_candidate(ci, eid, T_B_O_by_event[eid], T_ref=T_rough)
            Ts2.append(T_B_O_by_event[eid] @ inv_T(T_C_O))
            ws2.append(1.0 / max(pnp_obs[ci][eid]["err_mean"], 1e-9))

        T_avg, st = robust_weighted_se3_average(Ts2, ws2, return_stats=True)
        T_base_Ci[ci] = T_avg
        base_stats[f"T_base_C{ci}"] = st
        np.save(os.path.join(out_dir, f"T_base_C{ci}.npy"), T_avg)
        print(f"  T_base_C{ci}: {len(Ts2)}fr "
              f"rot={st['rotation_std_deg']:.3f}deg trans={st['translation_std_mm']:.2f}mm")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    summary = {
        "calibration_type": "unified_charuco_cube",
        "handeye_data_source": "charuco" if use_charuco else "cube_pnp",
        "gripper_cam_idx": int(gripper_cam_idx),
        "ref_fixed_cam_idx": int(ref_fixed) if ref_fixed is not None else None,
        "fixed_cam_ids": [int(x) for x in fixed_cam_ids],
        "all_cam_ids": [int(x) for x in all_cam_ids],
        "selected_handeye_method": best_method,
        "num_robot_poses": len(robot_T),
        "num_handeye_events": len(common_he),
        "num_charuco_frames": len(charuco_obs),
        "num_cube_pnp_gripper": len(pnp_obs.get(gripper_cam_idx, {})),
        "diagnostics": {
            "fixed_extrinsics": fixed_stats,
            "handeye_methods": {
                k: {"score": v["score"], "mean_trans_mm": v["mean_trans_mm"],
                     "mean_rot_deg": v["mean_rot_deg"], "stability": v["stability"]}
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

    print(f"\n{'=' * 60}")
    print("Calibration COMPLETE")
    print(f"{'=' * 60}")
    print(f"  source: {'ChArUco board' if use_charuco else 'cube PnP'} ({best_method})")
    print(f"  output: {out_dir}")
    for k in summary["transforms"]:
        print(f"    {k}")


if __name__ == "__main__":
    main()
