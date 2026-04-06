import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt
from config import CubeConfig
from cube_config_utils import (
    load_cube_config_from_calibration_summary,
    load_cube_config_from_json_file,
    load_cube_config_from_meta,
    load_fixed_cube_config,
    load_preferred_cube_config,
)
from robot_comm import euler_deg_to_matrix


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def candidate_face_count(cand: dict) -> int:
    used_ids = cand.get("used_ids", [])
    return max(len(set(int(x) for x in used_ids)), 1)


def candidate_face_weight(cand: dict, single_face_weight: float = 0.35) -> float:
    return float(single_face_weight) if candidate_face_count(cand) <= 1 else 1.0


def cube_candidate_rank(cand: dict) -> Tuple[float, float, int]:
    used_ids = cand.get("used_ids", [])
    err = float(cand.get("err_mean", 99.0))
    source = str(cand.get("source", "unknown"))
    source_prio = {
        "multi": 0,
        "meta": 1,
        "ippe0": 2,
        "ippe1": 3,
    }.get(source, 9)
    return (-len(set(int(x) for x in used_ids)), err, source_prio)


def cube_selection_profile_kwargs(profile: str = "default",
                                  cube_only_single_face_weight: float = 0.35,
                                  cube_only_single_face_penalty: float = 0.75) -> Dict[str, float]:
    if str(profile) == "cube_only_specialized":
        return {
            "single_face_weight": float(cube_only_single_face_weight),
            "single_face_penalty": float(cube_only_single_face_penalty),
        }
    return {
        "single_face_weight": 1.0,
        "single_face_penalty": 0.0,
    }


def select_primary_cube_candidate(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None
    return min(candidates, key=cube_candidate_rank)


def resolve_cube_config_for_run(root_folder: str,
                                calib_dir: Optional[str] = None,
                                cube_config_json: Optional[str] = None,
                                default_cfg: Optional[CubeConfig] = None) -> Tuple[CubeConfig, str]:
    cfg_template = default_cfg or CubeConfig()
    if cube_config_json:
        cfg, source = load_cube_config_from_json_file(cube_config_json, cfg_template)
        if cfg is not None:
            return cfg, source
    cfg, source, _ = load_fixed_cube_config(cfg_template)
    if cfg is not None:
        return cfg, source
    cfg, source, _ = load_preferred_cube_config(root_folder, cfg_template)
    if cfg is not None:
        return cfg, source
    if calib_dir:
        cfg, source = load_cube_config_from_calibration_summary(calib_dir, cfg_template)
        if cfg is not None:
            return cfg, source
    cfg, source = load_cube_config_from_meta(root_folder, cfg_template)
    return cfg, source


def load_intrinsics_color(intrinsics_dir: str, cam_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(os.path.join(intrinsics_dir, f"cam{cam_idx}.npz"), allow_pickle=True)
    return data["color_K"].astype(np.float64), data["color_D"].astype(np.float64)


def load_intrinsics_with_depth_scale(intrinsics_dir: str, cam_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    data = np.load(os.path.join(intrinsics_dir, f"cam{cam_idx}.npz"), allow_pickle=True)
    depth_scale = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else 0.001
    if not np.isfinite(depth_scale):
        depth_scale = 0.001
    return data["color_K"].astype(np.float64), data["color_D"].astype(np.float64), float(depth_scale)


def load_robot_pose_from_capture(cap: dict) -> Optional[np.ndarray]:
    T_base_gripper = None
    if "robot_pose_matrix_4x4" in cap:
        try:
            T_base_gripper = np.asarray(cap["robot_pose_matrix_4x4"], dtype=np.float64)
        except Exception:
            T_base_gripper = None
    if T_base_gripper is None and "capture_pose_matrix_4x4" in cap:
        try:
            T_base_gripper = np.asarray(cap["capture_pose_matrix_4x4"], dtype=np.float64)
        except Exception:
            T_base_gripper = None
    if T_base_gripper is None and "robot_pose_6dof" in cap:
        try:
            T_base_gripper = euler_deg_to_matrix(*cap["robot_pose_6dof"])
        except Exception:
            T_base_gripper = None
    if T_base_gripper is None and "capture_pose_6dof" in cap:
        try:
            T_base_gripper = euler_deg_to_matrix(*cap["capture_pose_6dof"])
        except Exception:
            T_base_gripper = None
    return T_base_gripper


def load_calib_dir(calib_dir: str) -> Dict[str, np.ndarray]:
    transforms = {}
    for filename in os.listdir(calib_dir):
        if filename.endswith(".npy"):
            transforms[filename.replace(".npy", "")] = np.load(os.path.join(calib_dir, filename))
    return transforms


def build_cube_pose_candidates(root_folder: str,
                               cinfo: dict,
                               K: np.ndarray,
                               D: np.ndarray,
                               cube: ArucoCubeTarget,
                               meta_reproj_thr: float = 3.0,
                               solve_reproj_thr: float = 5.0,
                               min_aspect: float = 0.0,
                               include_meta: bool = False) -> List[dict]:
    candidates: List[dict] = []
    cpnp = cinfo.get("cube_pnp")
    if include_meta and cpnp and cpnp.get("ok"):
        err = float(cpnp.get("reproj_mean_px", 99.0))
        T44 = cpnp.get("T_cam_cube_4x4")
        if T44 is not None and err <= float(meta_reproj_thr):
            candidates.append({
                "T_C_O": np.asarray(T44, dtype=np.float64),
                "err_mean": err,
                "n_points": int(cpnp.get("n_points", 4)),
                "used_ids": [int(x) for x in cpnp.get("used_ids", [])],
                "source": "meta",
            })

    rgb_path = os.path.join(root_folder, cinfo.get("rgb_path", ""))
    img = cv2.imread(rgb_path)
    if img is None:
        return candidates

    ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
        img, K, D,
        use_ransac=True, min_markers=1,
        reproj_thr_mean_px=float(solve_reproj_thr), return_reproj=True,
        min_aspect=float(min_aspect))
    if ok and reproj and reproj["err_mean"] <= float(solve_reproj_thr):
        candidates.append({
            "T_C_O": rodrigues_to_Rt(rvec, tvec),
            "err_mean": float(reproj["err_mean"]),
            "n_points": int(reproj["n_points"]),
            "used_ids": [int(x) for x in used],
            "source": "multi",
        })

    corners_list, ids = cube.detect(img)
    if ids is None:
        return candidates
    for corners, mid in zip(corners_list, ids):
        mid = int(mid)
        if not cube.model.has_marker(mid):
            continue
        obj = cube.model.marker_corners_in_rig(mid)
        img_pts = corners.reshape(4, 2).astype(np.float64)
        img_pts = cube.model.reorder_image_corners(mid, img_pts)
        if min_aspect > 0:
            edge_w = np.linalg.norm(img_pts[1] - img_pts[0])
            edge_h = np.linalg.norm(img_pts[3] - img_pts[0])
            aspect = min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6)
            if aspect < float(min_aspect):
                continue
        n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            obj.reshape(-1, 1, 3),
            img_pts.reshape(-1, 1, 2),
            K,
            D,
            flags=cv2.SOLVEPNP_IPPE,
        )
        for sol_idx in range(int(n_sol)):
            err_val = float(reproj_errs[sol_idx][0]) if reproj_errs is not None else 99.0
            if err_val > float(solve_reproj_thr):
                continue
            candidates.append({
                "T_C_O": rodrigues_to_Rt(rvecs[sol_idx], tvecs[sol_idx]),
                "err_mean": err_val,
                "n_points": 4,
                "used_ids": [mid],
                "source": f"ippe{sol_idx}",
            })
    return candidates


def get_event_base_camera_transform(cap: dict,
                                    cam_idx: int,
                                    transforms: Dict[str, np.ndarray],
                                    gripper_cam_idx: Optional[int]) -> Optional[np.ndarray]:
    key = f"T_base_C{int(cam_idx)}"
    if key in transforms:
        return np.asarray(transforms[key], dtype=np.float64)
    if gripper_cam_idx is not None and int(cam_idx) == int(gripper_cam_idx):
        T_gripper_cam = transforms.get("T_gripper_cam")
        T_base_gripper = load_robot_pose_from_capture(cap)
        if T_gripper_cam is not None and T_base_gripper is not None:
            return np.asarray(T_base_gripper, dtype=np.float64) @ np.asarray(T_gripper_cam, dtype=np.float64)
    return None


def select_consistent_event_cube_candidates(cap: dict,
                                            candidates_by_cam: Dict[int, List[dict]],
                                            transforms: Dict[str, np.ndarray],
                                            gripper_cam_idx: Optional[int],
                                            num_iters: int = 3,
                                            score_rot_weight: float = 5.0,
                                            score_err_weight: float = 10.0,
                                            single_face_weight: float = 1.0,
                                            single_face_penalty: float = 0.0) -> Dict[int, dict]:
    if not candidates_by_cam:
        return {}
    selected = {
        int(ci): select_primary_cube_candidate(cands)
        for ci, cands in candidates_by_cam.items()
        if select_primary_cube_candidate(cands) is not None
    }
    if len(selected) < 2:
        return selected

    for _ in range(max(int(num_iters), 1)):
        changed = 0
        updated = dict(selected)
        for ci, cands in candidates_by_cam.items():
            ci = int(ci)
            T_base_cam = get_event_base_camera_transform(cap, ci, transforms, gripper_cam_idx)
            if T_base_cam is None:
                continue
            reference_poses = []
            for cj, candj in selected.items():
                if int(cj) == ci or candj is None:
                    continue
                T_base_other = get_event_base_camera_transform(cap, int(cj), transforms, gripper_cam_idx)
                if T_base_other is None:
                    continue
                reference_poses.append(T_base_other @ np.asarray(candj["T_C_O"], dtype=np.float64))
            if not reference_poses:
                continue
            T_ref = reference_poses[0] if len(reference_poses) == 1 else weighted_pose_average(reference_poses)

            best = min(
                cands,
                key=lambda cand: (
                    (
                        float(np.linalg.norm((T_base_cam @ cand["T_C_O"])[:3, 3] - T_ref[:3, 3]) * 1000.0)
                        + float(score_rot_weight) * rotation_error_deg(
                            (T_base_cam @ cand["T_C_O"])[:3, :3],
                            T_ref[:3, :3],
                        )
                    ) / max(candidate_face_weight(cand, single_face_weight), 1e-6)
                    + float(score_err_weight) * float(cand.get("err_mean", 99.0))
                    + (float(single_face_penalty) if candidate_face_count(cand) <= 1 else 0.0),
                    cube_candidate_rank(cand),
                ),
            )
            if best is not selected.get(ci):
                changed += 1
            updated[ci] = best
        selected = updated
        if changed == 0:
            break
    return selected


def weighted_pose_average(T_list: List[np.ndarray]) -> np.ndarray:
    ts = np.asarray([T[:3, 3] for T in T_list], dtype=np.float64)
    Rs = np.asarray([T[:3, :3] for T in T_list], dtype=np.float64)
    t_mean = np.mean(ts, axis=0)
    R_mean = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_mean
    return T


def build_capture_cube_candidate_map(cap: dict,
                                     root_folder: str,
                                     K_map: Dict[int, np.ndarray],
                                     D_map: Dict[int, np.ndarray],
                                     cube: ArucoCubeTarget,
                                     gripper_cam_idx: Optional[int],
                                     include_meta: bool = False) -> Dict[int, List[dict]]:
    event_candidate_map: Dict[int, List[dict]] = {}
    for ci_str, cinfo in cap.get("cams", {}).items():
        ci = int(ci_str)
        if ci not in K_map or not cinfo.get("saved"):
            continue
        meta_thr = 5.0 if ci == gripper_cam_idx else 3.0
        candidates = build_cube_pose_candidates(
            root_folder, cinfo, K_map[ci], D_map[ci], cube,
            meta_reproj_thr=meta_thr, solve_reproj_thr=5.0,
            min_aspect=0.0, include_meta=include_meta)
        if candidates:
            event_candidate_map[ci] = candidates
    return event_candidate_map


def build_event_cube_selection(meta: dict,
                               transforms: Dict[str, np.ndarray],
                               intrinsics_dir: str,
                               root_folder: str,
                               all_cam_ids: List[int],
                               gripper_cam_idx: Optional[int],
                               cube_cfg: CubeConfig,
                               include_meta: bool = False,
                               selection_profile: str = "default") -> Dict[int, Dict[int, dict]]:
    cube = ArucoCubeTarget(cube_cfg)
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics_color(intrinsics_dir, ci)
    profile_kwargs = cube_selection_profile_kwargs(selection_profile)

    selection_by_event: Dict[int, Dict[int, dict]] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        event_candidate_map = build_capture_cube_candidate_map(
            cap, root_folder, K_map, D_map, cube, gripper_cam_idx,
            include_meta=include_meta)
        refined = select_consistent_event_cube_candidates(
            cap, event_candidate_map, transforms, gripper_cam_idx, **profile_kwargs) if event_candidate_map else {}
        if refined:
            selection_by_event[eid] = {
                int(ci): dict(cand)
                for ci, cand in refined.items()
            }
    return selection_by_event
