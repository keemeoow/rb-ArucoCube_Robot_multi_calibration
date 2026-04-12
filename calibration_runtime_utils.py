import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, depth_metrics_to_fields, rodrigues_to_Rt
from charuco_utils import CharucoTarget
from config import (
    CharucoBoardConfig,
    CubeConfig,
    get_default_cube_config,
    get_default_cube_config_source,
)
from capture_detection_utils import detect_cube_markers_in_frame
from cube_config_utils import (
    load_cube_config_from_json_file,
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


def filter_candidates_for_camera_role(candidates: List[dict],
                                      cam_idx: int,
                                      gripper_cam_idx: Optional[int],
                                      min_fixed_faces: int = 2,
                                      max_fixed_err: float = 2.75) -> List[dict]:
    filtered: List[dict] = []
    is_gripper = gripper_cam_idx is not None and int(cam_idx) == int(gripper_cam_idx)
    max_face_count = max((candidate_face_count(cand) for cand in candidates), default=0)
    for cand in candidates:
        face_count = candidate_face_count(cand)
        err_mean = float(cand.get("err_mean", 99.0))
        if max_face_count >= 2 and face_count < max_face_count:
            continue
        if not is_gripper:
            if face_count < int(min_fixed_faces):
                continue
            if max_fixed_err > 0 and err_mean > float(max_fixed_err):
                continue
        filtered.append(cand)
    return filtered


DEPTH_FIELD_KEYS = tuple(depth_metrics_to_fields(None).keys())


def copy_depth_fields(src: dict) -> Dict[str, object]:
    return {k: src[k] for k in DEPTH_FIELD_KEYS if k in src}


def candidate_has_depth(cand: dict) -> bool:
    return bool(cand.get("depth_valid")) and cand.get("depth_plane_mean_mm") is not None


def candidate_depth_rank(cand: dict) -> Tuple[int, float, int, int]:
    if not candidate_has_depth(cand):
        return (1, float("inf"), 0, 0)
    return (
        0,
        float(cand.get("depth_plane_mean_mm", float("inf"))),
        -int(cand.get("depth_num_samples", 0)),
        -int(cand.get("depth_num_markers", 0)),
    )


def candidate_depth_penalty(cand: dict, missing_penalty: float = 0.0) -> float:
    if not candidate_has_depth(cand):
        return float(missing_penalty)
    depth_err = max(float(cand.get("depth_plane_mean_mm", 0.0)), 0.0)
    sample_scale = min(max(int(cand.get("depth_num_samples", 0)), 1), 40) / 40.0
    marker_scale = min(max(int(cand.get("depth_num_markers", 0)), 1), 3) / 3.0
    return depth_err / max(sample_scale * marker_scale, 0.2)


def cube_candidate_rank(cand: dict) -> Tuple[float, float, int]:
    used_ids = cand.get("used_ids", [])
    err = float(cand.get("err_mean", 99.0))
    source = str(cand.get("source", "unknown"))
    source_prio = {
        "depth_svd": 0,
        "multi": 1,
        "meta": 2,
        "ippe0": 3,
        "ippe1": 4,
    }.get(source, 9)
    return (-len(set(int(x) for x in used_ids)),) + candidate_depth_rank(cand) + (err, source_prio)


def cube_selection_profile_kwargs(profile: str = "default",
                                  cube_only_single_face_weight: float = 0.35,
                                  cube_only_single_face_penalty: float = 0.75) -> Dict[str, float]:
    if str(profile) == "cube_only_specialized":
        return {
            "single_face_weight": float(cube_only_single_face_weight),
            "single_face_penalty": float(cube_only_single_face_penalty),
            "score_depth_weight": 1.5,
            "score_prior_weight": 5.0,
            "prior_translation_divisor_mm": 6.0,
            "prior_rotation_divisor_deg": 2.0,
            "max_consensus_translation_mm": 7.0,
            "max_consensus_rotation_deg": 1.0,
            "min_consensus_cams": 2,
        }
    return {
        "single_face_weight": 1.0,
        "single_face_penalty": 0.0,
        "score_depth_weight": 1.0,
        "score_prior_weight": 4.0,
        "prior_translation_divisor_mm": 6.0,
        "prior_rotation_divisor_deg": 2.0,
        "max_consensus_translation_mm": 7.0,
        "max_consensus_rotation_deg": 1.0,
        "min_consensus_cams": 2,
    }


def select_primary_cube_candidate(candidates: List[dict]) -> Optional[dict]:
    if not candidates:
        return None
    return min(candidates, key=cube_candidate_rank)


def resolve_cube_config_for_run(root_folder: str,
                                calib_dir: Optional[str] = None,
                                cube_config_json: Optional[str] = None,
                                default_cfg: Optional[CubeConfig] = None) -> Tuple[CubeConfig, str]:
    """Resolve the cube config for a run.

    Project policy: use the canonical project cube definition by default.
    The only supported way to use a different cube model is an explicit JSON override.
    """
    cfg_template = default_cfg or get_default_cube_config()
    if cube_config_json:
        cfg, source = load_cube_config_from_json_file(cube_config_json, cfg_template)
        if cfg is None:
            raise FileNotFoundError(f"Failed to load cube config JSON: {cube_config_json}")
        return cfg, f"explicit_json:{os.path.abspath(cube_config_json)} ({source})"
    return cfg_template, get_default_cube_config_source()


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


def load_robot_pose6_from_capture(cap: dict) -> Optional[np.ndarray]:
    for key in ("robot_pose_6dof", "capture_pose_6dof"):
        raw = cap.get(key)
        if not isinstance(raw, list) or len(raw) != 6:
            continue
        try:
            return np.asarray([float(x) for x in raw], dtype=np.float64)
        except Exception:
            continue
    return None


def get_capture_set_index(cap: dict) -> Optional[int]:
    raw = cap.get("set_index")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def get_capture_set_cube_center_transform_raw(cap: dict) -> Optional[np.ndarray]:
    raw = cap.get("set_cube_center_6dof")
    if not isinstance(raw, list) or len(raw) != 6:
        return None
    try:
        pose6 = [float(x) for x in raw]
    except Exception:
        return None
    return euler_deg_to_matrix(*pose6)


def get_capture_set_cube_prior(cap: dict,
                               transforms: Optional[Dict[str, Any]] = None,
                               allow_raw: bool = False) -> Optional[np.ndarray]:
    T_raw = get_capture_set_cube_center_transform_raw(cap)
    if T_raw is None:
        return None
    prior_info = None if transforms is None else transforms.get("set_cube_center_prior")
    if isinstance(prior_info, dict):
        delta = prior_info.get("T_set_cube_center_to_object")
        if delta is None:
            return T_raw if allow_raw else None
        try:
            T_delta = np.asarray(delta, dtype=np.float64).reshape(4, 4)
        except Exception:
            return T_raw if allow_raw else None
        return T_raw @ T_delta
    return T_raw if allow_raw else None


def get_object_anchor_key_for_set(set_index: Optional[int]) -> Optional[str]:
    if set_index is None:
        return None
    return f"T_base_O_set{int(set_index)}"


def get_capture_object_anchor(cap: dict,
                              transforms: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    set_index = get_capture_set_index(cap)
    set_key = get_object_anchor_key_for_set(set_index)
    if set_key and set_key in transforms:
        return np.asarray(transforms[set_key], dtype=np.float64), set_key
    T_base_O = transforms.get("T_base_O")
    if T_base_O is None:
        return None, None
    return np.asarray(T_base_O, dtype=np.float64), "T_base_O"


def load_calib_dir(calib_dir: str) -> Dict[str, Any]:
    transforms: Dict[str, Any] = {}
    for filename in os.listdir(calib_dir):
        if filename.endswith(".npy"):
            transforms[filename.replace(".npy", "")] = np.load(os.path.join(calib_dir, filename))
    internal_dir = os.path.join(calib_dir, "internal_runtime")
    if os.path.isdir(internal_dir):
        for filename in os.listdir(internal_dir):
            if filename.endswith(".npy"):
                transforms[filename.replace(".npy", "")] = np.load(os.path.join(internal_dir, filename))
    model_paths = [
        os.path.join(calib_dir, "gripper_base_pose_model.json"),
        os.path.join(internal_dir, "gripper_base_pose_model.json"),
    ]
    for model_path in model_paths:
        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                transforms["gripper_base_pose_model"] = json.load(f)
            break
    prior_paths = [
        os.path.join(calib_dir, "set_cube_center_prior.json"),
        os.path.join(internal_dir, "set_cube_center_prior.json"),
    ]
    for prior_path in prior_paths:
        if os.path.exists(prior_path):
            with open(prior_path, "r") as f:
                transforms["set_cube_center_prior"] = json.load(f)
            break
    return transforms


def predict_gripper_base_transform_from_model(cap: dict,
                                              transforms: Dict[str, Any],
                                              gripper_cam_idx: Optional[int]) -> Optional[np.ndarray]:
    if gripper_cam_idx is None:
        return None
    model = transforms.get("gripper_base_pose_model")
    if not isinstance(model, dict):
        return None
    model_gripper = model.get("gripper_cam_idx")
    if model_gripper is not None and int(model_gripper) != int(gripper_cam_idx):
        return None

    T_gripper_cam = transforms.get("T_gripper_cam")
    T_base_gripper = load_robot_pose_from_capture(cap)
    pose6 = load_robot_pose6_from_capture(cap)
    if T_gripper_cam is None or T_base_gripper is None or pose6 is None:
        return None

    samples = model.get("samples") or []
    if not samples:
        return None

    current_set = get_capture_set_index(cap)
    exact_tol_mm = float(model.get("exact_match_translation_tol_mm", 5.0))
    exact_tol_deg = float(model.get("exact_match_rotation_tol_deg", 5.0))
    feature_scale = np.asarray(model.get("feature_scale", [1, 1, 1, 1, 1, 1]), dtype=np.float64).reshape(6)
    feature_scale = np.where(np.abs(feature_scale) < 1e-9, 1.0, feature_scale)
    top_k = max(int(model.get("top_k", 3)), 1)
    sigma = max(float(model.get("kernel_sigma", 1.0)), 1e-6)
    max_norm_dist = float(model.get("max_normalized_dist", 2.5))

    rows = []
    for sample in samples:
        try:
            sample_pose = np.asarray(sample["robot_pose_6dof"], dtype=np.float64).reshape(6)
            delta_t_m = np.asarray(sample["delta_translation_mm"], dtype=np.float64).reshape(3) / 1000.0
            delta_rvec = np.asarray(sample["delta_rvec_rad"], dtype=np.float64).reshape(3)
        except Exception:
            continue
        raw_delta = sample_pose - pose6
        exact_match = (
            np.all(np.abs(raw_delta[:3]) <= exact_tol_mm) and
            np.all(np.abs(raw_delta[3:]) <= exact_tol_deg)
        )
        norm_dist = float(np.linalg.norm(raw_delta / feature_scale))
        set_match = (current_set is not None and sample.get("set_index") == int(current_set))
        rows.append({
            "norm_dist": norm_dist,
            "exact_match": exact_match,
            "set_match": set_match,
            "delta_t_m": delta_t_m,
            "delta_rvec": delta_rvec,
        })

    if not rows:
        return None

    exact_rows = [row for row in rows if row["exact_match"]]
    if exact_rows:
        exact_rows.sort(key=lambda row: (not row["set_match"], row["norm_dist"]))
        chosen = exact_rows[0]
        R_delta, _ = cv2.Rodrigues(np.asarray(chosen["delta_rvec"], dtype=np.float64).reshape(3, 1))
        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = np.asarray(chosen["delta_t_m"], dtype=np.float64).reshape(3)
        return T_delta @ (np.asarray(T_base_gripper, dtype=np.float64) @ np.asarray(T_gripper_cam, dtype=np.float64))

    rows.sort(key=lambda row: (not row["set_match"], row["norm_dist"]))
    nearest = rows[:top_k]
    if not nearest or nearest[0]["norm_dist"] > max_norm_dist:
        return None

    d = np.asarray([row["norm_dist"] for row in nearest], dtype=np.float64)
    w = np.exp(-0.5 * np.square(d / sigma))
    if current_set is not None:
        for idx, row in enumerate(nearest):
            if row["set_match"]:
                w[idx] *= 1.25
    w = w / (np.sum(w) + 1e-12)

    delta_t = np.sum(np.stack([row["delta_t_m"] for row in nearest], axis=0) * w[:, None], axis=0)
    delta_rvec = np.sum(np.stack([row["delta_rvec"] for row in nearest], axis=0) * w[:, None], axis=0)
    R_delta, _ = cv2.Rodrigues(np.asarray(delta_rvec, dtype=np.float64).reshape(3, 1))
    T_delta = np.eye(4, dtype=np.float64)
    T_delta[:3, :3] = R_delta
    T_delta[:3, 3] = delta_t
    return T_delta @ (np.asarray(T_base_gripper, dtype=np.float64) @ np.asarray(T_gripper_cam, dtype=np.float64))


def build_cube_pose_candidates(root_folder: str,
                               cinfo: dict,
                               K: np.ndarray,
                               D: np.ndarray,
                               cube: ArucoCubeTarget,
                               meta_reproj_thr: float = 3.0,
                               solve_reproj_thr: float = 5.0,
                               min_aspect: float = 0.0,
                               include_meta: bool = False,
                               depth_scale: Optional[float] = None,
                               include_depth_pose_candidate: bool = False) -> List[dict]:
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
                **copy_depth_fields(cpnp),
            })

    rgb_path = os.path.join(root_folder, cinfo.get("rgb_path", ""))
    img = cv2.imread(rgb_path)
    if img is None:
        return candidates
    depth = None
    depth_rel = cinfo.get("depth_path", "")
    if depth_scale is not None and depth_rel:
        depth = cv2.imread(os.path.join(root_folder, depth_rel), cv2.IMREAD_UNCHANGED)

    is_gripper = bool(cinfo.get("is_gripper"))
    charuco_det = CharucoTarget(CharucoBoardConfig()) if is_gripper else None
    detect_info = detect_cube_markers_in_frame(
        img,
        cube,
        cube_ids=cube.cfg.marker_ids,
        charuco=charuco_det,
        is_gripper=is_gripper,
        board_mask_pad_px=6.0,
    )
    cube_img = detect_info["cube_image"]
    corners_list = detect_info["corners"]
    ids = detect_info["ids"]

    ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
        cube_img, K, D,
        use_ransac=True, min_markers=1,
        reproj_thr_mean_px=float(solve_reproj_thr), return_reproj=True,
        min_aspect=float(min_aspect),
        depth_u16=depth,
        depth_scale=depth_scale)
    if ok and reproj and reproj["err_mean"] <= float(solve_reproj_thr):
        candidates.append({
            "T_C_O": rodrigues_to_Rt(rvec, tvec),
            "err_mean": float(reproj["err_mean"]),
            "n_points": int(reproj["n_points"]),
            "used_ids": [int(x) for x in used],
            "source": "multi",
            **depth_metrics_to_fields(reproj.get("depth_metrics")),
        })

    if bool(include_depth_pose_candidate):
        ok_depth, depth_pose = cube.solve_pose_from_depth(
            cube_img,
            depth,
            depth_scale,
            K,
            D,
            min_aspect=float(min_aspect),
            min_valid_per_marker=3,
            min_valid_points=6,
            max_reproj_mean_px=float(solve_reproj_thr),
        )
        if ok_depth and depth_pose is not None:
            candidates.append({
                "T_C_O": np.asarray(depth_pose["T_C_O"], dtype=np.float64),
                "err_mean": float(depth_pose["err_mean"]),
                "n_points": int(depth_pose["n_points"]),
                "used_ids": [int(x) for x in depth_pose.get("used_ids", [])],
                "source": "depth_svd",
                **depth_metrics_to_fields(depth_pose.get("depth_metrics")),
            })

    if ids is None:
        return candidates
    for corners, mid in zip(corners_list, ids):
        mid = int(mid)
        if not cube.model.has_marker(mid):
            continue
        img_pts = corners.reshape(4, 2).astype(np.float64)
        img_pts = cube.model.reorder_image_corners(mid, img_pts)
        if min_aspect > 0:
            edge_w = np.linalg.norm(img_pts[1] - img_pts[0])
            edge_h = np.linalg.norm(img_pts[3] - img_pts[0])
            aspect = min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6)
            if aspect < float(min_aspect):
                continue
        ippe_candidates = cube.single_marker_ippe_candidates(
            mid,
            corners.reshape(4, 2).astype(np.float64),
            K,
            D,
            corners_list=corners_list,
            ids=ids,
            depth_u16=depth,
            depth_scale=depth_scale,
        )
        if not ippe_candidates:
            continue
        best = min(ippe_candidates, key=lambda cand: cand["rank"])
        err_val = float(best["err_mean"])
        if err_val > float(solve_reproj_thr):
            continue
        candidates.append({
            "T_C_O": np.asarray(best["T_C_O"], dtype=np.float64),
            "err_mean": err_val,
            "n_points": 4,
            "used_ids": [mid],
            "source": f"ippe{int(best['solution_index'])}",
            "z_ok": bool(best["z_ok"]),
            "vis_ok": bool(best["vis_ok"]),
            "vis_score": float(best["vis_score"]),
            "visibility_tier": int(best["visibility_tier"]),
            **depth_metrics_to_fields(best.get("depth_metrics")),
        })
    return candidates


def get_event_base_camera_transform(cap: dict,
                                    cam_idx: int,
                                    transforms: Dict[str, Any],
                                    gripper_cam_idx: Optional[int]) -> Optional[np.ndarray]:
    eid = int(cap.get("event_id", -1))
    if eid >= 0:
        event_key = f"T_base_C{int(cam_idx)}_event{eid}"
        if event_key in transforms:
            return np.asarray(transforms[event_key], dtype=np.float64)
    if gripper_cam_idx is not None and int(cam_idx) == int(gripper_cam_idx):
        T_model = predict_gripper_base_transform_from_model(cap, transforms, gripper_cam_idx)
        if T_model is not None:
            return np.asarray(T_model, dtype=np.float64)
    key = f"T_base_C{int(cam_idx)}"
    if key in transforms:
        return np.asarray(transforms[key], dtype=np.float64)
    if gripper_cam_idx is not None and int(cam_idx) == int(gripper_cam_idx):
        T_gripper_cam = transforms.get("T_gripper_cam")
        T_base_gripper = load_robot_pose_from_capture(cap)
        if T_gripper_cam is not None and T_base_gripper is not None:
            return np.asarray(T_base_gripper, dtype=np.float64) @ np.asarray(T_gripper_cam, dtype=np.float64)
    return None


def weighted_pose_average(T_list: List[np.ndarray],
                          weights: Optional[List[float]] = None) -> np.ndarray:
    ts = np.asarray([T[:3, 3] for T in T_list], dtype=np.float64)
    Rs = np.asarray([T[:3, :3] for T in T_list], dtype=np.float64)
    if weights is None:
        w = np.ones((len(T_list),), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != len(T_list):
            raise ValueError("weights size must match T_list")
        w = np.maximum(w, 1e-12)
    w /= (np.sum(w) + 1e-12)
    t_mean = np.sum(w[:, None] * ts, axis=0)
    R_mean = np.sum(w[:, None, None] * Rs, axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_mean
    return T


def _event_candidate_pose_rows(cap: dict,
                               selected: Dict[int, dict],
                               transforms: Dict[str, Any],
                               gripper_cam_idx: Optional[int],
                               single_face_weight: float) -> List[dict]:
    rows: List[dict] = []
    for ci, cand in selected.items():
        T_base_cam = get_event_base_camera_transform(cap, int(ci), transforms, gripper_cam_idx)
        if T_base_cam is None:
            continue
        err_mean = max(float(cand.get("err_mean", 1.0)), 1e-9)
        depth_penalty = candidate_depth_penalty(cand, missing_penalty=6.0)
        weight = (
            candidate_face_weight(cand, single_face_weight)
            / err_mean
            / (1.0 + 0.15 * depth_penalty)
        )
        rows.append({
            "cam_idx": int(ci),
            "cand": cand,
            "T_base_obj": np.asarray(T_base_cam, dtype=np.float64) @ np.asarray(cand["T_C_O"], dtype=np.float64),
            "weight": float(max(weight, 1e-9)),
        })
    return rows


def _prune_inconsistent_event_selection(cap: dict,
                                        selected: Dict[int, dict],
                                        transforms: Dict[str, Any],
                                        gripper_cam_idx: Optional[int],
                                        single_face_weight: float,
                                        max_consensus_translation_mm: float,
                                        max_consensus_rotation_deg: float,
                                        min_consensus_cams: int,
                                        prune_score_rot_weight: float = 5.0) -> Dict[int, dict]:
    if len(selected) < 2:
        return selected
    if max_consensus_translation_mm <= 0.0 and max_consensus_rotation_deg <= 0.0:
        return selected

    active = {int(ci): cand for ci, cand in selected.items()}
    min_keep = max(int(min_consensus_cams), 2)

    for _ in range(max(len(active), 1)):
        rows = _event_candidate_pose_rows(cap, active, transforms, gripper_cam_idx, single_face_weight)
        if len(rows) < 2:
            break
        T_ref = weighted_pose_average(
            [row["T_base_obj"] for row in rows],
            [row["weight"] for row in rows],
        )
        residuals = []
        for row in rows:
            T_evt = row["T_base_obj"]
            residuals.append({
                "cam_idx": int(row["cam_idx"]),
                "dt_mm": float(np.linalg.norm(T_evt[:3, 3] - T_ref[:3, 3]) * 1000.0),
                "dr_deg": rotation_error_deg(T_evt[:3, :3], T_ref[:3, :3]),
            })
        violating = [
            row for row in residuals
            if row["dt_mm"] > float(max_consensus_translation_mm)
            or row["dr_deg"] > float(max_consensus_rotation_deg)
        ]
        if not violating:
            break
        if len(active) <= min_keep:
            break
        worst = max(
            violating,
            key=lambda row: (
                row["dt_mm"] > float(max_consensus_translation_mm)
                or row["dr_deg"] > float(max_consensus_rotation_deg),
                row["dt_mm"] + float(prune_score_rot_weight) * row["dr_deg"],
                row["dt_mm"],
                row["dr_deg"],
            ),
        )
        active.pop(int(worst["cam_idx"]), None)

    return active


def candidate_set_prior_penalty(cap: dict,
                                T_base_cam: np.ndarray,
                                cand: dict,
                                transforms: Optional[Dict[str, Any]],
                                prior_translation_divisor_mm: float,
                                prior_rotation_divisor_deg: float) -> float:
    T_prior = get_capture_set_cube_prior(cap, transforms=transforms, allow_raw=False)
    if T_prior is None:
        return 0.0
    T_base_obj = np.asarray(T_base_cam, dtype=np.float64) @ np.asarray(cand["T_C_O"], dtype=np.float64)
    dt_mm = float(np.linalg.norm(T_base_obj[:3, 3] - T_prior[:3, 3]) * 1000.0)
    dr_deg = rotation_error_deg(T_base_obj[:3, :3], T_prior[:3, :3])
    trans_div = max(float(prior_translation_divisor_mm), 1e-6)
    rot_div = max(float(prior_rotation_divisor_deg), 1e-6)
    return dt_mm / trans_div + dr_deg / rot_div


def select_consistent_event_cube_candidates(cap: dict,
                                            candidates_by_cam: Dict[int, List[dict]],
                                            transforms: Dict[str, Any],
                                            gripper_cam_idx: Optional[int],
                                            num_iters: int = 3,
                                            score_rot_weight: float = 5.0,
                                            score_err_weight: float = 10.0,
                                            score_depth_weight: float = 1.0,
                                            score_prior_weight: float = 4.0,
                                            prior_translation_divisor_mm: float = 6.0,
                                            prior_rotation_divisor_deg: float = 2.0,
                                            single_face_weight: float = 1.0,
                                            single_face_penalty: float = 0.0,
                                            max_consensus_translation_mm: float = 7.0,
                                            max_consensus_rotation_deg: float = 1.0,
                                            min_consensus_cams: int = 2) -> Dict[int, dict]:
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
                    + float(score_depth_weight) * candidate_depth_penalty(cand, missing_penalty=6.0)
                    + float(score_prior_weight) * candidate_set_prior_penalty(
                        cap,
                        T_base_cam,
                        cand,
                        transforms,
                        prior_translation_divisor_mm=float(prior_translation_divisor_mm),
                        prior_rotation_divisor_deg=float(prior_rotation_divisor_deg),
                    )
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
    return _prune_inconsistent_event_selection(
        cap,
        selected,
        transforms,
        gripper_cam_idx,
        single_face_weight=float(single_face_weight),
        max_consensus_translation_mm=float(max_consensus_translation_mm),
        max_consensus_rotation_deg=float(max_consensus_rotation_deg),
        min_consensus_cams=int(min_consensus_cams),
        prune_score_rot_weight=float(score_rot_weight),
    )


def build_capture_cube_candidate_map(cap: dict,
                                     root_folder: str,
                                     K_map: Dict[int, np.ndarray],
                                     D_map: Dict[int, np.ndarray],
                                     cube: ArucoCubeTarget,
                                     gripper_cam_idx: Optional[int],
                                     include_meta: bool = False,
                                     depth_scale_map: Optional[Dict[int, float]] = None,
                                     include_depth_pose_candidate: bool = False) -> Dict[int, List[dict]]:
    event_candidate_map: Dict[int, List[dict]] = {}
    for ci_str, cinfo in cap.get("cams", {}).items():
        ci = int(ci_str)
        if ci not in K_map or not cinfo.get("saved"):
            continue
        meta_thr = 5.0 if ci == gripper_cam_idx else 3.0
        candidates = build_cube_pose_candidates(
            root_folder, cinfo, K_map[ci], D_map[ci], cube,
            meta_reproj_thr=meta_thr, solve_reproj_thr=5.0,
            min_aspect=0.0, include_meta=include_meta,
            depth_scale=None if depth_scale_map is None else depth_scale_map.get(ci),
            include_depth_pose_candidate=include_depth_pose_candidate)
        candidates = filter_candidates_for_camera_role(candidates, ci, gripper_cam_idx)
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
                               selection_profile: str = "default",
                               include_depth_pose_candidate: bool = False) -> Dict[int, Dict[int, dict]]:
    cube = ArucoCubeTarget(cube_cfg)
    K_map, D_map, depth_scale_map = {}, {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], depth_scale_map[ci] = load_intrinsics_with_depth_scale(intrinsics_dir, ci)
    profile_kwargs = cube_selection_profile_kwargs(selection_profile)

    selection_by_event: Dict[int, Dict[int, dict]] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        event_candidate_map = build_capture_cube_candidate_map(
            cap, root_folder, K_map, D_map, cube, gripper_cam_idx,
            include_meta=include_meta, depth_scale_map=depth_scale_map,
            include_depth_pose_candidate=include_depth_pose_candidate)
        refined = select_consistent_event_cube_candidates(
            cap, event_candidate_map, transforms, gripper_cam_idx, **profile_kwargs) if event_candidate_map else {}
        if refined:
            selection_by_event[eid] = {
                int(ci): dict(cand)
                for ci, cand in refined.items()
            }
    return selection_by_event
