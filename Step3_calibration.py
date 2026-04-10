# Step3_calibration.py
"""
Step 3: 통합 캘리브레이션.

Step2_capture_capture.py의 meta.json을 입력으로 받아:
  A) 저장된/재검출 cube pose 후보를 읽는다
  B) 고정 카메라 상대변환을 cube 기준으로 계산한다
  C) Hand-eye는 gripper ChArUco 보드로 계산하되 cube 일관성까지 함께 평가한다
  D) 고정 카메라는 cube-primary, board-refine 방식으로 base 좌표계에 정렬한다
  E) set_cube_center_6dof가 있으면 cube object frame과의 상수 delta를 학습해 set prior로 재정렬한다
  F) 기본 경로에서는 depth pose를 직접 최적화 변수로 쓰지 않고, depth 품질과 set prior를
     candidate selection / refinement에만 반영한다.

출력:
  T_gripper_cam.npy      - gripper -> camera
  T_base_O.npy           - base -> object (최종 단일 출력)
  T_base_C{i}.npy        - base -> fixed camera
  T_C{ref}_C{i}.npy      - ref fixed camera -> other fixed camera
  internal_runtime/*     - event/set별 보조 runtime transform

실행:
  python Step3_calibration.py \
    --root_folder ./data/session \
    --intrinsics_dir ./intrinsics

기본 정책:
  - 새 데이터도 기본 `calib_out` 경로에서 현재 안정형 파이프라인을 그대로 사용한다.
  - `--enable_depth_pose_candidate`는 dimension/mesh 실험용 옵션이며 기본 OFF다.
"""

import os
import json
import argparse
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, depth_metrics_to_fields, rodrigues_to_Rt, inv_T
from calibration_runtime_utils import (
    copy_depth_fields,
    cube_selection_profile_kwargs,
    filter_candidates_for_camera_role,
    get_capture_set_index,
    get_capture_set_cube_center_transform_raw,
    get_object_anchor_key_for_set,
    load_intrinsics_with_depth_scale,
    select_consistent_event_cube_candidates,
    select_primary_cube_candidate,
)
from config import CubeConfig, get_default_cube_config
from utils_pose import robust_se3_average, se3_distance
from robot_comm import euler_deg_to_matrix


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def cleanup_legacy_public_outputs(out_dir: str) -> None:
    legacy_json = {
        "T_base_O_by_set.json",
        "gripper_base_pose_model.json",
        "T_base_C2_by_event.json",
    }
    legacy_patterns = [
        re.compile(r"^T_base_C\d+_event\d+\.npy$"),
        re.compile(r"^T_base_O_set\d+\.npy$"),
    ]
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        if name in legacy_json or any(pat.match(name) for pat in legacy_patterns):
            try:
                os.remove(path)
            except OSError:
                pass


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


def marker_aspect_ratio(img_pts: np.ndarray) -> float:
    pts = np.asarray(img_pts, dtype=np.float64).reshape(4, 2)
    edge_w = np.linalg.norm(pts[1] - pts[0])
    edge_h = np.linalg.norm(pts[3] - pts[0])
    return float(min(edge_w, edge_h) / (max(edge_w, edge_h) + 1e-6))


def candidate_weight(cand: dict, single_face_scale: float = 0.35) -> float:
    face_count = max(len(set(int(x) for x in cand.get("used_ids", []))), 1)
    weight = face_count / max(float(cand.get("err_mean", 1.0)), 1e-9)
    if face_count <= 1:
        weight *= float(single_face_scale)
    if bool(cand.get("depth_valid")) and cand.get("depth_plane_mean_mm") is not None:
        depth_err = max(float(cand.get("depth_plane_mean_mm", 0.0)), 0.0)
        sample_scale = min(max(int(cand.get("depth_num_samples", 0)), 1), 40) / 40.0
        marker_scale = min(max(int(cand.get("depth_num_markers", 0)), 1), 3) / 3.0
        weight *= sample_scale
        weight *= marker_scale
        weight /= (1.0 + depth_err / 3.0)
    else:
        weight *= 0.5
    return float(weight)


def stored_cube_pose_candidates(cinfo: dict,
                                cam_idx: int,
                                gripper_cam_idx: Optional[int],
                                max_err: float,
                                min_markers: int,
                                min_aspect: float) -> List[dict]:
    candidates: List[dict] = []
    aspect_by_marker: Dict[int, float] = {}

    for item in cinfo.get("markers", []):
        mid = int(item.get("marker_id", -1))
        corners = np.asarray(item.get("corners_2d", []), dtype=np.float64)
        aspect = None
        if corners.shape == (4, 2):
            aspect = marker_aspect_ratio(corners)
            aspect_by_marker[mid] = aspect

        pose_candidates = item.get("pose_candidates") or []
        if pose_candidates:
            for cand in pose_candidates:
                err = float(cand.get("reproj_error_mean_px", 99.0))
                if err > float(max_err):
                    continue
                if aspect is not None and aspect < float(min_aspect):
                    continue
                T44 = cand.get("T_cam_cube_4x4")
                if T44 is None:
                    continue
                sol_idx = int(cand.get("solution_index", 0))
                candidates.append({
                    "T_C_O": np.asarray(T44, dtype=np.float64),
                    "err_mean": err,
                    "n_points": 4,
                    "used_ids": [mid],
                    "source": f"ippe{sol_idx}",
                    "aspect_ratio": float(aspect) if aspect is not None else None,
                    **copy_depth_fields(cand),
                })
            continue

        T44 = item.get("T_cam_cube_4x4")
        err = float(item.get("reproj_error_mean_px", 99.0))
        if T44 is None or err > float(max_err):
            continue
        if aspect is not None and aspect < float(min_aspect):
            continue
        candidates.append({
            "T_C_O": np.asarray(T44, dtype=np.float64),
            "err_mean": err,
            "n_points": 4,
            "used_ids": [mid],
            "source": "ippe0",
            "aspect_ratio": float(aspect) if aspect is not None else None,
            **copy_depth_fields(item),
        })

    cpnp = cinfo.get("cube_pnp")
    if cpnp and cpnp.get("ok"):
        err = float(cpnp.get("reproj_mean_px", 99.0))
        used_ids = [int(x) for x in cpnp.get("used_ids", [])]
        T44 = cpnp.get("T_cam_cube_4x4")
        enough_ids = len(set(used_ids)) >= max(int(min_markers), 1)
        if T44 is not None and err <= float(max_err) and enough_ids:
            used_aspects = [aspect_by_marker[mid] for mid in used_ids if mid in aspect_by_marker]
            min_used_aspect = min(used_aspects) if used_aspects else None
            if min_used_aspect is None or min_used_aspect >= float(min_aspect):
                candidates.append({
                    "T_C_O": np.asarray(T44, dtype=np.float64),
                    "err_mean": err,
                    "n_points": int(cpnp.get("n_points", 4 * max(len(set(used_ids)), 1))),
                    "used_ids": used_ids,
                    "source": "meta",
                    "from_gripper": bool(cam_idx == gripper_cam_idx),
                    **copy_depth_fields(cpnp),
                })

    return candidates


def estimate_image_cube_pose_candidates(cube: ArucoCubeTarget,
                                        img: np.ndarray,
                                        K: np.ndarray,
                                        D: np.ndarray,
                                        max_err: float,
                                        min_markers: int,
                                        min_aspect: float,
                                        depth_u16: Optional[np.ndarray] = None,
                                        depth_scale: Optional[float] = None,
                                        include_depth_pose_candidate: bool = False) -> List[dict]:
    candidates: List[dict] = []

    ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
        img, K, D,
        use_ransac=True, min_markers=max(int(min_markers), 1),
        reproj_thr_mean_px=float(max_err), return_reproj=True,
        min_aspect=float(min_aspect),
        depth_u16=depth_u16,
        depth_scale=depth_scale)
    if ok and reproj and reproj["err_mean"] <= float(max_err):
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
            img,
            depth_u16,
            depth_scale,
            K,
            D,
            min_aspect=float(min_aspect),
            min_valid_per_marker=3,
            min_valid_points=6,
            max_reproj_mean_px=float(max_err),
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

    corners_list, ids = cube.detect(img)
    if ids is None:
        return candidates

    for c, mid in zip(corners_list, ids):
        mid = int(mid)
        if not cube.model.has_marker(mid):
            continue
        img_pts = cube.model.reorder_image_corners(mid, c.reshape(4, 2).astype(np.float64))
        if marker_aspect_ratio(img_pts) < float(min_aspect):
            continue
        ippe_candidates = cube.single_marker_ippe_candidates(
            mid,
            c.reshape(4, 2).astype(np.float64),
            K,
            D,
            corners_list=corners_list,
            ids=ids,
            depth_u16=depth_u16,
            depth_scale=depth_scale,
        )
        if not ippe_candidates:
            continue
        best = min(ippe_candidates, key=lambda cand: cand["rank"])
        if float(best["err_mean"]) > float(max_err):
            continue
        candidates.append({
            "T_C_O": np.asarray(best["T_C_O"], dtype=np.float64),
            "err_mean": float(best["err_mean"]),
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


def build_cube_anchor_from_fixed_cams(meta: dict,
                                      pnp_obs: Dict[int, Dict[int, dict]],
                                      T_base_Ci: Dict[int, np.ndarray],
                                      fixed_cam_ids: List[int],
                                      gripper_cam_idx: Optional[int],
                                      min_cams: int = 2,
                                      extra_transforms: Optional[Dict[str, Any]] = None):
    transforms: Dict[str, Any] = dict(extra_transforms or {})
    transforms.update({
        f"T_base_C{int(ci)}": np.asarray(T, dtype=np.float64)
        for ci, T in T_base_Ci.items()
        if int(ci) in fixed_cam_ids
    })
    profile_kwargs = cube_selection_profile_kwargs("cube_only_specialized")
    event_pose_map: Dict[int, np.ndarray] = {}
    event_pose_list: List[np.ndarray] = []
    event_weights: List[float] = []
    event_details: List[dict] = []
    total_keys = 0
    signature_counter: Counter = Counter()

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        event_candidate_map = {}
        for ci in fixed_cam_ids:
            tf_key = f"T_base_C{int(ci)}"
            if tf_key not in transforms:
                continue
            obs = pnp_obs.get(int(ci), {}).get(eid)
            if obs is None:
                continue
            cands = obs.get("_candidates") or [obs]
            event_candidate_map[int(ci)] = []
            for cand in cands:
                T_C_O = np.asarray(cand.get("T_C_O"), dtype=np.float64)
                if T_C_O.shape != (4, 4) or not np.all(np.isfinite(T_C_O)):
                    continue
                event_candidate_map[int(ci)].append({
                    "T_C_O": T_C_O,
                    "err_mean": float(cand.get("err_mean", 99.0)),
                    "n_points": int(cand.get("n_points", 4)),
                    "used_ids": [int(x) for x in cand.get("used_ids", [])],
                    "source": str(cand.get("source", "unknown")),
                })
            if not event_candidate_map[int(ci)]:
                event_candidate_map.pop(int(ci), None)

        if len(event_candidate_map) < max(int(min_cams), 1):
            continue

        total_keys += 1
        refined = select_consistent_event_cube_candidates(
            cap, event_candidate_map, transforms, gripper_cam_idx, **profile_kwargs)
        if len(refined) < max(int(min_cams), 1):
            continue

        Ts, ws = [], []
        detail_row = {"event_id": eid, "support": int(len(refined)), "cameras": {}}
        for ci, cand in refined.items():
            T_base_O = transforms[f"T_base_C{int(ci)}"] @ np.asarray(cand["T_C_O"], dtype=np.float64)
            Ts.append(T_base_O)
            ws.append(candidate_weight(cand))
            signature = (tuple(int(x) for x in cand.get("used_ids", [])), str(cand.get("source", "unknown")))
            signature_counter[signature] += 1
            detail_row["cameras"][str(ci)] = {
                "used_ids": [int(x) for x in cand.get("used_ids", [])],
                "source": str(cand.get("source", "unknown")),
                "err_mean": float(cand.get("err_mean", 99.0)),
            }

        T_event = robust_weighted_se3_average(Ts, ws)
        event_pose_map[eid] = T_event
        event_pose_list.append(T_event)
        event_weights.append(float(np.sum(ws)))
        event_details.append(detail_row)

    if not event_pose_list:
        return None, {}, {
            "support": 0,
            "total_keys": int(total_keys),
            "stability": {},
            "events": [],
        }

    T_avg, st_anchor = robust_weighted_se3_average(event_pose_list, event_weights, return_stats=True)
    diag = {
        "support": int(len(event_pose_list)),
        "total_keys": int(total_keys),
        "stability": st_anchor,
        "events": event_details,
    }
    if signature_counter:
        (used_ids, source), count = signature_counter.most_common(1)[0]
        diag["dominant_signature"] = {
            "used_ids": [int(x) for x in used_ids],
            "source": str(source),
            "support": int(count),
        }
    return T_avg, event_pose_map, diag


def build_gripper_cube_anchor(robot_T: Dict[int, np.ndarray],
                              pnp_obs: Dict[int, Dict[int, dict]],
                              gripper_cam_idx: int,
                              T_gTc: np.ndarray,
                              gripper_base_by_event: Optional[Dict[int, np.ndarray]] = None):
    base_keys = set(robot_T.keys())
    if gripper_base_by_event:
        base_keys |= set(int(eid) for eid in gripper_base_by_event.keys())
    common_cube = sorted(base_keys & set(pnp_obs.get(gripper_cam_idx, {}).keys()))
    event_pose_map: Dict[int, np.ndarray] = {}
    event_pose_list: List[np.ndarray] = []
    event_weights: List[float] = []

    for eid in common_cube:
        obs = pnp_obs[gripper_cam_idx][eid]
        T_base_cam = None
        if gripper_base_by_event is not None:
            T_base_cam = gripper_base_by_event.get(eid)
        if T_base_cam is None:
            if eid not in robot_T:
                continue
            T_base_cam = robot_T[eid] @ T_gTc
        T_B_O = np.asarray(T_base_cam, dtype=np.float64) @ np.asarray(obs["T_C_O"], dtype=np.float64)
        event_pose_map[eid] = T_B_O
        event_pose_list.append(T_B_O)
        event_weights.append(candidate_weight(obs))

    if not event_pose_list:
        return None, {}, {
            "support": 0,
            "total_keys": 0,
            "stability": {},
            "events": [],
        }

    T_avg, st_anchor = robust_weighted_se3_average(event_pose_list, event_weights, return_stats=True)
    return T_avg, event_pose_map, {
        "support": int(len(event_pose_list)),
        "total_keys": int(len(common_cube)),
        "stability": st_anchor,
        "events": [{"event_id": int(eid)} for eid in common_cube],
    }


def estimate_fixed_cameras_from_cube_anchor(
    pnp_obs: Dict[int, Dict[int, dict]],
    fixed_cam_ids: List[int],
    T_B_O_by_event: Dict[int, np.ndarray],
):
    T_base_Ci = {}
    base_stats = {}

    def _pick_candidate(ci, eid, T_B_O, T_ref=None):
        candidates = pnp_obs[ci][eid].get("_candidates")
        if not candidates or len(candidates) <= 1:
            return pnp_obs[ci][eid]["T_C_O"]
        best_T, best_score = None, 1e9
        for cand in candidates:
            T_sol = np.asarray(cand["T_C_O"], dtype=np.float64)
            err_sol = float(cand.get("err_mean", 99.0))
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
        common = sorted(set(pnp_obs.get(ci, {}).keys()) & set(T_B_O_by_event.keys()))
        if not common:
            continue

        Ts1, ws1 = [], []
        for eid in common:
            T_C_O = _pick_candidate(ci, eid, T_B_O_by_event[eid])
            Ts1.append(T_B_O_by_event[eid] @ inv_T(T_C_O))
            ws1.append(1.0 / max(float(pnp_obs[ci][eid].get("err_mean", 1.0)), 1e-9))
        T_rough = robust_weighted_se3_average(Ts1, ws1)

        Ts2, ws2 = [], []
        for eid in common:
            T_C_O = _pick_candidate(ci, eid, T_B_O_by_event[eid], T_ref=T_rough)
            Ts2.append(T_B_O_by_event[eid] @ inv_T(T_C_O))
            ws2.append(1.0 / max(float(pnp_obs[ci][eid].get("err_mean", 1.0)), 1e-9))

        T_avg, st = robust_weighted_se3_average(Ts2, ws2, return_stats=True)
        st["method"] = "cube_anchor_primary"
        st["support"] = int(len(common))
        st["total_keys"] = int(len(common))
        T_base_Ci[int(ci)] = T_avg
        base_stats[f"T_base_C{int(ci)}"] = st

    return T_base_Ci, base_stats


def merge_fixed_camera_base_transforms(
    cube_T_base_Ci: Dict[int, np.ndarray],
    cube_stats: Dict[str, dict],
    board_T_base_Ci: Dict[int, np.ndarray],
    board_stats: Dict[str, dict],
    mode: str = "cube_primary",
    board_refine_alpha: float = 0.35,
    max_refine_dt_mm: float = 25.0,
    max_refine_dr_deg: float = 5.0,
):
    merged = {}
    merged_stats = {}
    camera_ids = sorted(set(cube_T_base_Ci.keys()) | set(board_T_base_Ci.keys()))
    for ci in camera_ids:
        key = f"T_base_C{int(ci)}"
        T_cube = cube_T_base_Ci.get(int(ci))
        T_board = board_T_base_Ci.get(int(ci))
        st_cube = dict(cube_stats.get(key, {}))
        st_board = dict(board_stats.get(key, {}))

        if str(mode) == "board_primary":
            primary = T_board if T_board is not None else T_cube
            primary_stats = st_board if T_board is not None else st_cube
            secondary = T_cube if T_board is not None else None
        else:
            primary = T_cube if T_cube is not None else T_board
            primary_stats = st_cube if T_cube is not None else st_board
            secondary = T_board if T_cube is not None else None

        if primary is None:
            continue

        T_final = np.asarray(primary, dtype=np.float64)
        method = str(primary_stats.get("method", "unknown"))

        if secondary is not None and board_refine_alpha > 0.0:
            secondary = np.asarray(secondary, dtype=np.float64)
            dt_mm = float(np.linalg.norm(T_final[:3, 3] - secondary[:3, 3]) * 1000.0)
            dr_deg = rotation_error_deg(T_final[:3, :3], secondary[:3, :3])
            if dt_mm <= float(max_refine_dt_mm) and dr_deg <= float(max_refine_dr_deg):
                T_final = blend_rigid_transforms(T_final, secondary, float(board_refine_alpha))
                method = f"{method}+board_refined"
                primary_stats["board_refine_delta_mm"] = dt_mm
                primary_stats["board_refine_delta_deg"] = dr_deg

        primary_stats["method"] = method
        merged[int(ci)] = T_final
        merged_stats[key] = primary_stats

    return merged, merged_stats


def score_handeye_with_cube_support(
    meta: dict,
    robot_T: Dict[int, np.ndarray],
    pnp_obs: Dict[int, Dict[int, dict]],
    gripper_cam_idx: int,
    fixed_cam_ids: List[int],
    T_gTc: np.ndarray,
    min_cams: int = 2,
):
    _, gripper_anchor_events, _ = build_gripper_cube_anchor(
        robot_T, pnp_obs, gripper_cam_idx, T_gTc, gripper_base_by_event=None)
    if not gripper_anchor_events:
        return {
            "cube_score": 1e9,
            "cube_fixed_support": 0,
            "cube_anchor_support": 0,
            "cube_base_stability": {},
            "cube_anchor_stability": {},
        }

    cube_T_base_Ci, cube_base_stats = estimate_fixed_cameras_from_cube_anchor(
        pnp_obs, fixed_cam_ids, gripper_anchor_events)
    if not cube_T_base_Ci:
        return {
            "cube_score": 1e9,
            "cube_fixed_support": 0,
            "cube_anchor_support": 0,
            "cube_base_stability": {},
            "cube_anchor_stability": {},
        }

    anchor_avg, _, anchor_diag = build_cube_anchor_from_fixed_cams(
        meta, pnp_obs, cube_T_base_Ci, sorted(cube_T_base_Ci.keys()), gripper_cam_idx,
        min_cams=min_cams)
    base_rows = list(cube_base_stats.values())
    mean_base_trans = float(np.mean([float(r.get("translation_std_mm", 1e9)) for r in base_rows])) if base_rows else 1e9
    mean_base_rot = float(np.mean([float(r.get("rotation_std_deg", 1e9)) for r in base_rows])) if base_rows else 1e9
    anchor_st = anchor_diag.get("stability", {}) if anchor_avg is not None else {}
    anchor_trans = float(anchor_st.get("translation_std_mm", 1e9))
    anchor_rot = float(anchor_st.get("rotation_std_deg", 1e9))
    cube_score = mean_base_trans + 10.0 * mean_base_rot + 0.5 * anchor_trans + 5.0 * anchor_rot
    return {
        "cube_score": float(cube_score),
        "cube_fixed_support": int(len(cube_T_base_Ci)),
        "cube_anchor_support": int(anchor_diag.get("support", 0)),
        "cube_base_stability": {
            "mean_translation_std_mm": mean_base_trans,
            "mean_rotation_std_deg": mean_base_rot,
        },
        "cube_anchor_stability": anchor_st,
    }


def build_gripper_event_base_transforms_from_fixed_cams(
    meta: dict,
    pnp_obs: Dict[int, Dict[int, dict]],
    T_base_Ci: Dict[int, np.ndarray],
    fixed_cam_ids: List[int],
    gripper_cam_idx: int,
    robot_T: Dict[int, np.ndarray],
    T_gTc: np.ndarray,
    min_cams: int = 2,
    max_fixed_spread_mm: float = 15.0,
    max_fixed_spread_deg: float = 2.0,
    extra_transforms: Optional[Dict[str, Any]] = None,
):
    transforms: Dict[str, Any] = dict(extra_transforms or {})
    transforms.update({
        f"T_base_C{int(ci)}": np.asarray(T, dtype=np.float64)
        for ci, T in T_base_Ci.items()
        if int(ci) in fixed_cam_ids
    })
    profile_kwargs = cube_selection_profile_kwargs("default")
    event_base_map: Dict[int, np.ndarray] = {}
    diag_events: List[dict] = []

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        if eid not in pnp_obs.get(gripper_cam_idx, {}):
            continue
        if eid not in robot_T:
            continue

        fixed_candidate_map: Dict[int, List[dict]] = {}
        for ci in fixed_cam_ids:
            tf_key = f"T_base_C{int(ci)}"
            if tf_key not in transforms:
                continue
            obs = pnp_obs.get(int(ci), {}).get(eid)
            if obs is None:
                continue
            cands = obs.get("_candidates") or [obs]
            payload = []
            for cand in cands:
                T_C_O = np.asarray(cand.get("T_C_O"), dtype=np.float64)
                if T_C_O.shape != (4, 4) or not np.all(np.isfinite(T_C_O)):
                    continue
                payload.append({
                    "T_C_O": T_C_O,
                    "err_mean": float(cand.get("err_mean", 99.0)),
                    "n_points": int(cand.get("n_points", 4)),
                    "used_ids": [int(x) for x in cand.get("used_ids", [])],
                    "source": str(cand.get("source", "unknown")),
                    **copy_depth_fields(cand),
                })
            if payload:
                fixed_candidate_map[int(ci)] = payload

        if len(fixed_candidate_map) < max(int(min_cams), 1):
            continue

        refined_fixed = select_consistent_event_cube_candidates(
            cap, fixed_candidate_map, transforms, None, **profile_kwargs)
        if len(refined_fixed) < max(int(min_cams), 1):
            continue

        fixed_object_poses = []
        fixed_weights = []
        fixed_details = {}
        for ci, cand in refined_fixed.items():
            T_base_cam = transforms.get(f"T_base_C{int(ci)}")
            if T_base_cam is None:
                continue
            T_base_O = np.asarray(T_base_cam, dtype=np.float64) @ np.asarray(cand["T_C_O"], dtype=np.float64)
            fixed_object_poses.append(T_base_O)
            fixed_weights.append(candidate_weight(cand))
            fixed_details[str(int(ci))] = {
                "used_ids": [int(x) for x in cand.get("used_ids", [])],
                "source": str(cand.get("source", "unknown")),
                "err_mean": float(cand.get("err_mean", 99.0)),
            }

        if len(fixed_object_poses) < max(int(min_cams), 1):
            continue

        T_base_O_fixed = weighted_se3_average(fixed_object_poses, fixed_weights)
        fixed_spread_mm = [
            float(np.linalg.norm(T[:3, 3] - T_base_O_fixed[:3, 3]) * 1000.0)
            for T in fixed_object_poses
        ]
        fixed_spread_deg = [
            rotation_error_deg(T[:3, :3], T_base_O_fixed[:3, :3])
            for T in fixed_object_poses
        ]
        fixed_spread_mean_mm = float(np.mean(fixed_spread_mm)) if fixed_spread_mm else 0.0
        fixed_spread_mean_deg = float(np.mean(fixed_spread_deg)) if fixed_spread_deg else 0.0
        if fixed_spread_mean_mm > float(max_fixed_spread_mm):
            continue
        if fixed_spread_mean_deg > float(max_fixed_spread_deg):
            continue

        gripper_obs = pnp_obs[gripper_cam_idx][eid]
        gripper_candidates = gripper_obs.get("_candidates") or [gripper_obs]
        gripper_best = select_primary_cube_candidate(gripper_candidates)
        if gripper_best is None:
            continue

        T_C_O_gripper = np.asarray(gripper_best["T_C_O"], dtype=np.float64)
        T_base_cam_corrected = T_base_O_fixed @ inv_T(T_C_O_gripper)
        T_base_cam_nominal = robot_T[eid] @ T_gTc
        T_delta = T_base_cam_corrected @ inv_T(T_base_cam_nominal)

        event_base_map[eid] = T_base_cam_corrected
        diag_events.append({
            "event_id": int(eid),
            "set_index": get_capture_set_index(cap),
            "support": int(len(fixed_object_poses)),
            "fixed_spread_mean_mm": fixed_spread_mean_mm,
            "fixed_spread_mean_deg": fixed_spread_mean_deg,
            "delta_translation_mm": float(np.linalg.norm(T_delta[:3, 3]) * 1000.0),
            "delta_rotation_deg": rotation_error_deg(T_delta[:3, :3], np.eye(3, dtype=np.float64)),
            "delta_translation_xyz_mm": (T_delta[:3, 3] * 1000.0).tolist(),
            "fixed_cameras": fixed_details,
            "gripper_used_ids": [int(x) for x in gripper_best.get("used_ids", [])],
            "gripper_source": str(gripper_best.get("source", "unknown")),
            "gripper_err_mean": float(gripper_best.get("err_mean", 99.0)),
        })

    diag = {
        "support": int(len(event_base_map)),
        "events": diag_events,
    }
    return event_base_map, diag


def blend_rigid_transforms(T_a: np.ndarray,
                           T_b: np.ndarray,
                           alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return np.asarray(T_a, dtype=np.float64).copy()
    if alpha >= 1.0:
        return np.asarray(T_b, dtype=np.float64).copy()
    T_a = np.asarray(T_a, dtype=np.float64)
    T_b = np.asarray(T_b, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (1.0 - alpha) * T_a[:3, 3] + alpha * T_b[:3, 3]
    R_mean = (1.0 - alpha) * T_a[:3, :3] + alpha * T_b[:3, :3]
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T[:3, :3] = R
    return T


def refine_gripper_event_base_transforms_with_board_anchor(
    gripper_base_by_event: Dict[int, np.ndarray],
    charuco_obs: Dict[int, dict],
    blend_alpha: float = 0.8,
):
    blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    if blend_alpha <= 0.0:
        return gripper_base_by_event, {
            "support": 0,
            "blend_alpha": blend_alpha,
            "enabled": False,
            "events": [],
        }

    overlap = sorted(set(int(eid) for eid in gripper_base_by_event.keys()) & set(int(eid) for eid in charuco_obs.keys()))
    if len(overlap) < 3:
        return gripper_base_by_event, {
            "support": int(len(overlap)),
            "blend_alpha": blend_alpha,
            "enabled": False,
            "events": [],
        }

    board_list = []
    board_weights = []
    for eid in overlap:
        T_base_cam = np.asarray(gripper_base_by_event[eid], dtype=np.float64)
        T_cam_board = np.asarray(charuco_obs[eid]["T_cam_board"], dtype=np.float64)
        board_list.append(T_base_cam @ T_cam_board)
        board_weights.append(1.0 / max(float(charuco_obs[eid].get("reproj", 1.0)), 1e-9))

    T_base_board_ref, st_board = robust_weighted_se3_average(board_list, board_weights, return_stats=True)
    refined = {int(eid): np.asarray(T, dtype=np.float64).copy() for eid, T in gripper_base_by_event.items()}
    event_rows = []
    for eid in overlap:
        T_old = np.asarray(gripper_base_by_event[eid], dtype=np.float64)
        T_cam_board = np.asarray(charuco_obs[eid]["T_cam_board"], dtype=np.float64)
        T_board_aligned = T_base_board_ref @ inv_T(T_cam_board)
        T_new = blend_rigid_transforms(T_old, T_board_aligned, blend_alpha)
        refined[eid] = T_new
        T_delta = T_new @ inv_T(T_old)
        event_rows.append({
            "event_id": int(eid),
            "delta_translation_mm": float(np.linalg.norm(T_delta[:3, 3]) * 1000.0),
            "delta_rotation_deg": rotation_error_deg(T_delta[:3, :3], np.eye(3, dtype=np.float64)),
            "charuco_reproj_px": float(charuco_obs[eid].get("reproj", 1.0)),
            "charuco_corners": int(charuco_obs[eid].get("n_corners", 0)),
        })

    return refined, {
        "support": int(len(overlap)),
        "blend_alpha": blend_alpha,
        "enabled": True,
        "board_anchor_stability": st_board,
        "events": event_rows,
    }


def build_gripper_base_pose_model(meta: dict,
                                  robot_T: Dict[int, np.ndarray],
                                  T_gTc: np.ndarray,
                                  gripper_base_by_event: Dict[int, np.ndarray],
                                  gripper_cam_idx: int):
    samples = []
    pose_vectors = []

    cap_by_eid = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid >= 0:
            cap_by_eid[eid] = cap

    for eid, T_base_cam_corr in sorted(gripper_base_by_event.items()):
        if eid not in cap_by_eid or eid not in robot_T:
            continue
        cap = cap_by_eid[eid]
        pose6 = try_parse_pose6(cap.get("capture_pose_6dof"))
        if pose6 is None:
            pose6 = try_parse_pose6(cap.get("robot_pose_6dof"))
        if pose6 is None:
            continue

        T_base_cam_nom = np.asarray(robot_T[eid], dtype=np.float64) @ np.asarray(T_gTc, dtype=np.float64)
        T_delta = np.asarray(T_base_cam_corr, dtype=np.float64) @ inv_T(T_base_cam_nom)
        rvec_delta, _ = cv2.Rodrigues(np.asarray(T_delta[:3, :3], dtype=np.float64))
        pose_vec = np.asarray(pose6, dtype=np.float64).reshape(6)
        pose_vectors.append(pose_vec)
        samples.append({
            "event_id": int(eid),
            "set_index": get_capture_set_index(cap),
            "robot_pose_6dof": pose_vec.tolist(),
            "delta_translation_mm": (np.asarray(T_delta[:3, 3], dtype=np.float64) * 1000.0).tolist(),
            "delta_rvec_rad": np.asarray(rvec_delta, dtype=np.float64).reshape(3).tolist(),
        })

    if not samples:
        return None

    X = np.asarray(pose_vectors, dtype=np.float64)
    feature_scale = np.std(X, axis=0) if len(X) >= 2 else np.ones((6,), dtype=np.float64)
    min_scale = np.asarray([50.0, 50.0, 10.0, 15.0, 5.0, 15.0], dtype=np.float64)
    feature_scale = np.maximum(feature_scale, min_scale)

    return {
        "model_type": "pose_knn_delta_v1",
        "gripper_cam_idx": int(gripper_cam_idx),
        "support": int(len(samples)),
        "feature_order": ["x_mm", "y_mm", "z_mm", "rz_deg", "ry_deg", "rx_deg"],
        "feature_scale": feature_scale.tolist(),
        "exact_match_translation_tol_mm": 5.0,
        "exact_match_rotation_tol_deg": 5.0,
        "top_k": 3,
        "kernel_sigma": 1.0,
        "max_normalized_dist": 2.5,
        "samples": samples,
    }


def load_nominal_set_cube_transforms(meta: dict) -> Dict[int, np.ndarray]:
    priors: Dict[int, np.ndarray] = {}
    for cap in meta.get("captures", []):
        set_index = get_capture_set_index(cap)
        if set_index is None or int(set_index) in priors:
            continue
        T_nominal = get_capture_set_cube_center_transform_raw(cap)
        if T_nominal is not None:
            priors[int(set_index)] = np.asarray(T_nominal, dtype=np.float64)
    return priors


def build_event_pose_map_from_set_priors(meta: dict,
                                         set_priors_by_set: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    event_pose_map: Dict[int, np.ndarray] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        set_index = get_capture_set_index(cap)
        if set_index is None:
            continue
        T_prior = set_priors_by_set.get(int(set_index))
        if T_prior is None:
            continue
        event_pose_map[int(eid)] = np.asarray(T_prior, dtype=np.float64)
    return event_pose_map


def estimate_set_cube_prior_alignment(raw_set_priors: Dict[int, np.ndarray],
                                      estimated_by_set: Dict[int, np.ndarray],
                                      set_diag: Optional[dict] = None):
    delta_list: List[np.ndarray] = []
    delta_weights: List[float] = []
    per_set = {}

    for set_index in sorted(set(raw_set_priors.keys()) & set(estimated_by_set.keys())):
        T_raw = np.asarray(raw_set_priors[int(set_index)], dtype=np.float64)
        T_est = np.asarray(estimated_by_set[int(set_index)], dtype=np.float64)
        T_delta = inv_T(T_raw) @ T_est
        support = 1.0
        if set_diag is not None:
            support = float((set_diag.get("per_set", {}) or {}).get(str(int(set_index)), {}).get("support", 1.0))
        delta_list.append(T_delta)
        delta_weights.append(max(support, 1.0))
        per_set[str(int(set_index))] = {
            "support": int(support),
            "raw_transform": T_raw.tolist(),
            "estimated_transform": T_est.tolist(),
            "delta_transform": T_delta.tolist(),
        }

    if not delta_list:
        return None, {}, {
            "support": 0,
            "stability": {},
            "per_set": {},
        }

    T_delta_avg, st_delta = robust_weighted_se3_average(delta_list, delta_weights, return_stats=True)
    corrected_by_set: Dict[int, np.ndarray] = {}
    for set_index, T_raw in raw_set_priors.items():
        corrected_by_set[int(set_index)] = np.asarray(T_raw, dtype=np.float64) @ np.asarray(T_delta_avg, dtype=np.float64)

    residual_rows = {}
    for set_index in sorted(set(raw_set_priors.keys()) & set(estimated_by_set.keys())):
        T_corr = corrected_by_set[int(set_index)]
        T_est = np.asarray(estimated_by_set[int(set_index)], dtype=np.float64)
        residual_rows[str(int(set_index))] = {
            "dt_mm": float(np.linalg.norm(T_corr[:3, 3] - T_est[:3, 3]) * 1000.0),
            "dr_deg": rotation_error_deg(T_corr[:3, :3], T_est[:3, :3]),
        }
        per_set[str(int(set_index))].update(residual_rows[str(int(set_index))])

    diag = {
        "support": int(len(delta_list)),
        "stability": st_delta,
        "T_set_cube_center_to_object": np.asarray(T_delta_avg, dtype=np.float64).tolist(),
        "per_set": per_set,
    }
    return T_delta_avg, corrected_by_set, diag


def build_setwise_cube_anchors(meta: dict,
                               event_pose_map: Dict[int, np.ndarray],
                               set_prior_by_set: Optional[Dict[int, np.ndarray]] = None,
                               prior_blend_alpha: float = 0.25,
                               max_prior_dt_mm: float = 35.0,
                               max_prior_dr_deg: float = 8.0):
    poses_by_set: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    events_without_set_index: List[int] = []

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0 or eid not in event_pose_map:
            continue
        set_index = get_capture_set_index(cap)
        if set_index is None:
            events_without_set_index.append(int(eid))
            continue
        poses_by_set[int(set_index)].append((eid, np.asarray(event_pose_map[eid], dtype=np.float64)))

    transforms_by_set: Dict[int, np.ndarray] = {}
    diag = {
        "num_sets": 0,
        "set_indices": [],
        "per_set": {},
        "events_without_set_index": [int(eid) for eid in events_without_set_index],
        "global_average_is_compatibility_only": False,
    }

    for set_index in sorted(poses_by_set):
        items = poses_by_set[set_index]
        if not items:
            continue
        Ts = [T for _, T in items]
        T_avg, st = robust_weighted_se3_average(Ts, return_stats=True)
        if set_prior_by_set is not None and int(set_index) in set_prior_by_set:
            T_prior = np.asarray(set_prior_by_set[int(set_index)], dtype=np.float64)
            dt_mm = float(np.linalg.norm(T_avg[:3, 3] - T_prior[:3, 3]) * 1000.0)
            dr_deg = rotation_error_deg(T_avg[:3, :3], T_prior[:3, :3])
            if dt_mm <= float(max_prior_dt_mm) and dr_deg <= float(max_prior_dr_deg):
                T_avg = blend_rigid_transforms(T_avg, T_prior, float(prior_blend_alpha))
                st["prior_blend_dt_mm"] = dt_mm
                st["prior_blend_dr_deg"] = dr_deg
                st["prior_blend_alpha"] = float(prior_blend_alpha)
        transforms_by_set[int(set_index)] = T_avg
        diag["per_set"][str(int(set_index))] = {
            "support": int(len(items)),
            "events": [int(eid) for eid, _ in items],
            "stability": st,
        }

    diag["num_sets"] = int(len(transforms_by_set))
    diag["set_indices"] = [int(x) for x in sorted(transforms_by_set)]
    diag["global_average_is_compatibility_only"] = bool(len(transforms_by_set) > 1)
    return transforms_by_set, diag


def build_hybrid_setwise_cube_anchors(meta: dict,
                                      fixed_event_pose_map: Dict[int, np.ndarray],
                                      gripper_event_pose_map: Dict[int, np.ndarray],
                                      set_prior_by_set: Optional[Dict[int, np.ndarray]] = None,
                                      prior_blend_alpha: float = 0.25,
                                      max_prior_dt_mm: float = 35.0,
                                      max_prior_dr_deg: float = 8.0):
    fixed_by_set: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    gripper_by_set: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    events_without_set_index: List[int] = []

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        set_index = get_capture_set_index(cap)
        if set_index is None:
            if eid in fixed_event_pose_map or eid in gripper_event_pose_map:
                events_without_set_index.append(int(eid))
            continue
        if eid in fixed_event_pose_map:
            fixed_by_set[int(set_index)][eid] = np.asarray(fixed_event_pose_map[eid], dtype=np.float64)
        if eid in gripper_event_pose_map:
            gripper_by_set[int(set_index)][eid] = np.asarray(gripper_event_pose_map[eid], dtype=np.float64)

    transforms_by_set: Dict[int, np.ndarray] = {}
    diag = {
        "num_sets": 0,
        "set_indices": [],
        "per_set": {},
        "events_without_set_index": [int(eid) for eid in events_without_set_index],
        "global_average_is_compatibility_only": False,
        "strategy": "fixed_translation_gripper_rotation",
    }

    all_sets = sorted(set(fixed_by_set.keys()) | set(gripper_by_set.keys()))
    for set_index in all_sets:
        fixed_items = fixed_by_set.get(set_index, {})
        gripper_items = gripper_by_set.get(set_index, {})
        event_ids = sorted(set(fixed_items.keys()) | set(gripper_items.keys()))
        if not event_ids:
            continue

        hybrid_events: List[Tuple[int, np.ndarray]] = []
        for eid in event_ids:
            T_fixed = fixed_items.get(eid)
            T_gripper = gripper_items.get(eid)
            T_evt = np.eye(4, dtype=np.float64)
            if T_gripper is not None:
                T_evt[:3, :3] = T_gripper[:3, :3]
            elif T_fixed is not None:
                T_evt[:3, :3] = T_fixed[:3, :3]
            if T_fixed is not None:
                T_evt[:3, 3] = T_fixed[:3, 3]
            elif T_gripper is not None:
                T_evt[:3, 3] = T_gripper[:3, 3]
            hybrid_events.append((eid, T_evt))

        T_avg, st = robust_weighted_se3_average(
            [T for _, T in hybrid_events], return_stats=True)
        if set_prior_by_set is not None and int(set_index) in set_prior_by_set:
            T_prior = np.asarray(set_prior_by_set[int(set_index)], dtype=np.float64)
            dt_mm = float(np.linalg.norm(T_avg[:3, 3] - T_prior[:3, 3]) * 1000.0)
            dr_deg = rotation_error_deg(T_avg[:3, :3], T_prior[:3, :3])
            if dt_mm <= float(max_prior_dt_mm) and dr_deg <= float(max_prior_dr_deg):
                T_avg = blend_rigid_transforms(T_avg, T_prior, float(prior_blend_alpha))
                st["prior_blend_dt_mm"] = dt_mm
                st["prior_blend_dr_deg"] = dr_deg
                st["prior_blend_alpha"] = float(prior_blend_alpha)
        transforms_by_set[int(set_index)] = T_avg

        fixed_st = None
        if fixed_items:
            _, fixed_st = robust_weighted_se3_average(
                [T for T in fixed_items.values()], return_stats=True)
        gripper_st = None
        if gripper_items:
            _, gripper_st = robust_weighted_se3_average(
                [T for T in gripper_items.values()], return_stats=True)

        diag["per_set"][str(int(set_index))] = {
            "support": int(len(hybrid_events)),
            "events": [int(eid) for eid, _ in hybrid_events],
            "stability": st,
            "source": "fixed_translation+gripper_rotation",
            "fixed_support": int(len(fixed_items)),
            "gripper_support": int(len(gripper_items)),
            "fixed_stability": fixed_st,
            "gripper_stability": gripper_st,
        }

    diag["num_sets"] = int(len(transforms_by_set))
    diag["set_indices"] = [int(x) for x in sorted(transforms_by_set)]
    diag["global_average_is_compatibility_only"] = bool(len(transforms_by_set) > 1)
    return transforms_by_set, diag


def main():
    parser = argparse.ArgumentParser(description="Unified calibration (ChArUco hand-eye + cube multi-cam)")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--gripper_cam_idx", type=int, default=None)
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=None)
    parser.add_argument("--handeye_method", type=str, default="AUTO")
    parser.add_argument("--gripper_cube_min_markers", type=int, default=1)
    parser.add_argument("--gripper_cube_min_aspect", type=float, default=0.35)
    parser.add_argument("--event_anchor_min_cams", type=int, default=2)
    parser.add_argument("--gripper_board_blend_alpha", type=float, default=0.8)
    parser.add_argument("--common_object_mode", type=str, default="cube_primary",
                        choices=["cube_primary", "board_primary"])
    parser.add_argument("--fixed_board_refine_alpha", type=float, default=0.35)
    parser.add_argument("--enable_depth_pose_candidate", action="store_true",
                        help="Experimental: include depth-SVD cube pose candidates; can improve dimension metrics but may reduce hand-eye stability")
    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir
    out_dir = args.out_dir or os.path.join(root, "calib_out")
    ensure_dir(out_dir)
    cleanup_legacy_public_outputs(out_dir)
    internal_runtime_dir = ensure_dir(os.path.join(out_dir, "internal_runtime"))

    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)
    nominal_set_cube_priors = load_nominal_set_cube_transforms(meta)
    if nominal_set_cube_priors:
        print(f"[INFO] set_cube_center priors: {sorted(int(x) for x in nominal_set_cube_priors.keys())}")
    else:
        print("[INFO] set_cube_center priors: none")

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
    K_map, D_map, depth_scale_map = {}, {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], depth_scale_map[ci] = load_intrinsics_with_depth_scale(intr_dir, ci)

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
    cfg = get_default_cube_config()
    cube = ArucoCubeTarget(cfg)

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in pnp_obs or not cinfo.get("saved"):
                continue
            max_err = 5.0 if ci == gripper_cam_idx else 3.0
            min_markers = args.gripper_cube_min_markers if ci == gripper_cam_idx else 1
            min_aspect = args.gripper_cube_min_aspect if ci == gripper_cam_idx else 0.0

            candidates = stored_cube_pose_candidates(
                cinfo, ci, gripper_cam_idx,
                max_err=max_err,
                min_markers=min_markers,
                min_aspect=min_aspect,
            )

            rgb_path = os.path.join(root, cinfo.get("rgb_path", ""))
            img = cv2.imread(rgb_path) if rgb_path else None
            if img is not None:
                depth = None
                depth_rel = cinfo.get("depth_path", "")
                if depth_rel:
                    depth = cv2.imread(os.path.join(root, depth_rel), cv2.IMREAD_UNCHANGED)
                candidates = estimate_image_cube_pose_candidates(
                    cube, img, K_map[ci], D_map[ci],
                    max_err=max_err,
                    min_markers=min_markers,
                    min_aspect=min_aspect,
                    depth_u16=depth,
                    depth_scale=depth_scale_map.get(ci),
                    include_depth_pose_candidate=bool(args.enable_depth_pose_candidate),
                ) + candidates

            candidates = filter_candidates_for_camera_role(candidates, ci, gripper_cam_idx)

            if not candidates:
                continue

            best = select_primary_cube_candidate(candidates)
            if best is None:
                continue

            pnp_obs[ci][eid] = {
                "T_C_O": np.asarray(best["T_C_O"], dtype=np.float64),
                "err_mean": float(best["err_mean"]),
                "n_points": int(best.get("n_points", 4)),
                "used_ids": [int(x) for x in best.get("used_ids", [])],
                "source": str(best.get("source", "unknown")),
                **copy_depth_fields(best),
                "_candidates": [
                    {
                        "T_C_O": np.asarray(cand["T_C_O"], dtype=np.float64),
                        "err_mean": float(cand.get("err_mean", 99.0)),
                        "n_points": int(cand.get("n_points", 4)),
                        "used_ids": [int(x) for x in cand.get("used_ids", [])],
                        "source": str(cand.get("source", "unknown")),
                        **copy_depth_fields(cand),
                    }
                    for cand in candidates
                ],
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

            board_score = float(np.mean(trans_mm)) + 10.0 * float(np.mean(rot_deg))
            cube_score_info = score_handeye_with_cube_support(
                meta, robot_T, pnp_obs, gripper_cam_idx, fixed_cam_ids, T_gTc,
                min_cams=max(int(args.event_anchor_min_cams), 2),
            )
            if str(args.common_object_mode) == "cube_primary":
                score = board_score + float(cube_score_info.get("cube_score", 0.0))
            else:
                score = board_score
            method_results[mname] = {
                "T_gTc": T_gTc,
                "score": score,
                "board_score": board_score,
                "mean_trans_mm": float(np.mean(trans_mm)),
                "mean_rot_deg": float(np.mean(rot_deg)),
                "stability": st_bo,
                "cube_support": cube_score_info,
            }
            print(
                f"  [{mname}] score={score:.3f} "
                f"(board={board_score:.3f}, cube={cube_score_info.get('cube_score', 0.0):.3f}) "
                f"trans={np.mean(trans_mm):.2f}mm rot={np.mean(rot_deg):.3f}deg"
            )
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

    # ══════════════════════════════════════════════════════════
    # STEP D: Fixed cameras in robot base frame
    # D-1: Board-based (T_base_board[eid] @ inv(T_Ci_board)) - precise
    # D-2: Cube PnP chaining - fallback for cameras that can't see board
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("[STEP-D] Fixed cameras in robot base frame")
    print("=" * 60)

    from charuco_utils import CharucoTarget as _CT
    from config import CharucoBoardConfig as _CBC
    charuco_det = _CT(_CBC())

    # Per-event T_base_board from gripper camera (precise!)
    T_base_board_per_event = {}
    for eid in common_he:
        if eid not in charuco_obs or eid not in robot_T:
            continue
        T_base_board_per_event[eid] = robot_T[eid] @ T_gTc @ charuco_obs[eid]["T_cam_board"]

    board_T_base_Ci = {}
    board_base_stats = {}

    # D-1: Board-based calibration
    print("  --- D-1: Board-based (ChArUco) ---")
    for ci in fixed_cam_ids:
        if ci not in K_map:
            continue
        Ts, ws = [], []
        for cap in meta.get("captures", []):
            eid = int(cap.get("event_id", -1))
            if eid not in T_base_board_per_event:
                continue
            cinfo = cap.get("cams", {}).get(str(ci), {})
            rgb_path = os.path.join(root, cinfo.get("rgb_path", ""))
            img = cv2.imread(rgb_path)
            if img is None:
                continue
            try:
                ok, rv, tv, n_cor, reproj = charuco_det.estimate_pose(img, K_map[ci], D_map[ci])
            except Exception:
                ok = False
            if ok and n_cor >= 6:
                T_cam_board = rodrigues_to_Rt(rv, tv)
                T_est = T_base_board_per_event[eid] @ inv_T(T_cam_board)
                Ts.append(T_est)
                ws.append(1.0 / max(float(reproj), 1e-9))

        if len(Ts) >= 3:
            T_avg, st = robust_weighted_se3_average(Ts, ws, return_stats=True)
            st["method"] = "board_based"
            st["support"] = int(st.get("num_inliers", len(Ts)))
            st["total_keys"] = int(len(Ts))
            board_T_base_Ci[ci] = T_avg
            board_base_stats[f"T_base_C{ci}"] = st
            print(f"  T_base_C{ci}: {st['num_inliers']}/{len(Ts)}fr (board) "
                  f"rot={st['rotation_std_deg']:.3f}deg trans={st['translation_std_mm']:.2f}mm")
        else:
            print(f"  cam{ci}: board not visible ({len(Ts)} frames)")

    print("  --- D-1b: Cube-primary fixed camera base transforms ---")
    gripper_anchor_nominal, gripper_anchor_events_nominal, gripper_anchor_diag_nominal = build_gripper_cube_anchor(
        robot_T, pnp_obs, gripper_cam_idx, T_gTc, gripper_base_by_event=None)
    cube_T_base_Ci, cube_base_stats = estimate_fixed_cameras_from_cube_anchor(
        pnp_obs, fixed_cam_ids, gripper_anchor_events_nominal)
    for ci in fixed_cam_ids:
        key = f"T_base_C{ci}"
        st = cube_base_stats.get(key)
        if st is None:
            continue
        print(f"  {key}: {st.get('support', 0)}fr (cube) "
              f"rot={st['rotation_std_deg']:.3f}deg trans={st['translation_std_mm']:.2f}mm")

    T_base_Ci, base_stats = merge_fixed_camera_base_transforms(
        cube_T_base_Ci,
        cube_base_stats,
        board_T_base_Ci,
        board_base_stats,
        mode=str(args.common_object_mode),
        board_refine_alpha=float(args.fixed_board_refine_alpha),
    )
    for ci in fixed_cam_ids:
        key = f"T_base_C{ci}"
        T = T_base_Ci.get(ci)
        if T is None:
            continue
        np.save(os.path.join(out_dir, f"{key}.npy"), np.asarray(T, dtype=np.float64))
        st = base_stats.get(key, {})
        print(f"  [FINAL] {key}: method={st.get('method', 'unknown')} "
              f"rot={st.get('rotation_std_deg', 0.0):.3f}deg trans={st.get('translation_std_mm', 0.0):.2f}mm")

    # Build object anchor from the fixed-camera cube consensus.
    print("  --- D-1.5: Object anchor from fixed-camera cube consensus ---")
    T_B_O_avg = None
    T_B_O_by_event: Dict[int, np.ndarray] = {}
    cube_anchor_diag = {
        "support": 0,
        "total_keys": 0,
        "stability": {},
        "events": [],
    }
    t_base_o_source = "uninitialized"
    gripper_anchor = None
    gripper_anchor_events: Dict[int, np.ndarray] = {}
    gripper_anchor_diag = {
        "support": 0,
        "total_keys": 0,
        "stability": {},
        "events": [],
    }
    gripper_base_by_event: Dict[int, np.ndarray] = {}
    gripper_base_diag = {
        "support": 0,
        "events": [],
    }
    gripper_pose_model = None
    set_cube_center_prior = None
    set_cube_center_prior_diag = {
        "support": 0,
        "stability": {},
        "per_set": {},
    }

    gripper_base_by_event, gripper_base_diag = build_gripper_event_base_transforms_from_fixed_cams(
        meta, pnp_obs, T_base_Ci, sorted(T_base_Ci.keys()), gripper_cam_idx,
        robot_T, T_gTc, min_cams=max(int(args.event_anchor_min_cams), 2),
    )
    gripper_base_by_event, gripper_board_refine_diag = refine_gripper_event_base_transforms_with_board_anchor(
        gripper_base_by_event,
        charuco_obs,
        blend_alpha=float(args.gripper_board_blend_alpha),
    )
    gripper_base_diag["board_anchor_refinement"] = gripper_board_refine_diag
    for eid, T_evt in sorted(gripper_base_by_event.items()):
        np.save(
            os.path.join(internal_runtime_dir, f"T_base_C{gripper_cam_idx}_event{int(eid)}.npy"),
            np.asarray(T_evt, dtype=np.float64),
        )
    if gripper_base_diag.get("events"):
        with open(os.path.join(internal_runtime_dir, f"T_base_C{gripper_cam_idx}_by_event.json"), "w") as f:
            json.dump(gripper_base_diag, f, indent=2)
        print(
            f"  gripper event transforms from fixed consensus: "
            f"{gripper_base_diag.get('support', 0)} events"
        )
    gripper_pose_model = build_gripper_base_pose_model(
        meta, robot_T, T_gTc, gripper_base_by_event, gripper_cam_idx)
    if gripper_pose_model is not None:
        with open(os.path.join(internal_runtime_dir, "gripper_base_pose_model.json"), "w") as f:
            json.dump(gripper_pose_model, f, indent=2)
        print(f"  internal_runtime/gripper_base_pose_model.json ({gripper_pose_model.get('support', 0)} samples)")

    gripper_anchor, gripper_anchor_events, gripper_anchor_diag = build_gripper_cube_anchor(
        robot_T, pnp_obs, gripper_cam_idx, T_gTc, gripper_base_by_event=gripper_base_by_event)

    board_anchor, board_anchor_events, board_anchor_diag = build_cube_anchor_from_fixed_cams(
        meta, pnp_obs, T_base_Ci, sorted(T_base_Ci.keys()), gripper_cam_idx,
        min_cams=args.event_anchor_min_cams,
    )
    if board_anchor is not None:
        T_B_O_avg = board_anchor
        T_B_O_by_event = board_anchor_events
        cube_anchor_diag = board_anchor_diag
        t_base_o_source = "cube_consensus_fixed_cams"
        print(
            f"  T_base_O from cube consensus: "
            f"{cube_anchor_diag.get('support', 0)}/{cube_anchor_diag.get('total_keys', 0)} events"
        )
    else:
        if gripper_anchor is not None:
            T_B_O_avg = gripper_anchor
            T_B_O_by_event = gripper_anchor_events
            cube_anchor_diag = gripper_anchor_diag
            t_base_o_source = "gripper_cube_average_fallback"
            print(
                f"  [WARN] Falling back to gripper-only cube anchor: "
                f"{cube_anchor_diag.get('support', 0)}/{cube_anchor_diag.get('total_keys', 0)} events"
            )
        else:
            T_B_O_avg = np.eye(4, dtype=np.float64)
            t_base_o_source = "identity_fallback"
            print("  [WARN] No reliable cube anchor available; using identity fallback")

    set_anchor_source = t_base_o_source
    if t_base_o_source == "cube_consensus_fixed_cams" and gripper_anchor_events:
        T_B_O_by_set, cube_anchor_by_set_diag = build_hybrid_setwise_cube_anchors(
            meta, T_B_O_by_event, gripper_anchor_events)
        if T_B_O_by_set:
            set_anchor_source = "hybrid_fixed_translation_gripper_rotation"
            cube_anchor_diag["setwise_strategy"] = "fixed_translation_gripper_rotation"
            cube_anchor_diag["gripper_anchor"] = gripper_anchor_diag
    else:
        T_B_O_by_set, cube_anchor_by_set_diag = build_setwise_cube_anchors(meta, T_B_O_by_event)
    if T_B_O_by_set:
        if len(T_B_O_by_set) > 1:
            T_B_O_avg = weighted_se3_average([T_B_O_by_set[s] for s in sorted(T_B_O_by_set)])
        else:
            T_B_O_avg = next(iter(T_B_O_by_set.values()))
        cube_anchor_diag["set_indices"] = [int(x) for x in sorted(T_B_O_by_set)]
        cube_anchor_diag["multi_set_capture"] = bool(len(T_B_O_by_set) > 1)
        if len(T_B_O_by_set) > 1:
            cube_anchor_diag["note"] = (
                "T_base_O mixes multiple set_index groups. "
                "Set-specific anchors are stored under internal_runtime/."
            )

    if nominal_set_cube_priors and T_B_O_by_set:
        T_set_cube_center_to_object, corrected_set_priors, set_cube_center_prior_diag = estimate_set_cube_prior_alignment(
            nominal_set_cube_priors,
            T_B_O_by_set,
            cube_anchor_by_set_diag,
        )
        if T_set_cube_center_to_object is not None and corrected_set_priors:
            set_cube_center_prior = {
                "source": "set_cube_center_6dof_aligned_to_cube_object",
                "support": int(set_cube_center_prior_diag.get("support", 0)),
                "T_set_cube_center_to_object": np.asarray(T_set_cube_center_to_object, dtype=np.float64).tolist(),
                "per_set": set_cube_center_prior_diag.get("per_set", {}),
            }
            with open(os.path.join(internal_runtime_dir, "set_cube_center_prior.json"), "w") as f:
                json.dump(set_cube_center_prior, f, indent=2)
            print(
                "  internal_runtime/set_cube_center_prior.json "
                f"(support={set_cube_center_prior.get('support', 0)})"
            )

            print("  --- D-1.6: Set-prior + depth refinement pass ---")
            prior_event_pose_map = build_event_pose_map_from_set_priors(meta, corrected_set_priors)
            cube_T_base_Ci_prior, cube_base_stats_prior = estimate_fixed_cameras_from_cube_anchor(
                pnp_obs, fixed_cam_ids, prior_event_pose_map)
            if cube_T_base_Ci_prior:
                T_base_Ci, base_stats = merge_fixed_camera_base_transforms(
                    cube_T_base_Ci_prior,
                    cube_base_stats_prior,
                    board_T_base_Ci,
                    board_base_stats,
                    mode=str(args.common_object_mode),
                    board_refine_alpha=float(args.fixed_board_refine_alpha),
                )
                for ci in fixed_cam_ids:
                    key = f"T_base_C{ci}"
                    T = T_base_Ci.get(ci)
                    if T is None:
                        continue
                    np.save(os.path.join(out_dir, f"{key}.npy"), np.asarray(T, dtype=np.float64))
                    st = base_stats.get(key, {})
                    print(f"  [PRIOR] {key}: method={st.get('method', 'unknown')} "
                          f"rot={st.get('rotation_std_deg', 0.0):.3f}deg trans={st.get('translation_std_mm', 0.0):.2f}mm")

                extra_runtime_transforms = {
                    "set_cube_center_prior": set_cube_center_prior,
                }
                gripper_base_by_event, gripper_base_diag = build_gripper_event_base_transforms_from_fixed_cams(
                    meta, pnp_obs, T_base_Ci, sorted(T_base_Ci.keys()), gripper_cam_idx,
                    robot_T, T_gTc, min_cams=max(int(args.event_anchor_min_cams), 2),
                    extra_transforms=extra_runtime_transforms,
                )
                gripper_base_by_event, gripper_board_refine_diag = refine_gripper_event_base_transforms_with_board_anchor(
                    gripper_base_by_event,
                    charuco_obs,
                    blend_alpha=float(args.gripper_board_blend_alpha),
                )
                gripper_base_diag["board_anchor_refinement"] = gripper_board_refine_diag
                for eid, T_evt in sorted(gripper_base_by_event.items()):
                    np.save(
                        os.path.join(internal_runtime_dir, f"T_base_C{gripper_cam_idx}_event{int(eid)}.npy"),
                        np.asarray(T_evt, dtype=np.float64),
                    )
                if gripper_base_diag.get("events"):
                    with open(os.path.join(internal_runtime_dir, f"T_base_C{gripper_cam_idx}_by_event.json"), "w") as f:
                        json.dump(gripper_base_diag, f, indent=2)
                gripper_pose_model = build_gripper_base_pose_model(
                    meta, robot_T, T_gTc, gripper_base_by_event, gripper_cam_idx)
                if gripper_pose_model is not None:
                    with open(os.path.join(internal_runtime_dir, "gripper_base_pose_model.json"), "w") as f:
                        json.dump(gripper_pose_model, f, indent=2)

                gripper_anchor, gripper_anchor_events, gripper_anchor_diag = build_gripper_cube_anchor(
                    robot_T, pnp_obs, gripper_cam_idx, T_gTc, gripper_base_by_event=gripper_base_by_event)

                board_anchor, board_anchor_events, board_anchor_diag = build_cube_anchor_from_fixed_cams(
                    meta, pnp_obs, T_base_Ci, sorted(T_base_Ci.keys()), gripper_cam_idx,
                    min_cams=args.event_anchor_min_cams,
                    extra_transforms=extra_runtime_transforms,
                )
                if board_anchor is not None:
                    T_B_O_avg = board_anchor
                    T_B_O_by_event = board_anchor_events
                    cube_anchor_diag = board_anchor_diag
                    t_base_o_source = "cube_consensus_fixed_cams+set_prior_refined"
                elif gripper_anchor is not None:
                    T_B_O_avg = gripper_anchor
                    T_B_O_by_event = gripper_anchor_events
                    cube_anchor_diag = gripper_anchor_diag
                    t_base_o_source = "gripper_cube_average_fallback+set_prior_refined"

                set_anchor_source = t_base_o_source
                if t_base_o_source.startswith("cube_consensus_fixed_cams") and gripper_anchor_events:
                    T_B_O_by_set, cube_anchor_by_set_diag = build_hybrid_setwise_cube_anchors(
                        meta,
                        T_B_O_by_event,
                        gripper_anchor_events,
                        set_prior_by_set=corrected_set_priors,
                    )
                    if T_B_O_by_set:
                        set_anchor_source = "hybrid_fixed_translation_gripper_rotation+set_prior"
                        cube_anchor_diag["setwise_strategy"] = "fixed_translation_gripper_rotation"
                        cube_anchor_diag["gripper_anchor"] = gripper_anchor_diag
                else:
                    T_B_O_by_set, cube_anchor_by_set_diag = build_setwise_cube_anchors(
                        meta,
                        T_B_O_by_event,
                        set_prior_by_set=corrected_set_priors,
                    )
                if T_B_O_by_set:
                    if len(T_B_O_by_set) > 1:
                        T_B_O_avg = weighted_se3_average([T_B_O_by_set[s] for s in sorted(T_B_O_by_set)])
                    else:
                        T_B_O_avg = next(iter(T_B_O_by_set.values()))
                    cube_anchor_diag["set_indices"] = [int(x) for x in sorted(T_B_O_by_set)]
                    cube_anchor_diag["multi_set_capture"] = bool(len(T_B_O_by_set) > 1)
                    cube_anchor_diag["set_cube_center_prior"] = {
                        "support": int(set_cube_center_prior_diag.get("support", 0)),
                    }

    set_anchor_payload = {}
    for set_index in sorted(T_B_O_by_set):
        key = get_object_anchor_key_for_set(set_index)
        T_set = np.asarray(T_B_O_by_set[set_index], dtype=np.float64)
        np.save(os.path.join(internal_runtime_dir, f"{key}.npy"), T_set)
        set_anchor_payload[str(int(set_index))] = {
            "transform_key": key,
            "transform": T_set.tolist(),
            "source": set_anchor_source,
            **cube_anchor_by_set_diag.get("per_set", {}).get(str(int(set_index)), {}),
        }
        print(f"  internal_runtime/{key}.npy ({set_anchor_source}, support={set_anchor_payload[str(int(set_index))].get('support', 0)})")

    if set_anchor_payload:
        with open(os.path.join(internal_runtime_dir, "T_base_O_by_set.json"), "w") as f:
            json.dump(set_anchor_payload, f, indent=2)

    np.save(os.path.join(out_dir, "T_base_O.npy"), T_B_O_avg)
    print(f"  T_base_O.npy ({t_base_o_source})")

    # D-2: Cube PnP chaining for remaining cameras
    remaining = [ci for ci in fixed_cam_ids if ci not in T_base_Ci]

    def _pick_candidate(ci, eid, T_B_O, T_ref=None):
        candidates = pnp_obs[ci][eid].get("_candidates")
        if not candidates or len(candidates) <= 1:
            return pnp_obs[ci][eid]["T_C_O"]
        best_T, best_score = None, 1e9
        for cand in candidates:
            T_sol = np.asarray(cand["T_C_O"], dtype=np.float64)
            err_sol = float(cand.get("err_mean", 99.0))
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

    if remaining:
        print(f"  --- D-2: Cube PnP chaining for {remaining} ---")
        for ci in remaining:
            common = sorted(set(pnp_obs[ci].keys()) & set(T_B_O_by_event.keys()))
            if not common:
                print(f"  [WARN] cam{ci}: no overlap")
                continue

            Ts1, ws1 = [], []
            for eid in common:
                T_C_O = _pick_candidate(ci, eid, T_B_O_by_event[eid])
                Ts1.append(T_B_O_by_event[eid] @ inv_T(T_C_O))
                ws1.append(1.0 / max(pnp_obs[ci][eid]["err_mean"], 1e-9))
            T_rough = robust_weighted_se3_average(Ts1, ws1)

            Ts2, ws2 = [], []
            for eid in common:
                T_C_O = _pick_candidate(ci, eid, T_B_O_by_event[eid], T_ref=T_rough)
                Ts2.append(T_B_O_by_event[eid] @ inv_T(T_C_O))
                ws2.append(1.0 / max(pnp_obs[ci][eid]["err_mean"], 1e-9))

            T_avg, st = robust_weighted_se3_average(Ts2, ws2, return_stats=True)
            st["method"] = "cube_anchor"
            st["support"] = int(len(common))
            st["total_keys"] = int(len(common))
            T_base_Ci[ci] = T_avg
            base_stats[f"T_base_C{ci}"] = st
            np.save(os.path.join(out_dir, f"T_base_C{ci}.npy"), T_avg)
            print(f"  T_base_C{ci}: {len(Ts2)}fr (cube) "
                  f"rot={st['rotation_std_deg']:.3f}deg trans={st['translation_std_mm']:.2f}mm")

    # Step E joint optimization remains deferred until the cube anchor path
    # above is stable enough to avoid reinforcing ambiguous single-face poses.

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    summary = {
        "calibration_type": "unified_charuco_cube",
        "handeye_data_source": "charuco" if use_charuco else "cube_pnp",
        "gripper_cam_idx": int(gripper_cam_idx),
        "ref_fixed_cam_idx": int(ref_fixed) if ref_fixed is not None else None,
        "common_object_mode": str(args.common_object_mode),
        "fixed_cam_ids": [int(x) for x in fixed_cam_ids],
        "all_cam_ids": [int(x) for x in all_cam_ids],
        "selected_handeye_method": best_method,
        "num_robot_poses": len(robot_T),
        "num_handeye_events": len(common_he),
        "num_charuco_frames": len(charuco_obs),
        "num_cube_pnp_gripper": len(pnp_obs.get(gripper_cam_idx, {})),
        "num_object_sets": int(len(T_B_O_by_set)),
        "object_set_indices": [int(x) for x in sorted(T_B_O_by_set)],
        "t_base_o_source": t_base_o_source,
        "diagnostics": {
            "fixed_extrinsics": fixed_stats,
            "handeye_methods": {
                k: {"score": v["score"], "mean_trans_mm": v["mean_trans_mm"],
                     "mean_rot_deg": v["mean_rot_deg"], "stability": v["stability"]}
                for k, v in method_results.items()
            },
            "base_transforms": base_stats,
            "cube_anchor": cube_anchor_diag,
            "cube_anchor_by_set": cube_anchor_by_set_diag,
            "gripper_event_base_transforms": gripper_base_diag,
            "set_cube_center_prior": {
                "support": int(set_cube_center_prior_diag.get("support", 0)),
                "source": None if set_cube_center_prior is None else set_cube_center_prior.get("source"),
                "stability": set_cube_center_prior_diag.get("stability", {}),
            },
            "gripper_base_pose_model": {
                "support": 0 if gripper_pose_model is None else int(gripper_pose_model.get("support", 0)),
                "model_type": None if gripper_pose_model is None else gripper_pose_model.get("model_type"),
            },
            "internal_runtime_dir": "internal_runtime",
        },
        "transforms": {},
    }

    for ci, T in T_Cref_Ci.items():
        summary["transforms"][f"T_C{ref_fixed}_C{ci}"] = T.reshape(-1).tolist()
    summary["transforms"]["T_gripper_cam"] = T_gTc.reshape(-1).tolist()
    summary["transforms"]["T_base_O"] = T_B_O_avg.reshape(-1).tolist()
    for ci, T in T_base_Ci.items():
        summary["transforms"][f"T_base_C{ci}"] = T.reshape(-1).tolist()

    final_transforms = {
        "generated_by": "Step3_calibration.py",
        "root_folder": root,
        "calib_dir": out_dir,
        "transforms": {
            name: np.asarray(T, dtype=np.float64).tolist()
            for name, T in {
                **{f"T_C{ref_fixed}_C{ci}": T for ci, T in T_Cref_Ci.items()},
                "T_gripper_cam": T_gTc,
                "T_base_O": T_B_O_avg,
                **{f"T_base_C{ci}": T for ci, T in T_base_Ci.items()},
            }.items()
        },
    }
    with open(os.path.join(out_dir, "final_transforms_base_frame.json"), "w") as f:
        json.dump(final_transforms, f, indent=2)

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
