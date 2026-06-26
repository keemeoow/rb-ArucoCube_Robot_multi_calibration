#!/usr/bin/env python3
"""
Step3_compare_calibrartion.py

Purpose
-------
A compact, experiment-first refactor of Step3 calibration for comparing:

1) Per-camera PnP mean
2) Per-camera PnP + robust SE(3) averaging
3) PnP pose-consistency optimization
4) Direct reprojection-error optimization

Each method is run in two modes:
  - without robot-known cube prior
  - with robot-known cube prior, when set_cube_center_6dof exists in meta.json

Outputs
-------
<out_dir>/ablation_summary.csv
<out_dir>/ablation_summary.json
<out_dir>/<method>__<prior_mode>/T_base_C{i}.npy or T_ref_C{i}.npy
<out_dir>/<method>__<prior_mode>/diagnostics.json

Run
---
python Step3_compare_calibrartion.py \
  --root_folder ./data/session \
  --intrinsics_dir ./intrinsics \
  --out_dir ./data/session/calib_ablation

Notes
-----
- This file intentionally avoids depth-SVD candidates, board-primary blending,
  legacy runtime export, and many fallback policies in the old Step3. Those are useful
  for recovery/debugging, but they obscure algorithmic comparison.
- Direct reprojection optimization needs accessible cube model object-corner coordinates.
  The adapter tries common project APIs/fields. If your ArucoCubeTarget model uses a
  different accessor, edit `get_marker_object_corners()` only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from aruco_cube import ArucoCubeTarget, rodrigues_to_Rt
from calibration_runtime_utils import (
    copy_depth_fields,
    filter_candidates_for_camera_role,
    get_capture_set_index,
    get_capture_set_cube_center_transform_raw,
    load_intrinsics_with_depth_scale,
    resolve_cube_config_for_run,
    select_primary_cube_candidate,
)
from config import get_default_cube_config
from cube_config_utils import cube_configs_equivalent, load_cube_config_from_meta
from robot_comm import euler_deg_to_matrix


# -----------------------------
# Basic SE(3) utilities
# -----------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def make_T(rotvec: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def T_to_vec(T: np.ndarray) -> np.ndarray:
    v = np.zeros(6, dtype=np.float64)
    v[:3] = np.asarray(T[:3, 3], dtype=np.float64)
    v[3:] = R.from_matrix(T[:3, :3]).as_rotvec()
    return v


def vec_to_T(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(6)
    return make_T(v[3:], v[:3])


def se3_log_residual(T_err: np.ndarray, rot_scale_m_per_rad: float = 0.05) -> np.ndarray:
    """Return residual in meters: [dx,dy,dz, scaled_rotvec]."""
    r = np.zeros(6, dtype=np.float64)
    r[:3] = T_err[:3, 3]
    r[3:] = R.from_matrix(T_err[:3, :3]).as_rotvec() * float(rot_scale_m_per_rad)
    return r


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def weighted_se3_average(T_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    if not T_list:
        raise ValueError("weighted_se3_average got an empty T_list")
    if weights is None:
        w = np.ones(len(T_list), dtype=np.float64)
    else:
        w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    w = w / (w.sum() + 1e-12)

    t = np.sum(np.stack([T[:3, 3] for T in T_list], axis=0) * w[:, None], axis=0)
    M = np.sum(np.stack([T[:3, :3] for T in T_list], axis=0) * w[:, None, None], axis=0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def robust_se3_average(
    T_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    max_iters: int = 5,
    k_mad: float = 2.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if not T_list:
        raise ValueError("robust_se3_average got an empty T_list")
    if weights is None:
        weights = [1.0] * len(T_list)
    inliers = np.arange(len(T_list), dtype=int)
    T_avg = weighted_se3_average(T_list, weights)

    for _ in range(max_iters):
        errs = []
        for idx in inliers:
            e = se3_log_residual(inv_T(T_avg) @ T_list[idx])
            errs.append(float(np.linalg.norm(e[:3]) * 1000.0 + np.linalg.norm(e[3:]) * 1000.0))
        errs = np.asarray(errs, dtype=np.float64)
        med = float(np.median(errs))
        mad = float(np.median(np.abs(errs - med)) + 1e-12)
        thr = med + k_mad * 1.4826 * mad
        keep_local = errs <= thr
        if keep_local.sum() < max(3, int(0.4 * len(inliers))):
            break
        new_inliers = inliers[keep_local]
        if len(new_inliers) == len(inliers):
            break
        inliers = new_inliers
        T_avg = weighted_se3_average([T_list[i] for i in inliers], [weights[i] for i in inliers])

    T_avg = weighted_se3_average([T_list[i] for i in inliers], [weights[i] for i in inliers])
    trans = [float(np.linalg.norm(T[:3, 3] - T_avg[:3, 3]) * 1000.0) for T in [T_list[i] for i in inliers]]
    rot = [rotation_error_deg(T[:3, :3], T_avg[:3, :3]) for T in [T_list[i] for i in inliers]]
    return T_avg, {
        "num_total": int(len(T_list)),
        "num_inliers": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / max(1, len(T_list))),
        "translation_std_mm": float(np.std(trans)) if trans else 0.0,
        "rotation_std_deg": float(np.std(rot)) if rot else 0.0,
    }


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class PoseObs:
    cam: int
    event: int
    set_idx: Optional[int]
    T_C_O: np.ndarray
    err_px: float
    n_points: int
    source: str


@dataclass
class CornerObs:
    cam: int
    event: int
    set_idx: Optional[int]
    object_points: np.ndarray  # Nx3, cube/object frame
    image_points: np.ndarray   # Nx2
    err_hint_px: float


@dataclass
class MethodResult:
    method: str
    prior_mode: str
    ok: bool
    message: str
    n_pose_obs: int
    n_corner_obs: int
    reproj_rmse_px: Optional[float]
    reproj_median_px: Optional[float]
    pose_trans_rmse_mm: Optional[float]
    pose_rot_rmse_deg: Optional[float]
    prior_trans_rmse_mm: Optional[float]
    prior_rot_rmse_deg: Optional[float]
    output_dir: str


# -----------------------------
# Meta and detection adapters
# -----------------------------

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
            return [float(obj["x"]), float(obj["y"]), float(obj["z"]), float(obj["rz"]), float(obj["ry"]), float(obj["rx"])]
        for key in ["robot_pose_6dof", "tcp_pose_6dof", "pose_6dof", "pose"]:
            out = try_parse_pose6(obj.get(key))
            if out is not None:
                return out
    return None


def pose6_to_T_base_gripper(pose6: List[float]) -> np.ndarray:
    x, y, z, rz, ry, rx = [float(v) for v in pose6]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_deg_to_matrix(rz, ry, rx)
    # Project convention: robot poses are usually stored in mm.
    scale = 0.001 if max(abs(x), abs(y), abs(z)) > 10.0 else 1.0
    T[:3, 3] = np.array([x, y, z], dtype=np.float64) * scale
    return T


def load_nominal_set_cube_transforms(meta: Dict[str, Any]) -> Dict[int, np.ndarray]:
    priors: Dict[int, np.ndarray] = {}
    for cap in meta.get("captures", []):
        sidx = get_capture_set_index(cap)
        if sidx is None or sidx in priors:
            continue
        raw = get_capture_set_cube_center_transform_raw(cap)
        pose = try_parse_pose6(raw)
        if pose is None:
            pose = try_parse_pose6(cap.get("set_cube_center_6dof"))
        if pose is not None:
            priors[int(sidx)] = pose6_to_T_base_gripper(pose)
    return priors


def load_robot_poses_from_meta(meta: Dict[str, Any]) -> Dict[int, np.ndarray]:
    robot_T: Dict[int, np.ndarray] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        pose = None
        for key in ["robot_pose_6dof", "tcp_pose_6dof", "pose_6dof", "robot_pose"]:
            pose = try_parse_pose6(cap.get(key))
            if pose is not None:
                break
        if pose is not None:
            robot_T[eid] = pose6_to_T_base_gripper(pose)
    return robot_T


def marker_aspect_ratio(img_pts: np.ndarray) -> float:
    pts = np.asarray(img_pts, dtype=np.float64).reshape(4, 2)
    lens = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(min(lens) / max(max(lens), 1e-12))


def stored_cube_pose_candidates(
    cinfo: Dict[str, Any],
    cam_idx: int,
    gripper_cam_idx: Optional[int],
    max_err: float,
    min_markers: int,
    min_aspect: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    aspect_by_marker: Dict[int, float] = {}

    for item in cinfo.get("markers", []):
        mid = int(item.get("marker_id", -1))
        corners = np.asarray(item.get("corners_2d", []), dtype=np.float64)
        aspect = None
        if corners.shape == (4, 2):
            aspect = marker_aspect_ratio(corners)
            aspect_by_marker[mid] = aspect

        for cand in item.get("pose_candidates") or []:
            err = float(cand.get("reproj_error_mean_px", 99.0))
            T44 = cand.get("T_cam_cube_4x4")
            if T44 is None or err > max_err:
                continue
            if aspect is not None and aspect < min_aspect:
                continue
            candidates.append({
                "T_C_O": np.asarray(T44, dtype=np.float64),
                "err_mean": err,
                "n_points": 4,
                "used_ids": [mid],
                "source": "stored_ippe",
                **copy_depth_fields(cand),
            })

    cpnp = cinfo.get("cube_pnp")
    if cpnp and cpnp.get("ok"):
        err = float(cpnp.get("reproj_mean_px", 99.0))
        used_ids = [int(x) for x in cpnp.get("used_ids", [])]
        T44 = cpnp.get("T_cam_cube_4x4")
        if T44 is not None and err <= max_err and len(set(used_ids)) >= max(1, int(min_markers)):
            aspects = [aspect_by_marker[mid] for mid in used_ids if mid in aspect_by_marker]
            if not aspects or min(aspects) >= min_aspect:
                candidates.append({
                    "T_C_O": np.asarray(T44, dtype=np.float64),
                    "err_mean": err,
                    "n_points": int(cpnp.get("n_points", 4 * max(1, len(set(used_ids))))),
                    "used_ids": used_ids,
                    "source": "stored_cube_pnp",
                    **copy_depth_fields(cpnp),
                })
    return candidates


def get_marker_object_corners(cube: ArucoCubeTarget, marker_id: int) -> Optional[np.ndarray]:
    """Adapter for project-specific cube model APIs.

    Expected output order must match cube.model.reorder_image_corners(marker_id, corners).
    Add one branch here if your model exposes a different method/field name.
    """
    model = cube.model
    mid = int(marker_id)

    method_names = [
        "get_marker_object_corners",
        "marker_object_corners",
        "get_marker_corners_3d",
        "marker_corners_3d",
        "object_corners",
        "corners_3d",
    ]
    for name in method_names:
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                pts = np.asarray(fn(mid), dtype=np.float64)
                if pts.shape == (4, 3):
                    return pts
            except TypeError:
                pass
            except Exception:
                pass

    field_names = [
        "marker_corners_obj",
        "marker_corners_3d",
        "object_points_by_id",
        "corners_by_marker",
        "markers",
    ]
    for name in field_names:
        data = getattr(model, name, None)
        if isinstance(data, dict) and mid in data:
            val = data[mid]
            if isinstance(val, dict):
                for key in ["corners_3d", "object_points", "obj_pts", "points"]:
                    if key in val:
                        pts = np.asarray(val[key], dtype=np.float64)
                        if pts.shape == (4, 3):
                            return pts
            else:
                pts = np.asarray(val, dtype=np.float64)
                if pts.shape == (4, 3):
                    return pts
    return None


def detect_corner_observations(
    root: str,
    meta: Dict[str, Any],
    cube: ArucoCubeTarget,
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    all_cam_ids: List[int],
    gripper_cam_idx: int,
    max_err_fixed: float,
    max_err_gripper: float,
    min_aspect_fixed: float,
    min_aspect_gripper: float,
) -> List[CornerObs]:
    obs: List[CornerObs] = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        sidx = get_capture_set_index(cap)
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in all_cam_ids or not cinfo.get("saved"):
                continue
            rgb_rel = cinfo.get("rgb_path", "")
            if not rgb_rel:
                continue
            img = cv2.imread(os.path.join(root, rgb_rel))
            if img is None:
                continue
            try:
                corners_list, ids = cube.detect(img)
            except Exception:
                continue
            if ids is None:
                continue
            obj_all, img_all = [], []
            min_aspect = min_aspect_gripper if ci == gripper_cam_idx else min_aspect_fixed
            for corners, mid_raw in zip(corners_list, ids):
                mid = int(np.asarray(mid_raw).reshape(-1)[0])
                if not cube.model.has_marker(mid):
                    continue
                img_pts_raw = np.asarray(corners, dtype=np.float64).reshape(4, 2)
                try:
                    img_pts = np.asarray(cube.model.reorder_image_corners(mid, img_pts_raw), dtype=np.float64).reshape(4, 2)
                except Exception:
                    img_pts = img_pts_raw
                if marker_aspect_ratio(img_pts) < min_aspect:
                    continue
                obj_pts = get_marker_object_corners(cube, mid)
                if obj_pts is None:
                    continue
                obj_all.append(obj_pts)
                img_all.append(img_pts)
            if obj_all:
                obs.append(CornerObs(
                    cam=ci,
                    event=eid,
                    set_idx=int(sidx) if sidx is not None else None,
                    object_points=np.concatenate(obj_all, axis=0),
                    image_points=np.concatenate(img_all, axis=0),
                    err_hint_px=max_err_gripper if ci == gripper_cam_idx else max_err_fixed,
                ))
    return obs


def estimate_image_cube_pose(
    cube: ArucoCubeTarget,
    img: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    max_err: float,
    min_markers: int,
    min_aspect: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    try:
        ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
            img, K, D,
            use_ransac=True,
            min_markers=max(1, int(min_markers)),
            reproj_thr_mean_px=float(max_err),
            return_reproj=True,
            min_aspect=float(min_aspect),
        )
        if ok and reproj and float(reproj.get("err_mean", 99.0)) <= max_err:
            candidates.append({
                "T_C_O": rodrigues_to_Rt(rvec, tvec),
                "err_mean": float(reproj["err_mean"]),
                "n_points": int(reproj.get("n_points", 4)),
                "used_ids": [int(x) for x in used],
                "source": "redetect_cube_pnp",
            })
    except Exception:
        pass
    return candidates


def load_pose_observations(
    root: str,
    meta: Dict[str, Any],
    cube: ArucoCubeTarget,
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    all_cam_ids: List[int],
    gripper_cam_idx: int,
    reuse_stored_cube_candidates: bool,
    max_err_fixed: float,
    max_err_gripper: float,
    min_aspect_fixed: float,
    min_aspect_gripper: float,
    gripper_min_markers: int,
) -> List[PoseObs]:
    obs: List[PoseObs] = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        sidx = get_capture_set_index(cap)
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in all_cam_ids or not cinfo.get("saved"):
                continue
            max_err = max_err_gripper if ci == gripper_cam_idx else max_err_fixed
            min_aspect = min_aspect_gripper if ci == gripper_cam_idx else min_aspect_fixed
            min_markers = gripper_min_markers if ci == gripper_cam_idx else 1

            candidates = []
            if reuse_stored_cube_candidates:
                candidates.extend(stored_cube_pose_candidates(
                    cinfo, ci, gripper_cam_idx, max_err, min_markers, min_aspect
                ))
            rgb_rel = cinfo.get("rgb_path", "")
            if rgb_rel:
                img = cv2.imread(os.path.join(root, rgb_rel))
                if img is not None:
                    candidates = estimate_image_cube_pose(
                        cube, img, K_map[ci], D_map[ci], max_err, min_markers, min_aspect
                    ) + candidates
            candidates = filter_candidates_for_camera_role(candidates, ci, gripper_cam_idx)
            best = select_primary_cube_candidate(candidates) if candidates else None
            if best is None:
                continue
            obs.append(PoseObs(
                cam=ci,
                event=eid,
                set_idx=int(sidx) if sidx is not None else None,
                T_C_O=np.asarray(best["T_C_O"], dtype=np.float64),
                err_px=float(best.get("err_mean", 99.0)),
                n_points=int(best.get("n_points", 4)),
                source=str(best.get("source", "unknown")),
            ))
    return obs


# -----------------------------
# Calibration initialization
# -----------------------------

def observations_by_cam_event(pose_obs: List[PoseObs]) -> Dict[int, Dict[int, PoseObs]]:
    out: Dict[int, Dict[int, PoseObs]] = defaultdict(dict)
    for o in pose_obs:
        out[o.cam][o.event] = o
    return out


def build_ref_relative_from_pairwise(
    pose_obs: List[PoseObs],
    fixed_cam_ids: List[int],
    ref_cam: int,
    robust: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    by = observations_by_cam_event(pose_obs)
    T_ref_C: Dict[int, np.ndarray] = {ref_cam: np.eye(4, dtype=np.float64)}
    diag: Dict[str, Any] = {}
    for ci in fixed_cam_ids:
        if ci == ref_cam:
            continue
        common = sorted(set(by.get(ref_cam, {}).keys()) & set(by.get(ci, {}).keys()))
        Ts, ws = [], []
        for eid in common:
            T_ref_O = by[ref_cam][eid].T_C_O
            T_ci_O = by[ci][eid].T_C_O
            Ts.append(T_ref_O @ inv_T(T_ci_O))
            ws.append(1.0 / max(by[ref_cam][eid].err_px * by[ci][eid].err_px, 1e-9))
        if not Ts:
            continue
        if robust:
            T, st = robust_se3_average(Ts, ws)
        else:
            T = weighted_se3_average(Ts, None)
            st = {"num_total": len(Ts), "num_inliers": len(Ts), "inlier_ratio": 1.0}
        T_ref_C[ci] = T
        diag[f"T_ref_C{ci}"] = st
    return T_ref_C, diag


def initialize_ref_object_poses(
    pose_obs: List[PoseObs],
    T_ref_C: Dict[int, np.ndarray],
    fixed_cam_ids: List[int],
    ref_cam: int,
) -> Dict[int, np.ndarray]:
    by_event: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    for o in pose_obs:
        if o.cam not in fixed_cam_ids or o.cam not in T_ref_C:
            continue
        # T_ref_O = T_ref_Ci * T_Ci_O
        by_event[o.event].append((T_ref_C[o.cam] @ o.T_C_O, 1.0 / max(o.err_px, 1e-9)))
    out: Dict[int, np.ndarray] = {}
    for eid, pairs in by_event.items():
        out[eid] = weighted_se3_average([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def initialize_base_from_priors(
    pose_obs: List[PoseObs],
    fixed_cam_ids: List[int],
    set_priors: Dict[int, np.ndarray],
    robust: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any]]:
    """Initialize T_base_Ci and event object poses from nominal T_base_O_set priors."""
    T_base_C: Dict[int, np.ndarray] = {}
    diag: Dict[str, Any] = {}
    for ci in fixed_cam_ids:
        Ts, ws = [], []
        for o in pose_obs:
            if o.cam != ci or o.set_idx is None or o.set_idx not in set_priors:
                continue
            # T_base_Ci = T_base_O_prior * inv(T_Ci_O)
            Ts.append(set_priors[o.set_idx] @ inv_T(o.T_C_O))
            ws.append(1.0 / max(o.err_px, 1e-9))
        if not Ts:
            continue
        if robust:
            T, st = robust_se3_average(Ts, ws)
        else:
            T = weighted_se3_average(Ts, None)
            st = {"num_total": len(Ts), "num_inliers": len(Ts), "inlier_ratio": 1.0}
        T_base_C[ci] = T
        diag[f"T_base_C{ci}"] = st

    T_base_O_event: Dict[int, np.ndarray] = {}
    by_event: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    for o in pose_obs:
        if o.cam in T_base_C:
            by_event[o.event].append((T_base_C[o.cam] @ o.T_C_O, 1.0 / max(o.err_px, 1e-9)))
    for eid, pairs in by_event.items():
        T_base_O_event[eid] = weighted_se3_average([p[0] for p in pairs], [p[1] for p in pairs])
    return T_base_C, T_base_O_event, diag


def build_param_layout(cam_ids: List[int], event_ids: List[int], ref_cam: Optional[int]) -> Dict[str, Any]:
    cam_vars = [ci for ci in cam_ids if ref_cam is None or ci != ref_cam]
    layout = {
        "cam_vars": cam_vars,
        "event_vars": event_ids,
        "cam_slice": {},
        "event_slice": {},
        "n": 0,
    }
    k = 0
    for ci in cam_vars:
        layout["cam_slice"][ci] = slice(k, k + 6)
        k += 6
    for eid in event_ids:
        layout["event_slice"][eid] = slice(k, k + 6)
        k += 6
    layout["n"] = k
    return layout


def pack_params(T_cam: Dict[int, np.ndarray], T_obj: Dict[int, np.ndarray], layout: Dict[str, Any]) -> np.ndarray:
    x = np.zeros(layout["n"], dtype=np.float64)
    for ci, sl in layout["cam_slice"].items():
        x[sl] = T_to_vec(T_cam[ci])
    for eid, sl in layout["event_slice"].items():
        x[sl] = T_to_vec(T_obj[eid])
    return x


def unpack_params(x: np.ndarray, layout: Dict[str, Any], ref_cam: Optional[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    T_cam: Dict[int, np.ndarray] = {}
    if ref_cam is not None:
        T_cam[ref_cam] = np.eye(4, dtype=np.float64)
    for ci, sl in layout["cam_slice"].items():
        T_cam[ci] = vec_to_T(x[sl])
    T_obj = {eid: vec_to_T(x[sl]) for eid, sl in layout["event_slice"].items()}
    return T_cam, T_obj


# -----------------------------
# Metrics
# -----------------------------

def pose_consistency_metrics(
    pose_obs: List[PoseObs],
    T_cam: Dict[int, np.ndarray],
    T_obj_event: Dict[int, np.ndarray],
    fixed_cam_ids: List[int],
) -> Tuple[Optional[float], Optional[float]]:
    trans_mm, rot_deg = [], []
    for o in pose_obs:
        if o.cam not in fixed_cam_ids or o.cam not in T_cam or o.event not in T_obj_event:
            continue
        pred = inv_T(T_cam[o.cam]) @ T_obj_event[o.event]
        Terr = inv_T(o.T_C_O) @ pred
        trans_mm.append(float(np.linalg.norm(Terr[:3, 3]) * 1000.0))
        rot_deg.append(float(np.degrees(np.linalg.norm(R.from_matrix(Terr[:3, :3]).as_rotvec()))))
    if not trans_mm:
        return None, None
    return float(np.sqrt(np.mean(np.square(trans_mm)))), float(np.sqrt(np.mean(np.square(rot_deg))))


def prior_metrics(
    T_obj_event: Dict[int, np.ndarray],
    event_to_set: Dict[int, Optional[int]],
    set_priors: Dict[int, np.ndarray],
) -> Tuple[Optional[float], Optional[float]]:
    trans_mm, rot_deg = [], []
    for eid, T in T_obj_event.items():
        sidx = event_to_set.get(eid)
        if sidx is None or sidx not in set_priors:
            continue
        Terr = inv_T(set_priors[sidx]) @ T
        trans_mm.append(float(np.linalg.norm(Terr[:3, 3]) * 1000.0))
        rot_deg.append(float(np.degrees(np.linalg.norm(R.from_matrix(Terr[:3, :3]).as_rotvec()))))
    if not trans_mm:
        return None, None
    return float(np.sqrt(np.mean(np.square(trans_mm)))), float(np.sqrt(np.mean(np.square(rot_deg))))


def reprojection_errors(
    corner_obs: List[CornerObs],
    T_cam: Dict[int, np.ndarray],
    T_obj_event: Dict[int, np.ndarray],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    fixed_cam_ids: List[int],
) -> np.ndarray:
    errs: List[float] = []
    for o in corner_obs:
        if o.cam not in fixed_cam_ids or o.cam not in T_cam or o.event not in T_obj_event:
            continue
        T_C_O = inv_T(T_cam[o.cam]) @ T_obj_event[o.event]
        rvec = R.from_matrix(T_C_O[:3, :3]).as_rotvec().reshape(3, 1)
        tvec = T_C_O[:3, 3].reshape(3, 1)
        proj, _ = cv2.projectPoints(o.object_points.astype(np.float64), rvec, tvec, K_map[o.cam], D_map[o.cam])
        diff = proj.reshape(-1, 2) - o.image_points.reshape(-1, 2)
        errs.extend(np.linalg.norm(diff, axis=1).tolist())
    return np.asarray(errs, dtype=np.float64)


# -----------------------------
# Optimization methods
# -----------------------------

def optimize_pose_consistency(
    pose_obs: List[PoseObs],
    fixed_cam_ids: List[int],
    init_T_cam: Dict[int, np.ndarray],
    init_T_obj: Dict[int, np.ndarray],
    ref_cam: Optional[int],
    event_to_set: Dict[int, Optional[int]],
    set_priors: Optional[Dict[int, np.ndarray]],
    prior_weight: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any]]:
    event_ids = sorted(init_T_obj.keys())
    cam_ids = sorted([ci for ci in fixed_cam_ids if ci in init_T_cam])
    layout = build_param_layout(cam_ids, event_ids, ref_cam=ref_cam)
    x0 = pack_params(init_T_cam, init_T_obj, layout)

    usable = [o for o in pose_obs if o.cam in cam_ids and o.event in init_T_obj]
    if len(usable) < 4:
        return init_T_cam, init_T_obj, {"optimized": False, "reason": "not enough pose observations"}

    def residual(x: np.ndarray) -> np.ndarray:
        T_cam, T_obj = unpack_params(x, layout, ref_cam=ref_cam)
        res = []
        for o in usable:
            pred = inv_T(T_cam[o.cam]) @ T_obj[o.event]
            e = se3_log_residual(inv_T(o.T_C_O) @ pred)
            w = math.sqrt(min(50.0, 1.0 / max(o.err_px, 1e-6)))
            res.extend((e * w).tolist())
        if set_priors and prior_weight > 0.0:
            for eid, T in T_obj.items():
                sidx = event_to_set.get(eid)
                if sidx is None or sidx not in set_priors:
                    continue
                e = se3_log_residual(inv_T(set_priors[sidx]) @ T)
                res.extend((e * float(prior_weight)).tolist())
        return np.asarray(res, dtype=np.float64)

    r0 = residual(x0)
    opt = least_squares(
        residual,
        x0,
        method="trf",
        loss="huber",
        f_scale=0.003,
        max_nfev=300,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    T_cam, T_obj = unpack_params(opt.x, layout, ref_cam=ref_cam)
    r1 = residual(opt.x)
    return T_cam, T_obj, {
        "optimized": True,
        "success": bool(opt.success),
        "cost_initial": float(np.mean(r0 ** 2)) if r0.size else None,
        "cost_final": float(np.mean(r1 ** 2)) if r1.size else None,
        "nfev": int(opt.nfev),
    }


def optimize_reprojection(
    corner_obs: List[CornerObs],
    pose_obs: List[PoseObs],
    fixed_cam_ids: List[int],
    init_T_cam: Dict[int, np.ndarray],
    init_T_obj: Dict[int, np.ndarray],
    ref_cam: Optional[int],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    event_to_set: Dict[int, Optional[int]],
    set_priors: Optional[Dict[int, np.ndarray]],
    prior_weight: float,
    pose_regularizer_weight: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any]]:
    event_ids = sorted(init_T_obj.keys())
    cam_ids = sorted([ci for ci in fixed_cam_ids if ci in init_T_cam])
    layout = build_param_layout(cam_ids, event_ids, ref_cam=ref_cam)
    x0 = pack_params(init_T_cam, init_T_obj, layout)

    usable_corners = [o for o in corner_obs if o.cam in cam_ids and o.event in init_T_obj]
    usable_poses = [o for o in pose_obs if o.cam in cam_ids and o.event in init_T_obj]
    if len(usable_corners) < 4:
        return init_T_cam, init_T_obj, {"optimized": False, "reason": "not enough corner observations or cube object-corner API unavailable"}

    def residual(x: np.ndarray) -> np.ndarray:
        T_cam, T_obj = unpack_params(x, layout, ref_cam=ref_cam)
        res: List[float] = []
        for o in usable_corners:
            T_C_O = inv_T(T_cam[o.cam]) @ T_obj[o.event]
            rvec = R.from_matrix(T_C_O[:3, :3]).as_rotvec().reshape(3, 1)
            tvec = T_C_O[:3, 3].reshape(3, 1)
            proj, _ = cv2.projectPoints(o.object_points.astype(np.float64), rvec, tvec, K_map[o.cam], D_map[o.cam])
            diff = (proj.reshape(-1, 2) - o.image_points.reshape(-1, 2)).reshape(-1)
            # Pixel residual. Robust loss handles bad corners.
            res.extend(diff.tolist())
        if pose_regularizer_weight > 0.0:
            for o in usable_poses:
                pred = inv_T(T_cam[o.cam]) @ T_obj[o.event]
                e = se3_log_residual(inv_T(o.T_C_O) @ pred)
                res.extend((e * float(pose_regularizer_weight)).tolist())
        if set_priors and prior_weight > 0.0:
            for eid, T in T_obj.items():
                sidx = event_to_set.get(eid)
                if sidx is None or sidx not in set_priors:
                    continue
                # Convert meter-level prior to pixel-like scale. 100 px per meter is intentionally soft.
                e = se3_log_residual(inv_T(set_priors[sidx]) @ T)
                res.extend((e * float(prior_weight)).tolist())
        return np.asarray(res, dtype=np.float64)

    r0 = residual(x0)
    opt = least_squares(
        residual,
        x0,
        method="trf",
        loss="huber",
        f_scale=2.0,
        max_nfev=500,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    T_cam, T_obj = unpack_params(opt.x, layout, ref_cam=ref_cam)
    r1 = residual(opt.x)
    return T_cam, T_obj, {
        "optimized": True,
        "success": bool(opt.success),
        "cost_initial": float(np.mean(r0 ** 2)) if r0.size else None,
        "cost_final": float(np.mean(r1 ** 2)) if r1.size else None,
        "nfev": int(opt.nfev),
    }


# -----------------------------
# Evaluation runner
# -----------------------------

def save_transforms(out_dir: str, T_cam: Dict[int, np.ndarray], prior_mode: str, ref_cam: Optional[int]) -> None:
    ensure_dir(out_dir)
    for ci, T in sorted(T_cam.items()):
        if prior_mode == "with_robot_cube_prior":
            np.save(os.path.join(out_dir, f"T_base_C{ci}.npy"), T)
        else:
            np.save(os.path.join(out_dir, f"T_ref_C{ci}.npy"), T)
    if ref_cam is not None and prior_mode == "without_robot_cube_prior":
        with open(os.path.join(out_dir, "coordinate_note.txt"), "w") as f:
            f.write(f"Transforms are in cam{ref_cam} reference coordinates. T_ref_C{ref_cam}=I.\n")


def evaluate_and_save(
    method: str,
    prior_mode: str,
    base_out: str,
    pose_obs: List[PoseObs],
    corner_obs: List[CornerObs],
    T_cam: Dict[int, np.ndarray],
    T_obj: Dict[int, np.ndarray],
    fixed_cam_ids: List[int],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    event_to_set: Dict[int, Optional[int]],
    set_priors: Dict[int, np.ndarray],
    diag: Dict[str, Any],
    ref_cam: Optional[int],
) -> MethodResult:
    out_dir = ensure_dir(os.path.join(base_out, f"{method}__{prior_mode}"))
    save_transforms(out_dir, T_cam, prior_mode, ref_cam)

    e = reprojection_errors(corner_obs, T_cam, T_obj, K_map, D_map, fixed_cam_ids)
    reproj_rmse = float(np.sqrt(np.mean(e ** 2))) if e.size else None
    reproj_med = float(np.median(e)) if e.size else None
    pose_t, pose_r = pose_consistency_metrics(pose_obs, T_cam, T_obj, fixed_cam_ids)
    prior_t, prior_r = prior_metrics(T_obj, event_to_set, set_priors)

    diagnostics = {
        "method": method,
        "prior_mode": prior_mode,
        "n_pose_obs": len(pose_obs),
        "n_corner_obs": len(corner_obs),
        "reproj_rmse_px": reproj_rmse,
        "reproj_median_px": reproj_med,
        "pose_trans_rmse_mm": pose_t,
        "pose_rot_rmse_deg": pose_r,
        "prior_trans_rmse_mm": prior_t,
        "prior_rot_rmse_deg": prior_r,
        "extra": diag,
    }
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    return MethodResult(
        method=method,
        prior_mode=prior_mode,
        ok=True,
        message="ok",
        n_pose_obs=len(pose_obs),
        n_corner_obs=len(corner_obs),
        reproj_rmse_px=reproj_rmse,
        reproj_median_px=reproj_med,
        pose_trans_rmse_mm=pose_t,
        pose_rot_rmse_deg=pose_r,
        prior_trans_rmse_mm=prior_t,
        prior_rot_rmse_deg=prior_r,
        output_dir=out_dir,
    )


def run_method_suite(
    pose_obs: List[PoseObs],
    corner_obs: List[CornerObs],
    fixed_cam_ids: List[int],
    ref_cam: int,
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    event_to_set: Dict[int, Optional[int]],
    set_priors: Dict[int, np.ndarray],
    out_dir: str,
    with_prior: bool,
    args: argparse.Namespace,
) -> List[MethodResult]:
    prior_mode = "with_robot_cube_prior" if with_prior else "without_robot_cube_prior"
    results: List[MethodResult] = []

    # Choose coordinate system.
    # - No prior: cam-ref coordinate with T_ref_Cref=I.
    # - With prior: robot base coordinate using set_cube_center_6dof priors.
    ref_for_layout = None if with_prior else ref_cam

    if with_prior:
        if not set_priors:
            return [MethodResult("all", prior_mode, False, "no set_cube_center_6dof prior in meta.json", len(pose_obs), len(corner_obs), None, None, None, None, None, None, out_dir)]
        T_cam_mean, T_obj_mean, diag_mean = initialize_base_from_priors(pose_obs, fixed_cam_ids, set_priors, robust=False)
        T_cam_rob, T_obj_rob, diag_rob = initialize_base_from_priors(pose_obs, fixed_cam_ids, set_priors, robust=True)
    else:
        T_cam_mean, diag_mean = build_ref_relative_from_pairwise(pose_obs, fixed_cam_ids, ref_cam, robust=False)
        T_obj_mean = initialize_ref_object_poses(pose_obs, T_cam_mean, fixed_cam_ids, ref_cam)
        T_cam_rob, diag_rob = build_ref_relative_from_pairwise(pose_obs, fixed_cam_ids, ref_cam, robust=True)
        T_obj_rob = initialize_ref_object_poses(pose_obs, T_cam_rob, fixed_cam_ids, ref_cam)

    # 1) simple mean
    results.append(evaluate_and_save(
        "01_pnp_mean", prior_mode, out_dir, pose_obs, corner_obs, T_cam_mean, T_obj_mean,
        fixed_cam_ids, K_map, D_map, event_to_set, set_priors, diag_mean, ref_for_layout
    ))

    # 2) robust average
    results.append(evaluate_and_save(
        "02_pnp_robust_se3", prior_mode, out_dir, pose_obs, corner_obs, T_cam_rob, T_obj_rob,
        fixed_cam_ids, K_map, D_map, event_to_set, set_priors, diag_rob, ref_for_layout
    ))

    # 3) pose-level consistency optimization, initialized by robust.
    T_cam_pose, T_obj_pose, diag_pose = optimize_pose_consistency(
        pose_obs=pose_obs,
        fixed_cam_ids=fixed_cam_ids,
        init_T_cam=T_cam_rob,
        init_T_obj=T_obj_rob,
        ref_cam=ref_for_layout,
        event_to_set=event_to_set,
        set_priors=set_priors if with_prior else None,
        prior_weight=float(args.pose_prior_weight if with_prior else 0.0),
    )
    results.append(evaluate_and_save(
        "03_pose_consistency_opt", prior_mode, out_dir, pose_obs, corner_obs, T_cam_pose, T_obj_pose,
        fixed_cam_ids, K_map, D_map, event_to_set, set_priors, diag_pose, ref_for_layout
    ))

    # 4) direct reprojection optimization, initialized by pose opt.
    T_cam_repr, T_obj_repr, diag_repr = optimize_reprojection(
        corner_obs=corner_obs,
        pose_obs=pose_obs,
        fixed_cam_ids=fixed_cam_ids,
        init_T_cam=T_cam_pose,
        init_T_obj=T_obj_pose,
        ref_cam=ref_for_layout,
        K_map=K_map,
        D_map=D_map,
        event_to_set=event_to_set,
        set_priors=set_priors if with_prior else None,
        prior_weight=float(args.reproj_prior_weight if with_prior else 0.0),
        pose_regularizer_weight=float(args.reproj_pose_regularizer_weight),
    )
    results.append(evaluate_and_save(
        "04_direct_reprojection_opt", prior_mode, out_dir, pose_obs, corner_obs, T_cam_repr, T_obj_repr,
        fixed_cam_ids, K_map, D_map, event_to_set, set_priors, diag_repr, ref_for_layout
    ))

    return results


def write_summary(out_dir: str, results: List[MethodResult]) -> None:
    ensure_dir(out_dir)
    rows = [asdict(r) for r in results]
    with open(os.path.join(out_dir, "ablation_summary.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    fields = list(rows[0].keys()) if rows else []
    with open(os.path.join(out_dir, "ablation_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(results: List[MethodResult]) -> None:
    print("\n" + "=" * 96)
    print("ABLATION SUMMARY")
    print("=" * 96)
    header = f"{'method':30s} {'prior':25s} {'reprj_rmse':>11s} {'pose_t':>10s} {'pose_r':>9s} {'prior_t':>10s}"
    print(header)
    print("-" * len(header))
    for r in results:
        def fmt(v: Optional[float], nd: int = 3) -> str:
            return "NA" if v is None else f"{v:.{nd}f}"
        print(f"{r.method:30s} {r.prior_mode:25s} {fmt(r.reproj_rmse_px):>11s} {fmt(r.pose_trans_rmse_mm,2):>10s} {fmt(r.pose_rot_rmse_deg,3):>9s} {fmt(r.prior_trans_rmse_mm,2):>10s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refactored calibration ablation runner")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--gripper_cam_idx", type=int, default=None)
    parser.add_argument("--ref_fixed_cam_idx", type=int, default=None)
    parser.add_argument("--cube_config_json", type=str, default=None)
    parser.add_argument("--max_err_fixed", type=float, default=3.0)
    parser.add_argument("--max_err_gripper", type=float, default=5.0)
    parser.add_argument("--fixed_cube_min_aspect", type=float, default=0.0)
    parser.add_argument("--gripper_cube_min_aspect", type=float, default=0.35)
    parser.add_argument("--gripper_cube_min_markers", type=int, default=1)
    parser.add_argument("--pose_prior_weight", type=float, default=30.0,
                        help="Soft prior weight for pose-consistency optimization when robot cube prior is enabled.")
    parser.add_argument("--reproj_prior_weight", type=float, default=100.0,
                        help="Soft prior weight for direct reprojection optimization when robot cube prior is enabled.")
    parser.add_argument("--reproj_pose_regularizer_weight", type=float, default=2.0,
                        help="Soft pose regularizer used during direct reprojection optimization.")
    args = parser.parse_args()

    root = args.root_folder
    out_dir = ensure_dir(args.out_dir or os.path.join(root, "calib_ablation"))
    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)

    cfg, cfg_source = resolve_cube_config_for_run(
        root_folder=root,
        calib_dir=out_dir,
        cube_config_json=args.cube_config_json,
        default_cfg=get_default_cube_config(),
    )
    meta_cfg, _ = load_cube_config_from_meta(root, default_cfg=cfg)
    reuse_stored = cube_configs_equivalent(meta_cfg, cfg)
    cube = ArucoCubeTarget(cfg)

    all_cam_ids = sorted({
        int(k) for cap in meta.get("captures", [])
        for k, v in cap.get("cams", {}).items() if v.get("saved")
    })
    if not all_cam_ids:
        raise RuntimeError("No saved cameras found in meta.json")

    gripper_cam_idx = args.gripper_cam_idx
    if gripper_cam_idx is None:
        gripper_cam_idx = meta.get("gripper_cam_idx")
    if gripper_cam_idx is None:
        dm = os.path.join(args.intrinsics_dir, "device_map.json")
        if os.path.exists(dm):
            with open(dm, "r") as f:
                gripper_cam_idx = json.load(f).get("gripper_cam_idx")
    if gripper_cam_idx is None:
        raise RuntimeError("gripper_cam_idx is required or must exist in meta/device_map.json")

    fixed_cam_ids = [ci for ci in all_cam_ids if ci != int(gripper_cam_idx)]
    if len(fixed_cam_ids) < 2:
        raise RuntimeError("Need at least two fixed cameras for this comparison")
    ref_cam = args.ref_fixed_cam_idx if args.ref_fixed_cam_idx is not None else fixed_cam_ids[0]
    if ref_cam not in fixed_cam_ids:
        raise RuntimeError(f"ref_fixed_cam_idx cam{ref_cam} is not in fixed cams: {fixed_cam_ids}")

    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], _ = load_intrinsics_with_depth_scale(args.intrinsics_dir, ci)

    event_to_set: Dict[int, Optional[int]] = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid >= 0:
            sidx = get_capture_set_index(cap)
            event_to_set[eid] = int(sidx) if sidx is not None else None

    set_priors = load_nominal_set_cube_transforms(meta)

    print(f"[INFO] cube config source: {cfg_source}")
    print(f"[INFO] all cams={all_cam_ids}, fixed={fixed_cam_ids}, gripper=cam{gripper_cam_idx}, ref=cam{ref_cam}")
    print(f"[INFO] stored cube candidates reused: {reuse_stored}")
    print(f"[INFO] robot cube priors: {len(set_priors)} sets")

    pose_obs = load_pose_observations(
        root=root,
        meta=meta,
        cube=cube,
        K_map=K_map,
        D_map=D_map,
        all_cam_ids=all_cam_ids,
        gripper_cam_idx=int(gripper_cam_idx),
        reuse_stored_cube_candidates=reuse_stored,
        max_err_fixed=float(args.max_err_fixed),
        max_err_gripper=float(args.max_err_gripper),
        min_aspect_fixed=float(args.fixed_cube_min_aspect),
        min_aspect_gripper=float(args.gripper_cube_min_aspect),
        gripper_min_markers=int(args.gripper_cube_min_markers),
    )
    # This comparison is about fixed multi-camera calibration. Gripper observations are kept out of metrics.
    fixed_pose_obs = [o for o in pose_obs if o.cam in fixed_cam_ids]

    corner_obs = detect_corner_observations(
        root=root,
        meta=meta,
        cube=cube,
        K_map=K_map,
        D_map=D_map,
        all_cam_ids=fixed_cam_ids,
        gripper_cam_idx=int(gripper_cam_idx),
        max_err_fixed=float(args.max_err_fixed),
        max_err_gripper=float(args.max_err_gripper),
        min_aspect_fixed=float(args.fixed_cube_min_aspect),
        min_aspect_gripper=float(args.gripper_cube_min_aspect),
    )

    print(f"[INFO] fixed pose observations: {len(fixed_pose_obs)}")
    print(f"[INFO] fixed corner observations: {len(corner_obs)}")
    if len(corner_obs) == 0:
        print("[WARN] Direct reprojection optimization will be skipped unless get_marker_object_corners() is adapted to your cube model API.")

    results: List[MethodResult] = []
    results.extend(run_method_suite(
        fixed_pose_obs, corner_obs, fixed_cam_ids, ref_cam, K_map, D_map,
        event_to_set, set_priors, out_dir, with_prior=False, args=args
    ))
    results.extend(run_method_suite(
        fixed_pose_obs, corner_obs, fixed_cam_ids, ref_cam, K_map, D_map,
        event_to_set, set_priors, out_dir, with_prior=True, args=args
    ))

    write_summary(out_dir, results)
    print_summary(results)
    print(f"\n[DONE] summary: {os.path.join(out_dir, 'ablation_summary.csv')}")


if __name__ == "__main__":
    main()
