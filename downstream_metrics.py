import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, inv_T
from calibration_runtime_utils import (
    build_event_cube_selection,
    get_event_base_camera_transform,
    load_intrinsics_with_depth_scale,
)
from config import CharucoBoardConfig, CubeConfig
from charuco_utils import CharucoTarget
from Step3_calibration import (
    rotation_error_deg,
    weighted_se3_average,
)


def compute_board_reprojection_metrics(meta: dict,
                                       root_folder: str,
                                       intrinsics_dir: str,
                                       all_cam_ids: List[int]) -> dict:
    detector = CharucoTarget(CharucoBoardConfig())
    errors_all = []
    per_camera = {}
    for ci in all_cam_ids:
        K, D, _ = load_intrinsics_with_depth_scale(intrinsics_dir, ci)
        cam_errors = []
        for cap in meta.get("captures", []):
            cinfo = cap.get("cams", {}).get(str(ci), {})
            rgb_rel = cinfo.get("rgb_path", "")
            if not cinfo.get("saved") or not rgb_rel:
                continue
            img = cv2.imread(os.path.join(root_folder, rgb_rel))
            if img is None:
                continue
            ok, _, _, n_corners, reproj = detector.estimate_pose(img, K, D)
            if ok and n_corners >= 4 and reproj is not None:
                cam_errors.append(float(reproj))
                errors_all.append(float(reproj))
        per_camera[f"cam{ci}"] = {
            "frames": int(len(cam_errors)),
            "mean_px": None if not cam_errors else float(np.mean(cam_errors)),
        }
    return {
        "total_frames": int(len(errors_all)),
        "mean_px": None if not errors_all else float(np.mean(errors_all)),
        "median_px": None if not errors_all else float(np.median(errors_all)),
        "max_px": None if not errors_all else float(np.max(errors_all)),
        "pass": None if not errors_all else bool(np.mean(errors_all) < 0.5),
        "per_camera": per_camera,
    }


def compute_pose_repeatability_metrics(meta: dict,
                                       transforms: Dict[str, np.ndarray],
                                       intrinsics_dir: str,
                                       root_folder: str,
                                       all_cam_ids: List[int],
                                       gripper_cam_idx: Optional[int],
                                       cube_cfg: CubeConfig,
                                       include_meta: bool = False,
                                       selection_profile: str = "default") -> dict:
    selection = build_event_cube_selection(
        meta, transforms, intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta, selection_profile=selection_profile)
    event_dt = []
    event_dr = []
    num_events = 0
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        refined = selection.get(eid, {})
        if len(refined) < 2:
            continue
        Ts, ws = [], []
        for ci, cand in refined.items():
            T_base_cam = get_event_base_camera_transform(cap, ci, transforms, gripper_cam_idx)
            if T_base_cam is None:
                continue
            Ts.append(T_base_cam @ np.asarray(cand["T_C_O"], dtype=np.float64))
            ws.append(1.0 / max(float(cand.get("err_mean", 1.0)), 1e-9))
        if len(Ts) < 2:
            continue
        num_events += 1
        T_event = weighted_se3_average(Ts, ws)
        for T in Ts:
            event_dt.append(float(np.linalg.norm(T[:3, 3] - T_event[:3, 3]) * 1000.0))
            event_dr.append(rotation_error_deg(T[:3, :3], T_event[:3, :3]))

    if not event_dt:
        return {
            "num_events": 0,
            "mean_dt_mm": None,
            "mean_dr_deg": None,
            "max_dt_mm": None,
            "max_dr_deg": None,
            "pass": None,
        }
    return {
        "num_events": int(num_events),
        "mean_dt_mm": float(np.mean(event_dt)),
        "mean_dr_deg": float(np.mean(event_dr)),
        "max_dt_mm": float(np.max(event_dt)),
        "max_dr_deg": float(np.max(event_dr)),
        "pass": bool(np.mean(event_dt) < 5.0 and np.mean(event_dr) < 1.0),
    }


def _depth_roi_points(depth_u16: np.ndarray,
                      bbox: Tuple[int, int, int, int],
                      K: np.ndarray,
                      depth_scale: float,
                      z_min: float,
                      z_max: float,
                      stride: int = 2) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth_u16.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    pts = []
    for v in range(y0, y1, max(int(stride), 1)):
        for u in range(x0, x1, max(int(stride), 1)):
            d = int(depth_u16[v, u])
            if d <= 0:
                continue
            z = float(d) * float(depth_scale)
            if z < z_min or z > z_max:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)


def _mask_points_to_polygon(points_cam: np.ndarray,
                            K: np.ndarray,
                            polygon_xy: np.ndarray,
                            image_shape: Tuple[int, int]) -> np.ndarray:
    if points_cam.size == 0 or polygon_xy.size == 0:
        return np.zeros((points_cam.shape[0],), dtype=bool)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.maximum(points_cam[:, 2], 1e-9)
    u = points_cam[:, 0] * fx / z + cx
    v = points_cam[:, 1] * fy / z + cy
    h, w = image_shape[:2]
    mask = np.zeros((points_cam.shape[0],), dtype=bool)
    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 1, 2)
    for idx, (uu, vv) in enumerate(zip(u, v)):
        if uu < 0 or uu >= w or vv < 0 or vv >= h:
            continue
        inside = cv2.pointPolygonTest(poly, (float(uu), float(vv)), False)
        mask[idx] = inside >= 0
    return mask


def _project_cube_silhouette(T_C_O: np.ndarray,
                             cube_cfg: CubeConfig,
                             K: np.ndarray,
                             D: np.ndarray) -> np.ndarray:
    half_side = float(cube_cfg.cube_side_m) * 0.5
    vertices = np.asarray([
        [-half_side, -half_side, -half_side],
        [half_side, -half_side, -half_side],
        [half_side, half_side, -half_side],
        [-half_side, half_side, -half_side],
        [-half_side, -half_side, half_side],
        [half_side, -half_side, half_side],
        [half_side, half_side, half_side],
        [-half_side, half_side, half_side],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(np.asarray(T_C_O[:3, :3], dtype=np.float64))
    tvec = np.asarray(T_C_O[:3, 3], dtype=np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(vertices.reshape(-1, 1, 3), rvec, tvec, K, D)
    pts = proj.reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    return hull.reshape(-1, 2)


def _cube_surface_distance_mm(points_cube: np.ndarray, half_side_m: float) -> np.ndarray:
    if points_cube.size == 0:
        return np.empty((0,), dtype=np.float64)
    d = np.abs(np.abs(points_cube) - float(half_side_m))
    return np.min(d, axis=1) * 1000.0


def _estimate_cube_dimension_error_mm(points_cube: np.ndarray,
                                      cube_side_m: float,
                                      surface_band_m: float = 0.006) -> Optional[float]:
    if points_cube.size == 0:
        return None
    half_side = float(cube_side_m) * 0.5
    abs_pts = np.abs(points_cube)
    face_axis = np.argmax(abs_pts, axis=1)
    dim_errors = []
    for axis in range(3):
        axis_mask = face_axis == axis
        if not np.any(axis_mask):
            continue
        coords = points_cube[axis_mask, axis]
        pos = coords[coords > 0]
        neg = coords[coords < 0]
        pos = pos[np.abs(pos - half_side) <= surface_band_m]
        neg = neg[np.abs(neg + half_side) <= surface_band_m]
        if pos.size < 20 or neg.size < 20:
            continue
        pos_plane = float(np.median(pos))
        neg_plane = float(np.median(neg))
        side_est = pos_plane - neg_plane
        dim_errors.append(abs(side_est - float(cube_side_m)) * 1000.0)
    if not dim_errors:
        return None
    return float(np.mean(dim_errors))


def compute_depth_cube_metrics(meta: dict,
                               transforms: Dict[str, np.ndarray],
                               intrinsics_dir: str,
                               root_folder: str,
                               all_cam_ids: List[int],
                               gripper_cam_idx: Optional[int],
                               cube_cfg: CubeConfig,
                               include_meta: bool = False,
                               selection_profile: str = "default",
                               stride: int = 2) -> dict:
    selection = build_event_cube_selection(
        meta, transforms, intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta, selection_profile=selection_profile)
    cube = ArucoCubeTarget(cube_cfg)
    half_side = float(cube_cfg.cube_side_m) * 0.5
    mesh_rmse_mm = []
    dim_abs_err_mm = []
    usable_clouds = 0
    usable_events = 0

    intr_map = {
        int(ci): load_intrinsics_with_depth_scale(intrinsics_dir, int(ci))
        for ci in all_cam_ids
    }

    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        refined = selection.get(eid, {})
        if not refined:
            continue
        event_points_cube = []
        for ci, cand in refined.items():
            cinfo = cap.get("cams", {}).get(str(ci), {})
            depth_rel = cinfo.get("depth_path", "")
            rgb_rel = cinfo.get("rgb_path", "")
            if not depth_rel or not rgb_rel:
                continue
            depth = cv2.imread(os.path.join(root_folder, depth_rel), cv2.IMREAD_UNCHANGED)
            img = cv2.imread(os.path.join(root_folder, rgb_rel))
            if depth is None or img is None:
                continue
            obs_set = cube.collect_observations(img, min_aspect=0.0)
            bbox = obs_set.image_bbox(pad_px=24.0)
            if bbox is None:
                continue
            K, D, depth_scale = intr_map[int(ci)]
            z_center = float(np.asarray(cand["T_C_O"], dtype=np.float64)[2, 3])
            pts_cam = _depth_roi_points(
                depth, bbox, K, depth_scale,
                z_min=max(0.05, z_center - 0.08),
                z_max=z_center + 0.08,
                stride=stride,
            )
            if pts_cam.shape[0] < 40:
                continue
            hull = _project_cube_silhouette(np.asarray(cand["T_C_O"], dtype=np.float64), cube_cfg, K, D)
            if hull.size:
                keep_poly = _mask_points_to_polygon(pts_cam, K, hull, depth.shape)
                if np.any(keep_poly):
                    pts_cam = pts_cam[keep_poly]
            if pts_cam.shape[0] < 30:
                continue
            T_O_C = inv_T(np.asarray(cand["T_C_O"], dtype=np.float64))
            pts_cube = (T_O_C[:3, :3] @ pts_cam.T).T + T_O_C[:3, 3]
            keep = np.all(np.abs(pts_cube) <= (half_side + 0.012), axis=1)
            pts_cube = pts_cube[keep]
            if pts_cube.size:
                surface_dist_mm = _cube_surface_distance_mm(pts_cube, half_side)
                pts_cube = pts_cube[surface_dist_mm <= 8.0]
            if pts_cube.shape[0] < 30:
                continue
            usable_clouds += 1
            event_points_cube.append(pts_cube)
            dists = _cube_surface_distance_mm(pts_cube, half_side)
            if dists.size:
                mesh_rmse_mm.append(float(np.sqrt(np.mean(np.square(dists)))))

        if event_points_cube:
            usable_events += 1
            pts = np.concatenate(event_points_cube, axis=0)
            dim_err = _estimate_cube_dimension_error_mm(pts, cube_cfg.cube_side_m)
            if dim_err is not None:
                dim_abs_err_mm.append(float(dim_err))

    mesh_metrics = {
        "usable_clouds": int(usable_clouds),
        "usable_events": int(usable_events),
        "mean_rmse_mm": None if not mesh_rmse_mm else float(np.mean(mesh_rmse_mm)),
        "median_rmse_mm": None if not mesh_rmse_mm else float(np.median(mesh_rmse_mm)),
        "max_rmse_mm": None if not mesh_rmse_mm else float(np.max(mesh_rmse_mm)),
        "pass": None if not mesh_rmse_mm else bool(np.mean(mesh_rmse_mm) < 5.0),
    }
    dim_metrics = {
        "usable_events": int(len(dim_abs_err_mm)),
        "mean_abs_err_mm": None if not dim_abs_err_mm else float(np.mean(dim_abs_err_mm)),
        "median_abs_err_mm": None if not dim_abs_err_mm else float(np.median(dim_abs_err_mm)),
        "max_abs_err_mm": None if not dim_abs_err_mm else float(np.max(dim_abs_err_mm)),
        "pass": None if not dim_abs_err_mm else bool(np.mean(dim_abs_err_mm) < 3.0),
    }
    return {
        "mesh_alignment": mesh_metrics,
        "dimension_accuracy": dim_metrics,
    }
