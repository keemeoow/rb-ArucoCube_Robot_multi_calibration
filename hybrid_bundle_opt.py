import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from aruco_cube import inv_T


def pose6_from_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return np.concatenate([T[:3, 3], rvec.reshape(3)], axis=0).astype(np.float64)


def transform_from_pose6(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    R, _ = cv2.Rodrigues(x[3:].reshape(3, 1))
    T[:3, :3] = R
    T[:3, 3] = x[:3]
    return T


def transform_error_vector(T_pred: np.ndarray,
                           T_ref: np.ndarray,
                           trans_scale_mm: float = 1000.0,
                           rot_scale_deg: float = 180.0 / math.pi) -> np.ndarray:
    T_pred = np.asarray(T_pred, dtype=np.float64).reshape(4, 4)
    T_ref = np.asarray(T_ref, dtype=np.float64).reshape(4, 4)
    t_err = (T_pred[:3, 3] - T_ref[:3, 3]) * float(trans_scale_mm)
    R_err = T_ref[:3, :3].T @ T_pred[:3, :3]
    rvec, _ = cv2.Rodrigues(R_err)
    r_err = rvec.reshape(3) * float(rot_scale_deg)
    return np.concatenate([t_err, r_err], axis=0).astype(np.float64)


def _pack_variables(initial_transforms: Dict[str, np.ndarray],
                    fixed_cam_ids: List[int]) -> Tuple[np.ndarray, List[str]]:
    names = ["T_gripper_cam", "T_base_board", "T_base_O"]
    for ci in fixed_cam_ids:
        key = f"T_base_C{int(ci)}"
        if key in initial_transforms:
            names.append(key)
    x0 = np.concatenate([pose6_from_transform(initial_transforms[name]) for name in names], axis=0)
    return x0, names


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _unpack_variables(x: np.ndarray, names: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for i, name in enumerate(names):
        out[name] = transform_from_pose6(x[6 * i:6 * (i + 1)])
    return out


def _pack_cube_only_variables(initial_transforms: Dict[str, np.ndarray],
                              fixed_cam_ids: List[int],
                              observation_keys: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[str], Dict[Tuple[int, int], int]]:
    names = ["T_gripper_cam", "T_base_O"]
    for ci in fixed_cam_ids:
        key = f"T_base_C{int(ci)}"
        if key in initial_transforms:
            names.append(key)
    pose_part = np.concatenate([pose6_from_transform(initial_transforms[name]) for name in names], axis=0)
    switch_init = np.full((len(observation_keys),), 2.2, dtype=np.float64)
    switch_map = {tuple(map(int, key)): idx for idx, key in enumerate(observation_keys)}
    return np.concatenate([pose_part, switch_init], axis=0), names, switch_map


def refine_hybrid_calibration(
    initial_transforms: Dict[str, np.ndarray],
    fixed_cam_ids: List[int],
    robot_poses: Dict[int, np.ndarray],
    gripper_board_obs: Dict[int, dict],
    fixed_board_obs: Dict[int, Dict[int, dict]],
    selected_cube_obs: Dict[int, Dict[int, dict]],
    gripper_cam_idx: int,
    board_weight: float = 1.0,
    cube_weight: float = 0.35,
    pair_weight: float = 0.15,
    prior_weight: float = 0.03,
    max_nfev: int = 200,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Jointly refine board and cube alignment around a board-seeded solution."""
    base_required = ("T_gripper_cam", "T_base_board", "T_base_O")
    for key in base_required:
        if key not in initial_transforms:
            raise KeyError(f"Missing required initial transform: {key}")

    x0, names = _pack_variables(initial_transforms, fixed_cam_ids)
    initial_cache = {name: np.asarray(initial_transforms[name], dtype=np.float64).reshape(4, 4) for name in names}

    def _base_cam_transform(var_map: Dict[str, np.ndarray], eid: int, ci: int) -> Optional[np.ndarray]:
        if int(ci) == int(gripper_cam_idx):
            if eid not in robot_poses:
                return None
            return np.asarray(robot_poses[eid], dtype=np.float64) @ var_map["T_gripper_cam"]
        key = f"T_base_C{int(ci)}"
        return var_map.get(key)

    def residuals(x: np.ndarray) -> np.ndarray:
        var_map = _unpack_variables(x, names)
        res = []

        # Board residuals
        B = var_map["T_base_board"]
        for eid, obs in gripper_board_obs.items():
            if eid not in robot_poses:
                continue
            T_pred = np.asarray(robot_poses[eid], dtype=np.float64) @ var_map["T_gripper_cam"] @ np.asarray(obs["T_cam_board"], dtype=np.float64)
            w = math.sqrt(float(board_weight) / max(float(obs.get("reproj", 1.0)), 1e-6))
            res.extend((w * transform_error_vector(T_pred, B)).tolist())

        for ci, per_event in fixed_board_obs.items():
            key = f"T_base_C{int(ci)}"
            if key not in var_map:
                continue
            for eid, obs in per_event.items():
                T_pred = var_map[key] @ np.asarray(obs["T_cam_board"], dtype=np.float64)
                w = math.sqrt(float(board_weight) / max(float(obs.get("reproj", 1.0)), 1e-6))
                res.extend((w * transform_error_vector(T_pred, B)).tolist())

        # Cube residuals to a single static cube anchor in base.
        O = var_map["T_base_O"]
        event_pose_rows = []
        for ci, per_event in selected_cube_obs.items():
            for eid, obs in per_event.items():
                T_base_cam = _base_cam_transform(var_map, int(eid), int(ci))
                if T_base_cam is None:
                    continue
                T_pred = T_base_cam @ np.asarray(obs["T_C_O"], dtype=np.float64)
                w = math.sqrt(float(cube_weight) / max(float(obs.get("err_mean", 1.0)), 1e-6))
                err_vec = w * transform_error_vector(T_pred, O)
                res.extend(err_vec.tolist())
                event_pose_rows.append((int(eid), int(ci), T_pred, float(obs.get("err_mean", 1.0))))

        # Pairwise same-event consistency helps suppress cube family conflicts.
        by_event: Dict[int, List[Tuple[int, np.ndarray, float]]] = {}
        for eid, ci, T_pred, err_mean in event_pose_rows:
            by_event.setdefault(int(eid), []).append((int(ci), T_pred, float(err_mean)))
        for eid, rows in by_event.items():
            if len(rows) < 2:
                continue
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    ci, Ti, ei = rows[i]
                    cj, Tj, ej = rows[j]
                    w = math.sqrt(float(pair_weight) / max((ei + ej) * 0.5, 1e-6))
                    res.extend((w * transform_error_vector(Ti, Tj)).tolist())

        # Light prior keeps the optimizer close to the board-seeded physically valid solution.
        if prior_weight > 0:
            w_prior = math.sqrt(float(prior_weight))
            for name in names:
                res.extend((w_prior * transform_error_vector(var_map[name], initial_cache[name])).tolist())

        return np.asarray(res, dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        verbose=0,
    )

    refined = _unpack_variables(result.x, names)
    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "num_residuals": int(result.fun.size),
        "variables": list(names),
        "weights": {
            "board_weight": float(board_weight),
            "cube_weight": float(cube_weight),
            "pair_weight": float(pair_weight),
            "prior_weight": float(prior_weight),
        },
        "observation_counts": {
            "gripper_board": int(len(gripper_board_obs)),
            "fixed_board": int(sum(len(v) for v in fixed_board_obs.values())),
            "cube_total": int(sum(len(v) for v in selected_cube_obs.values())),
        },
    }
    return refined, report


def refine_cube_only_calibration(
    initial_transforms: Dict[str, np.ndarray],
    fixed_cam_ids: List[int],
    robot_poses: Dict[int, np.ndarray],
    selected_cube_obs: Dict[int, Dict[int, dict]],
    gripper_cam_idx: int,
    cube_weight: float = 1.0,
    pair_weight: float = 0.25,
    prior_weight: float = 0.03,
    switch_prior_weight: float = 1.0,
    switch_floor: float = 0.05,
    max_nfev: int = 250,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Cube-only bundle refinement with switchable constraints per observation."""
    base_required = ("T_gripper_cam", "T_base_O")
    for key in base_required:
        if key not in initial_transforms:
            raise KeyError(f"Missing required initial transform: {key}")

    observation_keys = sorted(
        (int(ci), int(eid))
        for ci, per_event in selected_cube_obs.items()
        for eid in per_event.keys()
    )
    x0, names, switch_map = _pack_cube_only_variables(initial_transforms, fixed_cam_ids, observation_keys)
    pose_dim = 6 * len(names)
    initial_cache = {name: np.asarray(initial_transforms[name], dtype=np.float64).reshape(4, 4) for name in names}

    def _base_cam_transform(var_map: Dict[str, np.ndarray], eid: int, ci: int) -> Optional[np.ndarray]:
        if int(ci) == int(gripper_cam_idx):
            if eid not in robot_poses:
                return None
            return np.asarray(robot_poses[eid], dtype=np.float64) @ var_map["T_gripper_cam"]
        return var_map.get(f"T_base_C{int(ci)}")

    def residuals(x: np.ndarray) -> np.ndarray:
        var_map = _unpack_variables(x[:pose_dim], names)
        switch_raw = x[pose_dim:]
        switch_vals = _sigmoid(switch_raw)
        O = var_map["T_base_O"]
        event_rows: Dict[int, List[Tuple[int, np.ndarray, float, float]]] = {}
        res = []

        for (ci, eid), idx in switch_map.items():
            obs = selected_cube_obs.get(int(ci), {}).get(int(eid))
            if obs is None:
                continue
            T_base_cam = _base_cam_transform(var_map, int(eid), int(ci))
            if T_base_cam is None:
                continue
            T_pred = T_base_cam @ np.asarray(obs["T_C_O"], dtype=np.float64)
            s = float(max(switch_vals[idx], switch_floor))
            face_weight = float(obs.get("face_weight", 1.0))
            err_mean = float(obs.get("err_mean", 1.0))
            w = math.sqrt(float(cube_weight) * face_weight / max(err_mean, 1e-6))
            res.extend((w * s * transform_error_vector(T_pred, O)).tolist())
            res.append(math.sqrt(float(switch_prior_weight)) * (1.0 - s))
            event_rows.setdefault(int(eid), []).append((int(ci), T_pred, err_mean, s))

        for eid, rows in event_rows.items():
            if len(rows) < 2:
                continue
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    ci, Ti, ei, si = rows[i]
                    cj, Tj, ej, sj = rows[j]
                    w = math.sqrt(float(pair_weight) * max(si * sj, switch_floor) / max((ei + ej) * 0.5, 1e-6))
                    res.extend((w * transform_error_vector(Ti, Tj)).tolist())

        if prior_weight > 0:
            w_prior = math.sqrt(float(prior_weight))
            for name in names:
                res.extend((w_prior * transform_error_vector(var_map[name], initial_cache[name])).tolist())

        return np.asarray(res, dtype=np.float64)

    result = least_squares(
        residuals,
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        verbose=0,
    )

    refined = _unpack_variables(result.x[:pose_dim], names)
    switch_vals = _sigmoid(result.x[pose_dim:])
    switch_report = {}
    for (ci, eid), idx in switch_map.items():
        switch_report[f"cam{int(ci)}_event{int(eid)}"] = float(switch_vals[idx])

    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "num_residuals": int(result.fun.size),
        "variables": list(names),
        "weights": {
            "cube_weight": float(cube_weight),
            "pair_weight": float(pair_weight),
            "prior_weight": float(prior_weight),
            "switch_prior_weight": float(switch_prior_weight),
        },
        "observation_counts": {
            "cube_total": int(len(observation_keys)),
        },
        "switch_values": switch_report,
        "switch_summary": {
            "mean": float(np.mean(switch_vals)) if switch_vals.size else None,
            "median": float(np.median(switch_vals)) if switch_vals.size else None,
            "min": float(np.min(switch_vals)) if switch_vals.size else None,
            "num_low_switch": int(np.sum(switch_vals < 0.5)) if switch_vals.size else 0,
        },
    }
    return refined, report
