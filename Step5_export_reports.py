#!/usr/bin/env python3
"""
Generate final-use exports from the stable calibration/verification pipeline.

Example:
  python Step5_export_reports.py \
    --root_folder "data/session" \
    --intrinsics_dir "intrinsics" \
    --calib_dir "data/session/calib_out"
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_step5_export_reports")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aruco_cube import ArucoCubeTarget, inv_T, rodrigues_to_Rt
from calibration_runtime_utils import (
    build_capture_cube_candidate_map,
    build_cube_pose_candidates,
    cube_selection_profile_kwargs,
    get_event_base_camera_transform,
    load_calib_dir,
    load_intrinsics_color,
    load_intrinsics_with_depth_scale,
    load_robot_pose_from_capture,
    resolve_cube_config_for_run,
    select_consistent_event_cube_candidates,
    select_primary_cube_candidate,
)
from config import CharucoBoardConfig, CubeConfig, get_default_cube_config
from cube_config_utils import (
    cube_config_to_dict,
)
from downstream_metrics import (
    compute_board_reprojection_metrics,
    compute_depth_cube_metrics,
    compute_pose_repeatability_metrics,
)
from Step4_verify import collect_cube_candidate_diagnostics


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


load_calib = load_calib_dir


def matrix_to_rzryrx_deg(T: np.ndarray) -> Tuple[float, float, float]:
    R = T[:3, :3]
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = np.arcsin(sy)
    cy = np.cos(ry)
    if abs(cy) > 1e-8:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = 0.0
        rz = np.arctan2(-R[0, 1], R[1, 1])
    return tuple(float(np.degrees(v)) for v in (rz, ry, rx))


def matrix_to_nested_list(T: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in np.asarray(T, dtype=np.float64)]


def format_used_ids(counter: Counter, limit: int = 3) -> str:
    parts = []
    for used_ids, count in counter.most_common(limit):
        label = "+".join(f"id{mid}" for mid in used_ids)
        parts.append(f"{label} x{count}")
    return ", ".join(parts)


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def bool_pass_fail(ok: Optional[bool]) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "N/A"


def write_final_use_bundle(export_dir: str,
                           summary: dict,
                           verification: dict,
                           cube_cfg: CubeConfig,
                           cube_cfg_source: str,
                           usable: dict,
                           excluded: dict) -> List[str]:
    final_dir = ensure_dir(os.path.join(export_dir, "final_use"))

    transforms_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cube_config_source": cube_cfg_source,
        "cube_config_used": cube_config_to_dict(cube_cfg),
        "verification": verification,
        "usable_transforms": usable,
        "excluded_transforms": excluded,
    }
    transforms_json = os.path.join(final_dir, "usable_transforms_final.json")
    transforms_npz = os.path.join(final_dir, "usable_transforms_final.npz")
    cube_config_json = os.path.join(final_dir, "cube_config_used.json")
    summary_json = os.path.join(final_dir, "calibration_summary_snapshot.json")
    verification_json = os.path.join(final_dir, "verification_metrics.json")
    readme_path = os.path.join(final_dir, "README.md")

    with open(transforms_json, "w") as f:
        json.dump(transforms_payload, f, indent=2)
    if usable:
        np.savez(transforms_npz, **{
            k: np.asarray(v["matrix_4x4"], dtype=np.float64)
            for k, v in usable.items()
        })
    with open(cube_config_json, "w") as f:
        json.dump(cube_config_to_dict(cube_cfg), f, indent=2)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(verification_json, "w") as f:
        json.dump(verification, f, indent=2)

    readme_lines = [
        "# Final Use Export",
        "",
        "This folder contains the final-use export package for the selected calibration run.",
        "",
        "## Included files",
        "- `usable_transforms_final.json`: pass-only transform export with verification snapshot",
        "- `usable_transforms_final.npz`: pass-only 4x4 matrices in NumPy format",
        "- `cube_config_used.json`: cube model/config used for this run",
        "- `calibration_summary_snapshot.json`: full calibration summary snapshot",
        "- `verification_metrics.json`: verification metrics snapshot",
        "",
        "## Current quality summary",
        f"- Cross-camera mean: {verification.get('cross_camera', {}).get('mean_mm')}",
        f"- Reprojection mean: {verification.get('reprojection', {}).get('mean_px')}",
        f"- Hand-eye pass: {verification.get('handeye', {}).get('pass')}",
        "",
        "## Included transforms",
    ]
    if usable:
        readme_lines.extend([f"- `{name}` ({payload.get('quality_tier', 'unknown')})" for name, payload in sorted(usable.items())])
    else:
        readme_lines.append("- None")
    save_text(readme_path, "\n".join(readme_lines) + "\n")

    saved = [transforms_json, cube_config_json, summary_json, verification_json, readme_path]
    if usable:
        saved.append(transforms_npz)
    return saved


def compute_cross_camera_metrics(meta: dict, transforms: Dict[str, np.ndarray],
                                 all_cam_ids: List[int], gripper_cam_idx: Optional[int],
                                 root_folder: Optional[str] = None,
                                 intrinsics_dir: Optional[str] = None,
                                 cube_cfg: Optional[CubeConfig] = None,
                                 include_meta: bool = False,
                                 selection_profile: str = "default") -> dict:
    tf = dict(transforms)

    use_current_cube = bool(root_folder and intrinsics_dir and cube_cfg is not None)
    cube = None
    K_map, D_map, depth_scale_map = {}, {}, {}
    if use_current_cube:
        cube = ArucoCubeTarget(cube_cfg)
        for ci in all_cam_ids:
            K_map[ci], D_map[ci], depth_scale_map[ci] = load_intrinsics_with_depth_scale(intrinsics_dir, ci)
    profile_kwargs = cube_selection_profile_kwargs(selection_profile)

    errors_mm = []
    n_events = 0
    for cap in meta.get("captures", []):
        positions = []
        event_candidate_map = {}
        if use_current_cube:
            event_candidate_map = build_capture_cube_candidate_map(
                cap, root_folder, K_map, D_map, cube, gripper_cam_idx,
                include_meta=include_meta, depth_scale_map=depth_scale_map)
        refined_selection = select_consistent_event_cube_candidates(
            cap, event_candidate_map, tf, gripper_cam_idx, **profile_kwargs) if event_candidate_map else {}

        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in all_cam_ids:
                continue
            T_cam_cube = None
            if use_current_cube:
                if ci not in refined_selection:
                    continue
                T_cam_cube = np.asarray(refined_selection[ci]["T_C_O"], dtype=np.float64)
            else:
                cpnp = cinfo.get("cube_pnp")
                if cpnp and cpnp.get("ok"):
                    T_cam_cube = np.asarray(cpnp["T_cam_cube_4x4"], dtype=np.float64)
            if T_cam_cube is None or not np.all(np.isfinite(T_cam_cube)):
                continue
            T_base_cam = get_event_base_camera_transform(cap, ci, tf, gripper_cam_idx)
            if T_base_cam is None:
                continue
            T_base_cube = T_base_cam @ T_cam_cube
            positions.append(T_base_cube[:3, 3] * 1000.0)
        if len(positions) < 2:
            continue
        n_events += 1
        positions = np.asarray(positions)
        mean_pos = positions.mean(axis=0)
        for pos in positions:
            errors_mm.append(float(np.linalg.norm(pos - mean_pos)))

    metrics = {
        "events_with_2plus_cams": int(n_events),
        "num_errors": int(len(errors_mm)),
        "mean_mm": None,
        "median_mm": None,
        "max_mm": None,
        "std_mm": None,
        "pass": None,
    }
    if errors_mm:
        arr = np.asarray(errors_mm, dtype=np.float64)
        metrics.update({
            "mean_mm": float(np.mean(arr)),
            "median_mm": float(np.median(arr)),
            "max_mm": float(np.max(arr)),
            "std_mm": float(np.std(arr)),
            "pass": bool(np.mean(arr) < 5.0),
        })
    return metrics


def compute_reprojection_metrics(meta: dict, transforms: Dict[str, np.ndarray], intrinsics_dir: str,
                                 all_cam_ids: List[int], root_folder: str, gripper_cam_idx: Optional[int],
                                 cube_cfg: CubeConfig, include_meta: bool = False) -> dict:
    cube = ArucoCubeTarget(cube_cfg)
    K_map, D_map, depth_scale_map = {}, {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], depth_scale_map[ci] = load_intrinsics_with_depth_scale(intrinsics_dir, ci)

    errors_px = []
    for cap in meta.get("captures", []):
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in K_map or not cinfo.get("saved"):
                continue
            meta_thr = 5.0 if ci == gripper_cam_idx else 3.0
            candidates = build_cube_pose_candidates(
                root_folder, cinfo, K_map[ci], D_map[ci], cube,
                meta_reproj_thr=meta_thr, solve_reproj_thr=5.0,
                min_aspect=0.0, include_meta=include_meta,
                depth_scale=depth_scale_map.get(ci))
            if not candidates:
                continue
            best = select_primary_cube_candidate(candidates)
            if best is None:
                continue

            rgb_rel = cinfo.get("rgb_path", "")
            if not rgb_rel:
                continue
            img = cv2.imread(os.path.join(root_folder, rgb_rel))
            if img is None:
                continue
            corners_list, ids = cube.detect(img)
            if ids is None:
                continue

            R = best["T_C_O"][:3, :3]
            t = best["T_C_O"][:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            total_err = []
            for corners, mid in zip(corners_list, ids):
                mid = int(mid)
                if not cube.model.has_marker(mid):
                    continue
                img_pts = cube.model.reorder_image_corners(mid, corners.reshape(4, 2).astype(np.float64))
                obj_pts = cube.model.marker_corners_in_rig(mid)
                proj, _ = cv2.projectPoints(
                    obj_pts.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K_map[ci], D_map[ci]
                )
                proj = proj.reshape(-1, 2)
                total_err.extend(np.linalg.norm(proj - img_pts, axis=1).tolist())
            if total_err:
                errors_px.append(float(np.mean(total_err)))

    metrics = {
        "total_observations": int(len(errors_px)),
        "mean_px": None,
        "median_px": None,
        "max_px": None,
        "lt_1px_count": 0,
        "pass": None,
    }
    if errors_px:
        arr = np.asarray(errors_px, dtype=np.float64)
        metrics.update({
            "mean_px": float(np.mean(arr)),
            "median_px": float(np.median(arr)),
            "max_px": float(np.max(arr)),
            "lt_1px_count": int(np.sum(arr < 1.0)),
            "pass": bool(np.mean(arr) < 2.0),
        })
    return metrics


def compute_handeye_metrics(meta: dict, transforms: Dict[str, np.ndarray], gripper_cam_idx: int,
                            root_folder: str, intrinsics_dir: str) -> dict:
    metrics = {
        "frames": 0,
        "board_position_std_mm": None,
        "board_position_max_mm": None,
        "board_rotation_mean_deg": None,
        "board_rotation_max_deg": None,
        "pass": None,
    }
    if transforms.get("T_gripper_cam") is None:
        return metrics

    charuco_by_event = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        gi_data = cap.get("cams", {}).get(str(gripper_cam_idx), {})
        ch = gi_data.get("charuco")
        if ch and ch.get("ok") and ch.get("T_cam_board_4x4") is not None:
            charuco_by_event[eid] = np.asarray(ch["T_cam_board_4x4"], dtype=np.float64)

    if len(charuco_by_event) < 2:
        from charuco_utils import CharucoTarget

        g_K, g_D = load_intrinsics_color(intrinsics_dir, gripper_cam_idx)
        charuco_det = CharucoTarget(CharucoBoardConfig())
        for cap in meta.get("captures", []):
            eid = int(cap.get("event_id", -1))
            if eid < 0 or eid in charuco_by_event:
                continue
            gi_data = cap.get("cams", {}).get(str(gripper_cam_idx), {})
            rgb_rel = gi_data.get("rgb_path", "")
            if not rgb_rel:
                continue
            img = cv2.imread(os.path.join(root_folder, rgb_rel))
            if img is None:
                continue
            ok, ch_rvec, ch_tvec, ch_n, _ = charuco_det.estimate_pose(img, g_K, g_D)
            if ok and ch_rvec is not None and ch_n >= 4:
                charuco_by_event[eid] = rodrigues_to_Rt(ch_rvec, ch_tvec)

    T_base_board_list = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        T_base_cam = get_event_base_camera_transform(cap, gripper_cam_idx, transforms, gripper_cam_idx)
        if T_base_cam is None:
            continue
        T_cam_board = charuco_by_event.get(eid)
        if T_cam_board is None:
            continue
        T_base_board_list.append(np.asarray(T_base_cam, dtype=np.float64) @ T_cam_board)

    if len(T_base_board_list) < 2:
        return metrics

    positions = np.array([T[:3, 3] * 1000.0 for T in T_base_board_list], dtype=np.float64)
    mean_pos = positions.mean(axis=0)
    pos_errors = np.array([np.linalg.norm(p - mean_pos) for p in positions], dtype=np.float64)
    rot_errors = np.array(
        [rotation_error_deg(T[:3, :3], T_base_board_list[0][:3, :3]) for T in T_base_board_list],
        dtype=np.float64,
    )
    metrics.update({
        "frames": int(len(T_base_board_list)),
        "board_position_std_mm": float(np.std(pos_errors)),
        "board_position_max_mm": float(np.max(pos_errors)),
        "board_rotation_mean_deg": float(np.mean(rot_errors)),
        "board_rotation_max_deg": float(np.max(rot_errors)),
        "pass": bool(np.std(pos_errors) < 3.0 and np.mean(rot_errors) < 1.0),
    })
    return metrics


def load_transform_set_from_summary(summary: dict, mode: str) -> Dict[str, np.ndarray]:
    raw = summary.get("transform_sets", {}).get(mode, {})
    return {
        str(name): np.asarray(vals, dtype=np.float64).reshape(4, 4)
        for name, vals in raw.items()
    }


def compute_full_verification_bundle(meta: dict,
                                     transforms: Dict[str, np.ndarray],
                                     intrinsics_dir: str,
                                     root_folder: str,
                                     all_cam_ids: List[int],
                                     gripper_cam_idx: int,
                                     cube_cfg: CubeConfig,
                                     include_meta: bool = False,
                                     selection_profile: str = "default") -> dict:
    verification = {
        "cross_camera": compute_cross_camera_metrics(
            meta, transforms, all_cam_ids, gripper_cam_idx,
            root_folder=root_folder, intrinsics_dir=intrinsics_dir,
            cube_cfg=cube_cfg, include_meta=False,
            selection_profile=selection_profile),
        "reprojection": compute_reprojection_metrics(
            meta, transforms, intrinsics_dir, all_cam_ids, root_folder,
            gripper_cam_idx, cube_cfg, include_meta=include_meta),
        "handeye": compute_handeye_metrics(meta, transforms, gripper_cam_idx, root_folder, intrinsics_dir),
        "board_reprojection": compute_board_reprojection_metrics(
            meta, root_folder, intrinsics_dir, all_cam_ids),
        "pose_repeatability": compute_pose_repeatability_metrics(
            meta, transforms, intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
            cube_cfg, include_meta=include_meta,
            selection_profile=selection_profile),
    }
    depth_bundle = compute_depth_cube_metrics(
        meta, transforms, intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta,
        selection_profile=selection_profile)
    verification["mesh_alignment"] = depth_bundle["mesh_alignment"]
    verification["dimension_accuracy"] = depth_bundle["dimension_accuracy"]
    return verification


def build_mode_comparison_rows(summary: dict,
                               meta: dict,
                               intrinsics_dir: str,
                               root_folder: str,
                               all_cam_ids: List[int],
                               gripper_cam_idx: int,
                               cube_cfg: CubeConfig,
                               include_meta: bool = False) -> Tuple[List[dict], Dict[str, dict]]:
    rows = []
    bundles = {}
    for mode in ("board_only", "cube_only", "hybrid"):
        transforms = load_transform_set_from_summary(summary, mode)
        if not transforms:
            continue
        selection_profile = "cube_only_specialized" if mode == "cube_only" else "default"
        bundle = compute_full_verification_bundle(
            meta, transforms, intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
            cube_cfg, include_meta=include_meta, selection_profile=selection_profile)
        bundles[mode] = bundle
        available_cams = sorted(
            int(name.replace("T_base_C", ""))
            for name in transforms.keys()
            if name.startswith("T_base_C")
        )
        rows.append({
            "mode": mode,
            "num_base_cameras": str(len(available_cams)),
            "base_cameras": ", ".join(f"cam{ci}" for ci in available_cams),
            "cross_camera_mean_mm": "" if bundle["cross_camera"]["mean_mm"] is None else f"{bundle['cross_camera']['mean_mm']:.2f}",
            "cube_reproj_mean_px": "" if bundle["reprojection"]["mean_px"] is None else f"{bundle['reprojection']['mean_px']:.3f}",
            "board_reproj_mean_px": "" if bundle["board_reprojection"]["mean_px"] is None else f"{bundle['board_reprojection']['mean_px']:.3f}",
            "mesh_rmse_mm": "" if bundle["mesh_alignment"]["mean_rmse_mm"] is None else f"{bundle['mesh_alignment']['mean_rmse_mm']:.2f}",
            "dimension_err_mm": "" if bundle["dimension_accuracy"]["mean_abs_err_mm"] is None else f"{bundle['dimension_accuracy']['mean_abs_err_mm']:.2f}",
            "pose_repeat_mm": "" if bundle["pose_repeatability"]["mean_dt_mm"] is None else f"{bundle['pose_repeatability']['mean_dt_mm']:.2f}",
            "pose_repeat_deg": "" if bundle["pose_repeatability"]["mean_dr_deg"] is None else f"{bundle['pose_repeatability']['mean_dr_deg']:.3f}",
            "handeye_pass": bool_pass_fail(bundle["handeye"]["pass"]),
        })
    return rows, bundles


def save_mode_comparison_report(path: str, rows: List[dict]) -> None:
    lines = [
        "# Calibration Mode Comparison",
        "",
        "Planar board seed, cube-only, and hybrid refinement were re-evaluated on the same dataset.",
        "",
        render_markdown_table(rows, [
            "mode",
            "num_base_cameras",
            "base_cameras",
            "cross_camera_mean_mm",
            "cube_reproj_mean_px",
            "board_reproj_mean_px",
            "mesh_rmse_mm",
            "dimension_err_mm",
            "pose_repeat_mm",
            "pose_repeat_deg",
            "handeye_pass",
        ]),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def evaluate_export_status(name: str, summary: dict, verification: dict) -> Tuple[str, str]:
    handeye = summary.get("diagnostics", {}).get("handeye_methods", {})
    selected_method = summary.get("selected_handeye_method")
    selected_handeye = handeye.get(selected_method, {})
    base_stats = summary.get("diagnostics", {}).get("base_transforms", {})
    cube_anchor = summary.get("diagnostics", {}).get("cube_anchor") or {}
    cube_anchor_by_set = summary.get("diagnostics", {}).get("cube_anchor_by_set") or {}
    cube_anchor_per_set = cube_anchor_by_set.get("per_set", {}) or {}
    hybrid = summary.get("diagnostics", {}).get("hybrid_refinement", {}) or {}
    cross_camera = verification.get("cross_camera", {}) or {}
    reprojection = verification.get("reprojection", {}) or {}
    handeye_verification = verification.get("handeye", {}) or {}
    board_reprojection = verification.get("board_reprojection", {}) or {}
    pose_repeatability = verification.get("pose_repeatability", {}) or {}
    mesh_alignment = verification.get("mesh_alignment", {}) or {}
    hybrid_cam3_ok = (
        hybrid.get("applied") is True and
        cross_camera.get("mean_mm") is not None and
        cross_camera.get("mean_mm") <= 8.0 and
        reprojection.get("pass") is True and
        mesh_alignment.get("pass") is True and
        pose_repeatability.get("mean_dt_mm") is not None and
        pose_repeatability.get("mean_dt_mm") <= 8.0 and
        pose_repeatability.get("mean_dr_deg") is not None and
        pose_repeatability.get("mean_dr_deg") <= 3.0
    )
    provisional_cam3_ok = (
        cross_camera.get("mean_mm") is not None and
        cross_camera.get("mean_mm") <= 8.0 and
        reprojection.get("pass") is True and
        mesh_alignment.get("pass") is True and
        pose_repeatability.get("mean_dt_mm") is not None and
        pose_repeatability.get("mean_dt_mm") <= 8.0 and
        pose_repeatability.get("mean_dr_deg") is not None and
        pose_repeatability.get("mean_dr_deg") <= 3.0
    )

    if name == "T_gripper_cam":
        ok = (
            selected_handeye.get("mean_trans_mm") is not None and
            selected_handeye.get("mean_rot_deg") is not None and
            selected_handeye.get("mean_trans_mm") <= 12.0 and
            selected_handeye.get("mean_rot_deg") <= 1.5 and
            handeye_verification.get("pass") is True and
            board_reprojection.get("pass") is True
        )
        return ("PASS" if ok else "FAIL",
                f"hand-eye {selected_method}: {selected_handeye.get('mean_trans_mm', float('nan')):.2f}mm / "
                f"{selected_handeye.get('mean_rot_deg', float('nan')):.3f}deg")

    if name in ("T_base_C0", "T_base_C1"):
        st = base_stats.get(name, {})
        ok = (
            st.get("translation_std_mm") is not None and
            st.get("rotation_std_deg") is not None and
            st.get("translation_std_mm") <= 5.0 and
            st.get("rotation_std_deg") <= 0.5 and
            st.get("method", "board") != "cube_anchor" and
            int(st.get("num_inliers", 0)) >= 10 and
            board_reprojection.get("pass") is True
        )
        return ("PASS" if ok else "FAIL",
                f"{st.get('method', 'board-based')}: {st.get('translation_std_mm', float('nan')):.2f}mm / "
                f"{st.get('rotation_std_deg', float('nan')):.3f}deg")

    if name == "T_C0_C1":
        s0, _ = evaluate_export_status("T_base_C0", summary, verification)
        s1, _ = evaluate_export_status("T_base_C1", summary, verification)
        ok = (s0 == "PASS" and s1 == "PASS")
        return ("PASS" if ok else "FAIL", "derived from PASS T_base_C0 and T_base_C1")

    if name == "T_base_C3":
        st = base_stats.get(name, {})
        support = int(st.get("support", 0))
        dom = st.get("dominant_signature") or {}
        method = st.get("method")
        ok = False
        if method == "cube_anchor_strict":
            ok = (
                st.get("translation_std_mm") is not None and
                st.get("rotation_std_deg") is not None and
                st.get("translation_std_mm") < 1.0 and
                st.get("rotation_std_deg") < 0.5 and
                support >= 4 and
                int(dom.get("support", 0)) >= 4
            )
        else:
            ok = (
                st.get("translation_std_mm") is not None and
                st.get("rotation_std_deg") is not None and
                st.get("translation_std_mm") < 3.0 and
                st.get("rotation_std_deg") < 1.0 and
                support >= 6 and
                method != "cube_anchor"
            )
        if not ok and hybrid_cam3_ok:
            ok = (
                st.get("translation_std_mm") is not None and
                st.get("rotation_std_deg") is not None and
                st.get("translation_std_mm") <= 8.0 and
                st.get("rotation_std_deg") <= 1.0 and
                support >= 12
            )
        if not ok and provisional_cam3_ok:
            ok = (
                st.get("translation_std_mm") is not None and
                st.get("rotation_std_deg") is not None and
                st.get("translation_std_mm") <= 8.0 and
                st.get("rotation_std_deg") <= 1.0 and
                support >= 12 and
                method not in (None, "cube_anchor")
            )
        return ("PASS" if ok else "FAIL",
                f"{method or 'unknown'} support={support}/{st.get('total_keys', 0)} "
                f"signature={dom.get('used_ids', [])}/{dom.get('source', 'n/a')}"
                f"{' + hybrid-refined provisional' if ok and hybrid_cam3_ok else ''}"
                f"{' + verified provisional' if ok and (not hybrid_cam3_ok) and provisional_cam3_ok else ''}")

    if name == "T_base_O":
        if cube_anchor_by_set.get("num_sets", 0) > 1:
            return (
                "FAIL",
                f"compatibility average across {cube_anchor_by_set.get('num_sets', 0)} set_index groups; "
                "use T_base_O_set*",
            )
        support = int(cube_anchor.get("support", 0))
        st = cube_anchor.get("stability", {})
        ok = (
            support >= 12 and
            st.get("translation_std_mm", 1e9) < 3.0 and
            st.get("rotation_std_deg", 1e9) < 0.5 and
            cross_camera.get("pass") is True and
            reprojection.get("pass") is True
        )
        return ("PASS" if ok else "FAIL",
                f"cube anchor support={support}/{cube_anchor.get('total_keys', 0)}")

    if name.startswith("T_base_O_set"):
        set_index = name.replace("T_base_O_set", "")
        set_diag = cube_anchor_per_set.get(str(set_index), {})
        st = set_diag.get("stability", {})
        support = int(set_diag.get("support", 0))
        ok = (
            support >= 2 and
            st.get("translation_std_mm", 1e9) < 3.0 and
            st.get("rotation_std_deg", 1e9) < 0.5 and
            reprojection.get("pass") is True
        )
        return (
            "PASS" if ok else "FAIL",
            f"set_index={set_index} support={support} "
            f"stability={st.get('translation_std_mm', float('nan')):.2f}mm / "
            f"{st.get('rotation_std_deg', float('nan')):.3f}deg",
        )

    if name == "T_C0_C3":
        s0, _ = evaluate_export_status("T_base_C0", summary, verification)
        s3, _ = evaluate_export_status("T_base_C3", summary, verification)
        ok = (s0 == "PASS" and s3 == "PASS")
        return ("PASS" if ok else "FAIL", "derived from T_base_C0 and T_base_C3")

    return ("FAIL", "unsupported")


def export_quality_tier(name: str, summary: dict) -> str:
    base_stats = summary.get("diagnostics", {}).get("base_transforms", {})
    hybrid = summary.get("diagnostics", {}).get("hybrid_refinement", {}) or {}
    if name in ("T_gripper_cam", "T_base_C0", "T_base_C1", "T_C0_C1"):
        return "production"
    if name.startswith("T_base_O_set"):
        return "diagnostic"
    method_c3 = str(base_stats.get("T_base_C3", {}).get("method", ""))
    c3_is_provisional = (
        "cube_anchor" in method_c3 or
        hybrid.get("applied") is True
    )
    if name == "T_base_C3" and c3_is_provisional:
        return "provisional"
    if name == "T_C0_C3" and c3_is_provisional:
        return "provisional"
    return "diagnostic"


def build_camera_root_cause(ci: int, row_list: List[dict], summary: dict) -> str:
    base_stats = summary.get("diagnostics", {}).get("base_transforms", {})
    key = f"T_base_C{ci}"
    st = base_stats.get(key, {})
    selected_counter = Counter(tuple(r["used_ids"]) for r in row_list)
    accepted_counter = Counter(tuple(r["used_ids"]) for r in row_list if r["accepted"])
    selected_top = selected_counter.most_common(1)
    accepted_top = accepted_counter.most_common(1)

    if st.get("method") == "cube_anchor":
        return (
            f"cam{ci} extrinsic itself is not board-verified. It depends on cube-anchor only, "
            f"support {st.get('support', 0)}/{st.get('total_keys', 0)}, "
            f"and accepted cases come from {format_used_ids(accepted_counter, 1) or 'none'}."
        )

    if ci == summary.get("gripper_cam_idx"):
        return (
            f"hand-eye is stable, but cube candidates disagree across markers. "
            f"Selected poses are dominated by {format_used_ids(selected_counter, 2) or 'none'}, "
            f"while accepted poses are only {format_used_ids(accepted_counter, 2) or 'none'}."
        )

    if selected_top and accepted_top and selected_top[0][0] != accepted_top[0][0]:
        return (
            f"camera extrinsic is stable, but the dominant selected cube marker "
            f"({format_used_ids(selected_counter, 1)}) is not the one that survives global checks "
            f"({format_used_ids(accepted_counter, 1)}). This points to marker-to-marker cube model inconsistency."
        )

    if not accepted_counter:
        return (
            f"camera extrinsic is stable, but no selected cube candidate survived the object/camera thresholds. "
            f"Dominant selected markers: {format_used_ids(selected_counter, 2) or 'none'}."
        )

    return (
        f"camera extrinsic is stable. Remaining failure comes from cube-model inconsistency, "
        f"not from the board-based camera calibration itself."
    )


def build_marker_root_cause(marker_id: int, diag_row: dict, accepted_count: int) -> str:
    current = diag_row.get("current", {})
    best = diag_row.get("best", {})
    parts = []

    if current.get("mean_reproj_px", 999.0) < 0.5 and current.get("mean_dt_mm", 0.0) > 150.0:
        parts.append("2D corner fit is good, but the 3D pose is inconsistent with the global cube model")
    if current.get("rank", 999) > 20:
        parts.append(
            f"current face/order ranks poorly ({current.get('rank')}) against alternatives such as "
            f"{best.get('face', 'N/A')}/{best.get('corner_permutation', 'N/A')}"
        )
    elif current.get("rank", 999) <= 3:
        parts.append("current mapping is near-best for this session")
    if current.get("num_inliers", 0) == 0:
        parts.append("it never reaches the global inlier threshold")
    elif current.get("num_inliers", 0) <= 4:
        parts.append(f"only {current.get('num_inliers')} observations reach global consensus")
    if accepted_count == 0:
        parts.append("no selected single-marker candidate was accepted")
    if not parts:
        parts.append("marker is not the main blocker")
    return "; ".join(parts)


def build_result_tables(summary: dict, verification: dict) -> Tuple[List[dict], List[dict]]:
    transforms = {
        name: np.asarray(vals, dtype=np.float64).reshape(4, 4)
        for name, vals in summary.get("transforms", {}).items()
    }
    handeye = summary.get("diagnostics", {}).get("handeye_methods", {})
    selected_method = summary.get("selected_handeye_method")
    selected_handeye = handeye.get(selected_method, {})
    base_stats = summary.get("diagnostics", {}).get("base_transforms", {})
    cube_anchor = summary.get("diagnostics", {}).get("cube_anchor") or {}
    cube_anchor_by_set = summary.get("diagnostics", {}).get("cube_anchor_by_set") or {}
    cube_anchor_per_set = cube_anchor_by_set.get("per_set", {}) or {}

    camera_rows = []
    entries = [
        ("cam2", "gripper", "T_gripper_cam", f"hand-eye:{selected_method}",
         selected_handeye.get("mean_trans_mm"), selected_handeye.get("mean_rot_deg"),
         selected_handeye.get("stability", {}).get("num_inliers"), summary.get("num_handeye_events"),
         f"gripper->camera extrinsic, {summary.get('num_charuco_frames')} ChArUco frames"),
        ("cam0", "fixed", "T_base_C0", "board-based",
         base_stats.get("T_base_C0", {}).get("translation_std_mm"),
         base_stats.get("T_base_C0", {}).get("rotation_std_deg"),
         base_stats.get("T_base_C0", {}).get("num_inliers"),
         base_stats.get("T_base_C0", {}).get("num_frames"),
         "fixed camera extrinsic from ChArUco board"),
        ("cam1", "fixed", "T_base_C1", "board-based",
         base_stats.get("T_base_C1", {}).get("translation_std_mm"),
         base_stats.get("T_base_C1", {}).get("rotation_std_deg"),
         base_stats.get("T_base_C1", {}).get("num_inliers"),
         base_stats.get("T_base_C1", {}).get("num_frames"),
         "fixed camera extrinsic from ChArUco board"),
        ("cam3", "fixed", "T_base_C3", base_stats.get("T_base_C3", {}).get("method", "cube-anchor"),
         base_stats.get("T_base_C3", {}).get("translation_std_mm"),
         base_stats.get("T_base_C3", {}).get("rotation_std_deg"),
         base_stats.get("T_base_C3", {}).get("num_inliers"),
         base_stats.get("T_base_C3", {}).get("num_frames"),
         f"fallback from cube anchor support={base_stats.get('T_base_C3', {}).get('support', 0)}/"
         f"{base_stats.get('T_base_C3', {}).get('total_keys', 0)}"),
        ("cube", "object", "T_base_O", summary.get("t_base_o_source", "unknown"),
         cube_anchor.get("stability", {}).get("translation_std_mm"),
         cube_anchor.get("stability", {}).get("rotation_std_deg"),
         cube_anchor.get("support"),
         cube_anchor.get("total_keys"),
         ("compatibility average across set anchors"
          if cube_anchor_by_set.get("num_sets", 0) > 1 else
          "static cube anchor in robot base")),
    ]

    for set_index in sorted(int(k) for k in cube_anchor_per_set.keys()):
        set_diag = cube_anchor_per_set.get(str(set_index), {})
        st = set_diag.get("stability", {})
        entries.append((
            f"cube_set_{set_index}",
            "object",
            f"T_base_O_set{set_index}",
            summary.get("t_base_o_source", "unknown"),
            st.get("translation_std_mm"),
            st.get("rotation_std_deg"),
            set_diag.get("support"),
            set_diag.get("support"),
            f"set_index={set_index} static cube anchor",
        ))

    for entity, role, name, method, trans_err, rot_err, num_inliers, num_total, note in entries:
        if name not in transforms:
            continue
        T = transforms[name]
        tx, ty, tz = (T[:3, 3] * 1000.0).tolist()
        rz, ry, rx = matrix_to_rzryrx_deg(T)
        status, reason = evaluate_export_status(name, summary, verification)
        camera_rows.append({
            "entity": entity,
            "role": role,
            "transform": name,
            "solve_method": method,
            "status": status,
            "x_mm": f"{tx:.3f}",
            "y_mm": f"{ty:.3f}",
            "z_mm": f"{tz:.3f}",
            "rz_deg": f"{rz:.3f}",
            "ry_deg": f"{ry:.3f}",
            "rx_deg": f"{rx:.3f}",
            "error_trans_mm": "" if trans_err is None else f"{float(trans_err):.3f}",
            "error_rot_deg": "" if rot_err is None else f"{float(rot_err):.3f}",
            "support_inliers": "" if num_inliers is None else str(int(num_inliers)),
            "support_total": "" if num_total is None else str(int(num_total)),
            "status_reason": reason,
            "notes": note,
        })

    relative_rows = []
    for name, note in [
        ("T_C0_C1", "derived from T_base_C0 and T_base_C1"),
        ("T_C0_C3", "derived from T_base_C0 and T_base_C3"),
    ]:
        if name not in transforms:
            continue
        T = transforms[name]
        tx, ty, tz = (T[:3, 3] * 1000.0).tolist()
        rz, ry, rx = matrix_to_rzryrx_deg(T)
        status, reason = evaluate_export_status(name, summary, verification)
        relative_rows.append({
            "transform": name,
            "status": status,
            "x_mm": f"{tx:.3f}",
            "y_mm": f"{ty:.3f}",
            "z_mm": f"{tz:.3f}",
            "rz_deg": f"{rz:.3f}",
            "ry_deg": f"{ry:.3f}",
            "rx_deg": f"{rx:.3f}",
            "status_reason": reason,
            "notes": note,
        })
    return camera_rows, relative_rows


def render_markdown_table(rows: List[dict], columns: List[str]) -> str:
    if not rows:
        return "_No rows_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep] + body) + "\n"


def save_calibration_report(path: str, summary: dict, verification: dict,
                            camera_rows: List[dict], relative_rows: List[dict]) -> None:
    lines = []
    lines.append("# Calibration Result Table")
    lines.append("")
    lines.append("## Verification Summary")
    lines.append("")
    lines.append(
        f"- Cross-camera: {bool_pass_fail(verification['cross_camera']['pass'])} "
        f"(mean {verification['cross_camera']['mean_mm']:.2f} mm, "
        f"median {verification['cross_camera']['median_mm']:.2f} mm, "
        f"max {verification['cross_camera']['max_mm']:.2f} mm)"
        if verification["cross_camera"]["mean_mm"] is not None else
        "- Cross-camera: N/A"
    )
    lines.append(
        f"- Cube reprojection: {bool_pass_fail(verification['reprojection']['pass'])} "
        f"(mean {verification['reprojection']['mean_px']:.3f} px, "
        f"median {verification['reprojection']['median_px']:.3f} px, "
        f"max {verification['reprojection']['max_px']:.3f} px)"
        if verification["reprojection"]["mean_px"] is not None else
        "- Cube reprojection: N/A"
    )
    lines.append(
        f"- Hand-eye board stability: {bool_pass_fail(verification['handeye']['pass'])} "
        f"(pos std {verification['handeye']['board_position_std_mm']:.2f} mm, "
        f"rot mean {verification['handeye']['board_rotation_mean_deg']:.3f} deg)"
        if verification["handeye"]["board_position_std_mm"] is not None else
        "- Hand-eye board stability: N/A"
    )
    lines.append(
        f"- Board reprojection: {bool_pass_fail(verification['board_reprojection']['pass'])} "
        f"(mean {verification['board_reprojection']['mean_px']:.3f} px)"
        if verification["board_reprojection"]["mean_px"] is not None else
        "- Board reprojection: N/A"
    )
    lines.append(
        f"- Mesh alignment: {bool_pass_fail(verification['mesh_alignment']['pass'])} "
        f"(mean RMSE {verification['mesh_alignment']['mean_rmse_mm']:.2f} mm)"
        if verification["mesh_alignment"]["mean_rmse_mm"] is not None else
        "- Mesh alignment: N/A"
    )
    lines.append(
        f"- Dimension accuracy: {bool_pass_fail(verification['dimension_accuracy']['pass'])} "
        f"(mean abs err {verification['dimension_accuracy']['mean_abs_err_mm']:.2f} mm)"
        if verification["dimension_accuracy"]["mean_abs_err_mm"] is not None else
        "- Dimension accuracy: N/A"
    )
    lines.append(
        f"- Pose repeatability: {bool_pass_fail(verification['pose_repeatability']['pass'])} "
        f"(mean {verification['pose_repeatability']['mean_dt_mm']:.2f} mm / "
        f"{verification['pose_repeatability']['mean_dr_deg']:.3f} deg)"
        if verification["pose_repeatability"]["mean_dt_mm"] is not None else
        "- Pose repeatability: N/A"
    )
    lines.append("")
    lines.append("## Camera / Object Transforms")
    lines.append("")
    lines.append(render_markdown_table(camera_rows, [
        "entity", "role", "transform", "solve_method", "status",
        "x_mm", "y_mm", "z_mm", "rz_deg", "ry_deg", "rx_deg",
        "error_trans_mm", "error_rot_deg", "support_inliers", "support_total", "notes",
    ]))
    lines.append("## Relative Transforms")
    lines.append("")
    lines.append(render_markdown_table(relative_rows, [
        "transform", "status", "x_mm", "y_mm", "z_mm", "rz_deg", "ry_deg", "rx_deg", "notes",
    ]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_failure_report(path: str, verification: dict,
                        camera_failure_rows: List[dict], marker_failure_rows: List[dict]) -> None:
    lines = []
    lines.append("# Cube Failure Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Cross-camera mean error: {verification['cross_camera']['mean_mm']:.2f} mm"
        if verification["cross_camera"]["mean_mm"] is not None else
        "- Cross-camera mean error: N/A"
    )
    lines.append(
        f"- Cube reprojection mean error: {verification['reprojection']['mean_px']:.3f} px"
        if verification["reprojection"]["mean_px"] is not None else
        "- Cube reprojection mean error: N/A"
    )
    lines.append(
        f"- Candidate diagnostics: {verification['candidate_summary']['selected']} selected / "
        f"{verification['candidate_summary']['accepted']} accepted"
    )
    lines.append("")
    lines.append("## Camera-Level Causes")
    lines.append("")
    lines.append(render_markdown_table(camera_failure_rows, [
        "camera", "selected_candidates", "accepted_candidates", "accept_rate",
        "mean_obj_dt_mm", "mean_obj_dr_deg", "mean_cam_dt_mm",
        "dominant_selected_markers", "dominant_accepted_markers", "root_cause",
    ]))
    lines.append("## Marker-Level Causes")
    lines.append("")
    lines.append(render_markdown_table(marker_failure_rows, [
        "marker_id", "num_observations", "current_face", "current_perm", "best_face", "best_perm",
        "current_rank", "num_inliers", "selected_single", "accepted_single",
        "mean_dt_mm", "mean_dr_deg", "mean_reproj_px", "seen_in_cameras", "root_cause",
    ]))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--cube_config_json", default=None,
                        help="Optional cube config JSON override. Leave unset to use the project's canonical cube definition.")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    root_folder = args.root_folder
    calib_dir = args.calib_dir
    out_dir = ensure_dir(args.out_dir or calib_dir)
    export_dir = ensure_dir(os.path.join(out_dir, "export"))

    with open(os.path.join(root_folder, "meta.json"), "r") as f:
        meta = json.load(f)
    with open(os.path.join(calib_dir, "calibration_summary.json"), "r") as f:
        summary = json.load(f)

    transforms = load_calib(calib_dir)
    all_cam_ids = sorted({
        int(k) for cap in meta.get("captures", [])
        for k in cap.get("cams", {}).keys()
    })
    gripper_cam_idx = summary.get("gripper_cam_idx")

    cube_cfg, cube_cfg_source = resolve_cube_config_for_run(
        root_folder, calib_dir=calib_dir, cube_config_json=args.cube_config_json, default_cfg=get_default_cube_config())
    include_meta = False

    verification = compute_full_verification_bundle(
        meta, transforms, args.intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta)

    candidate_rows = collect_cube_candidate_diagnostics(
        meta, transforms, args.intrinsics_dir, root_folder, gripper_cam_idx, all_cam_ids,
        cube_cfg=cube_cfg, include_meta=include_meta)
    selected_rows = [row for row in candidate_rows if row["selected"]]
    accepted_rows = [row for row in selected_rows if row["accepted"]]
    verification["candidate_summary"] = {
        "total": int(len(candidate_rows)),
        "selected": int(len(selected_rows)),
        "accepted": int(len(accepted_rows)),
    }

    override_path = os.path.join(calib_dir, "verify", "cube_override_diagnostic.json")
    if os.path.exists(override_path):
        with open(override_path, "r") as f:
            override_report = json.load(f)
    else:
        override_report = {}

    camera_rows, relative_rows = build_result_tables(summary, verification)
    mode_comparison_rows, mode_verification = build_mode_comparison_rows(
        summary, meta, args.intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta)

    camera_failure_rows = []
    by_cam = defaultdict(list)
    for row in selected_rows:
        by_cam[row["cam_idx"]].append(row)
    for ci in sorted(by_cam):
        rows_ci = by_cam[ci]
        acc_ci = [r for r in rows_ci if r["accepted"]]
        cam_dt_vals = [float(r["cam_dt_mm"]) for r in rows_ci if r["cam_dt_mm"] is not None]
        selected_counter = Counter(tuple(r["used_ids"]) for r in rows_ci)
        accepted_counter = Counter(tuple(r["used_ids"]) for r in acc_ci)
        camera_failure_rows.append({
            "camera": f"cam{ci}",
            "selected_candidates": str(len(rows_ci)),
            "accepted_candidates": str(len(acc_ci)),
            "accept_rate": f"{(len(acc_ci) / max(len(rows_ci), 1)):.3f}",
            "mean_obj_dt_mm": f"{np.mean([r['obj_dt_mm'] for r in rows_ci]):.2f}",
            "mean_obj_dr_deg": f"{np.mean([r['obj_dr_deg'] for r in rows_ci]):.2f}",
            "mean_cam_dt_mm": "" if not cam_dt_vals else f"{np.mean(cam_dt_vals):.2f}",
            "dominant_selected_markers": format_used_ids(selected_counter, 3),
            "dominant_accepted_markers": format_used_ids(accepted_counter, 3),
            "root_cause": build_camera_root_cause(ci, rows_ci, summary),
        })

    single_rows = [r for r in selected_rows if len(r["used_ids"]) == 1]
    per_marker_selected = defaultdict(list)
    for row in single_rows:
        per_marker_selected[row["used_ids"][0]].append(row)

    marker_failure_rows = []
    for mid_str, diag_row in sorted(override_report.items(), key=lambda kv: int(kv[0])):
        mid = int(mid_str)
        selected_marker_rows = per_marker_selected.get(mid, [])
        accepted_count = sum(1 for r in selected_marker_rows if r["accepted"])
        seen_cams = Counter(r["cam_idx"] for r in selected_marker_rows)
        current = diag_row.get("current", {})
        best = diag_row.get("best", {})
        marker_failure_rows.append({
            "marker_id": str(mid),
            "num_observations": str(diag_row.get("num_observations", 0)),
            "current_face": current.get("face", ""),
            "current_perm": current.get("corner_permutation", ""),
            "best_face": best.get("face", ""),
            "best_perm": best.get("corner_permutation", ""),
            "current_rank": str(current.get("rank", "")),
            "num_inliers": str(current.get("num_inliers", "")),
            "selected_single": str(len(selected_marker_rows)),
            "accepted_single": str(accepted_count),
            "mean_dt_mm": "" if current.get("mean_dt_mm") is None else f"{float(current['mean_dt_mm']):.2f}",
            "mean_dr_deg": "" if current.get("mean_dr_deg") is None else f"{float(current['mean_dr_deg']):.2f}",
            "mean_reproj_px": "" if current.get("mean_reproj_px") is None else f"{float(current['mean_reproj_px']):.3f}",
            "seen_in_cameras": ", ".join(f"cam{ci} x{count}" for ci, count in seen_cams.most_common()),
            "root_cause": build_marker_root_cause(mid, diag_row, accepted_count),
        })

    md_table_path = os.path.join(out_dir, "calibration_result_table.md")
    camera_csv_path = os.path.join(out_dir, "camera_calibration_table.csv")
    relative_csv_path = os.path.join(out_dir, "relative_transform_table.csv")
    verification_path = os.path.join(out_dir, "verification_metrics.json")
    failure_md_path = os.path.join(out_dir, "cube_failure_report.md")
    camera_failure_csv = os.path.join(out_dir, "cube_failure_camera_table.csv")
    marker_failure_csv = os.path.join(out_dir, "cube_failure_marker_table.csv")
    mode_compare_md = os.path.join(out_dir, "calibration_mode_comparison.md")
    mode_compare_csv = os.path.join(out_dir, "calibration_mode_comparison.csv")
    mode_compare_json = os.path.join(out_dir, "calibration_mode_comparison.json")

    save_calibration_report(md_table_path, summary, verification, camera_rows, relative_rows)
    write_csv(camera_csv_path, camera_rows, list(camera_rows[0].keys()) if camera_rows else [])
    write_csv(relative_csv_path, relative_rows, list(relative_rows[0].keys()) if relative_rows else [])
    save_failure_report(failure_md_path, verification, camera_failure_rows, marker_failure_rows)
    write_csv(camera_failure_csv, camera_failure_rows, list(camera_failure_rows[0].keys()) if camera_failure_rows else [])
    write_csv(marker_failure_csv, marker_failure_rows, list(marker_failure_rows[0].keys()) if marker_failure_rows else [])
    save_mode_comparison_report(mode_compare_md, mode_comparison_rows)
    write_csv(mode_compare_csv, mode_comparison_rows, list(mode_comparison_rows[0].keys()) if mode_comparison_rows else [])
    with open(mode_compare_json, "w") as f:
        json.dump(mode_verification, f, indent=2)

    usable = {}
    excluded = {}
    for name, values in summary.get("transforms", {}).items():
        T = np.asarray(values, dtype=np.float64).reshape(4, 4)
        status, reason = evaluate_export_status(name, summary, verification)
        payload = {
            "matrix_4x4": matrix_to_nested_list(T),
            "translation_mm": [float(x) for x in (T[:3, 3] * 1000.0)],
            "rotation_rz_ry_rx_deg": [float(x) for x in matrix_to_rzryrx_deg(T)],
            "quality_tier": export_quality_tier(name, summary),
            "status_reason": reason,
        }
        if status == "PASS":
            usable[name] = payload
        else:
            excluded[name] = payload

    export_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root_folder": os.path.abspath(root_folder),
        "source_calib_dir": os.path.abspath(calib_dir),
        "cube_config_source": cube_cfg_source,
        "cube_config_used": cube_config_to_dict(cube_cfg),
        "verification": verification,
        "usable_transforms": usable,
        "excluded_transforms": excluded,
    }
    export_json_path = os.path.join(export_dir, "usable_transforms.json")
    export_npz_path = os.path.join(export_dir, "usable_transforms.npz")
    with open(export_json_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    if usable:
        np.savez(export_npz_path, **{k: np.asarray(v["matrix_4x4"], dtype=np.float64) for k, v in usable.items()})

    with open(verification_path, "w") as f:
        json.dump(verification, f, indent=2)

    final_use_paths = write_final_use_bundle(
        export_dir,
        summary,
        verification,
        cube_cfg,
        cube_cfg_source,
        usable,
        excluded,
    )

    print(f"[SAVE] {md_table_path}")
    print(f"[SAVE] {camera_csv_path}")
    print(f"[SAVE] {relative_csv_path}")
    print(f"[SAVE] {verification_path}")
    print(f"[SAVE] {failure_md_path}")
    print(f"[SAVE] {camera_failure_csv}")
    print(f"[SAVE] {marker_failure_csv}")
    print(f"[SAVE] {mode_compare_md}")
    if mode_comparison_rows:
        print(f"[SAVE] {mode_compare_csv}")
    print(f"[SAVE] {mode_compare_json}")
    print(f"[SAVE] {export_json_path}")
    if usable:
        print(f"[SAVE] {export_npz_path}")
    for path in final_use_paths:
        print(f"[SAVE] {path}")


if __name__ == "__main__":
    main()
