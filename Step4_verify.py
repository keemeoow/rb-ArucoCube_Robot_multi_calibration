# Step4_verify.py
"""
Step 4: 캘리브레이션 검증 및 시각화.

검증 항목:
  1. Cross-camera consistency: 같은 event에서 camera별 cube pose가 base에서 얼마나 일치하는지
  2. Reprojection error: 현재 cube model/pose를 이미지로 다시 투영했을 때의 오차
  3. Hand-eye consistency: gripper board anchor가 이벤트마다 얼마나 안정적인지
  4. Depth metrics: depth cloud와 cube surface 정합, dimension accuracy
  5. 3D visualization: base 기준 camera/object 배치

기본 검증 정책:
  - Step3 기본 파이프라인과 같은 stable candidate path를 그대로 사용한다.
  - depth는 mesh/dimension 검증과 candidate 품질 평가에 반영하되,
    실험용 depth-SVD 후보는 명시적으로 켠 경우에만 사용한다.

실행:
  python Step4_verify.py \
    --root_folder ./data/session \
    --calib_dir ./data/session/calib_out \
    --intrinsics_dir ./intrinsics

3D 창 옵션 추가 CLI 인자:
--hide_gripper_trajectory
--camera_label_size
--object_label_size
--view_elev
--view_azim
"""

import os
import json
import argparse
import itertools
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
if "--no_show" in sys.argv or not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from aruco_cube import ArucoCubeTarget, ArucoCubeModel, rodrigues_to_Rt, inv_T
from calibration_runtime_utils import (
    build_event_cube_selection,
    build_capture_cube_candidate_map,
    build_cube_pose_candidates,
    cube_selection_profile_kwargs,
    get_capture_object_anchor,
    get_event_base_camera_transform,
    load_calib_dir,
    load_intrinsics_color,
    load_intrinsics_with_depth_scale,
    load_robot_pose_from_capture,
    resolve_cube_config_for_run,
    select_consistent_event_cube_candidates,
    select_primary_cube_candidate,
)
from config import CubeConfig, get_default_cube_config
from downstream_metrics import (
    compute_board_reprojection_metrics,
    compute_depth_cube_metrics,
    compute_pose_repeatability_metrics,
)
DIAG_FACES = ("+Z", "-Z", "+X", "-X", "+Y", "-Y")


def build_named_corner_permutations():
    named = {
        "r0": [0, 1, 2, 3],
        "r90": [1, 2, 3, 0],
        "r180": [2, 3, 0, 1],
        "r270": [3, 0, 1, 2],
        "flip_x": [1, 0, 3, 2],
        "flip_y": [3, 2, 1, 0],
        "flip_diag": [0, 3, 2, 1],
        "flip_anti": [2, 1, 0, 3],
    }
    for perm in itertools.permutations(range(4)):
        key = f"p{perm[0]}{perm[1]}{perm[2]}{perm[3]}"
        named.setdefault(key, list(perm))
    return named


DIAG_CORNER_PERMUTATIONS = build_named_corner_permutations()


def rotation_error_deg(Ra, Rb):
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


load_calib = load_calib_dir


def iter_object_anchor_items(transforms: Dict[str, np.ndarray]):
    set_keys = sorted(
        [k for k in transforms.keys() if k.startswith("T_base_O_set")],
        key=lambda k: int(k.replace("T_base_O_set", "")),
    )
    if set_keys:
        for key in set_keys:
            yield key, np.asarray(transforms[key], dtype=np.float64)
        return
    T_base_O = transforms.get("T_base_O")
    if T_base_O is not None:
        yield "T_base_O", np.asarray(T_base_O, dtype=np.float64)


def draw_frame(ax, T, label="", scale=30.0, lw=1.5, fontsize: float = 7.0):
    """Draw a coordinate frame (RGB = XYZ) at transform T."""
    o = T[:3, 3] * 1000.0  # m -> mm
    R = T[:3, :3]
    colors = ['r', 'g', 'b']
    for i, c in enumerate(colors):
        d = R[:, i] * scale
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                  color=c, linewidth=lw, arrow_length_ratio=0.15)
    if label:
        ax.text(o[0], o[1], o[2], f"  {label}", fontsize=fontsize)


def draw_camera(ax, T, label="", scale=20.0, color='blue', fontsize: float = 7.0):
    """Draw camera as a pyramid frustum."""
    o = T[:3, 3] * 1000.0
    R = T[:3, :3]

    # Camera frustum corners (in camera frame, pointing +Z)
    s = scale
    corners_cam = np.array([
        [-s, -s*0.75, s*1.5],
        [ s, -s*0.75, s*1.5],
        [ s,  s*0.75, s*1.5],
        [-s,  s*0.75, s*1.5],
    ], dtype=np.float64)

    corners_world = (R @ corners_cam.T).T + o
    # Draw frustum lines
    for c in corners_world:
        ax.plot3D([o[0], c[0]], [o[1], c[1]], [o[2], c[2]],
                  color=color, linewidth=0.8, alpha=0.6)
    # Draw rectangle
    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D([corners_world[i, 0], corners_world[j, 0]],
                  [corners_world[i, 1], corners_world[j, 1]],
                  [corners_world[i, 2], corners_world[j, 2]],
                  color=color, linewidth=0.8, alpha=0.6)
    if label:
        ax.text(o[0], o[1], o[2], f"  {label}", fontsize=fontsize, color=color)


def iter_public_transform_items(transforms: Dict[str, np.ndarray]):
    for name in sorted(transforms.keys()):
        if "_event" in name:
            continue
        if name.startswith("T_base_O_set"):
            continue
        T = transforms.get(name)
        if isinstance(T, np.ndarray):
            yield name, np.asarray(T, dtype=np.float64)


def _resize_pad_bgr(img: np.ndarray, width: int, height: int, bg=(245, 245, 245)) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((height, width, 3), bg, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(float(width) / max(w, 1), float(height) / max(h, 1))
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _put_text_lines(img: np.ndarray, lines: List[str], x: int = 8, y: int = 22,
                    line_h: int = 22, color=(20, 20, 20), scale: float = 0.55) -> np.ndarray:
    out = img.copy()
    for idx, line in enumerate(lines):
        cv2.putText(out, line, (x, y + idx * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    return out


def build_selected_event_contact_sheets(meta: dict,
                                        transforms: Dict[str, np.ndarray],
                                        intrinsics_dir: str,
                                        root_folder: str,
                                        all_cam_ids: List[int],
                                        gripper_cam_idx: Optional[int],
                                        cube_cfg: CubeConfig,
                                        include_meta: bool = False,
                                        selection_profile: str = "default",
                                        tile_w: int = 420,
                                        tile_h: int = 300,
                                        cols: int = 3,
                                        rows: int = 4):
    selection = build_event_cube_selection(
        meta, transforms, intrinsics_dir, root_folder, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta, selection_profile=selection_profile)

    manifest = []
    tiles = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        refined = selection.get(eid, {})
        if not refined:
            continue
        set_index = cap.get("set_index")
        for ci, cand in sorted(refined.items()):
            cinfo = cap.get("cams", {}).get(str(ci), {})
            rgb_rel = cinfo.get("rgb_path", "")
            if not rgb_rel:
                continue
            img = cv2.imread(os.path.join(root_folder, rgb_rel))
            if img is None:
                continue
            tile = _resize_pad_bgr(img, tile_w, tile_h)
            lines = [
                f"event {eid} | set {set_index} | cam{ci}",
                f"used_ids={cand.get('used_ids', [])}  src={cand.get('source', 'unknown')}",
                f"err={float(cand.get('err_mean', 99.0)):.3f}px",
            ]
            tile = _put_text_lines(tile, lines, color=(10, 10, 10))
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 2)
            tiles.append(tile)
            manifest.append({
                "event_id": int(eid),
                "set_index": None if set_index is None else int(set_index),
                "cam_idx": int(ci),
                "rgb_path": rgb_rel,
                "used_ids": [int(x) for x in cand.get("used_ids", [])],
                "source": str(cand.get("source", "unknown")),
                "err_mean_px": float(cand.get("err_mean", 99.0)),
            })

    if not tiles:
        return [], manifest

    pages = []
    page_cap = max(int(cols * rows), 1)
    for page_idx in range(0, len(tiles), page_cap):
        chunk = tiles[page_idx:page_idx + page_cap]
        canvas = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
        for idx, tile in enumerate(chunk):
            rr = idx // cols
            cc = idx % cols
            y0 = rr * tile_h
            x0 = cc * tile_w
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
        title = f"Selected event images {page_idx // page_cap + 1}"
        cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 180), 2, cv2.LINE_AA)
        pages.append(canvas)
    return pages, manifest


def _fit_world_to_rect(points: List[Tuple[float, float]], rect: Tuple[int, int, int, int]):
    x0, y0, w, h = rect
    if not points:
        return lambda px, py: (x0 + w // 2, y0 + h // 2)
    xs = np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    span_x = max(max_x - min_x, 50.0)
    span_y = max(max_y - min_y, 50.0)
    margin_x = 0.12 * span_x
    margin_y = 0.12 * span_y
    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    def project(px: float, py: float):
        u = x0 + int(round((px - min_x) / max(max_x - min_x, 1e-9) * w))
        v = y0 + h - int(round((py - min_y) / max(max_y - min_y, 1e-9) * h))
        return u, v

    return project


def build_base_frame_overview_cv(meta: dict,
                                 transforms: Dict[str, np.ndarray],
                                 gripper_cam_idx: Optional[int],
                                 all_cam_ids: List[int]) -> np.ndarray:
    width, height = 1800, 980
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    top_rect = (40, 70, 760, 820)
    side_rect = (840, 70, 760, 820)
    text_x0 = 1620

    camera_rows = []
    robot_rows = []
    gripper_cam_rows = []
    for ci in all_cam_ids:
        key = f"T_base_C{int(ci)}"
        T = transforms.get(key)
        if isinstance(T, np.ndarray):
            camera_rows.append((f"cam{ci}", np.asarray(T, dtype=np.float64), (200, 80, 40)))

    for cap in meta.get("captures", []):
        T_bg = load_robot_pose_from_capture(cap)
        if T_bg is not None:
            robot_rows.append((int(cap.get("event_id", -1)), np.asarray(T_bg, dtype=np.float64)))
        if gripper_cam_idx is not None:
            T_bc = get_event_base_camera_transform(cap, gripper_cam_idx, transforms, gripper_cam_idx)
            if T_bc is not None:
                gripper_cam_rows.append((int(cap.get("event_id", -1)), np.asarray(T_bc, dtype=np.float64)))

    T_obj = transforms.get("T_base_O")
    world_xy = []
    world_xz = []
    for _, T, _ in camera_rows:
        world_xy.append((float(T[0, 3] * 1000.0), float(T[1, 3] * 1000.0)))
        world_xz.append((float(T[0, 3] * 1000.0), float(T[2, 3] * 1000.0)))
    for _, T in robot_rows:
        world_xy.append((float(T[0, 3] * 1000.0), float(T[1, 3] * 1000.0)))
        world_xz.append((float(T[0, 3] * 1000.0), float(T[2, 3] * 1000.0)))
    for _, T in gripper_cam_rows:
        world_xy.append((float(T[0, 3] * 1000.0), float(T[1, 3] * 1000.0)))
        world_xz.append((float(T[0, 3] * 1000.0), float(T[2, 3] * 1000.0)))
    if isinstance(T_obj, np.ndarray):
        world_xy.append((float(T_obj[0, 3] * 1000.0), float(T_obj[1, 3] * 1000.0)))
        world_xz.append((float(T_obj[0, 3] * 1000.0), float(T_obj[2, 3] * 1000.0)))

    proj_xy = _fit_world_to_rect(world_xy, top_rect)
    proj_xz = _fit_world_to_rect(world_xz, side_rect)

    cv2.putText(canvas, "Base Frame Overview (XY top view)", (top_rect[0], 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Base Frame Overview (XZ side view)", (side_rect[0], 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (top_rect[0], top_rect[1]), (top_rect[0] + top_rect[2], top_rect[1] + top_rect[3]), (120, 120, 120), 2)
    cv2.rectangle(canvas, (side_rect[0], side_rect[1]), (side_rect[0] + side_rect[2], side_rect[1] + side_rect[3]), (120, 120, 120), 2)

    def draw_pose_pair(T: np.ndarray, label: str, color: Tuple[int, int, int], radius: int = 6):
        x_mm = float(T[0, 3] * 1000.0)
        y_mm = float(T[1, 3] * 1000.0)
        z_mm = float(T[2, 3] * 1000.0)
        u_xy, v_xy = proj_xy(x_mm, y_mm)
        u_xz, v_xz = proj_xz(x_mm, z_mm)
        cv2.circle(canvas, (u_xy, v_xy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (u_xz, v_xz), radius, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (u_xy + 8, v_xy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (u_xz + 8, v_xz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for idx in range(1, len(robot_rows)):
        p0 = robot_rows[idx - 1][1]
        p1 = robot_rows[idx][1]
        cv2.line(canvas, proj_xy(float(p0[0, 3] * 1000.0), float(p0[1, 3] * 1000.0)),
                 proj_xy(float(p1[0, 3] * 1000.0), float(p1[1, 3] * 1000.0)), (180, 180, 180), 2, cv2.LINE_AA)
        cv2.line(canvas, proj_xz(float(p0[0, 3] * 1000.0), float(p0[2, 3] * 1000.0)),
                 proj_xz(float(p1[0, 3] * 1000.0), float(p1[2, 3] * 1000.0)), (180, 180, 180), 2, cv2.LINE_AA)

    for label, T, color in camera_rows:
        draw_pose_pair(T, label, color, radius=8)
    for eid, T in robot_rows:
        draw_pose_pair(T, f"G{eid}", (110, 110, 110), radius=4)
    for eid, T in gripper_cam_rows:
        draw_pose_pair(T, f"C2@{eid}", (20, 140, 255), radius=4)
    if isinstance(T_obj, np.ndarray):
        draw_pose_pair(np.asarray(T_obj, dtype=np.float64), "Object", (30, 30, 220), radius=10)

    cv2.putText(canvas, "Final transforms", (text_x0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    text_y = 80
    for name, T in iter_public_transform_items(transforms):
        pos = T[:3, 3] * 1000.0
        lines = [
            name,
            f"  xyz_mm=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]",
        ]
        for line in lines:
            cv2.putText(canvas, line, (text_x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (30, 30, 30), 1, cv2.LINE_AA)
            text_y += 22
        text_y += 6
    return canvas


def show_cv_pages(title: str, pages: List[np.ndarray]) -> None:
    if not pages:
        return
    idx = 0
    while True:
        window_title = f"{title} ({idx + 1}/{len(pages)})"
        cv2.imshow(window_title, pages[idx])
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)
        if key in (27, ord('q')):
            break
        if key in (32, 13, ord('n')) and idx + 1 < len(pages):
            idx += 1
            continue
        if key in (ord('p'),) and idx > 0:
            idx -= 1
            continue
        break


def figure_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# ══════════════════════════════════════════════════════════════
# Verification tests
# ══════════════════════════════════════════════════════════════

def test_cross_camera_consistency(meta, transforms, all_cam_ids, gripper_cam_idx,
                                  root_folder=None, intrinsics_dir=None,
                                  cube_cfg=None, include_meta=False,
                                  selection_profile="default"):
    """Test: same cube seen from different cameras -> same position in base frame."""
    print("\n" + "=" * 60)
    print("[TEST 1] Cross-camera consistency")
    print("=" * 60)

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
        # Collect cube positions in base frame from each camera
        positions_base = []
        cam_labels = []
        event_candidate_map = {}

        if use_current_cube:
            event_candidate_map = build_capture_cube_candidate_map(
                cap, root_folder, K_map, D_map, cube, gripper_cam_idx,
                include_meta=include_meta, depth_scale_map=depth_scale_map)

        refined_selection = select_consistent_event_cube_candidates(
            cap, event_candidate_map, tf, gripper_cam_idx, **profile_kwargs) if event_candidate_map else {}

        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
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
            positions_base.append(T_base_cube[:3, 3] * 1000.0)  # mm
            cam_labels.append(ci)

        if len(positions_base) < 2:
            continue

        n_events += 1
        positions = np.array(positions_base)
        mean_pos = positions.mean(axis=0)

        for i, (pos, ci) in enumerate(zip(positions, cam_labels)):
            err = np.linalg.norm(pos - mean_pos)
            errors_mm.append(err)

    if errors_mm:
        print(f"  Events with 2+ cameras: {n_events}")
        print(f"  Position error (vs mean):")
        print(f"    mean:   {np.mean(errors_mm):.2f} mm")
        print(f"    median: {np.median(errors_mm):.2f} mm")
        print(f"    max:    {np.max(errors_mm):.2f} mm")
        print(f"    std:    {np.std(errors_mm):.2f} mm")
        ok = np.mean(errors_mm) < 5.0
        print(f"  Result: {'PASS' if ok else 'FAIL'} (threshold: 5mm)")
    else:
        print("  [SKIP] Not enough multi-camera observations")
        ok = None

    return errors_mm


def test_reprojection(meta, transforms, intrinsics_dir, all_cam_ids, root_folder,
                      gripper_cam_idx=None, cube_cfg=None, include_meta=False):
    """Test: solve current cube model and measure corner reprojection against fresh detections."""
    print("\n" + "=" * 60)
    print("[TEST 2] Reprojection verification")
    print("=" * 60)

    cfg = cube_cfg or resolve_cube_config_for_run(root_folder, default_cfg=get_default_cube_config())[0]
    cube = ArucoCubeTarget(cfg)
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
                    obj_pts.reshape(-1, 1, 3),
                    rvec,
                    t.reshape(3, 1),
                    K_map[ci],
                    D_map[ci],
                )
                proj = proj.reshape(-1, 2)
                total_err.extend(np.linalg.norm(proj - img_pts, axis=1).tolist())
            if total_err:
                errors_px.append(float(np.mean(total_err)))

    if errors_px:
        print(f"  Total observations: {len(errors_px)}")
        print(f"  Reprojection error (current cube solve):")
        print(f"    mean:   {np.mean(errors_px):.3f} px")
        print(f"    median: {np.median(errors_px):.3f} px")
        print(f"    max:    {np.max(errors_px):.3f} px")
        print(f"    <1px:   {sum(1 for e in errors_px if e < 1.0)}/{len(errors_px)}")
        ok = np.mean(errors_px) < 2.0
        print(f"  Result: {'PASS' if ok else 'FAIL'} (threshold: 2px)")
    else:
        print("  [SKIP] No reprojection data")

    return errors_px


def test_handeye_consistency(meta, transforms, gripper_cam_idx, root_folder=None, intrinsics_dir=None):
    """Test: T_base_board should be constant (board is fixed)."""
    print("\n" + "=" * 60)
    print("[TEST 3] Hand-eye consistency (board stability)")
    print("=" * 60)

    T_gTc = transforms.get("T_gripper_cam")
    if T_gTc is None:
        print("  [SKIP] T_gripper_cam not found")
        return []

    charuco_by_event = {}
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        if eid < 0:
            continue
        gi_data = cap.get("cams", {}).get(str(gripper_cam_idx), {})
        ch = gi_data.get("charuco")
        if ch and ch.get("ok") and ch.get("T_cam_board_4x4") is not None:
            charuco_by_event[eid] = np.asarray(ch["T_cam_board_4x4"], dtype=np.float64)

    if len(charuco_by_event) < 2 and root_folder and intrinsics_dir:
        try:
            from charuco_utils import CharucoTarget
            from config import CharucoBoardConfig

            g_K, g_D = load_intrinsics_color(intrinsics_dir, gripper_cam_idx)
            charuco_det = CharucoTarget(CharucoBoardConfig())
            print("  No/insufficient ChArUco in metadata, detecting from saved gripper images...")
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
                ch_ok, ch_rvec, ch_tvec, ch_n, _ = charuco_det.estimate_pose(img, g_K, g_D)
                if ch_ok and ch_rvec is not None and ch_n >= 4:
                    charuco_by_event[eid] = rodrigues_to_Rt(ch_rvec, ch_tvec)
        except Exception as e:
            print(f"  [WARN] ChArUco fallback detection failed: {e}")

    T_base_board_list = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        T_base_cam = get_event_base_camera_transform(cap, gripper_cam_idx, transforms, gripper_cam_idx)
        if T_base_cam is None:
            continue

        T_cam_board = charuco_by_event.get(eid)
        if T_cam_board is None:
            continue

        T_base_board = np.asarray(T_base_cam, dtype=np.float64) @ T_cam_board
        T_base_board_list.append(T_base_board)

    if len(T_base_board_list) < 2:
        print("  [SKIP] Not enough ChArUco observations")
        return []

    # Compute consistency
    positions = np.array([T[:3, 3] * 1000.0 for T in T_base_board_list])
    mean_pos = positions.mean(axis=0)

    pos_errors = [np.linalg.norm(p - mean_pos) for p in positions]
    rot_errors = [rotation_error_deg(T[:3, :3], T_base_board_list[0][:3, :3])
                  for T in T_base_board_list]

    print(f"  Frames: {len(T_base_board_list)}")
    print(f"  Board position stability:")
    print(f"    std: {np.std(pos_errors):.2f} mm")
    print(f"    max: {np.max(pos_errors):.2f} mm")
    print(f"  Board rotation stability:")
    print(f"    mean: {np.mean(rot_errors):.3f} deg")
    print(f"    max:  {np.max(rot_errors):.3f} deg")
    ok = np.std(pos_errors) < 3.0 and np.mean(rot_errors) < 1.0
    print(f"  Result: {'PASS' if ok else 'FAIL'} (pos<3mm, rot<1deg)")

    return pos_errors


# ══════════════════════════════════════════════════════════════
# Cube Candidate Diagnostics
# ══════════════════════════════════════════════════════════════

def collect_cube_candidate_diagnostics(meta, transforms, intrinsics_dir, root_folder,
                                       gripper_cam_idx, all_cam_ids, cube_cfg=None,
                                       include_meta=False,
                                       selection_profile="default"):
    T_gTc = transforms.get("T_gripper_cam")
    if not any(True for _ in iter_object_anchor_items(transforms)):
        print("  [SKIP] Candidate diagnostics need T_base_O or T_base_O_set*")
        return []

    cfg = cube_cfg or resolve_cube_config_for_run(root_folder, default_cfg=get_default_cube_config())[0]
    cube = ArucoCubeTarget(cfg)
    K_map, D_map, depth_scale_map = {}, {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], depth_scale_map[ci] = load_intrinsics_with_depth_scale(intrinsics_dir, ci)
    profile_kwargs = cube_selection_profile_kwargs(selection_profile)

    rows = []
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        T_base_O, anchor_key = get_capture_object_anchor(cap, transforms)
        if T_base_O is None:
            continue
        event_candidate_map = build_capture_cube_candidate_map(
            cap, root_folder, K_map, D_map, cube, gripper_cam_idx,
            include_meta=include_meta, depth_scale_map=depth_scale_map)

        refined_selection = select_consistent_event_cube_candidates(
            cap, event_candidate_map, transforms, gripper_cam_idx, **profile_kwargs) if event_candidate_map else {}

        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in K_map or not cinfo.get("saved"):
                continue
            T_base_cam = get_event_base_camera_transform(cap, ci, transforms, gripper_cam_idx)
            if T_base_cam is None:
                continue

            candidates = event_candidate_map.get(ci, [])
            if not candidates:
                continue

            event_rows = []
            for idx, cand in enumerate(candidates):
                T_base_O_cand = T_base_cam @ cand["T_C_O"]
                obj_dt = float(np.linalg.norm(T_base_O_cand[:3, 3] - T_base_O[:3, 3]) * 1000.0)
                obj_dr = rotation_error_deg(T_base_O_cand[:3, :3], T_base_O[:3, :3])
                cam_dt = None
                cam_dr = None
                cam_key = f"T_base_C{ci}"
                if cam_key in transforms:
                    T_base_C_from_anchor = T_base_O @ inv_T(cand["T_C_O"])
                    cam_dt = float(np.linalg.norm(
                        T_base_C_from_anchor[:3, 3] - transforms[cam_key][:3, 3]) * 1000.0)
                    cam_dr = rotation_error_deg(
                        T_base_C_from_anchor[:3, :3], transforms[cam_key][:3, :3])

                score = obj_dt + 5.0 * obj_dr + 10.0 * float(cand.get("err_mean", 1.0))
                if cand.get("used_ids"):
                    score -= 0.1 * len(set(int(x) for x in cand["used_ids"]))

                row = {
                    "cam_idx": ci,
                    "event_id": eid,
                    "image_path": os.path.join(root_folder, cinfo.get("rgb_path", "")),
                    "source": str(cand.get("source", "unknown")),
                    "used_ids": [int(x) for x in cand.get("used_ids", [])],
                    "n_points": int(cand.get("n_points", 0)),
                    "err_mean_px": float(cand.get("err_mean", 99.0)),
                    "obj_dt_mm": obj_dt,
                    "obj_dr_deg": obj_dr,
                    "cam_dt_mm": cam_dt,
                    "cam_dr_deg": cam_dr,
                    "score": float(score),
                    "anchor_key": anchor_key,
                    "selected": False,
                    "accepted": False,
                }
                rows.append(row)
                event_rows.append(row)

            if event_rows:
                selected_cand = refined_selection.get(ci)
                if selected_cand is not None:
                    best = min(
                        event_rows,
                        key=lambda r: (
                            0 if (
                                r["source"] == str(selected_cand.get("source", "unknown")) and
                                r["used_ids"] == [int(x) for x in selected_cand.get("used_ids", [])] and
                                abs(r["err_mean_px"] - float(selected_cand.get("err_mean", 99.0))) < 1e-9
                            ) else 1,
                            r["score"],
                        ),
                    )
                else:
                    best = min(event_rows, key=lambda r: r["score"])
                best["selected"] = True
                best["accepted"] = (
                    (best["obj_dt_mm"] <= 60.0 and best["obj_dr_deg"] <= 12.0) or
                    (best["cam_dt_mm"] is not None and best["cam_dt_mm"] <= 80.0 and
                     best["cam_dr_deg"] is not None and best["cam_dr_deg"] <= 15.0)
                )

    return rows


def _candidate_marker_corners(cap, ci: int, used_ids: List[int]):
    cinfo = cap.get("cams", {}).get(str(ci), {})
    markers = cinfo.get("markers", [])
    out = []
    for item in markers:
        mid = int(item.get("marker_id", -1))
        if mid in used_ids:
            pts = np.asarray(item.get("corners_2d", []), dtype=np.int32)
            if pts.shape == (4, 2):
                out.append((mid, pts))
    return out


def render_candidate_tile(meta_by_event, row):
    img = cv2.imread(row["image_path"])
    if img is None:
        return None

    cap = meta_by_event.get(row["event_id"])
    color = (40, 180, 40) if row["accepted"] else (30, 60, 220)
    if row["selected"] and not row["accepted"]:
        color = (0, 165, 255)
    for mid, pts in _candidate_marker_corners(cap, row["cam_idx"], row["used_ids"]):
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
        p0 = tuple(int(x) for x in pts[0])
        cv2.putText(img, f"id{mid}", p0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    lines = [
        f"cam{row['cam_idx']} e{row['event_id']} {'ACCEPT' if row['accepted'] else 'REJECT'}",
        f"{row['source']} ids={row['used_ids']} err={row['err_mean_px']:.3f}px",
        f"obj: {row['obj_dt_mm']:.1f}mm / {row['obj_dr_deg']:.1f}deg",
    ]
    if row["cam_dt_mm"] is not None and row["cam_dr_deg"] is not None:
        lines.append(f"cam: {row['cam_dt_mm']:.1f}mm / {row['cam_dr_deg']:.1f}deg")

    y = 18
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 18
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_cube_candidate_scatter(rows):
    if not rows:
        return None
    cams = sorted(set(r["cam_idx"] for r in rows))
    fig, axes = plt.subplots(1, len(cams), figsize=(5.5 * len(cams), 4.5), squeeze=False)
    marker_style = {"meta": "s", "multi": "o", "ippe0": "^", "ippe1": "v"}
    for ax, ci in zip(axes[0], cams):
        cam_rows = [r for r in rows if r["cam_idx"] == ci]
        for row in cam_rows:
            x = row["obj_dt_mm"]
            y = row["obj_dr_deg"]
            if row["selected"] and row["accepted"]:
                color, alpha, size = "green", 0.9, 70
            elif row["selected"]:
                color, alpha, size = "orange", 0.9, 70
            else:
                color, alpha, size = "crimson", 0.28, 35
            ax.scatter(
                x, y,
                c=color, alpha=alpha, s=size,
                marker=marker_style.get(row["source"], "o"),
                edgecolors="black" if row["selected"] else "none")
        for row in [r for r in cam_rows if r["selected"]]:
            ax.annotate(
                f"e{row['event_id']} {row['used_ids']}",
                (row["obj_dt_mm"], row["obj_dr_deg"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.axvline(60.0, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(12.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(f"cam{ci} Candidate Scores")
        ax.set_xlabel("Object Error to Assigned Anchor (mm)")
        ax.set_ylabel("Rotation Error (deg)")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def visualize_marker_health(rows):
    single_rows = [r for r in rows if len(r["used_ids"]) == 1]
    if not single_rows:
        return None

    best_rows = {}
    for row in single_rows:
        key = (row["cam_idx"], row["event_id"], row["used_ids"][0])
        if key not in best_rows or row["score"] < best_rows[key]["score"]:
            best_rows[key] = row

    per_marker = defaultdict(list)
    for row in best_rows.values():
        per_marker[row["used_ids"][0]].append(row)

    mids = sorted(per_marker)
    mean_dt = [float(np.mean([r["obj_dt_mm"] for r in per_marker[mid]])) for mid in mids]
    mean_dr = [float(np.mean([r["obj_dr_deg"] for r in per_marker[mid]])) for mid in mids]
    accept_rate = [
        float(np.mean([1.0 if r["accepted"] else 0.0 for r in per_marker[mid]]))
        for mid in mids
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(mids))
    axes[0].bar(x, mean_dt, color="steelblue")
    axes[0].set_xticks(x, [f"id{mid}" for mid in mids])
    axes[0].set_ylabel("Mean Object Error (mm)")
    axes[0].set_title("Best Single-Marker Distance to Assigned Anchor")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, mean_dr, color="darkorange")
    axes[1].set_xticks(x, [f"id{mid}" for mid in mids])
    axes[1].set_ylabel("Mean Rotation Error (deg)")
    axes[1].set_title("Best Single-Marker Rotation Error")
    axes[1].grid(True, axis="y", alpha=0.25)

    for ax, vals in zip(axes, [accept_rate, accept_rate]):
        for i, rate in enumerate(vals):
            ax.text(i, ax.get_ylim()[1] * 0.94, f"ok={rate:.2f}", ha="center", va="top", fontsize=8)

    plt.tight_layout()
    return fig


def visualize_candidate_examples(meta, rows, save_dir):
    if not rows:
        return []
    meta_by_event = {int(cap.get("event_id", -1)): cap for cap in meta.get("captures", [])}
    saved = []
    for ci in sorted(set(r["cam_idx"] for r in rows)):
        cam_rows = [r for r in rows if r["cam_idx"] == ci and r["selected"]]
        if not cam_rows:
            continue
        accepted = sorted([r for r in cam_rows if r["accepted"]], key=lambda r: r["score"])[:3]
        rejected = sorted(
            [r for r in cam_rows if not r["accepted"]],
            key=lambda r: (-r["obj_dt_mm"], -r["obj_dr_deg"], -r["score"]))[:3]
        chosen = accepted + rejected
        if not chosen:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.reshape(-1)
        for ax, row in zip(axes, chosen):
            tile = render_candidate_tile(meta_by_event, row)
            if tile is None:
                ax.axis("off")
                continue
            ax.imshow(tile)
            ax.set_axis_off()
        for ax in axes[len(chosen):]:
            ax.axis("off")
        fig.suptitle(f"cam{ci} accepted/rejected cube candidates", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"cam{ci}_candidate_examples.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved


def collect_marker_override_diagnostics(meta, transforms, intrinsics_dir, root_folder,
                                        gripper_cam_idx, all_cam_ids, cube_cfg=None):
    T_gTc = transforms.get("T_gripper_cam")
    if not any(True for _ in iter_object_anchor_items(transforms)):
        print("  [SKIP] Override diagnostics need T_base_O or T_base_O_set*")
        return {}

    cfg = cube_cfg or resolve_cube_config_for_run(root_folder, default_cfg=get_default_cube_config())[0]
    model = ArucoCubeModel(cfg)
    cube = ArucoCubeTarget(cfg)
    reorder_to_name = {tuple(v): k for k, v in DIAG_CORNER_PERMUTATIONS.items()}

    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci] = load_intrinsics_color(intrinsics_dir, ci)

    face_obj = {}
    for face in DIAG_FACES:
        c, u, v, _ = model.face_defs[face]
        face_obj[face] = np.asarray(
            [c + u * p[0] + v * p[1] for p in model.local_corners],
            dtype=np.float64)

    per_marker_obs = defaultdict(list)
    for cap in meta.get("captures", []):
        eid = int(cap.get("event_id", -1))
        T_B_G = load_robot_pose_from_capture(cap)
        T_base_O_event, _ = get_capture_object_anchor(cap, transforms)
        if T_base_O_event is None:
            continue
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            if ci not in K_map or not cinfo.get("saved"):
                continue
            T_base_cam = None
            if ci == gripper_cam_idx and T_B_G is not None and T_gTc is not None:
                T_base_cam = T_B_G @ T_gTc
            elif f"T_base_C{ci}" in transforms:
                T_base_cam = transforms[f"T_base_C{ci}"]
            if T_base_cam is None:
                continue

            image_path = os.path.join(root_folder, cinfo.get("rgb_path", ""))
            raw_rows = []
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    corners_list, ids = cube.detect(img)
                    if ids is not None:
                        for corners, mid in zip(corners_list, ids):
                            raw_rows.append({
                                "marker_id": int(mid),
                                "corners": corners.reshape(4, 2).astype(np.float64),
                            })

            if raw_rows:
                visible_ids = [row["marker_id"] for row in raw_rows]
                source_rows = raw_rows
            else:
                visible_ids = [int(m.get("marker_id", -1)) for m in cinfo.get("markers", [])]
                source_rows = []
                for item in cinfo.get("markers", []):
                    mid = int(item.get("marker_id", -1))
                    corners = np.asarray(item.get("corners_2d", []), dtype=np.float64)
                    if corners.shape != (4, 2):
                        continue
                    source_rows.append({
                        "marker_id": mid,
                        "corners": corners,
                    })

            for item in source_rows:
                mid = int(item["marker_id"])
                corners = np.asarray(item["corners"], dtype=np.float64)
                if mid not in cfg.marker_ids or corners.shape != (4, 2):
                    continue
                per_marker_obs[mid].append({
                    "cam_idx": ci,
                    "event_id": eid,
                    "image_path": image_path,
                    "corners": corners,
                    "visible_ids": visible_ids,
                    "area_px2": float(abs(cv2.contourArea(corners.astype(np.float32)))),
                    "T_base_cam": T_base_cam,
                    "T_base_O_anchor": T_base_O_event,
                })

    report = {}
    for mid in cfg.marker_ids:
        obs = per_marker_obs.get(mid, [])
        if not obs:
            continue

        current_perm_name = reorder_to_name.get(
            tuple(cfg.corner_reorder.get(mid, [0, 1, 2, 3])),
            "custom")
        current_face = cfg.id_to_face[mid]

        rankings = []
        for face in DIAG_FACES:
            for perm_name, reorder in DIAG_CORNER_PERMUTATIONS.items():
                rows = []
                for row in obs:
                    try:
                        n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                            face_obj[face].reshape(-1, 1, 3),
                            row["corners"][reorder].reshape(-1, 1, 2),
                            K_map[row["cam_idx"]], D_map[row["cam_idx"]],
                            flags=cv2.SOLVEPNP_IPPE)
                    except cv2.error:
                        continue

                    best = None
                    for si in range(int(n_sol)):
                        R, _ = cv2.Rodrigues(rvecs[si])
                        T = np.eye(4, dtype=np.float64)
                        T[:3, :3] = R
                        T[:3, 3] = tvecs[si].reshape(3)
                        T_base_O_cand = row["T_base_cam"] @ T
                        obj_dt = float(np.linalg.norm(
                            T_base_O_cand[:3, 3] - row["T_base_O_anchor"][:3, 3]) * 1000.0)
                        obj_dr = rotation_error_deg(
                            T_base_O_cand[:3, :3], row["T_base_O_anchor"][:3, :3])
                        reproj = float(reproj_errs[si][0]) if reproj_errs is not None else 99.0
                        score = obj_dt + 5.0 * obj_dr + 10.0 * reproj
                        if best is None or score < best["score"]:
                            best = {
                                "obj_dt_mm": obj_dt,
                                "obj_dr_deg": obj_dr,
                                "reproj_px": reproj,
                                "ippe_solution": int(si),
                                "score": score,
                            }
                    if best is not None:
                        rows.append(best)

                num_rows = len(rows)
                if num_rows:
                    num_inliers = int(sum(
                        1 for r in rows
                        if r["obj_dt_mm"] <= 60.0 and r["obj_dr_deg"] <= 15.0))
                    mean_dt = float(np.mean([r["obj_dt_mm"] for r in rows]))
                    mean_dr = float(np.mean([r["obj_dr_deg"] for r in rows]))
                    mean_reproj = float(np.mean([r["reproj_px"] for r in rows]))
                    score = mean_dt + 8.0 * mean_dr + 20.0 * (len(obs) - num_rows) + 2.0 * mean_reproj - 5.0 * num_inliers
                else:
                    num_inliers = 0
                    mean_dt = 1e9
                    mean_dr = 1e9
                    mean_reproj = 1e9
                    score = 1e9

                rankings.append({
                    "face": face,
                    "corner_permutation": perm_name,
                    "corner_reorder": list(reorder),
                    "num_obs": int(len(obs)),
                    "num_used": int(num_rows),
                    "num_inliers": int(num_inliers),
                    "mean_dt_mm": mean_dt,
                    "mean_dr_deg": mean_dr,
                    "mean_reproj_px": mean_reproj,
                    "score": float(score),
                })

        rankings.sort(key=lambda r: (r["score"], -r["num_inliers"], r["mean_reproj_px"]))
        current_rank = next(
            (i + 1 for i, item in enumerate(rankings)
             if item["face"] == current_face and item["corner_permutation"] == current_perm_name),
            None)
        current_entry = next(
            (item for item in rankings
             if item["face"] == current_face and item["corner_permutation"] == current_perm_name),
            None)
        best_entry = rankings[0]

        examples = []
        seen_keys = set()
        for row in sorted(obs, key=lambda r: (-r["area_px2"], r["cam_idx"], r["event_id"])):
            key = (row["cam_idx"], row["event_id"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            examples.append({
                "cam_idx": int(row["cam_idx"]),
                "event_id": int(row["event_id"]),
                "image_path": row["image_path"],
                "corners": row["corners"].astype(float).tolist(),
                "visible_ids": [int(x) for x in row["visible_ids"]],
                "area_px2": float(row["area_px2"]),
            })
            if len(examples) >= 6:
                break

        report[int(mid)] = {
            "marker_id": int(mid),
            "num_observations": int(len(obs)),
            "current": {
                "face": current_face,
                "corner_permutation": current_perm_name,
                "corner_reorder": list(cfg.corner_reorder[mid]),
                "rank": current_rank,
                "score": None if current_entry is None else current_entry["score"],
                "num_inliers": None if current_entry is None else current_entry["num_inliers"],
                "mean_dt_mm": None if current_entry is None else current_entry["mean_dt_mm"],
                "mean_dr_deg": None if current_entry is None else current_entry["mean_dr_deg"],
                "mean_reproj_px": None if current_entry is None else current_entry["mean_reproj_px"],
            },
            "best": best_entry,
            "top_candidates": rankings[:8],
            "score_improvement": None if current_entry is None else float(current_entry["score"] - best_entry["score"]),
            "examples": examples,
        }

    return report


def visualize_marker_override_summary(report):
    if not report:
        return None
    mids = sorted(report)
    fig, axes = plt.subplots(len(mids), 1, figsize=(10, 2.6 * len(mids)), squeeze=False)
    for ax, mid in zip(axes[:, 0], mids):
        row = report[mid]
        top = row["top_candidates"][:5]
        labels = [f"{c['face']}\n{c['corner_permutation']}" for c in top]
        scores = [c["score"] for c in top]
        colors = []
        for cand in top:
            if cand["face"] == row["best"]["face"] and cand["corner_permutation"] == row["best"]["corner_permutation"]:
                colors.append("forestgreen")
            elif (cand["face"] == row["current"]["face"] and
                  cand["corner_permutation"] == row["current"]["corner_permutation"]):
                colors.append("darkorange")
            else:
                colors.append("steelblue")
        ax.bar(np.arange(len(top)), scores, color=colors)
        ax.set_xticks(np.arange(len(top)), labels)
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(
            f"id{mid}: current rank={row['current']['rank']} "
            f"delta={0.0 if row['score_improvement'] is None else row['score_improvement']:.1f}")
    plt.tight_layout()
    return fig


def render_marker_gallery(report_row):
    examples = report_row.get("examples", [])
    if not examples:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.reshape(-1)
    current = report_row["current"]
    best = report_row["best"]
    for ax, item in zip(axes, examples):
        img = cv2.imread(item["image_path"])
        if img is None:
            ax.axis("off")
            continue
        raw_corners = np.asarray(item["corners"], dtype=np.int32)
        reorder = np.asarray(current.get("corner_reorder", [0, 1, 2, 3]), dtype=np.int32)
        corners = raw_corners[reorder]
        x, y, w, h = cv2.boundingRect(raw_corners.reshape(-1, 1, 2))
        pad = int(max(w, h) * 2.0)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, img.shape[1])
        y1 = min(y + h + pad, img.shape[0])

        cv2.polylines(img, [raw_corners.reshape(-1, 1, 2)], True, (160, 160, 160), 1)
        cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, (0, 220, 0), 2)
        for i, pt in enumerate(corners):
            cv2.circle(img, tuple(int(v) for v in pt), 4, (0, 255, 255), -1)
            cv2.putText(
                img, str(i), tuple(int(v) for v in pt + np.array([4, -4])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ax.imshow(crop)
        ax.set_axis_off()
        ax.set_title(
            f"cam{item['cam_idx']} e{item['event_id']} ids={item['visible_ids']}",
            fontsize=9)

    for ax in axes[len(examples):]:
        ax.axis("off")

    fig.suptitle(
        f"id{report_row['marker_id']} current={current['face']}/{current['corner_permutation']} "
        f"-> best={best['face']}/{best['corner_permutation']}",
        fontsize=12)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 3D Visualization
# ══════════════════════════════════════════════════════════════

def visualize_3d(meta,
                 transforms,
                 gripper_cam_idx,
                 all_cam_ids,
                 show_gripper_trajectory: bool = True,
                 camera_label_size: float = 7.0,
                 object_label_size: float = 8.0,
                 view_elev: float = 26.0,
                 view_azim: float = -58.0):
    """3D plot of robot base, cameras, cube positions, gripper poses."""
    print("\n" + "=" * 60)
    print("[VIS] 3D Visualization")
    print("=" * 60)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Robot base (origin)
    T_origin = np.eye(4)
    draw_frame(ax, T_origin, label="Robot Base", scale=40.0, lw=2.5, fontsize=object_label_size)

    # 2. Fixed cameras
    cam_colors = {0: 'blue', 1: 'green', 3: 'orange'}
    for ci in all_cam_ids:
        key = f"T_base_C{ci}"
        if key not in transforms:
            continue
        T = transforms[key]
        tag = "Gripper" if ci == gripper_cam_idx else "Fixed"
        color = cam_colors.get(ci, 'purple')
        if ci == gripper_cam_idx:
            color = 'red'
        draw_camera(ax, T, label=f"cam{ci} ({tag})", color=color, fontsize=camera_label_size)
        draw_frame(ax, T, scale=20.0, lw=1.0)

    # 3. Cube positions per event
    T_gTc = transforms.get("T_gripper_cam")
    cube_positions = []

    for cap in meta.get("captures", []):
        eid = cap.get("event_id", -1)

        # From fixed cameras
        for ci_str, cinfo in cap.get("cams", {}).items():
            ci = int(ci_str)
            cpnp = cinfo.get("cube_pnp")
            if not cpnp or not cpnp.get("ok"):
                continue
            key = f"T_base_C{ci}"
            if key not in transforms:
                continue
            T_cam_cube = np.asarray(cpnp["T_cam_cube_4x4"], dtype=np.float64)
            T_base_cube = transforms[key] @ T_cam_cube
            cube_positions.append(T_base_cube[:3, 3] * 1000.0)
            break  # one per event is enough

    if cube_positions:
        pts = np.array(cube_positions)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='cyan', s=30, marker='s', alpha=0.7, label='Cube positions')

    # 4. Gripper poses per event
    gripper_positions = []
    for cap in meta.get("captures", []):
        T_B_G = None
        if "robot_pose_matrix_4x4" in cap:
            try:
                T_B_G = np.asarray(cap["robot_pose_matrix_4x4"], dtype=np.float64)
            except Exception:
                pass
        if T_B_G is None and "robot_pose_6dof" in cap:
            try:
                T_B_G = euler_deg_to_matrix(*cap["robot_pose_6dof"])
            except Exception:
                pass
        if T_B_G is not None:
            gripper_positions.append(T_B_G[:3, 3] * 1000.0)

    if gripper_positions:
        pts = np.array(gripper_positions)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='red', s=15, marker='^', alpha=0.5, label='Gripper poses')
        if show_gripper_trajectory and len(pts) >= 2:
            ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                      color='red', linewidth=1.3, alpha=0.35, label='Gripper trajectory')

    # 5. Board position (average)
    drawn_anchor = False
    for key, T_base_O in iter_object_anchor_items(transforms):
        if key == "T_base_O":
            label = "Cube (avg)"
        else:
            label = f"Cube set {key.replace('T_base_O_set', '')}"
        draw_frame(ax, T_base_O, label=label, scale=25.0, lw=2.0, fontsize=object_label_size)
        drawn_anchor = True

    # Formatting
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Calibration Result - Robot Base Frame")
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    # Equal aspect ratio
    all_pts = []
    if cube_positions:
        all_pts.extend(cube_positions)
    if gripper_positions:
        all_pts.extend(gripper_positions)
    for ci in all_cam_ids:
        key = f"T_base_C{ci}"
        if key in transforms:
            all_pts.append(transforms[key][:3, 3] * 1000.0)
    all_pts.append(np.zeros(3))

    if all_pts:
        pts = np.array(all_pts)
        center = pts.mean(axis=0)
        max_range = max(pts.max(axis=0) - pts.min(axis=0)) / 2.0 * 1.2
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    return fig


def visualize_errors(cross_errors, reproj_errors, handeye_errors):
    """Plot error distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Cross-camera
    if cross_errors:
        axes[0].hist(cross_errors, bins=20, color='steelblue', edgecolor='black')
        axes[0].axvline(np.mean(cross_errors), color='red', linestyle='--',
                        label=f'mean={np.mean(cross_errors):.2f}mm')
        axes[0].set_xlabel("Position error (mm)")
        axes[0].set_title("Cross-camera consistency")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[0].transAxes)

    # Reprojection
    if reproj_errors:
        axes[1].hist(reproj_errors, bins=20, color='seagreen', edgecolor='black')
        axes[1].axvline(np.mean(reproj_errors), color='red', linestyle='--',
                        label=f'mean={np.mean(reproj_errors):.3f}px')
        axes[1].set_xlabel("Reprojection error (px)")
        axes[1].set_title("Reprojection error")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[1].transAxes)

    # Hand-eye consistency
    if handeye_errors:
        axes[2].hist(handeye_errors, bins=20, color='coral', edgecolor='black')
        axes[2].axvline(np.mean(handeye_errors), color='red', linestyle='--',
                        label=f'mean={np.mean(handeye_errors):.2f}mm')
        axes[2].set_xlabel("Board position error (mm)")
        axes[2].set_title("Hand-eye consistency")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No data", ha='center', va='center', transform=axes[2].transAxes)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Calibration verification & visualization")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--gripper_cam_idx", type=int, default=None)
    parser.add_argument("--cube_config_json", type=str, default=None)
    parser.add_argument("--cube_selection_profile", type=str, default="default",
                        choices=["default", "cube_only_specialized"])
    parser.add_argument("--hide_gripper_trajectory", action="store_true",
                        help="Do not draw gripper trajectory in 3D overview")
    parser.add_argument("--camera_label_size", type=float, default=7.0)
    parser.add_argument("--object_label_size", type=float, default=8.0)
    parser.add_argument("--view_elev", type=float, default=26.0)
    parser.add_argument("--view_azim", type=float, default=-58.0)
    parser.add_argument("--no_show", action="store_true", help="Save plots without showing")
    args = parser.parse_args()

    root = args.root_folder
    calib_dir = args.calib_dir or os.path.join(root, "calib_out")

    # Load meta
    meta_path = os.path.join(root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load calibration summary
    summary_path = os.path.join(calib_dir, "calibration_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    # Load transforms
    transforms = load_calib(calib_dir)
    public_keys = [name for name, _ in iter_public_transform_items(transforms)]
    internal_event_count = sum(1 for name in transforms.keys() if "_event" in name)
    print(f"[INFO] Loaded public transforms: {public_keys}")
    if internal_event_count:
        print(f"[INFO] Loaded internal runtime transforms: {internal_event_count} event-specific entries")

    # Camera info
    gripper_cam_idx = args.gripper_cam_idx
    if gripper_cam_idx is None:
        gripper_cam_idx = summary.get("gripper_cam_idx") or meta.get("gripper_cam_idx")

    all_cam_ids = sorted({
        int(k) for cap in meta.get("captures", [])
        for k in cap.get("cams", {}).keys()
    })
    print(f"[INFO] Cameras: {all_cam_ids}, gripper=cam{gripper_cam_idx}")
    cube_cfg, cube_cfg_source = resolve_cube_config_for_run(
        root, calib_dir=calib_dir, cube_config_json=args.cube_config_json, default_cfg=get_default_cube_config())
    include_meta_candidates = False
    print(f"[INFO] cube config source: {cube_cfg_source}")
    print(f"[INFO] cube id_to_face: {cube_cfg.id_to_face}")
    print(f"[INFO] cube corner_reorder: {cube_cfg.corner_reorder}")

    # Keep metric computation on the original transform set.
    # A gripper camera does not have a single global T_base_Ci; it changes per event.
    # Add a one-frame approximation only to the visualization copy.
    viz_transforms = dict(transforms)
    T_gTc = transforms.get("T_gripper_cam")
    if T_gTc is not None and gripper_cam_idx is not None:
        for cap in meta.get("captures", []):
            T_B_G = None
            if "robot_pose_matrix_4x4" in cap:
                try:
                    T_B_G = np.asarray(cap["robot_pose_matrix_4x4"], dtype=np.float64)
                except Exception:
                    pass
            if T_B_G is None and "robot_pose_6dof" in cap:
                try:
                    T_B_G = euler_deg_to_matrix(*cap["robot_pose_6dof"])
                except Exception:
                    pass
            if T_B_G is not None:
                key = f"T_base_C{gripper_cam_idx}"
                if key not in viz_transforms:
                    viz_transforms[key] = T_B_G @ T_gTc
                break

    # ─── Run tests ───
    cross_err = test_cross_camera_consistency(
        meta, transforms, all_cam_ids, gripper_cam_idx,
        root_folder=root, intrinsics_dir=args.intrinsics_dir,
        cube_cfg=cube_cfg, include_meta=False,
        selection_profile=args.cube_selection_profile)
    reproj_err = test_reprojection(
        meta, transforms, args.intrinsics_dir, all_cam_ids, root,
        gripper_cam_idx=gripper_cam_idx, cube_cfg=cube_cfg,
        include_meta=include_meta_candidates)
    he_err = test_handeye_consistency(
        meta, transforms, gripper_cam_idx, root_folder=root, intrinsics_dir=args.intrinsics_dir)
    board_reproj = compute_board_reprojection_metrics(
        meta, root, args.intrinsics_dir, all_cam_ids)
    pose_repeat = compute_pose_repeatability_metrics(
        meta, transforms, args.intrinsics_dir, root, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta_candidates,
        selection_profile=args.cube_selection_profile)
    depth_metrics = compute_depth_cube_metrics(
        meta, transforms, args.intrinsics_dir, root, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta_candidates,
        selection_profile=args.cube_selection_profile)

    # ─── Print calibration summary ───
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    if summary:
        print(f"  Hand-eye method: {summary.get('selected_handeye_method', 'N/A')}")
        print(f"  Data source: {summary.get('handeye_data_source', 'N/A')}")
        print(f"  Robot poses: {summary.get('num_robot_poses', 0)}")
        print(f"  Hand-eye events: {summary.get('num_handeye_events', 0)}")
        print(f"  ChArUco frames: {summary.get('num_charuco_frames', 0)}")
    if board_reproj["mean_px"] is not None:
        print(f"  Board reprojection mean: {board_reproj['mean_px']:.3f}px")
    if pose_repeat["mean_dt_mm"] is not None:
        print(f"  Pose repeatability: {pose_repeat['mean_dt_mm']:.2f}mm / {pose_repeat['mean_dr_deg']:.3f}deg")
    if depth_metrics["mesh_alignment"]["mean_rmse_mm"] is not None:
        print(
            f"  Depth mesh RMSE: {depth_metrics['mesh_alignment']['mean_rmse_mm']:.2f}mm | "
            f"dim err: {depth_metrics['dimension_accuracy']['mean_abs_err_mm']:.2f}mm"
        )

    # Print transforms
    print("\n  Transforms:")
    for name, T in iter_public_transform_items(transforms):
        pos = T[:3, 3] * 1000.0
        print(f"    {name}: pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm")

    verification = {
        "cross_camera": {
            "num_errors": int(len(cross_err)),
            "mean_mm": None if not cross_err else float(np.mean(cross_err)),
            "median_mm": None if not cross_err else float(np.median(cross_err)),
            "max_mm": None if not cross_err else float(np.max(cross_err)),
            "pass": None if not cross_err else bool(np.mean(cross_err) < 5.0),
        },
        "reprojection": {
            "total_observations": int(len(reproj_err)),
            "mean_px": None if not reproj_err else float(np.mean(reproj_err)),
            "median_px": None if not reproj_err else float(np.median(reproj_err)),
            "max_px": None if not reproj_err else float(np.max(reproj_err)),
            "pass": None if not reproj_err else bool(np.mean(reproj_err) < 2.0),
        },
        "handeye": {
            "frames": int(len(he_err)),
            "board_position_std_mm": None if not he_err else float(np.std(he_err)),
            "board_position_max_mm": None if not he_err else float(np.max(he_err)),
            "pass": None if not he_err else bool(np.std(he_err) < 3.0),
        },
        "board_reprojection": board_reproj,
        "pose_repeatability": pose_repeat,
        "mesh_alignment": depth_metrics["mesh_alignment"],
        "dimension_accuracy": depth_metrics["dimension_accuracy"],
    }

    save_dir = os.path.join(calib_dir, "verify")
    os.makedirs(save_dir, exist_ok=True)
    legacy_base_cv_path = os.path.join(save_dir, "base_frame_overview_cv.png")
    if os.path.exists(legacy_base_cv_path):
        try:
            os.remove(legacy_base_cv_path)
        except OSError:
            pass
    verification_path = os.path.join(calib_dir, "verification_metrics.json")
    with open(verification_path, "w") as f:
        json.dump(verification, f, indent=2)
    print(f"[SAVE] {verification_path}")

    selected_pages, selected_manifest = build_selected_event_contact_sheets(
        meta, transforms, args.intrinsics_dir, root, all_cam_ids, gripper_cam_idx,
        cube_cfg, include_meta=include_meta_candidates,
        selection_profile=args.cube_selection_profile)
    if selected_pages:
        manifest_path = os.path.join(save_dir, "selected_event_images_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(selected_manifest, f, indent=2)
        print(f"[SAVE] {manifest_path}")
        for page_idx, page in enumerate(selected_pages, start=1):
            page_path = os.path.join(save_dir, f"selected_event_images_cv_{page_idx:02d}.jpg")
            cv2.imwrite(page_path, page)
            print(f"[SAVE] {page_path}")

    base_overview_cv = build_base_frame_overview_cv(meta, transforms, gripper_cam_idx, all_cam_ids)

    fig_3d = visualize_3d(
        meta, viz_transforms, gripper_cam_idx, all_cam_ids,
        show_gripper_trajectory=not args.hide_gripper_trajectory,
        camera_label_size=float(args.camera_label_size),
        object_label_size=float(args.object_label_size),
        view_elev=float(args.view_elev),
        view_azim=float(args.view_azim),
    )
    fig_3d_path = os.path.join(save_dir, "3d_overview.png")
    fig_3d.savefig(fig_3d_path, dpi=150)
    print(f"[SAVE] {fig_3d_path}")
    fig_3d_cv = figure_to_bgr(fig_3d)
    fig_3d_cv_path = os.path.join(save_dir, "base_frame_overview_3d_cv.png")
    cv2.imwrite(fig_3d_cv_path, fig_3d_cv)
    print(f"[SAVE] {fig_3d_cv_path}")

    if args.no_show:
        print("\n[DONE] Verification complete")
        return

    # ─── Visualize ───
    fig_err = visualize_errors(cross_err, reproj_err, he_err)
    fig_err.savefig(os.path.join(save_dir, "error_histograms.png"), dpi=150)
    print(f"[SAVE] {os.path.join(save_dir, 'error_histograms.png')}")

    print("\n" + "=" * 60)
    print("[VIS] Cube Candidate Diagnostics")
    print("=" * 60)
    cand_rows = collect_cube_candidate_diagnostics(
        meta, transforms, args.intrinsics_dir, root, gripper_cam_idx, all_cam_ids,
        cube_cfg=cube_cfg, include_meta=include_meta_candidates,
        selection_profile=args.cube_selection_profile)
    if cand_rows:
        selected = [r for r in cand_rows if r["selected"]]
        accepted = [r for r in selected if r["accepted"]]
        print(f"  Candidates: {len(cand_rows)} total, {len(selected)} selected, {len(accepted)} accepted")

        fig_scatter = visualize_cube_candidate_scatter(cand_rows)
        if fig_scatter is not None:
            scatter_path = os.path.join(save_dir, "cube_candidate_scatter.png")
            fig_scatter.savefig(scatter_path, dpi=150)
            print(f"[SAVE] {scatter_path}")

        fig_health = visualize_marker_health(cand_rows)
        if fig_health is not None:
            health_path = os.path.join(save_dir, "cube_marker_health.png")
            fig_health.savefig(health_path, dpi=150)
            print(f"[SAVE] {health_path}")

        example_paths = visualize_candidate_examples(meta, cand_rows, save_dir)
        for p in example_paths:
            print(f"[SAVE] {p}")
    else:
        print("  [SKIP] No candidate diagnostics available")

    print("\n" + "=" * 60)
    print("[VIS] Marker Override Diagnostics")
    print("=" * 60)
    override_report = collect_marker_override_diagnostics(
        meta, transforms, args.intrinsics_dir, root, gripper_cam_idx, all_cam_ids, cube_cfg=cube_cfg)
    if override_report:
        out_json = os.path.join(save_dir, "cube_override_diagnostic.json")
        with open(out_json, "w") as f:
            json.dump({str(k): v for k, v in override_report.items()}, f, indent=2)
        print(f"[SAVE] {out_json}")

        fig_override = visualize_marker_override_summary(override_report)
        if fig_override is not None:
            override_path = os.path.join(save_dir, "cube_override_summary.png")
            fig_override.savefig(override_path, dpi=150)
            print(f"[SAVE] {override_path}")

        for mid in sorted(override_report):
            row = override_report[mid]
            print(
                f"  id{mid}: current={row['current']['face']}/{row['current']['corner_permutation']} "
                f"rank={row['current']['rank']} -> best={row['best']['face']}/{row['best']['corner_permutation']} "
                f"delta={0.0 if row['score_improvement'] is None else row['score_improvement']:.1f}")
            fig_gallery = render_marker_gallery(row)
            if fig_gallery is not None:
                gallery_path = os.path.join(save_dir, f"marker_id{mid}_gallery.png")
                fig_gallery.savefig(gallery_path, dpi=150)
                plt.close(fig_gallery)
                print(f"[SAVE] {gallery_path}")
    else:
        print("  [SKIP] No override diagnostics available")

    if selected_pages:
        show_cv_pages("Selected event images", selected_pages)
    show_cv_pages("Base frame overview", [base_overview_cv])
    show_cv_pages("Base frame overview 3D", [fig_3d_cv])

    if not args.no_show:
        plt.show()

    print("\n[DONE] Verification complete")


if __name__ == "__main__":
    main()
