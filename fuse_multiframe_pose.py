#!/usr/bin/env python3
"""
Multi-frame point cloud fusion + auto-tuned single ICP for object pose.

19 frame × 4 cam = 76 (mask, depth) 쌍에서 객체별 점군을 base 좌표계로 모은 뒤
GLB 모델과 한 번의 ICP로 정합하여 단일 안정 포즈 산출.

자동 튜닝:
  vote_ratio / voxel_mm / max_corr_mm 조합을 순회하며 객체별 최고 점수 채택.
  점수 = (fitness / (rmse + 0.001)) - extent_excess_penalty
  목표 조건 (fit≥0.95, rmse≤3mm) 충족 시 즉시 다음 객체로.

Comparison.png:
  각 cam의 대표 frame(중간 frame_009) 위에 변환된 GLB 메쉬 wireframe을 overlay.
  4-cam quad 이미지 + 객체별 단일 이미지 생성.

전제: data/pose_out_all/frame_*/sam_masks/*/*.png (per-frame SAM 마스크 이미 생성됨)

사용:
  python3 scripts/fuse_multiframe_pose.py
  python3 scripts/fuse_multiframe_pose.py --objects red,cream
  python3 scripts/fuse_multiframe_pose.py --quick   # 적은 trial만
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

DATA_DIR = Path("data")
CAPTURE_DIR = DATA_DIR / "object_capture"
SAM_ROOT = DATA_DIR / "pose_out_all"
INTR_DIR = Path("intrinsics")
CALIB_DIR = DATA_DIR / "session" / "calib_out"

GOAL_FIT = 0.95
GOAL_RMSE_MM = 3.0


# ────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────

def load_intrinsics():
    out = {}
    for ci in [0, 1, 2, 3]:
        npz = np.load(INTR_DIR / f"cam{ci}.npz", allow_pickle=True)
        K = npz["color_K"].astype(np.float64)
        D = npz["color_D"].astype(np.float64)
        depth_scale = float(npz.get("depth_scale", 0.001))
        out[ci] = (K, D, depth_scale)
    return out


def load_static_transforms():
    T_base_C0 = np.load(CALIB_DIR / "T_base_C0.npy")
    T_C0_C1 = np.load(CALIB_DIR / "T_C0_C1.npy")
    T_C0_C3 = np.load(CALIB_DIR / "T_C0_C3.npy")
    T_gripper_cam = np.load(CALIB_DIR / "T_gripper_cam.npy")
    return {
        0: T_base_C0,
        1: T_base_C0 @ T_C0_C1,
        3: T_base_C0 @ T_C0_C3,
    }, T_gripper_cam


# ────────────────────────────────────────────────────────────────────
# Point cloud building
# ────────────────────────────────────────────────────────────────────

def backproject(depth_u16, mask, K, depth_scale):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 3))
    d = depth_u16[ys, xs].astype(np.float64) * depth_scale
    valid = (d > 0.05) & (d < 1.5)
    xs, ys, d = xs[valid], ys[valid], d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    Z = d
    return np.stack([X, Y, Z], axis=1)


def transform_pts(pts, T):
    if len(pts) == 0:
        return pts
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def vote_static_mask(obj_name, cam_idx, n_frames, vote_ratio):
    masks = []
    for frame in range(n_frames):
        fid = f"{frame:06d}"
        mp = SAM_ROOT / f"frame_{fid}" / "sam_masks" / fid / f"{obj_name}_cam{cam_idx}.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        masks.append((m > 0).astype(np.uint8))
    if not masks:
        return None, 0
    vote = np.stack(masks, axis=0).sum(axis=0)
    voted = (vote >= max(1, int(vote_ratio * len(masks)))).astype(np.uint8) * 255
    return voted, len(masks)


def collect_fused_cloud(obj_name, intrinsics, static_T_base_cam, T_gripper_cam,
                        n_frames=19, vote_ratio=0.5, verbose=False):
    all_pts = []
    n_used = 0
    n_skipped = 0
    static_voted = {}
    for cam_idx in [0, 1, 3]:
        voted, n_m = vote_static_mask(obj_name, cam_idx, n_frames, vote_ratio)
        if voted is not None and voted.sum() > 50:
            static_voted[cam_idx] = voted
            if verbose:
                print(f"    cam{cam_idx} voted: {(voted>0).sum()} px / {n_m} masks")

    for frame in range(n_frames):
        fid = f"{frame:06d}"
        for cam_idx in [0, 1, 2, 3]:
            if cam_idx == 2:
                mp = SAM_ROOT / f"frame_{fid}" / "sam_masks" / fid / f"{obj_name}_cam{cam_idx}.png"
                if not mp.exists():
                    n_skipped += 1
                    continue
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            else:
                mask = static_voted.get(cam_idx)
            if mask is None or mask.sum() < 50:
                n_skipped += 1
                continue

            depth_path = CAPTURE_DIR / f"cam{cam_idx}" / f"depth_{fid}.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                n_skipped += 1
                continue

            K, _, depth_scale = intrinsics[cam_idx]
            pts_cam = backproject(depth, mask, K, depth_scale)
            if len(pts_cam) < 20:
                n_skipped += 1
                continue

            if cam_idx == 2:
                T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
                if not T_be_path.exists():
                    n_skipped += 1
                    continue
                T_be = np.load(T_be_path)
                T_base_cam = T_be @ T_gripper_cam
            else:
                T_base_cam = static_T_base_cam[cam_idx]

            pts_base = transform_pts(pts_cam, T_base_cam)
            all_pts.append(pts_base)
            n_used += 1

    if not all_pts:
        return np.zeros((0, 3)), n_used, n_skipped
    return np.vstack(all_pts), n_used, n_skipped


def collect_depth_above_table_cloud(intrinsics, static_T_base_cam, T_gripper_cam,
                                     center_xy_mm, half_extent_xy_mm,
                                     z_min_mm=5.0, z_max_mm=250.0,
                                     n_frames=19, depth_stride=4):
    """SAM 마스크 없이 모든 depth 점을 base 좌표로 변환 → 테이블 위 + xy bbox로 필터."""
    cx, cy = center_xy_mm
    hx, hy = half_extent_xy_mm
    all_pts = []
    n_used = 0
    for frame in range(n_frames):
        fid = f"{frame:06d}"
        for cam_idx in [0, 1, 2, 3]:
            depth_path = CAPTURE_DIR / f"cam{cam_idx}" / f"depth_{fid}.png"
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if depth_stride > 1:
                depth = depth[::depth_stride, ::depth_stride]
            K, _, depth_scale = intrinsics[cam_idx]
            # downsampled K
            Ks = K.copy()
            Ks[0, 0] /= depth_stride; Ks[1, 1] /= depth_stride
            Ks[0, 2] /= depth_stride; Ks[1, 2] /= depth_stride
            mask = (depth > 0).astype(np.uint8)
            pts_cam = backproject(depth, mask, Ks, depth_scale)
            if len(pts_cam) < 50:
                continue
            if cam_idx == 2:
                T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
                if not T_be_path.exists():
                    continue
                T_be = np.load(T_be_path)
                T_base_cam = T_be @ T_gripper_cam
            else:
                T_base_cam = static_T_base_cam[cam_idx]
            pts_base_m = transform_pts(pts_cam, T_base_cam)
            pts_base_mm = pts_base_m * 1000
            # z filter: above table, below max height
            zm = pts_base_mm[:, 2]
            inz = (zm > z_min_mm) & (zm < z_max_mm)
            # xy filter
            xm, ym = pts_base_mm[:, 0], pts_base_mm[:, 1]
            inxy = (np.abs(xm - cx) <= hx) & (np.abs(ym - cy) <= hy)
            keep = inz & inxy
            if keep.sum() < 20:
                continue
            all_pts.append(pts_base_m[keep])
            n_used += 1
    if not all_pts:
        return np.zeros((0, 3)), n_used
    return np.vstack(all_pts), n_used


def find_clusters(pcd, eps=0.015, min_points=50):
    """Return list of (cluster_pcd, centroid, extent_mm) sorted by size desc."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    out = []
    pts = np.asarray(pcd.points)
    for lab in np.unique(labels):
        if lab < 0:
            continue
        sel = labels == lab
        cp = pts[sel]
        if len(cp) < min_points:
            continue
        cpcd = o3d.geometry.PointCloud()
        cpcd.points = o3d.utility.Vector3dVector(cp)
        ext = (cp.max(0) - cp.min(0)) * 1000
        out.append({"pcd": cpcd, "centroid": cp.mean(0), "extent_mm": ext, "n": len(cp)})
    out.sort(key=lambda d: -d["n"])
    return out


def clean_cloud(pts_np, voxel_m=0.003, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_m)
    return pcd


def keep_largest_cluster(pcd, eps=0.015, min_points=30):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.max() < 0:
        return pcd
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    biggest = unique[np.argmax(counts)]
    keep = labels == biggest
    pts = np.asarray(pcd.points)[keep]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    return out


# ────────────────────────────────────────────────────────────────────
# Model + ICP
# ────────────────────────────────────────────────────────────────────

def load_glb_as_pcd(glb_path, n_samples=8000, center=True, prescale=1.0):
    """GLB를 점군으로 변환. center=True면 객체 bbox 중심을 원점으로 이동
    (이렇게 해야 ICP 결과 T_base_object가 블록 중심 좌표가 됨).
    prescale: scalar 면 uniform 스케일, (3,) 배열이면 anisotropic per-axis 스케일.
    """
    m = trimesh.load(str(glb_path), force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if center:
        offset = m.bounds.mean(axis=0)
        m.apply_translation(-offset)
    if hasattr(prescale, "__len__") and len(prescale) == 3:
        # Anisotropic scaling: trimesh apply_scale 은 vector 입력시 per-axis scale
        m.apply_scale(np.asarray(prescale, dtype=np.float64))
    elif float(prescale) != 1.0:
        m.apply_scale(float(prescale))
    pts, _ = trimesh.sample.sample_surface_even(m, n_samples)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20))
    return pcd, m


def estimate_object_dimensions_from_cluster(target_pcd):
    """클러스터의 oriented bbox 크기 (sorted desc, mm). 블록 실측 추정에 사용."""
    pts_arr = np.asarray(target_pcd.points)
    if len(pts_arr) < 30:
        return None
    try:
        obb = target_pcd.get_oriented_bounding_box()
        extent_mm = np.asarray(obb.extent) * 1000
    except Exception:
        # OBB 실패 시 axis-aligned bbox로 fallback
        extent_mm = (pts_arr.max(0) - pts_arr.min(0)) * 1000
    return sorted([float(e) for e in extent_mm], reverse=True)


def auto_estimate_block_real_size(obj_name, table_v_threshold,
                                    intrinsics, static_T_base_cam, T_gripper_cam,
                                    n_frames=19,
                                    cam_ids=(0, 1, 2, 3)):
    """76-view fused cloud + cluster → real block dimensions (mm, sorted desc).
    GLB를 ICP scale 학습에 맡기지 않고, 관측 점군에서 직접 객체 크기를 추정.
    """
    pts_np, _ = collect_table_aware_cloud(
        obj_name, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
        n_frames=n_frames, gripper_weight=1,  # weight 없이 raw 분포 확인
        cam_ids=cam_ids,
        require_unambiguous_mask=False,  # 좀 더 관대히 — 크기 추정만 목적
        top2_ratio=0.95)
    if len(pts_np) < 300:
        return None
    target_pcd = clean_cloud(pts_np, voxel_m=0.002)
    target_pcd = keep_largest_cluster(target_pcd, eps=0.012, min_points=50)
    if len(target_pcd.points) < 80:
        return None
    return estimate_object_dimensions_from_cluster(target_pcd)


def initial_T(model_pcd, target_pcd):
    c_m = np.asarray(model_pcd.points).mean(axis=0)
    c_t = np.asarray(target_pcd.points).mean(axis=0)
    T = np.eye(4)
    T[:3, 3] = c_t - c_m
    return T


def run_icp_multi_init(model_pcd, target_pcd, T_init,
                       max_corr_coarse, max_corr_fine, yaw_grid=24):
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    best = None
    best_score = -np.inf
    for k in range(yaw_grid):
        ang = 2 * np.pi * k / yaw_grid
        Rz = np.eye(4)
        Rz[:2, :2] = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
        T0 = T_init @ Rz
        r1 = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, max_corr_coarse, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, 60))
        r2 = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, max_corr_fine, r1.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-9, 1e-9, 120))
        score = r2.fitness / (r2.inlier_rmse + 1e-4)
        if score > best_score:
            best_score = score
            best = r2
    return best


# ────────────────────────────────────────────────────────────────────
# Auto-tune
# ────────────────────────────────────────────────────────────────────

def evaluate_combo(obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                   vote_ratio, voxel_mm, max_corr_mm, dbscan_eps_mm,
                   n_frames=19, n_samples=8000, verbose=False):
    pts_np, n_used, n_skipped = collect_fused_cloud(
        obj, intrinsics, static_T_base_cam, T_gripper_cam,
        n_frames=n_frames, vote_ratio=vote_ratio, verbose=verbose)
    if len(pts_np) < 100:
        return None
    target_pcd = clean_cloud(pts_np, voxel_m=voxel_mm / 1000)
    target_pcd = keep_largest_cluster(target_pcd, eps=dbscan_eps_mm / 1000, min_points=30)
    target_arr = np.asarray(target_pcd.points)
    if len(target_arr) < 50:
        return None
    ext = (target_arr.max(0) - target_arr.min(0)) * 1000  # mm
    model_pcd, mesh = load_glb_as_pcd(glb_path, n_samples=n_samples)
    T0 = initial_T(model_pcd, target_pcd)
    result = run_icp_multi_init(
        model_pcd, target_pcd, T0,
        max_corr_coarse=max_corr_mm * 4 / 1000,
        max_corr_fine=max_corr_mm / 1000,
        yaw_grid=24)
    # extent excess: 모델 대비 관측 extent가 1.6배 넘으면 패널티
    model_ext = (np.asarray(model_pcd.points).max(0) - np.asarray(model_pcd.points).min(0)) * 1000
    ratio = max(ext) / (max(model_ext) + 1e-4)
    extent_penalty = max(0, ratio - 1.6) * 50  # 1.6배 초과시 비례 패널티
    score = result.fitness / (result.inlier_rmse + 0.001) - extent_penalty
    return {
        "params": {"vote_ratio": vote_ratio, "voxel_mm": voxel_mm,
                   "max_corr_mm": max_corr_mm, "dbscan_eps_mm": dbscan_eps_mm},
        "T_base_object": result.transformation,
        "fit": float(result.fitness),
        "rmse_mm": float(result.inlier_rmse * 1000),
        "extent_mm": [float(x) for x in ext],
        "model_extent_mm": [float(x) for x in model_ext],
        "extent_ratio": float(ratio),
        "score": float(score),
        "n_views_used": n_used,
        "n_points_clean": int(len(target_arr)),
        "target_pcd": target_pcd,
        "mesh": mesh,
        "extent_penalty": float(extent_penalty),
    }


def detect_table_v_threshold(n_frames=19, sample_frames=(0, 5, 10, 15)):
    """검정/회색 테이블 자동 감지. dominant V (가장 흔한 밝기) 위 1.5σ를
    전경/배경 임계값으로 사용. 정적 cam만 샘플링 (배경 일관).
    """
    samples = []
    for ci in [0, 1, 3]:
        for fi in sample_frames:
            p = CAPTURE_DIR / f"cam{ci}" / f"rgb_{fi:06d}.jpg"
            rgb = cv2.imread(str(p))
            if rgb is None:
                continue
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            V = hsv[:, :, 2].flatten()
            samples.append(V[::15])
    if not samples:
        return 80.0, 0.0  # fallback
    samples = np.concatenate(samples)
    hist, edges = np.histogram(samples, bins=48)
    dom_idx = int(np.argmax(hist))
    dom_v = float((edges[dom_idx] + edges[dom_idx + 1]) / 2)
    # std within ±25 of dominant V
    near = samples[np.abs(samples - dom_v) < 25]
    dom_std = float(near.std()) if len(near) > 5 else 10.0
    v_thr = dom_v + max(20.0, 1.5 * dom_std)
    return v_thr, dom_v


# 객체별 기본 color prior — configs/objects/<obj>.json 가 없을 때 fallback.
# 이렇게 코드 안에 내장하면 configs 폴더가 사라져도 파이프라인이 동작.
_DEFAULT_COLOR_SPECS = {
    "red":   {"enabled": True,  "hue_ref": 2.0,  "hue_radius": 7.0,
              "s_min": 150, "s_max": 255, "v_min": 80,  "v_max": 230},
    "cream": {"enabled": True,  "hue_ref": 22.0, "hue_radius": 8.0,
              "s_min": 50,  "s_max": 200, "v_min": 200, "v_max": 255},
    "blue":  {"enabled": True,  "hue_ref": 88.0, "hue_radius": 9.0,
              "s_min": 80,  "s_max": 255, "v_min": 140, "v_max": 240},
    "box":   {"enabled": False, "hue_ref": 0.0,  "hue_radius": 180.0,
              "s_min": 0,   "s_max": 25,  "v_min": 180, "v_max": 255},
}


def load_obj_color_spec(obj_name):
    """configs/objects/<obj>.json 의 color_prior 파싱. 파일 없으면 코드 내장 default 사용."""
    cfg_path = Path("configs/objects") / f"{obj_name}.json"
    if cfg_path.exists():
        try:
            cfg = json.load(open(cfg_path))
            cp = cfg.get("color_prior", {})
        except Exception:
            cp = {}
    else:
        cp = _DEFAULT_COLOR_SPECS.get(obj_name, {})
    # 모든 필드 default fallback 까지 적용
    default = _DEFAULT_COLOR_SPECS.get(obj_name, {})
    return {
        "enabled": bool(cp.get("enabled", default.get("enabled", True))),
        "hue_ref": float(cp.get("hue_ref", default.get("hue_ref", 0))),
        "hue_radius": float(cp.get("hue_radius", default.get("hue_radius", 12))),
        "s_min": int(cp.get("s_min", default.get("s_min", 0))),
        "s_max": int(cp.get("s_max", default.get("s_max", 255))),
        "v_min": int(cp.get("v_min", default.get("v_min", 0))),
        "v_max": int(cp.get("v_max", default.get("v_max", 255))),
    }


def table_aware_mask(rgb_bgr, color_spec, table_v_threshold, margin=10):
    """전경 (table_v_threshold 위) ∩ 객체 색 prior."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(np.int32), hsv[:, :, 1], hsv[:, :, 2]
    fg = V > table_v_threshold
    if color_spec["enabled"] and color_spec["hue_radius"] < 90:
        # 일반 컬러 객체: hue 거리 + S/V 범위
        href = color_spec["hue_ref"]
        hrad = color_spec["hue_radius"]
        h_dist = np.minimum(np.abs(H - href), 180 - np.abs(H - href))
        obj = ((h_dist <= hrad) &
               (S >= color_spec["s_min"]) & (S <= color_spec["s_max"]) &
               (V >= color_spec["v_min"]) & (V <= color_spec["v_max"]))
    else:
        # 흰색 (hue 무관): S 낮고 V 높은 픽셀
        obj = ((S < color_spec["s_max"]) &
               (V >= color_spec["v_min"]) & (V <= color_spec["v_max"]))
    mask = (fg & obj).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def is_unambiguous_mask(mask, min_size=200, top2_ratio=0.7):
    """마스크가 명확한 단일 객체인지 검사.
    - 최대 connected component가 충분히 크고 (>min_size)
    - 두 번째 큰 component가 top1의 top2_ratio 이하여야 통과 (= 단일 객체).
    """
    if mask.sum() < min_size:
        return False, 0, 0
    binmask = (mask > 0).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binmask, 8)
    if n_labels < 2:
        return False, 0, 0
    sizes = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)],
                   reverse=True)
    top1 = sizes[0]
    top2 = sizes[1] if len(sizes) >= 2 else 0
    if top1 < min_size:
        return False, top1, top2
    # 두 번째 cluster가 top1의 ratio 넘으면 ambiguous
    if top2 > top1 * top2_ratio:
        return False, top1, top2
    return True, top1, top2


def collect_table_aware_cloud(obj_name, table_v_threshold,
                               intrinsics, static_T_base_cam, T_gripper_cam,
                               z_min_mm=5.0, z_max_mm=300.0, n_frames=19,
                               gripper_weight=20,
                               workspace_bbox_mm=((-500, 300), (150, 900)),
                               require_unambiguous_mask=True,
                               min_mask_size=200, top2_ratio=0.7,
                               cam_ids=(0, 1, 2, 3),
                               frame_subset=None):
    """gripper_weight: gripper cam(cam2) 점을 N배 복제하여 ICP 가중치 효과.
    workspace_bbox_mm: ((x_min, x_max), (y_min, y_max)) — 점군에서 작업면 외 제외.
    require_unambiguous_mask: top2 component 비율이 top2_ratio 이하인 frame만 사용.
    frame_subset: 사용할 frame index list. None 이면 range(n_frames) 전체.
    """
    color_spec = load_obj_color_spec(obj_name)
    all_pts = []
    n_used = 0
    n_gripper_views = 0
    n_static_views = 0
    n_skipped_ambiguous = 0
    (xmin, xmax), (ymin, ymax) = workspace_bbox_mm
    frames_iter = frame_subset if frame_subset is not None else range(n_frames)
    for frame in frames_iter:
        fid = f"{frame:06d}"
        for ci in cam_ids:
            rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
            if rgb is None:
                continue
            mask = table_aware_mask(rgb, color_spec, table_v_threshold)
            if mask.sum() < min_mask_size:
                continue
            # ── B: 마스크 단일성 검사 — 모호하면 해당 (frame,cam) 스킵
            if require_unambiguous_mask:
                ok, t1, t2 = is_unambiguous_mask(mask, min_mask_size, top2_ratio)
                if not ok:
                    n_skipped_ambiguous += 1
                    continue
            depth = cv2.imread(
                str(CAPTURE_DIR / f"cam{ci}" / f"depth_{fid}.png"),
                cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            K, _, depth_scale = intrinsics[ci]
            pts_cam = backproject(depth, mask, K, depth_scale)
            if len(pts_cam) < 20:
                continue
            if ci == 2:
                T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
                if not T_be_path.exists():
                    continue
                T_be = np.load(T_be_path)
                T_base_cam = T_be @ T_gripper_cam
            else:
                T_base_cam = static_T_base_cam[ci]
            pts_base = transform_pts(pts_cam, T_base_cam)
            pts_mm = pts_base * 1000
            zm = pts_mm[:, 2]
            xm = pts_mm[:, 0]
            ym = pts_mm[:, 1]
            # ── D: 워크스페이스 bbox 필터
            in_ws = ((zm > z_min_mm) & (zm < z_max_mm)
                     & (xm >= xmin) & (xm <= xmax)
                     & (ym >= ymin) & (ym <= ymax))
            if in_ws.sum() < 20:
                continue
            kept = pts_base[in_ws]
            # ── C: gripper cam 가중치만큼 복제 (default 20×)
            if ci == 2 and gripper_weight > 1:
                kept = np.tile(kept, (gripper_weight, 1))
                n_gripper_views += 1
            elif ci == 2:
                n_gripper_views += 1
            else:
                n_static_views += 1
            all_pts.append(kept)
            n_used += 1
    if not all_pts:
        return np.zeros((0, 3)), n_used
    print(f"    views: gripper={n_gripper_views} (weight×{gripper_weight}), "
          f"static={n_static_views}, skipped_ambig={n_skipped_ambiguous}")
    return np.vstack(all_pts), n_used


def _extract_scale_from_T(T):
    """Recover uniform scale baked into the 3x3 by ICP `with_scaling=True`."""
    return float(np.linalg.det(T[:3, :3]) ** (1.0 / 3.0))


def _strip_scale_from_T(T):
    """Replace scaled 3x3 with closest pure rotation (SVD). Translation untouched."""
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_pure = U @ Vt
    if np.linalg.det(R_pure) < 0:
        U[:, -1] *= -1
        R_pure = U @ Vt
    Tn = T.copy()
    Tn[:3, :3] = R_pure
    return Tn


def icp_with_scale(model_pcd, target_pcd, T_init,
                    corr_mm_schedule=(20, 10, 5, 3), yaw_grid=24,
                    with_scaling=True,
                    scale_lo=0.7, scale_hi=1.3):
    """ICP with optional uniform scaling.
    Scale collapse 차단: 학습된 scale 이 [scale_lo, scale_hi] 밖이면 그 yaw seed는
    rigid-only 로 재계산해서 사용. 이렇게 하면 degenerate scale 결과가 best 로 채택되지 않음.
    """
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    corr0 = corr_mm_schedule[0] / 1000
    best_T = None
    best_score = -np.inf
    best_result = None
    for k in range(yaw_grid):
        ang = 2 * np.pi * k / yaw_grid
        Rz = np.eye(4)
        Rz[:2, :2] = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
        T0 = T_init @ Rz
        r = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, corr0, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(
                with_scaling=with_scaling),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, 60))
        # Scale guard: 범위 벗어나면 rigid-only 로 재계산
        if with_scaling:
            s = _extract_scale_from_T(r.transformation)
            if s < scale_lo or s > scale_hi:
                r = o3d.pipelines.registration.registration_icp(
                    model_pcd, target_pcd, corr0, T0,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        with_scaling=False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, 60))
        sc = r.fitness / (r.inlier_rmse + 1e-4)
        if sc > best_score:
            best_score = sc
            best_T = r.transformation
            best_result = r
    for corr_mm in corr_mm_schedule[1:]:
        r = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, corr_mm / 1000, best_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-9, 1e-9, 150))
        # Point-to-plane은 scale 변경 안 하지만, 추가 안전 가드
        s = _extract_scale_from_T(r.transformation)
        if abs(s - 1.0) > 0.01:
            r_t = r.transformation.copy()
            r_t[:3, :3] = r_t[:3, :3] / s  # normalize back to scale=1
        best_T = r.transformation
        best_result = r
    return best_result


def multiscale_icp(model_pcd, target_pcd, T_init,
                    corr_mm_schedule=(20, 10, 5, 3), yaw_grid=24):
    """Coarse → fine ICP. yaw grid는 첫 단계에서만, 이후는 그 결과를 init으로."""
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    # 1단계: yaw grid 탐색
    corr0 = corr_mm_schedule[0] / 1000
    best_T = None
    best_score = -np.inf
    best_result = None
    for k in range(yaw_grid):
        ang = 2 * np.pi * k / yaw_grid
        Rz = np.eye(4)
        Rz[:2, :2] = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
        T0 = T_init @ Rz
        r = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, corr0, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, 60))
        sc = r.fitness / (r.inlier_rmse + 1e-4)
        if sc > best_score:
            best_score = sc
            best_T = r.transformation
            best_result = r
    # 후속 단계: point-to-plane, 점차 좁히기
    for corr_mm in corr_mm_schedule[1:]:
        r = o3d.pipelines.registration.registration_icp(
            model_pcd, target_pcd, corr_mm / 1000, best_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(1e-9, 1e-9, 150))
        # 다음 단계 init으로 갱신
        best_T = r.transformation
        best_result = r
    return best_result


def estimate_pose_single_view(obj, frame_id, cam_id, table_v_threshold,
                                intrinsics, static_T_base_cam, T_gripper_cam,
                                glb_prescale=1.0,
                                voxel_mm=1.5, max_corr_mm=8.0):
    """단일 (frame, cam) point cloud 에서 객체 포즈 추정.
    19개 cam2 frame 각각이 독립 view → 독립 추정. 이후 robust median 으로 ensemble.
    """
    color_spec = load_obj_color_spec(obj)
    fid = f"{frame_id:06d}"
    rgb = cv2.imread(str(CAPTURE_DIR / f"cam{cam_id}" / f"rgb_{fid}.jpg"))
    if rgb is None:
        return None
    mask = table_aware_mask(rgb, color_spec, table_v_threshold)
    if mask.sum() < 200:
        return None
    depth = cv2.imread(str(CAPTURE_DIR / f"cam{cam_id}" / f"depth_{fid}.png"),
                        cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    K, _, depth_scale = intrinsics[cam_id]
    pts_cam = backproject(depth, mask, K, depth_scale)
    if len(pts_cam) < 30:
        return None
    # cam → base
    if cam_id == 2:
        T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
        if not T_be_path.exists():
            return None
        T_be = np.load(T_be_path)
        T_base_cam = T_be @ T_gripper_cam
    else:
        T_base_cam = static_T_base_cam[cam_id]
    pts_base = transform_pts(pts_cam, T_base_cam)
    pts_mm = pts_base * 1000
    inz = (pts_mm[:, 2] > 5) & (pts_mm[:, 2] < 300)
    if inz.sum() < 30:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_base[inz])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_mm / 1000)
    if len(pcd.points) < 30:
        return None
    # 단일 view 라 cluster 1개만 있을 확률 높음 → 그대로 사용
    glb_path = DATA_DIR / f"{obj}.glb"
    model_pcd, _ = load_glb_as_pcd(glb_path, n_samples=4000,
                                    center=True, prescale=glb_prescale)
    T0 = initial_T(model_pcd, pcd)
    result = icp_with_scale(model_pcd, pcd, T0,
                             corr_mm_schedule=(15, 8, 4), yaw_grid=18,
                             with_scaling=True, scale_lo=0.7, scale_hi=1.3)
    # 매우 낮은 가드만 — fit 은 ensemble 에서 filter
    if result.inlier_rmse < 0.0002:
        return None
    if result.fitness < 0.05:  # 거의 매칭 없음
        return None
    return {
        "T": result.transformation,
        "fit": float(result.fitness),
        "rmse_mm": float(result.inlier_rmse * 1000),
        "n_pts": int(len(pcd.points)),
        "frame": int(frame_id),
        "cam": int(cam_id),
    }


def robust_se3_median(T_list, weights=None, max_mad_factor=2.5):
    """SE3 robust median.
    - Translation: 위치 component 의 element-wise median + MAD outlier rejection
    - Rotation: scipy Rotation.mean (quaternion-based)
    """
    if not T_list:
        return None
    Ts = np.asarray(T_list, dtype=np.float64)
    ts = Ts[:, :3, 3]
    t_med = np.median(ts, axis=0)
    dists = np.linalg.norm(ts - t_med, axis=1)
    mad = float(np.median(np.abs(dists - np.median(dists)))) + 1e-9
    keep = dists < (np.median(dists) + max_mad_factor * 1.4826 * mad)
    if keep.sum() < max(2, len(T_list) // 3):
        keep = np.ones(len(T_list), dtype=bool)  # fallback: keep all if too few
    Ts_kept = Ts[keep]

    # Translation: median of kept
    t_final = np.median(Ts_kept[:, :3, 3], axis=0)

    # Rotation: quaternion-based mean (scipy)
    from scipy.spatial.transform import Rotation as _R
    Rs = Ts_kept[:, :3, :3]
    # 각 R의 scale 제거 (rotation pure)
    Rs_pure = []
    for R in Rs:
        U, _, Vt = np.linalg.svd(R)
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0:
            U[:, -1] *= -1
            Rp = U @ Vt
        Rs_pure.append(Rp)
    try:
        rot_mean = _R.from_matrix(np.asarray(Rs_pure)).mean().as_matrix()
    except Exception:
        rot_mean = Rs_pure[0]
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = rot_mean
    T_out[:3, 3] = t_final
    return T_out, {
        "n_total": len(T_list), "n_kept": int(keep.sum()),
        "mad_mm": float(mad * 1000),
    }


def _estimate_pose_from_cloud(pts_np, glb_path, glb_prescale,
                                voxel_mm=2.0, max_corr_mm=10.0,
                                yaw_grid=18):
    """단일 fused 점군에 대해 ICP → (T, fit, rmse_mm, n_pts) 또는 None."""
    if len(pts_np) < 80:
        return None
    target_pcd = clean_cloud(pts_np, voxel_m=voxel_mm / 1000)
    target_pcd = keep_largest_cluster(target_pcd, eps=0.012, min_points=30)
    if len(target_pcd.points) < 40:
        return None
    model_pcd, _ = load_glb_as_pcd(glb_path, n_samples=4000, center=True,
                                    prescale=glb_prescale)
    T0 = initial_T(model_pcd, target_pcd)
    res = icp_with_scale(model_pcd, target_pcd, T0,
                          corr_mm_schedule=(15, 8, 4), yaw_grid=yaw_grid,
                          with_scaling=True, scale_lo=0.7, scale_hi=1.3)
    if res.inlier_rmse < 0.0002 or res.fitness < 0.05:
        return None
    return {
        "T": res.transformation,
        "fit": float(res.fitness),
        "rmse_mm": float(res.inlier_rmse * 1000),
        "n_pts": int(len(target_pcd.points)),
    }


def evaluate_per_frame_ensemble(obj, table_v_threshold,
                                 intrinsics, static_T_base_cam, T_gripper_cam,
                                 glb_path, n_frames=19,
                                 glb_prescale=1.0,
                                 min_fit=0.15,
                                 n_batches=4,
                                 verbose=True):
    """Cam2 frame batch K개 + 정적 cam 3개 → robust SE3 median.
    각 batch (≈5 frame) 를 fused cloud 로 만들어 ICP. 단일 frame 보다 dense 해서
    ICP 가 안정적이고, batch 간 분산으로 view-diversity 활용.
    """
    estimates = []
    # cam2 batches
    batch_frames = np.array_split(np.arange(n_frames), n_batches)
    for bi, frames in enumerate(batch_frames):
        if len(frames) == 0:
            continue
        pts_np, _ = collect_table_aware_cloud(
            obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
            n_frames=n_frames, gripper_weight=1,
            cam_ids=(2,), frame_subset=list(frames))
        r = _estimate_pose_from_cloud(pts_np, glb_path, glb_prescale)
        if r is not None and r["fit"] >= min_fit:
            r["cam"] = 2
            r["batch"] = bi
            estimates.append(r)
            if verbose:
                print(f"     cam2 batch{bi} frames={list(frames)} "
                      f"fit={r['fit']:.3f} rmse={r['rmse_mm']:5.2f}mm n={r['n_pts']}")
    # static cams: each cam fuses all frames (same view but cleaner)
    for ci in [0, 1, 3]:
        pts_np, _ = collect_table_aware_cloud(
            obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
            n_frames=n_frames, gripper_weight=1, cam_ids=(ci,))
        r = _estimate_pose_from_cloud(pts_np, glb_path, glb_prescale)
        if r is not None and r["fit"] >= min_fit:
            r["cam"] = ci
            estimates.append(r)
            if verbose:
                print(f"     cam{ci} fit={r['fit']:.3f} rmse={r['rmse_mm']:5.2f}mm "
                      f"n={r['n_pts']}")
    if verbose:
        n_c2 = sum(1 for e in estimates if e['cam'] == 2)
        n_st = sum(1 for e in estimates if e['cam'] != 2)
        print(f"     ensemble: {len(estimates)} estimates "
              f"(cam2 batches: {n_c2}, static cams: {n_st})")
    if len(estimates) < 3:
        if verbose:
            print(f"     [ensemble] not enough estimates ({len(estimates)} < 3), skipping")
        return None

    # Robust median
    Ts = [e["T"] for e in estimates]
    T_med, diag = robust_se3_median(Ts)

    # 후처리: pure rotation (scale 제거)
    R_med = T_med[:3, :3]
    U, _, Vt = np.linalg.svd(R_med)
    R_pure = U @ Vt
    if np.linalg.det(R_pure) < 0:
        U[:, -1] *= -1
        R_pure = U @ Vt
    T_med[:3, :3] = R_pure  # rigid SE3

    # 평균 RMSE / fit
    rmse_mean = float(np.mean([e["rmse_mm"] for e in estimates]))
    fit_mean = float(np.mean([e["fit"] for e in estimates]))
    glb_path_p = Path(glb_path)
    model_pcd, mesh = load_glb_as_pcd(glb_path_p, n_samples=6000,
                                       center=True, prescale=glb_prescale)
    model_ext = (np.asarray(model_pcd.points).max(0)
                 - np.asarray(model_pcd.points).min(0)) * 1000

    # target_pcd 를 다시 만들어 ICP rmse 평가 (이 ensemble pose 에 대해)
    # — fused cloud 에 대해 GLB at T_med 의 inlier rmse 계산
    pts_np, _ = collect_table_aware_cloud(
        obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
        n_frames=n_frames, gripper_weight=1)
    if len(pts_np) > 200:
        target_pcd = clean_cloud(pts_np, voxel_m=0.002)
        target_pcd = keep_largest_cluster(target_pcd, eps=0.012, min_points=40)
        target_arr = np.asarray(target_pcd.points)
        ext = (target_arr.max(0) - target_arr.min(0)) * 1000
        # Open3D eval
        model_T = o3d.geometry.PointCloud()
        model_T.points = model_pcd.points
        model_T.transform(T_med)
        eval_res = o3d.pipelines.registration.evaluate_registration(
            model_T, target_pcd, max_correspondence_distance=0.005)
        fit_final = float(eval_res.fitness)
        rmse_final = float(eval_res.inlier_rmse * 1000)
    else:
        ext = np.array([0, 0, 0]); fit_final = fit_mean; rmse_final = rmse_mean

    ratio = max(ext) / (max(model_ext) + 1e-4) if model_ext.max() > 0 else 1.0
    return {
        "params": {"seg": "per_frame_ensemble",
                   "n_estimates": len(estimates),
                   "n_kept": diag["n_kept"],
                   "mad_mm": diag["mad_mm"]},
        "T_base_object": T_med,
        "fit": fit_final,
        "rmse_mm": rmse_final,
        "extent_mm": [float(x) for x in ext],
        "model_extent_mm": [float(x) for x in model_ext],
        "extent_ratio": float(ratio),
        "score": fit_final / (rmse_final / 1000 + 0.001),
        "n_views_used": len(estimates),
        "n_points_clean": int(len(target_arr)) if len(pts_np) > 200 else 0,
        "target_pcd": target_pcd if len(pts_np) > 200 else None,
        "mesh": mesh,
        "extent_penalty": 0.0,
    }


def estimate_per_cam_pose(obj, cam_id, table_v_threshold,
                          intrinsics, static_T_base_cam, T_gripper_cam,
                          glb_path, n_frames=19,
                          voxel_mm=1.5, max_corr_mm=8.0):
    """단일 cam만 사용하여 객체 위치 추정. outlier cam 검출에 사용."""
    pts_np, n_used = collect_table_aware_cloud(
        obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
        n_frames=n_frames, gripper_weight=1, top2_ratio=0.85,
        cam_ids=(cam_id,))
    if len(pts_np) < 100:
        return None
    target_pcd = clean_cloud(pts_np, voxel_m=voxel_mm / 1000)
    target_pcd = keep_largest_cluster(target_pcd, eps=0.012, min_points=20)
    if len(target_pcd.points) < 30:
        return None
    target_arr = np.asarray(target_pcd.points)
    model_pcd, _ = load_glb_as_pcd(glb_path, n_samples=4000, center=True)
    T0 = initial_T(model_pcd, target_pcd)
    result = icp_with_scale(model_pcd, target_pcd, T0,
                             corr_mm_schedule=(15, 8, 4), yaw_grid=18,
                             with_scaling=True)
    if result.inlier_rmse < 0.0002:
        return None
    return result.transformation[:3, 3] * 1000  # mm


def select_inlier_cams(obj, table_v_threshold,
                       intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                       n_frames=19, max_dev_mm=30.0):
    """각 cam의 단독 추정 위치 비교하여 median에서 max_dev_mm 안의 cam만 keep.
    gripper cam(cam2)은 항상 keep (가장 정확하다고 가정)."""
    per_cam_pos = {}
    for ci in [0, 1, 2, 3]:
        pos = estimate_per_cam_pose(
            obj, ci, table_v_threshold, intrinsics,
            static_T_base_cam, T_gripper_cam, glb_path, n_frames=n_frames)
        if pos is not None:
            per_cam_pos[ci] = pos
    if 2 not in per_cam_pos:
        # gripper cam 없으면 모든 cam median 기준
        if not per_cam_pos:
            return [0, 1, 2, 3]
        positions = np.array(list(per_cam_pos.values()))
        ref = np.median(positions, axis=0)
    else:
        # gripper cam이 reference (가장 신뢰)
        ref = per_cam_pos[2]
    inlier_cams = []
    diagnostics = {}
    for ci in [0, 1, 2, 3]:
        if ci == 2 and ci in per_cam_pos:
            inlier_cams.append(ci)
            diagnostics[ci] = {"in": True, "dev_mm": 0.0, "reason": "gripper_anchor"}
            continue
        if ci not in per_cam_pos:
            diagnostics[ci] = {"in": False, "reason": "no_pose"}
            continue
        dev = float(np.linalg.norm(per_cam_pos[ci] - ref))
        in_set = dev <= max_dev_mm
        diagnostics[ci] = {"in": in_set, "dev_mm": dev,
                           "pos_mm": [float(v) for v in per_cam_pos[ci]]}
        if in_set:
            inlier_cams.append(ci)
    # per_cam_pos 도 반환 (depth_cluster prefer_position 으로 사용 가능)
    return inlier_cams, diagnostics, per_cam_pos


def evaluate_table_aware(obj, table_v_threshold,
                          intrinsics, static_T_base_cam, T_gripper_cam,
                          glb_path, voxel_mm, dbscan_eps_mm,
                          n_frames=19, gripper_weight=10, with_scaling=True,
                          top2_ratio=0.85, cam_ids=(0, 1, 2, 3),
                          glb_prescale=1.0):
    pts_np, n_used = collect_table_aware_cloud(
        obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
        n_frames=n_frames, gripper_weight=gripper_weight,
        top2_ratio=top2_ratio, cam_ids=cam_ids)
    if len(pts_np) < 200:
        return None
    target_pcd = clean_cloud(pts_np, voxel_m=voxel_mm / 1000)
    target_pcd = keep_largest_cluster(target_pcd,
                                      eps=dbscan_eps_mm / 1000, min_points=50)
    target_arr = np.asarray(target_pcd.points)
    # 충분한 점이 있어야 ICP가 의미있게 6DoF 결정. 너무 적으면 degenerate fit 위험.
    if len(target_arr) < 400:
        return None
    ext = (target_arr.max(0) - target_arr.min(0)) * 1000
    # GLB 사전 스케일 적용 (auto-estimated dims) — ICP 는 scale 학습 안 함 (rigid)
    model_pcd, mesh = load_glb_as_pcd(glb_path, n_samples=8000, prescale=glb_prescale)
    model_ext = (np.asarray(model_pcd.points).max(0)
                 - np.asarray(model_pcd.points).min(0)) * 1000
    # 점 분포가 GLB 대비 너무 작으면 degenerate 가능성 → 거부
    ratio_check = max(ext) / (max(model_ext) + 1e-4)
    if ratio_check < 0.50:
        # target 이 model 대비 절반 미만 = 객체 일부만 보임 → 신뢰 불가
        return None
    T0 = initial_T(model_pcd, target_pcd)
    # scale-enabled ICP: GLB ↔ 실제 블록 치수 자동 보정
    result = icp_with_scale(model_pcd, target_pcd, T0,
                             corr_mm_schedule=(20, 10, 5, 3),
                             yaw_grid=24, with_scaling=with_scaling)
    # Degenerate fit guard: RMSE 가 비현실적으로 낮으면 (<0.2mm) 의심 → 거부
    if result.inlier_rmse < 0.0002:
        return None
    # ICP에서 학습된 스케일 추출
    learned_scale = float(np.linalg.norm(result.transformation[:3, 0]))
    ratio = max(ext) / (max(model_ext) + 1e-4)
    extent_penalty = max(0, ratio - 1.6) * 50
    score = result.fitness / (result.inlier_rmse + 0.001) - extent_penalty
    return {
        "params": {"seg": "table_aware", "voxel_mm": voxel_mm,
                   "dbscan_eps_mm": dbscan_eps_mm,
                   "table_v_threshold": float(table_v_threshold),
                   "gripper_weight": gripper_weight,
                   "with_scaling": with_scaling},
        "T_base_object": result.transformation,
        "learned_scale": learned_scale,
        "fit": float(result.fitness),
        "rmse_mm": float(result.inlier_rmse * 1000),
        "extent_mm": [float(x) for x in ext],
        "model_extent_mm": [float(x) for x in model_ext],
        "extent_ratio": float(ratio),
        "score": float(score),
        "n_views_used": n_used,
        "n_points_clean": int(len(target_arr)),
        "target_pcd": target_pcd,
        "mesh": mesh,
        "extent_penalty": float(extent_penalty),
    }


def color_white_mask(rgb_bgr, s_max=30, v_min=130):
    """HSV-based white-pixel mask. 검정 테이블 위 흰 박스에 특화."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    mask = ((S < s_max) & (V > v_min)).astype(np.uint8) * 255
    # morphological cleanup
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def collect_color_cloud(intrinsics, static_T_base_cam, T_gripper_cam,
                        s_max=30, v_min=130,
                        z_min_mm=5.0, z_max_mm=300.0,
                        n_frames=19):
    """순수 색 임계 마스크로 점군 누적 (SAM 우회). 흰박스용."""
    all_pts = []
    n_used = 0
    for frame in range(n_frames):
        fid = f"{frame:06d}"
        for ci in [0, 1, 2, 3]:
            rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
            if rgb is None:
                continue
            mask = color_white_mask(rgb, s_max=s_max, v_min=v_min)
            if mask.sum() < 100:
                continue
            depth = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"depth_{fid}.png"),
                                cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            K, _, depth_scale = intrinsics[ci]
            pts_cam = backproject(depth, mask, K, depth_scale)
            if len(pts_cam) < 20:
                continue
            if ci == 2:
                T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
                if not T_be_path.exists():
                    continue
                T_be = np.load(T_be_path)
                T_base_cam = T_be @ T_gripper_cam
            else:
                T_base_cam = static_T_base_cam[ci]
            pts_base = transform_pts(pts_cam, T_base_cam)
            pts_mm = pts_base * 1000
            zm = pts_mm[:, 2]
            inz = (zm > z_min_mm) & (zm < z_max_mm)
            if inz.sum() < 20:
                continue
            all_pts.append(pts_base[inz])
            n_used += 1
    if not all_pts:
        return np.zeros((0, 3)), n_used
    return np.vstack(all_pts), n_used


def evaluate_color_seg(obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                       s_max, v_min, voxel_mm, max_corr_mm, n_frames=19):
    """색-임계 + 깊이 필터 + 최대 클러스터 → ICP. SAM 우회 모드."""
    pts_np, n_used = collect_color_cloud(
        intrinsics, static_T_base_cam, T_gripper_cam,
        s_max=s_max, v_min=v_min, n_frames=n_frames)
    if len(pts_np) < 200:
        return None
    target_pcd = clean_cloud(pts_np, voxel_m=voxel_mm / 1000)
    # 최대 클러스터만 유지
    target_pcd = keep_largest_cluster(target_pcd, eps=0.015, min_points=80)
    target_arr = np.asarray(target_pcd.points)
    if len(target_arr) < 50:
        return None
    ext = (target_arr.max(0) - target_arr.min(0)) * 1000
    model_pcd, mesh = load_glb_as_pcd(glb_path, n_samples=8000)
    model_ext = (np.asarray(model_pcd.points).max(0)
                 - np.asarray(model_pcd.points).min(0)) * 1000
    T0 = initial_T(model_pcd, target_pcd)
    result = run_icp_multi_init(
        model_pcd, target_pcd, T0,
        max_corr_coarse=max_corr_mm * 4 / 1000,
        max_corr_fine=max_corr_mm / 1000,
        yaw_grid=24)
    ratio = max(ext) / (max(model_ext) + 1e-4)
    extent_penalty = max(0, ratio - 1.6) * 50
    score = result.fitness / (result.inlier_rmse + 0.001) - extent_penalty
    return {
        "params": {"seg": "color_seg", "s_max": s_max, "v_min": v_min,
                   "voxel_mm": voxel_mm, "max_corr_mm": max_corr_mm},
        "T_base_object": result.transformation,
        "fit": float(result.fitness),
        "rmse_mm": float(result.inlier_rmse * 1000),
        "extent_mm": [float(x) for x in ext],
        "model_extent_mm": [float(x) for x in model_ext],
        "extent_ratio": float(ratio),
        "score": float(score),
        "n_views_used": n_used,
        "n_points_clean": int(len(target_arr)),
        "target_pcd": target_pcd,
        "mesh": mesh,
        "extent_penalty": float(extent_penalty),
    }


def evaluate_depth_cluster(obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                            voxel_mm, max_corr_mm, n_frames=19,
                            z_min_mm=5.0, z_max_mm=250.0,
                            workspace_half_xy_mm=(800.0, 800.0),
                            ws_center_xy_mm=(0.0, 600.0),
                            exclude_positions_mm=None,
                            min_exclude_dist_mm=40.0,
                            prefer_position_mm=None,
                            prefer_radius_mm=80.0):
    """SAM 우회: 모든 depth 점 -> z 필터(테이블 위) -> xy 워크스페이스 필터 -> DBSCAN
    -> GLB extent와 가장 비슷한 클러스터 선택 -> ICP.
    검정 테이블 환경에서 흰박스 추출에 유리.
    """
    pts_np, n_used = collect_depth_above_table_cloud(
        intrinsics, static_T_base_cam, T_gripper_cam,
        center_xy_mm=ws_center_xy_mm,
        half_extent_xy_mm=workspace_half_xy_mm,
        z_min_mm=z_min_mm, z_max_mm=z_max_mm,
        n_frames=n_frames, depth_stride=4)
    if len(pts_np) < 200:
        return None
    pcd = clean_cloud(pts_np, voxel_m=voxel_mm / 1000)
    clusters = find_clusters(pcd, eps=0.018, min_points=80)
    if not clusters:
        return None
    # Load GLB to get target extent
    model_pcd, mesh = load_glb_as_pcd(glb_path, n_samples=6000)
    model_ext = (np.asarray(model_pcd.points).max(0) -
                 np.asarray(model_pcd.points).min(0)) * 1000
    target_ext = sorted(model_ext, reverse=True)
    # Pick cluster: 1순위 prefer_position_mm 근처 (per-cam 추정 위치) 안에서 extent 매칭,
    # 2순위 단순 extent 매칭. exclude_positions_mm (다른 객체 위치) 는 항상 거부.
    excl = exclude_positions_mm or []
    pref = np.asarray(prefer_position_mm) if prefer_position_mm is not None else None
    best_cluster = None
    best_diff = np.inf
    best_dist_to_pref = np.inf
    for c in clusters[:12]:  # top-12 후보
        ce = sorted(c["extent_mm"], reverse=True)
        centroid_mm = c["centroid"] * 1000
        # exclude: 이미 다른 객체로 추정된 위치 근처
        too_close = any(
            float(np.linalg.norm(centroid_mm - np.asarray(ex_pos))) < min_exclude_dist_mm
            for ex_pos in excl)
        if too_close:
            continue
        diff = sum(abs(ce[i] - target_ext[i]) for i in range(3))
        if ce[0] > 2.0 * target_ext[0]:
            continue  # size mismatch too big
        # prefer_position 우선: 그 안에 있는 cluster 중 best diff
        if pref is not None:
            d_pref = float(np.linalg.norm(centroid_mm - pref))
            if d_pref < prefer_radius_mm:
                # prefer 영역 내 → 우선 채택 (diff 가 best 보다 작거나 비슷할 때)
                if best_cluster is None or (
                        best_dist_to_pref >= prefer_radius_mm or diff < best_diff * 1.5):
                    best_diff = diff
                    best_cluster = c
                    best_dist_to_pref = d_pref
            else:
                # prefer 밖이면 prefer 안에 있는 게 없을 때만 채택
                if best_dist_to_pref >= prefer_radius_mm and diff < best_diff:
                    best_diff = diff
                    best_cluster = c
                    best_dist_to_pref = d_pref
        else:
            if diff < best_diff:
                best_diff = diff
                best_cluster = c
    if best_cluster is None:
        return None
    target_pcd = best_cluster["pcd"]
    target_arr = np.asarray(target_pcd.points)
    ext = (target_arr.max(0) - target_arr.min(0)) * 1000
    T0 = initial_T(model_pcd, target_pcd)
    result = run_icp_multi_init(
        model_pcd, target_pcd, T0,
        max_corr_coarse=max_corr_mm * 4 / 1000,
        max_corr_fine=max_corr_mm / 1000,
        yaw_grid=24)
    ratio = max(ext) / (max(model_ext) + 1e-4)
    extent_penalty = max(0, ratio - 1.6) * 50
    score = result.fitness / (result.inlier_rmse + 0.001) - extent_penalty
    return {
        "params": {"seg": "depth_cluster", "voxel_mm": voxel_mm,
                   "max_corr_mm": max_corr_mm,
                   "z_min_mm": z_min_mm, "z_max_mm": z_max_mm},
        "T_base_object": result.transformation,
        "fit": float(result.fitness),
        "rmse_mm": float(result.inlier_rmse * 1000),
        "extent_mm": [float(x) for x in ext],
        "model_extent_mm": [float(x) for x in model_ext],
        "extent_ratio": float(ratio),
        "score": float(score),
        "n_views_used": n_used,
        "n_points_clean": int(len(target_arr)),
        "target_pcd": target_pcd,
        "mesh": mesh,
        "extent_penalty": float(extent_penalty),
        "n_clusters_found": len(clusters),
    }


def auto_tune_object(obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                     quick=False, table_v_threshold=None,
                     reject_outlier_cams=True, max_cam_dev_mm=30.0,
                     exclude_positions_mm=None,
                     glb_prescale=1.0):
    if quick:
        vote_ratios = [0.5, 0.7]
        voxels = [3.0]
        max_corrs = [10.0]
        dbscans = [15.0]
    else:
        vote_ratios = [0.3, 0.5, 0.6, 0.7, 0.8]
        voxels = [2.0, 3.0]
        max_corrs = [8.0, 12.0]
        dbscans = [12.0, 20.0]
    best = None
    trial = 0

    # ── Per-cam outlier rejection (cam0/cam3 노이즈가 큰 경우 자동 제외) ──
    cam_ids_use = (0, 1, 2, 3)
    prefer_pos_mm = None
    if reject_outlier_cams and table_v_threshold is not None:
        sel = select_inlier_cams(
            obj, table_v_threshold, intrinsics, static_T_base_cam,
            T_gripper_cam, glb_path, max_dev_mm=max_cam_dev_mm)
        if isinstance(sel, tuple) and len(sel) >= 3:
            cam_ids_use, diag, per_cam_pos = sel
        elif isinstance(sel, tuple):
            cam_ids_use, diag = sel
            per_cam_pos = {}
        else:
            cam_ids_use = sel
            diag = {}
            per_cam_pos = {}
        # gripper cam (cam2) 우선 — 가장 정확
        if 2 in per_cam_pos:
            prefer_pos_mm = per_cam_pos[2]
        elif per_cam_pos:
            # cam2 없으면 median of available
            positions = np.array(list(per_cam_pos.values()))
            prefer_pos_mm = np.median(positions, axis=0)
        if len(cam_ids_use) < 4:
            dropped = [c for c in [0, 1, 2, 3] if c not in cam_ids_use]
            print(f"  outlier cam rejection: keep={cam_ids_use}, drop={dropped}")
            for ci, info in diag.items():
                if "dev_mm" in info:
                    flag = "IN" if info["in"] else "OUT"
                    print(f"    cam{ci}: {flag}  dev={info['dev_mm']:.1f}mm")
        else:
            print(f"  outlier cam rejection: all 4 cams within {max_cam_dev_mm}mm")

    cam_ids_t = tuple(cam_ids_use)

    # ── Per-frame ensemble (1차 시도, 가장 정확) ──
    # 각 cam2 frame 독립 ICP × 19 + 정적 cam 3개 → robust SE3 median
    if table_v_threshold is not None:
        print(f"  trying per-frame ensemble (cam2 × 19 frames + static cams) ...")
        res = evaluate_per_frame_ensemble(
            obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
            glb_path, glb_prescale=glb_prescale)
        if res is not None:
            flag = "★" if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM and
                           res["extent_ratio"] <= 1.6) else " "
            print(f"   [ensemble n={res['params']['n_estimates']} kept={res['params']['n_kept']} "
                  f"mad={res['params']['mad_mm']:.1f}mm]  "
                  f"fit={res['fit']:.3f}  rmse={res['rmse_mm']:5.2f}mm  "
                  f"ext_ratio={res['extent_ratio']:4.2f}  score={res['score']:7.2f} {flag}")
            if best is None or res["score"] > best["score"]:
                best = res
            if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM
                    and res["extent_ratio"] <= 1.4):
                print(f"   -> ensemble good enough, stopping")
                return best

    # ── Fallback: table-aware color seg + multi-scale ICP ──
    if table_v_threshold is not None:
        print(f"  trying table-aware mode (V_thr={table_v_threshold:.0f}, cams={cam_ids_t}) ...")
        ta_combos = [(2.0, 12.0), (2.0, 18.0), (3.0, 15.0), (1.5, 10.0)]
        for vox, ds in ta_combos:
            res = evaluate_table_aware(
                obj, table_v_threshold, intrinsics, static_T_base_cam, T_gripper_cam,
                glb_path, voxel_mm=vox, dbscan_eps_mm=ds,
                cam_ids=cam_ids_t,
                glb_prescale=glb_prescale,
                with_scaling=True)  # scale guard 가 범위 벗어나면 자동 rigid 대체
            if res is None:
                print(f"   [ta vox={vox} ds={ds}] -> too few pts")
                continue
            flag = "★" if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM and
                           res["extent_ratio"] <= 1.6) else " "
            print(f"   [ta vox={vox} ds={ds}]  fit={res['fit']:.3f}  "
                  f"rmse={res['rmse_mm']:5.2f}mm  ext_ratio={res['extent_ratio']:4.2f}  "
                  f"n_pts={res['n_points_clean']}  score={res['score']:7.2f} {flag}")
            if best is None or res["score"] > best["score"]:
                best = res
            if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM
                    and res["extent_ratio"] <= 1.4):
                print(f"   -> table-aware good enough, stopping")
                return best

    combos = list(itertools.product(vote_ratios, voxels, max_corrs, dbscans))
    print(f"  trying {len(combos)} SAM-vote combos...")
    for vr, vox, mc, ds in combos:
        trial += 1
        res = evaluate_combo(obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                             vote_ratio=vr, voxel_mm=vox, max_corr_mm=mc, dbscan_eps_mm=ds)
        if res is None:
            print(f"   [{trial:>2}/{len(combos)}] vr={vr} vox={vox} mc={mc} ds={ds} -> too few pts")
            continue
        flag = "★" if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM and
                       res["extent_ratio"] <= 1.6) else " "
        print(f"   [{trial:>2}/{len(combos)}] vr={vr} vox={vox} mc={mc} ds={ds}  "
              f"fit={res['fit']:.3f}  rmse={res['rmse_mm']:5.2f}mm  "
              f"ext_ratio={res['extent_ratio']:4.2f}  score={res['score']:7.2f} {flag}")
        if best is None or res["score"] > best["score"]:
            best = res
        if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM
                and res["extent_ratio"] <= 1.4):
            print(f"   -> good enough, stopping SAM auto-tune")
            break

    # Fallback A: 색 임계 (흰박스 특화). SAM 우회.
    needs_fallback = (best is None
                      or best["fit"] < GOAL_FIT
                      or best["rmse_mm"] > GOAL_RMSE_MM
                      or best["extent_ratio"] > 1.6)
    if needs_fallback and obj == "box":
        print("  trying color-segment fallback (S<smax, V>vmin) ...")
        for sm, vm, vox, mc in [(30, 130, 2.0, 8.0), (40, 120, 2.0, 8.0),
                                 (50, 100, 2.0, 10.0), (35, 150, 3.0, 8.0)]:
            res = evaluate_color_seg(
                obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                s_max=sm, v_min=vm, voxel_mm=vox, max_corr_mm=mc)
            if res is None:
                print(f"   [color s_max={sm} v_min={vm}] -> too few pts")
                continue
            flag = "★" if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM and
                           res["extent_ratio"] <= 1.6) else " "
            print(f"   [color s_max={sm} v_min={vm} vox={vox} mc={mc}]  "
                  f"fit={res['fit']:.3f}  rmse={res['rmse_mm']:5.2f}mm  "
                  f"ext_ratio={res['extent_ratio']:4.2f}  score={res['score']:7.2f} {flag}")
            if best is None or res["score"] > best["score"]:
                best = res
            if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM
                    and res["extent_ratio"] <= 1.4):
                print(f"   -> color-seg good enough, stopping")
                break

    # Fallback B: depth-above-table clustering (SAM/color-free).
    needs_fallback = (best is None
                      or best["fit"] < GOAL_FIT
                      or best["rmse_mm"] > GOAL_RMSE_MM
                      or best["extent_ratio"] > 1.6)
    if needs_fallback:
        print("  trying depth-cluster fallback (SAM-free) ...")
        depth_combos = [(2.0, 8.0), (2.0, 12.0), (3.0, 8.0), (3.0, 12.0)]
        for vox, mc in depth_combos:
            res = evaluate_depth_cluster(
                obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
                voxel_mm=vox, max_corr_mm=mc,
                exclude_positions_mm=exclude_positions_mm,
                prefer_position_mm=prefer_pos_mm)
            if res is None:
                print(f"   [depth vox={vox} mc={mc}] -> no valid cluster")
                continue
            flag = "★" if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM and
                           res["extent_ratio"] <= 1.6) else " "
            print(f"   [depth vox={vox} mc={mc}]  fit={res['fit']:.3f}  "
                  f"rmse={res['rmse_mm']:5.2f}mm  ext_ratio={res['extent_ratio']:4.2f}  "
                  f"clusters_found={res['n_clusters_found']}  score={res['score']:7.2f} {flag}")
            if best is None or res["score"] > best["score"]:
                best = res
            if (res["fit"] >= GOAL_FIT and res["rmse_mm"] <= GOAL_RMSE_MM
                    and res["extent_ratio"] <= 1.4):
                print(f"   -> depth-cluster good enough, stopping")
                break

    return best


# ────────────────────────────────────────────────────────────────────
# Comparison rendering
# ────────────────────────────────────────────────────────────────────

def project_mesh_to_image(mesh, T_cam_obj, K, D, img_shape):
    """Project mesh edges onto image. Returns mask (h,w) + edge image."""
    h, w = img_shape[:2]
    verts_obj = np.asarray(mesh.vertices)
    if len(verts_obj) == 0:
        return None, None
    # transform to cam frame
    verts_cam = (T_cam_obj[:3, :3] @ verts_obj.T).T + T_cam_obj[:3, 3]
    # in front of cam only
    z = verts_cam[:, 2]
    if (z > 0.05).sum() < 3:
        return None, None
    # project
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    pts2d, _ = cv2.projectPoints(verts_cam.astype(np.float64), rvec, tvec,
                                  K.astype(np.float64), D.astype(np.float64))
    pts2d = pts2d.reshape(-1, 2)
    # Edges from faces
    faces = np.asarray(mesh.faces)
    edge_img = np.zeros((h, w, 3), dtype=np.uint8)
    silhouette = np.zeros((h, w), dtype=np.uint8)
    # Fill triangles for silhouette
    for tri in faces:
        a, b, c = tri
        if z[a] <= 0.05 or z[b] <= 0.05 or z[c] <= 0.05:
            continue
        pts = np.array([pts2d[a], pts2d[b], pts2d[c]], dtype=np.int32)
        # within image
        if np.any(pts[:, 0] < -100) or np.any(pts[:, 0] > w + 100):
            continue
        if np.any(pts[:, 1] < -100) or np.any(pts[:, 1] > h + 100):
            continue
        cv2.fillPoly(silhouette, [pts], 255)
    # Find contour for clean overlay
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return silhouette, contours


def render_overlay(rgb_bgr, mesh, T_cam_obj, K, D, color_bgr=(0, 255, 255)):
    """Overlay GLB silhouette + outline on RGB image."""
    if rgb_bgr is None or mesh is None:
        return rgb_bgr
    sil, contours = project_mesh_to_image(mesh, T_cam_obj, K, D, rgb_bgr.shape)
    out = rgb_bgr.copy()
    if sil is None:
        return out
    # semi-transparent fill
    color_img = np.zeros_like(out)
    color_img[:, :] = color_bgr
    alpha = 0.25
    mask_3 = (sil > 0)[..., None]
    out = np.where(mask_3, (out * (1 - alpha) + color_img * alpha).astype(np.uint8), out)
    # outline
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, 2, cv2.LINE_AA)
    return out


def render_comparison(obj, mesh, T_base_obj, intrinsics,
                       static_T_base_cam, T_gripper_cam,
                       cam_color=(0, 255, 255), ref_frame=9):
    """Per-cam overlay; returns dict cam_idx -> image."""
    images = {}
    fid = f"{ref_frame:06d}"
    for cam_idx in [0, 1, 2, 3]:
        rgb_path = CAPTURE_DIR / f"cam{cam_idx}" / f"rgb_{fid}.jpg"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            continue
        if cam_idx == 2:
            T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_path.exists():
                continue
            T_be = np.load(T_be_path)
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[cam_idx]
        T_cam_base = np.linalg.inv(T_base_cam)
        T_cam_obj = T_cam_base @ T_base_obj
        K, D, _ = intrinsics[cam_idx]
        ov = render_overlay(rgb, mesh, T_cam_obj, K, D, color_bgr=cam_color)
        # label
        label = f"cam{cam_idx} | {obj}"
        cv2.rectangle(ov, (5, 5), (5 + 10 * len(label) + 10, 32), (0, 0, 0), -1)
        cv2.putText(ov, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        images[cam_idx] = ov
    return images


def quad_image(images_by_cam, target_w=640):
    """Compose 2x2 quad from cam0,1,2,3."""
    tiles = []
    for ci in [0, 1, 2, 3]:
        im = images_by_cam.get(ci)
        if im is None:
            im = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(im, f"cam{ci} (no image)", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 255), 2)
        h, w = im.shape[:2]
        tw = target_w
        th = int(h * tw / w)
        tiles.append(cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA))
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top, bot])


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default="red,cream,blue,box")
    ap.add_argument("--out_dir", default="data/pose_fused")
    ap.add_argument("--quick", action="store_true", help="fewer trials")
    ap.add_argument("--n_frames", type=int, default=19)
    ap.add_argument("--ref_frame", type=int, default=9, help="comparison overlay reference frame")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = [s.strip() for s in args.objects.split(",") if s.strip()]

    print("Loading calibration / intrinsics ...")
    intrinsics = load_intrinsics()
    static_T_base_cam, T_gripper_cam = load_static_transforms()

    print("Detecting table color (auto) ...")
    table_v_thr, table_v_dom = detect_table_v_threshold(n_frames=args.n_frames)
    print(f"  dominant V (table) = {table_v_dom:.1f}  ->  foreground V_threshold = {table_v_thr:.1f}")

    print(f"\nObjects: {objects}   frames: 0..{args.n_frames - 1}   "
          f"(goal: fit≥{GOAL_FIT}, rmse≤{GOAL_RMSE_MM}mm)\n")

    summary = {}
    comparison_tiles = {}
    obj_colors = {
        "red":   (0, 0, 255),
        "cream": (140, 220, 240),
        "blue":  (255, 200, 100),
        "box":   (220, 220, 220),
    }
    accepted_positions_mm = []  # 누적된 다른 객체 위치 — depth_cluster fallback에서 중복 회피
    auto_dims_by_obj = {}  # 객체별 자동 추정 실제 크기 (mm, sorted desc)
    glb_prescale_by_obj = {}  # GLB → 추정 크기 uniform 스케일
    print("\n--- Auto-estimating block dimensions from 76-view fused clouds ---")
    for obj in objects:
        glb_path = DATA_DIR / f"{obj}.glb"
        if not glb_path.exists():
            continue
        dims = auto_estimate_block_real_size(
            obj, table_v_thr, intrinsics, static_T_base_cam, T_gripper_cam,
            n_frames=args.n_frames)
        m = trimesh.load(str(glb_path), force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        glb_ext_raw = np.asarray(m.extents * 1000, dtype=np.float64)  # x, y, z
        glb_ext_sorted = sorted([float(e) for e in glb_ext_raw], reverse=True)
        if dims is None:
            print(f"  {obj:>6}: dim estimation failed → prescale=1.0 (use GLB original)")
            auto_dims_by_obj[obj] = None
            glb_prescale_by_obj[obj] = 1.0
            continue
        # Uniform prescale: GLB max-axis 가 observed max-axis 와 일치하도록 scalar 스케일.
        # 안전 범위 [0.5, 2.0] 클램핑 — degenerate 차단
        prescale = dims[0] / glb_ext_sorted[0] if glb_ext_sorted[0] > 0 else 1.0
        prescale = float(np.clip(prescale, 0.5, 2.0))
        auto_dims_by_obj[obj] = dims
        glb_prescale_by_obj[obj] = prescale
        print(f"  {obj:>6}: GLB ext (sorted) [{glb_ext_sorted[0]:.1f},{glb_ext_sorted[1]:.1f},{glb_ext_sorted[2]:.1f}]mm  "
              f"→ observed [{dims[0]:.1f},{dims[1]:.1f},{dims[2]:.1f}]mm  "
              f"prescale={prescale:.3f}")
    print()

    for obj in objects:
        glb_path = DATA_DIR / f"{obj}.glb"
        if not glb_path.exists():
            print(f"[SKIP] {obj}: missing GLB at {glb_path}")
            continue
        print(f"━━━ {obj} ━━━")
        best = auto_tune_object(
            obj, intrinsics, static_T_base_cam, T_gripper_cam, glb_path,
            quick=args.quick, table_v_threshold=table_v_thr,
            exclude_positions_mm=list(accepted_positions_mm),
            glb_prescale=glb_prescale_by_obj.get(obj, 1.0))
        if best is None:
            print(f"  [FAIL] no successful combo")
            summary[obj] = {"pass": False, "reason": "no_valid_combo"}
            continue

        T_b_o = best["T_base_object"]
        t = T_b_o[:3, 3] * 1000
        R = T_b_o[:3, :3]
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

        passed = (best["fit"] >= GOAL_FIT and best["rmse_mm"] <= GOAL_RMSE_MM
                  and best["extent_ratio"] <= 1.6)
        # 다음 객체 추정 시 이 위치와 충돌하지 않도록 누적 (depth_cluster fallback 보호)
        accepted_positions_mm.append([float(t[0]), float(t[1]), float(t[2])])
        print(f"  best: fit={best['fit']:.3f}  rmse={best['rmse_mm']:.2f}mm  "
              f"ext_ratio={best['extent_ratio']:.2f}  pass={passed}")
        print(f"  params: {best['params']}")
        print(f"  pose (mm/deg): x={t[0]:+.2f} y={t[1]:+.2f} z={t[2]:+.2f}  "
              f"yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f}")

        obj_out = out_dir / obj
        obj_out.mkdir(parents=True, exist_ok=True)
        np.save(obj_out / "T_base_object.npy", T_b_o)
        o3d.io.write_point_cloud(str(obj_out / "fused_cloud.ply"), best["target_pcd"])
        mesh_T = best["mesh"].copy()
        mesh_T.apply_transform(T_b_o)
        mesh_T.export(str(obj_out / f"{obj}_posed_fused.glb"))
        json.dump({
            "object": obj,
            "pass": passed,
            "best_params": best["params"],
            "n_views_used": best["n_views_used"],
            "n_points_clean": best["n_points_clean"],
            "icp_fit": best["fit"],
            "icp_rmse_m": best["rmse_mm"] / 1000,
            "extent_mm": best["extent_mm"],
            "model_extent_mm": best["model_extent_mm"],
            "extent_ratio": best["extent_ratio"],
            "T_base_object_4x4": T_b_o.tolist(),
            "position_mm": [float(v) for v in t],
            "euler_zyx_deg": [float(yaw), float(pitch), float(roll)],
        }, open(obj_out / "pose.json", "w"), indent=2)

        # 객체별 비교 이미지 저장 안 함 (요구사항). 파이프라인 단계별만 저장.

        summary[obj] = {
            "pass": passed,
            "fit": best["fit"],
            "rmse_mm": best["rmse_mm"],
            "extent_ratio": best["extent_ratio"],
            "params": best["params"],
            "position_mm": [float(v) for v in t],
            "euler_zyx_deg": [float(yaw), float(pitch), float(roll)],
            "n_views_used": best["n_views_used"],
        }
        print()

    # ── 파이프라인 단계별 이미지 저장 (객체별 비교 X, 단계별 ○) ──
    print("\n=== Saving pipeline-stage visualizations ===")
    stages_dir = out_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    fid = f"{args.ref_frame:06d}"

    def _quad(images_by_cam, target_w=640, label_prefix=""):
        tiles = []
        for ci in [0, 1, 2, 3]:
            im = images_by_cam.get(ci)
            if im is None:
                im = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(im, f"cam{ci} (n/a)", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 255), 2)
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            h, w = im.shape[:2]
            tw = target_w; th = int(h * tw / w)
            tile = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            label = f"cam{ci} {label_prefix}"
            cv2.rectangle(tile, (5, 5), (5 + 14 * len(label) + 10, 32), (0, 0, 0), -1)
            cv2.putText(tile, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])

    # Stage 1: 원본 RGB
    raw_imgs = {ci: cv2.imread(str(CAPTURE_DIR / f"cam{ci}/rgb_{fid}.jpg")) for ci in [0,1,2,3]}
    cv2.imwrite(str(stages_dir / "stage1_input_rgb.png"),
                _quad(raw_imgs, label_prefix=f"RGB f{fid}"))
    print(f"  saved: {stages_dir / 'stage1_input_rgb.png'}")

    # Stage 2: HSV 색 마스크 (모든 객체 색별 overlay)
    masks_overlay = {}
    for ci in [0, 1, 2, 3]:
        rgb = raw_imgs.get(ci)
        if rgb is None: continue
        out = rgb.copy()
        for obj in objects:
            cs = load_obj_color_spec(obj)
            mask = table_aware_mask(rgb, cs, table_v_thr)
            color_img = np.zeros_like(out); color_img[:] = obj_colors.get(obj, (0,255,255))
            mask3 = (mask > 0)[..., None]
            out = np.where(mask3, (out*0.5 + color_img*0.5).astype(np.uint8), out)
        masks_overlay[ci] = out
    cv2.imwrite(str(stages_dir / "stage2_hsv_masks.png"),
                _quad(masks_overlay, label_prefix="HSV masks"))
    print(f"  saved: {stages_dir / 'stage2_hsv_masks.png'}")

    # Stage 3: 포인트 클라우드 top-down (base frame xy)
    cloud_canvas = np.full((640, 640, 3), 30, dtype=np.uint8)
    # 작업면 그리드 (-500 ~ +500 in x, 0 ~ +1000 in y) → 640px
    cv2.rectangle(cloud_canvas, (10, 10), (630, 630), (60, 60, 60), 1)
    for obj in objects:
        cl_path = out_dir / obj / "fused_cloud.ply"
        if not cl_path.exists(): continue
        pcd = o3d.io.read_point_cloud(str(cl_path))
        pts = np.asarray(pcd.points) * 1000  # mm
        if len(pts) == 0: continue
        # x: -500..+500 → 0..640;  y: 0..+1000 → 0..640 (위에서 본 viewpoint)
        u = ((pts[:, 0] + 500) / 1000 * 640).astype(np.int32)
        v = (640 - (pts[:, 1] / 1000 * 640)).astype(np.int32)
        valid = (u >= 0) & (u < 640) & (v >= 0) & (v < 640)
        col = obj_colors.get(obj, (0, 255, 255))
        for x, y in zip(u[valid], v[valid]):
            cv2.circle(cloud_canvas, (x, y), 1, col, -1)
    cv2.putText(cloud_canvas, "Top-down (base xy)  -500<=x<=+500, 0<=y<=+1000 mm",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(stages_dir / "stage3_fused_clouds_topdown.png"), cloud_canvas)
    print(f"  saved: {stages_dir / 'stage3_fused_clouds_topdown.png'}")

    # Stage 4 & 5: 초기/최종 pose overlay — 사실 우리 파이프라인은 단일 ICP, 둘이 같음.
    # 대신: GLB at ICP-fit pose 와 GLB at original-size 두 가지 비교.
    final_imgs = {}
    for cam_idx in [0, 1, 2, 3]:
        rgb = raw_imgs.get(cam_idx)
        if rgb is None: continue
        K, D, _ = intrinsics[cam_idx]
        if cam_idx == 2:
            T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_path.exists(): continue
            T_be = np.load(T_be_path)
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[cam_idx]
        T_cam_base = np.linalg.inv(T_base_cam)
        out = rgb.copy()
        for obj in objects:
            posed_glb = out_dir / obj / f"{obj}_posed_fused.glb"
            if not posed_glb.exists(): continue
            mesh = trimesh.load(str(posed_glb), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            T_cam_obj = T_cam_base
            out = render_overlay(out, mesh, T_cam_obj, K, D,
                                 color_bgr=obj_colors.get(obj, (0, 255, 255)))
        final_imgs[cam_idx] = out
    cv2.imwrite(str(stages_dir / "stage5_final_pose_overlay.png"),
                _quad(final_imgs, label_prefix="final GLB pose"))
    print(f"  saved: {stages_dir / 'stage5_final_pose_overlay.png'}")

    # Combined comparison (all objects overlayed on ref frame, per cam)
    print("Rendering combined comparison (all objects on ref frame)...")
    combined_imgs = {}
    for cam_idx in [0, 1, 2, 3]:
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{cam_idx}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        K, D, _ = intrinsics[cam_idx]
        if cam_idx == 2:
            T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_path.exists():
                continue
            T_be = np.load(T_be_path)
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[cam_idx]
        T_cam_base = np.linalg.inv(T_base_cam)
        out = rgb.copy()
        for obj in objects:
            sf = out_dir / obj / "pose.json"
            if not sf.exists() or not summary.get(obj, {}).get("pass", False):
                # still overlay even if fail, with red tint
                pass
            if not (out_dir / obj / f"{obj}_posed_fused.glb").exists():
                continue
            mesh = trimesh.load(str(out_dir / obj / f"{obj}_posed_fused.glb"), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            # mesh is already in BASE frame (posed_fused.glb)
            T_cam_obj = T_cam_base  # mesh in base, so transform mesh→cam = inv(T_base_cam)
            out = render_overlay(out, mesh, T_cam_obj, K, D,
                                 color_bgr=obj_colors.get(obj, (0, 255, 255)))
        cv2.rectangle(out, (5, 5), (5 + 12 * 22, 32), (0, 0, 0), -1)
        cv2.putText(out, f"cam{cam_idx}  all objects", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        combined_imgs[cam_idx] = out
    if combined_imgs:
        quad = quad_image(combined_imgs, target_w=640)
        cv2.imwrite(str(out_dir / "comparison.png"), quad)
        print(f"saved combined: {out_dir / 'comparison.png'}")

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    print("\n=== Summary ===")
    print(f"{'obj':>6} {'pass':>5} {'fit':>6} {'rmse_mm':>8} {'ext_ratio':>9} "
          f"{'views':>5}   pos_mm")
    print("-" * 90)
    for obj, s in summary.items():
        if "position_mm" not in s:
            print(f"{obj:>6}  FAIL  ({s.get('reason', '?')})")
            continue
        pos = s["position_mm"]
        flag = "OK" if s["pass"] else "FAIL"
        print(f"{obj:>6} {flag:>5} {s['fit']:6.3f} {s['rmse_mm']:8.2f} "
              f"{s['extent_ratio']:9.2f} {s['n_views_used']:>5}   "
              f"[{pos[0]:+7.1f}, {pos[1]:+7.1f}, {pos[2]:+7.1f}]")
    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
