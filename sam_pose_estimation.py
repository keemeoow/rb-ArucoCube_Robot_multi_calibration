#!/usr/bin/env python3
"""SAM-first 6D pose estimation pipeline.

Step 1: SAM auto-mask per cam image (cam0/1/3: 1 frame each, cam2: all 19 frames)
Step 2: Backproject SAM mask + depth → 3D cluster (uses calibration + intrinsics)
Step 3: Cross-cam clustering by 3D centroid → group masks belonging to same object
Step 4: For each group, fit each GLB candidate (red/cream/blue/box) → identify + pose
Step 5: Render-and-compare IoU refinement (grid search around ICP pose)

사용:
  python3 sam_pose_estimation.py
  python3 sam_pose_estimation.py --skip_refine    # Step 5 생략
  python3 sam_pose_estimation.py --ref_frame 9    # static cam frame 선택
"""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

SAM_CACHE_PATH = Path("data/pose_sam/_sam_cache.pkl")

from fuse_multiframe_pose import (
    CAPTURE_DIR, DATA_DIR,
    load_intrinsics, load_static_transforms,
    backproject, transform_pts, clean_cloud,
    load_glb_as_pcd, initial_T, run_icp_multi_init, icp_with_scale,
)


OUT_DIR = Path("data/pose_sam")
CANDIDATE_OBJECTS = ["red", "cream", "blue", "box"]
WORKSPACE_BBOX_MM = ((-500, 300), (150, 900), (5, 300))  # x, y, z
N_FRAMES = 19

# Per-object color for overlay viz
OBJ_COLORS = {
    "red":   (0, 0, 255),
    "cream": (140, 220, 240),
    "blue":  (255, 200, 100),
    "box":   (240, 240, 240),
}


# ───────────────────────── SAM setup ─────────────────────────

_sam_auto = None


def get_sam_auto_generator():
    global _sam_auto
    if _sam_auto is None:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
        sam.to("cpu")
        sam.eval()
        _sam_auto = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,             # 32 default → 16 으로 가속 (2x 빠름)
            pred_iou_thresh=0.85,
            stability_score_thresh=0.88,
            box_nms_thresh=0.7,
            min_mask_region_area=400,
        )
    return _sam_auto


# ───────────── Step 1-2: SAM mask → 3D cluster ─────────────

def mask_to_3d_cluster(mask, depth, K, depth_scale, T_base_cam,
                       workspace_bbox_mm=WORKSPACE_BBOX_MM,
                       erode_px=2):
    """Mask 픽셀 + depth → base frame 의 3D points.
    erode_px: mask boundary 의 depth edge halo 제거 위한 erosion."""
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8),
                          (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    # Erode mask 2 pixels: depth sensor edge halo (~5-10mm) 제거
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (erode_px * 2 + 1, erode_px * 2 + 1))
        mask = cv2.erode(mask.astype(np.uint8), kernel)
    pts_cam = backproject(depth, mask, K, depth_scale)
    if len(pts_cam) < 30:
        return None
    pts_base = transform_pts(pts_cam, T_base_cam)
    pts_mm = pts_base * 1000
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = workspace_bbox_mm
    inws = ((pts_mm[:, 0] >= xmin) & (pts_mm[:, 0] <= xmax)
            & (pts_mm[:, 1] >= ymin) & (pts_mm[:, 1] <= ymax)
            & (pts_mm[:, 2] >= zmin) & (pts_mm[:, 2] <= zmax))
    if inws.sum() < 30:
        return None
    return pts_base[inws]


def _merge_adjacent_masks(masks, rgb, dilate_px=3, overlap_thr=0.30,
                            color_delta_thr=40.0,
                            max_combined_area_frac=0.35):
    """SAM auto-mask 의 face 분리 문제 해결.
    조건 (3개 모두 충족 시 같은 객체):
      1. dilated mask 끼리 inter > min_area * overlap_thr (인접)
      2. mean color 차이 < color_delta_thr (같은 객체 face 면 색 유사)
      3. 합쳐서도 max_combined_area_frac (image area) 이하 (background 흡수 방지)
    """
    if len(masks) <= 1:
        return masks
    H, W = rgb.shape[:2]
    image_area = H * W
    n = len(masks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_px * 2 + 1, dilate_px * 2 + 1))
    dilated_segs = []
    mean_colors = []
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        dilated_segs.append(cv2.dilate(seg, kernel))
        # Mean RGB of mask area
        mc = rgb[seg.astype(bool)].mean(axis=0) if seg.any() else np.array([0, 0, 0])
        mean_colors.append(mc)
    mean_colors = np.asarray(mean_colors)

    for i in range(n):
        for j in range(i + 1, n):
            inter = int((dilated_segs[i] & dilated_segs[j]).sum())
            min_area = min(masks[i]["area"], masks[j]["area"])
            if inter <= min_area * overlap_thr:
                continue
            # Color similarity (Euclidean RGB distance)
            color_diff = float(np.linalg.norm(mean_colors[i] - mean_colors[j]))
            if color_diff > color_delta_thr:
                continue
            # 병합해도 image 의 35% 이하 (background 흡수 방지)
            combined_area = int((dilated_segs[i] | dilated_segs[j]).sum())
            if combined_area > image_area * max_combined_area_frac:
                continue
            union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idx_list in groups.values():
        combined = np.zeros((H, W), dtype=bool)
        for idx in idx_list:
            combined |= masks[idx]["segmentation"].astype(bool)
        m_int = combined.astype(np.uint8)
        merged.append({
            "segmentation": m_int,
            "area": int(m_int.sum()),
            "bbox": list(cv2.boundingRect(m_int)),
            "n_faces_merged": len(idx_list),
        })
    return merged


def sam_segment_cluster_one_view(sam_gen, rgb_bgr, depth, K, depth_scale, T_base_cam,
                                  min_mask_area_px=400, max_mask_area_px=80000):
    """단일 (cam, frame) view → list of 3D clusters.
    BGR→RGB 변환 + 인접 face 마스크 병합 (입체 객체의 여러 면 → 하나의 mask)."""
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    raw_masks = sam_gen.generate(rgb)
    # 인접 + 색상 유사한 face mask 병합 (입체 객체의 여러 면 → 1 mask)
    masks = _merge_adjacent_masks(raw_masks, rgb, dilate_px=3,
                                    overlap_thr=0.30, color_delta_thr=40.0)
    clusters = []
    for m in masks:
        area = m["area"]
        if area < min_mask_area_px or area > max_mask_area_px:
            continue
        seg = m["segmentation"].astype(np.uint8)
        pts_base = mask_to_3d_cluster(seg, depth, K, depth_scale, T_base_cam)
        if pts_base is None or len(pts_base) < 40:
            continue
        pts_mm = pts_base * 1000
        centroid_mm = pts_mm.mean(axis=0)
        ext_mm = pts_mm.max(0) - pts_mm.min(0)
        # 너무 크면 (table edge 등) 제외
        if max(ext_mm) > 250:
            continue
        clusters.append({
            "pts_base": pts_base,           # meters
            "pts_mm": pts_mm,
            "centroid_mm": centroid_mm,
            "extent_mm": ext_mm,
            "n_pts": int(len(pts_base)),
            "mask_2d": seg,
            "area_px": int(area),
        })
    return clusters


def collect_all_clusters(intrinsics, static_T_base_cam, T_gripper_cam,
                          ref_frame=0, verbose=True, use_cache=True):
    """cam0/1/3 each 1 frame + cam2 all 19 frames → 모든 SAM cluster 수집.
    각 cluster 에는 어느 cam/frame 에서 왔는지 metadata 포함.
    use_cache=True: data/pose_sam/_sam_cache.pkl 있으면 SAM 재실행 없이 reload."""
    if use_cache and SAM_CACHE_PATH.exists():
        cache_key = f"ref_frame_{ref_frame}"
        try:
            with open(SAM_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == cache_key:
                if verbose:
                    print(f"  [cache hit] loading {len(cached['clusters'])} clusters "
                          f"from {SAM_CACHE_PATH}")
                return cached["clusters"]
        except Exception as e:
            if verbose:
                print(f"  [cache miss/err] {e} → re-running SAM")

    sam_gen = get_sam_auto_generator()
    all_clusters = []

    def load_view(ci, fr):
        fid = f"{fr:06d}"
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            return None, None, None
        depth = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"depth_{fid}.png"),
                            cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None, None, None
        if ci == 2:
            T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_path.exists():
                return None, None, None
            T_be = np.load(T_be_path)
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[ci]
        return rgb, depth, T_base_cam

    # Static cams: 1 frame each
    for ci in [0, 1, 3]:
        rgb, depth, T_bc = load_view(ci, ref_frame)
        if rgb is None:
            continue
        K, _, depth_scale = intrinsics[ci]
        cls = sam_segment_cluster_one_view(sam_gen, rgb, depth, K, depth_scale, T_bc)
        for c in cls:
            c["cam"] = ci
            c["frame"] = ref_frame
        if verbose:
            print(f"  cam{ci} fr{ref_frame}: {len(cls)} masks kept (after filters)")
        all_clusters.extend(cls)

    # cam2: 모든 frame (19) 활용 — 더 많은 SAM observation → robust median ↑
    for fr in range(N_FRAMES):
        rgb, depth, T_bc = load_view(2, fr)
        if rgb is None:
            continue
        K, _, depth_scale = intrinsics[2]
        cls = sam_segment_cluster_one_view(sam_gen, rgb, depth, K, depth_scale, T_bc)
        for c in cls:
            c["cam"] = 2
            c["frame"] = fr
        if verbose:
            print(f"  cam2 fr{fr}: {len(cls)} masks kept")
        all_clusters.extend(cls)

    if verbose:
        print(f"Total clusters across all views: {len(all_clusters)}")

    # Save cache for next run
    if use_cache:
        SAM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(SAM_CACHE_PATH, "wb") as f:
                pickle.dump({"key": f"ref_frame_{ref_frame}",
                              "clusters": all_clusters}, f)
            if verbose:
                print(f"  [cache saved] → {SAM_CACHE_PATH}")
        except Exception as e:
            if verbose:
                print(f"  [cache save err] {e}")
    return all_clusters


# ───────────── Step 3: Cross-cam clustering ─────────────

def group_clusters_by_centroid(clusters, eps_mm=30.0, max_extent_mm=180.0):
    """Centroid 가 eps_mm 이내인 cluster 들을 같은 object 로 묶음.
    DBSCAN 으로 centroid 공간에서 grouping. 그룹의 centroid extent 가
    max_extent_mm 를 넘으면 K-means 로 재분할 (chain-link 아티팩트 방지)."""
    if not clusters:
        return []
    centroids = np.array([c["centroid_mm"] for c in clusters])
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(centroids / 1000)  # meters
    labels = np.array(pcd_c.cluster_dbscan(eps=eps_mm / 1000,
                                              min_points=2,
                                              print_progress=False))
    groups = {}
    for cl, lab in zip(clusters, labels):
        if lab < 0:
            key = f"single_{id(cl)}"
        else:
            key = f"g{lab}"
        groups.setdefault(key, []).append(cl)

    def _split_if_oversized(group_clusters, depth=0, max_depth=4):
        """Split if FUSED point extent exceeds max_extent_mm in 2+ axes.
        Single-axis elongation can be legitimate (object viewed from one angle);
        2-axis inflation is the smoking-gun for multi-object chain merge."""
        if len(group_clusters) < 4 or depth >= max_depth:
            return [group_clusters]
        fused = np.vstack([gc["pts_base"] for gc in group_clusters]) * 1000
        ext = fused.max(0) - fused.min(0)
        if int((ext > max_extent_mm).sum()) < 2:
            return [group_clusters]
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print(f"  [WARN] sklearn missing — group split skipped (ext={ext.round(1)})")
            return [group_clusters]
        c = np.array([gc["centroid_mm"] for gc in group_clusters])
        km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(c)
        sub = [[], []]
        for gc, lb in zip(group_clusters, km.labels_):
            sub[int(lb)].append(gc)
        out = []
        for s in sub:
            if s:
                out.extend(_split_if_oversized(s, depth + 1, max_depth))
        return out

    groups_list = []
    for k, gs in groups.items():
        if k.startswith("single_") and gs[0]["n_pts"] < 400:
            continue
        for sub_gs in _split_if_oversized(gs):
            groups_list.append(sub_gs)
    return groups_list


# ───────────── Step 4: Multi-GLB ID + ICP ─────────────

def snap_to_ground(T_base_obj, mesh, table_z_m=0.0):
    """Physical prior: 객체는 table 위에 놓임 → GLB 최저점 z = table_z.
    Cluster centroid 가 top-biased 라 GLB center 가 진짜 중심보다 위에 placed
    되는 systematic offset 보정 (z 만 조정).
    """
    verts_obj = np.asarray(mesh.vertices)
    R = T_base_obj[:3, :3].copy()
    # Strip scale (pure rotation)
    U, _, Vt = np.linalg.svd(R)
    R_pure = U @ Vt
    if np.linalg.det(R_pure) < 0:
        U[:, -1] *= -1
        R_pure = U @ Vt
    verts_base = (R_pure @ verts_obj.T).T + T_base_obj[:3, 3]
    z_min = float(verts_base[:, 2].min())
    T_new = T_base_obj.copy()
    T_new[2, 3] += (table_z_m - z_min)
    return T_new


def compute_iou_for_T(T_base_obj, mesh, cluster_observations,
                       intrinsics, static_T_base_cam, T_gripper_cam,
                       sam_edge_px=0):
    """주어진 T_base_obj 에서 각 cluster 의 SAM mask 와 rendered silhouette IoU.
    Mask area 로 weighted average (큰 mask 더 신뢰).
    sam_edge_px > 0: SAM mask 를 N px erosion (boundary 보수화).
    sam_edge_px < 0: SAM mask 를 |N| px dilation."""
    weighted_iou = 0.0
    total_weight = 0.0
    n = 0
    kernel = None
    if sam_edge_px != 0:
        k = abs(int(sam_edge_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
    for c in cluster_observations:
        ci = c["cam"]
        fr = c["frame"]
        fid = f"{fr:06d}"
        if ci == 2:
            T_be_p = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_p.exists():
                continue
            T_bc = np.load(T_be_p) @ T_gripper_cam
        else:
            T_bc = static_T_base_cam[ci]
        K, D, _ = intrinsics[ci]
        T_cam_obj = np.linalg.inv(T_bc) @ T_base_obj
        sam = c["mask_2d"]
        if kernel is not None:
            sam_u8 = (sam > 0).astype(np.uint8)
            if sam_edge_px > 0:
                sam = cv2.erode(sam_u8, kernel)
            else:
                sam = cv2.dilate(sam_u8, kernel)
        sil = project_mesh_silhouette(mesh, T_cam_obj, K, D, sam.shape)
        if sil.sum() < 30 or sam.sum() < 30:
            continue
        iou = iou_2d(sil, sam)
        w = float(sam.sum())  # mask area 로 가중
        weighted_iou += iou * w
        total_weight += w
        n += 1
    if total_weight == 0:
        return 0.0, 0
    return float(weighted_iou / total_weight), n


def fit_each_glb(fused_cloud_pts_base, glb_paths, voxel_m=0.002,
                  ds_eps_m=0.012, yaw_grid=24, verbose=False,
                  cluster_observations=None, intrinsics=None,
                  static_T_base_cam=None, T_gripper_cam=None):
    """각 candidate GLB 로 ICP fit + IoU scoring → 최고 score 의 결과 반환.
    cluster_observations 가 주어지면 IoU 가 주요 metric, 아니면 ICP fit/rmse."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_cloud_pts_base)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_m)
    # Cluster cleanup (largest)
    labels = np.array(pcd.cluster_dbscan(eps=ds_eps_m, min_points=30,
                                            print_progress=False))
    if (labels >= 0).any():
        biggest = np.bincount(labels[labels >= 0]).argmax()
        pcd = pcd.select_by_index(np.where(labels == biggest)[0])
    if len(pcd.points) < 50:
        return None

    target_arr = np.asarray(pcd.points)
    target_ext_mm = (target_arr.max(0) - target_arr.min(0)) * 1000

    # Target 의 모든 3 dim 정렬 (cluster 가 노이즈 inflate 가능, halo 5~15mm)
    target_ext_sorted = np.sort(target_ext_mm)[::-1]

    results = []
    for obj_name, glb_path in glb_paths.items():
        if not Path(glb_path).exists():
            continue
        # ── 1단계: RAW GLB 로 size gate (객체 식별용)
        raw_pcd, _ = load_glb_as_pcd(glb_path, n_samples=2000, center=True)
        raw_arr = np.asarray(raw_pcd.points)
        raw_ext_mm = (raw_arr.max(0) - raw_arr.min(0)) * 1000
        raw_ext_sorted = np.sort(raw_ext_mm)[::-1]
        # 임시 변수 (gate 검사용)
        model_ext_sorted = raw_ext_sorted
        model_ext_mm = raw_ext_mm

        # ── Per-axis size gate (sorted desc): top-down view 에서 z 축(=가장 작은 dim)
        # 은 자연스럽게 absent → 더 lenient. 큰 축은 halo 로 inflated → upper bound 여유.
        # dim0 (longest): [0.65, 1.7] — halo 까지 허용
        # dim1 (middle):  [0.5, 1.7]
        # dim2 (shortest, 자주 z): [0.25, 1.7] — top-down 시 절반 이하로 보일 수 있음
        per_axis_ratio = target_ext_sorted / (model_ext_sorted + 1e-4)
        AXIS_LO = np.array([0.65, 0.5, 0.25])
        AXIS_HI = np.array([1.7, 1.7, 1.7])
        if (np.any(per_axis_ratio > AXIS_HI)
                or np.any(per_axis_ratio < AXIS_LO)):
            if verbose:
                print(f"    skip {obj_name}: per-axis ratio {per_axis_ratio.round(2)} "
                      f"out of [{list(AXIS_LO)}, {list(AXIS_HI)}]")
            continue

        # ── 2단계: RAW GLB 로 ICP (식별 위해)
        model_pcd, mesh = load_glb_as_pcd(glb_path, n_samples=6000, center=True)
        model_arr = np.asarray(model_pcd.points)
        model_ext_mm = (model_arr.max(0) - model_arr.min(0)) * 1000

        T0 = initial_T(model_pcd, pcd)
        # Note: uniform scaling 은 x/y 도 줄여서 IoU 악화 (실험 확인) → rigid ICP 유지
        icp_res = run_icp_multi_init(
            model_pcd, pcd, T0,
            max_corr_coarse=0.020, max_corr_fine=0.005,
            yaw_grid=yaw_grid)
        T_snapped = icp_res.transformation

        # ── Z 진단: 실제 cluster 와 GLB의 z 범위 비교
        cluster_z = target_arr[:, 2]
        verts_b = (T_snapped[:3, :3] @ np.asarray(mesh.vertices).T).T + T_snapped[:3, 3]
        glb_z = verts_b[:, 2]
        if verbose:
            print(f"    [{obj_name} z-diag] cluster z=[{cluster_z.min()*1000:.1f},"
                  f"{cluster_z.max()*1000:.1f}] (mean {cluster_z.mean()*1000:.1f})  "
                  f"GLB z=[{glb_z.min()*1000:.1f},{glb_z.max()*1000:.1f}] "
                  f"(center T={T_snapped[2,3]*1000:.1f})")
        ratio = float(max(target_ext_mm) / (max(model_ext_mm) + 1e-4))

        # ── IoU 계산 (cluster_observations 있을 때 — 주요 metric)
        iou_mean = 0.0
        iou_n = 0
        if cluster_observations is not None and intrinsics is not None:
            iou_mean, iou_n = compute_iou_for_T(
                T_snapped, mesh, cluster_observations,
                intrinsics, static_T_base_cam, T_gripper_cam)

        # Size match (보조 metric)
        xy_dev = float(np.mean(np.abs(per_axis_ratio[:2] - 1.0)))
        z_dev = float(np.abs(per_axis_ratio[2] - 1.0))
        size_match = float(np.exp(-2.0 * xy_dev - 0.3 * z_dev))

        # Score: IoU × (1 + size_match) × icp_fit
        # IoU 가 비슷할 때 size_match 와 ICP fit 으로 차별화 (작은 GLB 의 false-positive 방지)
        if iou_n > 0:
            score = iou_mean * (1.0 + size_match) * float(icp_res.fitness)
        else:
            score = (icp_res.fitness / (icp_res.inlier_rmse + 1e-4)) * size_match

        results.append({
            "obj": obj_name,
            "T": T_snapped,
            "fit": float(icp_res.fitness),
            "rmse_mm": float(icp_res.inlier_rmse * 1000),
            "extent_ratio": float(ratio),
            "iou_mean": iou_mean,
            "iou_n": iou_n,
            "size_match": size_match,
            "score": float(score),
            "mesh": mesh,
            "model_ext_mm": model_ext_mm,
            "target_ext_mm": target_ext_mm,
            "target_pcd": pcd,
        })
        if verbose:
            print(f"    {obj_name}: fit={icp_res.fitness:.3f}  "
                  f"rmse={icp_res.inlier_rmse*1000:5.2f}mm  "
                  f"ext_ratio={ratio:4.2f}  IoU={iou_mean:.3f} (n={iou_n})  "
                  f"score={score:.3f}")
    if not results:
        return None
    return max(results, key=lambda r: r["score"])


# ───────────── Step 5: Render-and-compare IoU refinement ─────────────

_MESH_CACHE_LOWPOLY = {}


def get_lowpoly_mesh(mesh, target_faces=800):
    """Decimate mesh for fast silhouette rendering. Cached by id(mesh)."""
    key = id(mesh)
    if key in _MESH_CACHE_LOWPOLY:
        return _MESH_CACHE_LOWPOLY[key]
    n_faces = len(mesh.faces)
    if n_faces <= target_faces:
        _MESH_CACHE_LOWPOLY[key] = mesh
        return mesh
    try:
        # trimesh decimation
        ratio = target_faces / n_faces
        m_low = mesh.simplify_quadric_decimation(face_count=target_faces)
        _MESH_CACHE_LOWPOLY[key] = m_low
        return m_low
    except Exception:
        _MESH_CACHE_LOWPOLY[key] = mesh
        return mesh


def project_mesh_silhouette(mesh, T_cam_obj, K, D, img_shape, strip_scale=False):
    """Vectorized: batch projection + single fillPoly call for all triangles.
    strip_scale=False: T_cam_obj 의 3x3 에 포함된 scale 사용 (ICP with_scaling 결과 유지)."""
    h, w = img_shape[:2]
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    R = T_cam_obj[:3, :3].copy()
    if strip_scale:
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
    verts_cam = (R @ verts.T).T + T_cam_obj[:3, 3]
    z = verts_cam[:, 2]
    if (z > 0.05).sum() < 3:
        return np.zeros((h, w), dtype=np.uint8)
    pts2d, _ = cv2.projectPoints(
        verts_cam.astype(np.float64),
        np.zeros(3), np.zeros(3),
        K.astype(np.float64), D.astype(np.float64))
    pts2d = pts2d.reshape(-1, 2)
    faces = np.asarray(mesh.faces)

    # Vectorized triangle filter: all 3 vertices z>0.05 AND in/near image
    z_ok = (z[faces[:, 0]] > 0.05) & (z[faces[:, 1]] > 0.05) & (z[faces[:, 2]] > 0.05)
    tri_pts = pts2d[faces]  # (N_faces, 3, 2)
    # bbox check
    xs = tri_pts[..., 0]
    ys = tri_pts[..., 1]
    in_x = (xs.min(1) > -200) & (xs.max(1) < w + 200)
    in_y = (ys.min(1) > -200) & (ys.max(1) < h + 200)
    valid = z_ok & in_x & in_y
    if not valid.any():
        return np.zeros((h, w), dtype=np.uint8)
    tri_kept = tri_pts[valid].astype(np.int32)
    sil = np.zeros((h, w), dtype=np.uint8)
    # cv2.fillPoly: 한 번에 모든 triangle 채움 (Python 루프 제거)
    cv2.fillPoly(sil, [t for t in tri_kept], 255)
    return sil


def iou_2d(mask_a, mask_b):
    inter = ((mask_a > 0) & (mask_b > 0)).sum()
    union = ((mask_a > 0) | (mask_b > 0)).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _single_view_iou_refine(T_init, mesh, sam_mask, K, D, T_base_cam, shape,
                              dxy=(-6, -3, 0, 3, 6), dyaw=(-12, -6, 0, 6, 12)):
    """Single cam 의 SAM mask 와 GLB silhouette IoU 최대화 (grid search).
    오직 1 cam 의 정보만 사용 → 빠르고 단순."""
    from scipy.spatial.transform import Rotation as Rot
    best_T = T_init.copy()
    best_iou = 0.0
    for dx in dxy:
        for dy in dxy:
            for ang in dyaw:
                T = T_init.copy()
                Rz = Rot.from_euler('z', ang, degrees=True).as_matrix()
                T[:3, :3] = Rz @ T_init[:3, :3]
                T[0, 3] = T_init[0, 3] + dx / 1000
                T[1, 3] = T_init[1, 3] + dy / 1000
                T_cam_obj = np.linalg.inv(T_base_cam) @ T
                sil = project_mesh_silhouette(mesh, T_cam_obj, K, D, shape)
                if sil.sum() < 50:
                    continue
                iou = iou_2d(sil, sam_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_T = T
    return best_T, best_iou


def _robust_se3_median(Ts):
    """SE3 robust median: translation 은 median, rotation 은 quaternion mean."""
    if not Ts:
        return None
    Ts_arr = np.asarray(Ts)
    # Translation: median per axis
    t_med = np.median(Ts_arr[:, :3, 3], axis=0)
    # Rotation: quaternion mean (scipy)
    from scipy.spatial.transform import Rotation as Rot
    Rs = []
    for T in Ts:
        R = T[:3, :3]
        U, _, Vt = np.linalg.svd(R)
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0:
            U[:, -1] *= -1
            Rp = U @ Vt
        Rs.append(Rp)
    try:
        R_mean = Rot.from_matrix(np.asarray(Rs)).mean().as_matrix()
    except Exception:
        R_mean = Rs[0]
    T_out = np.eye(4)
    T_out[:3, :3] = R_mean
    T_out[:3, 3] = t_med
    return T_out


def iou_refine(T_init, mesh, ref_observations, ground_snap=True):
    """Joint IoU grid search (2-pass coarse-to-fine).
    각 (dxy, dyaw) 후보 → 모든 cam SAM mask 와의 평균 IoU 계산 → 최대 선택.
    scipy 보다 robust (local min 회피), per-cam median 보다 strict (joint criterion).
    """
    from scipy.spatial.transform import Rotation as Rot

    def _apply_delta(T_anchor, dx_mm, dy_mm, dz_mm, dyaw_deg):
        T = T_anchor.copy()
        Rz = Rot.from_euler('z', dyaw_deg, degrees=True).as_matrix()
        T[:3, :3] = Rz @ T_anchor[:3, :3]
        T[0, 3] = T_anchor[0, 3] + dx_mm / 1000
        T[1, 3] = T_anchor[1, 3] + dy_mm / 1000
        T[2, 3] = T_anchor[2, 3] + dz_mm / 1000
        return T

    def _joint_iou_weighted(T, current_mesh):
        """모든 cam SAM mask 와의 IoU (mask area 가중 평균)."""
        weighted = 0.0
        total_w = 0.0
        for sam_mask, K, D, T_base_cam, shape in ref_observations:
            T_cam_obj = np.linalg.inv(T_base_cam) @ T
            sil = project_mesh_silhouette(current_mesh, T_cam_obj, K, D, shape)
            if sil.sum() < 50:
                continue
            iou = iou_2d(sil, sam_mask)
            w = float(sam_mask.sum())
            weighted += iou * w
            total_w += w
        return weighted / total_w if total_w > 0 else 0.0

    def _joint_iou(T):
        return _joint_iou_weighted(T, mesh)

    def _grid_search(T_anchor, dxy_grid, dz_grid, dyaw_grid):
        best_T = T_anchor.copy()
        best_iou = _joint_iou(T_anchor)
        for dx in dxy_grid:
            for dy in dxy_grid:
                for dz in dz_grid:
                    for dyaw in dyaw_grid:
                        T = _apply_delta(T_anchor, dx, dy, dz, dyaw)
                        iou = _joint_iou(T)
                        if iou > best_iou:
                            best_iou = iou
                            best_T = T
        return best_T, best_iou

    # Pass 1: 넓은 coarse grid (5×5×5 = 125)
    T1, iou1 = _grid_search(
        T_init,
        dxy_grid=(-12, -6, 0, 6, 12),
        dz_grid=(0,),
        dyaw_grid=(-20, -10, 0, 10, 20))

    # Pass 2: 좁은 fine grid (5×5×5 = 125)
    T2, iou2 = _grid_search(
        T1,
        dxy_grid=(-3, -1.5, 0, 1.5, 3),
        dz_grid=(0,),
        dyaw_grid=(-4, -2, 0, 2, 4))

    # Pass 3: scale grid search at T2 (GLB size 가 SAM mask 와 맞도록 학습)
    raw_verts = np.asarray(mesh.vertices).copy()
    raw_faces = np.asarray(mesh.faces).copy()

    def _make_scaled_mesh(scale_xyz):
        new_verts = raw_verts * scale_xyz
        return trimesh.Trimesh(vertices=new_verts, faces=raw_faces, process=False)

    def _joint_iou_scaled(scale_xyz, T):
        return _joint_iou_weighted(T, _make_scaled_mesh(scale_xyz))

    scale_grid_coarse = (0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20)
    scale_grid_fine = (0.92, 0.96, 0.98, 1.0, 1.02, 1.04, 1.08)

    best_scale = np.array([1.0, 1.0, 1.0])
    best_T = T2
    best_iou_final = iou2

    # ── Pass 3a: coarse scale at T2
    for sx in scale_grid_coarse:
        for sy in scale_grid_coarse:
            for sz in scale_grid_coarse:
                iou_s = _joint_iou_scaled(np.array([sx, sy, sz]), best_T)
                if iou_s > best_iou_final:
                    best_iou_final = iou_s
                    best_scale = np.array([sx, sy, sz])

    # ── Pass 3b: T re-optimization with current best_scale (mesh 바뀌면 T 도 살짝 이동)
    scaled_mesh_cur = _make_scaled_mesh(best_scale)
    def _eval_T_with_scaled(T):
        return _joint_iou_weighted(T, scaled_mesh_cur)
    # Fine T grid around best_T
    for dx in (-3, -1.5, 0, 1.5, 3):
        for dy in (-3, -1.5, 0, 1.5, 3):
            for dyaw in (-4, -2, 0, 2, 4):
                T_test = _apply_delta(best_T, dx, dy, 0, dyaw)
                iou_t = _eval_T_with_scaled(T_test)
                if iou_t > best_iou_final:
                    best_iou_final = iou_t
                    best_T = T_test

    # ── Pass 3c: fine scale at best_T
    for sx in scale_grid_fine:
        for sy in scale_grid_fine:
            for sz in scale_grid_fine:
                cand_scale = np.array([sx, sy, sz]) * best_scale
                iou_s = _joint_iou_scaled(cand_scale, best_T)
                if iou_s > best_iou_final:
                    best_iou_final = iou_s
                    best_scale = cand_scale

    # 최종 mesh 갱신 (scale 적용)
    if not np.allclose(best_scale, 1.0):
        scaled_final = _make_scaled_mesh(best_scale)
        mesh.vertices = scaled_final.vertices

    return best_T, best_iou_final


# ───────────── Visualization ─────────────

def make_4cam_grid(per_cam_imgs, target_w=720, labels=None):
    """4-cam 이미지를 2x2 grid 로 결합. labels=cam 별 텍스트 (선택)."""
    tiles = []
    for ci in [0, 1, 2, 3]:
        img = per_cam_imgs.get(ci)
        if img is None:
            tiles.append(np.zeros((400, target_w, 3), dtype=np.uint8))
            continue
        h, w = img.shape[:2]
        th = int(h * target_w / w)
        t = cv2.resize(img, (target_w, th), interpolation=cv2.INTER_AREA)
        if labels and labels.get(ci):
            cv2.rectangle(t, (5, 5), (5 + 14 * len(labels[ci]) + 10, 36),
                          (0, 0, 0), -1)
            cv2.putText(t, labels[ci], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    # Pad to same height
    max_h = max(t.shape[0] for t in tiles)
    tiles = [
        np.pad(t, ((0, max_h - t.shape[0]), (0, 0), (0, 0)),
               mode='constant', constant_values=0)
        for t in tiles
    ]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def save_stage1_rgb(intrinsics, static_T_base_cam, T_gripper_cam,
                     ref_frame=0, out_path=None):
    """Stage 1: 입력 RGB images (4 cam)."""
    per_cam = {}
    for ci in [0, 1, 2, 3]:
        fid = f"{ref_frame:06d}"
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is not None:
            per_cam[ci] = rgb
    labels = {ci: f"cam{ci} RGB" for ci in [0, 1, 2, 3]}
    grid = make_4cam_grid(per_cam, labels=labels)
    if out_path:
        cv2.imwrite(str(out_path), grid)
    return grid


def save_stage2_sam(all_clusters, intrinsics, static_T_base_cam, T_gripper_cam,
                     ref_frame=0, out_path=None):
    """Stage 2: SAM auto-masks overlaid on RGB (모든 mask 를 색상별 채움)."""
    np.random.seed(42)
    palette = (np.random.rand(50, 3) * 200 + 55).astype(np.uint8)
    per_cam = {}
    for ci in [0, 1, 2, 3]:
        # Static cam: ref_frame; cam2: ref_frame
        fid = f"{ref_frame:06d}"
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        out = rgb.copy()
        for idx, c in enumerate(all_clusters):
            if c["cam"] != ci or c["frame"] != ref_frame:
                continue
            mask = c["mask_2d"]
            color = palette[idx % 50].tolist()
            color_img = np.zeros_like(out)
            color_img[:, :] = color
            mask3 = (mask > 0)[..., None]
            out = np.where(mask3, (out * 0.5 + color_img * 0.5).astype(np.uint8), out)
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
        per_cam[ci] = out
    labels = {ci: f"cam{ci} SAM masks" for ci in [0, 1, 2, 3]}
    grid = make_4cam_grid(per_cam, labels=labels)
    if out_path:
        cv2.imwrite(str(out_path), grid)
    return grid


def save_stage3_clusters(groups, intrinsics, static_T_base_cam, T_gripper_cam,
                          ref_frame=0, out_path=None):
    """Stage 3: 3D cluster groups (object 별 색상) projected to image."""
    obj_colors_list = [(0, 0, 255), (140, 220, 240), (255, 200, 100),
                        (240, 240, 240), (100, 255, 100), (200, 100, 255),
                        (255, 100, 100), (100, 100, 255)]
    per_cam = {}
    for ci in [0, 1, 2, 3]:
        fid = f"{ref_frame:06d}"
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        if ci == 2:
            T_be_p = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_p.exists():
                continue
            T_bc = np.load(T_be_p) @ T_gripper_cam
        else:
            T_bc = static_T_base_cam[ci]
        K, D, _ = intrinsics[ci]
        out = rgb.copy()
        for gi, gs in enumerate(groups):
            n_pts_total = sum(c["n_pts"] for c in gs)
            if n_pts_total < 1000:
                continue  # skip small noise groups
            color = obj_colors_list[gi % len(obj_colors_list)]
            # Fuse points
            pts = np.vstack([c["pts_base"] for c in gs])
            pts_cam = (np.linalg.inv(T_bc)[:3, :3] @ pts.T).T + np.linalg.inv(T_bc)[:3, 3]
            z = pts_cam[:, 2]
            valid = z > 0.05
            if valid.sum() < 5:
                continue
            pts2d, _ = cv2.projectPoints(
                pts_cam[valid].astype(np.float64),
                np.zeros(3), np.zeros(3),
                K.astype(np.float64), D.astype(np.float64))
            pts2d = pts2d.reshape(-1, 2).astype(np.int32)
            h, w = out.shape[:2]
            in_img = ((pts2d[:, 0] >= 0) & (pts2d[:, 0] < w)
                      & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h))
            for p in pts2d[in_img][::5]:  # every 5th pt
                cv2.circle(out, tuple(p), 1, color, -1)
        per_cam[ci] = out
    labels = {ci: f"cam{ci} 3D clusters" for ci in [0, 1, 2, 3]}
    grid = make_4cam_grid(per_cam, labels=labels)
    if out_path:
        cv2.imwrite(str(out_path), grid)
    return grid


def save_stage_pose(obj_poses, intrinsics, static_T_base_cam, T_gripper_cam,
                     ref_frame=0, out_path=None, stage_label="Stage4"):
    """Pose overlay: SAM mask outline (cyan) + GLB silhouette (object color)."""
    per_cam = {}
    for ci in [0, 1, 2, 3]:
        fid = f"{ref_frame:06d}"
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        if ci == 2:
            T_be_p = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
            if not T_be_p.exists():
                continue
            T_bc = np.load(T_be_p) @ T_gripper_cam
        else:
            T_bc = static_T_base_cam[ci]
        K, D, _ = intrinsics[ci]
        out = rgb.copy()
        for obj, info in obj_poses.items():
            # Draw SAM mask outline (cyan) for cluster in this cam (any frame)
            for c in info["clusters"]:
                if c["cam"] != ci:
                    continue
                if c["frame"] != ref_frame and ci != 2:
                    continue
                # cam2 mask 는 cam2 frame 마다 다름 → ref_frame 만 표시
                if ci == 2 and c["frame"] != ref_frame:
                    continue
                contours, _ = cv2.findContours(c["mask_2d"].astype(np.uint8),
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(out, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
            # Draw GLB silhouette outline (object color)
            mesh = info["mesh"]
            T = info["T"]
            T_cam_obj = np.linalg.inv(T_bc) @ T
            sil = project_mesh_silhouette(mesh, T_cam_obj, K, D, rgb.shape)
            if sil.sum() < 30:
                continue
            color = OBJ_COLORS.get(obj, (0, 255, 255))
            color_img = np.zeros_like(out)
            color_img[:, :] = color
            mask3 = (sil > 0)[..., None]
            out = np.where(mask3, (out * 0.65 + color_img * 0.35).astype(np.uint8), out)
            contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
        per_cam[ci] = out
    labels = {ci: f"cam{ci} {stage_label}" for ci in [0, 1, 2, 3]}
    grid = make_4cam_grid(per_cam, labels=labels)
    if out_path:
        cv2.imwrite(str(out_path), grid)
    return grid


def render_overlay_image(obj_poses, intrinsics, static_T_base_cam, T_gripper_cam,
                         ref_frame=0, out_path=None):
    """4-cam grid: 각 cam 의 RGB 에 모든 객체 GLB silhouette 오버레이."""
    fid = f"{ref_frame:06d}"
    tiles = []
    for ci in [0, 1, 2, 3]:
        rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
        if rgb is None:
            continue
        if ci == 2:
            T_be = np.load(CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy")
            T_base_cam = T_be @ T_gripper_cam
        else:
            T_base_cam = static_T_base_cam[ci]
        K, D, _ = intrinsics[ci]
        out = rgb.copy()
        for obj, info in obj_poses.items():
            mesh = info["mesh"]
            T = info["T"]
            T_cam_obj = np.linalg.inv(T_base_cam) @ T
            sil = project_mesh_silhouette(mesh, T_cam_obj, K, D, rgb.shape)
            if sil.sum() < 30:
                continue
            color = OBJ_COLORS.get(obj, (0, 255, 255))
            color_img = np.zeros_like(out)
            color_img[:, :] = color
            mask3 = (sil > 0)[..., None]
            out = np.where(mask3, (out * 0.55 + color_img * 0.45).astype(np.uint8), out)
            contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
        cv2.rectangle(out, (5, 5), (200, 32), (0, 0, 0), -1)
        cv2.putText(out, f"cam{ci}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        h, w = out.shape[:2]
        th = int(h * 720 / w)
        tiles.append(cv2.resize(out, (720, th), interpolation=cv2.INTER_AREA))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    if out_path:
        cv2.imwrite(str(out_path), grid)
        print(f"saved overlay: {out_path}")
    return grid


# ───────────── Main ─────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref_frame", type=int, default=0,
                     help="static cam frame 선택 (cam0/1/3 용)")
    ap.add_argument("--skip_refine", action="store_true",
                     help="Step 5 IoU refinement 생략")
    ap.add_argument("--no_cache", action="store_true",
                     help="SAM mask cache 무시, 다시 segment")
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = out_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    print("Loading calibration & intrinsics ...")
    intrinsics = load_intrinsics()
    static_T_base_cam, T_gripper_cam = load_static_transforms()

    glb_paths = {obj: DATA_DIR / f"{obj}.glb" for obj in CANDIDATE_OBJECTS}
    missing = [o for o, p in glb_paths.items() if not p.exists()]
    if missing:
        print(f"WARN: missing GLBs: {missing}")
        glb_paths = {o: p for o, p in glb_paths.items() if p.exists()}

    print(f"\n=== Step 1-2: SAM segmentation + 3D backprojection ===")
    print(f"  static cams 0,1,3 use frame {args.ref_frame} only; cam2 uses all {N_FRAMES} frames")
    all_clusters = collect_all_clusters(
        intrinsics, static_T_base_cam, T_gripper_cam,
        ref_frame=args.ref_frame, verbose=True,
        use_cache=not args.no_cache)

    if not all_clusters:
        print("No clusters found. Exiting.")
        return

    # ── Stage 1: 원본 RGB
    print(f"\n  saving Stage 1: input RGB → {stages_dir}/stage1_input_rgb.png")
    save_stage1_rgb(intrinsics, static_T_base_cam, T_gripper_cam,
                     ref_frame=args.ref_frame,
                     out_path=stages_dir / "stage1_input_rgb.png")
    # ── Stage 2: SAM masks
    print(f"  saving Stage 2: SAM masks → {stages_dir}/stage2_sam_masks.png")
    save_stage2_sam(all_clusters, intrinsics, static_T_base_cam, T_gripper_cam,
                     ref_frame=args.ref_frame,
                     out_path=stages_dir / "stage2_sam_masks.png")

    print(f"\n=== Step 3: Cross-view clustering by 3D centroid ===")
    groups = group_clusters_by_centroid(all_clusters, eps_mm=30.0)
    print(f"  {len(groups)} object group(s) found")
    for i, gs in enumerate(groups):
        cams_used = sorted(set(c["cam"] for c in gs))
        n_pts_total = sum(c["n_pts"] for c in gs)
        centroids = np.array([c["centroid_mm"] for c in gs])
        center = centroids.mean(axis=0)
        print(f"  group {i}: {len(gs)} masks  cams={cams_used}  "
              f"total_pts={n_pts_total}  center≈[{center[0]:.0f},{center[1]:.0f},{center[2]:.0f}]mm")

    # ── Stage 3: 3D clusters projected to image
    print(f"\n  saving Stage 3: 3D clusters → {stages_dir}/stage3_3d_clusters.png")
    save_stage3_clusters(groups, intrinsics, static_T_base_cam, T_gripper_cam,
                          ref_frame=args.ref_frame,
                          out_path=stages_dir / "stage3_3d_clusters.png")

    print(f"\n=== Step 4: Multi-GLB ID via Hungarian assignment ===")
    # 큰 group 만 후보 (n_pts >= 5000) — noise group 제외
    main_groups = [gs for gs in groups if sum(c["n_pts"] for c in gs) >= 5000]
    main_groups.sort(key=lambda gs: -sum(c["n_pts"] for c in gs))
    print(f"  main groups (n_pts ≥ 5000): {len(main_groups)}")

    # ── 모든 (group, GLB) 쌍에 대해 IoU + ICP 계산 → cost matrix
    all_glbs = list(glb_paths.keys())
    n_groups = len(main_groups)
    n_glbs = len(all_glbs)
    # cost[g, o] = -score (낮을수록 좋음) ; -inf 면 invalid
    cost = np.full((n_groups, n_glbs), 1e9)
    all_results = {}  # (group_idx, obj_name) → result dict

    for gi, gs in enumerate(main_groups):
        fused = np.vstack([c["pts_base"] for c in gs])
        print(f"\n  group {gi} (n_pts={len(fused)}):")
        for oi, obj_name in enumerate(all_glbs):
            single_candidate = {obj_name: glb_paths[obj_name]}
            res = fit_each_glb(
                fused, single_candidate, verbose=False,
                cluster_observations=gs,
                intrinsics=intrinsics,
                static_T_base_cam=static_T_base_cam,
                T_gripper_cam=T_gripper_cam)
            if res is None:
                print(f"    {obj_name}: skipped (size gate)")
                continue
            cost[gi, oi] = -res["score"]
            all_results[(gi, oi)] = res
            print(f"    {obj_name}: IoU={res['iou_mean']:.3f}  "
                  f"size_match={res['size_match']:.3f}  fit={res['fit']:.3f}  "
                  f"score={res['score']:.3f}")

    # ── Hungarian assignment (전체 매칭 동시 최적화)
    from scipy.optimize import linear_sum_assignment
    # Pad cost if more groups than GLBs or vice versa (linear_sum_assignment handles unbalanced)
    row_ind, col_ind = linear_sum_assignment(cost)

    object_poses = {}
    print(f"\n  Hungarian assignment:")
    for gi, oi in zip(row_ind, col_ind):
        if cost[gi, oi] >= 1e9:
            print(f"    group {gi} ↔ {all_glbs[oi]}: NO VALID FIT")
            continue
        obj_name = all_glbs[oi]
        res = all_results[(gi, oi)]
        res["group_idx"] = int(gi)
        res["clusters"] = main_groups[gi]
        object_poses[obj_name] = res
        T = res["T"]
        print(f"    group {gi} ↔ {obj_name}: IoU={res['iou_mean']:.3f}  "
              f"pos=[{T[0,3]*1000:+.1f},{T[1,3]*1000:+.1f},{T[2,3]*1000:+.1f}]mm")

    if not object_poses:
        print("\nNo objects identified. Exiting.")
        return

    # ── Stage 4: ICP 초기 pose
    print(f"\n  saving Stage 4: initial ICP pose → {stages_dir}/stage4_icp_pose.png")
    save_stage_pose(object_poses, intrinsics, static_T_base_cam, T_gripper_cam,
                     ref_frame=args.ref_frame,
                     out_path=stages_dir / "stage4_icp_pose.png",
                     stage_label="Step4 ICP")

    # ── 식별 후 객체별 best prescale 자동 선택
    # 3가지 전략 (raw / z-only / 3-axis) 각각 IoU 측정 → 최고 선택
    print(f"\n=== Post-ID: per-object best prescale 선택 ===")

    def _eval_iou_for_mesh(test_mesh, T, clusters, intrinsics, static_T_base_cam,
                            T_gripper_cam):
        """주어진 mesh + T 에 대해 평균 IoU 계산."""
        iou, _ = compute_iou_for_T(T, test_mesh, clusters, intrinsics,
                                     static_T_base_cam, T_gripper_cam)
        return iou

    for obj, info in object_poses.items():
        glb_path = glb_paths[obj]
        raw_mesh = info["mesh"]
        raw_ext_mm = (np.asarray(raw_mesh.vertices).max(0)
                      - np.asarray(raw_mesh.vertices).min(0)) * 1000
        target_ext_mm = info["target_ext_mm"]

        candidates = {}
        # Strategy 1: raw GLB (no scaling)
        candidates["raw"] = (raw_mesh, np.array([1.0, 1.0, 1.0]))

        # Strategy 2: Z-only (smallest axis 만 cluster smallest 에 맞춤)
        raw_min_axis = int(np.argmin(raw_ext_mm))
        cluster_min = float(np.min(target_ext_mm))
        z_scale = cluster_min / (raw_ext_mm[raw_min_axis] + 1e-4)
        if 0.4 <= z_scale <= 1.2:
            prescale_z = np.ones(3)
            prescale_z[raw_min_axis] = z_scale
            _, mesh_z = load_glb_as_pcd(glb_path, n_samples=2000, center=True,
                                          prescale=prescale_z)
            candidates["z_only"] = (mesh_z, prescale_z)

        # Strategy 3: 3-axis (no halo)
        glb_order = np.argsort(-raw_ext_mm)
        target_sorted = np.sort(target_ext_mm)[::-1]
        prescale_3 = np.ones(3)
        for k in range(3):
            glb_ax = int(glb_order[k])
            scale_k = target_sorted[k] / (raw_ext_mm[glb_ax] + 1e-4)
            prescale_3[glb_ax] = float(np.clip(scale_k, 0.4, 1.3))
        _, mesh_3 = load_glb_as_pcd(glb_path, n_samples=2000, center=True,
                                      prescale=prescale_3)
        candidates["3axis"] = (mesh_3, prescale_3)

        # IoU 측정 후 best 선택
        T = info["T"]
        best_name = "raw"
        best_iou = 0.0
        best_mesh = raw_mesh
        best_scale = np.array([1.0, 1.0, 1.0])
        for name, (cand_mesh, cand_scale) in candidates.items():
            iou_c = _eval_iou_for_mesh(cand_mesh, T, info["clusters"], intrinsics,
                                         static_T_base_cam, T_gripper_cam)
            print(f"  {obj}: {name:6s} scale={cand_scale.round(2).tolist()}  "
                  f"IoU={iou_c:.3f}")
            if iou_c > best_iou:
                best_iou = iou_c
                best_name = name
                best_mesh = cand_mesh
                best_scale = cand_scale
        info["mesh"] = best_mesh
        info["prescale_strategy"] = best_name
        info["prescale_xyz"] = best_scale.tolist()
        info["iou_after_prescale"] = float(best_iou)
        print(f"  {obj}: → BEST = '{best_name}' IoU={best_iou:.3f}\n")

    if not args.skip_refine:
        print(f"\n=== Step 5: IoU refinement (grid search) ===")
        for obj, info in object_poses.items():
            # Collect SAM masks for this object's group (subsample: max 4 per cam)
            from collections import defaultdict
            by_cam = defaultdict(list)
            for c in info["clusters"]:
                by_cam[c["cam"]].append(c)
            sampled = []
            for ci, cs in by_cam.items():
                # Largest masks first → take top 2 (cam 별, refinement 빠르게)
                cs.sort(key=lambda c: -c["n_pts"])
                sampled.extend(cs[:2])
            ref_obs = []
            for c in sampled:
                ci = c["cam"]
                fr = c["frame"]
                fid = f"{fr:06d}"
                if ci == 2:
                    T_be_p = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
                    if not T_be_p.exists():
                        continue
                    T_bc = np.load(T_be_p) @ T_gripper_cam
                else:
                    T_bc = static_T_base_cam[ci]
                K, D, _ = intrinsics[ci]
                ref_obs.append((c["mask_2d"], K, D, T_bc, c["mask_2d"].shape))
            if not ref_obs:
                continue
            print(f"  {obj}: refining with {len(ref_obs)} views (scipy.optimize) ...")
            T_init = info["T"]
            T_refined, best_iou = iou_refine(T_init, info["mesh"], ref_obs)
            info["T"] = T_refined
            info["iou_refined"] = best_iou
            dt = (T_refined[:3, 3] - T_init[:3, 3]) * 1000
            print(f"  {obj}: IoU={best_iou:.3f}  "
                  f"Δ=[{dt[0]:+.1f},{dt[1]:+.1f},{dt[2]:+.1f}]mm")

        # ── Stage 5: optimized pose
        print(f"\n  saving Stage 5: optimized pose → {stages_dir}/stage5_optimized_pose.png")
        save_stage_pose(object_poses, intrinsics, static_T_base_cam, T_gripper_cam,
                         ref_frame=args.ref_frame,
                         out_path=stages_dir / "stage5_optimized_pose.png",
                         stage_label="Step5 scipy")

    print(f"\n=== Saving artifacts to {out_dir} ===")
    summary = {}
    for obj, info in object_poses.items():
        T = info["T"]
        pos_mm = (T[:3, 3] * 1000).tolist()
        # Euler ZYX from pure rotation
        U, _, Vt = np.linalg.svd(T[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        # ZYX Euler
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if sy > 1e-6:
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            pitch = float(np.arctan2(-R[2, 0], sy))
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        else:
            roll = float(np.arctan2(-R[1, 2], R[1, 1]))
            pitch = float(np.arctan2(-R[2, 0], sy))
            yaw = 0.0
        summary[obj] = {
            "fit": info["fit"],
            "rmse_mm": info["rmse_mm"],
            "extent_ratio": info["extent_ratio"],
            "position_mm": pos_mm,
            "euler_zyx_deg": [np.degrees(yaw), np.degrees(pitch), np.degrees(roll)],
            "iou_refined": info.get("iou_refined"),
            "n_clusters_used": len(info["clusters"]),
            "group_idx": info["group_idx"],
        }
        # Save T per object
        obj_dir = out_dir / obj
        obj_dir.mkdir(parents=True, exist_ok=True)
        np.save(obj_dir / "T_base_object.npy", T)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary.json saved")

    # Overlay viz
    render_overlay_image(object_poses, intrinsics, static_T_base_cam, T_gripper_cam,
                          ref_frame=args.ref_frame,
                          out_path=out_dir / "overlay_all.png")

    print(f"\n=== Summary ===")
    print(f"{'obj':>6}  {'fit':>5}  {'rmse_mm':>7}  {'ext_ratio':>9}  {'IoU':>5}  pos_mm")
    print("-" * 80)
    for obj, s in summary.items():
        iou_s = f"{s['iou_refined']:.3f}" if s["iou_refined"] is not None else "  -- "
        print(f"{obj:>6}  {s['fit']:5.3f}  {s['rmse_mm']:7.2f}  "
              f"{s['extent_ratio']:9.2f}  {iou_s:>5}  "
              f"[{s['position_mm'][0]:+7.1f},{s['position_mm'][1]:+7.1f},"
              f"{s['position_mm'][2]:+7.1f}]")


if __name__ == "__main__":
    main()
