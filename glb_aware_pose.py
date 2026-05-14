#!/usr/bin/env python3
"""
GLB-aware iterative pose refinement.

Flow:
  1. 초기 pose 추정: fuse_multiframe_pose 의 결과 사용 (HSV + multi-frame fusion + ICP)
  2. GLB를 현재 추정 pose로 각 cam 좌표계에 렌더링 → 예측 silhouette
  3. 예측 silhouette의 bbox를 SAM bbox prompt로 사용 → 정밀 SAM 마스크
  4. SAM 마스크로 점군 재추출 → ICP 재실행 → 새 pose
  5. 수렴까지 (또는 max_iter) 반복

이전 한계점들이 해결되는 이유:
  - SAM with bbox prompt = 점 prompt보다 전체 객체 정확히 분리
  - GLB-driven silhouette = "예측" 마스크 → SAM에 강한 사전 정보 제공
  - render-and-compare = 가설 → 검증 → 정제 반복으로 수렴

사용:
  python3 glb_aware_pose.py [--objects red,cream,blue,box] [--max_iter 3]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

import sys
sys.path.insert(0, ".")
from fuse_multiframe_pose import (
    load_intrinsics, load_static_transforms, detect_table_v_threshold,
    load_obj_color_spec, table_aware_mask, backproject, transform_pts,
    clean_cloud, keep_largest_cluster, load_glb_as_pcd, initial_T,
    icp_with_scale,
    CAPTURE_DIR, DATA_DIR, CALIB_DIR
)

# Lazy SAM load
_sam_predictor = None


def get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None:
        from mobile_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
        sam.to("cpu")
        sam.eval()
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def project_mesh_to_silhouette(mesh, T_cam_obj, K, D, img_shape):
    """GLB 메쉬를 cam 좌표계로 변환 후 image plane에 silhouette 렌더."""
    h, w = img_shape[:2]
    verts_obj = np.asarray(mesh.vertices)
    if len(verts_obj) == 0:
        return None
    verts_cam = (T_cam_obj[:3, :3] @ verts_obj.T).T + T_cam_obj[:3, 3]
    z = verts_cam[:, 2]
    if (z > 0.05).sum() < 3:
        return None
    rvec = np.zeros(3); tvec = np.zeros(3)
    pts2d, _ = cv2.projectPoints(
        verts_cam.astype(np.float64), rvec, tvec,
        K.astype(np.float64), D.astype(np.float64))
    pts2d = pts2d.reshape(-1, 2)
    faces = np.asarray(mesh.faces)
    silhouette = np.zeros((h, w), dtype=np.uint8)
    for tri in faces:
        a, b, c = tri
        if z[a] <= 0.05 or z[b] <= 0.05 or z[c] <= 0.05:
            continue
        pts = np.array([pts2d[a], pts2d[b], pts2d[c]], dtype=np.int32)
        if np.any(pts[:, 0] < -200) or np.any(pts[:, 0] > w + 200):
            continue
        if np.any(pts[:, 1] < -200) or np.any(pts[:, 1] > h + 200):
            continue
        cv2.fillPoly(silhouette, [pts], 255)
    return silhouette


def silhouette_bbox(silhouette, pad_px=20):
    """silhouette mask에서 bbox 추출 + 패딩."""
    ys, xs = np.where(silhouette > 0)
    if len(xs) < 5:
        return None
    h, w = silhouette.shape
    x_min = max(0, int(xs.min()) - pad_px)
    x_max = min(w - 1, int(xs.max()) + pad_px)
    y_min = max(0, int(ys.min()) - pad_px)
    y_max = min(h - 1, int(ys.max()) + pad_px)
    return [x_min, y_min, x_max, y_max]


def sam_segment_with_bbox(rgb_bgr, bbox_xyxy):
    """SAM with bbox prompt → 단일 마스크 반환."""
    predictor = get_sam_predictor()
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb_rgb)
    box_np = np.array([bbox_xyxy], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=box_np, multimask_output=False)
    if masks is None or len(masks) == 0:
        return None
    return (masks[0] * 255).astype(np.uint8)


def collect_sam_masked_cloud(obj_name, T_base_obj, mesh,
                              intrinsics, static_T_base_cam, T_gripper_cam,
                              z_min_mm=5.0, z_max_mm=300.0,
                              frame_ids=None,
                              gripper_weight=10,
                              workspace_bbox_mm=((-500, 300), (150, 900)),
                              cam_ids=(0, 1, 2, 3)):
    """현재 T_base_obj에서 GLB silhouette 예측 → SAM bbox prompt → 마스크 → 점군."""
    all_pts = []
    n_used = 0
    n_sam_calls = 0
    sam_mask_cache = {}  # (frame, ci) → mask 캐시 (같은 iter 내 재사용)
    (xmin, xmax), (ymin, ymax) = workspace_bbox_mm
    if frame_ids is None:
        frame_ids = list(range(19))
    for frame in frame_ids:
        fid = f"{frame:06d}"
        for ci in cam_ids:
            rgb = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
            if rgb is None:
                continue
            # cam 좌표계로 객체 변환
            if ci == 2:
                T_be_path = CAPTURE_DIR / "cam2" / f"T_base_ee_{fid}.npy"
                if not T_be_path.exists():
                    continue
                T_be = np.load(T_be_path)
                T_base_cam = T_be @ T_gripper_cam
            else:
                T_base_cam = static_T_base_cam[ci]
            T_cam_base = np.linalg.inv(T_base_cam)
            T_cam_obj = T_cam_base @ T_base_obj

            K, _, depth_scale = intrinsics[ci]
            # GLB silhouette 예측
            silhouette = project_mesh_to_silhouette(mesh, T_cam_obj, K, np.zeros(5), rgb.shape)
            if silhouette is None:
                continue
            bbox = silhouette_bbox(silhouette, pad_px=25)
            if bbox is None:
                continue

            # SAM with bbox prompt → 정밀 마스크
            mask = sam_segment_with_bbox(rgb, bbox)
            n_sam_calls += 1
            if mask is None or mask.sum() < 100:
                continue

            depth = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"depth_{fid}.png"),
                                cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            pts_cam = backproject(depth, mask, K, depth_scale)
            if len(pts_cam) < 20:
                continue

            pts_base = transform_pts(pts_cam, T_base_cam)
            pts_mm = pts_base * 1000
            zm = pts_mm[:, 2]; xm = pts_mm[:, 0]; ym = pts_mm[:, 1]
            in_ws = ((zm > z_min_mm) & (zm < z_max_mm)
                     & (xm >= xmin) & (xm <= xmax)
                     & (ym >= ymin) & (ym <= ymax))
            if in_ws.sum() < 20:
                continue
            kept = pts_base[in_ws]
            if ci == 2 and gripper_weight > 1:
                kept = np.tile(kept, (gripper_weight, 1))
            all_pts.append(kept)
            n_used += 1

    if not all_pts:
        return np.zeros((0, 3)), n_used, n_sam_calls
    return np.vstack(all_pts), n_used, n_sam_calls


def refine_pose_iterative(obj_name, initial_T_base_obj, glb_path,
                           intrinsics, static_T_base_cam, T_gripper_cam,
                           max_iter=3, conv_thresh_mm=0.3, frame_ids=None):
    """초기 pose에서 시작해 GLB silhouette → SAM mask → ICP 반복.
    수렴 (pose 변화 < conv_thresh_mm) 또는 max_iter 까지.
    """
    print(f"  [glb-aware] iterative refinement for {obj_name}")
    # GLB 로드 (centered)
    model_pcd, mesh = load_glb_as_pcd(glb_path, n_samples=8000, center=True)

    T_curr = np.asarray(initial_T_base_obj, dtype=np.float64)
    best_rmse = np.inf
    best_fit = -1.0
    history = [{"iter": 0, "T": T_curr.tolist(), "rmse_mm": None,
                "fit": None, "delta_mm": None}]
    for it in range(1, max_iter + 1):
        print(f"  iter {it}/{max_iter}: rendering GLB silhouettes → SAM with bbox prompts")
        pts_np, n_used, n_sam = collect_sam_masked_cloud(
            obj_name, T_curr, mesh, intrinsics,
            static_T_base_cam, T_gripper_cam,
            gripper_weight=10, frame_ids=frame_ids)
        if len(pts_np) < 200:
            print(f"    [WARN] too few points ({len(pts_np)}) — keeping previous pose")
            break

        target_pcd = clean_cloud(pts_np, voxel_m=0.002)
        target_pcd = keep_largest_cluster(target_pcd, eps=0.012, min_points=40)
        target_arr = np.asarray(target_pcd.points)
        if len(target_arr) < 50:
            print(f"    [WARN] cluster too small — keeping previous pose")
            break

        # iter 1: yaw grid 24-step (orientation 탐색)
        # iter 2+: 현재 T_curr 그대로 init으로 fine refinement만 (yaw 안정)
        if it == 1:
            T0 = initial_T(model_pcd, target_pcd)
            result = icp_with_scale(model_pcd, target_pcd, T0,
                                     corr_mm_schedule=(15, 8, 4, 2.5),
                                     yaw_grid=24, with_scaling=True)
        else:
            # 이전 pose가 거의 정확하다고 가정, fine refinement만
            result = icp_with_scale(model_pcd, target_pcd, T_curr,
                                     corr_mm_schedule=(5, 3, 2),
                                     yaw_grid=1, with_scaling=True)
        T_new = result.transformation
        rmse = float(result.inlier_rmse * 1000)
        fit = float(result.fitness)

        # pose 변화 측정
        dt = float(np.linalg.norm((T_new[:3, 3] - T_curr[:3, 3]) * 1000))
        dR = T_new[:3, :3] @ T_curr[:3, :3].T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        print(f"    iter {it}: fit={fit:.3f}  rmse={rmse:.2f}mm  n_pts={len(target_arr)}  "
              f"sam_calls={n_sam}  Δpos={dt:.2f}mm Δrot={ang:.2f}°")

        # 악화 가드 — fit 떨어지거나 RMSE 너무 커지면 revert
        if fit < 0.5 or (it > 1 and rmse > best_rmse * 1.5):
            print(f"    [WARN] worsened (fit={fit:.3f}, rmse={rmse:.2f}); reverting to best")
            history.append({"iter": it, "T": T_new.tolist(), "rmse_mm": rmse,
                             "fit": fit, "delta_mm": dt, "delta_deg": ang,
                             "reverted": True})
            break

        history.append({
            "iter": it, "T": T_new.tolist(), "rmse_mm": rmse, "fit": fit,
            "delta_mm": dt, "delta_deg": ang,
        })
        T_curr = T_new
        if rmse < best_rmse:
            best_rmse = rmse
            best_fit = fit
        if dt < conv_thresh_mm and ang < 0.3:
            print(f"    -> converged (Δ < {conv_thresh_mm}mm, < 0.3°)")
            break

    return T_curr, mesh, history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", default="red,cream,blue,box")
    ap.add_argument("--max_iter", type=int, default=3)
    ap.add_argument("--initial_dir", default="data/pose_fused",
                    help="초기 pose 읽을 디렉토리 (기존 fuse 결과)")
    ap.add_argument("--out_dir", default="data/pose_fused_glb_aware")
    ap.add_argument("--frames", default="0-18",
                    help="frame id 명시. 형식: '0-18' 전체, '0,4,8,12,16' 5개, '0-4' 첫5개")
    args = ap.parse_args()

    # frame_ids 파싱
    fs = args.frames.strip()
    if "-" in fs and "," not in fs:
        a, b = fs.split("-")
        frame_ids = list(range(int(a), int(b) + 1))
    else:
        frame_ids = [int(x) for x in fs.split(",") if x.strip()]
    print(f"Using frame_ids: {frame_ids}")

    initial_dir = Path(args.initial_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = [s.strip() for s in args.objects.split(",") if s.strip()]

    print("Loading calibration / intrinsics ...")
    intrinsics = load_intrinsics()
    static_T_base_cam, T_gripper_cam = load_static_transforms()
    print("Loading SAM ...")
    _ = get_sam_predictor()

    print(f"\nObjects: {objects}  max_iter: {args.max_iter}\n")
    summary = {}
    for obj in objects:
        glb_path = DATA_DIR / f"{obj}.glb"
        if not glb_path.exists():
            print(f"[SKIP] {obj}: missing GLB")
            continue
        init_T_path = initial_dir / obj / "T_base_object.npy"
        if not init_T_path.exists():
            print(f"[SKIP] {obj}: missing initial pose at {init_T_path}")
            continue
        T_init = np.load(init_T_path)
        print(f"━━━ {obj} ━━━")
        print(f"  initial T_base_obj position (mm): {(T_init[:3,3]*1000).round(2)}")

        T_final, mesh, history = refine_pose_iterative(
            obj, T_init, glb_path, intrinsics, static_T_base_cam, T_gripper_cam,
            max_iter=args.max_iter, frame_ids=frame_ids)

        # save
        obj_out = out_dir / obj
        obj_out.mkdir(parents=True, exist_ok=True)
        np.save(obj_out / "T_base_object.npy", T_final)
        mesh_T = mesh.copy()
        mesh_T.apply_transform(T_final)
        mesh_T.export(str(obj_out / f"{obj}_posed_glb_aware.glb"))
        with open(obj_out / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        t = T_final[:3, 3] * 1000
        t_init = T_init[:3, 3] * 1000
        delta_total = float(np.linalg.norm(t - t_init))
        summary[obj] = {
            "initial_mm": [float(v) for v in t_init],
            "final_mm": [float(v) for v in t],
            "delta_mm": delta_total,
            "n_iter": len(history) - 1,
            "final_rmse_mm": history[-1].get("rmse_mm"),
            "final_fit": history[-1].get("fit"),
        }
        print(f"  → final pose: {t.round(2)} mm  (moved {delta_total:.2f}mm from initial)")
        print()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(f"{'obj':>7}  {'init pos':>22}  {'final pos':>22}  {'Δ pos':>8} {'iter':>5} {'rmse':>7}")
    print("-" * 90)
    for obj, s in summary.items():
        i = s["initial_mm"]
        f_ = s["final_mm"]
        rmse = s.get("final_rmse_mm", 0) or 0
        print(f"{obj:>7}  ({i[0]:+6.1f},{i[1]:+6.1f},{i[2]:+6.1f})  "
              f"({f_[0]:+6.1f},{f_[1]:+6.1f},{f_[2]:+6.1f})  "
              f"{s['delta_mm']:6.2f}mm  {s['n_iter']:>5} {rmse:6.2f}mm")


if __name__ == "__main__":
    main()
